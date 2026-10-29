"""DDL router for versioning statements.

``CREATE SNAPSHOT TABLE``, ``CREATE TABLE ... CLONE``,
``CREATE MATERIALIZED VIEW``, ``REFRESH MATERIALIZED VIEW``, ``DROP
SNAPSHOT TABLE``, and ``DROP MATERIALIZED VIEW`` are versioning DDL
statements that SQLGlot either round-trips opaquely or rejects. They
never go through the regular SQL translator — instead, the job
executor asks this module whether the incoming statement is a
versioning DDL, and if so delegates the whole job to the matching
manager.

The snapshot / clone / drop / refresh forms are detected by regex
(similar in spirit to ``sql/rewriter/information_schema.py``) so we can
intercept before the translator rejects the statement. ``CREATE
MATERIALIZED VIEW`` is parsed with SQLGlot instead: its grammar carries
``IF NOT EXISTS``, ``PARTITION BY``, ``CLUSTER BY`` and ``OPTIONS(...)``
(all of which dbt-bigquery emits), and the body must reach the
materialized-view manager verbatim for BigQuery's query rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
import re
from typing import TYPE_CHECKING

import sqlglot
from sqlglot import TokenType, exp

from bqemulator.domain.errors import InvalidQueryError

if TYPE_CHECKING:  # pragma: no cover
    from bqemulator.api.dependencies import AppContext


class VersioningDDLKind(StrEnum):
    """Types of Phase-7 DDL that bypass the SQL translator."""

    CREATE_SNAPSHOT = "CREATE_SNAPSHOT"
    DROP_SNAPSHOT = "DROP_SNAPSHOT"
    CREATE_CLONE = "CREATE_CLONE"
    CREATE_MATERIALIZED_VIEW = "CREATE_MATERIALIZED_VIEW"
    DROP_MATERIALIZED_VIEW = "DROP_MATERIALIZED_VIEW"
    REFRESH_MATERIALIZED_VIEW = "REFRESH_MATERIALIZED_VIEW"


# Patterns compiled with ``re.IGNORECASE | re.DOTALL`` so they work
# against arbitrarily-cased multi-line DDL bodies.
_ID = r"`?([A-Za-z0-9_\-]+)`?"
_QUALIFIED_NAME = rf"(?:{_ID}\s*\.\s*)?(?:{_ID}\s*\.\s*)?{_ID}"

_CREATE_SNAPSHOT_RE = re.compile(
    rf"^\s*CREATE\s+(?:OR\s+REPLACE\s+)?SNAPSHOT\s+TABLE\s+(?P<dest>{_QUALIFIED_NAME})"
    rf"\s+(?:CLONE|COPY)\s+(?P<src>{_QUALIFIED_NAME})"
    r"\s*;?\s*$",
    flags=re.IGNORECASE | re.DOTALL,
)
_DROP_SNAPSHOT_RE = re.compile(
    rf"^\s*DROP\s+SNAPSHOT\s+TABLE\s+(?:IF\s+EXISTS\s+)?(?P<dest>{_QUALIFIED_NAME})\s*;?\s*$",
    flags=re.IGNORECASE | re.DOTALL,
)
_CREATE_CLONE_RE = re.compile(
    rf"^\s*CREATE\s+(?:OR\s+REPLACE\s+)?TABLE\s+(?P<dest>{_QUALIFIED_NAME})"
    rf"\s+CLONE\s+(?P<src>{_QUALIFIED_NAME})"
    r"\s*;?\s*$",
    flags=re.IGNORECASE | re.DOTALL,
)
_MATERIALIZED_VIEW_RE = re.compile(r"MATERIALIZED\s+VIEW", flags=re.IGNORECASE)
_DROP_MATERIALIZED_VIEW_RE = re.compile(
    rf"^\s*DROP\s+MATERIALIZED\s+VIEW\s+(?:IF\s+EXISTS\s+)?(?P<dest>{_QUALIFIED_NAME})\s*;?\s*$",
    flags=re.IGNORECASE | re.DOTALL,
)
_REFRESH_MATERIALIZED_VIEW_RE = re.compile(
    rf"^\s*CALL\s+BQ\.REFRESH_MATERIALIZED_VIEW\s*\(\s*['\"](?P<dest>[^'\"]+)['\"]\s*\)\s*;?\s*$"
    rf"|^\s*REFRESH\s+MATERIALIZED\s+VIEW\s+(?P<dest2>{_QUALIFIED_NAME})\s*;?\s*$",
    flags=re.IGNORECASE | re.DOTALL,
)


@dataclass(slots=True, frozen=True)
class VersioningDDL:
    """Parsed Phase-7 DDL statement."""

    kind: VersioningDDLKind
    target_project: str
    target_dataset: str
    target_table: str
    source_project: str | None = None
    source_dataset: str | None = None
    source_table: str | None = None
    view_query: str | None = None
    # CREATE MATERIALIZED VIEW only.
    replace: bool = False
    if_not_exists: bool = False
    partition_by: str | None = None  # BigQuery SQL of the PARTITION BY expression
    cluster_by: tuple[str, ...] = ()
    options: dict[str, object] = field(default_factory=dict)  # lower-cased OPTIONS(...)


def is_versioning_ddl(sql: str) -> bool:
    """Quick check — returns True if ``sql`` might be a Phase-7 DDL."""
    upper = sql.upper()
    return (
        "SNAPSHOT" in upper
        or "CLONE" in upper
        or "REFRESH_MATERIALIZED_VIEW" in upper
        or _MATERIALIZED_VIEW_RE.search(sql) is not None
    )


class VersioningDDLRouter:
    """Parses and routes Phase-7 DDL."""

    def __init__(self, default_project_id: str) -> None:
        self._project_id = default_project_id

    def parse(self, sql: str) -> VersioningDDL | None:
        """Parse ``sql`` into a :class:`VersioningDDL`, or return ``None``."""
        sql = sql.strip().rstrip(";").strip()

        m = _CREATE_SNAPSHOT_RE.match(sql)
        if m is not None:
            dest_p, dest_d, dest_t = self._split_qualified(m.group("dest"))
            src_p, src_d, src_t = self._split_qualified(m.group("src"))
            return VersioningDDL(
                kind=VersioningDDLKind.CREATE_SNAPSHOT,
                target_project=dest_p,
                target_dataset=dest_d,
                target_table=dest_t,
                source_project=src_p,
                source_dataset=src_d,
                source_table=src_t,
            )

        m = _DROP_SNAPSHOT_RE.match(sql)
        if m is not None:
            dest_p, dest_d, dest_t = self._split_qualified(m.group("dest"))
            return VersioningDDL(
                kind=VersioningDDLKind.DROP_SNAPSHOT,
                target_project=dest_p,
                target_dataset=dest_d,
                target_table=dest_t,
            )

        m = _CREATE_CLONE_RE.match(sql)
        if m is not None:
            dest_p, dest_d, dest_t = self._split_qualified(m.group("dest"))
            src_p, src_d, src_t = self._split_qualified(m.group("src"))
            return VersioningDDL(
                kind=VersioningDDLKind.CREATE_CLONE,
                target_project=dest_p,
                target_dataset=dest_d,
                target_table=dest_t,
                source_project=src_p,
                source_dataset=src_d,
                source_table=src_t,
            )

        create_mv = self._parse_create_materialized_view(sql)
        if create_mv is not None:
            return create_mv

        m = _DROP_MATERIALIZED_VIEW_RE.match(sql)
        if m is not None:
            dest_p, dest_d, dest_t = self._split_qualified(m.group("dest"))
            return VersioningDDL(
                kind=VersioningDDLKind.DROP_MATERIALIZED_VIEW,
                target_project=dest_p,
                target_dataset=dest_d,
                target_table=dest_t,
            )

        m = _REFRESH_MATERIALIZED_VIEW_RE.match(sql)
        if m is not None:
            raw = m.group("dest") or m.group("dest2")
            dest_p, dest_d, dest_t = self._split_qualified(raw)
            return VersioningDDL(
                kind=VersioningDDLKind.REFRESH_MATERIALIZED_VIEW,
                target_project=dest_p,
                target_dataset=dest_d,
                target_table=dest_t,
            )

        return None

    def _parse_create_materialized_view(self, sql: str) -> VersioningDDL | None:
        """Parse ``CREATE [OR REPLACE] MATERIALIZED VIEW [IF NOT EXISTS] …``.

        The head (target, ``PARTITION BY``, ``CLUSTER BY``, ``OPTIONS``)
        comes from SQLGlot's :class:`exp.Create`; the body is sliced from
        the original text after the first top-level ``AS`` so the manager
        stores and validates exactly what the caller wrote. Returns
        ``None`` for anything that is not a materialized-view CREATE,
        including SQL SQLGlot cannot parse (the translator reports that).
        """
        if _MATERIALIZED_VIEW_RE.search(sql) is None:
            return None
        try:
            tree = sqlglot.parse_one(sql, read="bigquery")
        except Exception:  # noqa: BLE001 — not ours to diagnose
            return None
        if not isinstance(tree, exp.Create) or (tree.kind or "").upper() != "VIEW":
            return None
        properties = tree.args.get("properties")
        props = list(properties.expressions) if isinstance(properties, exp.Properties) else []
        if not any(isinstance(p, exp.MaterializedProperty) for p in props):
            return None
        target = tree.this.this if isinstance(tree.this, exp.Schema) else tree.this
        if not isinstance(target, exp.Table):
            return None
        body = _body_after_top_level_as(sql)
        if body is None:
            raise InvalidQueryError("CREATE MATERIALIZED VIEW requires an AS <query> body")
        partition_by: str | None = None
        cluster_by: tuple[str, ...] = ()
        options: dict[str, object] = {}
        for prop in props:
            if isinstance(prop, exp.PartitionedByProperty):
                partition_by = prop.this.sql("bigquery")
            elif isinstance(prop, exp.ClusterProperty):
                cluster_by = tuple(e.name or e.sql("bigquery") for e in prop.expressions)
            elif type(prop) is exp.Property:
                options[prop.name.lower()] = _option_value(prop.args.get("value"))
        qualified = ".".join(part for part in (target.catalog, target.db, target.name) if part)
        dest_p, dest_d, dest_t = self._split_qualified(qualified)
        return VersioningDDL(
            kind=VersioningDDLKind.CREATE_MATERIALIZED_VIEW,
            target_project=dest_p,
            target_dataset=dest_d,
            target_table=dest_t,
            view_query=body,
            replace=bool(tree.args.get("replace")),
            if_not_exists=bool(tree.args.get("exists")),
            partition_by=partition_by,
            cluster_by=cluster_by,
            options=options,
        )

    def _split_qualified(self, raw: str) -> tuple[str, str, str]:
        """Split ``[project.]dataset.table`` into its three parts."""
        # Drop backticks the regex may have retained.
        cleaned = raw.replace("`", "")
        parts = [p.strip() for p in cleaned.split(".") if p.strip()]
        if len(parts) == _PARTS_FULLY_QUALIFIED:
            return parts[0], parts[1], parts[2]
        if len(parts) == _PARTS_DATASET_QUALIFIED:
            return self._project_id, parts[0], parts[1]
        if len(parts) == _PARTS_BARE:
            raise InvalidQueryError(
                "Versioning DDL requires a fully-qualified or dataset-qualified table reference",
            )
        raise InvalidQueryError(
            f"Could not parse versioning DDL target: {raw!r}",
        )


_PARTS_FULLY_QUALIFIED = 3
_PARTS_DATASET_QUALIFIED = 2
_PARTS_BARE = 1


def _unwrap_source(ddl: VersioningDDL) -> tuple[str, str, str]:
    """Return the ``(project, dataset, table)`` source triple.

    Snapshot / clone DDLs require the source to be set by the parser
    before dispatch reaches this layer; returning the narrowed triple
    lets the caller pass it through to the manager without casting at
    every argument position.

    Uses an explicit raise rather than ``assert`` so the invariant
    survives ``python -O`` and so the failure mode is a normal client
    error rather than ``AssertionError``.
    """
    if ddl.source_project is None or ddl.source_dataset is None or ddl.source_table is None:
        raise InvalidQueryError("Snapshot/clone DDL requires a source table reference")
    return ddl.source_project, ddl.source_dataset, ddl.source_table


async def execute_versioning_ddl(
    ddl: VersioningDDL,
    ctx: AppContext,
) -> None:
    """Dispatch a parsed Phase-7 DDL to the matching manager.

    The caller holds neither the write lock nor the snapshot manager;
    every manager acquires its own resources as needed.
    """
    from bqemulator.versioning.clone import CloneManager
    from bqemulator.versioning.materialized_views import MaterializedViewManager
    from bqemulator.versioning.snapshot_table import SnapshotTableManager

    if ddl.kind is VersioningDDLKind.CREATE_SNAPSHOT:
        src_project, src_dataset, src_table = _unwrap_source(ddl)
        await SnapshotTableManager(ctx).create(
            ddl.target_project,
            ddl.target_dataset,
            ddl.target_table,
            src_project,
            src_dataset,
            src_table,
        )
    elif ddl.kind is VersioningDDLKind.DROP_SNAPSHOT:
        await SnapshotTableManager(ctx).drop(
            ddl.target_project,
            ddl.target_dataset,
            ddl.target_table,
        )
    elif ddl.kind is VersioningDDLKind.CREATE_CLONE:
        src_project, src_dataset, src_table = _unwrap_source(ddl)
        await CloneManager(ctx).create(
            ddl.target_project,
            ddl.target_dataset,
            ddl.target_table,
            src_project,
            src_dataset,
            src_table,
        )
    elif ddl.kind is VersioningDDLKind.CREATE_MATERIALIZED_VIEW:
        assert ddl.view_query is not None  # noqa: S101
        await MaterializedViewManager(ctx).create(
            ddl.target_project,
            ddl.target_dataset,
            ddl.target_table,
            ddl.view_query,
            replace=ddl.replace,
            if_not_exists=ddl.if_not_exists,
            partition_by=ddl.partition_by,
            cluster_by=ddl.cluster_by,
            options=ddl.options,
        )
    elif ddl.kind is VersioningDDLKind.DROP_MATERIALIZED_VIEW:
        await MaterializedViewManager(ctx).drop(
            ddl.target_project,
            ddl.target_dataset,
            ddl.target_table,
        )
    elif ddl.kind is VersioningDDLKind.REFRESH_MATERIALIZED_VIEW:
        await MaterializedViewManager(ctx).refresh(
            ddl.target_project,
            ddl.target_dataset,
            ddl.target_table,
        )
    else:  # pragma: no cover — all branches enumerated
        raise InvalidQueryError(f"Unknown Phase-7 DDL: {ddl.kind}")


def _body_after_top_level_as(sql: str) -> str | None:
    """Return the text after the first ``AS`` outside parentheses, or ``None``.

    ``PARTITION BY DATE(CAST(ts AS TIMESTAMP))`` and ``OPTIONS(description=
    'as is')`` both contain ``AS`` tokens, so the tokenizer tracks depth
    and string literals rather than searching the raw text.
    """
    tokens = sqlglot.Dialect.get_or_raise("bigquery").tokenizer_class().tokenize(sql)
    depth = 0
    for token in tokens:
        if token.token_type is TokenType.L_PAREN:
            depth += 1
        elif token.token_type is TokenType.R_PAREN:
            depth -= 1
        elif token.token_type is TokenType.ALIAS and depth == 0:
            body = sql[token.end + 1 :].strip()
            return body or None
    return None


def _option_value(value: exp.Expr | None) -> object:
    """Python value of an ``OPTIONS(name=value)`` entry.

    Booleans and integer / string literals become their Python
    counterparts; anything else (``INTERVAL``, arrays, struct literals)
    stays as BigQuery SQL text.
    """
    if value is None:
        return None
    if isinstance(value, exp.Boolean):
        return value.this
    if isinstance(value, exp.Literal):
        if value.is_string:
            return value.this
        if value.is_int:
            return int(value.this)
        return float(value.this)
    if isinstance(value, exp.Null):
        return None
    return value.sql("bigquery")


__all__ = [
    "VersioningDDL",
    "VersioningDDLKind",
    "VersioningDDLRouter",
    "execute_versioning_ddl",
    "is_versioning_ddl",
]
