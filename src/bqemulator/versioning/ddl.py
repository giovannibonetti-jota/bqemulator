"""DDL router for versioning statements.

``CREATE SNAPSHOT TABLE``, ``CREATE TABLE ... CLONE``, ``CREATE
MATERIALIZED VIEW``, ``DROP SNAPSHOT TABLE``, ``DROP MATERIALIZED VIEW``
and ``CALL BQ.REFRESH_MATERIALIZED_VIEW`` are versioning DDL statements
DuckDB cannot run. They never go through the regular SQL translator —
instead, the job executor asks this module whether the incoming
statement is a versioning DDL, and if so delegates the whole job to the
matching manager.

Routing is by SQLGlot node type. ``CREATE MATERIALIZED VIEW``, ``CREATE
TABLE ... CLONE`` and ``DROP MATERIALIZED VIEW`` parse into typed
:class:`exp.Create` / :class:`exp.Drop` nodes that carry ``OR REPLACE``,
``IF [NOT] EXISTS``, ``PARTITION BY``, ``CLUSTER BY`` and ``OPTIONS``
(all of which dbt-bigquery emits), plus leading comments and backticked
or hyphenated names for free. The snapshot heads, a clone with
``OPTIONS`` or ``FOR SYSTEM_TIME``, and ``CALL BQ.REFRESH`` come back as
an opaque :class:`exp.Command`, and the emulator's own ``REFRESH
MATERIALIZED VIEW`` extension (ADR 0017) is not BigQuery syntax at all;
those go through the :mod:`bqemulator.sql.ddl_heads` grammar.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
import re
from typing import TYPE_CHECKING

import sqlglot
from sqlglot import TokenType, exp

from bqemulator.domain.errors import InvalidQueryError
from bqemulator.sql.ddl_heads import (
    CallRefresh,
    CreateClone,
    CreateSnapshot,
    DropSnapshot,
    RefreshMaterializedView,
    option_value,
    parse_ddl_head,
)
from bqemulator.sql.statement import parse_single_statement

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


_MATERIALIZED_VIEW_RE = re.compile(r"MATERIALIZED\s+VIEW", flags=re.IGNORECASE)


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
    # CREATE forms.
    replace: bool = False
    if_not_exists: bool = False
    options: dict[str, object] = field(default_factory=dict)  # lower-cased OPTIONS(...)
    # DROP forms.
    if_exists: bool = False
    # Snapshot / clone: the ``FOR SYSTEM_TIME AS OF`` expression, as BigQuery SQL.
    system_time: str | None = None
    # CREATE MATERIALIZED VIEW only.
    partition_by: str | None = None  # BigQuery SQL of the PARTITION BY expression
    cluster_by: tuple[str, ...] = ()


def is_versioning_ddl(sql: str) -> bool:
    """Cheap textual pre-filter — True if ``sql`` might be a Phase-7 DDL."""
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

    def parse(self, sql: str, tree: exp.Expr | None = None) -> VersioningDDL | None:
        """Parse ``sql`` into a :class:`VersioningDDL`, or return ``None``.

        ``tree`` is the statement's SQLGlot parse when the caller already
        has it (the executor and interpreter parse each statement once);
        otherwise ``sql`` is parsed here. A statement SQLGlot cannot type
        (an ``exp.Command``) or cannot parse at all goes to the DDL-head
        grammar. Anything that is not a single versioning statement
        returns ``None`` so the normal translator path reports it.
        """
        if tree is None:
            tree = parse_single_statement(sql)
        if tree is None or isinstance(tree, exp.Command):
            return self._parse_untyped(sql)
        if isinstance(tree, exp.Create):
            return self._parse_create(sql, tree)
        if isinstance(tree, exp.Drop):
            return self._parse_drop(tree)
        return None

    def _parse_create(self, sql: str, tree: exp.Create) -> VersioningDDL | None:
        """Route ``CREATE MATERIALIZED VIEW`` and ``CREATE TABLE … CLONE``."""
        kind = (tree.kind or "").upper()
        if kind == "VIEW":
            return self._parse_create_materialized_view(sql, tree)
        clone = tree.args.get("clone")
        if kind == "TABLE" and isinstance(clone, exp.Clone) and not clone.args.get("copy"):
            source = clone.this
            if not isinstance(source, exp.Table):
                return None
            dest_p, dest_d, dest_t = self._table_parts(tree.this)
            src_p, src_d, src_t = self._table_parts(source)
            return VersioningDDL(
                kind=VersioningDDLKind.CREATE_CLONE,
                target_project=dest_p,
                target_dataset=dest_d,
                target_table=dest_t,
                source_project=src_p,
                source_dataset=src_d,
                source_table=src_t,
                replace=bool(tree.args.get("replace")),
                if_not_exists=bool(tree.args.get("exists")),
            )
        return None

    def _parse_drop(self, tree: exp.Drop) -> VersioningDDL | None:
        """Route ``DROP MATERIALIZED VIEW [IF EXISTS]``."""
        if (tree.kind or "").upper() != "VIEW" or not tree.args.get("materialized"):
            return None
        dest_p, dest_d, dest_t = self._table_parts(tree.this)
        return VersioningDDL(
            kind=VersioningDDLKind.DROP_MATERIALIZED_VIEW,
            target_project=dest_p,
            target_dataset=dest_d,
            target_table=dest_t,
            if_exists=bool(tree.args.get("exists")),
        )

    def _parse_untyped(self, sql: str) -> VersioningDDL | None:
        """Route the heads SQLGlot cannot type through the DDL-head grammar."""
        head = parse_ddl_head(sql)
        if isinstance(head, CreateSnapshot | CreateClone):
            dest_p, dest_d, dest_t = self._split_qualified(head.target)
            src_p, src_d, src_t = self._split_qualified(head.source)
            kind = (
                VersioningDDLKind.CREATE_SNAPSHOT
                if isinstance(head, CreateSnapshot)
                else VersioningDDLKind.CREATE_CLONE
            )
            return VersioningDDL(
                kind=kind,
                target_project=dest_p,
                target_dataset=dest_d,
                target_table=dest_t,
                source_project=src_p,
                source_dataset=src_d,
                source_table=src_t,
                replace=head.replace,
                if_not_exists=head.if_not_exists,
                options={name: option_value(raw) for name, raw in head.options.items()},
                system_time=head.system_time,
            )
        if isinstance(head, DropSnapshot):
            dest_p, dest_d, dest_t = self._split_qualified(head.target)
            return VersioningDDL(
                kind=VersioningDDLKind.DROP_SNAPSHOT,
                target_project=dest_p,
                target_dataset=dest_d,
                target_table=dest_t,
                if_exists=head.if_exists,
            )
        if isinstance(head, CallRefresh | RefreshMaterializedView):
            dest_p, dest_d, dest_t = self._split_qualified(head.target)
            return VersioningDDL(
                kind=VersioningDDLKind.REFRESH_MATERIALIZED_VIEW,
                target_project=dest_p,
                target_dataset=dest_d,
                target_table=dest_t,
            )
        return None

    def _parse_create_materialized_view(self, sql: str, tree: exp.Create) -> VersioningDDL | None:
        """Parse ``CREATE [OR REPLACE] MATERIALIZED VIEW [IF NOT EXISTS] …``.

        The head (target, ``PARTITION BY``, ``CLUSTER BY``, ``OPTIONS``)
        comes from SQLGlot's :class:`exp.Create`; the body is sliced from
        the original text after the first top-level ``AS`` so the manager
        stores and validates exactly what the caller wrote. Returns
        ``None`` for a plain (non-materialized) view.
        """
        properties = tree.args.get("properties")
        props = list(properties.expressions) if isinstance(properties, exp.Properties) else []
        if not any(isinstance(p, exp.MaterializedProperty) for p in props):
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
        dest_p, dest_d, dest_t = self._table_parts(tree.this)
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

    def _table_parts(self, node: exp.Expr) -> tuple[str, str, str]:
        """``(project, dataset, table)`` of a DDL target node.

        SQLGlot already splits a whole-backticked three-part name into
        catalog / db / name, so this only applies the default project and
        the "dataset-qualified at least" rule.
        """
        table = node.this if isinstance(node, exp.Schema) else node
        if not isinstance(table, exp.Table):
            raise InvalidQueryError(
                f"Could not parse versioning DDL target: {node.sql('bigquery')!r}",
            )
        qualified = ".".join(part for part in (table.catalog, table.db, table.name) if part)
        return self._split_qualified(qualified)

    def _split_qualified(self, raw: str) -> tuple[str, str, str]:
        """Split ``[project.]dataset.table`` into its three parts."""
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

_LABEL_PAIR_RE = re.compile(r"\(\s*'([^']*)'\s*,\s*'([^']*)'\s*\)")


@dataclass(slots=True, frozen=True)
class TableOptions:
    """The ``TableMeta`` fields a snapshot / clone ``OPTIONS(...)`` clause sets."""

    description: str | None = None
    friendly_name: str | None = None
    labels: dict[str, str] = field(default_factory=dict)


def table_metadata_options(options: Mapping[str, object]) -> TableOptions:
    """Metadata from a snapshot / clone ``OPTIONS(...)`` clause.

    ``description`` and ``friendly_name`` carry over as strings and
    ``labels`` from its ``[('k', 'v'), …]`` literal; the other options
    (expiration, KMS key) have no emulator counterpart and are ignored.
    """
    description = options.get("description")
    friendly_name = options.get("friendly_name")
    labels = options.get("labels")
    return TableOptions(
        description=description if isinstance(description, str) else None,
        friendly_name=friendly_name if isinstance(friendly_name, str) else None,
        labels=dict(_LABEL_PAIR_RE.findall(labels)) if isinstance(labels, str) else {},
    )


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
            replace=ddl.replace,
            if_not_exists=ddl.if_not_exists,
            system_time=ddl.system_time,
            options=ddl.options,
        )
    elif ddl.kind is VersioningDDLKind.DROP_SNAPSHOT:
        await SnapshotTableManager(ctx).drop(
            ddl.target_project,
            ddl.target_dataset,
            ddl.target_table,
            if_exists=ddl.if_exists,
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
            replace=ddl.replace,
            if_not_exists=ddl.if_not_exists,
            system_time=ddl.system_time,
            options=ddl.options,
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
            if_exists=ddl.if_exists,
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
            body = sql[token.end + 1 :].strip().rstrip(";").strip()
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
    "TableOptions",
    "VersioningDDL",
    "VersioningDDLKind",
    "VersioningDDLRouter",
    "execute_versioning_ddl",
    "is_versioning_ddl",
    "table_metadata_options",
]
