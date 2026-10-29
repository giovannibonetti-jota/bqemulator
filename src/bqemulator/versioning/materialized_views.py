"""Materialized-view registry, refresh, and staleness tracking.

Implements the MV model from ADR 0017 — event-driven staleness
flagging plus lazy recompute on read.

Lifecycle:

1. ``CREATE MATERIALIZED VIEW mv [PARTITION BY …] [CLUSTER BY …]
   [OPTIONS(…)] AS <query>``
   → parse ``<query>`` with SQLGlot, check it against BigQuery's
   materialized-view query rules (:mod:`bqemulator.versioning.mv_rules`;
   ``allow_non_incremental_definition`` selects the looser set), extract
   base-table refs (plain tables only — BigQuery refuses views, other
   materialized views and snapshots), run the query through the regular
   SQL translation pipeline, materialise the rows into a regular dataset
   table, store ``TableMeta`` (with the partitioning / clustering) +
   ``MaterializedViewMeta`` + dependency entries in the catalog,
   subscribe the MV to ``TableDataChanged`` events on each base table.
2. ``TableDataChanged(base)``
   → every subscribed MV flips ``is_stale=True`` in the catalog.
3. ``SELECT ... FROM mv``
   → the jobs executor consults
   :meth:`MaterializedViewManager.refresh_if_stale` before executing;
   a stale MV is recomputed under the write lock.
4. ``REFRESH MATERIALIZED VIEW mv``
   → :meth:`MaterializedViewManager.refresh` forces a recompute
   regardless of the flag.
5. ``DROP MATERIALIZED VIEW mv``
   → drop the physical table, clear subscriptions, remove catalog
   entries.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sqlglot
from sqlglot import exp

from bqemulator.catalog.etag import generate_etag
from bqemulator.catalog.models import (
    Clustering,
    MaterializedViewMeta,
    RangePartitioning,
    TableFieldSchema,
    TableMeta,
    TableSchema,
    TimePartitioning,
)
from bqemulator.domain.errors import (
    InvalidQueryError,
    ResourceRef,
    resource_already_exists,
    resource_not_found,
)
from bqemulator.domain.events import TableDataChanged
from bqemulator.observability.logging_ import get_logger
from bqemulator.sql.table_rewriter import rewrite_table_refs
from bqemulator.sql.translator import SQLTranslator
from bqemulator.storage.sql_identifiers import quoted_table_ref
from bqemulator.versioning.mv_rules import validate_materialized_view_query

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable, Mapping

    from bqemulator.api.dependencies import AppContext
    from bqemulator.domain.events import DomainEvent


_log = get_logger(__name__)


class MaterializedViewManager:
    """Coordinates MV lifecycle + refresh."""

    def __init__(self, ctx: AppContext) -> None:
        self._ctx = ctx
        self._translator = SQLTranslator()

    # -- Public API ------------------------------------------------------

    async def create(
        self,
        project_id: str,
        dataset_id: str,
        table_id: str,
        view_query: str,
        *,
        replace: bool = False,
        if_not_exists: bool = False,
        partition_by: str | None = None,
        cluster_by: tuple[str, ...] = (),
        options: Mapping[str, object] | None = None,
    ) -> MaterializedViewMeta:
        """Materialise a new MV and register its dependencies.

        ``options`` is the lower-cased ``OPTIONS(...)`` clause.
        ``allow_non_incremental_definition`` switches the query rules to
        the non-incremental set and, as in BigQuery, must come with
        ``max_staleness``; ``description`` and ``friendly_name`` land on
        the table metadata; the refresh options are accepted and ignored
        (the emulator refreshes on read).
        """
        opts = dict(options or {})
        non_incremental = bool(opts.get("allow_non_incremental_definition"))
        if non_incremental and "max_staleness" not in opts:
            raise InvalidQueryError(
                "allow_non_incremental_definition must be accompanied by the max_staleness option",
            )
        if self._ctx.catalog.get_dataset(project_id, dataset_id) is None:
            raise resource_not_found(
                ResourceRef("dataset", project_id, dataset_id),
            )
        existing = self._ctx.catalog.get_table(project_id, dataset_id, table_id)
        if existing is not None:
            is_mv = existing.table_type == "MATERIALIZED_VIEW"
            current = self._ctx.catalog.get_materialized_view(project_id, dataset_id, table_id)
            if if_not_exists and is_mv and current is not None:
                return current
            if not (replace and is_mv):
                raise resource_already_exists(
                    ResourceRef("table", project_id, dataset_id, table_id),
                )

        tree = _parse_view_query(view_query)
        validate_materialized_view_query(tree, non_incremental=non_incremental)
        base_tables = extract_base_tables(tree, project_id)
        if not base_tables:
            raise InvalidQueryError(
                "CREATE MATERIALIZED VIEW requires at least one base table",
            )
        # Every base table must exist or the MV cannot refresh, and must
        # be a plain table: BigQuery does not nest materialized views and
        # refuses logical views (preview) and snapshots as sources.
        for bp, bd, bt in base_tables:
            base = self._ctx.catalog.get_table(bp, bd, bt)
            if base is None:
                raise resource_not_found(ResourceRef("table", bp, bd, bt))
            problem = _BASE_TABLE_KIND_PROBLEM.get(base.table_type)
            if problem is not None:
                raise InvalidQueryError(
                    f"Invalid materialized view query: {problem} ({bp}.{bd}.{bt})",
                )
        time_partitioning, range_partitioning = _partitioning(partition_by)

        if existing is not None:
            await self.drop(project_id, dataset_id, table_id)

        now = self._ctx.clock.now()
        target_ref = quoted_table_ref(project_id, dataset_id, table_id)
        duckdb_sql = self._translate(view_query, project_id)

        async with self._ctx.engine.write_lock():
            self._ctx.engine.execute(
                f"CREATE OR REPLACE TABLE {target_ref} AS {duckdb_sql}",
            )
            schema = self._read_arrow_schema(project_id, dataset_id, table_id)
            try:
                _check_layout_columns(
                    schema,
                    cluster_by=cluster_by,
                    time_partitioning=time_partitioning,
                    range_partitioning=range_partitioning,
                )
            except InvalidQueryError:
                self._ctx.engine.execute(f"DROP TABLE IF EXISTS {target_ref}")
                raise
            count_row = self._ctx.engine.execute(
                f"SELECT COUNT(*) FROM {target_ref}",
            ).fetchone()
            num_rows = int(count_row[0]) if count_row else 0

            description = opts.get("description")
            friendly_name = opts.get("friendly_name")
            table_meta = TableMeta(
                project_id=project_id,
                dataset_id=dataset_id,
                table_id=table_id,
                table_type="MATERIALIZED_VIEW",
                schema=schema,
                friendly_name=friendly_name if isinstance(friendly_name, str) else None,
                description=description if isinstance(description, str) else None,
                time_partitioning=time_partitioning,
                range_partitioning=range_partitioning,
                clustering=Clustering(fields=cluster_by) if cluster_by else None,
                creation_time=now,
                last_modified_time=now,
                num_rows=num_rows,
                num_bytes=0,
                etag=generate_etag(
                    project_id,
                    dataset_id,
                    table_id,
                    "MATERIALIZED_VIEW",
                    str(now),
                ),
                view_query=view_query,
            )
            self._ctx.catalog.create_table(table_meta)

            mv_meta = MaterializedViewMeta(
                project_id=project_id,
                dataset_id=dataset_id,
                table_id=table_id,
                view_query=view_query,
                base_tables=tuple(base_tables),
                last_refresh_time=now,
                is_stale=False,
            )
            self._ctx.catalog.upsert_materialized_view(mv_meta)

        _subscribe_mv(self._ctx, mv_meta)
        return mv_meta

    async def drop(
        self,
        project_id: str,
        dataset_id: str,
        table_id: str,
    ) -> None:
        """Drop an MV and clear its subscriptions."""
        existing_table = self._ctx.catalog.get_table(project_id, dataset_id, table_id)
        if existing_table is None or existing_table.table_type != "MATERIALIZED_VIEW":
            raise resource_not_found(
                ResourceRef("materialized_view", project_id, dataset_id, table_id),
            )
        target_ref = quoted_table_ref(project_id, dataset_id, table_id)
        async with self._ctx.engine.write_lock():
            self._ctx.engine.execute(f"DROP TABLE IF EXISTS {target_ref}")
            self._ctx.catalog.delete_table(
                project_id,
                dataset_id,
                table_id,
                not_found_ok=True,
            )
            self._ctx.catalog.delete_materialized_view(
                project_id,
                dataset_id,
                table_id,
                not_found_ok=True,
            )
        _unsubscribe_mv(self._ctx, project_id, dataset_id, table_id)

    async def refresh(
        self,
        project_id: str,
        dataset_id: str,
        table_id: str,
    ) -> MaterializedViewMeta:
        """Force-refresh an MV regardless of staleness."""
        mv = self._ctx.catalog.get_materialized_view(project_id, dataset_id, table_id)
        if mv is None:
            raise resource_not_found(
                ResourceRef("materialized_view", project_id, dataset_id, table_id),
            )
        return await self._do_refresh(mv, force=True)

    async def refresh_if_stale(
        self,
        project_id: str,
        dataset_id: str,
        table_id: str,
    ) -> MaterializedViewMeta | None:
        """Refresh the MV only when ``is_stale=True``.

        Returns the updated :class:`MaterializedViewMeta` (or the current
        one if no refresh was needed). Returns ``None`` if the table is
        not an MV.

        The staleness flag is re-checked inside :meth:`_do_refresh`'s
        write lock to collapse concurrent stale-readers onto a single
        recompute — the second reader sees ``is_stale=False`` and
        skips the redundant CTAS.
        """
        mv = self._ctx.catalog.get_materialized_view(project_id, dataset_id, table_id)
        if mv is None:
            return None
        if not mv.is_stale:
            return mv
        return await self._do_refresh(mv)

    # -- Internals -------------------------------------------------------

    async def _do_refresh(
        self,
        mv: MaterializedViewMeta,
        *,
        force: bool = False,
    ) -> MaterializedViewMeta:
        duckdb_sql = self._translate(mv.view_query, mv.project_id)
        target_ref = quoted_table_ref(mv.project_id, mv.dataset_id, mv.table_id)
        now = self._ctx.clock.now()

        async with self._ctx.engine.write_lock():
            # Re-check inside the lock so concurrent stale-readers
            # collapse onto a single recompute. ``force=True`` bypasses
            # the recheck for explicit ``REFRESH MATERIALIZED VIEW``.
            if not force:
                fresh = self._ctx.catalog.get_materialized_view(
                    mv.project_id,
                    mv.dataset_id,
                    mv.table_id,
                )
                if fresh is None:
                    return mv
                if not fresh.is_stale:
                    return fresh
                mv = fresh
            self._ctx.engine.execute(
                f"CREATE OR REPLACE TABLE {target_ref} AS {duckdb_sql}",
            )
            count_row = self._ctx.engine.execute(
                f"SELECT COUNT(*) FROM {target_ref}",
            ).fetchone()
            num_rows = int(count_row[0]) if count_row else 0

            table_meta = self._ctx.catalog.get_table(
                mv.project_id,
                mv.dataset_id,
                mv.table_id,
            )
            if table_meta is not None:
                self._ctx.catalog.update_table(
                    table_meta.model_copy(
                        update={
                            "num_rows": num_rows,
                            "last_modified_time": now,
                        },
                    ),
                )
            refreshed = mv.model_copy(
                update={"last_refresh_time": now, "is_stale": False},
            )
            self._ctx.catalog.upsert_materialized_view(refreshed)

        return refreshed

    def _translate(self, bq_sql: str, project_id: str) -> str:
        """Translate + table-rewrite the MV source query."""
        from bqemulator.domain.result import Err, Ok

        match self._translator.translate(bq_sql):
            case Ok(duckdb_sql):
                pass
            case Err(error):
                raise error
        return rewrite_table_refs(duckdb_sql, project_id)

    def _read_arrow_schema(
        self,
        project_id: str,
        dataset_id: str,
        table_id: str,
    ) -> TableSchema:
        """Inspect the freshly-materialised table to build a TableSchema."""
        from bqemulator.storage.arrow_bridge import (
            arrow_type_to_bq_type_name,
            introspect_arrow_schema,
        )

        target_ref = quoted_table_ref(project_id, dataset_id, table_id)
        schema = introspect_arrow_schema(self._ctx.engine, target_ref)
        fields = tuple(
            TableFieldSchema(
                name=schema.field(i).name,
                type=arrow_type_to_bq_type_name(schema.field(i).type),
                mode="NULLABLE",
            )
            for i in range(len(schema))
        )
        return TableSchema(fields=fields)


# ---------------------------------------------------------------------------
# Dependency extraction
# ---------------------------------------------------------------------------


def _parse_view_query(view_query: str) -> exp.Expr:
    """Parse the MV body as BigQuery SQL; :class:`InvalidQueryError` when it cannot be."""
    try:
        return sqlglot.parse_one(view_query, read="bigquery")
    except Exception as exc:
        raise InvalidQueryError(
            f"Cannot parse materialized-view query: {exc}",
        ) from exc


def extract_base_tables(
    view_query: str | exp.Expr,
    project_id: str,
) -> list[tuple[str, str, str]]:
    """Walk ``view_query`` (SQL text or its parsed tree) and return every base-table reference.

    Returns a list of ``(project, dataset, table)`` triples, deduplicated
    while preserving iteration order. Raises :class:`InvalidQueryError`
    if the query cannot be parsed.
    """
    tree = _parse_view_query(view_query) if isinstance(view_query, str) else view_query

    seen: set[tuple[str, str, str]] = set()
    out: list[tuple[str, str, str]] = []
    for table_node in tree.find_all(exp.Table):
        if isinstance(table_node.this, exp.Anonymous):
            # TVF call — not a base table.
            continue
        table_name = table_node.name
        dataset_name = table_node.db
        project_name = table_node.catalog or project_id
        if not table_name or not dataset_name:
            continue
        key = (project_name, dataset_name, table_name)
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


# What BigQuery says when a materialized view reads something other than a
# plain table (CLONE is a plain table once created, so it is allowed).
_BASE_TABLE_KIND_PROBLEM: dict[str, str] = {
    "VIEW": "logical views are not supported as base tables",
    "MATERIALIZED_VIEW": "materialized views cannot be nested",
    "SNAPSHOT": "table snapshots are not supported as base tables",
    "EXTERNAL": "external tables are not supported as base tables",
}

_TIME_UNITS = frozenset({"DAY", "HOUR", "MONTH", "YEAR"})


def _partitioning(
    partition_by: str | None,
) -> tuple[TimePartitioning | None, RangePartitioning | None]:
    """Map a ``PARTITION BY`` expression to the catalog's partitioning models.

    Accepts the forms BigQuery allows on a materialized view: a bare
    ``DATE`` column, ``DATE(col)``, ``{TIMESTAMP,DATETIME,DATE}_TRUNC(col,
    unit)``, ``_PARTITIONDATE`` / ``_PARTITIONTIME`` (ingestion time) and
    ``RANGE_BUCKET(col, GENERATE_ARRAY(start, end, interval))``.
    """
    if partition_by is None:
        return None, None
    try:
        expr = sqlglot.parse_one(partition_by, read="bigquery")
    except Exception as exc:
        raise InvalidQueryError(f"Cannot parse PARTITION BY {partition_by}: {exc}") from exc
    if isinstance(expr, (exp.Column, exp.Identifier)):
        name = expr.name
        if name.upper() in {"_PARTITIONDATE", "_PARTITIONTIME"}:
            return TimePartitioning(type="DAY", field=None), None
        return TimePartitioning(type="DAY", field=name), None
    if isinstance(expr, exp.Date) and isinstance(expr.this, exp.Column):
        return TimePartitioning(type="DAY", field=expr.this.name), None
    if isinstance(expr, (exp.TimestampTrunc, exp.DatetimeTrunc, exp.DateTrunc)) and isinstance(
        expr.this, exp.Column
    ):
        unit = expr.args.get("unit")
        unit_name = unit.name.upper() if isinstance(unit, exp.Expr) else "DAY"
        if unit_name not in _TIME_UNITS:
            raise InvalidQueryError(
                f"PARTITION BY {partition_by}: a materialized view can only be partitioned "
                "by DAY, HOUR, MONTH or YEAR",
            )
        return TimePartitioning(type=unit_name, field=expr.this.name), None  # type: ignore[arg-type]
    range_partitioning = _range_partitioning(expr)
    if range_partitioning is not None:
        return None, range_partitioning
    raise InvalidQueryError(
        f"PARTITION BY {partition_by} is not a valid materialized view partitioning "
        "expression (use a DATE column, DATE(col), TIMESTAMP_TRUNC(col, unit) or RANGE_BUCKET)",
    )


def _range_partitioning(expr: exp.Expr) -> RangePartitioning | None:
    """``RANGE_BUCKET(col, GENERATE_ARRAY(start, end, interval))`` with literal bounds."""
    if not (isinstance(expr, exp.Anonymous) and expr.name.upper() == "RANGE_BUCKET"):
        return None
    args = expr.expressions
    column_ok = len(args) == 2 and isinstance(args[0], exp.Column)  # noqa: PLR2004
    generate = args[1] if column_ok else None
    if not (
        isinstance(generate, exp.GenerateSeries)
        or (isinstance(generate, exp.Anonymous) and generate.name.upper() == "GENERATE_ARRAY")
    ):
        return None
    if isinstance(generate, exp.GenerateSeries):
        bounds = [generate.args.get("start"), generate.args.get("end"), generate.args.get("step")]
    else:
        bounds = list(generate.expressions)
    if len(bounds) != 3 or not all(isinstance(b, exp.Literal) and b.is_int for b in bounds):  # noqa: PLR2004
        return None
    start, end, interval = (int(b.this) for b in bounds)  # type: ignore[union-attr]
    return RangePartitioning(field=args[0].name, start=start, end=end, interval=interval)


def _check_layout_columns(
    schema: TableSchema,
    *,
    cluster_by: tuple[str, ...],
    time_partitioning: TimePartitioning | None,
    range_partitioning: RangePartitioning | None,
) -> None:
    """``CLUSTER BY`` / ``PARTITION BY`` columns must be outputs of the query."""
    outputs = {f.name for f in schema.fields}
    for column in cluster_by:
        if column not in outputs:
            raise InvalidQueryError(
                f"CLUSTER BY column {column} is not an output column of the materialized view",
            )
    partition_field = (time_partitioning.field if time_partitioning is not None else None) or (
        range_partitioning.field if range_partitioning is not None else None
    )
    if partition_field is not None and partition_field not in outputs:
        raise InvalidQueryError(
            f"PARTITION BY column {partition_field} is not an output column of the "
            "materialized view",
        )


# ---------------------------------------------------------------------------
# Event-bus subscription wiring
# ---------------------------------------------------------------------------


def _subscribe_mv(ctx: AppContext, mv: MaterializedViewMeta) -> None:
    """Register a ``TableDataChanged`` handler that marks ``mv`` stale."""
    handler = _make_stale_handler(ctx, mv)
    _register_handler(ctx, mv, handler)
    ctx.events.subscribe(TableDataChanged, handler)


def _unsubscribe_mv(
    ctx: AppContext,
    project_id: str,
    dataset_id: str,
    table_id: str,
) -> None:
    """Remove the ``TableDataChanged`` handler for a dropped MV."""
    handlers = _handler_registry(ctx)
    key = (project_id, dataset_id, table_id)
    handler = handlers.pop(key, None)
    if handler is not None:
        ctx.events.unsubscribe(TableDataChanged, handler)


def _make_stale_handler(
    ctx: AppContext,
    mv: MaterializedViewMeta,
) -> Callable[[DomainEvent], None]:
    """Return a handler that flips ``is_stale=True`` on base-table changes."""
    base_set = {tuple(b) for b in mv.base_tables}
    key = (mv.project_id, mv.dataset_id, mv.table_id)

    def _handler(event: DomainEvent) -> None:
        if not isinstance(event, TableDataChanged):
            return
        if (event.project_id, event.dataset_id, event.table_id) not in base_set:
            return
        fresh = ctx.catalog.get_materialized_view(*key)
        if fresh is None or fresh.is_stale:
            return
        ctx.catalog.upsert_materialized_view(
            fresh.model_copy(update={"is_stale": True}),
        )

    return _handler


_MV_HANDLER_REGISTRY: dict[
    int,
    dict[tuple[str, str, str], Callable[[DomainEvent], None]],
] = {}


def _handler_registry(ctx: AppContext) -> dict[tuple[str, str, str], Callable[[DomainEvent], None]]:
    """Return (creating if needed) the per-context MV handler registry.

    Keyed by ``id(ctx)`` so each ``EmulatorServer`` instance has its own
    registry without mutating the frozen :class:`AppContext`. The entry
    is cleared via :func:`clear_subscriptions_for_context` when the
    server stops.
    """
    key = id(ctx)
    registry = _MV_HANDLER_REGISTRY.get(key)
    if registry is None:
        registry = {}
        _MV_HANDLER_REGISTRY[key] = registry
    return registry


def clear_subscriptions_for_context(ctx: AppContext) -> None:
    """Remove every MV subscription attached to ``ctx``.

    Called by the composition root during shutdown so stopped servers
    don't keep dead handlers alive.
    """
    registry = _MV_HANDLER_REGISTRY.pop(id(ctx), None)
    if registry is None:
        return
    for handler in registry.values():
        ctx.events.unsubscribe(TableDataChanged, handler)


def _register_handler(
    ctx: AppContext,
    mv: MaterializedViewMeta,
    handler: Callable[[DomainEvent], None],
) -> None:
    reg = _handler_registry(ctx)
    reg[(mv.project_id, mv.dataset_id, mv.table_id)] = handler


def hydrate_subscriptions(ctx: AppContext) -> None:
    """Rebuild MV event subscriptions from the catalog.

    Called by the composition root on startup so restarts don't lose
    staleness tracking for persisted MVs.
    """
    for mv in ctx.catalog.list_all_materialized_views():
        _subscribe_mv(ctx, mv)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


__all__ = [
    "MaterializedViewManager",
    "clear_subscriptions_for_context",
    "extract_base_tables",
    "hydrate_subscriptions",
]
