"""``CREATE TABLE ... CLONE`` — physical copy with clone table-type.

BigQuery clones share storage with the source until the first DML
against the clone, then diverge. DuckDB has no copy-on-write primitive,
so we do a physical ``CREATE TABLE AS SELECT`` and label the resulting
table with ``table_type=CLONE`` + ``base_table`` so REST responses and
``tables.list`` distinguish clones from regular tables. The
user-observable semantics match BigQuery: writes to the clone never
touch the source.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from bqemulator.catalog.etag import generate_etag
from bqemulator.catalog.models import TableMeta, TableSchema
from bqemulator.domain.errors import (
    ResourceRef,
    UnsupportedFeatureError,
    resource_already_exists,
    resource_not_found,
)
from bqemulator.storage.sql_identifiers import quoted_table_ref
from bqemulator.versioning.ddl import table_metadata_options

if TYPE_CHECKING:  # pragma: no cover
    from bqemulator.api.dependencies import AppContext


#: Table types ``CREATE OR REPLACE TABLE … CLONE`` may overwrite.
_REPLACEABLE_TABLE_TYPES = frozenset({"TABLE", "CLONE"})


class CloneManager:
    """Handles ``CREATE TABLE ... CLONE`` DDL."""

    def __init__(self, ctx: AppContext) -> None:
        self._ctx = ctx

    async def create(
        self,
        target_project: str,
        target_dataset: str,
        target_table: str,
        source_project: str,
        source_dataset: str,
        source_table: str,
        *,
        replace: bool = False,
        if_not_exists: bool = False,
        system_time: str | None = None,
        options: Mapping[str, object] | None = None,
    ) -> TableMeta:
        """Materialise a clone and register it in the catalog.

        ``if_not_exists`` returns the existing table untouched;
        ``replace`` drops an existing plain table or clone first (as in
        BigQuery, a snapshot or materialized view in the way is still an
        "already exists" error). ``options`` is the lower-cased
        ``OPTIONS(...)`` clause (see :func:`table_metadata_options`).
        ``system_time`` (``FOR SYSTEM_TIME AS OF``) is parsed but not yet
        served: the clone always copies the current state, so it is
        refused rather than silently ignored.
        """
        if system_time is not None:
            raise UnsupportedFeatureError(
                "CREATE TABLE ... CLONE ... FOR SYSTEM_TIME AS OF is not supported; "
                "bqemulator clones the table's current state only.",
            )
        src_meta = self._ctx.catalog.get_table(
            source_project,
            source_dataset,
            source_table,
        )
        if src_meta is None:
            raise resource_not_found(
                ResourceRef("table", source_project, source_dataset, source_table),
            )

        dst_meta = self._ctx.catalog.get_table(
            target_project,
            target_dataset,
            target_table,
        )
        if dst_meta is not None:
            if if_not_exists:
                return dst_meta
            if not (replace and dst_meta.table_type in _REPLACEABLE_TABLE_TYPES):
                raise resource_already_exists(
                    ResourceRef("table", target_project, target_dataset, target_table),
                )

        # Ensure the destination dataset exists; clones land inside the
        # dataset's regular schema just like normal tables.
        if self._ctx.catalog.get_dataset(target_project, target_dataset) is None:
            raise resource_not_found(
                ResourceRef("dataset", target_project, target_dataset),
            )

        src_ref = quoted_table_ref(source_project, source_dataset, source_table)
        dst_ref = quoted_table_ref(target_project, target_dataset, target_table)

        now = self._ctx.clock.now()
        table_options = table_metadata_options(options or {})

        async with self._ctx.engine.write_lock():
            if dst_meta is not None:
                self._ctx.engine.execute(f"DROP TABLE IF EXISTS {dst_ref}")
                self._ctx.catalog.delete_table(
                    target_project,
                    target_dataset,
                    target_table,
                    not_found_ok=True,
                )
            self._ctx.engine.execute(
                f"CREATE TABLE {dst_ref} AS SELECT * FROM {src_ref}",
            )
            count_row = self._ctx.engine.execute(
                f"SELECT COUNT(*) FROM {dst_ref}",
            ).fetchone()
            num_rows = int(count_row[0]) if count_row else 0

            # Preserve the source's schema, partitioning, and clustering.
            meta = TableMeta(
                project_id=target_project,
                dataset_id=target_dataset,
                table_id=target_table,
                table_type="CLONE",
                schema=(src_meta.schema_ or TableSchema()),
                description=table_options.description,
                friendly_name=table_options.friendly_name,
                labels=table_options.labels,
                time_partitioning=src_meta.time_partitioning,
                range_partitioning=src_meta.range_partitioning,
                clustering=src_meta.clustering,
                creation_time=now,
                last_modified_time=now,
                num_rows=num_rows,
                num_bytes=0,
                etag=generate_etag(
                    target_project,
                    target_dataset,
                    target_table,
                    "CLONE",
                    str(now),
                ),
                base_table=(f"{source_project}.{source_dataset}.{source_table}"),
            )
            self._ctx.catalog.create_table(meta)

        return meta


__all__ = ["CloneManager"]
