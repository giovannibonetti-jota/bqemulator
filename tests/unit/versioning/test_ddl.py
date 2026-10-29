"""Unit tests for the Phase 7 DDL router."""

from __future__ import annotations

import pytest

from bqemulator.domain.errors import InvalidQueryError
from bqemulator.versioning.ddl import (
    VersioningDDLKind,
    VersioningDDLRouter,
    is_versioning_ddl,
)

pytestmark = pytest.mark.unit


def test_is_versioning_ddl_keywords() -> None:
    assert is_versioning_ddl("CREATE SNAPSHOT TABLE x CLONE y")
    assert is_versioning_ddl("create table x clone y")
    assert is_versioning_ddl("CREATE MATERIALIZED VIEW v AS SELECT 1")
    assert is_versioning_ddl("REFRESH MATERIALIZED VIEW v")
    assert is_versioning_ddl("CALL BQ.REFRESH_MATERIALIZED_VIEW('p.ds.v')")
    assert is_versioning_ddl("call bq.refresh_materialized_view('p.ds.v')")
    assert not is_versioning_ddl("SELECT 1")
    assert not is_versioning_ddl("INSERT INTO ds.t VALUES (1)")


def test_parse_create_snapshot_table_dataset_qualified() -> None:
    router = VersioningDDLRouter("p")
    parsed = router.parse(
        "CREATE SNAPSHOT TABLE ds.copy_t CLONE ds.t",
    )
    assert parsed is not None
    assert parsed.kind is VersioningDDLKind.CREATE_SNAPSHOT
    assert parsed.target_project == "p"
    assert parsed.target_dataset == "ds"
    assert parsed.target_table == "copy_t"
    assert parsed.source_project == "p"
    assert parsed.source_dataset == "ds"
    assert parsed.source_table == "t"


def test_parse_create_snapshot_table_fully_qualified() -> None:
    router = VersioningDDLRouter("default")
    parsed = router.parse(
        "CREATE SNAPSHOT TABLE `proj.ds.copy_t` CLONE `proj.ds.t`",
    )
    assert parsed is not None
    assert parsed.target_project == "proj"
    assert parsed.source_project == "proj"


def test_parse_drop_snapshot_table_if_exists() -> None:
    router = VersioningDDLRouter("p")
    parsed = router.parse("DROP SNAPSHOT TABLE IF EXISTS ds.copy_t")
    assert parsed is not None
    assert parsed.kind is VersioningDDLKind.DROP_SNAPSHOT
    assert parsed.target_table == "copy_t"


def test_parse_create_clone() -> None:
    router = VersioningDDLRouter("p")
    parsed = router.parse("CREATE TABLE ds.workcopy CLONE ds.t;")
    assert parsed is not None
    assert parsed.kind is VersioningDDLKind.CREATE_CLONE
    assert parsed.source_dataset == "ds"
    assert parsed.target_table == "workcopy"


def test_parse_create_materialized_view_with_subquery() -> None:
    router = VersioningDDLRouter("p")
    parsed = router.parse(
        """CREATE MATERIALIZED VIEW ds.daily AS
        SELECT DATE(placed_at) AS day, SUM(amount) AS total
        FROM ds.orders GROUP BY day""",
    )
    assert parsed is not None
    assert parsed.kind is VersioningDDLKind.CREATE_MATERIALIZED_VIEW
    assert parsed.view_query is not None
    assert "SUM(amount)" in parsed.view_query


def test_parse_refresh_materialized_view() -> None:
    router = VersioningDDLRouter("p")
    parsed = router.parse("REFRESH MATERIALIZED VIEW ds.daily")
    assert parsed is not None
    assert parsed.kind is VersioningDDLKind.REFRESH_MATERIALIZED_VIEW
    assert parsed.target_dataset == "ds"


def test_parse_refresh_materialized_view_call_form() -> None:
    router = VersioningDDLRouter("p")
    parsed = router.parse("CALL BQ.REFRESH_MATERIALIZED_VIEW('p.ds.daily')")
    assert parsed is not None
    assert parsed.kind is VersioningDDLKind.REFRESH_MATERIALIZED_VIEW
    assert parsed.target_project == "p"
    assert parsed.target_dataset == "ds"
    assert parsed.target_table == "daily"


def test_parse_drop_materialized_view() -> None:
    router = VersioningDDLRouter("p")
    parsed = router.parse("DROP MATERIALIZED VIEW IF EXISTS ds.daily")
    assert parsed is not None
    assert parsed.kind is VersioningDDLKind.DROP_MATERIALIZED_VIEW


def test_parse_returns_none_for_non_versioning_sql() -> None:
    router = VersioningDDLRouter("p")
    assert router.parse("SELECT 1") is None
    assert router.parse("INSERT INTO ds.t VALUES (1)") is None
    assert router.parse("CREATE TABLE ds.t (x INT64)") is None


def test_parse_rejects_bare_table_reference() -> None:
    router = VersioningDDLRouter("p")
    with pytest.raises(InvalidQueryError):
        router.parse("CREATE TABLE single CLONE other_single")


def test_parse_create_materialized_view_dbt_shape() -> None:
    """The ``IF NOT EXISTS`` / ``PARTITION BY`` / ``CLUSTER BY`` / ``OPTIONS`` shape dbt emits."""
    router = VersioningDDLRouter("p")
    parsed = router.parse(
        """
        create materialized view if not exists `q`.`ds`.`mv`
        partition by DATE(CAST(ts AS TIMESTAMP))
        cluster by created_at, country
        OPTIONS(
          enable_refresh=True,
          refresh_interval_minutes=30,
          description='as is',
          max_staleness=INTERVAL "0:30:0" HOUR TO SECOND,
          allow_non_incremental_definition=true
        )
        as (SELECT *, TIMESTAMP_MICROS(id >> 11) AS created_at FROM `q`.`ds`.`orders`);
        """,
    )
    assert parsed is not None
    assert parsed.kind is VersioningDDLKind.CREATE_MATERIALIZED_VIEW
    assert (parsed.target_project, parsed.target_dataset, parsed.target_table) == ("q", "ds", "mv")
    assert parsed.if_not_exists is True
    assert parsed.replace is False
    assert parsed.partition_by == "DATE(CAST(ts AS TIMESTAMP))"
    assert parsed.cluster_by == ("created_at", "country")
    assert parsed.options == {
        "enable_refresh": True,
        "refresh_interval_minutes": 30,
        "description": "as is",
        "max_staleness": "INTERVAL '0:30:0' HOUR TO SECOND",
        "allow_non_incremental_definition": True,
    }
    # The body is the caller's text, not a SQLGlot re-rendering.
    assert parsed.view_query == (
        "(SELECT *, TIMESTAMP_MICROS(id >> 11) AS created_at FROM `q`.`ds`.`orders`)"
    )


def test_parse_create_or_replace_materialized_view() -> None:
    router = VersioningDDLRouter("p")
    parsed = router.parse(
        "CREATE OR REPLACE MATERIALIZED VIEW `my-proj.ds.mv` AS SELECT a FROM ds.t",
    )
    assert parsed is not None
    assert parsed.replace is True
    assert parsed.if_not_exists is False
    assert parsed.target_project == "my-proj"
    assert parsed.view_query == "SELECT a FROM ds.t"
    assert parsed.options == {}


def test_parse_create_materialized_view_without_body() -> None:
    router = VersioningDDLRouter("p")
    with pytest.raises(InvalidQueryError, match="AS <query>"):
        router.parse("CREATE MATERIALIZED VIEW ds.mv OPTIONS(enable_refresh=true)")


def test_parse_leaves_plain_views_and_tables_alone() -> None:
    router = VersioningDDLRouter("p")
    assert router.parse("CREATE VIEW ds.v AS SELECT 1") is None
    assert router.parse("CREATE TABLE ds.t AS SELECT 1") is None
