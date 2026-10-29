"""Integration tests for ``CREATE SNAPSHOT TABLE`` / ``DROP SNAPSHOT TABLE``."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
import pytest_asyncio

from bqemulator.config import PersistenceMode, Settings
from bqemulator.server import EmulatorServer

pytestmark = pytest.mark.integration


@pytest_asyncio.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    settings = Settings(
        persistence_mode=PersistenceMode.EPHEMERAL,
        rest_port=0,
        grpc_port=0,
    )
    s = EmulatorServer(settings)
    await s.start()
    try:
        async with httpx.AsyncClient(base_url=s.rest_url, timeout=20.0) as c:
            await c.post(
                "/bigquery/v2/projects/p/datasets",
                json={"datasetReference": {"projectId": "p", "datasetId": "ds"}},
            )
            await c.post(
                "/bigquery/v2/projects/p/datasets/ds/tables",
                json={
                    "tableReference": {
                        "projectId": "p",
                        "datasetId": "ds",
                        "tableId": "t",
                    },
                    "schema": {
                        "fields": [
                            {"name": "id", "type": "INT64"},
                            {"name": "val", "type": "STRING"},
                        ],
                    },
                },
            )
            yield c
    finally:
        await s.stop()


async def _run(client: httpx.AsyncClient, sql: str) -> dict[str, Any]:
    r = await client.post(
        "/bigquery/v2/projects/p/queries",
        json={"query": sql, "useLegacySql": False},
    )
    r.raise_for_status()
    return r.json()


def _rows(payload: dict[str, Any]) -> list[list[str]]:
    return [[c["v"] for c in row["f"]] for row in payload.get("rows", [])]


async def test_create_snapshot_table_captures_point_in_time_copy(
    client: httpx.AsyncClient,
) -> None:
    await _run(client, "INSERT INTO ds.t VALUES (1, 'a'), (2, 'b')")
    await _run(client, "CREATE SNAPSHOT TABLE ds.t_snap CLONE ds.t")
    await _run(client, "INSERT INTO ds.t VALUES (3, 'c')")

    snap_rows = await _run(client, "SELECT id FROM ds.t_snap ORDER BY id")
    live_rows = await _run(client, "SELECT id FROM ds.t ORDER BY id")
    assert _rows(snap_rows) == [["1"], ["2"]]
    assert _rows(live_rows) == [["1"], ["2"], ["3"]]


async def test_snapshot_table_listed_with_snapshot_type(
    client: httpx.AsyncClient,
) -> None:
    await _run(client, "INSERT INTO ds.t VALUES (1, 'a')")
    await _run(client, "CREATE SNAPSHOT TABLE ds.t_snap CLONE ds.t")

    r = await client.get("/bigquery/v2/projects/p/datasets/ds/tables")
    r.raise_for_status()
    tables = {t["tableReference"]["tableId"]: t for t in r.json()["tables"]}
    assert tables["t_snap"]["type"] == "SNAPSHOT"


async def test_drop_snapshot_table_removes_it(
    client: httpx.AsyncClient,
) -> None:
    await _run(client, "INSERT INTO ds.t VALUES (1, 'a')")
    await _run(client, "CREATE SNAPSHOT TABLE ds.t_snap CLONE ds.t")
    await _run(client, "DROP SNAPSHOT TABLE ds.t_snap")

    r = await client.get("/bigquery/v2/projects/p/datasets/ds/tables")
    table_ids = [t["tableReference"]["tableId"] for t in r.json()["tables"]]
    assert "t_snap" not in table_ids


async def test_snapshot_table_immutable_after_source_drops(
    client: httpx.AsyncClient,
) -> None:
    """A user snapshot is durable even after the source table goes away."""
    await _run(client, "INSERT INTO ds.t VALUES (1, 'a'), (2, 'b')")
    await _run(client, "CREATE SNAPSHOT TABLE ds.t_snap CLONE ds.t")

    # Drop the source.
    r = await client.delete("/bigquery/v2/projects/p/datasets/ds/tables/t")
    assert r.status_code == 204

    rows = await _run(client, "SELECT id FROM ds.t_snap ORDER BY id")
    assert _rows(rows) == [["1"], ["2"]]


async def test_dml_against_snapshot_rejected(client: httpx.AsyncClient) -> None:
    """``INSERT INTO snapshot_table`` is rejected with a clear error.

    P3.a / ADR 0022 §3: SQL execution errors surface as HTTP 200 with
    the job's ``errors[0]`` (real BigQuery wire behaviour), not as a
    direct 4xx response.
    """
    await _run(client, "INSERT INTO ds.t VALUES (1, 'a')")
    await _run(client, "CREATE SNAPSHOT TABLE ds.t_snap CLONE ds.t")
    r = await client.post(
        "/bigquery/v2/projects/p/queries",
        json={"query": "INSERT INTO ds.t_snap VALUES (99, 'z')", "useLegacySql": False},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["errors"][0]["reason"] == "invalidQuery"
    assert "immutable" in body["errors"][0]["message"].lower()


async def test_snapshot_ddl_full_grammar(client: httpx.AsyncClient) -> None:
    """``IF NOT EXISTS`` / ``OR REPLACE`` / ``OPTIONS`` / ``IF EXISTS`` route to the manager.

    The regex this replaced knew only the bare form, so each of these fell
    through to DuckDB and failed. The dbt-style leading comment is part of
    the shape under test.
    """
    await _run(client, "INSERT INTO ds.t VALUES (1, 'a')")
    await _run(
        client,
        '/* {"app": "dbt"} */ CREATE SNAPSHOT TABLE IF NOT EXISTS `p.ds.t_snap` CLONE ds.t '
        "OPTIONS(description='nightly copy', labels=[('env', 'test')], "
        "expiration_timestamp=TIMESTAMP '2099-01-01')",
    )
    # IF NOT EXISTS is a no-op the second time.
    await _run(client, "CREATE SNAPSHOT TABLE IF NOT EXISTS ds.t_snap CLONE ds.t")
    r = await client.get("/bigquery/v2/projects/p/datasets/ds/tables/t_snap")
    assert r.status_code == 200
    body = r.json()
    assert body["type"] == "SNAPSHOT"
    assert body["description"] == "nightly copy"
    assert body["labels"] == {"env": "test"}
    # OR REPLACE re-snapshots the source's current rows.
    await _run(client, "INSERT INTO ds.t VALUES (2, 'b')")
    await _run(client, "CREATE OR REPLACE SNAPSHOT TABLE ds.t_snap CLONE ds.t")
    rows = _rows(await _run(client, "SELECT id FROM ds.t_snap ORDER BY id"))
    assert rows == [["1"], ["2"]]
    await _run(client, "DROP SNAPSHOT TABLE IF EXISTS ds.t_snap;")
    await _run(client, "DROP SNAPSHOT TABLE IF EXISTS ds.t_snap")  # missing: no-op
    r = await client.get("/bigquery/v2/projects/p/datasets/ds/tables/t_snap")
    assert r.status_code == 404


async def test_snapshot_for_system_time_is_refused_explicitly(client: httpx.AsyncClient) -> None:
    """``FOR SYSTEM_TIME AS OF`` parses and is refused as unsupported, not as DuckDB syntax."""
    r = await client.post(
        "/bigquery/v2/projects/p/queries",
        json={
            "query": "CREATE SNAPSHOT TABLE ds.t_snap CLONE ds.t FOR SYSTEM_TIME AS OF "
            "TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 HOUR)",
            "useLegacySql": False,
        },
    )
    body = r.json()
    error = body["errors"][0] if r.status_code == 200 else body["error"]["errors"][0]
    assert error["reason"] == "notImplemented"
    assert "FOR SYSTEM_TIME AS OF" in error["message"]


async def test_malformed_snapshot_ddl_is_a_syntax_error(client: httpx.AsyncClient) -> None:
    """A statement that starts like the head but does not parse reports BigQuery's wording."""
    r = await client.post(
        "/bigquery/v2/projects/p/queries",
        json={"query": "CREATE SNAPSHOT TABLE ds.t_snap FROM ds.t", "useLegacySql": False},
    )
    body = r.json()
    error = body["errors"][0] if r.status_code == 200 else body["error"]["errors"][0]
    assert error["reason"] == "invalidQuery"
    assert error["message"].startswith("Syntax error: CREATE SNAPSHOT TABLE")
