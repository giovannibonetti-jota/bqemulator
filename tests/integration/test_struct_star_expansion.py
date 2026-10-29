"""End-to-end ``expr.*`` expansion through ``jobs.query`` and a view.

The dbt model that motivated the pass projects
``ARRAY(SELECT AS STRUCT <aggregates> FROM UNNEST(...))[OFFSET(0)].*``
from a view: the struct's fields have to become view columns with
their own names and types, and the typed ``ARRAY<STRUCT>`` column
case has to resolve its names from the catalog.
"""

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
    """Start an in-process ephemeral emulator and create dataset ``p.ds``."""
    settings = Settings(persistence_mode=PersistenceMode.EPHEMERAL, rest_port=0, grpc_port=0)
    s = EmulatorServer(settings)
    await s.start()
    try:
        async with httpx.AsyncClient(base_url=s.rest_url, timeout=20.0) as c:
            r = await c.post(
                "/bigquery/v2/projects/p/datasets",
                json={"datasetReference": {"projectId": "p", "datasetId": "ds"}},
            )
            r.raise_for_status()
            yield c
    finally:
        await s.stop()


async def _run(client: httpx.AsyncClient, sql: str) -> dict[str, Any]:
    """Run *sql* through ``jobs.query`` and return the response payload."""
    r = await client.post(
        "/bigquery/v2/projects/p/queries",
        json={"query": sql, "useLegacySql": False},
    )
    r.raise_for_status()
    return r.json()


def _rows(payload: dict[str, Any]) -> list[list[Any]]:
    """Flatten a ``jobs.query`` response into ``[[cell, …], …]``."""
    return [[c["v"] for c in row["f"]] for row in payload.get("rows", [])]


def _fields(payload: dict[str, Any]) -> list[tuple[str, str]]:
    return [(f["name"], f["type"]) for f in payload["schema"]["fields"]]


async def test_array_subquery_element_star_in_a_view(client: httpx.AsyncClient) -> None:
    await _run(client, "CREATE TABLE ds.req (id INT64, query_param_values STRING)")
    await _run(client, "INSERT INTO ds.req VALUES (1, 'a=1&b=2'), (2, NULL), (3, '')")
    await _run(
        client,
        "CREATE VIEW ds.req_view AS SELECT id, ARRAY("
        "  SELECT AS STRUCT STRING_AGG(p, '&') AS query_string, COUNT(*) AS param_count"
        "  FROM UNNEST(SPLIT(query_param_values, '&')) AS p WHERE LENGTH(p) > 0"
        ")[OFFSET(0)].*, query_param_values AS raw FROM ds.req",
    )

    payload = await _run(client, "SELECT * FROM ds.req_view ORDER BY id")
    assert _fields(payload) == [
        ("id", "INTEGER"),
        ("query_string", "STRING"),
        ("param_count", "INTEGER"),
        ("raw", "STRING"),
    ]
    assert _rows(payload) == [
        ["1", "a=1&b=2", "2", "a=1&b=2"],
        ["2", None, "0", None],
        ["3", None, "0", ""],
    ]

    filtered = await _run(client, "SELECT id FROM ds.req_view WHERE query_string IS NOT NULL")
    assert _rows(filtered) == [["1"]]


async def test_typed_array_struct_column_star(client: httpx.AsyncClient) -> None:
    await _run(
        client,
        "CREATE TABLE ds.events (id INT64, tags ARRAY<STRUCT<name STRING, weight INT64>>)",
    )
    await _run(
        client,
        "INSERT INTO ds.events VALUES (1, [STRUCT('x', 10), STRUCT('y', 20)]), (2, [])",
    )

    payload = await _run(
        client, "SELECT id, tags[SAFE_OFFSET(0)].* EXCEPT (weight) FROM ds.events ORDER BY id"
    )
    assert _fields(payload) == [("id", "INTEGER"), ("name", "STRING")]
    assert _rows(payload) == [["1", "x"], ["2", None]]


async def test_unnamed_struct_fields_are_reported_as_unsupported(
    client: httpx.AsyncClient,
) -> None:
    r = await client.post(
        "/bigquery/v2/projects/p/queries",
        json={"query": "SELECT STRUCT(1, 2).*", "useLegacySql": False},
    )
    assert r.status_code == 501
    assert "Dot-star expansion" in r.json()["error"]["message"]
