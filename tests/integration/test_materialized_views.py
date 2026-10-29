"""Integration tests for materialized views: create, refresh, INFORMATION_SCHEMA."""

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
                        "tableId": "orders",
                    },
                    "schema": {
                        "fields": [
                            {"name": "country", "type": "STRING"},
                            {"name": "amount", "type": "INT64"},
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


async def test_create_and_query_materialized_view(
    client: httpx.AsyncClient,
) -> None:
    await _run(client, "INSERT INTO ds.orders VALUES ('US', 10), ('US', 5), ('CA', 20)")
    await _run(
        client,
        "CREATE MATERIALIZED VIEW ds.country_totals AS "
        "SELECT country, SUM(amount) AS total FROM ds.orders GROUP BY country",
    )
    r = await _run(
        client,
        "SELECT country, total FROM ds.country_totals ORDER BY country",
    )
    assert _rows(r) == [["CA", "20"], ["US", "15"]]


async def test_mv_auto_refreshes_after_base_change(
    client: httpx.AsyncClient,
) -> None:
    await _run(client, "INSERT INTO ds.orders VALUES ('US', 10)")
    await _run(
        client,
        "CREATE MATERIALIZED VIEW ds.totals AS SELECT SUM(amount) AS total FROM ds.orders",
    )
    # Add a row — staleness should trigger a recompute on the next read.
    await _run(client, "INSERT INTO ds.orders VALUES ('CA', 90)")

    r = await _run(client, "SELECT total FROM ds.totals")
    assert _rows(r) == [["100"]]


async def test_explicit_refresh_recomputes(client: httpx.AsyncClient) -> None:
    await _run(client, "INSERT INTO ds.orders VALUES ('US', 1)")
    await _run(
        client,
        "CREATE MATERIALIZED VIEW ds.totals AS SELECT SUM(amount) AS total FROM ds.orders",
    )
    await _run(client, "INSERT INTO ds.orders VALUES ('CA', 9)")
    await _run(client, "REFRESH MATERIALIZED VIEW ds.totals")

    r = await _run(client, "SELECT total FROM ds.totals")
    assert _rows(r) == [["10"]]


async def test_information_schema_materialized_views(
    client: httpx.AsyncClient,
) -> None:
    await _run(client, "INSERT INTO ds.orders VALUES ('US', 10)")
    await _run(
        client,
        "CREATE MATERIALIZED VIEW ds.totals AS SELECT SUM(amount) AS total FROM ds.orders",
    )
    r = await _run(
        client,
        "SELECT table_name, is_stale FROM ds.INFORMATION_SCHEMA.MATERIALIZED_VIEWS",
    )
    rows = _rows(r)
    assert len(rows) == 1
    assert rows[0][0] == "totals"
    assert rows[0][1] in ("false", "FALSE", "0")  # boolean rendering varies


async def test_dml_against_mv_rejected(client: httpx.AsyncClient) -> None:
    """Direct DML against a materialized view is rejected — they refresh.

    P3.a / ADR 0022 §3: SQL execution errors surface as HTTP 200 with
    the job's ``errors[0]`` (real BigQuery wire behaviour), not as a
    direct 4xx response.
    """
    await _run(client, "INSERT INTO ds.orders VALUES ('US', 1)")
    await _run(
        client,
        "CREATE MATERIALIZED VIEW ds.totals AS SELECT SUM(amount) AS total FROM ds.orders",
    )
    r = await client.post(
        "/bigquery/v2/projects/p/queries",
        json={"query": "INSERT INTO ds.totals VALUES (999)", "useLegacySql": False},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["errors"][0]["reason"] == "invalidQuery"
    assert "immutable" in body["errors"][0]["message"].lower()


async def test_drop_materialized_view(client: httpx.AsyncClient) -> None:
    await _run(client, "INSERT INTO ds.orders VALUES ('US', 1)")
    await _run(
        client,
        "CREATE MATERIALIZED VIEW ds.totals AS SELECT SUM(amount) AS total FROM ds.orders",
    )
    await _run(client, "DROP MATERIALIZED VIEW ds.totals")
    r = await client.get("/bigquery/v2/projects/p/datasets/ds/tables")
    table_ids = [t["tableReference"]["tableId"] for t in r.json()["tables"]]
    assert "totals" not in table_ids


async def test_mv_with_join_refreshes_when_either_base_changes(
    client: httpx.AsyncClient,
) -> None:
    await client.post(
        "/bigquery/v2/projects/p/datasets/ds/tables",
        json={
            "tableReference": {"projectId": "p", "datasetId": "ds", "tableId": "rates"},
            "schema": {
                "fields": [
                    {"name": "country", "type": "STRING"},
                    {"name": "rate", "type": "FLOAT64"},
                ],
            },
        },
    )
    await _run(client, "INSERT INTO ds.orders VALUES ('US', 100), ('CA', 100)")
    await _run(client, "INSERT INTO ds.rates VALUES ('US', 1.0), ('CA', 0.7)")
    await _run(
        client,
        "CREATE MATERIALIZED VIEW ds.usd_totals AS "
        "SELECT o.country, SUM(o.amount * r.rate) AS usd "
        "FROM ds.orders AS o JOIN ds.rates AS r USING (country) GROUP BY o.country",
    )
    # Change a row in the rates side; MV should refresh.
    await _run(client, "UPDATE ds.rates SET rate = 0.5 WHERE country = 'CA'")

    r = await _run(
        client,
        "SELECT country, usd FROM ds.usd_totals ORDER BY country",
    )
    rows = _rows(r)
    assert rows[0][0] == "CA" and rows[0][1].startswith("50")
    assert rows[1][0] == "US" and rows[1][1].startswith("100")


async def _error(client: httpx.AsyncClient, sql: str) -> str:
    """Run ``sql`` expecting a job error; return its ``invalidQuery`` message."""
    r = await client.post(
        "/bigquery/v2/projects/p/queries",
        json={"query": sql, "useLegacySql": False},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["errors"][0]["reason"] == "invalidQuery", body
    return str(body["errors"][0]["message"])


async def _table_ids(client: httpx.AsyncClient) -> list[str]:
    r = await client.get("/bigquery/v2/projects/p/datasets/ds/tables")
    return [t["tableReference"]["tableId"] for t in r.json().get("tables", [])]


DBT_SHAPED_DDL = """
    create materialized view if not exists `p`.`ds`.`orders_with_day`
    partition by DATE(placed_at)
    cluster by country
    OPTIONS(
      enable_refresh=True,
      description='orders plus their day',
      refresh_interval_minutes=30
    )
    as
SELECT *, TIMESTAMP_MICROS(amount << 20) AS placed_at FROM `p`.`ds`.`orders`
"""


async def test_dbt_shaped_create_registers_the_view(client: httpx.AsyncClient) -> None:
    """``IF NOT EXISTS`` / ``PARTITION BY`` / ``CLUSTER BY`` / ``OPTIONS`` register a real MV.

    dbt-bigquery emits exactly this shape; before the SQLGlot-based parse
    it slipped past the versioning router and became an invisible DuckDB
    view.
    """
    await _run(client, "INSERT INTO ds.orders VALUES ('US', 10), ('CA', 20)")
    await _run(client, DBT_SHAPED_DDL)
    r = await client.get("/bigquery/v2/projects/p/datasets/ds/tables/orders_with_day")
    assert r.status_code == 200
    body = r.json()
    assert body["type"] == "MATERIALIZED_VIEW"
    assert body["clustering"] == {"fields": ["country"]}
    assert body["timePartitioning"]["type"] == "DAY"
    assert body["timePartitioning"]["field"] == "placed_at"
    assert body["description"] == "orders plus their day"
    rows = _rows(
        await _run(client, "SELECT country, placed_at FROM ds.orders_with_day ORDER BY country")
    )
    assert [row[0] for row in rows] == ["CA", "US"]
    # IF NOT EXISTS is a no-op the second time, not an "already exists" error.
    await _run(client, DBT_SHAPED_DDL)
    assert _table_ids(client) is not None
    schema_rows = _rows(
        await _run(
            client,
            "SELECT table_name, table_type FROM ds.INFORMATION_SCHEMA.TABLES "
            "WHERE table_name = 'orders_with_day'",
        ),
    )
    assert schema_rows == [["orders_with_day", "MATERIALIZED VIEW"]]


async def test_create_or_replace_swaps_the_query(client: httpx.AsyncClient) -> None:
    await _run(client, "INSERT INTO ds.orders VALUES ('US', 10), ('CA', 20)")
    await _run(
        client,
        "CREATE MATERIALIZED VIEW ds.totals AS SELECT SUM(amount) AS total FROM ds.orders",
    )
    await _run(
        client,
        "CREATE OR REPLACE MATERIALIZED VIEW ds.totals AS "
        "SELECT country, SUM(amount) AS total FROM ds.orders GROUP BY country",
    )
    rows = _rows(await _run(client, "SELECT country, total FROM ds.totals ORDER BY country"))
    assert rows == [["CA", "20"], ["US", "10"]]
    # A plain CREATE over an existing MV still conflicts.
    r = await client.post(
        "/bigquery/v2/projects/p/queries",
        json={
            "query": (
                "CREATE MATERIALIZED VIEW ds.totals AS SELECT SUM(amount) AS total FROM ds.orders"
            ),
            "useLegacySql": False,
        },
    )
    assert r.status_code == 200
    assert r.json()["errors"][0]["reason"] == "duplicate"


# One body per bullet of BigQuery's "Query limitations" for incremental
# views, plus the clauses outside its WITH/SELECT/FROM/WHERE/GROUP BY
# pattern. Each must fail at CREATE time and leave no table behind.
INCREMENTAL_REJECTED = [
    pytest.param(
        "SELECT country FROM ds.orders UNION ALL SELECT country FROM ds.rates",
        "UNION ALL",
        id="union-all",
    ),
    pytest.param(
        "SELECT o.country, SUM(o.amount) AS t FROM ds.orders AS o "
        "LEFT OUTER JOIN ds.rates AS r USING (country) GROUP BY o.country",
        "LEFT OUTER JOIN",
        id="left-outer-join",
    ),
    pytest.param(
        "SELECT o.country FROM ds.orders AS o RIGHT JOIN ds.rates AS r USING (country)",
        "RIGHT OUTER JOIN",
        id="right-join",
    ),
    pytest.param(
        "SELECT o.country FROM ds.orders AS o FULL OUTER JOIN ds.rates AS r USING (country)",
        "FULL OUTER JOIN",
        id="full-join",
    ),
    pytest.param(
        "SELECT a.country FROM ds.orders AS a JOIN ds.orders AS b USING (country)",
        "self-join",
        id="self-join",
    ),
    pytest.param(
        "SELECT country, ROW_NUMBER() OVER (ORDER BY amount) AS rn FROM ds.orders",
        "window function",
        id="window-function",
    ),
    pytest.param(
        "SELECT country, ARRAY(SELECT x FROM UNNEST([amount]) AS x) AS xs FROM ds.orders",
        "ARRAY subquery",
        id="array-subquery",
    ),
    pytest.param("SELECT country, RAND() AS r FROM ds.orders", "RAND()", id="rand"),
    pytest.param(
        "SELECT country, CURRENT_DATE() AS d FROM ds.orders", "CURRENT_DATE()", id="current-date"
    ),
    pytest.param(
        "SELECT country, SESSION_USER() AS u FROM ds.orders", "SESSION_USER()", id="session-user"
    ),
    pytest.param(
        "SELECT country, CURRENT_TIME() AS t FROM ds.orders", "CURRENT_TIME()", id="current-time"
    ),
    pytest.param("SELECT ds.shout(country) AS c FROM ds.orders", "user-defined function", id="udf"),
    pytest.param(
        "SELECT country FROM ds.orders TABLESAMPLE SYSTEM (10 PERCENT)",
        "TABLESAMPLE",
        id="tablesample",
    ),
    pytest.param(
        "SELECT country FROM ds.orders FOR SYSTEM_TIME AS OF TIMESTAMP '2024-01-01'",
        "FOR SYSTEM_TIME AS OF",
        id="system-time",
    ),
    pytest.param(
        "SELECT AI.GENERATE_BOOL(country) AS g FROM ds.orders",
        "generative AI function",
        id="generative-ai",
    ),
    pytest.param(
        "SELECT country, COUNT(*) AS c FROM ds.orders GROUP BY country HAVING COUNT(*) > 1",
        "HAVING",
        id="having",
    ),
    pytest.param("SELECT country FROM ds.orders ORDER BY country", "ORDER BY", id="order-by"),
    pytest.param("SELECT country FROM ds.orders LIMIT 1", "LIMIT", id="limit"),
    pytest.param(
        "SELECT country FROM ds.orders QUALIFY ROW_NUMBER() OVER (ORDER BY amount) = 1",
        "QUALIFY",
        id="qualify",
    ),
    pytest.param(
        "SELECT country, COUNT(*) / 10 AS cnt FROM ds.orders GROUP BY country",
        "not an output column",
        id="computed-aggregate",
    ),
    pytest.param(
        "SELECT country, STRING_AGG(country) AS s FROM ds.orders GROUP BY country",
        "aggregate function STRING_AGG",
        id="unsupported-aggregate",
    ),
    pytest.param(
        "WITH tmp AS (SELECT country, COUNT(*) AS n FROM ds.orders GROUP BY country) "
        "SELECT n, COUNT(*) AS cnt FROM tmp GROUP BY n",
        "more than one GROUP BY level",
        id="two-group-bys",
    ),
]


@pytest.mark.parametrize(("body", "expected"), INCREMENTAL_REJECTED)
async def test_incremental_view_rejects_unsupported_query(
    client: httpx.AsyncClient, body: str, expected: str
) -> None:
    await client.post(
        "/bigquery/v2/projects/p/datasets/ds/tables",
        json={
            "tableReference": {"projectId": "p", "datasetId": "ds", "tableId": "rates"},
            "schema": {"fields": [{"name": "country", "type": "STRING"}]},
        },
    )
    message = await _error(client, f"CREATE MATERIALIZED VIEW ds.bad AS {body}")
    assert message.startswith("Invalid materialized view query:")
    assert expected in message
    assert "bad" not in await _table_ids(client)


NON_INCREMENTAL_OPTIONS = (
    'OPTIONS(allow_non_incremental_definition=true, max_staleness=INTERVAL "4" HOUR)'
)


async def test_non_incremental_definition_lifts_the_incremental_rules(
    client: httpx.AsyncClient,
) -> None:
    """``allow_non_incremental_definition`` admits OUTER JOIN, UNION, HAVING and analytics."""
    await _run(client, "INSERT INTO ds.orders VALUES ('US', 10), ('US', 5), ('CA', 20)")
    await _run(
        client,
        f"CREATE MATERIALIZED VIEW ds.ranked {NON_INCREMENTAL_OPTIONS} AS "
        "SELECT country, SUM(amount) AS total, "
        "ROW_NUMBER() OVER (ORDER BY SUM(amount) DESC) AS rank "
        "FROM ds.orders GROUP BY country HAVING SUM(amount) > 1",
    )
    rows = _rows(await _run(client, "SELECT country, rank FROM ds.ranked ORDER BY rank"))
    assert rows == [["CA", "1"], ["US", "2"]]
    await _run(
        client,
        f"CREATE MATERIALIZED VIEW ds.unioned {NON_INCREMENTAL_OPTIONS} AS "
        "SELECT country FROM ds.orders UNION ALL SELECT country FROM ds.orders",
    )
    assert "unioned" in await _table_ids(client)


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        pytest.param("SELECT country, RAND() AS r FROM ds.orders", "RAND()", id="rand"),
        pytest.param(
            "SELECT ds.shout(country) AS c FROM ds.orders", "user-defined function", id="udf"
        ),
        pytest.param(
            "SELECT country FROM ds.orders TABLESAMPLE SYSTEM (10 PERCENT)",
            "TABLESAMPLE",
            id="tablesample",
        ),
        pytest.param(
            "SELECT country FROM ds.orders FOR SYSTEM_TIME AS OF TIMESTAMP '2024-01-01'",
            "FOR SYSTEM_TIME AS OF",
            id="system-time",
        ),
    ],
)
async def test_non_incremental_definition_keeps_the_universal_rules(
    client: httpx.AsyncClient, body: str, expected: str
) -> None:
    message = await _error(
        client, f"CREATE MATERIALIZED VIEW ds.bad {NON_INCREMENTAL_OPTIONS} AS {body}"
    )
    assert expected in message
    assert "bad" not in await _table_ids(client)


async def test_non_incremental_definition_requires_max_staleness(
    client: httpx.AsyncClient,
) -> None:
    message = await _error(
        client,
        "CREATE MATERIALIZED VIEW ds.bad OPTIONS(allow_non_incremental_definition=true) AS "
        "SELECT country FROM ds.orders",
    )
    assert "max_staleness" in message


async def test_base_table_must_be_a_plain_table(client: httpx.AsyncClient) -> None:
    await _run(client, "CREATE VIEW ds.orders_v AS SELECT country, amount FROM ds.orders")
    await _run(
        client,
        "CREATE MATERIALIZED VIEW ds.totals AS SELECT SUM(amount) AS total FROM ds.orders",
    )
    assert "logical views" in await _error(
        client, "CREATE MATERIALIZED VIEW ds.bad AS SELECT country FROM ds.orders_v"
    )
    assert "nested" in await _error(
        client, "CREATE MATERIALIZED VIEW ds.bad AS SELECT total FROM ds.totals"
    )
    assert "bad" not in await _table_ids(client)


async def test_cluster_and_partition_columns_must_be_outputs(client: httpx.AsyncClient) -> None:
    assert "CLUSTER BY column region" in await _error(
        client,
        "CREATE MATERIALIZED VIEW ds.bad CLUSTER BY region AS SELECT country FROM ds.orders",
    )
    assert "PARTITION BY column placed_at" in await _error(
        client,
        "CREATE MATERIALIZED VIEW ds.bad PARTITION BY DATE(placed_at) AS "
        "SELECT country FROM ds.orders",
    )
    assert "bad" not in await _table_ids(client)
