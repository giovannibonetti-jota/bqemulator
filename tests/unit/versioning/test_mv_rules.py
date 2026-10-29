"""Unit tests for BigQuery's materialized-view query rules.

One case per bullet of
https://docs.cloud.google.com/bigquery/docs/materialized-views-create#query_limitations
(incremental views) and of the universal rules that
``allow_non_incremental_definition`` does not lift.
"""

from __future__ import annotations

import pytest
import sqlglot

from bqemulator.domain.errors import InvalidQueryError
from bqemulator.versioning.mv_rules import (
    query_problems,
    validate_materialized_view_query,
)

pytestmark = pytest.mark.unit


def _problems(sql: str, *, non_incremental: bool = False) -> list[str]:
    return query_problems(sqlglot.parse_one(sql, read="bigquery"), non_incremental=non_incremental)


# "Unsupported SQL features" that every materialized view rejects.
UNIVERSAL = [
    pytest.param("SELECT a, RAND() AS r FROM ds.t", "non-deterministic function RAND()", id="rand"),
    pytest.param(
        "SELECT a, CURRENT_DATE() AS d FROM ds.t",
        "non-deterministic function CURRENT_DATE()",
        id="current_date",
    ),
    pytest.param(
        "SELECT a, CURRENT_TIME() AS d FROM ds.t",
        "non-deterministic function CURRENT_TIME()",
        id="current_time",
    ),
    pytest.param(
        "SELECT a, CURRENT_TIMESTAMP() AS d FROM ds.t",
        "non-deterministic function CURRENT_TIMESTAMP()",
        id="current_timestamp",
    ),
    pytest.param(
        "SELECT a, SESSION_USER() AS u FROM ds.t",
        "non-deterministic function SESSION_USER()",
        id="session_user",
    ),
    pytest.param(
        "SELECT a, GENERATE_UUID() AS u FROM ds.t",
        "non-deterministic function GENERATE_UUID()",
        id="generate_uuid",
    ),
    pytest.param("SELECT ds.fn(a) AS f FROM ds.t", "user-defined function ds.fn(a)", id="udf"),
    pytest.param(
        "SELECT `p`.`ds`.fn(a) AS f FROM ds.t",
        "user-defined function `p`.`ds`.fn(a)",
        id="udf-fully-qualified",
    ),
    pytest.param(
        "SELECT AI.GENERATE_BOOL(a) AS g FROM ds.t",
        "generative AI function AI.GENERATE_BOOL",
        id="ai-generate",
    ),
    pytest.param(
        "SELECT ML.GENERATE_TEXT(MODEL ds.m, TABLE ds.t) AS g FROM ds.t",
        "generative AI function ML.GENERATE_TEXT",
        id="ml-generate-text",
    ),
    pytest.param(
        "SELECT a FROM ds.t TABLESAMPLE SYSTEM (10 PERCENT)", "TABLESAMPLE", id="tablesample"
    ),
    pytest.param(
        "SELECT a FROM ds.t FOR SYSTEM_TIME AS OF TIMESTAMP '2024-01-01'",
        "FOR SYSTEM_TIME AS OF",
        id="system-time",
    ),
    pytest.param(
        "SELECT a, @@session_id AS s FROM ds.t", "system variable @@session_id", id="sysvar"
    ),
]

# Rejected by incremental views, allowed with allow_non_incremental_definition.
INCREMENTAL_ONLY = [
    pytest.param("SELECT a FROM ds.t UNION ALL SELECT a FROM ds.u", "UNION ALL", id="union-all"),
    pytest.param(
        "SELECT a FROM ds.t UNION DISTINCT SELECT a FROM ds.u",
        "UNION DISTINCT",
        id="union-distinct",
    ),
    pytest.param(
        "SELECT a FROM ds.t INTERSECT DISTINCT SELECT a FROM ds.u",
        "INTERSECT DISTINCT",
        id="intersect",
    ),
    pytest.param(
        "SELECT a FROM ds.t EXCEPT DISTINCT SELECT a FROM ds.u", "EXCEPT DISTINCT", id="except"
    ),
    pytest.param(
        "SELECT t.a, SUM(u.b) AS s FROM ds.t LEFT OUTER JOIN ds.u ON t.a = u.a GROUP BY t.a",
        "LEFT OUTER JOIN",
        id="left-outer-join",
    ),
    pytest.param(
        "SELECT t.a FROM ds.t LEFT JOIN ds.u ON t.a = u.a", "LEFT OUTER JOIN", id="left-join"
    ),
    pytest.param(
        "SELECT t.a FROM ds.t RIGHT JOIN ds.u ON t.a = u.a", "RIGHT OUTER JOIN", id="right-join"
    ),
    pytest.param(
        "SELECT t.a FROM ds.t FULL OUTER JOIN ds.u ON t.a = u.a", "FULL OUTER JOIN", id="full-join"
    ),
    pytest.param(
        "SELECT t1.a FROM ds.t AS t1 JOIN ds.t AS t2 ON t1.a = t2.a",
        "self-join on ds.t (referenced 2 times)",
        id="self-join",
    ),
    pytest.param(
        "SELECT a, ROW_NUMBER() OVER (ORDER BY a) AS rn FROM ds.t", "window function", id="window"
    ),
    pytest.param(
        "SELECT a, ARRAY(SELECT x FROM UNNEST(arr) AS x) AS xs FROM ds.t",
        "ARRAY subquery",
        id="array-subquery",
    ),
    pytest.param(
        "SELECT a, COUNT(*) AS c FROM ds.t GROUP BY a HAVING c > 1", "HAVING", id="having"
    ),
    pytest.param(
        "SELECT a FROM ds.t QUALIFY ROW_NUMBER() OVER (ORDER BY a) = 1", "QUALIFY", id="qualify"
    ),
    pytest.param("SELECT a FROM ds.t ORDER BY a", "ORDER BY", id="order-by"),
    pytest.param("SELECT a FROM ds.t LIMIT 10", "LIMIT", id="limit"),
    pytest.param(
        "SELECT TIMESTAMP_TRUNC(ts, HOUR) AS h, COUNT(*) / 10 AS cnt FROM ds.t GROUP BY h",
        "aggregate COUNT is not an output column",
        id="computed-aggregate",
    ),
    pytest.param(
        "SELECT a, STRING_AGG(b) AS s FROM ds.t GROUP BY a",
        "aggregate function STRING_AGG",
        id="unsupported-aggregate",
    ),
    pytest.param(
        "SELECT a, APPROX_QUANTILES(b, 2) AS q FROM ds.t GROUP BY a",
        "aggregate function APPROX_QUANTILES",
        id="unsupported-aggregate-approx-quantiles",
    ),
    pytest.param(
        "WITH tmp AS (SELECT city, COUNT(*) AS population FROM ds.t GROUP BY city) "
        "SELECT population, COUNT(*) AS cnt FROM tmp GROUP BY population",
        "more than one GROUP BY level",
        id="two-group-bys",
    ),
    pytest.param(
        "SELECT a FROM ds.t WHERE a > (SELECT MAX(a) FROM ds.u)",
        "aggregate in a subquery expression",
        id="filter-on-aggregate",
    ),
]

ACCEPTED = [
    pytest.param("SELECT country, SUM(amount) AS total FROM ds.orders GROUP BY country", id="sum"),
    pytest.param(
        "SELECT *, TIMESTAMP_MICROS(id >> 11) AS created_at FROM `p`.`ds`.`orders`", id="dbt-star"
    ),
    pytest.param(
        "SELECT ts_hour, ANY_VALUE(a) AS av, APPROX_COUNT_DISTINCT(a) AS ad, ARRAY_AGG(a) AS aa, "
        "AVG(x) AS avg, BIT_AND(x) AS b1, BIT_OR(x) AS b2, BIT_XOR(x) AS b3, COUNT(*) AS c, "
        "COUNTIF(x > 1) AS ci, HLL_COUNT.INIT(a) AS h, LOGICAL_AND(y) AS la, LOGICAL_OR(y) AS lo, "
        "MAX(x) AS mx, MIN(x) AS mn, MAX_BY(a, x) AS mb, MIN_BY(a, x) AS mnb, SUM(x) AS s "
        "FROM ds.t GROUP BY ts_hour",
        id="every-allowed-aggregate",
    ),
    pytest.param(
        "WITH tmp AS (SELECT TIMESTAMP_TRUNC(ts, HOUR) AS ts_hour, * FROM ds.t) "
        "SELECT ts_hour, COUNT(*) AS cnt FROM tmp GROUP BY ts_hour",
        id="cte",
    ),
    pytest.param(
        "SELECT o.country, SUM(o.amount * r.rate) AS usd FROM ds.orders AS o "
        "JOIN ds.rates AS r USING (country) GROUP BY o.country",
        id="inner-join",
    ),
    pytest.param("SELECT t.a, u.b FROM ds.t CROSS JOIN ds.u", id="cross-join"),
    pytest.param(
        "SELECT country, SUM(amount) AS t FROM "
        "(SELECT country, amount FROM ds.orders WHERE amount > 0) GROUP BY country",
        id="derived-table",
    ),
    pytest.param(
        "SELECT SAFE.SUBSTR(a, 1) AS s, NET.HOST(a) AS h FROM ds.t", id="builtin-namespaces"
    ),
    pytest.param("SELECT a, ARRAY_AGG(b ORDER BY c) AS bs FROM ds.t GROUP BY a", id="agg-order-by"),
]


@pytest.mark.parametrize(("sql", "expected"), UNIVERSAL)
def test_universal_rule_rejected_in_both_modes(sql: str, expected: str) -> None:
    assert expected in _problems(sql)
    assert expected in _problems(sql, non_incremental=True)


@pytest.mark.parametrize(("sql", "expected"), INCREMENTAL_ONLY)
def test_incremental_rule_rejected_only_without_allow_non_incremental(
    sql: str, expected: str
) -> None:
    assert any(expected in p for p in _problems(sql)), _problems(sql)
    assert _problems(sql, non_incremental=True) == []


@pytest.mark.parametrize("sql", ACCEPTED)
def test_valid_incremental_query_has_no_problems(sql: str) -> None:
    assert _problems(sql) == []


def test_validate_raises_invalid_query_with_every_problem() -> None:
    tree = sqlglot.parse_one("SELECT a, RAND() AS r FROM ds.t ORDER BY a LIMIT 1", read="bigquery")
    with pytest.raises(InvalidQueryError, match=r"RAND\(\); ORDER BY; LIMIT"):
        validate_materialized_view_query(tree)
    validate_materialized_view_query(
        sqlglot.parse_one("SELECT a FROM ds.t", read="bigquery"), non_incremental=True
    )
