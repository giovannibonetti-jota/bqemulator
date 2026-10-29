"""Unit tests for ``FROM UNNEST(<array of STRUCT>)`` field exposure.

Every case runs the translated SQL on a real DuckDB connection: the
pass exists because DuckDB's table-function ``UNNEST`` hides the
struct's fields behind the element column, so only execution proves
that bare field references bind.
"""

from __future__ import annotations

import duckdb
import pytest

from bqemulator.domain.result import Ok
from bqemulator.sql.rewriter.unnest_offset import rewrite_unnest_offset
from bqemulator.sql.translator import SQLTranslator

pytestmark = pytest.mark.unit

_DERIVED = (
    "UNNEST(ARRAY(SELECT AS STRUCT p AS label, LENGTH(p) AS size FROM UNNEST(['aa', 'b']) AS p))"
)


@pytest.fixture
def t() -> SQLTranslator:
    return SQLTranslator()


@pytest.fixture
def con() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("CREATE TABLE t AS SELECT 1 AS id, 'a=1&b=2' AS qs UNION ALL SELECT 2, NULL")
    return con


def _run(
    t: SQLTranslator, con: duckdb.DuckDBPyConnection, sql: str
) -> tuple[list[str], list[tuple[object, ...]]]:
    result = t.translate(sql)
    assert isinstance(result, Ok), result
    cur = con.execute(result.value)
    return [d[0] for d in cur.description or []], cur.fetchall()


class TestDerivedArray:
    def test_bare_field_references(self, t: SQLTranslator, con: duckdb.DuckDBPyConnection) -> None:
        cols, rows = _run(t, con, f"SELECT label, size FROM {_DERIVED} AS s ORDER BY label")
        assert cols == ["label", "size"]
        assert rows == [("aa", 2), ("b", 1)]

    def test_qualified_field_references(
        self, t: SQLTranslator, con: duckdb.DuckDBPyConnection
    ) -> None:
        cols, rows = _run(t, con, f"SELECT s.label FROM {_DERIVED} AS s ORDER BY 1")
        assert cols == ["label"]
        assert rows == [("aa",), ("b",)]

    def test_select_star_yields_the_fields(
        self, t: SQLTranslator, con: duckdb.DuckDBPyConnection
    ) -> None:
        cols, _ = _run(t, con, f"SELECT * FROM {_DERIVED} AS s")
        assert cols == ["label", "size"]

    def test_with_offset_is_zero_based(
        self, t: SQLTranslator, con: duckdb.DuckDBPyConnection
    ) -> None:
        # The query pipeline rebases ``o`` before translating; mirror it.
        sql = rewrite_unnest_offset(
            f"SELECT label, o AS idx FROM {_DERIVED} AS s WITH OFFSET AS o ORDER BY idx"
        )
        cols, rows = _run(t, con, sql)
        assert cols == ["label", "idx"]
        assert rows == [("aa", 0), ("b", 1)]

    def test_correlated_array_in_a_join(
        self, t: SQLTranslator, con: duckdb.DuckDBPyConnection
    ) -> None:
        cols, rows = _run(
            t,
            con,
            "SELECT t.id, name, value FROM t, UNNEST(ARRAY("
            "  SELECT AS STRUCT SPLIT(p, '=')[OFFSET(0)] AS name, SPLIT(p, '=')[OFFSET(1)] AS value"
            "  FROM UNNEST(SPLIT(t.qs, '&')) AS p)) AS kv "
            "WHERE LENGTH(name) > 0 ORDER BY 1, 2",
        )
        assert cols == ["id", "name", "value"]
        assert rows == [(1, "a", "1"), (1, "b", "2")]

    def test_left_join_keeps_rows_without_elements(
        self, t: SQLTranslator, con: duckdb.DuckDBPyConnection
    ) -> None:
        cols, rows = _run(
            t,
            con,
            "SELECT t.id, name FROM t LEFT JOIN UNNEST(ARRAY("
            "  SELECT AS STRUCT p AS name FROM UNNEST(SPLIT(t.qs, '&')) AS p)) AS kv ON TRUE "
            "ORDER BY 1, 2",
        )
        assert cols == ["id", "name"]
        assert rows == [(1, "a=1"), (1, "b=2"), (2, None)]


class TestUntouched:
    @pytest.mark.parametrize(
        ("sql", "expected"),
        [
            ("SELECT x FROM UNNEST([3, 1, 2]) AS x ORDER BY x", [(1,), (2,), (3,)]),
            ("SELECT p FROM UNNEST(SPLIT('a&b', '&')) AS p ORDER BY p", [("a",), ("b",)]),
            (
                "SELECT label, value FROM UNNEST([STRUCT('a' AS label, 1 AS value)]) AS s",
                [("a", 1)],
            ),
        ],
    )
    def test_scalar_and_literal_arrays_keep_working(
        self,
        t: SQLTranslator,
        con: duckdb.DuckDBPyConnection,
        sql: str,
        expected: list[tuple[object, ...]],
    ) -> None:
        _, rows = _run(t, con, sql)
        assert rows == expected

    def test_scalar_array_sql_is_not_reserialised(self, t: SQLTranslator) -> None:
        result = t.translate("SELECT x FROM UNNEST([1, 2]) AS x")
        assert isinstance(result, Ok)
        assert "max_depth" not in result.value
