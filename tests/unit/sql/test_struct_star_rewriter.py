"""Unit tests for the ``expr.*`` struct dot-star expansion.

Every case runs the translated SQL on a real DuckDB connection: the
point of the pass is that DuckDB's parser rejects ``.*`` on anything
but a bare struct column, so only execution proves the expansion.
"""

from __future__ import annotations

import duckdb
import pytest

from bqemulator.domain.errors import UnsupportedFeatureError
from bqemulator.domain.result import Err, Ok
from bqemulator.sql.translator import SQLTranslator

pytestmark = pytest.mark.unit

_SCHEMA = {
    "t": {
        "id": "BIGINT",
        "qs": "VARCHAR",
        "s": "STRUCT(x BIGINT, y VARCHAR)",
        "arr": "STRUCT(k BIGINT, v VARCHAR)[]",
    }
}


@pytest.fixture
def t() -> SQLTranslator:
    return SQLTranslator()


@pytest.fixture
def con() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE t AS "
        "SELECT 1 AS id, 'a=1&b=2' AS qs, {'x': 10, 'y': 'p'} AS s, [{'k': 1, 'v': 'one'}] AS arr "
        "UNION ALL SELECT 2, NULL, NULL, []"
    )
    return con


def _run(
    t: SQLTranslator,
    con: duckdb.DuckDBPyConnection,
    sql: str,
    *,
    schema: dict[str, dict[str, str]] | None = None,
) -> tuple[list[str], list[tuple[object, ...]]]:
    result = t.translate(sql, schema=schema)
    assert isinstance(result, Ok), result
    cur = con.execute(result.value)
    return [d[0] for d in cur.description or []], cur.fetchall()


class TestStaticFieldNames:
    """Operands SQLGlot cannot type: the names come from the struct literal."""

    def test_array_subquery_element_dbt_shape(
        self, t: SQLTranslator, con: duckdb.DuckDBPyConnection
    ) -> None:
        cols, rows = _run(
            t,
            con,
            "SELECT id, ARRAY(SELECT AS STRUCT STRING_AGG(p, '&') AS query_string, COUNT(*) AS n "
            "FROM UNNEST(SPLIT(t.qs, '&')) AS p WHERE LENGTH(p) > 0)[OFFSET(0)].*, qs AS tail "
            "FROM t ORDER BY id",
        )
        assert cols == ["id", "query_string", "n", "tail"]
        assert rows == [(1, "a=1&b=2", 2, "a=1&b=2"), (2, None, 0, None)]

    def test_struct_literal(self, t: SQLTranslator, con: duckdb.DuckDBPyConnection) -> None:
        cols, rows = _run(t, con, "SELECT STRUCT(1 AS a, 'x' AS b).* FROM t")
        assert cols == ["a", "b"]
        assert rows == [(1, "x"), (1, "x")]

    def test_scalar_subquery(self, t: SQLTranslator, con: duckdb.DuckDBPyConnection) -> None:
        cols, rows = _run(t, con, "SELECT (SELECT AS STRUCT id AS a, 2 AS b FROM t LIMIT 1).* ")
        assert cols == ["a", "b"]
        assert rows == [(1, 2)]

    def test_array_literal_element(self, t: SQLTranslator, con: duckdb.DuckDBPyConnection) -> None:
        cols, rows = _run(t, con, "SELECT [STRUCT(1 AS a, 2 AS b)][OFFSET(0)].* FROM t LIMIT 1")
        assert cols == ["a", "b"]
        assert rows == [(1, 2)]

    def test_safe_offset_out_of_range_yields_null_fields(
        self, t: SQLTranslator, con: duckdb.DuckDBPyConnection
    ) -> None:
        cols, rows = _run(t, con, "SELECT [STRUCT(1 AS a)][SAFE_OFFSET(5)].* FROM t LIMIT 1")
        assert cols == ["a"]
        assert rows == [(None,)]

    def test_expanded_columns_usable_downstream(
        self, t: SQLTranslator, con: duckdb.DuckDBPyConnection
    ) -> None:
        cols, rows = _run(
            t,
            con,
            "SELECT a, COUNT(*) AS c FROM (SELECT STRUCT(id AS a, qs AS b).* FROM t) "
            "GROUP BY a ORDER BY a DESC",
        )
        assert cols == ["a", "c"]
        assert rows == [(2, 1), (1, 1)]


class TestAnnotatedFieldNames:
    """Operands the catalog schema types through ``annotate_types``."""

    def test_array_column_element(self, t: SQLTranslator, con: duckdb.DuckDBPyConnection) -> None:
        cols, rows = _run(t, con, "SELECT id, arr[OFFSET(0)].* FROM t ORDER BY id", schema=_SCHEMA)
        assert cols == ["id", "k", "v"]
        assert rows == [(1, 1, "one"), (2, None, None)]

    @pytest.mark.parametrize("projection", ["t.s.*", "s.*"])
    def test_struct_column(
        self, t: SQLTranslator, con: duckdb.DuckDBPyConnection, projection: str
    ) -> None:
        cols, rows = _run(t, con, f"SELECT {projection} FROM t ORDER BY id", schema=_SCHEMA)
        assert cols == ["x", "y"]
        assert rows == [(10, "p"), (None, None)]


class TestModifiers:
    def test_except_drops_fields(self, t: SQLTranslator, con: duckdb.DuckDBPyConnection) -> None:
        cols, rows = _run(
            t, con, "SELECT STRUCT(1 AS a, 2 AS b, 3 AS c).* EXCEPT (b) FROM t LIMIT 1"
        )
        assert cols == ["a", "c"]
        assert rows == [(1, 3)]

    def test_replace_substitutes_projection(
        self, t: SQLTranslator, con: duckdb.DuckDBPyConnection
    ) -> None:
        cols, rows = _run(
            t,
            con,
            "SELECT STRUCT(1 AS a, 2 AS b).* REPLACE (id * 10 AS a) FROM t ORDER BY id",
        )
        assert cols == ["a", "b"]
        assert rows == [(10, 2), (20, 2)]


class TestUntouched:
    def test_table_wildcard(self, t: SQLTranslator, con: duckdb.DuckDBPyConnection) -> None:
        cols, _ = _run(t, con, "SELECT t.* FROM t")
        assert cols == ["id", "qs", "s", "arr"]

    def test_plain_query_sql_is_not_reserialised(self, t: SQLTranslator) -> None:
        result = t.translate("SELECT id FROM t")
        assert isinstance(result, Ok)
        assert result.value == "SELECT id FROM t"


class TestUnresolvable:
    @pytest.mark.parametrize(
        "sql",
        [
            "SELECT STRUCT(1, 2).* FROM t",
            "SELECT ARRAY(SELECT AS STRUCT * FROM t)[OFFSET(0)].* FROM t",
            "SELECT arr[OFFSET(0)].* FROM t",
        ],
    )
    def test_unnamed_fields_are_unsupported(self, t: SQLTranslator, sql: str) -> None:
        result = t.translate(sql)
        assert isinstance(result, Err)
        assert isinstance(result.error, UnsupportedFeatureError)
        assert "Dot-star expansion" in result.error.message
