"""Unit tests for the computed-index ``OFFSET`` shift."""

from __future__ import annotations

import duckdb
import pytest

from bqemulator.domain.result import Ok
from bqemulator.sql.rewriter.array_offset import rewrite_array_offset
from bqemulator.sql.rewriter.unnest_offset import rewrite_unnest_offset
from bqemulator.sql.translator import SQLTranslator

pytestmark = pytest.mark.unit


@pytest.fixture
def t() -> SQLTranslator:
    return SQLTranslator()


def _rows(t: SQLTranslator, sql: str) -> list[tuple[object, ...]]:
    result = t.translate(sql)
    assert isinstance(result, Ok), result
    return duckdb.connect().execute(result.value).fetchall()


def test_computed_offset_reads_the_zero_based_element(t: SQLTranslator) -> None:
    rows = _rows(t, "SELECT i, ['x', 'y'][OFFSET(i)] AS v FROM UNNEST([0, 1]) AS i ORDER BY i")
    assert rows == [(0, "x"), (1, "y")]


def test_computed_safe_offset_is_null_out_of_range(t: SQLTranslator) -> None:
    rows = _rows(
        t, "SELECT i, ['x', 'y'][SAFE_OFFSET(i)] AS v FROM UNNEST([0, 1, 2]) AS i ORDER BY i"
    )
    assert rows == [(0, "x"), (1, "y"), (2, None)]


def test_offset_from_with_offset_column(t: SQLTranslator) -> None:
    """A ``WITH OFFSET`` index, rebased by the query pipeline, addresses the same element."""
    sql = "SELECT p, ['x', 'y'][OFFSET(o)] AS v FROM UNNEST(['a', 'b']) AS p WITH OFFSET AS o"
    rows = _rows(t, rewrite_unnest_offset(sql))
    assert sorted(rows) == [("a", "x"), ("b", "y")]


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT ['x', 'y'][OFFSET(1)] AS v",
        "SELECT ['x', 'y'][ORDINAL(i)] AS v FROM UNNEST([1, 2]) AS i",
        "SELECT ['x', 'y'][i] AS v FROM UNNEST([1, 2]) AS i",
        "SELECT 1 AS v",
    ],
)
def test_literal_ordinal_and_bare_indexes_are_untouched(sql: str) -> None:
    assert rewrite_array_offset(sql) == sql


def test_rewritten_form(t: SQLTranslator) -> None:
    assert rewrite_array_offset("SELECT arr[OFFSET(i)] FROM t") == (
        "SELECT arr[ORDINAL(i + 1)] FROM t"
    )
