"""Unit tests for :mod:`bqemulator.sql.statement`."""

from __future__ import annotations

import pytest
from sqlglot import exp

from bqemulator.sql.statement import parse_single_statement

pytestmark = pytest.mark.unit


def test_single_statement_returns_its_tree() -> None:
    tree = parse_single_statement("/* dbt */ SELECT 1 AS x;")
    assert isinstance(tree, exp.Select)


@pytest.mark.parametrize(
    "sql",
    [
        "",
        "   ",
        ";",
        "-- only a comment",
        "/* c */",
        "SELECT 1 AS x; SELECT 2",  # two statements: a router must not match either
        "CREATE MATERIALIZED VIEW ds.mv AS SELECT 1 AS x; SELECT 2",
        "SELEC 1",  # not parseable
    ],
)
def test_anything_but_one_statement_is_none(sql: str) -> None:
    assert parse_single_statement(sql) is None


def test_empty_trailing_statements_are_ignored() -> None:
    assert isinstance(parse_single_statement("SELECT 1;;"), exp.Select)
