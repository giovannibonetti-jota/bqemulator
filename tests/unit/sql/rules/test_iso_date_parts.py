"""Tests for the BigQuery ISO date-part translation rules.

The two rules in :mod:`bqemulator.sql.rules.iso_date_parts` bridge a
DuckDB parser gap (no ``ISOWEEK`` specifier inside ``EXTRACT``) and a
type-fidelity gap (``DATE_TRUNC(date, ISOYEAR)`` returns ``TIMESTAMP``
instead of ``DATE``). Each rule is exercised against a real DuckDB
connection so the BigQuery wire-format expectations are honoured
end-to-end.
"""

from __future__ import annotations

import datetime as _dt

import duckdb
import pytest

from bqemulator.domain.result import Ok
from bqemulator.sql.translator import SQLTranslator

pytestmark = pytest.mark.unit


@pytest.fixture
def t() -> SQLTranslator:
    return SQLTranslator()


@pytest.fixture
def con() -> duckdb.DuckDBPyConnection:
    return duckdb.connect()


def _execute(t: SQLTranslator, con: duckdb.DuckDBPyConnection, sql: str) -> object:
    result = t.translate(sql)
    assert isinstance(result, Ok), result
    return con.execute(result.value).fetchone()


class TestExtractIsoweek:
    """``EXTRACT(ISOWEEK FROM x)`` → ``EXTRACT(WEEK FROM x)``."""

    def test_rewrites_specifier(self, t: SQLTranslator) -> None:
        result = t.translate("SELECT EXTRACT(ISOWEEK FROM DATE '2024-03-15') AS w")
        assert isinstance(result, Ok)
        assert "ISOWEEK" not in result.value.upper()
        assert "WEEK" in result.value.upper()

    def test_returns_iso_week(self, t: SQLTranslator, con: duckdb.DuckDBPyConnection) -> None:
        row = _execute(t, con, "SELECT EXTRACT(ISOWEEK FROM DATE '2024-03-15') AS w")
        assert row == (11,)

    def test_unrelated_extract_untouched(
        self, t: SQLTranslator, con: duckdb.DuckDBPyConnection
    ) -> None:
        # The rule must NOT alter ``EXTRACT(YEAR FROM ...)`` or other
        # non-ISOWEEK specifiers.
        row = _execute(t, con, "SELECT EXTRACT(YEAR FROM DATE '2024-03-15') AS y")
        assert row == (2024,)


class TestDateTruncIsoyear:
    """``DATE_TRUNC(date, ISOYEAR)`` → ``CAST(DATE_TRUNC('ISOYEAR', date) AS DATE)``."""

    def test_wraps_in_date_cast(self, t: SQLTranslator) -> None:
        result = t.translate("SELECT DATE_TRUNC(DATE '2024-01-02', ISOYEAR) AS d")
        assert isinstance(result, Ok)
        upper = result.value.upper()
        assert "CAST" in upper
        assert "AS DATE" in upper

    def test_returns_date_value(self, t: SQLTranslator, con: duckdb.DuckDBPyConnection) -> None:
        row = _execute(t, con, "SELECT DATE_TRUNC(DATE '2024-01-02', ISOYEAR) AS d")
        assert row == (_dt.date(2024, 1, 1),)

    def test_unrelated_unit_untouched(
        self, t: SQLTranslator, con: duckdb.DuckDBPyConnection
    ) -> None:
        # The ISOYEAR rule must only intercept the ``ISOYEAR`` unit; the
        # surrounding ``DateTruncCalendarUnitRule`` handles MONTH on a
        # DATE operand by casting the TIMESTAMP result back to DATE.
        # Either way, the truncated value must match BigQuery's expected
        # 2024-03-01 — so we accept both shapes via the date() coercion.
        row = _execute(t, con, "SELECT DATE_TRUNC(DATE '2024-03-15', MONTH) AS d")
        assert row is not None
        truncated = row[0]
        if isinstance(truncated, _dt.datetime):
            truncated = truncated.date()
        assert truncated == _dt.date(2024, 3, 1)

    def test_timestamp_operand_not_cast_to_date(
        self, t: SQLTranslator, con: duckdb.DuckDBPyConnection
    ) -> None:
        # TIMESTAMP_TRUNC(..., ISOYEAR) parses to the same node shape as
        # DATE_TRUNC(..., ISOYEAR) once re-parsed as DuckDB SQL, but the
        # rule must NOT cast a TIMESTAMP operand to DATE — DuckDB's
        # natural behaviour (return TIMESTAMP) is the correct one here.
        result = t.translate(
            "SELECT TIMESTAMP_TRUNC(TIMESTAMP '2024-01-02 12:00:00 UTC', ISOYEAR) AS d",
        )
        assert isinstance(result, Ok)
        desc = con.execute(result.value).description
        assert "TIMESTAMP" in str(desc[0][1]).upper()

    def test_datetime_operand_not_cast_to_date(
        self, t: SQLTranslator, con: duckdb.DuckDBPyConnection
    ) -> None:
        # Same as above for DATETIME_TRUNC(..., ISOYEAR).
        result = t.translate(
            "SELECT DATETIME_TRUNC(DATETIME '2024-01-02 12:00:00', ISOYEAR) AS d",
        )
        assert isinstance(result, Ok)
        desc = con.execute(result.value).description
        assert "TIMESTAMP" in str(desc[0][1]).upper() or "DATETIME" in str(desc[0][1]).upper()
        assert desc[0][1] != "DATE"


class TestDateTruncQuarter:
    """``DATE_TRUNC(date, QUARTER)`` → ``CAST(... AS DATE)``."""

    def test_returns_date(self, t: SQLTranslator, con: duckdb.DuckDBPyConnection) -> None:
        # Wed 2024-05-15 → start of Q2 = 2024-04-01.
        result = t.translate("SELECT DATE_TRUNC(DATE '2024-05-15', QUARTER) AS d")
        assert isinstance(result, Ok)
        desc = con.execute(result.value).description
        assert desc[0][1] == "DATE"
        row = con.execute(result.value).fetchone()
        assert row == (_dt.date(2024, 4, 1),)

    def test_passes_through_timestamp_operand(
        self, t: SQLTranslator, con: duckdb.DuckDBPyConnection
    ) -> None:
        # When the operand is TIMESTAMP, the rule must NOT cast to
        # DATE — DuckDB's natural behaviour is to return TIMESTAMP.
        result = t.translate(
            "SELECT DATE_TRUNC(TIMESTAMP '2024-05-15 12:00:00 UTC', QUARTER) AS d",
        )
        assert isinstance(result, Ok)
        desc = con.execute(result.value).description
        # No CAST AS DATE was injected — TIMESTAMP stays.
        assert "DATE" not in str(desc[0][1]).upper() or "TIMESTAMP" in str(desc[0][1]).upper()


class TestDateTruncWeek:
    """``DATE_TRUNC(date, WEEK)`` → Sunday-start cast to DATE."""

    def test_returns_sunday_date(self, t: SQLTranslator, con: duckdb.DuckDBPyConnection) -> None:
        # 2024-05-15 is a Wednesday; the Sunday on-or-before it is
        # 2024-05-12.
        result = t.translate("SELECT DATE_TRUNC(DATE '2024-05-15', WEEK) AS d")
        assert isinstance(result, Ok)
        desc = con.execute(result.value).description
        assert desc[0][1] == "DATE"
        row = con.execute(result.value).fetchone()
        assert row == (_dt.date(2024, 5, 12),)

    def test_sunday_input_returns_self(
        self, t: SQLTranslator, con: duckdb.DuckDBPyConnection
    ) -> None:
        # When the input *is* Sunday the rule returns the same date.
        result = t.translate("SELECT DATE_TRUNC(DATE '2024-05-12', WEEK) AS d")
        assert isinstance(result, Ok)
        row = con.execute(result.value).fetchone()
        assert row == (_dt.date(2024, 5, 12),)

    def test_saturday_input_rolls_back_to_sunday(
        self, t: SQLTranslator, con: duckdb.DuckDBPyConnection
    ) -> None:
        # 2024-05-18 (Saturday) → previous Sunday 2024-05-12.
        result = t.translate("SELECT DATE_TRUNC(DATE '2024-05-18', WEEK) AS d")
        assert isinstance(result, Ok)
        row = con.execute(result.value).fetchone()
        assert row == (_dt.date(2024, 5, 12),)


class TestDateTruncOverTableColumn:
    """``DATE_TRUNC(date_col, …)`` over a schema-typed table column.

    Exercises the rules in :mod:`bqemulator.sql.rules.iso_date_parts`
    with a real table column rather than a ``DATE '…'`` literal:
    ``_is_date_typed`` must recognise the column via the annotated
    ``.type`` SQLGlot's ``annotate_types`` pass attaches when
    ``translate()`` is called with a ``schema``, and the rules must
    match the ``exp.TimestampTrunc`` node shape SQLGlot's DuckDB
    dialect produces when re-parsing the transpiled SQL.
    """

    def test_week_over_column_matches_literal(
        self, t: SQLTranslator, con: duckdb.DuckDBPyConnection
    ) -> None:
        con.execute('CREATE SCHEMA "test-project__bqemu"')
        con.execute('CREATE TABLE "test-project__bqemu"."placements" (placed_on DATE)')
        con.execute(
            'INSERT INTO "test-project__bqemu"."placements" VALUES '
            "('2024-05-15'), ('2024-05-12'), ('2024-05-18')",  # Wed / Sun / Sat
        )
        schema = {"placements": {"placed_on": "DATE"}}
        result = t.translate(
            "SELECT DATE_TRUNC(placed_on, WEEK) AS d "
            "FROM `test-project.bqemu.placements` ORDER BY placed_on",
            schema=schema,
        )
        assert isinstance(result, Ok)
        duckdb_sql = result.value.replace(
            '"test-project"."bqemu"."placements"',
            '"test-project__bqemu"."placements"',
        )
        rows = con.execute(duckdb_sql).fetchall()
        # All three inputs fall in the same Sunday-start week: 2024-05-12.
        assert rows == [(_dt.date(2024, 5, 12),)] * 3
        desc = con.execute(duckdb_sql).description
        assert desc[0][1] == "DATE"

    def test_isoyear_over_column_casts_to_date(
        self, t: SQLTranslator, con: duckdb.DuckDBPyConnection
    ) -> None:
        con.execute('CREATE SCHEMA "test-project__bqemu"')
        con.execute('CREATE TABLE "test-project__bqemu"."placements" (placed_on DATE)')
        con.execute('INSERT INTO "test-project__bqemu"."placements" VALUES (\'2024-01-02\')')
        schema = {"placements": {"placed_on": "DATE"}}
        result = t.translate(
            "SELECT DATE_TRUNC(placed_on, ISOYEAR) AS d FROM `test-project.bqemu.placements`",
            schema=schema,
        )
        assert isinstance(result, Ok)
        duckdb_sql = result.value.replace(
            '"test-project"."bqemu"."placements"',
            '"test-project__bqemu"."placements"',
        )
        desc = con.execute(duckdb_sql).description
        assert desc[0][1] == "DATE"
        row = con.execute(duckdb_sql).fetchone()
        assert row == (_dt.date(2024, 1, 1),)

    def test_isoyear_over_timestamp_column_not_cast_to_date(
        self, t: SQLTranslator, con: duckdb.DuckDBPyConnection
    ) -> None:
        con.execute('CREATE SCHEMA "test-project__bqemu"')
        con.execute('CREATE TABLE "test-project__bqemu"."events" (happened_at TIMESTAMP)')
        con.execute(
            'INSERT INTO "test-project__bqemu"."events" VALUES (\'2024-01-02 12:00:00\')',
        )
        schema = {"events": {"happened_at": "TIMESTAMP"}}
        result = t.translate(
            "SELECT TIMESTAMP_TRUNC(happened_at, ISOYEAR) AS d FROM `test-project.bqemu.events`",
            schema=schema,
        )
        assert isinstance(result, Ok)
        duckdb_sql = result.value.replace(
            '"test-project"."bqemu"."events"',
            '"test-project__bqemu"."events"',
        )
        desc = con.execute(duckdb_sql).description
        assert desc[0][1] != "DATE"

    def test_week_over_column_survives_unrelated_date_add_in_same_query(
        self, t: SQLTranslator, con: duckdb.DuckDBPyConnection
    ) -> None:
        # ``rewrite_datetime_helpers`` (the ``DATE_ADD``/``DATE_SUB`` /
        # ``DATE_FROM_UNIX_DATE`` pre-translator) re-parses the entire
        # BigQuery SQL text and re-serialises it whenever the query
        # contains any of those calls — even ones unrelated to the
        # DATE_TRUNC this test cares about — which drops the
        # ``WEEK(SUNDAY)`` day qualifier down to bare ``WEEK``. The
        # DateTruncWeekRule safety net for bare ``WEEK`` must still
        # produce the correct Sunday-start result over a real DATE column.
        con.execute('CREATE SCHEMA "test-project__bqemu"')
        con.execute('CREATE TABLE "test-project__bqemu"."placements" (placed_on DATE)')
        con.execute('INSERT INTO "test-project__bqemu"."placements" VALUES (\'2024-05-15\')')
        schema = {"placements": {"placed_on": "DATE"}}
        result = t.translate(
            "SELECT DATE_TRUNC(placed_on, WEEK(SUNDAY)) AS week_start, "
            "DATE_ADD(placed_on, INTERVAL 1 DAY) AS unrelated "
            "FROM `test-project.bqemu.placements`",
            schema=schema,
        )
        assert isinstance(result, Ok)
        duckdb_sql = result.value.replace(
            '"test-project"."bqemu"."placements"',
            '"test-project__bqemu"."placements"',
        )
        row = con.execute(duckdb_sql).fetchone()
        assert row[0] == _dt.date(2024, 5, 12)  # Sunday-start week, not Monday-start
