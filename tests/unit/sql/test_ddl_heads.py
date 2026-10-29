"""Unit tests for :mod:`bqemulator.sql.ddl_heads`."""

from __future__ import annotations

import pytest

from bqemulator.domain.errors import InvalidQueryError
from bqemulator.sql.ddl_heads import (
    CallRefresh,
    CreateClone,
    CreateRowAccessPolicy,
    CreateSnapshot,
    DropAllRowAccessPolicies,
    DropRowAccessPolicy,
    DropSnapshot,
    DropTableFunction,
    RefreshMaterializedView,
    option_value,
    parse_ddl_head,
    parse_transaction_control,
)

pytestmark = pytest.mark.unit

DBT_COMMENT = '/* {"app": "dbt", "node_id": "model.x.y"} */\n'


class TestCreateSnapshot:
    def test_bare_form(self) -> None:
        assert parse_ddl_head("CREATE SNAPSHOT TABLE ds.snap CLONE ds.src") == CreateSnapshot(
            "ds.snap", "ds.src"
        )

    def test_every_clause(self) -> None:
        head = parse_ddl_head(
            DBT_COMMENT + "create or replace snapshot table if not exists `p-1.ds.snap`\n"
            "  copy `p-1`.`ds`.`src`\n"
            "  for system_time as of TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 HOUR)\n"
            "  options(expiration_timestamp=TIMESTAMP '2030-01-01', "
            "description='a (tricky) \\'one\\'', labels=[('k','v')]);"
        )
        assert head == CreateSnapshot(
            target="p-1.ds.snap",
            source="p-1.ds.src",
            replace=True,
            if_not_exists=True,
            system_time="TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 HOUR)",
            options={
                "expiration_timestamp": "TIMESTAMP '2030-01-01'",
                "description": "'a (tricky) \\'one\\''",
                "labels": "[('k','v')]",
            },
        )
        assert option_value(head.options["description"]) == "a (tricky) 'one'"

    def test_system_time_stops_at_end_without_options(self) -> None:
        head = parse_ddl_head(
            "CREATE SNAPSHOT TABLE ds.snap CLONE ds.src "
            "FOR SYSTEM_TIME AS OF '2030-01-01 00:00:00';"
        )
        assert isinstance(head, CreateSnapshot)
        assert head.system_time == "'2030-01-01 00:00:00'"

    def test_malformed_head_is_a_syntax_error(self) -> None:
        with pytest.raises(InvalidQueryError, match="Syntax error: CREATE SNAPSHOT TABLE") as info:
            parse_ddl_head("CREATE SNAPSHOT TABLE ds.snap FROM ds.src")
        assert "CLONE" in str(info.value)
        assert "at [1:31]" in str(info.value)  # the column of FROM


class TestDropSnapshot:
    @pytest.mark.parametrize(
        ("sql", "if_exists"),
        [
            ("DROP SNAPSHOT TABLE ds.snap", False),
            ("-- c\n  drop snapshot table if exists `p.ds.snap` ;", True),
        ],
    )
    def test_forms(self, sql: str, if_exists: bool) -> None:
        head = parse_ddl_head(sql)
        assert isinstance(head, DropSnapshot)
        assert (head.target.endswith("ds.snap"), head.if_exists) == (True, if_exists)


class TestCreateClone:
    def test_options_form(self) -> None:
        head = parse_ddl_head(
            "CREATE TABLE IF NOT EXISTS ds.copy CLONE ds.src "
            "OPTIONS(description='x', friendly_name=\"y\")"
        )
        assert head == CreateClone(
            "ds.copy",
            "ds.src",
            if_not_exists=True,
            options={"description": "'x'", "friendly_name": '"y"'},
        )

    def test_system_time_form(self) -> None:
        head = parse_ddl_head(
            "CREATE OR REPLACE TABLE ds.copy CLONE ds.src FOR SYSTEM_TIME AS OF "
            "TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 DAY) OPTIONS()"
        )
        assert head == CreateClone(
            "ds.copy",
            "ds.src",
            replace=True,
            system_time="TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)",
        )

    def test_plain_create_table_is_not_a_head(self) -> None:
        assert parse_ddl_head("CREATE TABLE ds.t (x INT64)") is None
        assert parse_ddl_head("CREATE TABLE ds.t AS SELECT 1 AS x") is None


class TestRefreshForms:
    @pytest.mark.parametrize(
        "sql",
        [
            "CALL BQ.REFRESH_MATERIALIZED_VIEW('p.ds.mv')",
            '/* c */ call bq.refresh_materialized_view ( "p.ds.mv" ) ;',
        ],
    )
    def test_call_forms(self, sql: str) -> None:
        assert parse_ddl_head(sql) == CallRefresh("p.ds.mv")

    def test_other_procedures_are_not_heads(self) -> None:
        assert parse_ddl_head("CALL ds.my_proc(1)") is None

    def test_refresh_extension(self) -> None:
        assert parse_ddl_head("REFRESH MATERIALIZED VIEW `p.ds.mv`;") == RefreshMaterializedView(
            "p.ds.mv"
        )


class TestRowAccessPolicy:
    def test_create_every_clause(self) -> None:
        head = parse_ddl_head(
            "CREATE OR REPLACE ROW ACCESS POLICY IF NOT EXISTS `eu` ON `my-proj`.`ds`.`tbl`\n"
            "GRANT TO ('user:a)b@c.d', \"group:g@c.d\")\n"
            "FILTER USING (region IN ('EU', ')') AND (tier > 1));"
        )
        assert head == CreateRowAccessPolicy(
            policy="eu",
            table="my-proj.ds.tbl",
            filter="region IN ('EU', ')') AND (tier > 1)",
            grantees=("user:a)b@c.d", "group:g@c.d"),
            replace=True,
            if_not_exists=True,
        )

    def test_create_without_grant(self) -> None:
        head = parse_ddl_head(
            DBT_COMMENT + "CREATE ROW ACCESS POLICY eu ON proj.ds.tbl FILTER USING (region = 'EU')"
        )
        assert head == CreateRowAccessPolicy("eu", "proj.ds.tbl", "region = 'EU'")

    def test_create_missing_filter_is_a_syntax_error(self) -> None:
        with pytest.raises(InvalidQueryError, match="FILTER USING"):
            parse_ddl_head("CREATE ROW ACCESS POLICY eu ON proj.ds.tbl GRANT TO ('user:a@b.c')")

    @pytest.mark.parametrize(
        ("sql", "if_exists"),
        [
            ("DROP ROW ACCESS POLICY p1 ON `ds.t`", False),
            ("DROP ROW ACCESS POLICY IF EXISTS `p1` ON ds.t;", True),
        ],
    )
    def test_drop(self, sql: str, if_exists: bool) -> None:
        assert parse_ddl_head(sql) == DropRowAccessPolicy("p1", "ds.t", if_exists)

    def test_drop_all(self) -> None:
        assert parse_ddl_head("DROP ALL ROW ACCESS POLICIES ON `p.ds.t`") == (
            DropAllRowAccessPolicies("p.ds.t")
        )


class TestDropTableFunction:
    @pytest.mark.parametrize(
        ("sql", "if_exists"),
        [
            ("DROP TABLE FUNCTION `p.ds.tvf`", False),
            ("drop table function if exists ds.tvf ;", True),
        ],
    )
    def test_forms(self, sql: str, if_exists: bool) -> None:
        head = parse_ddl_head(sql)
        assert isinstance(head, DropTableFunction)
        assert (head.name.endswith("ds.tvf"), head.if_exists) == (True, if_exists)

    def test_drop_table_is_not_a_head(self) -> None:
        assert parse_ddl_head("DROP TABLE FUNCTIONS_LOG") is None
        assert parse_ddl_head("DROP TABLE ds.t") is None


class TestRobustness:
    @pytest.mark.parametrize("sql", ["SELECT 1", "INSERT INTO ds.t VALUES (1)", "", "/* only */"])
    def test_non_heads_are_none(self, sql: str) -> None:
        assert parse_ddl_head(sql) is None

    def test_a_second_statement_is_a_syntax_error(self) -> None:
        """Callers hand over single statements; a trailing one is never swallowed."""
        with pytest.raises(InvalidQueryError, match="end of statement"):
            parse_ddl_head("CREATE TABLE ds.t CLONE ds.s OPTIONS(); SELECT 1")

    def test_deep_nesting_does_not_recurse(self) -> None:
        depth = 5000
        nested = "(" * depth + "1" + ")" * depth
        head = parse_ddl_head(f"CREATE SNAPSHOT TABLE ds.snap CLONE ds.src OPTIONS(a={nested})")
        assert isinstance(head, CreateSnapshot)
        assert head.options["a"] == nested

    def test_unbalanced_option_parens_is_a_syntax_error(self) -> None:
        with pytest.raises(InvalidQueryError, match="Syntax error"):
            parse_ddl_head("CREATE SNAPSHOT TABLE ds.snap CLONE ds.src OPTIONS(description='x'")


class TestOptionValue:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("'x'", "x"),
            ('"it\\"s"', 'it"s'),
            ("TRUE", True),
            ("false", False),
            ("NULL", None),
            ("30", 30),
            ("TIMESTAMP '2030-01-01'", "TIMESTAMP '2030-01-01'"),
            ("[('k','v')]", "[('k','v')]"),
        ],
    )
    def test_literals(self, raw: str, expected: object) -> None:
        assert option_value(raw) == expected


class TestTransactionControl:
    @pytest.mark.parametrize(
        "sql",
        [
            "BEGIN",
            "BEGIN;",
            "BEGIN TRANSACTION",
            "  begin   transaction  ;",
            "START TRANSACTION",
            "-- c\nStart Transaction;",
        ],
    )
    def test_begin_forms(self, sql: str) -> None:
        assert parse_transaction_control(sql) == "BEGIN"

    @pytest.mark.parametrize(
        "sql", ["COMMIT", "COMMIT TRANSACTION", "COMMIT WORK", "commit;", "END TRANSACTION"]
    )
    def test_commit_forms(self, sql: str) -> None:
        assert parse_transaction_control(sql) == "COMMIT"

    @pytest.mark.parametrize(
        "sql", ["ROLLBACK", "ROLLBACK TRANSACTION", "ROLLBACK WORK", "  rollback  ;"]
    )
    def test_rollback_forms(self, sql: str) -> None:
        assert parse_transaction_control(sql) == "ROLLBACK"

    @pytest.mark.parametrize(
        "sql",
        [
            "SELECT 1",
            "BEGIN SELECT 1; END",  # a block, not a transaction
            "END",  # the block terminator alone
            "ROLLBACK TO SAVEPOINT sp",
            "BEGINNING",
            "COMMITTED",
        ],
    )
    def test_other_sql_is_not_transaction_control(self, sql: str) -> None:
        assert parse_transaction_control(sql) is None
