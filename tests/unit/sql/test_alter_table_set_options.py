"""Pin the dbt-bigquery integration fixes."""

import pytest

from bqemulator.sql.rewriter.alter_table_set_options import (
    rewrite_alter_table_set_options,
)


@pytest.mark.parametrize(
    "sql",
    [
        "alter table `proj`.`ds`.`t` set OPTIONS()",
        'ALTER TABLE `p`.`d`.`t` SET OPTIONS(description="x", expiration_timestamp=NULL)',
        # dbt wraps every statement in a job-tag comment.
        '/* {"app": "dbt", "node_id": "seed.x"} */\n\n'
        "    alter table `bqemu-demo`.`dbt_local_raw`.`customers` set OPTIONS()\n  ",
        "-- dbt comment\nALTER TABLE p.d.t SET OPTIONS()",
        "ALTER TABLE IF EXISTS `p-1.d.t` SET OPTIONS(labels=[('a', 'b')]);",
    ],
)
def test_alter_table_set_options_becomes_a_no_op(sql: str) -> None:
    assert rewrite_alter_table_set_options(sql) == "SELECT 1"


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM t",
        "ALTER TABLE t ADD COLUMN x INT64",
        # ``ALTER VIEW`` is not collapsed (parity with the previous scanner).
        "ALTER VIEW v SET OPTIONS()",
        # The other SET form BigQuery has stays with the translator.
        "ALTER TABLE t SET DEFAULT COLLATE 'und:ci'",
        # Malformed statements pass through unchanged so the translator reports them.
        "ALTER TABLE t",
        "ALTER TABLE t SET OPTIONS",
        "ALTER TABLE t SET OPTIONS(description='no closing",
        "ALTER TABLE t SET OPTIONS() AND THEN SOME",
        "/* just a comment */",
        "",
        "/* never closed",
    ],
)
def test_other_shapes_pass_through(sql: str) -> None:
    assert rewrite_alter_table_set_options(sql) == sql
