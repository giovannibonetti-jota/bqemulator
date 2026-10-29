r"""Pre-translator: collapse ``ALTER TABLE ... SET OPTIONS(...)`` to a no-op.

BigQuery accepts ``ALTER TABLE \`proj\`.\`ds\`.\`t\` SET OPTIONS(...)``
as the canonical way to set / clear table-level metadata
(``description``, ``labels``, ``expiration_timestamp``, etc.).
``dbt-bigquery`` emits the bare ``SET OPTIONS()`` form (with empty
parens) at the tail of every ``dbt seed`` / ``dbt run`` to clear
prior options on a managed table.

DuckDB doesn't support ``ALTER TABLE ... SET OPTIONS``. SQLGlot's
BigQuery → DuckDB transpile gives up partway through and emits the
*unparseable* truncated form ``ALTER TABLE "..." SET`` (no clause
after ``SET``), which then trips DuckDB's parser with::

    Parser Error: syntax error at end of input

bqemulator already doesn't model table-level option metadata
(``options`` is a v1.0.1 polish item), so the right behaviour is to
no-op the statement — exactly what real BigQuery does for an empty
``SET OPTIONS()`` call on a table with no options to clear.

This module rewrites any ``ALTER TABLE … SET OPTIONS(…)`` (with or
without contents in the parens) into ``SELECT 1`` so DuckDB executes
a trivially-successful query and the job-state machine reports
success. The statement is recognised on SQLGlot's typed parse
(``exp.Alter`` carrying a single ``exp.AlterSet`` action), which sees
through leading comments and any identifier quoting for free; the
other ``SET`` form BigQuery has, ``SET DEFAULT COLLATE``, parses to a
``CollateProperty`` and is left alone. The metadata-aware future
implementation will replace this rule with a catalog-side update.
"""

from __future__ import annotations

from sqlglot import exp

from bqemulator.sql.statement import parse_single_statement


def rewrite_alter_table_set_options(bq_sql: str) -> str:
    """Return a no-op statement when ``bq_sql`` is ``ALTER TABLE ... SET OPTIONS(...)``.

    Returns the original SQL unchanged for every other shape, including
    anything SQLGlot cannot parse (the translator reports those).
    """
    tree = parse_single_statement(bq_sql)
    if not isinstance(tree, exp.Alter) or (tree.kind or "").upper() != "TABLE":
        return bq_sql
    actions = tree.args.get("actions") or []
    if len(actions) != 1 or not isinstance(actions[0], exp.AlterSet):
        return bq_sql
    # ``SET OPTIONS()`` parses to a single empty slot; ``SET OPTIONS(k=v, …)``
    # to one ``Properties`` of plain ``Property`` entries. Anything else
    # (``SET DEFAULT COLLATE``, a bare ``SET OPTIONS`` without parens) is
    # not this statement.
    clause = actions[0].expressions
    if len(clause) != 1:
        return bq_sql
    options = clause[0]
    if options is None:
        # SQLGlot parses a bare ``SET OPTIONS`` (no parentheses, a BigQuery
        # syntax error) to the same empty slot; keep letting DuckDB reject it.
        return "SELECT 1" if "(" in bq_sql else bq_sql
    if isinstance(options, exp.Properties) and all(
        type(prop) is exp.Property for prop in options.expressions
    ):
        return "SELECT 1"
    return bq_sql


__all__ = ["rewrite_alter_table_set_options"]
