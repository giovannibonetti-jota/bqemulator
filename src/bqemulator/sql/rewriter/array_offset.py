"""Pre-translator for ``arr[OFFSET(i)]`` / ``arr[SAFE_OFFSET(i)]`` with a computed index.

BigQuery's ``OFFSET`` is 0-based and DuckDB's bracket is 1-based.
SQLGlot shifts the index when it is an integer literal
(``arr[OFFSET(1)]`` → ``arr[2]``) but emits any other index verbatim,
so ``arr[OFFSET(idx)]`` reads the element *before* the one BigQuery
reads and ``arr[OFFSET(0 + idx)]`` with ``idx = 0`` reads nothing.

This pass rewrites every such bracket on the BigQuery AST to the
1-based ``ORDINAL`` form with the index incremented —
``arr[ORDINAL(idx + 1)]`` — which SQLGlot then emits as ``arr[idx + 1]``.
``SAFE_OFFSET`` keeps its ``safe`` flag. Literal indexes are left for
SQLGlot to shift.
"""

from __future__ import annotations

import sqlglot
from sqlglot import exp


def rewrite_array_offset(bq_sql: str) -> str:
    """``arr[OFFSET(expr)]`` → ``arr[ORDINAL(expr + 1)]`` for a non-literal ``expr``.

    Returns the input unchanged when no bracket needs the shift or the
    SQL does not parse (the transpile step reports parse errors).
    """
    if "OFFSET(" not in bq_sql.upper():
        return bq_sql
    try:
        tree = sqlglot.parse_one(bq_sql, read="bigquery")
    except sqlglot.errors.ParseError:
        return bq_sql
    modified = False
    for bracket in list(tree.find_all(exp.Bracket)):
        if bracket.args.get("offset") != 0 or len(bracket.expressions) != 1:
            continue
        index = bracket.expressions[0]
        if isinstance(index, exp.Literal):
            continue
        bracket.set("expressions", [exp.Add(this=index.copy(), expression=exp.Literal.number(1))])
        bracket.set("offset", 1)
        modified = True
    if not modified:
        return bq_sql
    return tree.sql(dialect="bigquery")


__all__ = ["rewrite_array_offset"]
