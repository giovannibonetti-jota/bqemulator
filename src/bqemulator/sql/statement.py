"""Single-statement SQLGlot parse shared by the statement routers.

The executor, the script interpreter and the versioning router all
dispatch on the *type* of the parsed statement (``exp.Create``,
``exp.Drop``, ``exp.Export`` …) rather than on regexes over the SQL
text, so each statement is parsed once here and the tree handed down.
The tree is read-only for every consumer: the translator re-parses
before it rewrites.
"""

from __future__ import annotations

import sqlglot
from sqlglot import exp


def parse_single_statement(sql: str) -> exp.Expr | None:
    """Parse ``sql`` as exactly one BigQuery statement, or return ``None``.

    ``None`` covers three cases the callers treat alike — SQLGlot cannot
    parse the text (the translator reports the syntax error), the text
    is empty or only comments, or it holds several statements (scripts
    are split by the scripting parser before reaching a router, so a
    multi-statement string here must never match a single-statement
    head; the old regexes swallowed ``…; SELECT 2`` into a view body).
    """
    try:
        statements = sqlglot.parse(sql, read="bigquery")
    except Exception:  # noqa: BLE001 — not ours to diagnose
        return None
    trees = [tree for tree in statements if tree is not None]
    if len(trees) != 1:
        return None
    return trees[0]


__all__ = ["parse_single_statement"]
