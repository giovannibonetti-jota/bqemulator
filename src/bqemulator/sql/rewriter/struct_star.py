"""Post-translator for dot-star on a STRUCT expression (``expr.*``).

BigQuery expands ``<struct expression>.*`` in a SELECT list into one
column per field: ``arr[OFFSET(0)].*``, ``STRUCT(1 AS a).*``,
``(SELECT AS STRUCT …).*`` and ``t.s.*`` all work. DuckDB only expands
``.*`` on a bare struct *column*; every other operand is a parser error
(``syntax error at or near "*"``), and ``unnest(expr)`` cannot replace
it because DuckDB refuses to unnest an expression that carries a
correlated subquery.

This pass rewrites ``Dot(expr, Star)`` select items on the DuckDB AST
into ``expr.f1 AS f1, expr.f2 AS f2, …`` — plain field access, which
DuckDB accepts on any struct-typed expression. The field names come
from, in order:

1. the ``annotate_types`` result when the translator ran it with a
   catalog schema (table columns, ``ARRAY<STRUCT>`` elements, struct
   literals, uncorrelated scalar subqueries);
2. the operand's own shape when SQLGlot could not type it: the
   ``{'name': …}`` struct literal that ``SELECT AS STRUCT a AS name``
   transpiles to, reached through ``ARRAY(SELECT …)[i]``, ``(SELECT …)``
   and ``[STRUCT(…)][i]`` wrappers.

``.* EXCEPT (…)`` drops the named fields and ``.* REPLACE (expr AS f)``
substitutes the field's projection, as BigQuery does. An operand whose
fields cannot be named — a positional ``STRUCT(1, 2)`` (``_0``, ``_1``
after transpile) or an opaque expression — raises
:class:`~bqemulator.domain.errors.UnsupportedFeatureError`, which is
more useful than the DuckDB parser error it would otherwise become.
The operand is duplicated once per field; DuckDB evaluates each copy,
so a correlated subquery runs N times for N fields.
"""

from __future__ import annotations

from sqlglot import exp

from bqemulator.sql.errors import sql_unsupported
from bqemulator.sql.struct_fields import is_anonymous_field, struct_field_names


def expand_struct_star(tree: exp.Expression) -> bool:
    """``<struct expr>.*`` select item → ``expr.f1 AS f1, expr.f2 AS f2, …``.

    Returns ``True`` when the tree was modified. ``t.*`` on a table
    alias is a :class:`exp.Column` with a ``Star`` head, not a
    :class:`exp.Dot`, so table wildcards are never touched.
    """
    modified = False
    for dot in list(tree.find_all(exp.Dot)):
        star = dot.expression
        if not isinstance(star, exp.Star) or not isinstance(dot.parent, exp.Select):
            continue
        if dot.arg_key != "expressions":
            continue
        operand = dot.this
        names = struct_field_names(operand)
        if names is None or any(is_anonymous_field(n) for n in names):
            raise sql_unsupported(
                f"Dot-star expansion of {operand.sql(dialect='duckdb')!r} "
                "(the struct's field names cannot be resolved)"
            )
        dot.replace(_projections(operand, names, star))
        modified = True
    return modified


def _projections(operand: exp.Expression, names: list[str], star: exp.Star) -> list[exp.Expression]:
    """Build ``operand.f AS f`` for every kept field, honouring EXCEPT / REPLACE."""
    excluded = {_name_of(e).lower() for e in star.args.get("except_") or []}
    replaced = {_name_of(r).lower(): r for r in star.args.get("replace") or []}
    out: list[exp.Expression] = []
    for name in names:
        key = name.lower()
        if key in excluded:
            continue
        if key in replaced:
            out.append(replaced[key].copy())
            continue
        field = exp.Dot(this=operand.copy(), expression=exp.to_identifier(name))
        out.append(exp.alias_(field, name))
    return out


def _name_of(node: exp.Expression) -> str:
    """Field name an EXCEPT column or REPLACE alias refers to."""
    if isinstance(node, exp.Alias):
        return node.alias
    return node.name


__all__ = ["expand_struct_star"]
