"""Post-translator for ``FROM UNNEST(<array of STRUCT>) AS s``.

BigQuery's ``UNNEST`` of an array of structs yields a table with one
column per field, so ``SELECT label FROM UNNEST(arr) AS s`` and
``s.label`` both bind. DuckDB's table-function ``UNNEST`` yields one
struct column: ``s.label`` binds, bare ``label`` does not. SQLGlot
already emits ``(SELECT UNNEST(arr, max_depth => 2)) AS s`` — which
unnests the list *and* the struct — when the array is a literal
``[STRUCT(…), …]``, the only case it can recognise syntactically.

This pass gives every other ``FROM`` / ``JOIN`` ``UNNEST`` whose
element struct can be named — an ``ARRAY(SELECT AS STRUCT …)``
subquery, a cast, an array column whose catalog type SQLGlot managed
to attach (it only does when it can qualify the whole query, which a
bare field reference defeats) — the equivalent shape ``(SELECT UNNEST(elem) FROM UNNEST(arr)
AS …(elem)) AS s``: the inner table-function ``UNNEST`` evaluates the
array once and yields one struct per row, the outer ``UNNEST`` of that
struct value spreads its fields into columns. ``WITH ORDINALITY``
rides along as a second inner column. Arrays whose elements cannot be
named are left alone, so scalar arrays keep DuckDB's native behaviour.

Like SQLGlot's literal rewrite, the BigQuery alias becomes the
subquery's table alias: ``s.field`` and bare ``field`` bind, a bare
``s`` referring to the whole struct does not.
"""

from __future__ import annotations

from sqlglot import exp

from bqemulator.sql.struct_fields import element_field_names, is_anonymous_field

_ROWS_ALIAS = "__bqemu_rows"
_ELEMENT_ALIAS = "__bqemu_elem"


def expose_unnest_struct_columns(tree: exp.Expression) -> bool:
    """``FROM UNNEST(<array of STRUCT>) AS s`` → ``(SELECT UNNEST(elem) FROM UNNEST(arr) …) AS s``.

    Returns ``True`` when the tree was modified.
    """
    modified = False
    for unnest in list(tree.find_all(exp.Unnest)):
        if not isinstance(unnest.parent, (exp.From, exp.Join)) or len(unnest.expressions) != 1:
            continue
        array = unnest.expressions[0]
        names = element_field_names(array)
        if not names or any(is_anonymous_field(n) for n in names):
            continue
        unnest.replace(_struct_rows(unnest, array))
        modified = True
    return modified


def _struct_rows(unnest: exp.Unnest, array: exp.Expression) -> exp.Subquery:
    """Build ``(SELECT UNNEST(e.elem)[, e.o AS o] FROM UNNEST(arr) [WITH ORDINALITY] AS e(…))``.

    The array is evaluated once by the inner table-function ``UNNEST``;
    the outer ``UNNEST(elem)`` of a struct *value* spreads its fields
    into columns. Zipping a second ``UNNEST`` over a copy of the array
    would be simpler but DuckDB refuses to copy an array built by a
    subquery (``Cannot copy BoundSubqueryExpression``).
    """
    element = exp.to_identifier(_ELEMENT_ALIAS)
    projections: list[exp.Expression] = [
        exp.Anonymous(this="UNNEST", expressions=[exp.column(element.copy(), _ROWS_ALIAS)])
    ]
    # SQLGlot keeps the ordinal column's name in ``offset`` (``False``
    # when absent) and renders ``WITH ORDINALITY AS rows(elem, name)``.
    offset = unnest.args.get("offset")
    ordinal = offset.copy() if isinstance(offset, exp.Identifier) else None
    if ordinal is not None:
        projections.append(exp.alias_(exp.column(ordinal.copy(), _ROWS_ALIAS), ordinal.copy()))
    rows = exp.Unnest(
        expressions=[array.copy()],
        offset=ordinal,
        alias=exp.TableAlias(this=exp.to_identifier(_ROWS_ALIAS), columns=[element]),
    )
    alias = unnest.args.get("alias")
    table_alias: exp.TableAlias | None = None
    if isinstance(alias, exp.TableAlias):
        name = alias.columns[0] if alias.columns else alias.this
        table_alias = exp.TableAlias(this=name.copy())
    select = exp.Select(expressions=projections).from_(rows)
    return exp.Subquery(this=select, alias=table_alias)


__all__ = ["expose_unnest_struct_columns"]
