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

_ANONYMOUS_FIELD_PREFIX = "_"


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
        names = _struct_field_names(operand)
        if names is None or any(_is_anonymous(n) for n in names):
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


def _is_anonymous(name: str) -> bool:
    """Whether *name* is the ``_N`` SQLGlot gives a positional struct field."""
    return name.startswith(_ANONYMOUS_FIELD_PREFIX) and name[1:].isdigit()


def _struct_field_names(node: exp.Expression) -> list[str] | None:
    """Resolve the field names of the STRUCT *node* evaluates to, or ``None``."""
    names = _names_from_type(node.type)
    if names is not None:
        return names
    if isinstance(node, exp.Struct):
        return _names_from_struct_literal(node)
    if isinstance(node, exp.Cast):
        return _names_from_type(node.to)
    if isinstance(node, exp.Paren):
        return _struct_field_names(node.this)
    if isinstance(node, exp.Subquery):
        return _names_from_select(node.this)
    if isinstance(node, exp.Bracket):
        return _element_field_names(node.this)
    return None


def _element_field_names(node: exp.Expression) -> list[str] | None:
    """Field names of the elements of the array *node* evaluates to."""
    names = _names_from_type(_element_type(node.type))
    if names is not None:
        return names
    if isinstance(node, exp.Paren):
        return _element_field_names(node.this)
    if isinstance(node, exp.Array):
        first = node.expressions[0] if node.expressions else None
        if isinstance(first, exp.Query):
            return _names_from_select(first)
        return _struct_field_names(first) if first is not None else None
    if isinstance(node, exp.Cast):
        return _names_from_type(_element_type(node.to))
    return None


def _names_from_select(query: exp.Expression) -> list[str] | None:
    """Field names of a single-projection ``SELECT {'a': …}`` query."""
    if not isinstance(query, exp.Select) or len(query.expressions) != 1:
        return None
    projection = query.expressions[0]
    if isinstance(projection, exp.Alias):
        projection = projection.this
    return _struct_field_names(projection)


def _names_from_struct_literal(struct: exp.Struct) -> list[str] | None:
    """Keys of a ``{'a': x, 'b': y}`` literal; ``None`` when any entry is unnamed."""
    names: list[str] = []
    for entry in struct.expressions:
        if not isinstance(entry, exp.PropertyEQ):
            return None
        names.append(entry.this.name)
    return names


def _names_from_type(data_type: exp.DataType | None) -> list[str] | None:
    """Column names of a ``STRUCT(a T, b U)`` type annotation."""
    if data_type is None or not data_type.is_type(exp.DataType.Type.STRUCT):
        return None
    names: list[str] = []
    for field in data_type.expressions:
        if not isinstance(field, exp.ColumnDef):
            return None
        names.append(field.name)
    return names


def _element_type(data_type: exp.DataType | None) -> exp.DataType | None:
    """Element type of an ``ARRAY<T>`` annotation, or ``None``."""
    if data_type is None or not data_type.is_type(exp.DataType.Type.ARRAY):
        return None
    element = data_type.expressions[0] if data_type.expressions else None
    return element if isinstance(element, exp.DataType) else None


__all__ = ["expand_struct_star"]
