"""Field-name resolution for STRUCT-valued DuckDB expressions.

BigQuery lets a query expand a struct by name (``expr.*``) or unnest an
array of structs into one column per field without ever spelling the
field names out; DuckDB needs them spelled out. These helpers recover
the names from, in order, SQLGlot's ``annotate_types`` result when the
translator ran it with a catalog schema, and the expression's own shape
otherwise: the ``{'name': …}`` literal ``SELECT AS STRUCT a AS name``
transpiles to, reached through ``ARRAY(SELECT …)``, ``(SELECT …)``,
``[STRUCT(…)]``, ``CAST(… AS STRUCT(…))`` and parentheses.
"""

from __future__ import annotations

from sqlglot import exp


def struct_field_names(node: exp.Expression) -> list[str] | None:
    """Field names of the STRUCT *node* evaluates to, or ``None`` when unknown."""
    names = _names_from_type(node.type)
    if names is not None:
        return names
    if isinstance(node, exp.Struct):
        return _names_from_struct_literal(node)
    if isinstance(node, exp.Cast):
        return _names_from_type(node.to)
    if isinstance(node, exp.Paren):
        return struct_field_names(node.this)
    if isinstance(node, exp.Subquery):
        return _names_from_select(node.this)
    if isinstance(node, exp.Bracket):
        return element_field_names(node.this)
    return None


def element_field_names(node: exp.Expression) -> list[str] | None:
    """Field names of the STRUCT elements of the array *node* evaluates to."""
    names = _names_from_type(_element_type(node.type))
    if names is not None:
        return names
    if isinstance(node, exp.Paren):
        return element_field_names(node.this)
    if isinstance(node, exp.Array):
        first = node.expressions[0] if node.expressions else None
        if isinstance(first, exp.Query):
            return _names_from_select(first)
        return struct_field_names(first) if first is not None else None
    if isinstance(node, exp.Cast):
        return _names_from_type(_element_type(node.to))
    return None


def is_anonymous_field(name: str) -> bool:
    """Whether *name* is the ``_N`` SQLGlot gives a positional struct field."""
    return name.startswith("_") and name[1:].isdigit()


def _names_from_select(query: exp.Expression | exp.Query) -> list[str] | None:
    """Field names of a single-projection ``SELECT {'a': …}`` query."""
    if not isinstance(query, exp.Select) or len(query.expressions) != 1:
        return None
    projection = query.expressions[0]
    if isinstance(projection, exp.Alias):
        projection = projection.this
    return struct_field_names(projection)


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


__all__ = ["element_field_names", "is_anonymous_field", "struct_field_names"]
