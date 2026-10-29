"""BigQuery's materialized-view query rules, checked before a view is materialised.

BigQuery accepts only a subset of GoogleSQL as a materialized-view body
(https://docs.cloud.google.com/bigquery/docs/materialized-views-create#query_limitations);
DuckDB would materialise any SELECT. The checks here run on the parsed
body so a view BigQuery would refuse never exists in the emulator.

Two rule sets apply:

* **Every** materialized view: no non-deterministic functions, no
  user-defined functions, no generative-AI functions, no ``TABLESAMPLE``,
  no ``FOR SYSTEM_TIME AS OF``, no system variables.
* **Incremental** views (the default) additionally follow the
  ``WITH / SELECT / FROM / WHERE / GROUP BY`` pattern: no set operations
  (``UNION ALL`` included — its incremental support is a preview the
  emulator does not model), only ``INNER``/``CROSS`` joins (``LEFT OUTER
  JOIN`` is the same preview), no self-joins, no window functions, no
  ``ARRAY`` subqueries, no ``HAVING`` / ``QUALIFY`` / ``ORDER BY`` /
  ``LIMIT``, and aggregates restricted to BigQuery's list, each a plain
  select-list output of the single aggregating SELECT.

``allow_non_incremental_definition=true`` lifts the second set only, as
BigQuery does. Restrictions that need column types (``ANY_VALUE`` /
``ARRAY_AGG`` / ``MAX_BY`` / ``MIN_BY`` over ``STRUCT``, ``ARRAY_AGG``
over ``ARRAY``) are not checked.
"""

from __future__ import annotations

from collections.abc import Iterator

from sqlglot import exp

from bqemulator.domain.errors import InvalidQueryError

_NON_DETERMINISTIC: tuple[type[exp.Expr], ...] = (
    exp.Rand,
    exp.CurrentDate,
    exp.CurrentTime,
    exp.CurrentTimestamp,
    exp.CurrentDatetime,
    exp.SessionUser,
    exp.Uuid,
)

# Function namespaces BigQuery ships; anything else before a dot is a UDF.
_BUILTIN_NAMESPACES = frozenset(
    {"AI", "BQ", "HLL_COUNT", "KLL_QUANTILES", "ML", "NET", "SAFE", "D3A_COUNT"},
)

_GENERATIVE_AI: tuple[type[exp.Expr], ...] = tuple(
    getattr(exp, name)
    for name in (
        "AIAgg",
        "AIClassify",
        "AIEmbed",
        "AIForecast",
        "AIGenerate",
        "AISimilarity",
        "AISummarizeAgg",
        "GenerateBool",
        "GenerateDouble",
        "GenerateEmbedding",
        "GenerateInt",
        "GenerateTable",
        "GenerateText",
    )
    if hasattr(exp, name)
)

# Aggregates an incremental materialized view may use (docs: "Aggregate
# requirements"), by SQLGlot node; ``HLL_COUNT.INIT`` parses as a dotted
# anonymous call and is matched by name.
_INCREMENTAL_AGGREGATES: tuple[type[exp.Expr], ...] = (
    exp.AnyValue,
    exp.ApproxDistinct,
    exp.ArrayAgg,
    exp.Avg,
    exp.BitwiseAndAgg,
    exp.BitwiseOrAgg,
    exp.BitwiseXorAgg,
    exp.Count,
    exp.CountIf,
    exp.LogicalAnd,
    exp.LogicalOr,
    exp.Max,
    exp.Min,
    exp.ArgMax,
    exp.ArgMin,
    exp.Sum,
)

_OUTER_JOIN_SIDES = frozenset({"LEFT", "RIGHT", "FULL"})
_QUERY_NODES: tuple[type[exp.Expr], ...] = (exp.Select, exp.Subquery, exp.SetOperation)


def validate_materialized_view_query(
    tree: exp.Expr,
    *,
    non_incremental: bool = False,
) -> None:
    """Raise :class:`InvalidQueryError` if ``tree`` is not a valid MV body."""
    problems = query_problems(tree, non_incremental=non_incremental)
    if problems:
        raise InvalidQueryError(
            "Invalid materialized view query: " + "; ".join(problems),
        )


def query_problems(
    tree: exp.Expr,
    *,
    non_incremental: bool = False,
) -> list[str]:
    """Return every rule ``tree`` breaks, in document order of the rules.

    Empty when the body is a valid BigQuery materialized-view query.
    """
    problems = list(_universal_problems(tree))
    if not non_incremental:
        problems.extend(_incremental_problems(tree))
    return problems


# ---------------------------------------------------------------------------
# Rules for every materialized view
# ---------------------------------------------------------------------------


def _universal_problems(tree: exp.Expr) -> Iterator[str]:
    for func in tree.find_all(exp.Func):
        if isinstance(func, _NON_DETERMINISTIC):
            yield f"non-deterministic function {_call_name(func)}()"
    for dot in tree.find_all(exp.Dot):
        namespace, call = _dotted_call(dot)
        if call is None:
            continue
        if namespace not in _BUILTIN_NAMESPACES:
            yield f"user-defined function {dot.sql('bigquery')}"
        elif namespace == "AI" or (namespace == "ML" and _call_name(call).startswith("GENERATE")):
            yield f"generative AI function {namespace}.{_call_name(call)}"
    for func in tree.find_all(exp.Func):
        if isinstance(func, _GENERATIVE_AI) and not isinstance(func.parent, exp.Dot):
            yield f"generative AI function {_call_name(func)}"
    if tree.find(exp.TableSample):
        yield "TABLESAMPLE"
    if tree.find(exp.Version):
        yield "FOR SYSTEM_TIME AS OF"
    for parameter in tree.find_all(exp.Parameter, exp.SessionParameter):
        rendered = parameter.sql("bigquery")
        if rendered.startswith("@@"):
            yield f"system variable {rendered}"


# ---------------------------------------------------------------------------
# Rules for incremental materialized views
# ---------------------------------------------------------------------------


def _incremental_problems(tree: exp.Expr) -> Iterator[str]:
    for set_op in tree.find_all(exp.SetOperation):
        yield _set_operation_name(set_op)
    for join in tree.find_all(exp.Join):
        if join.side in _OUTER_JOIN_SIDES:
            yield f"{join.side} OUTER JOIN"
    yield from _self_join_problems(tree)
    if tree.find(exp.Window):
        yield "window function"
    for array in tree.find_all(exp.Array):
        if any(isinstance(e, _QUERY_NODES) for e in array.expressions):
            yield "ARRAY subquery"
    if tree.find(exp.Having):
        yield "HAVING"
    if tree.find(exp.Qualify):
        yield "QUALIFY"
    if any(isinstance(o.parent, (exp.Select, exp.SetOperation)) for o in tree.find_all(exp.Order)):
        yield "ORDER BY"
    if tree.find(exp.Limit):
        yield "LIMIT"
    yield from _aggregate_problems(tree)


def _self_join_problems(tree: exp.Expr) -> Iterator[str]:
    """Flag a physical table referenced more than once (CTE names excluded)."""
    cte_names = {cte.alias_or_name for cte in tree.find_all(exp.CTE)}
    seen: dict[str, int] = {}
    for table in tree.find_all(exp.Table):
        if isinstance(table.this, exp.Anonymous):  # table-valued function
            continue
        if not table.db and table.name in cte_names:
            continue
        key = ".".join(part for part in (table.catalog, table.db, table.name) if part)
        seen[key] = seen.get(key, 0) + 1
    for key, count in seen.items():
        if count > 1:
            yield f"self-join on {key} (referenced {count} times)"


def _aggregate_problems(tree: exp.Expr) -> Iterator[str]:
    """Apply the "Aggregate requirements": allowed list, output-only, one level."""
    aggregating: dict[int, exp.Select] = {}
    for select in tree.find_all(exp.Select):
        if select.args.get("group"):
            aggregating[id(select)] = select
    for node in _aggregate_calls(tree):
        name = _call_name(node)
        if not (isinstance(node, _INCREMENTAL_AGGREGATES) or name == "HLL_COUNT.INIT"):
            yield f"aggregate function {name}"
        owner = node.find_ancestor(exp.Select)
        if owner is not None:
            aggregating.setdefault(id(owner), owner)
        if not _is_select_item(node):
            yield f"aggregate {name} is not an output column (aggregates must be select-list items)"
    if len(aggregating) > 1:
        yield "more than one GROUP BY level (aggregates over aggregates)"
    for select in aggregating.values():
        if not _is_query_source(select):
            yield "aggregate in a subquery expression (filtering or computing on an aggregate)"


def _aggregate_calls(tree: exp.Expr) -> Iterator[exp.Expr]:
    """Yield aggregate calls, skipping the analytic (``OVER``) uses."""
    for agg in tree.find_all(exp.AggFunc):
        if not isinstance(agg.parent, exp.Window):
            yield agg
    for dot in tree.find_all(exp.Dot):
        namespace, call = _dotted_call(dot)
        if call is not None and namespace == "HLL_COUNT" and _call_name(call) == "INIT":
            yield dot


def _is_select_item(node: exp.Expr) -> bool:
    """True when ``node`` is a select-list item itself or under its alias."""
    parent = node.parent
    if isinstance(parent, exp.Alias):
        parent = parent.parent
    return isinstance(parent, exp.Select)


def _is_query_source(select: exp.Select) -> bool:
    """True when ``select`` is the statement, a CTE body, or a FROM/JOIN source."""
    parent = select.parent
    if parent is None or isinstance(parent, (exp.CTE, exp.SetOperation)):
        return True
    if isinstance(parent, exp.Subquery):
        return isinstance(parent.parent, (exp.From, exp.Join))
    return False


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _dotted_call(node: exp.Dot) -> tuple[str, exp.Func | None]:
    """Return ``(leftmost namespace, call)`` for ``ns.fn(...)``; call is None otherwise."""
    call = node.expression
    if not isinstance(call, exp.Func):
        return "", None
    head: exp.Expr = node.this
    while isinstance(head, exp.Dot):
        head = head.this
    return head.name.upper(), call


def _call_name(node: exp.Expr) -> str:
    """The function name as BigQuery spells it (``STRING_AGG``, ``HLL_COUNT.INIT``)."""
    return node.sql("bigquery").split("(", 1)[0].strip().upper()


def _set_operation_name(node: exp.SetOperation) -> str:
    keyword = {exp.Union: "UNION", exp.Intersect: "INTERSECT", exp.Except: "EXCEPT"}.get(
        type(node), type(node).__name__.upper()
    )
    return f"{keyword} {'DISTINCT' if node.args.get('distinct') else 'ALL'}"


__all__ = ["query_problems", "validate_materialized_view_query"]
