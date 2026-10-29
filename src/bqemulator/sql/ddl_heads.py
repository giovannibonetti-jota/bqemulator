"""parsy grammar for the DDL heads SQLGlot leaves untyped.

SQLGlot's BigQuery dialect returns ``CREATE / DROP SNAPSHOT TABLE``,
``CREATE TABLE … CLONE … OPTIONS(…)``, ``CALL BQ.REFRESH_MATERIALIZED_VIEW``
and the ``ROW ACCESS POLICY`` statements as an opaque :class:`exp.Command`,
raises on ``DROP TABLE FUNCTION``, and types the transaction-control
words only in some spellings (``START TRANSACTION`` parses as a column
alias). Each of those heads used to be a hand-written regex that solved
comments, backticks, hyphenated project ids and string literals on its
own, with the gaps (and the ReDoS incidents) that implies.

This module keeps one token layer — whitespace and both comment forms,
case-insensitive keywords, identifiers, string literals, balanced
brackets — and one combinator per head. Bodies that are real SQL (a
``FILTER USING`` predicate, a ``FOR SYSTEM_TIME AS OF`` expression) are
returned as text for the SQL translator; the grammar never parses
expressions. Brackets are matched by an iterative leaf rather than a
recursive combinator, so nesting depth is bounded by the input, not by
the interpreter's recursion limit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import ClassVar

from parsy import ParseError, Parser, Result, regex, seq, string

from bqemulator.domain.errors import InvalidQueryError

# ---------------------------------------------------------------------------
# Token layer
# ---------------------------------------------------------------------------

#: Whitespace plus ``--`` and ``/* */`` comments. Every alternative consumes
#: at least one character through exactly one branch, so matching is linear.
_WS = regex(r"(?:\s+|--[^\n]*|/\*(?:[^*]|\*(?!/))*\*/)*").desc("whitespace")


def _kw(words: str) -> Parser:
    """Case-insensitive keyword sequence, e.g. ``_kw("IF NOT EXISTS")``.

    Each word is anchored on a word boundary, so ``BEGIN`` never matches
    the start of ``BEGINNING``.
    """
    parts = [regex(rf"{re.escape(w)}\b", flags=re.IGNORECASE) << _WS for w in words.split()]
    return seq(*parts).result(words).desc(words)


def _skip_string(stream: str, start: int) -> int:
    """Index just past the string literal that opens at ``start``."""
    quote = stream[start]
    i = start + 1
    n = len(stream)
    while i < n and stream[i] != quote:
        i += 2 if stream[i] == "\\" else 1
    return min(i + 1, n)


def _group(open_ch: str, close_ch: str) -> Parser:
    """Balanced-bracket leaf; the parse result is the text inside the brackets.

    An iterative scan that respects string literals and nesting, wrapped
    as a parsy leaf so it composes like any other token.
    """

    def group(stream: str, index: int) -> Result:
        if index >= len(stream) or stream[index] != open_ch:
            return Result.failure(index, f"'{open_ch}'")
        depth = 0
        i = index
        n = len(stream)
        while i < n:
            ch = stream[i]
            if ch in "'\"":
                i = _skip_string(stream, i)
                continue
            if ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    return Result.success(i + 1, stream[index + 1 : i])
            i += 1
        return Result.failure(index, f"'{close_ch}' closing the '{open_ch}'")

    return Parser(group) << _WS


def _text_until(*stops: str) -> Parser:
    """Raw SQL text up to a top-level stop keyword, ``;`` or the end of input.

    Used for the expression bodies the grammar hands to the translator
    verbatim: the scan tracks string literals and bracket depth so a stop
    word inside either does not end the text.
    """
    stop_re = re.compile("|".join(rf"{re.escape(s)}\b" for s in stops), re.IGNORECASE)

    def until(stream: str, index: int) -> Result:
        depth = 0
        i = index
        n = len(stream)
        while i < n:
            ch = stream[i]
            if ch in "'\"":
                i = _skip_string(stream, i)
                continue
            if ch in "([":
                depth += 1
            elif ch in ")]":
                depth -= 1
            elif (depth == 0 and ch == ";") or (
                depth == 0 and (i == index or stream[i - 1].isspace()) and stop_re.match(stream, i)
            ):
                break
            i += 1
        text = stream[index:i].strip()
        if not text:
            return Result.failure(index, "expression")
        return Result.success(i, text)

    return Parser(until)


_PARENS = _group("(", ")")
_BRACKETS = _group("[", "]")
_END = (_WS >> string(";").optional() >> _WS >> regex(r"$")).desc("end of statement")

_STRING = regex(r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"").desc("string literal")
_STRING_RAW = _STRING << _WS
_STRING_VALUE = _STRING.map(lambda s: s[1:-1]) << _WS

# Backticked (possibly whole-name ``\`p.ds.t\```) or bare; bare parts admit
# the hyphen of a project id.
_IDENT = (regex(r"`[^`]+`").map(lambda s: s[1:-1]) | regex(r"[A-Za-z_][A-Za-z0-9_-]*")).desc(
    "identifier"
) << _WS
_NAME = _IDENT.sep_by(string(".") << _WS, min=1).map(".".join).desc("qualified name")
_POLICY_ID = (
    regex(r"`[A-Za-z_][A-Za-z0-9_]*`").map(lambda s: s[1:-1]) | regex(r"[A-Za-z_][A-Za-z0-9_]*")
).desc("policy id") << _WS

# OPTIONS(name = value, …): values stay as BigQuery text; ``option_value``
# turns the literal ones into Python values.
_OPTION_KEY = regex(r"[A-Za-z_][A-Za-z0-9_]*").desc("option name") << _WS
_OPTION_ATOM = (
    _STRING_RAW
    | _PARENS.map(lambda s: f"({s})")
    | _BRACKETS.map(lambda s: f"[{s}]")
    | regex(r"""[^,()\[\]'"\s]+""") << _WS
)
_OPTION_VALUE = _OPTION_ATOM.at_least(1).map(" ".join).desc("option value")
_OPTION = seq(_OPTION_KEY.map(str.lower) << string("=") << _WS, _OPTION_VALUE).map(tuple)
_OPTIONS = (
    _kw("OPTIONS") >> string("(") >> _WS >> _OPTION.sep_by(string(",") << _WS) << string(")") << _WS
).map(dict)
_OPTIONS_OR_EMPTY = _OPTIONS.optional().map(lambda d: d or {})


def option_value(raw: str) -> object:
    """Python value of an ``OPTIONS`` entry: literals convert, anything else stays text."""
    if len(raw) >= 2 and raw[0] in "'\"" and raw[-1] == raw[0]:  # noqa: PLR2004
        return raw[1:-1].replace("\\" + raw[0], raw[0])
    upper = raw.upper()
    if upper == "TRUE":
        return True
    if upper == "FALSE":
        return False
    if upper == "NULL":
        return None
    if re.fullmatch(r"-?\d+", raw):
        return int(raw)
    return raw


# ---------------------------------------------------------------------------
# Heads
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CreateSnapshot:
    """``CREATE [OR REPLACE] SNAPSHOT TABLE [IF NOT EXISTS] t CLONE s [FOR SYSTEM_TIME …]``."""

    statement_type: ClassVar[str] = "CREATE_SNAPSHOT_TABLE"
    target: str
    source: str
    replace: bool = False
    if_not_exists: bool = False
    system_time: str | None = None
    options: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DropSnapshot:
    """``DROP SNAPSHOT TABLE [IF EXISTS] t``."""

    statement_type: ClassVar[str] = "DROP_SNAPSHOT_TABLE"
    target: str
    if_exists: bool = False


@dataclass(frozen=True, slots=True)
class CreateClone:
    """``CREATE [OR REPLACE] TABLE [IF NOT EXISTS] t CLONE s [FOR SYSTEM_TIME AS OF …] [OPTIONS]``.

    Only the forms SQLGlot does not type reach this head (a system time or
    an ``OPTIONS`` clause); the bare form is an ``exp.Create``.
    """

    statement_type: ClassVar[str] = "CREATE_TABLE"
    target: str
    source: str
    replace: bool = False
    if_not_exists: bool = False
    system_time: str | None = None
    options: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CallRefresh:
    """``CALL BQ.REFRESH_MATERIALIZED_VIEW('p.ds.mv')``.

    Reaches the grammar only outside a script (the scripting parser owns
    ``CALL``), so it carries no statement type of its own.
    """

    statement_type: ClassVar[str] = ""
    target: str


@dataclass(frozen=True, slots=True)
class RefreshMaterializedView:
    """``REFRESH MATERIALIZED VIEW mv`` — the emulator's own extension (ADR 0017)."""

    statement_type: ClassVar[str] = ""
    target: str


@dataclass(frozen=True, slots=True)
class CreateRowAccessPolicy:
    """``CREATE [OR REPLACE] ROW ACCESS POLICY [IF NOT EXISTS] p ON t [GRANT TO …] FILTER …``."""

    statement_type: ClassVar[str] = "CREATE_ROW_ACCESS_POLICY"
    policy: str
    table: str
    filter: str
    grantees: tuple[str, ...] | None = None
    replace: bool = False
    if_not_exists: bool = False


@dataclass(frozen=True, slots=True)
class DropRowAccessPolicy:
    """``DROP ROW ACCESS POLICY [IF EXISTS] p ON t``."""

    statement_type: ClassVar[str] = "DROP_ROW_ACCESS_POLICY"
    policy: str
    table: str
    if_exists: bool = False


@dataclass(frozen=True, slots=True)
class DropAllRowAccessPolicies:
    """``DROP ALL ROW ACCESS POLICIES ON t``."""

    statement_type: ClassVar[str] = "DROP_ROW_ACCESS_POLICY"
    table: str


@dataclass(frozen=True, slots=True)
class DropTableFunction:
    """``DROP TABLE FUNCTION [IF EXISTS] f``."""

    statement_type: ClassVar[str] = "DROP_TABLE_FUNCTION"
    name: str
    if_exists: bool = False


DDLHead = (
    CreateSnapshot
    | DropSnapshot
    | CreateClone
    | CallRefresh
    | RefreshMaterializedView
    | CreateRowAccessPolicy
    | DropRowAccessPolicy
    | DropAllRowAccessPolicies
    | DropTableFunction
)

_CREATE = _WS >> _kw("CREATE") >> _kw("OR REPLACE").optional().map(bool)
_IF_NOT_EXISTS = _kw("IF NOT EXISTS").optional().map(bool)
_IF_EXISTS = _kw("IF EXISTS").optional().map(bool)
_SYSTEM_TIME = (_kw("FOR SYSTEM_TIME AS OF") >> _text_until("OPTIONS")).optional()

_create_snapshot = (
    seq(
        _CREATE << _kw("SNAPSHOT TABLE"),
        _IF_NOT_EXISTS,
        _NAME,
        (_kw("CLONE") | _kw("COPY")) >> _NAME,
        _SYSTEM_TIME,
        _OPTIONS_OR_EMPTY,
    ).combine(
        lambda replace, ine, target, source, system_time, options: CreateSnapshot(
            target, source, replace, ine, system_time, options
        )
    )
    << _END
)
_drop_snapshot = (
    seq(_WS >> _kw("DROP SNAPSHOT TABLE") >> _IF_EXISTS, _NAME).combine(
        lambda if_exists, target: DropSnapshot(target, if_exists)
    )
    << _END
)
_create_clone = (
    seq(
        _CREATE << _kw("TABLE"),
        _IF_NOT_EXISTS,
        _NAME << _kw("CLONE"),
        _NAME,
        _SYSTEM_TIME,
        _OPTIONS_OR_EMPTY,
    ).combine(
        lambda replace, ine, target, source, system_time, options: CreateClone(
            target, source, replace, ine, system_time, options
        )
    )
    << _END
)
_call_refresh = (
    _WS
    >> _kw("CALL")
    >> _kw("BQ.REFRESH_MATERIALIZED_VIEW")
    >> string("(")
    >> _WS
    >> _STRING_VALUE.map(CallRefresh)
    << string(")")
    << _END
)
_refresh_extension = (
    _WS >> _kw("REFRESH MATERIALIZED VIEW") >> _NAME.map(RefreshMaterializedView)
) << _END
_GRANTEES = (
    _kw("GRANT TO")
    >> string("(")
    >> _WS
    >> _STRING_VALUE.sep_by(string(",") << _WS, min=1)
    << string(")")
    << _WS
).map(tuple)
_create_rap = (
    seq(
        _CREATE << _kw("ROW ACCESS POLICY"),
        _IF_NOT_EXISTS,
        _POLICY_ID,
        _kw("ON") >> _NAME,
        _GRANTEES.optional(),
        _kw("FILTER USING") >> _PARENS.map(str.strip),
    ).combine(
        lambda replace, ine, policy, table, grantees, predicate: CreateRowAccessPolicy(
            policy, table, predicate, grantees, replace, ine
        )
    )
    << _END
)
_drop_rap = (
    seq(
        _WS >> _kw("DROP ROW ACCESS POLICY") >> _IF_EXISTS,
        _POLICY_ID,
        _kw("ON") >> _NAME,
    ).combine(lambda if_exists, policy, table: DropRowAccessPolicy(policy, table, if_exists))
    << _END
)
_drop_all_rap = (
    _WS >> _kw("DROP ALL ROW ACCESS POLICIES") >> _kw("ON") >> _NAME.map(DropAllRowAccessPolicies)
) << _END
_drop_table_function = (
    seq(_WS >> _kw("DROP TABLE FUNCTION") >> _IF_EXISTS, _NAME).combine(
        lambda if_exists, name: DropTableFunction(name, if_exists)
    )
    << _END
)

#: ``(statement name, prefix that identifies the head, full parser)``. The
#: prefix decides which head a statement is *trying* to be, so a malformed
#: snapshot DDL is reported as such instead of falling through to DuckDB.
_HEADS: tuple[tuple[str, Parser, Parser], ...] = (
    (
        "CREATE SNAPSHOT TABLE",
        _CREATE >> _kw("SNAPSHOT TABLE"),
        _create_snapshot,
    ),
    ("DROP SNAPSHOT TABLE", _WS >> _kw("DROP SNAPSHOT TABLE"), _drop_snapshot),
    (
        "CREATE TABLE … CLONE",
        _CREATE >> _kw("TABLE") >> _IF_NOT_EXISTS >> _NAME >> _kw("CLONE"),
        _create_clone,
    ),
    (
        "CALL BQ.REFRESH_MATERIALIZED_VIEW",
        _WS >> _kw("CALL") >> _kw("BQ.REFRESH_MATERIALIZED_VIEW"),
        _call_refresh,
    ),
    (
        "REFRESH MATERIALIZED VIEW",
        _WS >> _kw("REFRESH MATERIALIZED VIEW"),
        _refresh_extension,
    ),
    (
        "CREATE ROW ACCESS POLICY",
        _CREATE >> _kw("ROW ACCESS POLICY"),
        _create_rap,
    ),
    ("DROP ROW ACCESS POLICY", _WS >> _kw("DROP ROW ACCESS POLICY"), _drop_rap),
    (
        "DROP ALL ROW ACCESS POLICIES",
        _WS >> _kw("DROP ALL ROW ACCESS POLICIES"),
        _drop_all_rap,
    ),
    ("DROP TABLE FUNCTION", _WS >> _kw("DROP TABLE FUNCTION"), _drop_table_function),
)


def parse_ddl_head(sql: str) -> DDLHead | None:
    """Parse ``sql`` as one of the untyped DDL heads, or return ``None``.

    ``sql`` is one statement (scripts are split by the scripting parser
    before any router sees them). ``None`` means the statement is none of
    the heads, so the caller continues with its normal path. A statement
    that starts like one of the heads but does not parse — a second
    statement after it included — raises :class:`InvalidQueryError` with
    BigQuery's ``Syntax error`` wording and the offending position.
    """
    for name, prefix, full in _HEADS:
        try:
            prefix.parse_partial(sql)
        except ParseError:
            continue
        try:
            head: DDLHead = full.parse(sql)
        except ParseError as exc:
            raise InvalidQueryError(_syntax_error(name, sql, exc)) from exc
        return head
    return None


def _syntax_error(name: str, sql: str, exc: ParseError) -> str:
    """BigQuery-shaped message: ``Syntax error: … at [line:column]``."""
    line = sql.count("\n", 0, exc.index) + 1
    column = exc.index - (sql.rfind("\n", 0, exc.index) + 1) + 1
    expected = ", ".join(sorted(exc.expected))
    got = sql[exc.index : exc.index + 20].split("\n", 1)[0].strip() or "end of input"
    return f"Syntax error: {name}: expected {expected} but got {got!r} at [{line}:{column}]"


# ---------------------------------------------------------------------------
# Transaction control
# ---------------------------------------------------------------------------

_TRANSACTION_CONTROL = (
    _WS
    >> (
        (_kw("BEGIN TRANSACTION") | _kw("START TRANSACTION") | _kw("BEGIN")).result("BEGIN")
        | (
            _kw("COMMIT TRANSACTION") | _kw("COMMIT WORK") | _kw("END TRANSACTION") | _kw("COMMIT")
        ).result("COMMIT")
        | (_kw("ROLLBACK TRANSACTION") | _kw("ROLLBACK WORK") | _kw("ROLLBACK")).result("ROLLBACK")
    )
    << _END
)


def parse_transaction_control(sql: str) -> str | None:
    """``"BEGIN"`` / ``"COMMIT"`` / ``"ROLLBACK"`` for transaction control, else ``None``.

    Whole-statement only: ``BEGIN … END`` blocks and ``ROLLBACK TO
    SAVEPOINT`` are not transaction control and return ``None``.
    """
    try:
        operation: str = _TRANSACTION_CONTROL.parse(sql)
    except ParseError:
        return None
    return operation


__all__ = [
    "CallRefresh",
    "CreateClone",
    "CreateRowAccessPolicy",
    "CreateSnapshot",
    "DDLHead",
    "DropAllRowAccessPolicies",
    "DropRowAccessPolicy",
    "DropSnapshot",
    "DropTableFunction",
    "RefreshMaterializedView",
    "option_value",
    "parse_ddl_head",
    "parse_transaction_control",
]
