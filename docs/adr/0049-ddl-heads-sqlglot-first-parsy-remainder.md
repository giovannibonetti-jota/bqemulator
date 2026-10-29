# ADR 0049: Statement heads — SQLGlot's AST first, a parsy grammar for the remainder

**Status**: Accepted
**Date**: 2026-10-29

## Context

The job executor, the script interpreter and the versioning router each
recognised a family of statements DuckDB cannot run — materialized views,
snapshots, clones, row access policies, `EXPORT DATA`, `CREATE MODEL`,
`DROP TABLE FUNCTION`, transaction control — with a hand-written regex per
head. Each regex re-solved tokenisation on its own: comments, backticked
and hyphenated identifiers, string literals with escaped quotes, nested
parentheses. Each got a different subset right. The gaps were ordinary
BigQuery:

- dbt-bigquery's `CREATE MATERIALIZED VIEW IF NOT EXISTS … CLUSTER BY …
  OPTIONS(…)` missed the router entirely and became an invisible DuckDB
  view;
- `CREATE TABLE … CLONE` rejected `IF NOT EXISTS` / `OR REPLACE`;
  snapshots rejected `IF NOT EXISTS`, `OPTIONS(...)` and
  `FOR SYSTEM_TIME AS OF`; `DROP MATERIALIZED VIEW` ignored `IF EXISTS`;
- the transaction, `EXPORT DATA` and row-access-policy heads rejected the
  comment dbt prefixes to every statement; a grantee containing `)` broke
  the policy parse; `DROP ALL ROW ACCESS POLICIES` was unsupported;
- a trailing `; SELECT 2` was swallowed into a materialized view body;
- three regexes tripped CodeQL's ReDoS detector over time and were each
  replaced by a bespoke scanner (221 lines for `ALTER TABLE SET OPTIONS`).

## Decision

1. **Parse once, dispatch on node type.** The executor and the interpreter
   parse a single statement once (`sql/statement.py`) and hand the tree to
   every router. SQLGlot 30.x types `CREATE MATERIALIZED VIEW`, `CREATE
   TABLE … CLONE`, `DROP MATERIALIZED VIEW`, `EXPORT DATA`, `CREATE
   MODEL` and `ALTER TABLE SET OPTIONS`, so those heads dispatch on
   `exp.Create` / `exp.Drop` / `exp.Export` / `exp.Alter` with no grammar
   of their own. Multi-statement input never matches a head.

2. **One parsy grammar for what SQLGlot leaves untyped**
   (`sql/ddl_heads.py`). SQLGlot returns `CREATE / DROP SNAPSHOT TABLE`,
   `CREATE TABLE … CLONE … OPTIONS`, `CALL BQ.REFRESH_MATERIALIZED_VIEW`
   and the `ROW ACCESS POLICY` statements as an opaque `exp.Command`,
   raises on `DROP TABLE FUNCTION` and on the emulator's own `REFRESH
   MATERIALIZED VIEW` extension (ADR 0017), and types the transaction
   words only in some spellings (`START TRANSACTION` parses as a column
   alias). The grammar keeps one token layer (whitespace and both comment
   forms, case-insensitive keywords, identifiers, string literals,
   balanced brackets) and one combinator per head. Expression bodies — a
   `FILTER USING` predicate, a `FOR SYSTEM_TIME AS OF` expression — are
   returned as text for the SQL translator; the grammar never parses
   expressions.

3. **Statement-shaped errors.** A statement that starts like one of the
   heads but does not parse raises BigQuery's `Syntax error: … at
   [line:column]` instead of falling through to DuckDB's parser.

## Consequences

- The grammar's bracket matcher is an iterative leaf, not a recursive
  combinator, so nesting depth is bounded by the input rather than the
  interpreter's recursion limit (a recursive prototype crashed at ~75
  levels). It respects string literals, so a `)` inside a grantee or a
  predicate no longer ends the clause.
- parsy (MIT, pure Python, no dependencies) becomes a runtime dependency.
  A head parse costs a few hundred microseconds, a fraction of the
  SQLGlot parse that follows every statement anyway.
- The in-query rewriters (`INFORMATION_SCHEMA`, wildcard tables,
  `RANGE_SESSIONIZE`) substitute inside a query; they are SQLGlot
  transforms, not statement heads, and stay out of the grammar.
- `FOR SYSTEM_TIME AS OF` on snapshots and clones is now parsed and
  refused with an explicit `UnsupportedFeatureError` (the copy is always
  of the current state) instead of a DuckDB syntax error.
