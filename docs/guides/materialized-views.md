# Materialized views

bqemulator's materialized-view subsystem follows the model locked in by
[ADR 0017](../adr/0017-materialized-view-refresh.md): event-driven
staleness flagging plus lazy recompute on read.

## Lifecycle

Create, query, refresh, drop:

```sql
-- Materialise the view's rows into a regular DuckDB table.
CREATE MATERIALIZED VIEW sales.daily_totals AS
SELECT DATE(placed_at) AS day, SUM(amount) AS total
FROM sales.orders
GROUP BY day;

-- Read like any other table.
SELECT day, total FROM sales.daily_totals ORDER BY day;

-- Force a recompute even if not stale.
REFRESH MATERIALIZED VIEW sales.daily_totals;
-- BigQuery's documented builtin procedure form is also accepted.
CALL BQ.REFRESH_MATERIALIZED_VIEW('project.sales.daily_totals');

-- Drop the view + its rows.
DROP MATERIALIZED VIEW sales.daily_totals;
```

`tableType` is `MATERIALIZED_VIEW`. Direct DML is rejected:

```sql
-- Rejected: the MV refreshes from its base tables.
INSERT INTO sales.daily_totals VALUES (...);
```

## Refresh model

1. **Create**: the view query is parsed with SQLGlot, base-table
   references are extracted, and the rows are materialised once. The
   catalog records the dependency edges.
2. **Stale flag**: every `TableDataChanged` event published by a write
   path checks the dependency edges and flips `is_stale=True` on every
   dependent MV.
3. **Lazy refresh**: the next read against a stale MV runs
   `CREATE OR REPLACE TABLE … AS <query>` under the engine write lock,
   then resets the stale flag.
4. **Forced refresh**: `REFRESH MATERIALIZED VIEW` (or BigQuery's
   documented `CALL BQ.REFRESH_MATERIALIZED_VIEW('<fqdn>')` builtin
   procedure) runs the same path regardless of staleness.

Concurrent stale-readers collapse onto a single recompute — the second
reader rechecks `is_stale` after acquiring the write lock and skips the
redundant CTAS.

## INFORMATION_SCHEMA.MATERIALIZED_VIEWS

The MV catalog is exposed through the standard
`INFORMATION_SCHEMA.MATERIALIZED_VIEWS` virtual table. Project- and
dataset-qualified forms are both supported:

```sql
SELECT table_name, last_refresh_time, refresh_watermark, is_stale
FROM `my_project`.sales.INFORMATION_SCHEMA.MATERIALIZED_VIEWS;
```

Columns:

| Column | Notes |
|---|---|
| `table_catalog` | Project id of the MV. |
| `table_schema` | Dataset id of the MV. |
| `table_name` | Name of the MV. |
| `last_refresh_time` | Wall-clock time of the most recent refresh. |
| `refresh_watermark` | Equal to `last_refresh_time` (the emulator does not support incremental watermarking). |
| `enable_refresh` | Always `TRUE`. |
| `refresh_interval_minutes` | Reported as `30` for compatibility; not used. |
| `is_stale` | `TRUE` once a base-table change has fired and a refresh has not yet run. |

## Query rules

`CREATE MATERIALIZED VIEW` checks the body against BigQuery's
[query limitations](https://docs.cloud.google.com/bigquery/docs/materialized-views-create#query_limitations)
before materialising anything, so a view BigQuery would refuse fails
here with the same `invalidQuery` job error and leaves no table behind.

Every materialized view rejects non-deterministic functions
(`RAND`, `CURRENT_DATE`, `CURRENT_TIME`, `CURRENT_TIMESTAMP`,
`SESSION_USER`, `GENERATE_UUID`), user-defined functions, generative-AI
functions (`AI.*`, `ML.GENERATE_*`), `TABLESAMPLE`, `FOR SYSTEM_TIME AS
OF` and system variables, and may only read plain tables (no views,
other materialized views or snapshots).

An incremental view (the default) must also follow the
`WITH / SELECT / FROM / WHERE / GROUP BY` pattern: no set operations
(`UNION ALL` included — its incremental support is a BigQuery preview
the emulator does not model), only `INNER` / `CROSS` joins (`LEFT OUTER
JOIN` is the same preview), no self-joins, no window functions, no
`ARRAY` subqueries, no `HAVING` / `QUALIFY` / `ORDER BY` / `LIMIT`, and
only BigQuery's aggregate list (`ANY_VALUE`, `APPROX_COUNT_DISTINCT`,
`ARRAY_AGG`, `AVG`, `BIT_AND`, `BIT_OR`, `BIT_XOR`, `COUNT`, `COUNTIF`,
`HLL_COUNT.INIT`, `LOGICAL_AND`, `LOGICAL_OR`, `MAX`, `MIN`, `MAX_BY`,
`MIN_BY`, `SUM`), each a plain select-list output of the single
aggregating `SELECT`.

`OPTIONS(allow_non_incremental_definition = true, max_staleness = …)`
lifts the second set, as in BigQuery (`max_staleness` is mandatory with
it). `PARTITION BY` and `CLUSTER BY` are accepted and recorded on the
table (`timePartitioning` / `clustering`); their columns must be
outputs of the query. The refresh options (`enable_refresh`,
`refresh_interval_minutes`) are accepted and ignored.

Not checked: the type-dependent restrictions (`ANY_VALUE`, `ARRAY_AGG`,
`MAX_BY`, `MIN_BY` over `STRUCT`; `ARRAY_AGG` over `ARRAY`).

## Limitations

- Refresh is full recompute — no incremental refresh, no partition
  pruning. Documented in
  [`docs/reference/out-of-scope.md`](../reference/out-of-scope.md).
- Dropping a base table leaves dependent MVs in a "broken" state — the
  next refresh will fail; the stored rows continue to satisfy reads
  until then. Drop the dependent MV explicitly if the base table is
  going away.
