//! # physical-plan
//!
//! Physical execution plans, operators, and expression evaluation. The largest
//! module in the workspace.
//!
//! ## What this crate provides
//!
//! - **Plan trait** — [`ExecutionPlan`]: the trait
//!   every operator implements. Has `schema()`, `children()`,
//!   `with_new_children(...)`, `execute(&ctx)`, and `as_any()` for runtime
//!   downcasting.
//! - **Operators** —
//!   [`ProjectionExec`],
//!   [`FilterExec`],
//!   [`GlobalLimitExec`],
//!   [`LocalLimitExec`],
//!   [`AggregateExec`],
//!   [`HashJoinExec`],
//!   [`ShuffleReaderExec`],
//!   [`ShuffleWriterExec`].
//! - **Expressions** — the `PhysicalExpr` family
//!   covers column references, literals, binary expressions (with numeric
//!   coercion), boolean comparisons and logical operators, arithmetic, casts,
//!   date arithmetic, and unary math. Each expression evaluates a `RecordBatch`
//!   into an output column.
//! - **Aggregation** — `AggregateExpr`,
//!   `Min`/`Max`/`Sum`/`Count`/`Avg`, and [`AggregateMode`]
//!   (`Partial` / `Final` / `Complete`).
//! - **Shuffle and task** — [`Task`], and re-exports of
//!   [`ShuffleLocation`] and
//!   [`ShuffleManager`] (which live in
//!   `fdapquery-execution`. Arrow IPC writer/reader for
//!   shuffle files).
//!
//! ## Design — traits, not enums
//!
//! `PhysicalPlan` and `PhysicalExpr` are Rust **traits** referenced through
//! `Arc<dyn PhysicalPlan>` / `Arc<dyn PhysicalExpr>`, *not* enums. This reverses
//! the "closed interface → enum" rule applied to `logical_plan`, because the
//! physical operator/expression set is large and open in spirit (adding a new
//! operator is adding a new file, not editing a central enum).
//! `as_any().downcast_ref::<X>()` is the standard pattern for recovering a
//! concrete type.

// ==============================================================
// Per-file modules.
// ==============================================================
pub mod action;
pub mod aggregate_exec;
// `aggregates` hosts `PhysicalGroupBy`, the strict mirror of
// `datafusion::physical_plan::aggregates::PhysicalGroupBy` consumed by
// `AggregateExec`. Same layout as DataFusion (both live under the
// `aggregates` submodule of the physical-plan crate).
pub mod aggregates;
// `display` hosts the `displayable(plan).indent(verbose)`
// builder that replaces the dropped `pretty` / `format` free functions.
// Same layout as DataFusion's `datafusion::physical_plan::display`.
pub mod display;
// Operator-level metrics. Mirrors
// `datafusion::physical_plan::metrics`. The display module references
// `MetricsSet` / `MetricType` / `MetricCategory` through here.
pub mod metrics;
// Strict mirror of
// `datafusion::physical_plan::render_tree`. Provides the `RenderTree`
// 2D grid + `RenderTreeNode` precomputation consumed by
// `display::TreeRenderVisitor`. Lives as its own module to match
// DataFusion's file layout.
pub mod hash_join_exec;
pub mod render_tree;
// `LimitExec` split into `GlobalLimitExec`
// (skip+fetch over merged result; emitted by the planner) and
// `LocalLimitExec` (per-partition early termination; scaffolded for
// upcoming repartition work). Matches DataFusion's split.
pub mod global_limit_exec;
pub mod local_limit_exec;
pub mod partitioning;
pub mod physical_plan;
pub mod plan_properties;
pub mod projection_exec;
// `physical_planner` (folded in from the dissolved
// `fdapquery-query-planner` crate; renamed from
// `query_planner` to match DataFusion's
// `physical_planner.rs`) exposes the `PhysicalPlanner` trait that
// translates `LogicalPlan` to `Arc<dyn ExecutionPlan>`. The concrete
// `DefaultPhysicalPlanner` impl lives in the umbrella `fdapquery`
// crate (matches DataFusion's split: trait in `datafusion-physical-plan`,
// default impl in `datafusion-core`).
pub mod filter_exec;
// `MemoryExec`: leaf plan over pre-loaded batches. Replaces the
// private `InMemoryPlan` that used to live in `fdapquery`'s
// `parallel_context`. Same shape as DataFusion's (historic)
// `datafusion::physical_plan::memory::MemoryExec`, backed by
// `fdapquery_execution::MemoryStream`.
pub mod memory_exec;
pub mod physical_planner;
// `scan_exec` is gone. The catalog
// providers build `DataSourceExec` (in `fdapquery-datasource`) directly
// from inside `TableProvider::scan`. The file is left in place as a
// tombstone for source-control archaeology; the module is no longer
// declared here. Strict mirror of DataFusion's
// `datafusion-physical-plan` (no scan operator).
pub mod shuffle_reader_exec;
pub mod shuffle_writer_exec;
// `SortExec` lives under a `sorts/` submodule to match
// DataFusion's file layout (`datafusion::physical_plan::sorts::sort`).
// Only the in-memory sort path is ported in v0.1; spilling /
// sort-preserving-merge / partial-sort are deferred (see sorts/mod.rs).
pub mod sorts;
// Moved `stream` to `fdapquery-execution`. The
// re-export below makes `crate::stream::{...}` paths inside this
// crate's own modules continue to resolve, and external consumers can
// keep writing `fdapquery_physical_plan::stream::{...}` too. The
// individual type re-exports near the bottom of this file keep
// `use fdapquery_physical_plan::SendableRecordBatchStream;` working.
pub use fdapquery_execution::stream;
pub mod task;
// Moved `task_context`, `shuffle_manager`, and `shuffle_location`
// into `fdapquery-execution`. Re-exports near the bottom of this file keep
// external consumers' `use fdapquery_physical_plan::{TaskContext,
// ShuffleManager, ShuffleLocation, …};` working unchanged.

// Internal helper: a float-aware hashable row key used by `HashJoinExec` for
// its join keys (and the same shape `AggregateExec` uses for group keys).
// See `row_key.rs`.
mod row_key;

// Test-only `TestSourceExec`
// leaf operator + employee-schema/batches fixture. Previously the
// operator tests in this crate drove `ScanExec` over the catalog's
// `CsvDataSource`; with the cycle broken the catalog crate is no
// longer reachable from here, and `TestSourceExec` replaces the
// fixture for in-crate operator tests.
#[cfg(test)]
mod test_util;

// ==============================================================
// Re-exports for convenient downstream `use physical_plan::*;` ergonomics.
// Only the phase-1 types are exported so far; later phases add to this list.
// ==============================================================
// Re-export from fdapquery-physical-expr — the expression types were
// split out but external consumers can still write
// `use fdapquery_physical_plan::PhysicalExpr;` via these re-exports
// (matches DataFusion's pattern of re-exporting from
// datafusion-physical-expr at datafusion-physical-plan's surface).
pub use fdapquery_physical_expr::BinaryExpr;
pub use fdapquery_physical_expr::CastExpr;
pub use fdapquery_physical_expr::Column;
// `ColumnarValue` is the result type of
// `PhysicalExpr::evaluate`. Mirrors
// `datafusion::physical_plan::ColumnarValue`.
pub use fdapquery_physical_expr::ColumnarValue;
// The five sibling literal types were collapsed into a
// single `Literal { value: ScalarValue }` plus a `lit()` factory.
pub use fdapquery_physical_expr::{Accumulator, AccumulatorValue, Literal, PhysicalExpr, lit};
// The 12 sibling binary re-exports collapsed into the
// unified [`BinaryExpr`] re-export already declared above. Construct via
// `BinaryExpr::new(left, fdapquery_expr::Operator::Eq, right)` etc.
// The rquery-DNA `pub fn format(plan: &dyn
// ExecutionPlan)` free function (and its follow-on rename to `pretty`,
// plus the parallel `ExecutionPlan::pretty(&self) -> String` trait
// method) have been removed. DataFusion has no equivalent of either —
// plan dumping goes through the `displayable(plan).indent(verbose)`
// builder. fdapquery mirrors that: see [`crate::display::displayable`]
// and [`crate::display::DisplayableExecutionPlan`].
// Strict mirror of `datafusion::physical_plan::display`.
// Every public symbol DataFusion exposes from `display` is re-exported
// here at the same crate root path, so external consumers can write
// `use fdapquery_physical_plan::{displayable, DisplayAs, DisplayFormatType, …};`
// exactly like `use datafusion_physical_plan::{displayable, DisplayAs, DisplayFormatType, …};`.
pub use display::{
    DefaultDisplay, DisplayAs, DisplayFormatType, DisplayableExecutionPlan, PlanType,
    ProjectSchemaDisplay, StringifiedPlan, VerboseDisplay, display_orderings, displayable,
};
// Metrics types referenced by the display module's wrapper builder.
// `MetricValue` is now an **enum** mirroring
// DataFusion's full variant set (`OutputRows`, `ElapsedCompute`,
// `SpillCount`, `SpilledBytes`, `OutputBytes`, `OutputBatches`,
// `SpilledRows`, `CurrentMemoryUsage`, `Count`, `Gauge`,
// `PeakMemoryUsage`, `Time`, `StartTimestamp`, `EndTimestamp`,
// `PruningMetrics`, `Ratio`).  `Metric` wraps a value with partition,
// metric type, category, and labels — same shape DataFusion's
// `MetricsSet` returns from `MetricsSet::iter()`.
pub use metrics::{
    BaselineMetrics, Count, CustomMetricValue, ExecutionPlanMetricsSet, Gauge, Label, LabelValue,
    Metric, MetricBuilder, MetricCategory, MetricType, MetricValue, MetricsSet, PruningMetrics,
    RatioMergeStrategy, RatioMetrics, RecordOutput, ScopedTimerGuard, SpillMetrics, SplitMetrics,
    Time, Timestamp,
};
// `ExecutionPlanVisitor` + `accept` are part of the display infrastructure
// — the indent / graphviz / one_line walkers all dispatch through them.
pub use physical_plan::{ExecutionPlan, ExecutionPlanVisitor, accept};
// The `PhysicalPlanner` trait — same shape as DataFusion's
// `datafusion::physical_planner::PhysicalPlanner`. The concrete
// `DefaultPhysicalPlanner` impl lives in the umbrella `fdapquery` crate
// (matching DataFusion's split: trait in physical-plan, default impl
// in the umbrella) so the concrete planner can freely compose
// `fdapquery-catalog` + `fdapquery-physical-plan` + `fdapquery-datasource`
// without forcing those deps on every consumer of this crate.
// Consumers that want trait-object dispatch use
// `Arc<dyn fdapquery_physical_plan::PhysicalPlanner>`; consumers that hold
// the concrete type call `fdapquery::DefaultPhysicalPlanner::new()` directly.
pub use physical_planner::PhysicalPlanner;
// Foundation: per-task context, partitioning descriptor, and the
// async record-batch stream surface. These live in `fdapquery-physical-plan`
// because the `ExecutionPlan` trait that references them is
// in this crate; Future work migrates them to `fdapquery-execution`.
pub use partitioning::Partitioning;
pub use plan_properties::PlanProperties;
// Stream types live in fdapquery-execution now.
pub use fdapquery_execution::{
    EmptyRecordBatchStream, MemoryStream, RecordBatchStream, RecordBatchStreamAdapter,
    SendableRecordBatchStream,
};
// The runtime types moved to `fdapquery-execution`. Re-export
// them here so existing call sites that wrote
// `use fdapquery_physical_plan::{TaskContext, RuntimeEnv, SessionConfig};`
// keep compiling.
pub use fdapquery_execution::{RuntimeEnv, SessionConfig, TaskContext};
// Phase-2 operators.
pub use filter_exec::FilterExec;
pub use global_limit_exec::GlobalLimitExec;
pub use local_limit_exec::LocalLimitExec;
pub use memory_exec::MemoryExec;
pub use projection_exec::ProjectionExec;
// `ScanExec` re-export removed
// alongside the deletion of the operator. External consumers should
// use `fdapquery_datasource::DataSourceExec` (built by
// `TableProvider::scan` inside `fdapquery-catalog`).
// Phase-3 scalar expressions (re-exported from fdapquery-physical-expr).
pub use fdapquery_physical_expr::{DateAddIntervalExpr, DateSubtractIntervalExpr};
pub use fdapquery_physical_expr::{Log, Sqrt, UnaryMathExpr};
// Phase-3 aggregation (expressions re-exported from physical-expr; AggregateExec
// is the operator that consumes them and stays here).
pub use aggregate_exec::AggregateExec;
pub use aggregates::PhysicalGroupBy;
pub use fdapquery_physical_expr::AggregateExpr;
pub use fdapquery_physical_expr::AggregateMode;
pub use fdapquery_physical_expr::{AvgAccumulator, AvgExpr};
pub use fdapquery_physical_expr::{CountAccumulator, CountExpr};
pub use fdapquery_physical_expr::{MaxAccumulator, MaxExpr};
pub use fdapquery_physical_expr::{MinAccumulator, MinExpr};
pub use fdapquery_physical_expr::{SumAccumulator, SumExpr};
// Phase-4 join + shuffle/task scaffolding.
pub use action::{Action, QueryAction, ShuffleIdAction};
pub use hash_join_exec::{ColumnIndex, HashJoinExec, JoinFilter, JoinOn, JoinOnRef, PartitionMode};
// `ShuffleLocation` / `ShuffleManager` moved to
// `fdapquery-execution` alongside `TaskContext`. Re-export from there so
// existing `use fdapquery_physical_plan::ShuffleManager;` keeps compiling.
pub use fdapquery_execution::{ShuffleLocation, ShuffleManager};
pub use shuffle_reader_exec::ShuffleReaderExec;
pub use shuffle_writer_exec::ShuffleWriterExec;
// `SortExec` and the sort-expression types it consumes.
// Same crate-root location as DataFusion's
// `datafusion_physical_plan::sorts::sort::SortExec` (and the
// `PhysicalSortExpr` / `LexOrdering` re-exported from physical-expr).
pub use fdapquery_physical_expr::{LexOrdering, PhysicalSortExpr};
pub use sorts::sort::SortExec;
pub use task::Task;
