//! # physical-plan
//!
//! Physical execution plans, operators, and expression evaluation. The largest
//! module in the workspace.
//!
//! ## What this crate provides
//!
//! - **Plan trait** — [`PhysicalPlan`](physical_plan::PhysicalPlan): the trait
//!   every operator implements. Has `schema()`, `children()`,
//!   `with_new_children(...)`, `execute(&ctx)`, and `as_any()` for runtime
//!   downcasting.
//! - **Operators** — [`ScanExec`](scan_exec::ScanExec),
//!   [`ProjectionExec`](projection_exec::ProjectionExec),
//!   [`FilterExec`](filter_exec::FilterExec),
//!   [`GlobalLimitExec`](global_limit_exec::GlobalLimitExec),
//!   [`LocalLimitExec`](local_limit_exec::LocalLimitExec),
//!   [`AggregateExec`](aggregate_exec::AggregateExec),
//!   [`HashJoinExec`](hash_join_exec::HashJoinExec),
//!   [`ShuffleReaderExec`](shuffle_reader_exec::ShuffleReaderExec),
//!   [`ShuffleWriterExec`](shuffle_writer_exec::ShuffleWriterExec).
//! - **Expressions** — the [`PhysicalExpr`](physical_plan::PhysicalPlan) family
//!   covers column references, literals, binary expressions (with numeric
//!   coercion), boolean comparisons and logical operators, arithmetic, casts,
//!   date arithmetic, and unary math. Each expression evaluates a `RecordBatch`
//!   into an output column.
//! - **Aggregation** — [`AggregateExpr`](aggregate_expression),
//!   `Min`/`Max`/`Sum`/`Count`/`Avg`, and [`AggregateMode`](aggregate_mode)
//!   (`Partial` / `Final` / `Complete`).
//! - **Shuffle and task** — [`Task`](task), and re-exports of
//!   [`ShuffleLocation`](fdapquery_execution::ShuffleLocation) and
//!   [`ShuffleManager`](fdapquery_execution::ShuffleManager) (which live in
//!   `fdapquery-execution` since Session 15c, Arrow IPC writer/reader for
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
pub mod hash_join_exec;
// Session 15d-1 #105 — `LimitExec` split into `GlobalLimitExec`
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
// `fdapquery-query-planner` crate in Session 12; renamed from
// `query_planner` in Session 15d-1 #109 to match DataFusion's
// `physical_planner.rs`) translates `LogicalPlan` to
// `Arc<dyn ExecutionPlan>`. Exposes the `PhysicalPlanner` trait + the
// `DefaultPhysicalPlanner` impl, same shape as DataFusion's.
pub mod filter_exec;
pub mod physical_planner;
pub mod scan_exec;
pub mod shuffle_reader_exec;
pub mod shuffle_writer_exec;
// Session 15d-1 #92 moved `stream` to `fdapquery-execution`. The
// re-export below makes `crate::stream::{...}` paths inside this
// crate's own modules continue to resolve, and external consumers can
// keep writing `fdapquery_physical_plan::stream::{...}` too. The
// individual type re-exports near the bottom of this file keep
// `use fdapquery_physical_plan::SendableRecordBatchStream;` working.
pub use fdapquery_execution::stream;
pub mod task;
// Session 15c moved `task_context`, `shuffle_manager`, and `shuffle_location`
// into `fdapquery-execution`. Re-exports near the bottom of this file keep
// external consumers' `use fdapquery_physical_plan::{TaskContext,
// ShuffleManager, ShuffleLocation, …};` working unchanged.

// Internal helper: a float-aware hashable row key used by `HashJoinExec` for
// its join keys (and the same shape `AggregateExec` uses for group keys).
// See `row_key.rs`.
mod row_key;

// ==============================================================
// Re-exports for convenient downstream `use physical_plan::*;` ergonomics.
// Only the phase-1 types are exported so far; later phases add to this list.
// ==============================================================
// Re-export from fdapquery-physical-expr — the expression types were
// split out in Session 13a but external consumers can still write
// `use fdapquery_physical_plan::PhysicalExpr;` via these re-exports
// (matches DataFusion's pattern of re-exporting from
// datafusion-physical-expr at datafusion-physical-plan's surface).
pub use fdapquery_physical_expr::BinaryExpr;
pub use fdapquery_physical_expr::CastExpr;
pub use fdapquery_physical_expr::Column;
pub use fdapquery_physical_expr::{
    Accumulator, AccumulatorValue, LiteralDate, LiteralDouble, LiteralIntervalDays, LiteralLong,
    LiteralString, PhysicalExpr,
};
pub use fdapquery_physical_expr::{AddExpr, DivideExpr, MathExpr, MultiplyExpr, SubtractExpr};
pub use fdapquery_physical_expr::{
    AndExpr, BooleanExpr, EqExpr, GtEqExpr, GtExpr, LtEqExpr, LtExpr, NeqExpr, OrExpr,
};
pub use physical_plan::{ExecutionPlan, format};
// The folded `DefaultPhysicalPlanner` + `PhysicalPlanner` trait — same shape
// as DataFusion's. Trait at `datafusion-core/src/physical_planner.rs:128`,
// `DefaultPhysicalPlanner` struct at line 266. Consumers that want
// trait-object dispatch use `Arc<dyn PhysicalPlanner>`; consumers that hold
// the concrete type call `DefaultPhysicalPlanner::new()` directly.
pub use physical_planner::{DefaultPhysicalPlanner, PhysicalPlanner};
// Phase B foundation: per-task context, partitioning descriptor, and the
// async record-batch stream surface. These live in `fdapquery-physical-plan`
// during Phase B because the `ExecutionPlan` trait that references them is
// in this crate; Phase C migrates them to `fdapquery-execution`.
pub use partitioning::Partitioning;
pub use plan_properties::PlanProperties;
// Session 15d-1 #92 — stream types live in fdapquery-execution now.
pub use fdapquery_execution::{
    RecordBatchStream, RecordBatchStreamAdapter, SendableRecordBatchStream,
};
// Session 15c — the runtime types moved to `fdapquery-execution`. Re-export
// them here so existing call sites that wrote
// `use fdapquery_physical_plan::{TaskContext, RuntimeEnv, SessionConfig};`
// keep compiling.
pub use fdapquery_execution::{RuntimeEnv, SessionConfig, TaskContext};
// Phase-2 operators.
pub use filter_exec::FilterExec;
pub use global_limit_exec::GlobalLimitExec;
pub use local_limit_exec::LocalLimitExec;
pub use projection_exec::ProjectionExec;
pub use scan_exec::ScanExec;
// Phase-3 scalar expressions (re-exported from fdapquery-physical-expr).
pub use fdapquery_physical_expr::{DateAddIntervalExpr, DateSubtractIntervalExpr};
pub use fdapquery_physical_expr::{Log, Sqrt, UnaryMathExpr};
// Phase-3 aggregation (expressions re-exported from physical-expr; AggregateExec
// is the operator that consumes them and stays here).
pub use aggregate_exec::AggregateExec;
pub use fdapquery_physical_expr::AggregateExpr;
pub use fdapquery_physical_expr::AggregateMode;
pub use fdapquery_physical_expr::{AvgAccumulator, AvgExpr};
pub use fdapquery_physical_expr::{CountAccumulator, CountExpr};
pub use fdapquery_physical_expr::{MaxAccumulator, MaxExpr};
pub use fdapquery_physical_expr::{MinAccumulator, MinExpr};
pub use fdapquery_physical_expr::{SumAccumulator, SumExpr};
// Phase-4 join + shuffle/task scaffolding.
pub use action::{Action, QueryAction, ShuffleIdAction};
pub use hash_join_exec::HashJoinExec;
// Session 15c — `ShuffleLocation` / `ShuffleManager` moved to
// `fdapquery-execution` alongside `TaskContext`. Re-export from there so
// existing `use fdapquery_physical_plan::ShuffleManager;` keeps compiling.
pub use fdapquery_execution::{ShuffleLocation, ShuffleManager};
pub use shuffle_reader_exec::ShuffleReaderExec;
pub use shuffle_writer_exec::ShuffleWriterExec;
pub use task::Task;
