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
//!   [`SelectionExec`](selection_exec::SelectionExec),
//!   [`LimitExec`](limit_exec::LimitExec),
//!   [`HashAggregateExec`](hash_aggregate_exec::HashAggregateExec),
//!   [`HashJoinExec`](hash_join_exec::HashJoinExec),
//!   [`ShuffleReaderExec`](shuffle_reader_exec::ShuffleReaderExec),
//!   [`ShuffleWriterExec`](shuffle_writer_exec::ShuffleWriterExec).
//! - **Expressions** — the [`Expression`](physical_plan::PhysicalPlan) family
//!   covers column references, literals, binary expressions (with numeric
//!   coercion), boolean comparisons and logical operators, arithmetic, casts,
//!   date arithmetic, and unary math. Each expression evaluates a `RecordBatch`
//!   into an output column.
//! - **Aggregation** — [`AggregateExpression`](aggregate_expression),
//!   `Min`/`Max`/`Sum`/`Count`/`Avg`, and [`AggregateMode`](aggregate_mode)
//!   (`Partial` / `Final` / `Complete`).
//! - **Shuffle and task** — [`Task`](task), [`ShuffleLocation`](shuffle_location),
//!   [`ShuffleManager`](shuffle_manager) (Arrow IPC writer/reader for shuffle
//!   files), and [`ExecutorContext`](executor_context) (per-executor identity
//!   + shuffle storage handle that operators receive via `execute(&ctx)`).
//!
//! ## Design — traits, not enums
//!
//! `PhysicalPlan` and `Expression` are Rust **traits** referenced through
//! `Arc<dyn PhysicalPlan>` / `Arc<dyn Expression>`, *not* enums. This reverses
//! the "closed interface → enum" rule applied to `logical_plan`, because the
//! physical operator/expression set is large and open in spirit (adding a new
//! operator is adding a new file, not editing a central enum).
//! `as_any().downcast_ref::<X>()` is the standard pattern for recovering a
//! concrete type.

// ==============================================================
// Per-file modules.
// ==============================================================
pub mod action;
pub mod aggregate_expression;
pub mod aggregate_mode;
pub mod avg_expression;
pub mod binary_expression;
pub mod boolean_expression;
pub mod cast_expression;
pub mod column_expression;
pub mod count_expression;
pub mod date_expression;
// `executor_context.rs` bundles the per-executor identity
// (executor_id / host / port) with the shuffle storage handle (Arc<ShuffleManager>)
// into a single value that operators receive through `execute(&ctx)`. Placement
// here in `physical-plan/` is forced by the dependency graph (every other
// candidate crate transitively depends on this one).
pub mod executor_context;
pub mod expressions;
pub mod hash_aggregate_exec;
pub mod hash_join_exec;
pub mod limit_exec;
pub mod math_expression;
pub mod max_expression;
pub mod min_expression;
pub mod physical_plan;
pub mod projection_exec;
pub mod scan_exec;
pub mod selection_exec;
pub mod shuffle_location;
pub mod shuffle_manager;
pub mod shuffle_reader_exec;
pub mod shuffle_writer_exec;
pub mod sum_expression;
pub mod task;
pub mod unary_math_expression;

// Internal helper: a float-aware hashable row key used by `HashJoinExec` for
// its join keys (and the same shape `HashAggregateExec` uses for group keys).
// See `row_key.rs`.
mod row_key;

// ==============================================================
// Re-exports for convenient downstream `use physical_plan::*;` ergonomics.
// Only the phase-1 types are exported so far; later phases add to this list.
// ==============================================================
pub use binary_expression::BinaryExpression;
pub use boolean_expression::{
    AndExpression, BooleanExpression, EqExpression, GtEqExpression, GtExpression, LtEqExpression,
    LtExpression, NeqExpression, OrExpression,
};
pub use cast_expression::CastExpression;
pub use column_expression::ColumnExpression;
pub use expressions::{
    Accumulator, AccumulatorValue, Expression, LiteralDateExpression, LiteralDoubleExpression,
    LiteralIntervalDaysExpression, LiteralLongExpression, LiteralStringExpression,
};
pub use math_expression::{
    AddExpression, DivideExpression, MathExpression, MultiplyExpression, SubtractExpression,
};
pub use physical_plan::{PhysicalPlan, format};
// Phase-2 operators.
pub use limit_exec::LimitExec;
pub use projection_exec::ProjectionExec;
pub use scan_exec::ScanExec;
pub use selection_exec::SelectionExec;
// Phase-3 scalar expressions.
pub use date_expression::{DateAddIntervalExpression, DateSubtractIntervalExpression};
pub use unary_math_expression::{Log, Sqrt, UnaryMathExpression};
// Phase-3 aggregation.
pub use aggregate_expression::AggregateExpression;
pub use aggregate_mode::AggregateMode;
pub use avg_expression::{AvgAccumulator, AvgExpression};
pub use count_expression::{CountAccumulator, CountExpression};
pub use hash_aggregate_exec::HashAggregateExec;
pub use max_expression::{MaxAccumulator, MaxExpression};
pub use min_expression::{MinAccumulator, MinExpression};
pub use sum_expression::{SumAccumulator, SumExpression};
// Phase-4 join + shuffle/task scaffolding.
pub use action::{Action, QueryAction, ShuffleIdAction};
pub use hash_join_exec::HashJoinExec;
pub use shuffle_location::ShuffleLocation;
pub use shuffle_manager::ShuffleManager;
pub use shuffle_reader_exec::ShuffleReaderExec;
pub use shuffle_writer_exec::ShuffleWriterExec;
pub use task::Task;
// Executor context — consumed by flight-server and the shuffle operator
// bodies.
pub use executor_context::ExecutorContext;
