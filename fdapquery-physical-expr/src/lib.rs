//! # physical-expr
//!
//! Physical expression trait, accumulator trait, and the concrete
//! expression / aggregate types. Mirrors DataFusion's
//! `datafusion-physical-expr` crate.
//!
//! ## What this crate provides
//!
//! - **PhysicalExpr trait** — [`PhysicalExpr`]:
//!   the trait every concrete physical expression implements. Each
//!   expression evaluates a `RecordBatch` into an output column.
//! - **Concrete expressions** — column references, literals, binary
//!   expressions (with numeric coercion), boolean comparisons and
//!   logical operators, arithmetic, casts, date arithmetic, and unary
//!   math.
//! - **Aggregation** — [`AggregateExpr`],
//!   `Min`/`Max`/`Sum`/`Count`/`Avg`, [`AggregateMode`]
//!   (`Partial` / `Final` / `Complete`), and the
//!   [`Accumulator`] trait used by all of
//!   them.
//!
//! ## Relationship to `fdapquery-physical-plan`
//!
//! Operators in `fdapquery-physical-plan` (`AggregateExec`,
//! `ProjectionExec`, `FilterExec`, …) consume these expression
//! types via path-dependency. `fdapquery-physical-plan/src/lib.rs`
//! re-exports the items here for the `use fdapquery_physical_plan::*`
//! ergonomics callers may want.

// ==============================================================
// Per-file modules.
// ==============================================================
pub mod aggregate_expression;
pub mod aggregate_mode;
pub mod avg_expression;
pub mod columnar_value;
// A single `BinaryExpr { left, op, right }` here,
// strict-mirroring DataFusion's `BinaryExpr`. The math and boolean
// kernels live inline in `binary_expression.rs`.
pub mod binary_expression;
pub mod cast_expression;
pub mod column_expression;
pub mod count_expression;
pub mod date_expression;
pub mod expressions;
pub mod max_expression;
pub mod min_expression;
// `PhysicalSortExpr` + `LexOrdering` mirror DataFusion's
// `datafusion::physical_expr_common::sort_expr` (consumed by `SortExec`).
pub mod sort_expr;
pub mod sum_expression;
pub mod unary_math_expression;
// `utils::scatter` is used by the default `PhysicalExpr::evaluate_selection`
// impl. Strict mirror (fallback arm only) of DataFusion's
// `datafusion_physical_expr_common::utils::scatter`.
pub mod utils;

// ==============================================================
// Per-item re-exports — matches DataFusion-physical-expr's
// top-level surface.
// ==============================================================
pub use aggregate_expression::AggregateExpr;
pub use aggregate_mode::AggregateMode;
pub use avg_expression::{AvgAccumulator, AvgExpr};
// The unified `BinaryExpr { left, op, right }` struct
// strict-mirrors `datafusion_physical_expr::expressions::BinaryExpr`.
// Call sites build `BinaryExpr::new(left, op, right)` parameterised by
// [`fdapquery_expr::Operator`].
pub use binary_expression::BinaryExpr;
pub use cast_expression::CastExpr;
pub use column_expression::Column;
// `ColumnarValue` is the result type of
// `PhysicalExpr::evaluate`. Strict mirror of DataFusion's
// `datafusion::physical_plan::ColumnarValue` (re-exported there from
// `datafusion_expr_common::columnar_value::ColumnarValue`).
pub use columnar_value::ColumnarValue;
pub use count_expression::{CountAccumulator, CountExpr};
pub use date_expression::{DateAddIntervalExpr, DateSubtractIntervalExpr};
// The five sibling literal types (`LiteralLong`,
// `LiteralDouble`, `LiteralString`, `LiteralDate`, `LiteralIntervalDays`)
// were collapsed into a single `Literal { value: ScalarValue }` plus a
// `lit()` factory, mirroring DataFusion's
// `datafusion_physical_expr::expressions::Literal` exactly.
pub use expressions::{Accumulator, AccumulatorValue, Literal, PhysicalExpr, lit};
pub use max_expression::{MaxAccumulator, MaxExpr};
pub use min_expression::{MinAccumulator, MinExpr};
// Re-export the sort-expression types at the crate root
// so consumers can write `use fdapquery_physical_expr::PhysicalSortExpr;`
// (matches DataFusion's `use datafusion_physical_expr::PhysicalSortExpr;`).
pub use sort_expr::{LexOrdering, PhysicalSortExpr};
pub use sum_expression::{SumAccumulator, SumExpr};
pub use unary_math_expression::{Log, Sqrt, UnaryMathExpr};
