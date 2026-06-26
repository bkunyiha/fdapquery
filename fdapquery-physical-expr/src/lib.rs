//! # physical-expr
//!
//! Physical expression trait, accumulator trait, and the concrete
//! expression / aggregate types. Mirrors DataFusion's
//! `datafusion-physical-expr` crate.
//!
//! ## What this crate provides
//!
//! - **PhysicalExpr trait** — [`PhysicalExpr`](expressions::PhysicalExpr):
//!   the trait every concrete physical expression implements. Each
//!   expression evaluates a `RecordBatch` into an output column.
//! - **Concrete expressions** — column references, literals, binary
//!   expressions (with numeric coercion), boolean comparisons and
//!   logical operators, arithmetic, casts, date arithmetic, and unary
//!   math.
//! - **Aggregation** — [`AggregateExpr`](aggregate_expression),
//!   `Min`/`Max`/`Sum`/`Count`/`Avg`, [`AggregateMode`](aggregate_mode)
//!   (`Partial` / `Final` / `Complete`), and the
//!   [`Accumulator`](expressions::Accumulator) trait used by all of
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
pub mod binary_expression;
pub mod boolean_expression;
pub mod cast_expression;
pub mod column_expression;
pub mod count_expression;
pub mod date_expression;
pub mod expressions;
pub mod math_expression;
pub mod max_expression;
pub mod min_expression;
pub mod sum_expression;
pub mod unary_math_expression;

// ==============================================================
// Per-item re-exports — matches DataFusion-physical-expr's
// top-level surface.
// ==============================================================
pub use aggregate_expression::AggregateExpr;
pub use aggregate_mode::AggregateMode;
pub use avg_expression::{AvgAccumulator, AvgExpr};
pub use binary_expression::BinaryExpr;
pub use boolean_expression::{
    AndExpr, BooleanExpr, EqExpr, GtEqExpr, GtExpr, LtEqExpr, LtExpr, NeqExpr, OrExpr,
};
pub use cast_expression::CastExpr;
pub use column_expression::Column;
pub use count_expression::{CountAccumulator, CountExpr};
pub use date_expression::{DateAddIntervalExpr, DateSubtractIntervalExpr};
pub use expressions::{
    Accumulator, AccumulatorValue, LiteralDate, LiteralDouble, LiteralIntervalDays, LiteralLong,
    LiteralString, PhysicalExpr,
};
pub use math_expression::{AddExpr, DivideExpr, MathExpr, MultiplyExpr, SubtractExpr};
pub use max_expression::{MaxAccumulator, MaxExpr};
pub use min_expression::{MinAccumulator, MinExpr};
pub use sum_expression::{SumAccumulator, SumExpr};
pub use unary_math_expression::{Log, Sqrt, UnaryMathExpr};
