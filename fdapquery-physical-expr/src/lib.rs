//! # physical-expr
//!
//! Physical expression trait, accumulator trait, and the concrete
//! expression / aggregate types. Mirrors DataFusion's
//! `datafusion-physical-expr` crate.
//!
//! ## What this crate provides
//!
//! - **Expression trait** — [`Expression`](expressions::Expression):
//!   the trait every concrete physical expression implements. Each
//!   expression evaluates a `RecordBatch` into an output column.
//! - **Concrete expressions** — column references, literals, binary
//!   expressions (with numeric coercion), boolean comparisons and
//!   logical operators, arithmetic, casts, date arithmetic, and unary
//!   math.
//! - **Aggregation** — [`AggregateExpression`](aggregate_expression),
//!   `Min`/`Max`/`Sum`/`Count`/`Avg`, [`AggregateMode`](aggregate_mode)
//!   (`Partial` / `Final` / `Complete`), and the
//!   [`Accumulator`](expressions::Accumulator) trait used by all of
//!   them.
//!
//! ## Relationship to `fdapquery-physical-plan`
//!
//! Operators in `fdapquery-physical-plan` (`HashAggregateExec`,
//! `ProjectionExec`, `SelectionExec`, …) consume these expression
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
pub use aggregate_expression::AggregateExpression;
pub use aggregate_mode::AggregateMode;
pub use avg_expression::{AvgAccumulator, AvgExpression};
pub use binary_expression::BinaryExpression;
pub use boolean_expression::{
    AndExpression, BooleanExpression, EqExpression, GtEqExpression, GtExpression, LtEqExpression,
    LtExpression, NeqExpression, OrExpression,
};
pub use cast_expression::CastExpression;
pub use column_expression::ColumnExpression;
pub use count_expression::{CountAccumulator, CountExpression};
pub use date_expression::{DateAddIntervalExpression, DateSubtractIntervalExpression};
pub use expressions::{
    Accumulator, AccumulatorValue, Expression, LiteralDateExpression, LiteralDoubleExpression,
    LiteralIntervalDaysExpression, LiteralLongExpression, LiteralStringExpression,
};
pub use math_expression::{
    AddExpression, DivideExpression, MathExpression, MultiplyExpression, SubtractExpression,
};
pub use max_expression::{MaxAccumulator, MaxExpression};
pub use min_expression::{MinAccumulator, MinExpression};
pub use sum_expression::{SumAccumulator, SumExpression};
pub use unary_math_expression::{Log, Sqrt, UnaryMathExpression};
