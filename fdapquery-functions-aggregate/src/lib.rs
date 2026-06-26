//! # functions-aggregate
//!
//! Built-in aggregate-expression types. v0.1 re-exports the existing
//! types from `fdapquery-physical-expr`; Phase D introduces the
//! `AggregateUDF` shape (matching DataFusion's `AggregateUDF`) and
//! moves the real implementations here.

pub use fdapquery_physical_expr::{
    AggregateExpression, AggregateMode, AvgAccumulator, AvgExpression, CountAccumulator,
    CountExpression, MaxAccumulator, MaxExpression, MinAccumulator, MinExpression, SumAccumulator,
    SumExpression,
};
