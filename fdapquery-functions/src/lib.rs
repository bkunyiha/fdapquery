//! # functions
//!
//! Built-in scalar-expression types. v0.1 re-exports the existing
//! types from `fdapquery-physical-expr`; Phase D introduces the
//! `ScalarUDF` shape (matching DataFusion's `ScalarUDF`) and moves
//! the real implementations here.

pub use fdapquery_physical_expr::{
    // Arithmetic.
    AddExpression,
    // Comparison + boolean.
    AndExpression,
    BooleanExpression,
    // Cast.
    CastExpression,
    // Column reference.
    ColumnExpression,
    // Date arithmetic.
    DateAddIntervalExpression,
    DateSubtractIntervalExpression,
    DivideExpression,
    EqExpression,
    GtEqExpression,
    GtExpression,
    // Literals.
    LiteralDateExpression,
    LiteralDoubleExpression,
    LiteralIntervalDaysExpression,
    LiteralLongExpression,
    LiteralStringExpression,
    // Unary math.
    Log,
    LtEqExpression,
    LtExpression,
    MathExpression,
    MultiplyExpression,
    NeqExpression,
    OrExpression,
    Sqrt,
    SubtractExpression,
    UnaryMathExpression,
};
