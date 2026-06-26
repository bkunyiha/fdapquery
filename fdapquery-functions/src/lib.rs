//! # functions
//!
//! Built-in scalar-expression types. v0.1 re-exports the existing
//! types from `fdapquery-physical-expr`; Phase D introduces the
//! `ScalarUDF` shape (matching DataFusion's `ScalarUDF`) and moves
//! the real implementations here.

pub use fdapquery_physical_expr::{
    // Arithmetic.
    AddExpr,
    // Comparison + boolean.
    AndExpr,
    BooleanExpr,
    // Cast.
    CastExpr,
    // Column reference.
    Column,
    // Date arithmetic.
    DateAddIntervalExpr,
    DateSubtractIntervalExpr,
    DivideExpr,
    EqExpr,
    GtEqExpr,
    GtExpr,
    // Literals.
    LiteralDate,
    LiteralDouble,
    LiteralIntervalDays,
    LiteralLong,
    LiteralString,
    // Unary math.
    Log,
    LtEqExpr,
    LtExpr,
    MathExpr,
    MultiplyExpr,
    NeqExpr,
    OrExpr,
    Sqrt,
    SubtractExpr,
    UnaryMathExpr,
};
