//! # functions
//!
//! Built-in scalar-expression types. v0.1 re-exports the existing
//! types from `fdapquery-physical-expr`; Future work introduces the
//! `ScalarUDF` shape (matching DataFusion's `ScalarUDF`) and moves
//! the real implementations here.

// The 12 sibling binary types (`AddExpr`,
// `SubtractExpr`, `MultiplyExpr`, `DivideExpr`, `AndExpr`, `OrExpr`,
// `EqExpr`, `NeqExpr`, `LtExpr`, `LtEqExpr`, `GtExpr`, `GtEqExpr`)
// collapsed into the unified [`BinaryExpr`] re-export below, strict-
// mirroring `datafusion_physical_expr::expressions::BinaryExpr`. Call
// sites build `BinaryExpr::new(left, fdapquery_expr::Operator::Eq,
// right)` etc.
pub use fdapquery_physical_expr::{
    // Unified binary expression.
    BinaryExpr,
    // Cast.
    CastExpr,
    // Column reference.
    Column,
    // Date arithmetic.
    DateAddIntervalExpr,
    DateSubtractIntervalExpr,
    // Literal — collapsed the five sibling literal types
    // (`LiteralLong`, `LiteralDouble`, `LiteralString`, `LiteralDate`,
    // `LiteralIntervalDays`) into a single `Literal { value: ScalarValue }`
    // plus a `lit()` factory, mirroring DataFusion's
    // `datafusion_physical_expr::expressions::Literal` exactly.
    Literal,
    // Unary math.
    Log,
    Sqrt,
    UnaryMathExpr,
    lit,
};
