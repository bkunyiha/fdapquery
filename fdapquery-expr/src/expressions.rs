//! # What lives here vs. in `logical_expr.rs`
//!
//! This module holds two kinds of thing: (1) convenience constructors for
//! `Expr`, and (2) the `self`-consuming builder methods for comparison
//! and arithmetic (`a.eq(b)`, `a.add(b)`, …).
//!
//! - The convenience constructors are introduction forms — functions into
//!   the type (`col: &str -> Expr`, `sum`/`min`/`max`/`avg`/`count`/
//!   `count_distinct` — each returning `Expr` directly via the
//!   `Expr::AggregateFunction` variant). They live here rather than in
//!   `logical_expr.rs` so the enum definition stays narrowly focused.
//!
//! The single literal factory `lit<T: Literal>(value: T) -> Expr` lives in
//! its own `expr_fn` module (alongside DataFusion's layout); the `Literal`
//! trait that powers it lives in `literal.rs`. Collapsed the
//! per-type `lit_string` / `lit_long` / `lit_float` / `lit_double` /
//! `lit_date` family into that single generic entry point.
//!
//! Comparison and arithmetic builders remain `self`-consuming methods
//! (`a.eq(b)`, `a.mult(b).alias("x")`).
//!
//! ## DSL constructors return `Expr`
//!
//! Every aggregate is a single `Expr::AggregateFunction(AggregateFunction)`
//! variant, and the DSL constructors return `Expr` directly —
//! byte-for-byte the same signatures DataFusion uses for `min`, `max`,
//! `sum`, `avg`, and `count` in `datafusion/functions-aggregate/src/`
//! (where `make_udaf_expr_and_func!` emits `pub fn min(expr: Expr) -> Expr`
//! etc., wrapping the call in `Expr::AggregateFunction(...)`).

use crate::aggregate_function::{AggregateFunction, AggregateFunctionKind};
use crate::logical_expr::Expr;
use crate::operator::Operator;
use arrow_schema::DataType;

// ==============================================================
// `self`-consuming builder methods for comparison and arithmetic.
// ==============================================================
// Every builder constructs the unified
// `Expr::BinaryExpr { left, op, right }` (parameterised by [`Operator`]).
// Same shape as DataFusion's `Expr` builder helpers in
// `datafusion/expr/src/expr.rs`.
// `add` / `div` are deliberately named methods (alongside `subtract` / `mult` /
// `modulus`); they build AST nodes, not compute values, so they are
// intentionally *not* `std::ops::{Add, Div}` impls.
#[allow(clippy::should_implement_trait)]
impl Expr {
    /// Build `self op rhs`.
    fn binary(self, op: Operator, rhs: Expr) -> Expr {
        Expr::BinaryExpr {
            left: Box::new(self),
            op,
            right: Box::new(rhs),
        }
    }

    pub fn eq(self, rhs: Expr) -> Expr {
        self.binary(Operator::Eq, rhs)
    }
    pub fn neq(self, rhs: Expr) -> Expr {
        self.binary(Operator::NotEq, rhs)
    }
    pub fn gt(self, rhs: Expr) -> Expr {
        self.binary(Operator::Gt, rhs)
    }
    pub fn gteq(self, rhs: Expr) -> Expr {
        self.binary(Operator::GtEq, rhs)
    }
    pub fn lt(self, rhs: Expr) -> Expr {
        self.binary(Operator::Lt, rhs)
    }
    pub fn lteq(self, rhs: Expr) -> Expr {
        self.binary(Operator::LtEq, rhs)
    }
    pub fn and(self, rhs: Expr) -> Expr {
        self.binary(Operator::And, rhs)
    }
    pub fn or(self, rhs: Expr) -> Expr {
        self.binary(Operator::Or, rhs)
    }
    pub fn add(self, rhs: Expr) -> Expr {
        self.binary(Operator::Plus, rhs)
    }
    pub fn subtract(self, rhs: Expr) -> Expr {
        self.binary(Operator::Minus, rhs)
    }
    pub fn mult(self, rhs: Expr) -> Expr {
        self.binary(Operator::Multiply, rhs)
    }
    pub fn div(self, rhs: Expr) -> Expr {
        self.binary(Operator::Divide, rhs)
    }
    pub fn modulus(self, rhs: Expr) -> Expr {
        self.binary(Operator::Modulo, rhs)
    }
    pub fn alias(self, alias: impl Into<String>) -> Expr {
        Expr::Alias {
            expr: Box::new(self),
            alias: alias.into(),
        }
    }
}

// ==============================================================
// Convenience constructors for `Expr`.
// ==============================================================

/// Create a column reference by name.
pub fn col(name: impl Into<String>) -> Expr {
    Expr::Column(name.into())
}

/// Cast `expr` to `data_type`.
pub fn cast(expr: Expr, data_type: DataType) -> Expr {
    Expr::Cast {
        expr: Box::new(expr),
        data_type,
    }
}

/// Construct an aggregate-function expression with no DISTINCT, FILTER,
/// ORDER BY, or null treatment — the default the SQL planner emits today.
/// Mirrors the body that DataFusion's `make_udaf_expr_and_func!` macro
/// generates for each built-in aggregate (e.g. the `min` / `max` /
/// `sum` / `avg` / `count` functions in
/// `datafusion/functions-aggregate/src/{min_max, sum, average, count}.rs`).
fn agg(func: AggregateFunctionKind, expr: Expr, distinct: bool) -> Expr {
    Expr::AggregateFunction(AggregateFunction::new(
        func,
        vec![expr],
        distinct,
        None,
        Vec::new(),
        None,
    ))
}

pub fn sum(expr: Expr) -> Expr {
    agg(AggregateFunctionKind::Sum, expr, false)
}
pub fn min(expr: Expr) -> Expr {
    agg(AggregateFunctionKind::Min, expr, false)
}
pub fn max(expr: Expr) -> Expr {
    agg(AggregateFunctionKind::Max, expr, false)
}
pub fn avg(expr: Expr) -> Expr {
    agg(AggregateFunctionKind::Avg, expr, false)
}
pub fn count(expr: Expr) -> Expr {
    agg(AggregateFunctionKind::Count, expr, false)
}
pub fn count_distinct(expr: Expr) -> Expr {
    agg(AggregateFunctionKind::Count, expr, true)
}
