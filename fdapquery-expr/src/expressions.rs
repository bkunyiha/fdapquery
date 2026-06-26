//! # What lives here vs. in `logical_expr.rs`
//!
//! This module holds two kinds of thing: (1) the `AggregateExpr` family, and
//! (2) the convenience constructors for `Expr` and `AggregateExpr`.
//!
//! - (1) `AggregateExpr` is its own sum type — a narrow family (`Sum`, `Min`,
//!   `Max`, `Avg`, `Count`, `CountDistinct`) that is also part of the broader
//!   `Expr` family. The `Aggregate` plan ranges over a typed
//!   `Vec<AggregateExpr>`, and the `From<AggregateExpr> for Expr` impl
//!   below bridges an aggregate back into `Expr` via the single
//!   `Expr::AggregateExpr` variant — exactly the shape of DataFusion's
//!   `Expr::AggregateFunction`.
//!
//! - (2) The convenience constructors are introduction forms — functions into
//!   a type (`col: &str -> Expr`, `lit_long: i64 -> Expr`, the
//!   `eq`/`add`/… builder methods, and `sum`/`min`/… which build an
//!   `AggregateExpr`). They live here rather than in `logical_expr.rs` so the
//!   enum definition stays narrowly focused.
//!
//! Literal constructors are spelled out per type (`lit_string`, `lit_long`,
//! …) because Rust has no function overloading; comparison and arithmetic
//! builders are `self`-consuming methods (`a.eq(b)`, `a.mult(b).alias("x")`).

use crate::logical_expr::Expr;
use crate::logical_plan::LogicalPlan;
use arrow_schema::DataType;
use fdapquery_datatypes::{Field, Result};
use std::fmt;

/// Aggregate functions: `Sum` / `Min` / `Max` / `Avg` / `Count` /
/// `CountDistinct`. Kept as its own enum so the `Aggregate` plan and
/// `DataFrame::aggregate` keep a typed `Vec<AggregateExpr>`; bridged into
/// `Expr` (for nesting inside expressions, e.g. `HAVING`) by the
/// `From<AggregateExpr> for Expr` impl below — the analogue of
/// DataFusion's `Expr::AggregateFunction`.
#[derive(Debug, Clone, PartialEq)]
pub enum AggregateExpr {
    Sum(Expr),
    Min(Expr),
    Max(Expr),
    Avg(Expr),
    Count(Expr),
    CountDistinct(Expr),
}

impl AggregateExpr {
    /// Compute the output `Field` for this aggregate against `input`'s schema.
    /// SUM/MIN/MAX/AVG carry the data type of their input expression; COUNT
    /// and COUNT DISTINCT are integer counts.
    pub fn to_field(&self, input: &LogicalPlan) -> Result<Field> {
        match self {
            AggregateExpr::Sum(e) => Ok(Field::new(
                "SUM",
                e.to_field(input)?.data_type().clone(),
                true,
            )),
            AggregateExpr::Min(e) => Ok(Field::new(
                "MIN",
                e.to_field(input)?.data_type().clone(),
                true,
            )),
            AggregateExpr::Max(e) => Ok(Field::new(
                "MAX",
                e.to_field(input)?.data_type().clone(),
                true,
            )),
            AggregateExpr::Avg(e) => Ok(Field::new(
                "AVG",
                e.to_field(input)?.data_type().clone(),
                true,
            )),
            AggregateExpr::Count(_) => Ok(Field::new("COUNT", arrow_schema::DataType::Int32, true)),
            AggregateExpr::CountDistinct(_) => Ok(Field::new(
                "COUNT_DISTINCT",
                arrow_schema::DataType::UInt32,
                true,
            )),
        }
    }
}

impl fmt::Display for AggregateExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AggregateExpr::Sum(e) => write!(f, "SUM({e})"),
            AggregateExpr::Min(e) => write!(f, "MIN({e})"),
            AggregateExpr::Max(e) => write!(f, "MAX({e})"),
            AggregateExpr::Avg(e) => write!(f, "AVG({e})"),
            AggregateExpr::Count(e) => write!(f, "COUNT({e})"),
            AggregateExpr::CountDistinct(e) => write!(f, "COUNT(DISTINCT {e})"),
        }
    }
}

/// The bridge: inject an `AggregateExpr` into `Expr` so it can nest
/// inside any expression (cf. DataFusion's `Expr::AggregateFunction`).
impl From<AggregateExpr> for Expr {
    fn from(agg: AggregateExpr) -> Self {
        Expr::AggregateExpr(Box::new(agg))
    }
}

// ==============================================================
// `self`-consuming builder methods for comparison and arithmetic.
// ==============================================================
// `add` / `div` are deliberately named methods (alongside `subtract` / `mult` /
// `modulus`); they build AST nodes, not compute values, so they are
// intentionally *not* `std::ops::{Add, Div}` impls.
#[allow(clippy::should_implement_trait)]
impl Expr {
    pub fn eq(self, rhs: Expr) -> Expr {
        Expr::Eq {
            l: Box::new(self),
            r: Box::new(rhs),
        }
    }
    pub fn neq(self, rhs: Expr) -> Expr {
        Expr::Neq {
            l: Box::new(self),
            r: Box::new(rhs),
        }
    }
    pub fn gt(self, rhs: Expr) -> Expr {
        Expr::Gt {
            l: Box::new(self),
            r: Box::new(rhs),
        }
    }
    pub fn gteq(self, rhs: Expr) -> Expr {
        Expr::GtEq {
            l: Box::new(self),
            r: Box::new(rhs),
        }
    }
    pub fn lt(self, rhs: Expr) -> Expr {
        Expr::Lt {
            l: Box::new(self),
            r: Box::new(rhs),
        }
    }
    pub fn lteq(self, rhs: Expr) -> Expr {
        Expr::LtEq {
            l: Box::new(self),
            r: Box::new(rhs),
        }
    }
    pub fn and(self, rhs: Expr) -> Expr {
        Expr::And {
            l: Box::new(self),
            r: Box::new(rhs),
        }
    }
    pub fn or(self, rhs: Expr) -> Expr {
        Expr::Or {
            l: Box::new(self),
            r: Box::new(rhs),
        }
    }
    pub fn add(self, rhs: Expr) -> Expr {
        Expr::Add {
            l: Box::new(self),
            r: Box::new(rhs),
        }
    }
    pub fn subtract(self, rhs: Expr) -> Expr {
        Expr::Subtract {
            l: Box::new(self),
            r: Box::new(rhs),
        }
    }
    pub fn mult(self, rhs: Expr) -> Expr {
        Expr::Multiply {
            l: Box::new(self),
            r: Box::new(rhs),
        }
    }
    pub fn div(self, rhs: Expr) -> Expr {
        Expr::Divide {
            l: Box::new(self),
            r: Box::new(rhs),
        }
    }
    pub fn modulus(self, rhs: Expr) -> Expr {
        Expr::Modulus {
            l: Box::new(self),
            r: Box::new(rhs),
        }
    }
    pub fn alias(self, alias: impl Into<String>) -> Expr {
        Expr::Alias {
            expr: Box::new(self),
            alias: alias.into(),
        }
    }
}

// ==============================================================
// Convenience constructors for `Expr` and `AggregateExpr`.
// ==============================================================

/// Create a column reference by name.
pub fn col(name: impl Into<String>) -> Expr {
    Expr::Column(name.into())
}

/// Literal string.
pub fn lit_string(value: impl Into<String>) -> Expr {
    Expr::LiteralString(value.into())
}
/// Literal `i64`.
pub fn lit_long(value: i64) -> Expr {
    Expr::LiteralLong(value)
}
/// Literal `f32`.
pub fn lit_float(value: f32) -> Expr {
    Expr::LiteralFloat(value)
}
/// Literal `f64`.
pub fn lit_double(value: f64) -> Expr {
    Expr::LiteralDouble(value)
}
/// Literal date.
pub fn lit_date(value: chrono::NaiveDate) -> Expr {
    Expr::LiteralDate(value)
}

/// Cast `expr` to `data_type`.
pub fn cast(expr: Expr, data_type: DataType) -> Expr {
    Expr::Cast {
        expr: Box::new(expr),
        data_type,
    }
}

pub fn sum(expr: Expr) -> AggregateExpr {
    AggregateExpr::Sum(expr)
}
pub fn min(expr: Expr) -> AggregateExpr {
    AggregateExpr::Min(expr)
}
pub fn max(expr: Expr) -> AggregateExpr {
    AggregateExpr::Max(expr)
}
pub fn avg(expr: Expr) -> AggregateExpr {
    AggregateExpr::Avg(expr)
}
pub fn count(expr: Expr) -> AggregateExpr {
    AggregateExpr::Count(expr)
}
pub fn count_distinct(expr: Expr) -> AggregateExpr {
    AggregateExpr::CountDistinct(expr)
}
