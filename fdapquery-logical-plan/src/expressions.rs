//! # What lives here vs. in `logical_expr.rs`
//!
//! This module holds two kinds of thing: (1) the `AggregateExpr` family, and
//! (2) the convenience constructors for `LogicalExpr` and `AggregateExpr`.
//!
//! - (1) `AggregateExpr` is its own sum type — a narrow family (`Sum`, `Min`,
//!   `Max`, `Avg`, `Count`, `CountDistinct`) that is also part of the broader
//!   `LogicalExpr` family. The `Aggregate` plan ranges over a typed
//!   `Vec<AggregateExpr>`, and the `From<AggregateExpr> for LogicalExpr` impl
//!   below bridges an aggregate back into `LogicalExpr` via the single
//!   `LogicalExpr::AggregateExpr` variant — exactly the shape of DataFusion's
//!   `Expr::AggregateFunction`.
//!
//! - (2) The convenience constructors are introduction forms — functions into
//!   a type (`col: &str -> LogicalExpr`, `lit_long: i64 -> LogicalExpr`, the
//!   `eq`/`add`/… builder methods, and `sum`/`min`/… which build an
//!   `AggregateExpr`). They live here rather than in `logical_expr.rs` so the
//!   enum definition stays narrowly focused.
//!
//! Literal constructors are spelled out per type (`lit_string`, `lit_long`,
//! …) because Rust has no function overloading; comparison and arithmetic
//! builders are `self`-consuming methods (`a.eq(b)`, `a.mult(b).alias("x")`).

use crate::logical_expr::LogicalExpr;
use crate::logical_plan::LogicalPlan;
use arrow_schema::DataType;
use datatypes::Field;
use datatypes::arrow_types::{INT32_TYPE, UINT32_TYPE};
use std::fmt;

/// Aggregate functions: `Sum` / `Min` / `Max` / `Avg` / `Count` /
/// `CountDistinct`. Kept as its own enum so the `Aggregate` plan and
/// `DataFrame::aggregate` keep a typed `Vec<AggregateExpr>`; bridged into
/// `LogicalExpr` (for nesting inside expressions, e.g. `HAVING`) by the
/// `From<AggregateExpr> for LogicalExpr` impl below — the analogue of
/// DataFusion's `Expr::AggregateFunction`.
#[derive(Debug, Clone, PartialEq)]
pub enum AggregateExpr {
    Sum(LogicalExpr),
    Min(LogicalExpr),
    Max(LogicalExpr),
    Avg(LogicalExpr),
    Count(LogicalExpr),
    CountDistinct(LogicalExpr),
}

impl AggregateExpr {
    /// Compute the output `Field` for this aggregate against `input`'s schema.
    /// SUM/MIN/MAX/AVG carry the data type of their input expression; COUNT
    /// and COUNT DISTINCT are integer counts.
    pub fn to_field(&self, input: &LogicalPlan) -> Field {
        match self {
            AggregateExpr::Sum(e) => Field::new("SUM", e.to_field(input).data_type),
            AggregateExpr::Min(e) => Field::new("MIN", e.to_field(input).data_type),
            AggregateExpr::Max(e) => Field::new("MAX", e.to_field(input).data_type),
            AggregateExpr::Avg(e) => Field::new("AVG", e.to_field(input).data_type),
            AggregateExpr::Count(_) => Field::new("COUNT", INT32_TYPE),
            AggregateExpr::CountDistinct(_) => Field::new("COUNT_DISTINCT", UINT32_TYPE),
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

/// The bridge: inject an `AggregateExpr` into `LogicalExpr` so it can nest
/// inside any expression (cf. DataFusion's `Expr::AggregateFunction`).
impl From<AggregateExpr> for LogicalExpr {
    fn from(agg: AggregateExpr) -> Self {
        LogicalExpr::AggregateExpr(Box::new(agg))
    }
}

// ==============================================================
// `self`-consuming builder methods for comparison and arithmetic.
// ==============================================================
// `add` / `div` are deliberately named methods (alongside `subtract` / `mult` /
// `modulus`); they build AST nodes, not compute values, so they are
// intentionally *not* `std::ops::{Add, Div}` impls.
#[allow(clippy::should_implement_trait)]
impl LogicalExpr {
    pub fn eq(self, rhs: LogicalExpr) -> LogicalExpr {
        LogicalExpr::Eq {
            l: Box::new(self),
            r: Box::new(rhs),
        }
    }
    pub fn neq(self, rhs: LogicalExpr) -> LogicalExpr {
        LogicalExpr::Neq {
            l: Box::new(self),
            r: Box::new(rhs),
        }
    }
    pub fn gt(self, rhs: LogicalExpr) -> LogicalExpr {
        LogicalExpr::Gt {
            l: Box::new(self),
            r: Box::new(rhs),
        }
    }
    pub fn gteq(self, rhs: LogicalExpr) -> LogicalExpr {
        LogicalExpr::GtEq {
            l: Box::new(self),
            r: Box::new(rhs),
        }
    }
    pub fn lt(self, rhs: LogicalExpr) -> LogicalExpr {
        LogicalExpr::Lt {
            l: Box::new(self),
            r: Box::new(rhs),
        }
    }
    pub fn lteq(self, rhs: LogicalExpr) -> LogicalExpr {
        LogicalExpr::LtEq {
            l: Box::new(self),
            r: Box::new(rhs),
        }
    }
    pub fn and(self, rhs: LogicalExpr) -> LogicalExpr {
        LogicalExpr::And {
            l: Box::new(self),
            r: Box::new(rhs),
        }
    }
    pub fn or(self, rhs: LogicalExpr) -> LogicalExpr {
        LogicalExpr::Or {
            l: Box::new(self),
            r: Box::new(rhs),
        }
    }
    pub fn add(self, rhs: LogicalExpr) -> LogicalExpr {
        LogicalExpr::Add {
            l: Box::new(self),
            r: Box::new(rhs),
        }
    }
    pub fn subtract(self, rhs: LogicalExpr) -> LogicalExpr {
        LogicalExpr::Subtract {
            l: Box::new(self),
            r: Box::new(rhs),
        }
    }
    pub fn mult(self, rhs: LogicalExpr) -> LogicalExpr {
        LogicalExpr::Multiply {
            l: Box::new(self),
            r: Box::new(rhs),
        }
    }
    pub fn div(self, rhs: LogicalExpr) -> LogicalExpr {
        LogicalExpr::Divide {
            l: Box::new(self),
            r: Box::new(rhs),
        }
    }
    pub fn modulus(self, rhs: LogicalExpr) -> LogicalExpr {
        LogicalExpr::Modulus {
            l: Box::new(self),
            r: Box::new(rhs),
        }
    }
    pub fn alias(self, alias: impl Into<String>) -> LogicalExpr {
        LogicalExpr::Alias {
            expr: Box::new(self),
            alias: alias.into(),
        }
    }
}

// ==============================================================
// Convenience constructors for `LogicalExpr` and `AggregateExpr`.
// ==============================================================

/// Create a column reference by name.
pub fn col(name: impl Into<String>) -> LogicalExpr {
    LogicalExpr::Column(name.into())
}

/// Literal string.
pub fn lit_string(value: impl Into<String>) -> LogicalExpr {
    LogicalExpr::LiteralString(value.into())
}
/// Literal `i64`.
pub fn lit_long(value: i64) -> LogicalExpr {
    LogicalExpr::LiteralLong(value)
}
/// Literal `f32`.
pub fn lit_float(value: f32) -> LogicalExpr {
    LogicalExpr::LiteralFloat(value)
}
/// Literal `f64`.
pub fn lit_double(value: f64) -> LogicalExpr {
    LogicalExpr::LiteralDouble(value)
}
/// Literal date.
pub fn lit_date(value: chrono::NaiveDate) -> LogicalExpr {
    LogicalExpr::LiteralDate(value)
}

/// Cast `expr` to `data_type`.
pub fn cast(expr: LogicalExpr, data_type: DataType) -> LogicalExpr {
    LogicalExpr::Cast {
        expr: Box::new(expr),
        data_type,
    }
}

pub fn sum(expr: LogicalExpr) -> AggregateExpr {
    AggregateExpr::Sum(expr)
}
pub fn min(expr: LogicalExpr) -> AggregateExpr {
    AggregateExpr::Min(expr)
}
pub fn max(expr: LogicalExpr) -> AggregateExpr {
    AggregateExpr::Max(expr)
}
pub fn avg(expr: LogicalExpr) -> AggregateExpr {
    AggregateExpr::Avg(expr)
}
pub fn count(expr: LogicalExpr) -> AggregateExpr {
    AggregateExpr::Count(expr)
}
pub fn count_distinct(expr: LogicalExpr) -> AggregateExpr {
    AggregateExpr::CountDistinct(expr)
}
