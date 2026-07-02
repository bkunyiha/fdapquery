//! `Sort` expression and `NullTreatment` enum — byte-for-byte mirror of
//! the same types in DataFusion's `datafusion/expr/src/expr.rs`.
//!
//! Introduced so the new `AggregateFunctionParams`
//! struct can carry `order_by: Vec<Sort>` and `null_treatment:
//! Option<NullTreatment>` byte-for-byte (cf. DataFusion
//! `datafusion_expr::expr::AggregateFunctionParams`). Neither field is
//! populated by the SQL planner today — fdapquery has no
//! `ORDER BY` / `IGNORE NULLS` SQL surface — but the struct shape
//! must match DataFusion so the field set is identical.

use crate::logical_expr::Expr;
use std::fmt;

/// How an aggregate or window function treats nulls in its argument.
/// Byte-for-byte mirror of DataFusion's `datafusion_expr::NullTreatment`.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub enum NullTreatment {
    IgnoreNulls,
    RespectNulls,
}

impl fmt::Display for NullTreatment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            NullTreatment::IgnoreNulls => "IGNORE NULLS",
            NullTreatment::RespectNulls => "RESPECT NULLS",
        })
    }
}

/// SORT expression — byte-for-byte mirror of DataFusion's
/// `datafusion_expr::expr::Sort` struct. DataFusion's `Sort` derives
/// `Eq` / `Hash` / `PartialOrd` because its `Expr` does; fdapquery's `Expr`
/// derives only `Debug, Clone, PartialEq`, so `Sort` inherits
/// the same minimal derive set here.
#[derive(Debug, Clone, PartialEq)]
pub struct Sort {
    /// The expression to sort on
    pub expr: Expr,
    /// The direction of the sort
    pub asc: bool,
    /// Whether to put Nulls before all other data values
    pub nulls_first: bool,
}

impl Sort {
    /// Create a new Sort expression
    pub fn new(expr: Expr, asc: bool, nulls_first: bool) -> Self {
        Self {
            expr,
            asc,
            nulls_first,
        }
    }
}

impl fmt::Display for Sort {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.expr)?;
        if self.asc {
            write!(f, " ASC")?;
        } else {
            write!(f, " DESC")?;
        }
        if self.nulls_first {
            write!(f, " NULLS FIRST")?;
        } else {
            write!(f, " NULLS LAST")?;
        }
        Ok(())
    }
}
