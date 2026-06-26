//!
//! Logical plan representing a filter (a.k.a. filter) against an input.
//! Filter does not change the schema of its input.

use crate::logical_expr::Expr;
use crate::logical_plan::LogicalPlan;
use fdapquery_datatypes::{Result, Schema};
use std::fmt;

#[derive(Clone)]
pub struct Filter {
    pub input: Box<LogicalPlan>,
    pub expr: Expr,
}

impl Filter {
    pub fn new(input: LogicalPlan, expr: Expr) -> Self {
        Self {
            input: Box::new(input),
            expr,
        }
    }

    pub fn schema(&self) -> Result<Schema> {
        self.input.schema()
    }

    pub fn children(&self) -> Vec<&LogicalPlan> {
        // self.input is likely a Box<LogicalPlan>, so we need to dereference it
        // to get the actual LogicalPlan reference(&LogicalPlan).
        vec![self.input.as_ref()]
    }
}

impl fmt::Display for Filter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Filter: {}", self.expr)
    }
}
