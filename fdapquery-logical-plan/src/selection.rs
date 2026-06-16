//!
//! Logical plan representing a selection (a.k.a. filter) against an input.
//! Selection does not change the schema of its input.

use crate::logical_expr::LogicalExpr;
use crate::logical_plan::LogicalPlan;
use datatypes::Schema;
use std::fmt;

#[derive(Clone)]
pub struct Selection {
    pub input: Box<LogicalPlan>,
    pub expr: LogicalExpr,
}

impl Selection {
    pub fn new(input: LogicalPlan, expr: LogicalExpr) -> Self {
        Self {
            input: Box::new(input),
            expr,
        }
    }

    pub fn schema(&self) -> Schema {
        self.input.schema()
    }

    pub fn children(&self) -> Vec<&LogicalPlan> {
        // self.input is likely a Box<LogicalPlan>, so we need to dereference it
        // to get the actual LogicalPlan reference(&LogicalPlan).
        vec![self.input.as_ref()]
    }
}

impl fmt::Display for Selection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Selection: {}", self.expr)
    }
}
