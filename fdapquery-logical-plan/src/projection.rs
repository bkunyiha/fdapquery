//!
//! Logical plan representing a projection (evaluating a list of expressions)
//! against an input.

use crate::logical_expr::LogicalExpr;
use crate::logical_plan::LogicalPlan;
use datatypes::Schema;
use std::fmt;

#[derive(Clone)]
pub struct Projection {
    pub input: Box<LogicalPlan>, // input is boxed because LogicalPlan is recursive.
    pub expr: Vec<LogicalExpr>,
}

impl Projection {
    pub fn new(input: LogicalPlan, expr: Vec<LogicalExpr>) -> Self {
        Self {
            input: Box::new(input),
            expr,
        }
    }

    pub fn schema(&self) -> Schema {
        Schema::new(self.expr.iter().map(|e| e.to_field(&self.input)).collect())
    }

    pub fn children(&self) -> Vec<&LogicalPlan> {
        vec![self.input.as_ref()]
    }
}

impl fmt::Display for Projection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let exprs: Vec<String> = self.expr.iter().map(|e| e.to_string()).collect();
        write!(f, "Projection: {}", exprs.join(", "))
    }
}
