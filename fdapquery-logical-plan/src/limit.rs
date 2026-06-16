//!
//! Logical plan representing a limit. Does not change the input schema.

use crate::logical_plan::LogicalPlan;
use datatypes::Schema;
use std::fmt;

#[derive(Clone)]
pub struct Limit {
    pub input: Box<LogicalPlan>,
    pub limit: i32,
}

impl Limit {
    pub fn new(input: LogicalPlan, limit: i32) -> Self {
        Self {
            input: Box::new(input),
            limit,
        }
    }

    pub fn schema(&self) -> Schema {
        self.input.schema()
    }

    pub fn children(&self) -> Vec<&LogicalPlan> {
        vec![self.input.as_ref()]
    }
}

impl fmt::Display for Limit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Limit: {}", self.limit)
    }
}
