//!
//! References a column in the input batch by its position. Evaluating it simply
//! hands back that column unchanged — the simplest possible physical expression.

use crate::expressions::Expression;
use datatypes::{ColumnVector, RecordBatch, record_batch};
use std::fmt;

/// Reference a column in a batch by index.
pub struct ColumnExpression {
    pub i: usize,
}

impl ColumnExpression {
    pub fn new(i: usize) -> Self {
        Self { i }
    }
}

impl Expression for ColumnExpression {
    fn evaluate(&self, input: &RecordBatch) -> Box<dyn ColumnVector> {
        // `record_batch::field` wraps the existing arrow `ArrayRef`
        // (cheap, Arc-cloned) as a ColumnVector.
        Box::new(record_batch::field(input, self.i))
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl fmt::Display for ColumnExpression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "#{}", self.i)
    }
}
