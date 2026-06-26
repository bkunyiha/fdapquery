//!
//! References a column in the input batch by its position. Evaluating it simply
//! hands back that column unchanged — the simplest possible physical expression.

use crate::expressions::PhysicalExpr;
use fdapquery_datatypes::{ColumnVector, RecordBatch, Result, record_batch};
use std::fmt;

/// Reference a column in a batch by index.
pub struct Column {
    pub i: usize,
}

impl Column {
    pub fn new(i: usize) -> Self {
        Self { i }
    }
}

impl PhysicalExpr for Column {
    fn evaluate(&self, input: &RecordBatch) -> Result<Box<dyn ColumnVector>> {
        // `record_batch::field` wraps the existing arrow `ArrayRef`
        // (cheap, Arc-cloned) as a ColumnVector.
        Ok(Box::new(record_batch::field(input, self.i)))
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl fmt::Display for Column {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "#{}", self.i)
    }
}
