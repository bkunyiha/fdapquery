//!
//! `COUNT(expr)` — number of non-null values. Always returns an `Int32` (the
//! count is `0` for an empty/all-null group, never null).

use crate::aggregate_expression::AggregateExpr;
use crate::expressions::{Accumulator, AccumulatorValue, PhysicalExpr, number_to_i64};
use fdapquery_common::{Result, ScalarValue};
use std::fmt;
use std::sync::Arc;

/// `COUNT(expr)`.
#[derive(Debug)]
pub struct CountExpr {
    expr: Arc<dyn PhysicalExpr>,
}

impl CountExpr {
    pub fn new(expr: Arc<dyn PhysicalExpr>) -> Self {
        Self { expr }
    }
}

impl AggregateExpr for CountExpr {
    fn input_expression(&self) -> Arc<dyn PhysicalExpr> {
        Arc::clone(&self.expr)
    }
    fn create_accumulator(&self) -> Box<dyn Accumulator> {
        Box::new(CountAccumulator::new())
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl fmt::Display for CountExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "COUNT({})", self.expr)
    }
}

/// Counts non-null values.
pub struct CountAccumulator {
    count: i32,
}

impl CountAccumulator {
    pub fn new() -> Self {
        Self { count: 0 }
    }
}

impl Default for CountAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

impl Accumulator for CountAccumulator {
    fn accumulate(&mut self, value: &ScalarValue) -> Result<()> {
        if !value.is_null() {
            self.count += 1;
        }
        Ok(())
    }

    fn final_value(&self) -> Result<ScalarValue> {
        Ok(ScalarValue::Int32(self.count))
    }

    fn merge(&mut self, other: &AccumulatorValue) -> Result<()> {
        // COUNT merges by adding the partial counts together.
        if let AccumulatorValue::Scalar(s) = other {
            if !s.is_null() {
                self.count += number_to_i64(s)? as i32;
            }
        }
        Ok(())
    }
}
