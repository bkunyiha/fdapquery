//!
//! `MIN(expr)` — keeps the smallest non-null value seen.

use crate::aggregate_expression::{AggregateExpr, scalar_lt};
use crate::expressions::{Accumulator, AccumulatorValue, PhysicalExpr};
use fdapquery_common::{Result, ScalarValue};
use std::fmt;
use std::sync::Arc;

/// `MIN(expr)`.
#[derive(Debug)]
pub struct MinExpr {
    expr: Arc<dyn PhysicalExpr>,
}

impl MinExpr {
    pub fn new(expr: Arc<dyn PhysicalExpr>) -> Self {
        Self { expr }
    }
}

impl AggregateExpr for MinExpr {
    fn input_expression(&self) -> Arc<dyn PhysicalExpr> {
        Arc::clone(&self.expr)
    }
    fn create_accumulator(&self) -> Box<dyn Accumulator> {
        Box::new(MinAccumulator::new())
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl fmt::Display for MinExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MIN({})", self.expr)
    }
}

/// Keeps the running minimum. `ScalarValue::Null` is the "no value yet" state.
pub struct MinAccumulator {
    value: ScalarValue,
}

impl MinAccumulator {
    pub fn new() -> Self {
        Self {
            value: ScalarValue::Null,
        }
    }
}

impl Default for MinAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

impl Accumulator for MinAccumulator {
    fn accumulate(&mut self, value: &ScalarValue) -> Result<()> {
        if value.is_null() {
            return Ok(());
        }
        if self.value.is_null() || scalar_lt(value, &self.value)? {
            self.value = value.clone();
        }
        Ok(())
    }

    fn final_value(&self) -> Result<ScalarValue> {
        Ok(self.value.clone())
    }

    fn merge(&mut self, other: &AccumulatorValue) -> Result<()> {
        // For MIN, merging a partial state is the same as accumulating it.
        if let AccumulatorValue::Scalar(v) = other {
            self.accumulate(v)?;
        }
        Ok(())
    }
}
