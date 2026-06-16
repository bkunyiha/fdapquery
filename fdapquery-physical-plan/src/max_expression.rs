//!
//! `MAX(expr)` — keeps the largest non-null value seen.

use crate::aggregate_expression::{AggregateExpression, scalar_gt};
use crate::expressions::{Accumulator, AccumulatorValue, Expression};
use datatypes::ScalarValue;
use std::fmt;
use std::sync::Arc;

/// `MAX(expr)`.
pub struct MaxExpression {
    expr: Arc<dyn Expression>,
}

impl MaxExpression {
    pub fn new(expr: Arc<dyn Expression>) -> Self {
        Self { expr }
    }
}

impl AggregateExpression for MaxExpression {
    fn input_expression(&self) -> Arc<dyn Expression> {
        Arc::clone(&self.expr)
    }
    fn create_accumulator(&self) -> Box<dyn Accumulator> {
        Box::new(MaxAccumulator::new())
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl fmt::Display for MaxExpression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MAX({})", self.expr)
    }
}

/// Keeps the running maximum.
pub struct MaxAccumulator {
    value: ScalarValue,
}

impl MaxAccumulator {
    pub fn new() -> Self {
        Self {
            value: ScalarValue::Null,
        }
    }
}

impl Default for MaxAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

impl Accumulator for MaxAccumulator {
    fn accumulate(&mut self, value: &ScalarValue) {
        if value.is_null() {
            return;
        }
        if self.value.is_null() || scalar_gt(value, &self.value) {
            self.value = value.clone();
        }
    }

    fn final_value(&self) -> ScalarValue {
        self.value.clone()
    }

    fn merge(&mut self, other: &AccumulatorValue) {
        // For MAX, merging a partial state is the same as accumulating it.
        if let AccumulatorValue::Scalar(v) = other {
            self.accumulate(v);
        }
    }
}
