//!
//! `MIN(expr)` — keeps the smallest non-null value seen.

use crate::aggregate_expression::{AggregateExpression, scalar_lt};
use crate::expressions::{Accumulator, AccumulatorValue, Expression};
use datatypes::ScalarValue;
use std::fmt;
use std::sync::Arc;

/// `MIN(expr)`.
pub struct MinExpression {
    expr: Arc<dyn Expression>,
}

impl MinExpression {
    pub fn new(expr: Arc<dyn Expression>) -> Self {
        Self { expr }
    }
}

impl AggregateExpression for MinExpression {
    fn input_expression(&self) -> Arc<dyn Expression> {
        Arc::clone(&self.expr)
    }
    fn create_accumulator(&self) -> Box<dyn Accumulator> {
        Box::new(MinAccumulator::new())
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl fmt::Display for MinExpression {
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
    fn accumulate(&mut self, value: &ScalarValue) {
        if value.is_null() {
            return;
        }
        if self.value.is_null() || scalar_lt(value, &self.value) {
            self.value = value.clone();
        }
    }

    fn final_value(&self) -> ScalarValue {
        self.value.clone()
    }

    fn merge(&mut self, other: &AccumulatorValue) {
        // For MIN, merging a partial state is the same as accumulating it.
        if let AccumulatorValue::Scalar(v) = other {
            self.accumulate(v);
        }
    }
}
