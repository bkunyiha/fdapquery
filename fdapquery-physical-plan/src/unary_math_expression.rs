//!
//! Unary math functions over a numeric column, producing a `Float64` column:
//! `Sqrt` and `Log` (natural log).
//!
//! ## Trait with a default method
//! As with the binary/boolean/math families, the shared "evaluate input, map
//! each non-null value through `apply`" logic lives in a trait
//! ([`UnaryMathExpression`]) with a default method (`evaluate_unary`) and a
//! required `apply` kernel. Each concrete function implements
//! `UnaryMathExpression` and a one-line `Expression` delegate.

use crate::expressions::{Expression, number_to_f64};
use datatypes::arrow_types::DOUBLE_TYPE;
use datatypes::{ArrowVectorBuilder, ColumnVector, RecordBatch, ScalarValue};
use std::fmt;
use std::sync::Arc;

/// A unary math function.
pub trait UnaryMathExpression: Expression {
    /// The input expression whose values are transformed.
    fn input(&self) -> &Arc<dyn Expression>;

    /// The function applied to each non-null value.
    fn apply(&self, value: f64) -> f64;

    /// Template method: evaluate the input, then map
    /// each non-null value through `apply`, producing a `Float64` column.
    fn evaluate_unary(&self, input: &RecordBatch) -> Box<dyn ColumnVector> {
        let n = self.input().evaluate(input);
        let mut builder = ArrowVectorBuilder::new(&DOUBLE_TYPE, n.size());
        for i in 0..n.size() {
            let value = n.get_value(i);
            if value.is_null() {
                builder.append_null();
            } else {
                builder.append_value(&ScalarValue::Float64(self.apply(number_to_f64(&value))));
            }
        }
        builder.set_value_count(n.size());
        Box::new(builder.build())
    }
}

/// Square root.
pub struct Sqrt {
    expr: Arc<dyn Expression>,
}

impl Sqrt {
    pub fn new(expr: Arc<dyn Expression>) -> Self {
        Self { expr }
    }
}

impl UnaryMathExpression for Sqrt {
    fn input(&self) -> &Arc<dyn Expression> {
        &self.expr
    }
    fn apply(&self, value: f64) -> f64 {
        value.sqrt()
    }
}

impl Expression for Sqrt {
    fn evaluate(&self, input: &RecordBatch) -> Box<dyn ColumnVector> {
        self.evaluate_unary(input)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl fmt::Display for Sqrt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "sqrt({})", self.expr)
    }
}

/// Natural logarithm.
pub struct Log {
    expr: Arc<dyn Expression>,
}

impl Log {
    pub fn new(expr: Arc<dyn Expression>) -> Self {
        Self { expr }
    }
}

impl UnaryMathExpression for Log {
    fn input(&self) -> &Arc<dyn Expression> {
        &self.expr
    }
    fn apply(&self, value: f64) -> f64 {
        value.ln()
    }
}

impl Expression for Log {
    fn evaluate(&self, input: &RecordBatch) -> Box<dyn ColumnVector> {
        self.evaluate_unary(input)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl fmt::Display for Log {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "log({})", self.expr)
    }
}
