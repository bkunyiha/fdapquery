//!
//! Unary math functions over a numeric column, producing a `Float64` column:
//! `Sqrt` and `Log` (natural log).
//!
//! ## Trait with a default method
//! As with the binary/boolean/math families, the shared "evaluate input, map
//! each non-null value through `apply`" logic lives in a trait
//! ([`UnaryMathExpr`]) with a default method (`evaluate_unary`) and a
//! required `apply` kernel. Each concrete function implements
//! `UnaryMathExpr` and a one-line `PhysicalExpr` delegate.

use crate::columnar_value::ColumnarValue;
use crate::expressions::{PhysicalExpr, number_to_f64};
use arrow_schema::{DataType, Schema};
use fdapquery_common::{ArrowVectorBuilder, Result, ScalarValue};
use fdapquery_datatypes::{RecordBatch, record_batch};
use std::fmt;
use std::sync::Arc;

/// A unary math function.
pub trait UnaryMathExpr: PhysicalExpr {
    /// The input expression whose values are transformed.
    fn input(&self) -> &Arc<dyn PhysicalExpr>;

    /// The function applied to each non-null value.
    fn apply(&self, value: f64) -> f64;

    /// Template method: evaluate the input, then map
    /// each non-null value through `apply`, producing a `Float64` column.
    fn evaluate_unary(&self, batch: &RecordBatch) -> Result<ColumnarValue> {
        let num_rows = record_batch::row_count(batch);
        let n = self.input().evaluate(batch)?.into_array(num_rows)?;
        let mut builder = ArrowVectorBuilder::new(&arrow_schema::DataType::Float64, n.len());
        for i in 0..n.len() {
            let value = ScalarValue::try_from_array(&n, i)?;
            if value.is_null() {
                builder.append_null();
            } else {
                builder.append_value(&ScalarValue::Float64(self.apply(number_to_f64(&value)?)));
            }
        }
        Ok(ColumnarValue::Array(builder.build()))
    }
}

/// Square root.
#[derive(Debug)]
pub struct Sqrt {
    expr: Arc<dyn PhysicalExpr>,
}

impl Sqrt {
    pub fn new(expr: Arc<dyn PhysicalExpr>) -> Self {
        Self { expr }
    }
}

impl UnaryMathExpr for Sqrt {
    fn input(&self) -> &Arc<dyn PhysicalExpr> {
        &self.expr
    }
    fn apply(&self, value: f64) -> f64 {
        value.sqrt()
    }
}

impl PhysicalExpr for Sqrt {
    fn evaluate(&self, batch: &RecordBatch) -> Result<ColumnarValue> {
        self.evaluate_unary(batch)
    }

    /// `sqrt(x)` is always evaluated in `Float64` — the runtime builds a
    /// `Float64` result column in [`UnaryMathExpr::evaluate_unary`].
    /// DataFusion exposes `sqrt` as a scalar UDF whose declared
    /// `return_type` is also `Float64`.
    fn data_type(&self, _input_schema: &Schema) -> Result<DataType> {
        Ok(DataType::Float64)
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
#[derive(Debug)]
pub struct Log {
    expr: Arc<dyn PhysicalExpr>,
}

impl Log {
    pub fn new(expr: Arc<dyn PhysicalExpr>) -> Self {
        Self { expr }
    }
}

impl UnaryMathExpr for Log {
    fn input(&self) -> &Arc<dyn PhysicalExpr> {
        &self.expr
    }
    fn apply(&self, value: f64) -> f64 {
        value.ln()
    }
}

impl PhysicalExpr for Log {
    fn evaluate(&self, batch: &RecordBatch) -> Result<ColumnarValue> {
        self.evaluate_unary(batch)
    }

    /// `ln(x)` is always evaluated in `Float64` — the runtime builds a
    /// `Float64` result column in [`UnaryMathExpr::evaluate_unary`].
    /// DataFusion's `ln` scalar UDF declares `return_type` `Float64`.
    fn data_type(&self, _input_schema: &Schema) -> Result<DataType> {
        Ok(DataType::Float64)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expressions::Literal;
    use fdapquery_common::ScalarValue;

    /// `Sqrt` and `Log` always produce a `Float64` column —
    /// `data_type` returns `Float64` regardless of input schema or inner
    /// expression's type. Matches DataFusion's `sqrt`/`ln` UDFs.
    #[test]
    fn data_type_is_float64() {
        let schema = arrow_schema::Schema::empty();
        let inner = Arc::new(Literal::new(ScalarValue::Int64(4))) as Arc<dyn PhysicalExpr>;

        let sqrt = Sqrt::new(inner.clone());
        assert_eq!(sqrt.data_type(&schema).unwrap(), DataType::Float64);

        let log = Log::new(inner);
        assert_eq!(log.data_type(&schema).unwrap(), DataType::Float64);
    }
}
