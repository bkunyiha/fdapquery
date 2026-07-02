//!
//! References a column in the input batch by its position. Evaluating it simply
//! hands back that column unchanged — the simplest possible physical expression.
//!
//! Mirrors DataFusion's `datafusion_physical_expr::expressions::Column`: the
//! struct carries both the source-schema column **name** (debug / display) and
//! its **index** (the only field actually used at evaluation time). The
//! Display format is `{name}@{index}`, byte-for-byte identical to
//! DataFusion's `write!(f, "{}@{}", self.name, self.index)`.

use crate::columnar_value::ColumnarValue;
use crate::expressions::PhysicalExpr;
use arrow_schema::{DataType, Schema};
use fdapquery_datatypes::{RecordBatch, Result};
use std::fmt;

/// Reference a column in a batch by index.
///
/// The `name` is carried for display/debug purposes only — `index` is the
/// authoritative selector used at evaluation time. This mirrors DataFusion's
/// `expressions::Column { name, index }` shape and Display format
/// (`name@index`).
#[derive(Debug)]
pub struct Column {
    /// The name of the column (used for debugging and display purposes).
    pub name: String,
    /// The index of the column in its schema.
    pub index: usize,
}

impl Column {
    /// Create a new column expression referencing column `index` in the schema,
    /// recording `name` for display.
    pub fn new(name: &str, index: usize) -> Self {
        Self {
            name: name.to_owned(),
            index,
        }
    }

    /// Get the column's name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the column's schema index.
    pub fn index(&self) -> usize {
        self.index
    }
}

impl PhysicalExpr for Column {
    fn evaluate(&self, batch: &RecordBatch) -> Result<ColumnarValue> {
        // `batch.column(i)` returns an `&ArrayRef` (= `&Arc<dyn Array>`),
        // which we Arc-clone (cheap — refcount bump) to wrap in
        // `ColumnarValue::Array`.
        Ok(ColumnarValue::Array(batch.column(self.index).clone()))
    }

    /// The column's Arrow type is the type recorded in the input schema's
    /// field at `self.index`. Mirrors DataFusion's
    /// `Column::data_type`:
    ///
    /// ```text
    /// fn data_type(&self, input_schema: &Schema) -> Result<DataType> {
    ///     self.bounds_check(input_schema)?;
    ///     Ok(input_schema.field(self.index).data_type().clone())
    /// }
    /// ```
    ///
    /// fdapquery omits the explicit `bounds_check` helper — `Schema::field`
    /// panics on an out-of-bounds index, matching the same effective
    /// invariant (the planner has resolved indices against this schema).
    fn data_type(&self, input_schema: &Schema) -> Result<DataType> {
        Ok(input_schema.field(self.index).data_type().clone())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl fmt::Display for Column {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}@{}", self.name, self.index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_schema::{Field as ArrowField, Schema as ArrowSchema};

    /// `Column::data_type` reads the field at `self.index` from the input
    /// schema — mirrors DataFusion's
    /// `Ok(input_schema.field(self.index).data_type().clone())`. Verified
    /// across three fields to confirm the right index is used (not
    /// hard-coded to 0) and the right `DataType` variant is returned.
    #[test]
    fn data_type_reads_field_at_self_index() {
        let schema = ArrowSchema::new(vec![
            ArrowField::new("a", DataType::Int32, true),
            ArrowField::new("b", DataType::Utf8, true),
            ArrowField::new("c", DataType::Float64, false),
        ]);
        assert_eq!(
            Column::new("a", 0).data_type(&schema).unwrap(),
            DataType::Int32
        );
        assert_eq!(
            Column::new("b", 1).data_type(&schema).unwrap(),
            DataType::Utf8
        );
        assert_eq!(
            Column::new("c", 2).data_type(&schema).unwrap(),
            DataType::Float64
        );
    }
}
