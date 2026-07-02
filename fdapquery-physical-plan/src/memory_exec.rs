//! Leaf `ExecutionPlan` that replays pre-loaded `Vec<RecordBatch>` batches.
//!
//! Strict mirror of `datafusion::physical_plan::memory::MemoryExec` (now
//! superseded upstream by `MemorySourceConfig` in `datafusion-datasource`
//! — fdapquery keeps the historic `MemoryExec` name, matching the
//! v0.1-target DataFusion surface).
//!
//! At construction it caches its output schema and a
//! single-partition/unknown-distribution `PlanProperties`. `execute(0, _)`
//! hands the buffered batches off through a
//! [`fdapquery_execution::MemoryStream`], which yields each batch in
//! order and honours the (optional) column projection.
//!
//! This type replaces the private `InMemoryPlan` that used to live in
//! `fdapquery::parallel_context`. The public surface is the same: a
//! leaf plan built from `(schema, Vec<RecordBatch>)`. `MemoryExec`
//! additionally exposes DataFusion's `try_new(data, schema, projection)`
//! constructor shape.

use crate::display::{DisplayAs, DisplayFormatType};
use crate::physical_plan::ExecutionPlan;
use crate::plan_properties::PlanProperties;
use crate::stream::{MemoryStream, SendableRecordBatchStream};
use fdapquery_datatypes::{FdapQueryError, RecordBatch, Result, Schema, SchemaRef};
use fdapquery_execution::TaskContext;
use std::fmt;
use std::sync::Arc;

/// Leaf physical plan over pre-loaded batches. Mirrors DataFusion's
/// (historic) `MemoryExec`: caches its output schema and
/// `PlanProperties` at construction, and on `execute` returns a
/// [`MemoryStream`] over the buffered `Vec<RecordBatch>`.
///
/// A `projection` argument (`Option<Vec<usize>>`) narrows the yielded
/// columns to the requested indices; the reported schema is the
/// projected schema when a projection is set.
#[derive(Debug)]
pub struct MemoryExec {
    /// Output schema after projection (if any).
    schema: Schema,
    /// The buffered input batches, in the yield order used by `execute`.
    batches: Vec<RecordBatch>,
    /// Optional column-index projection applied on every yielded batch.
    projection: Option<Vec<usize>>,
    /// Cached plan properties (single partition, unknown distribution).
    properties: PlanProperties,
}

impl MemoryExec {
    /// Construct a `MemoryExec` from an already-materialised
    /// `Vec<RecordBatch>` plus the schema those batches share.
    ///
    /// Kept for API continuity with the private `InMemoryPlan` this
    /// replaces (same two-argument shape). Callers wanting a projection
    /// use [`Self::try_new`].
    pub fn new(schema: Schema, batches: Vec<RecordBatch>) -> Self {
        Self {
            schema,
            batches,
            projection: None,
            properties: PlanProperties::single_partition_unknown(),
        }
    }

    /// DataFusion-shape constructor: `try_new(data, schema, projection)`.
    ///
    /// `schema` is the *output* schema — if `projection` is `Some(_)`,
    /// pass the already-projected schema so it matches what the yielded
    /// `RecordBatch`es will carry.
    pub fn try_new(
        data: Vec<RecordBatch>,
        schema: Schema,
        projection: Option<Vec<usize>>,
    ) -> Result<Self> {
        // Validate projection against the first batch (if any) so
        // out-of-bounds indices surface at construction rather than
        // first poll.
        if let (Some(indices), Some(first)) = (projection.as_ref(), data.first()) {
            let ncols = first.num_columns();
            for &i in indices {
                if i >= ncols {
                    return Err(FdapQueryError::ArrowError(
                        arrow_schema::ArrowError::SchemaError(format!(
                            "MemoryExec projection index {i} out of bounds for {ncols} columns"
                        )),
                    ));
                }
            }
        }
        Ok(Self {
            schema,
            batches: data,
            projection,
            properties: PlanProperties::single_partition_unknown(),
        })
    }
}

impl DisplayAs for MemoryExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MemoryExec: batches={}", self.batches.len())
    }
}

impl fmt::Display for MemoryExec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        <Self as DisplayAs>::fmt_as(self, DisplayFormatType::Default, f)
    }
}

impl ExecutionPlan for MemoryExec {
    fn name(&self) -> &'static str {
        "MemoryExec"
    }

    fn schema(&self) -> Schema {
        self.schema.clone()
    }

    fn properties(&self) -> &PlanProperties {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        Vec::new()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn execute(
        &self,
        partition: usize,
        _ctx: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        if partition != 0 {
            return Err(FdapQueryError::Internal(format!(
                "MemoryExec has 1 output partition; partition {partition} is out of range"
            )));
        }
        // arrow `RecordBatch` is `Arc`-backed, so cloning the vec is cheap.
        let schema_ref: SchemaRef = Arc::new(self.schema.clone());
        let ms = MemoryStream::try_new(
            self.batches.clone(),
            schema_ref,
            self.projection.clone(),
        )?;
        Ok(Box::pin(ms))
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if !children.is_empty() {
            return Err(FdapQueryError::Internal(format!(
                "MemoryExec is a leaf and expects no children, got {}",
                children.len()
            )));
        }
        Ok(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Int32Array, RecordBatch as ArrowBatch};
    use arrow_schema::{DataType, Field};
    use futures::TryStreamExt;

    fn ctx() -> Arc<TaskContext> {
        Arc::new(TaskContext::default_test())
    }

    fn one_col_schema() -> Schema {
        Schema::new(vec![Field::new("a", DataType::Int32, true)])
    }

    fn batch(vals: &[i32], schema: &Schema) -> RecordBatch {
        ArrowBatch::try_new(
            Arc::new(schema.clone()),
            vec![Arc::new(Int32Array::from(vals.to_vec()))],
        )
        .unwrap()
    }

    #[tokio::test]
    async fn memory_exec_execute_yields_batches_in_order() {
        let schema = one_col_schema();
        let b1 = batch(&[1, 2], &schema);
        let b2 = batch(&[3, 4], &schema);
        let plan = MemoryExec::new(schema, vec![b1.clone(), b2.clone()]);
        let stream = plan.execute(0, ctx()).unwrap();
        let collected: Vec<RecordBatch> = stream.try_collect().await.unwrap();
        assert_eq!(collected.len(), 2);
        assert_eq!(collected[0], b1);
        assert_eq!(collected[1], b2);
    }

    #[tokio::test]
    async fn memory_exec_rejects_nonzero_partition() {
        let schema = one_col_schema();
        let plan = MemoryExec::new(schema.clone(), vec![batch(&[1], &schema)]);
        let err = plan.execute(1, ctx()).err().unwrap();
        assert!(format!("{err}").contains("out of range"));
    }

    #[test]
    fn memory_exec_with_new_children_rejects_non_empty() {
        let schema = one_col_schema();
        let plan: Arc<MemoryExec> = Arc::new(MemoryExec::new(schema.clone(), vec![]));
        // Empty children — should succeed.
        assert!(Arc::clone(&plan).with_new_children(vec![]).is_ok());
        // Non-empty children — should error.
        let child: Arc<dyn ExecutionPlan> =
            Arc::new(MemoryExec::new(schema, vec![]));
        assert!(plan.with_new_children(vec![child]).is_err());
    }
}
