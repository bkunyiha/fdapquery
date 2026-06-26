//!
//! Evaluates a list of expressions against each input batch and assembles the
//! results into an output batch with the projection's schema.

use crate::Expression;
use crate::physical_plan::ExecutionPlan;
use crate::plan_properties::PlanProperties;
use crate::stream::{RecordBatchStreamAdapter, SendableRecordBatchStream};
use crate::task_context::TaskContext;
use fdapquery_datatypes::{ColumnVector, FdapQueryError, Result, Schema, record_batch};
use futures::StreamExt;
use std::fmt;
use std::sync::Arc;

/// Execute a projection.
///
/// The output schema is supplied explicitly (the query planner computes it) — a
/// projection can rename or compute columns, so it cannot always be derived from
/// the input.
pub struct ProjectionExec {
    pub input: Arc<dyn ExecutionPlan>,
    pub schema: Schema,
    pub expr: Vec<Arc<dyn Expression>>,
    properties: PlanProperties,
}

impl ProjectionExec {
    pub fn new(
        input: Arc<dyn ExecutionPlan>,
        schema: Schema,
        expr: Vec<Arc<dyn Expression>>,
    ) -> Self {
        let properties = PlanProperties::single_partition_unknown();
        Self {
            input,
            schema,
            expr,
            properties,
        }
    }
}

impl ExecutionPlan for ProjectionExec {
    fn name(&self) -> &str {
        "ProjectionExec"
    }

    fn schema(&self) -> Schema {
        self.schema.clone()
    }

    fn properties(&self) -> &PlanProperties {
        &self.properties
    }

    fn execute(
        &self,
        partition: usize,
        ctx: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        if partition != 0 {
            return Err(FdapQueryError::Internal(format!(
                "ProjectionExec has 1 output partition; partition {partition} is out of range"
            )));
        }
        // Projection just evaluates expressions per batch — no context use; pass
        // through to the input so shuffle-bearing children downstream can find it.
        let input_stream = self.input.execute(0, Arc::clone(&ctx))?;
        let schema = self.schema.clone();
        let exprs = self.expr.clone();
        let projected = input_stream.map(move |batch_res| {
            let batch = batch_res?;
            let columns: Vec<Box<dyn ColumnVector>> = exprs
                .iter()
                .map(|e| e.evaluate(&batch))
                .collect::<Result<Vec<_>>>()?;
            record_batch::create(&schema, columns)
        });
        let arrow_schema = Arc::new(self.schema.to_arrow());
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            arrow_schema,
            projected,
        )))
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    /// Rebuild this projection with a new input child.
    ///
    /// `ProjectionExec` has arity 1 (one input relation), so the incoming
    /// `children` vec always has exactly one element. We consume the vec via
    /// `into_iter().next().unwrap()` to take ownership of that single
    /// `Arc<dyn ExecutionPlan>` without an atomic refcount bump. We reuse
    /// `self.schema` and `self.expr` — they don't depend on which concrete
    /// input feeds this projection.
    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if children.len() != 1 {
            return Err(FdapQueryError::Internal(format!(
                "ProjectionExec::with_new_children expected 1 child, got {}",
                children.len()
            )));
        }
        Ok(Arc::new(ProjectionExec::new(
            children.into_iter().next().unwrap(),
            self.schema.clone(),
            self.expr.clone(),
        )))
    }
}

impl fmt::Display for ProjectionExec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Render as `ProjectionExec: [a, b, c]` — bracketed, comma-separated.
        let exprs: Vec<String> = self.expr.iter().map(|e| e.to_string()).collect();
        write!(f, "ProjectionExec: [{}]", exprs.join(", "))
    }
}
