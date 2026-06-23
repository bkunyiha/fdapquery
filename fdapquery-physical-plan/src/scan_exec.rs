//!
//! Scans a data source with an optional push-down projection. It is the only leaf
//! operator: it has no child plan and produces its batches by delegating to the
//! `DataSource` (which the optimizer's `ProjectionPushDownRule` has already
//! trimmed to just the columns the query needs).

use crate::executor_context::ExecutorContext;
use crate::physical_plan::PhysicalPlan;
use fdapquery_datasource::DataSource;
use fdapquery_datatypes::{FdapQueryError, RecordBatch, Result, Schema};
use std::fmt;
use std::sync::Arc;

/// Scan a data source with optional push-down projection.
///
/// `ds` is held as `Arc<dyn DataSource>` (matching the logical `Scan` operator),
/// so the same source can be shared across plan nodes. The output schema is
/// computed once at construction (`Schema::select` over the projection) and
/// cached — matching DataFusion's `ExecutionPlan::schema(&self) -> SchemaRef`
/// shape, where schema is infallible because it's known at plan-build time.
pub struct ScanExec {
    pub ds: Arc<dyn DataSource>,
    pub projection: Vec<String>,
    pub schema: Schema,
}

impl ScanExec {
    /// Build a `ScanExec`, validating the projection against the data-source
    /// schema. Invalid projection (a column name not present in the source)
    /// surfaces as `Err(SchemaError(_))`.
    pub fn new(ds: Arc<dyn DataSource>, projection: Vec<String>) -> Result<Self> {
        let schema = ds.schema().select(&projection)?;
        Ok(Self {
            ds,
            projection,
            schema,
        })
    }
}

impl PhysicalPlan for ScanExec {
    fn schema(&self) -> Schema {
        self.schema.clone()
    }

    fn execute(
        &self,
        _ctx: &ExecutorContext,
    ) -> Result<Box<dyn Iterator<Item = Result<RecordBatch>>>> {
        // A leaf scan needs no executor context — the `DataSource` reads from
        // its own configured location (CSV path / Parquet path). `_ctx` is
        // present in the signature only so the trait contract is uniform.
        let iter = self.ds.scan(&self.projection)?;
        Ok(Box::new(iter))
    }

    fn children(&self) -> Vec<&Arc<dyn PhysicalPlan>> {
        // A scan is a leaf — no inputs.
        vec![]
    }

    /// See the `PhysicalPlan::as_any` docstring for the rationale.
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    /// Rebuild this scan with new children. See the trait-level
    /// `PhysicalPlan::with_new_children` doc for the general rewrite pattern.
    ///
    /// Arity 0 (leaf): a scan has no input — it reads directly from a
    /// `DataSource`. The incoming `children` vec is always empty, so there's
    /// nothing to substitute. We hand back `self` unchanged (it's already an
    /// `Arc<Self>`, which is exactly the return type). No new allocation
    /// happens — the refcount just stays where it was.
    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn PhysicalPlan>>,
    ) -> Result<Arc<dyn PhysicalPlan>> {
        if !children.is_empty() {
            return Err(FdapQueryError::Internal(format!(
                "ScanExec is a leaf and expects no children, got {}",
                children.len()
            )));
        }
        Ok(self)
    }
}

impl fmt::Display for ScanExec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // The datatypes `Schema` has no `Display`, so use its `Debug` form.
        write!(
            f,
            "ScanExec: schema={:?}, projection={:?}",
            self.schema, self.projection
        )
    }
}
