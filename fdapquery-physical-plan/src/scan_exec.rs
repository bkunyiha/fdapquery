//!
//! Scans a data source with an optional push-down projection. It is the only leaf
//! operator: it has no child plan and produces its batches by delegating to the
//! `DataSource` (which the optimizer's `ProjectionPushDownRule` has already
//! trimmed to just the columns the query needs).

use crate::executor_context::ExecutorContext;
use crate::physical_plan::PhysicalPlan;
use datasource::DataSource;
use datatypes::{RecordBatch, Schema};
use std::fmt;
use std::sync::Arc;

/// Scan a data source with optional push-down projection.
///
/// `ds` is held as `Arc<dyn DataSource>` (matching the logical `Scan` operator),
/// so the same source can be shared across plan nodes.
pub struct ScanExec {
    pub ds: Arc<dyn DataSource>,
    pub projection: Vec<String>,
}

impl ScanExec {
    pub fn new(ds: Arc<dyn DataSource>, projection: Vec<String>) -> Self {
        Self { ds, projection }
    }
}

impl PhysicalPlan for ScanExec {
    fn schema(&self) -> Schema {
        self.ds.schema().select(&self.projection)
    }

    fn execute(&self, _ctx: &ExecutorContext) -> Box<dyn Iterator<Item = RecordBatch>> {
        // A leaf scan needs no executor context — the `DataSource` reads from
        // its own configured location (CSV path / Parquet path). `_ctx` is
        // present in the signature only so the trait contract is uniform.
        self.ds.scan(&self.projection)
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
    ) -> Arc<dyn PhysicalPlan> {
        assert!(
            children.is_empty(),
            "ScanExec is a leaf and expects no children"
        );
        self
    }
}

impl fmt::Display for ScanExec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // The datatypes `Schema` has no `Display`, so use its `Debug` form.
        write!(
            f,
            "ScanExec: schema={:?}, projection={:?}",
            self.schema(),
            self.projection
        )
    }
}
