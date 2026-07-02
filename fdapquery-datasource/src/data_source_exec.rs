//! `DataSourceExec` — the source-of-record physical operator for reading
//! a `DataSource`. Mirrors `datafusion_datasource::source::DataSourceExec`.
//!
//! Wraps an `Arc<dyn DataSource>` and dispatches `execute(partition, ctx)`
//! to `DataSource::open(partition, ctx)`. The `Display` / `DisplayAs`
//! output is `"DataSourceExec: "` followed by the inner source's
//! `fmt_as` output, matching DataFusion's format string.

use std::fmt;
use std::sync::Arc;

use fdapquery_common::FdapQueryError;
use fdapquery_datatypes::{Result, Schema};
use fdapquery_execution::TaskContext;
use fdapquery_physical_plan::display::{DisplayAs, DisplayFormatType};
use fdapquery_physical_plan::physical_plan::ExecutionPlan;
use fdapquery_physical_plan::plan_properties::PlanProperties;
use fdapquery_physical_plan::stream::SendableRecordBatchStream;

use crate::data_source::DataSource;

/// The source-of-record physical operator for reading a [`DataSource`].
///
/// Strict mirror of `datafusion_datasource::source::DataSourceExec`.
/// Wraps an `Arc<dyn DataSource>` and delegates `execute(partition, ctx)`
/// to `DataSource::open(partition, ctx)`. The operator's `PlanProperties`
/// is inherited from the inner source's [`DataSource::properties`] at
/// construction time and cached.
#[derive(Debug)]
pub struct DataSourceExec {
    source: Arc<dyn DataSource>,
    properties: PlanProperties,
}

impl DataSourceExec {
    /// Wrap a `DataSource` in a `DataSourceExec`. Inherits the source's
    /// `properties()` as the operator's `PlanProperties`.
    pub fn new(source: Arc<dyn DataSource>) -> Self {
        let properties = source.properties().clone();
        Self { source, properties }
    }

    /// The inner data source.
    pub fn source(&self) -> &Arc<dyn DataSource> {
        &self.source
    }
}

impl DisplayAs for DataSourceExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DataSourceExec: ")?;
        self.source.fmt_as(t, f)
    }
}

impl fmt::Display for DataSourceExec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        <Self as DisplayAs>::fmt_as(self, DisplayFormatType::Default, f)
    }
}

impl ExecutionPlan for DataSourceExec {
    fn name(&self) -> &'static str {
        "DataSourceExec"
    }

    fn schema(&self) -> Schema {
        self.source.schema()
    }

    fn properties(&self) -> &PlanProperties {
        &self.properties
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        self.source.open(partition, context)
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        // A data source is a leaf — no input plans.
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if !children.is_empty() {
            return Err(FdapQueryError::Internal(format!(
                "DataSourceExec is a leaf and expects no children, got {}",
                children.len()
            )));
        }
        Ok(self)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
