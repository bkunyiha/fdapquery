//!
//! Scans a data source with an optional push-down projection. It is the only
//! leaf operator in the query tree: it has no child plan and produces its
//! batches by delegating to the [`DataSource`] (which the optimiser's
//! `ProjectionPushDownRule` has already trimmed to just the columns the
//! query needs).
//!
//! ## Pilot for the Phase B trait shape
//! Session 8 (Phase B opens) uses `ScanExec` as the pilot for the new
//! `ExecutionPlan` trait — it's the simplest operator (no input plan, no
//! cross-batch state), so the conversion from sync `Iterator` to async
//! `SendableRecordBatchStream` is one line: wrap the sync iterator
//! `DataSource::scan` returns with `futures::stream::iter` and adapt it
//! with `RecordBatchStreamAdapter`.

use crate::physical_plan::ExecutionPlan;
use crate::plan_properties::PlanProperties;
use crate::stream::{RecordBatchStreamAdapter, SendableRecordBatchStream};
use fdapquery_catalog::TableProvider;
use fdapquery_datatypes::{FdapQueryError, Result, Schema};
use fdapquery_execution::TaskContext;
use std::fmt;
use std::sync::Arc;

/// TableScan a table provider with optional push-down projection.
///
/// `provider` is held as `Arc<dyn TableProvider>` (matching the logical
/// `TableScan` operator), so the same source can be shared across plan
/// nodes. The output schema is computed once at construction
/// (`Schema::select` over the projection) and cached — matching
/// DataFusion's `ExecutionPlan::schema(&self) -> SchemaRef` shape,
/// where schema is infallible because it's known at plan-build time.
///
/// `properties` is also computed at construction (single output
/// partition, unknown distribution) and cached for the same reason.
pub struct ScanExec {
    pub provider: Arc<dyn TableProvider>,
    pub projection: Vec<String>,
    pub schema: Schema,
    properties: PlanProperties,
}

impl ScanExec {
    /// Build a `ScanExec`, validating the projection against the
    /// provider's schema. An invalid projection (a column name not
    /// present in the source) surfaces as `Err(SchemaError(_))`.
    pub fn new(provider: Arc<dyn TableProvider>, projection: Vec<String>) -> Result<Self> {
        let source_schema = provider.schema();
        let schema = if projection.is_empty() {
            source_schema
        } else {
            let indices: Vec<usize> = projection
                .iter()
                .map(|name| {
                    source_schema
                        .fields()
                        .iter()
                        .position(|f| f.name() == name)
                        .ok_or_else(|| {
                            FdapQueryError::SchemaError(format!(
                                "ScanExec: projection column '{name}' not in source schema"
                            ))
                        })
                })
                .collect::<Result<Vec<usize>>>()?;
            source_schema.project(&indices)?
        };
        let properties = PlanProperties::single_partition_unknown();
        Ok(Self {
            provider,
            projection,
            schema,
            properties,
        })
    }
}

impl ExecutionPlan for ScanExec {
    fn name(&self) -> &str {
        "ScanExec"
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
        _ctx: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        // A leaf scan has one output partition. Any other partition index
        // is a planner bug.
        if partition != 0 {
            return Err(FdapQueryError::Internal(format!(
                "ScanExec has 1 output partition; partition {partition} is out of range"
            )));
        }
        // `TableProvider::scan` returns a `SendableRecordBatchStream`
        // (a pin-boxed async `Stream` over `Result<RecordBatch>`).
        // Wrap it with the projected schema via `RecordBatchStreamAdapter`
        // so it satisfies the `RecordBatchStream` contract that
        // `SendableRecordBatchStream` aliases.
        let stream = self.provider.scan(&self.projection)?;
        let arrow_schema = Arc::new(self.schema.clone());
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            arrow_schema,
            stream,
        )))
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        // A scan is a leaf — no inputs.
        vec![]
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    /// Rebuild this scan with new children. A scan is a leaf, so the
    /// incoming `children` vec is always empty and we hand `self` back
    /// unchanged — no new allocation, just keep the existing Arc.
    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use fdapquery_catalog::CsvDataSource;
    use futures::TryStreamExt;

    /// Drive `ScanExec::execute` over the shared employee.csv fixture and
    /// verify the row count matches the file (4 rows).
    #[tokio::test]
    async fn scan_yields_all_rows_via_async_stream() {
        let ds: Arc<dyn TableProvider> = Arc::new(CsvDataSource::new(
            "../testdata/employee.csv",
            None,
            true,
            1024,
        ));
        let columns: Vec<String> = ds
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect();
        let scan = ScanExec::new(Arc::clone(&ds), columns).unwrap();

        let ctx = Arc::new(TaskContext::default_test());
        let stream = scan.execute(0, ctx).unwrap();
        let batches = stream.try_collect::<Vec<_>>().await.unwrap();

        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 4);
    }

    /// Out-of-range partition is an Internal error, not a panic.
    #[tokio::test]
    async fn scan_rejects_non_zero_partition() {
        let ds: Arc<dyn TableProvider> = Arc::new(CsvDataSource::new(
            "../testdata/employee.csv",
            None,
            true,
            1024,
        ));
        let columns: Vec<String> = ds
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect();
        let scan = ScanExec::new(Arc::clone(&ds), columns).unwrap();

        let ctx = Arc::new(TaskContext::default_test());
        let err = scan.execute(1, ctx).map(|_| ()).unwrap_err();
        assert!(matches!(err, FdapQueryError::Internal(_)));
    }

    /// `properties()` reports the cached single-partition descriptor.
    #[test]
    fn properties_returns_cached_single_partition() {
        let ds: Arc<dyn TableProvider> = Arc::new(CsvDataSource::new(
            "../testdata/employee.csv",
            None,
            true,
            1024,
        ));
        let columns: Vec<String> = ds
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect();
        let scan = ScanExec::new(Arc::clone(&ds), columns).unwrap();

        assert_eq!(scan.properties().partition_count(), 1);
    }

    /// `with_new_children` rejects any non-empty child set.
    #[test]
    fn with_new_children_rejects_non_empty() {
        let ds: Arc<dyn TableProvider> = Arc::new(CsvDataSource::new(
            "../testdata/employee.csv",
            None,
            true,
            1024,
        ));
        let columns: Vec<String> = ds
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect();
        let scan = Arc::new(ScanExec::new(Arc::clone(&ds), columns).unwrap());

        // A "child" — well, just another ScanExec — to force a non-empty vec.
        let inner: Arc<dyn ExecutionPlan> = scan.clone();
        let err = Arc::clone(&scan)
            .with_new_children(vec![inner])
            .map(|_| ())
            .unwrap_err();
        assert!(matches!(err, FdapQueryError::Internal(_)));
    }
}
