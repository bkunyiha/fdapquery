//! Test-only `ExecutionPlan` helpers internal to `fdapquery-physical-plan`.
//!
//! The crate cannot reach `fdapquery-catalog` (there is no catalog ↔
//! physical-plan dep edge), so operator tests that would otherwise
//! drive `DataSourceExec` over `CsvDataSource` use [`TestSourceExec`], a
//! minimal in-crate `ExecutionPlan` that replays a pre-loaded
//! `Vec<RecordBatch>` as an async stream.
//!
//! Same shape as [`crate::MemoryExec`] and DataFusion's `MemoryExec`;
//! kept local to the test cfg so the production `physical_plan` re-exports
//! stay narrow. (`TestSourceExec` predates `MemoryExec`; call sites are
//! being flipped to `MemoryExec` incrementally.)

use crate::display::{DisplayAs, DisplayFormatType};
use crate::physical_plan::ExecutionPlan;
use crate::plan_properties::PlanProperties;
use crate::stream::{RecordBatchStreamAdapter, SendableRecordBatchStream};
use fdapquery_datatypes::{FdapQueryError, RecordBatch, Result, Schema};
use fdapquery_execution::TaskContext;
use std::fmt;
use std::sync::Arc;

/// Minimal leaf operator over a pre-loaded `Vec<RecordBatch>`.
///
/// Cached `schema` and `PlanProperties` at construction (single-partition,
/// unknown distribution). `execute(0, _)` wraps the batches in
/// `futures::stream::iter` and produces a schema-aware
/// `RecordBatchStreamAdapter`. Any partition other than 0 surfaces as
/// `FdapQueryError::Internal` — same shape every other leaf operator in
/// the crate uses.
#[derive(Debug)]
pub struct TestSourceExec {
    schema: Schema,
    batches: Vec<RecordBatch>,
    properties: PlanProperties,
}

impl TestSourceExec {
    pub fn new(schema: Schema, batches: Vec<RecordBatch>) -> Self {
        let properties = PlanProperties::single_partition_unknown();
        Self {
            schema,
            batches,
            properties,
        }
    }
}

impl DisplayAs for TestSourceExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TestSourceExec: batches={}", self.batches.len())
    }
}

impl fmt::Display for TestSourceExec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        <Self as DisplayAs>::fmt_as(self, DisplayFormatType::Default, f)
    }
}

impl ExecutionPlan for TestSourceExec {
    fn name(&self) -> &'static str {
        "TestSourceExec"
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
        if partition != 0 {
            return Err(FdapQueryError::Internal(format!(
                "TestSourceExec has 1 output partition; partition {partition} is out of range"
            )));
        }
        let arrow_schema = Arc::new(self.schema.clone());
        let stream = futures::stream::iter(self.batches.clone().into_iter().map(Ok));
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            arrow_schema,
            stream,
        )))
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if !children.is_empty() {
            return Err(FdapQueryError::Internal(format!(
                "TestSourceExec is a leaf and expects no children, got {}",
                children.len()
            )));
        }
        Ok(self)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

// =============================================================================
// employee.csv fixture, replayed in-memory.
// =============================================================================

/// Schema of `testdata/employee.csv`. Inferred once from the CSV header
/// and pinned here so the physical-plan crate can produce employee
/// batches without depending on `fdapquery-catalog`.
///
/// Columns: `id: Int64, first_name: Utf8, last_name: Utf8, state: Utf8,
/// job_title: Utf8, salary: Int64`.
pub fn employee_schema() -> Schema {
    use arrow_schema::DataType;
    use fdapquery_datatypes::Field;
    Schema::new(vec![
        Field::new("id", DataType::Int64, true),
        Field::new("first_name", DataType::Utf8, true),
        Field::new("last_name", DataType::Utf8, true),
        Field::new("state", DataType::Utf8, true),
        Field::new("job_title", DataType::Utf8, true),
        Field::new("salary", DataType::Int64, true),
    ])
}

/// A single-batch fixture mirroring the 4-row contents of
/// `testdata/employee.csv`. Same data, no file I/O — lets the
/// physical-plan operator tests assert end-to-end row/value behaviour
/// without depending on the catalog crate.
pub fn employee_batches() -> Vec<RecordBatch> {
    use arrow_array::{ArrayRef, Int64Array, StringArray};
    // Mirrors testdata/employee.csv exactly: 4 rows; row 4 has a NULL
    // state (the source CSV has an empty field there, which arrow's
    // CSV reader infers as Null over a nullable column). The salary
    // sequence matches the file: 12000, 10000, 11500, 11500 — so the
    // CO group spans rows 2 and 3 (min=10000, max=11500, count=2) and
    // the null-state group is row 4 alone (min=max=11500, count=1).
    let id: ArrayRef = Arc::new(Int64Array::from(vec![1, 2, 3, 4]));
    let first_name: ArrayRef = Arc::new(StringArray::from(vec!["Bill", "Gregg", "John", "Von"]));
    let last_name: ArrayRef = Arc::new(StringArray::from(vec![
        "Hopkins", "Langford", "Travis", "Mill",
    ]));
    // Row 4 (index 3) is None — matches the empty CSV field.
    let state: ArrayRef = Arc::new(StringArray::from(vec![
        Some("CA"),
        Some("CO"),
        Some("CO"),
        None,
    ]));
    let job_title: ArrayRef = Arc::new(StringArray::from(vec![
        "Manager",
        "Driver",
        "Manager, Software",
        "Defensive End",
    ]));
    let salary: ArrayRef = Arc::new(Int64Array::from(vec![12000, 10000, 11500, 11500]));
    let schema = Arc::new(employee_schema());
    let batch = RecordBatch::try_new(
        schema,
        vec![id, first_name, last_name, state, job_title, salary],
    )
    .expect("employee_batches: RecordBatch::try_new");
    vec![batch]
}

/// Convenience: construct an `Arc<dyn ExecutionPlan>` over the
/// employee fixture. Sugar for `Arc::new(TestSourceExec::new(...))`.
pub fn employee_source() -> Arc<dyn ExecutionPlan> {
    Arc::new(TestSourceExec::new(employee_schema(), employee_batches()))
}
