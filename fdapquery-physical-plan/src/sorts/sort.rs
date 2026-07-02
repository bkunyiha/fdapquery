//! `SortExec` — sorts an input stream by a list of `PhysicalSortExpr`
//! keys, optionally truncating to a `fetch` budget (top-K).
//!
//! Strict mirror of DataFusion's
//! `datafusion::physical_plan::sorts::sort::SortExec`. Field set,
//! constructor signature, `DisplayAs::fmt_as` output, and `ExecutionPlan`
//! impl shape match DataFusion byte-for-byte at the **in-memory sort
//! path** — the path DataFusion takes when no spilling is required.
//!
//! ## Documented deferrals from DataFusion's `SortExec`
//!
//! The fields below are present in DataFusion's `SortExec` but elided in
//! fdapquery v0.1. Each elision is a deferred port, not an intentional
//! design divergence:
//!
//! - `metrics_set: ExecutionPlanMetricsSet` — included. fdapquery's
//!   `ExecutionPlanMetricsSet` is the same shape as DataFusion's, so the
//!   field carries over directly.
//! - `common_sort_prefix: Vec<PhysicalSortExpr>` — elided. Populated by
//!   `equivalence_properties().extract_common_sort_prefix(...)` which
//!   requires the equivalence-property machinery
//!   (`fdapquery-physical-plan/src/equivalence/`, not yet ported). Used
//!   by `DisplayAs` to print `sort_prefix=[...]` after `expr=[...]` when
//!   the input is partially sorted, and by `TopK::try_new` to dedupe the
//!   filter. fdapquery's `DisplayAs` always renders an empty prefix.
//! - `cache: Arc<PlanProperties>` — included as a plain `PlanProperties`
//!   field (fdapquery doesn't yet wrap `PlanProperties` in `Arc` because
//!   the equivalence-property machinery that motivates the `Arc` is not
//!   yet ported; see the `cache` field on DataFusion's
//!   `datafusion::physical_plan::sorts::sort::SortExec`).
//! - `filter: Option<Arc<RwLock<TopKDynamicFilters>>>` — elided. This is
//!   the TopK dynamic-filter-pushdown plumbing
//!   (`datafusion/physical-plan/src/topk/mod.rs`); useless without the
//!   `gather_filters_for_pushdown` / `handle_child_pushdown_result`
//!   path, which itself depends on `FilterPushdownPhase` and
//!   `DynamicFilterPhysicalExpr` (not yet ported).
//!
//! ## Documented deferrals from DataFusion's `execute` body
//!
//! DataFusion's `execute` dispatches on `(sort_satisfied, fetch)` to one
//! of four code paths:
//!
//! 1. `(true, Some)` — input is already sorted, slice with `LimitStream`.
//! 2. `(true, None)` — input is already sorted, pass through.
//! 3. `(false, Some)` — TopK heap (`crate::topk::TopK`).
//! 4. `(false, None)` — `ExternalSorter` (in-memory + spill).
//!
//! fdapquery v0.1 always takes the `(false, ...)` arm because
//! `equivalence_properties().ordering_satisfy(...)` is not yet wired up.
//! Within that arm, fdapquery uses a single **in-memory** path: gather
//! all input batches, concat, `lexsort_to_indices`, `take_record_batch`,
//! truncate by `fetch` if set. This matches DataFusion's `ExternalSorter`
//! behaviour **when the input fits in memory** — DataFusion only spills
//! when `MemoryReservation::try_grow` fails. The follow-up task that adds
//! `RuntimeEnv::disk_manager` + `MemoryPool` + `arrow-row` is what gates
//! the external-sort port.
//!
//! ## Documented deferrals from DataFusion's `ExecutionPlan` impl
//!
//! - `required_input_distribution`, `benefits_from_input_partitioning`,
//!   `metrics`, `statistics_with_args`, `cardinality_effect`,
//!   `try_swapping_with_projection`, `gather_filters_for_pushdown`,
//!   `handle_child_pushdown_result`, `reset_state` — all elided. None of
//!   them are in fdapquery's `ExecutionPlan` trait yet (see
//!   `fdapquery_physical_plan::ExecutionPlan` trait). Adding any
//!   one is a follow-up that touches the trait first.

use crate::display::{DisplayAs, DisplayFormatType};
use crate::metrics::ExecutionPlanMetricsSet;
use crate::physical_plan::ExecutionPlan;
use crate::plan_properties::PlanProperties;
use crate::stream::{RecordBatchStreamAdapter, SendableRecordBatchStream};
use arrow::compute::{concat_batches, lexsort_to_indices, take_record_batch};
use arrow_array::Array;
use fdapquery_datatypes::{FdapQueryError, Result, Schema};
use fdapquery_execution::TaskContext;
use fdapquery_physical_expr::LexOrdering;
use futures::TryStreamExt;
use std::any::Any;
use std::fmt;
use std::sync::Arc;

/// Sort the input by a list of expressions, optionally truncating to
/// `fetch` rows (top-K). Strict mirror of DataFusion's
/// `datafusion::physical_plan::sorts::sort::SortExec`.
///
/// DataFusion source-of-truth (`datafusion::physical_plan::sorts::sort::SortExec`):
///
/// ```text
/// pub struct SortExec {
///     pub(crate) input: Arc<dyn ExecutionPlan>,
///     expr: LexOrdering,
///     metrics_set: ExecutionPlanMetricsSet,
///     preserve_partitioning: bool,
///     fetch: Option<usize>,
///     common_sort_prefix: Vec<PhysicalSortExpr>,
///     cache: Arc<PlanProperties>,
///     filter: Option<Arc<RwLock<TopKDynamicFilters>>>,
/// }
/// ```
///
/// The fdapquery v0.1 layout below carries every field that does not
/// depend on an as-yet-unported substrate (see module docs for the
/// deferral list).
#[derive(Debug, Clone)]
pub struct SortExec {
    /// The input plan whose output this `SortExec` sorts.
    pub(crate) input: Arc<dyn ExecutionPlan>,
    /// Sort expressions (lex order).
    expr: LexOrdering,
    /// Per-operator metrics.
    metrics_set: ExecutionPlanMetricsSet,
    /// If false, this `SortExec` sorts and merges all input partitions
    /// into a single sorted output partition. If true, each input
    /// partition is sorted independently.
    preserve_partitioning: bool,
    /// If `Some(n)`, only the first `n` rows of the sorted output are
    /// emitted (top-K). If `None`, the full sorted stream is emitted.
    fetch: Option<usize>,
    /// Cache of static plan properties (output partitioning, etc.).
    cache: PlanProperties,
}

impl SortExec {
    /// Create a new sort execution plan.
    ///
    /// Mirrors DataFusion's `SortExec::new(expr: LexOrdering, input:
    /// Arc<dyn ExecutionPlan>) -> Self` (`datafusion::physical_plan::sorts::sort::SortExec::new`) — including
    /// the **argument order** (`expr` first, then `input`). The default
    /// is `preserve_partitioning = false`, `fetch = None`, and the
    /// output partitioning is `UnknownPartitioning(1)` (single-partition
    /// merged output).
    pub fn new(expr: LexOrdering, input: Arc<dyn ExecutionPlan>) -> Self {
        let preserve_partitioning = false;
        let cache = Self::compute_properties(&input, preserve_partitioning);
        Self {
            input,
            expr,
            metrics_set: ExecutionPlanMetricsSet::new(),
            preserve_partitioning,
            fetch: None,
            cache,
        }
    }

    /// Whether this `SortExec` preserves partitioning of the children.
    /// Mirrors DataFusion `SortExec::preserve_partitioning`.
    pub fn preserve_partitioning(&self) -> bool {
        self.preserve_partitioning
    }

    /// Specify the partitioning behaviour of this sort exec.
    ///
    /// If `preserve_partitioning` is true, sorts each partition
    /// individually, producing one sorted stream per input partition. If
    /// false, sorts and merges all input partitions into a single sorted
    /// output partition.
    ///
    /// Mirrors DataFusion `SortExec::with_preserve_partitioning`. fdapquery's v0.1 only ever
    /// drives single-partition inputs (the planner doesn't emit
    /// repartitioning yet), so `preserve_partitioning = true` is a
    /// no-op at execute time — but the field is preserved so the
    /// strict-mirror surface area matches DataFusion.
    pub fn with_preserve_partitioning(mut self, preserve_partitioning: bool) -> Self {
        self.preserve_partitioning = preserve_partitioning;
        self.cache = Self::compute_properties(&self.input, preserve_partitioning);
        self
    }

    /// Modify how many rows to include in the result.
    ///
    /// If `None`, then all rows will be returned, in sorted order. If
    /// `Some(n)`, only the first `n` rows are emitted (top-K), which can
    /// reduce memory pressure since rows beyond the budget can be
    /// dropped.
    ///
    /// Mirrors DataFusion `SortExec::with_fetch`. DataFusion's TopK path
    /// additionally allocates a dynamic-filter wrapper for predicate
    /// pushdown; fdapquery v0.1 omits that wrapper (see module docs).
    pub fn with_fetch(&self, fetch: Option<usize>) -> Self {
        let mut new_sort = self.clone();
        new_sort.fetch = fetch;
        new_sort
    }

    /// Input plan. Mirrors DataFusion `SortExec::input`.
    pub fn input(&self) -> &Arc<dyn ExecutionPlan> {
        &self.input
    }

    /// Sort expressions. Mirrors DataFusion `SortExec::expr`.
    pub fn expr(&self) -> &LexOrdering {
        &self.expr
    }

    /// If `Some(fetch)`, limits output to only the first `fetch` items.
    /// Mirrors DataFusion `SortExec::fetch`.
    pub fn fetch(&self) -> Option<usize> {
        self.fetch
    }

    /// Compute the cached `PlanProperties`. Mirrors the partitioning
    /// branch of DataFusion's `compute_properties`
    /// (`SortExec::compute_properties` +
    /// `SortExec::output_partitioning_helper`): preserve →
    /// inherit the input's partitioning; else single-partition merged
    /// output (`UnknownPartitioning(1)`).
    fn compute_properties(
        input: &Arc<dyn ExecutionPlan>,
        preserve_partitioning: bool,
    ) -> PlanProperties {
        if preserve_partitioning {
            // Inherit the input's partitioning shape.
            PlanProperties::new(input.properties().output_partitioning.clone())
        } else {
            // Single-partition merged output.
            PlanProperties::single_partition_unknown()
        }
    }
}

impl DisplayAs for SortExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Byte-for-byte mirror of DataFusion
        // `<SortExec as DisplayAs>::fmt_as`. The
        // sort_prefix / filter sections are gated on substrates fdapquery
        // hasn't ported yet (equivalence properties, dynamic filters); see
        // module-level deferral notes. The two emitted format strings
        // match DataFusion exactly when those gates are not exercised.
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                let preserve_partitioning = self.preserve_partitioning;
                let exprs = format_exprs(&self.expr);
                match self.fetch {
                    Some(fetch) => write!(
                        f,
                        "SortExec: TopK(fetch={fetch}), expr=[{exprs}], preserve_partitioning=[{preserve_partitioning}]"
                    ),
                    None => write!(
                        f,
                        "SortExec: expr=[{exprs}], preserve_partitioning=[{preserve_partitioning}]"
                    ),
                }
            }
            DisplayFormatType::TreeRender => {
                let exprs = format_exprs(&self.expr);
                match self.fetch {
                    Some(fetch) => {
                        writeln!(f, "{exprs}")?;
                        writeln!(f, "limit={fetch}")
                    }
                    None => writeln!(f, "{exprs}"),
                }
            }
        }
    }
}

impl fmt::Display for SortExec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        <Self as DisplayAs>::fmt_as(self, DisplayFormatType::Default, f)
    }
}

/// Format the lex-order list the same way DataFusion's
/// `LexOrdering: Display` does — comma-separated `PhysicalSortExpr`
/// `Display` output with no enclosing brackets. The brackets are added
/// by the `expr=[...]` wrapper in `fmt_as` above. Byte-for-byte mirror
/// of `impl Display for datafusion_physical_expr_common::sort_expr::LexOrdering`.
fn format_exprs(exprs: &LexOrdering) -> String {
    let mut out = String::new();
    let mut first = true;
    for sort_expr in exprs {
        if first {
            first = false;
        } else {
            out.push_str(", ");
        }
        out.push_str(&sort_expr.to_string());
    }
    out
}

impl ExecutionPlan for SortExec {
    fn name(&self) -> &str {
        // DataFusion returns `"SortExec(TopK)"` when fetch is set
        // (see `<SortExec as ExecutionPlan>::name`). fdapquery v0.1's `ExecutionPlan::name`
        // returns `&str` (not `&'static str`) so we cannot directly
        // return a runtime-conditional string slice without a leak; both
        // names are statically known, so we match on the field.
        match self.fetch {
            Some(_) => "SortExec(TopK)",
            None => "SortExec",
        }
    }

    fn schema(&self) -> Schema {
        // Sort preserves the input schema — DataFusion's `SortExec`
        // returns `self.input.schema()` via the trait's default-derived
        // path. We do the same explicitly.
        self.input.schema()
    }

    fn properties(&self) -> &PlanProperties {
        &self.cache
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        // Mirrors DataFusion `<SortExec as ExecutionPlan>::children`.
        vec![&self.input]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        // Mirrors DataFusion
        // `<SortExec as ExecutionPlan>::with_new_children`: arity-1, reuse expr /
        // fetch / preserve_partitioning, recompute cached properties from
        // the new input.
        if children.len() != 1 {
            return Err(FdapQueryError::Internal(format!(
                "SortExec::with_new_children expected 1 child, got {}",
                children.len()
            )));
        }
        let new_input = children.into_iter().next().unwrap();
        let cache = Self::compute_properties(&new_input, self.preserve_partitioning);
        Ok(Arc::new(SortExec {
            input: new_input,
            expr: self.expr.clone(),
            metrics_set: self.metrics_set.clone(),
            preserve_partitioning: self.preserve_partitioning,
            fetch: self.fetch,
            cache,
        }))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn execute(
        &self,
        partition: usize,
        ctx: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        // v0.1 single-partition merged output (preserve_partitioning =
        // false default). The full DataFusion dispatch on
        // (sort_satisfied, fetch) is documented in the module header —
        // we always take the (false, _) branch here. When the input is
        // larger than memory, DataFusion's `ExternalSorter` spills to
        // disk; this in-memory mirror does not.
        if !self.preserve_partitioning && partition != 0 {
            return Err(FdapQueryError::Internal(format!(
                "SortExec has 1 output partition; partition {partition} is out of range"
            )));
        }

        let input = self.input.execute(partition, Arc::clone(&ctx))?;
        let schema = self.input.schema();
        let arrow_schema = Arc::new(schema.clone());
        let expr = self.expr.clone();
        let fetch = self.fetch;

        let arrow_schema_for_stream = Arc::clone(&arrow_schema);
        let stream = async_stream::try_stream! {
            // Step 1 — collect every input batch into memory. DataFusion's
            // `ExternalSorter::insert_batch` performs the same gather on
            // the no-spill path.
            let batches: Vec<_> = input.try_collect().await?;

            if batches.is_empty() {
                // Empty input → empty output, schema preserved.
                return;
            }

            // Step 2 — concat into a single RecordBatch. Matches
            // DataFusion's `in_mem_sort_stream` which calls
            // `concat_batches(&schema, &in_mem_batches)` before sorting.
            let combined = concat_batches(&arrow_schema_for_stream, &batches)
                .map_err(|e| FdapQueryError::Internal(format!(
                    "SortExec: concat_batches failed: {e}"
                )))?;

            // Step 3 — evaluate each PhysicalSortExpr against the
            // combined batch to produce a SortColumn (array + options).
            // Mirrors arrow's `PhysicalSortExpr::evaluate_to_sort_column`
            // pattern used by DataFusion.
            let mut sort_columns = Vec::with_capacity(expr.len());
            for sort_expr in &expr {
                let value = sort_expr.expr.evaluate(&combined)?;
                let array = value.into_array(combined.num_rows())?;
                sort_columns.push(arrow::compute::SortColumn {
                    values: array,
                    options: Some(sort_expr.options),
                });
            }

            // Step 4 — compute the sort permutation. `lexsort_to_indices`
            // accepts a `limit` so we pass `fetch` straight through; the
            // arrow kernel will only materialise the top-N indices.
            let indices = lexsort_to_indices(&sort_columns, fetch)
                .map_err(|e| FdapQueryError::Internal(format!(
                    "SortExec: lexsort_to_indices failed: {e}"
                )))?;

            // Step 5 — apply the permutation to materialise the sorted
            // RecordBatch. `take_record_batch` walks every column and
            // calls `arrow::compute::take` per column.
            let sorted = take_record_batch(&combined, &indices as &dyn Array)
                .map_err(|e| FdapQueryError::Internal(format!(
                    "SortExec: take_record_batch failed: {e}"
                )))?;

            yield sorted;
        };

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            arrow_schema,
            stream,
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::{TestSourceExec, employee_source};
    use arrow::compute::SortOptions;
    use arrow_array::{Int64Array, RecordBatch};
    use fdapquery_datatypes::Field;
    use fdapquery_execution::TaskContext;
    use fdapquery_physical_expr::{Column, PhysicalSortExpr};
    use futures::TryStreamExt;

    fn test_ctx() -> Arc<TaskContext> {
        Arc::new(TaskContext::default_test())
    }

    /// Build a `LexOrdering` with a single column key.
    fn key(col: &str, idx: usize, descending: bool, nulls_first: bool) -> LexOrdering {
        vec![PhysicalSortExpr::new(
            Arc::new(Column::new(col, idx)),
            SortOptions {
                descending,
                nulls_first,
            },
        )]
    }

    /// One-batch Int64 source over a column named "v".
    fn int_source(values: Vec<i64>) -> Arc<dyn ExecutionPlan> {
        let schema = Schema::new(vec![Field::new("v", arrow_schema::DataType::Int64, false)]);
        let arr = Arc::new(Int64Array::from(values));
        let batch = RecordBatch::try_new(Arc::new(schema.clone()), vec![arr]).unwrap();
        Arc::new(TestSourceExec::new(schema, vec![batch]))
    }

    async fn collect_v(plan: Arc<dyn ExecutionPlan>) -> Vec<i64> {
        let batches: Vec<RecordBatch> = plan
            .execute(0, test_ctx())
            .unwrap()
            .try_collect()
            .await
            .unwrap();
        let mut out = Vec::new();
        for b in batches {
            let col = b
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .clone();
            for i in 0..col.len() {
                out.push(col.value(i));
            }
        }
        out
    }

    /// Byte-for-byte mirror of DataFusion's `DisplayFormatType::Default`
    /// output for `SortExec`. Source-of-truth: DataFusion
    /// `<SortExec as DisplayAs>::fmt_as`. The
    /// two emitted templates are:
    /// - `"SortExec: expr=[...], preserve_partitioning=[{p}]"` (no fetch)
    /// - `"SortExec: TopK(fetch={n}), expr=[...], preserve_partitioning=[{p}]"`
    #[test]
    fn display_default_single_key_no_fetch() {
        let plan = SortExec::new(
            key("a", 0, /*desc=*/ false, /*nulls_first=*/ false),
            employee_source(),
        );
        assert_eq!(
            format!("{plan}"),
            "SortExec: expr=[a@0 ASC NULLS LAST], preserve_partitioning=[false]"
        );
    }

    #[test]
    fn display_default_with_fetch_renders_as_topk() {
        let plan = SortExec::new(key("a", 0, false, false), employee_source()).with_fetch(Some(10));
        assert_eq!(
            format!("{plan}"),
            "SortExec: TopK(fetch=10), expr=[a@0 ASC NULLS LAST], preserve_partitioning=[false]"
        );
    }

    #[test]
    fn display_default_multi_key_mixed_directions() {
        let mut lex = key("a", 0, false, false);
        lex.extend(key(
            "b", 1, /*desc=*/ true, /*nulls_first=*/ false,
        ));
        let plan = SortExec::new(lex, employee_source());
        assert_eq!(
            format!("{plan}"),
            "SortExec: expr=[a@0 ASC NULLS LAST, b@1 DESC NULLS LAST], preserve_partitioning=[false]"
        );
    }

    /// `displayable(plan).indent(false)` round-trips the same first line —
    /// confirms the operator reaches users via the tree walker too.
    #[test]
    fn displayable_indent_default_first_line() {
        let plan: Arc<dyn ExecutionPlan> =
            Arc::new(SortExec::new(key("a", 0, false, false), employee_source()));
        let rendered = format!(
            "{}",
            crate::display::displayable(plan.as_ref()).indent(false)
        );
        let first_line = rendered.lines().next().unwrap();
        assert_eq!(
            first_line,
            "SortExec: expr=[a@0 ASC NULLS LAST], preserve_partitioning=[false]"
        );
    }

    #[tokio::test]
    async fn sorts_single_key_ascending() {
        let plan: Arc<dyn ExecutionPlan> = Arc::new(SortExec::new(
            key("v", 0, /*desc=*/ false, /*nulls_first=*/ false),
            int_source(vec![3, 1, 2]),
        ));
        assert_eq!(collect_v(plan).await, vec![1, 2, 3]);
    }

    #[tokio::test]
    async fn sorts_single_key_descending() {
        let plan: Arc<dyn ExecutionPlan> = Arc::new(SortExec::new(
            key("v", 0, /*desc=*/ true, /*nulls_first=*/ false),
            int_source(vec![1, 2, 3]),
        ));
        assert_eq!(collect_v(plan).await, vec![3, 2, 1]);
    }

    #[tokio::test]
    async fn topk_limits_to_fetch() {
        // 100 rows, fetch=5 → 5 rows after sorting.
        let mut values: Vec<i64> = (0..100).collect();
        values.reverse(); // 99, 98, …, 0
        let plan: Arc<dyn ExecutionPlan> = Arc::new(
            SortExec::new(key("v", 0, false, false), int_source(values)).with_fetch(Some(5)),
        );
        assert_eq!(collect_v(plan).await, vec![0, 1, 2, 3, 4]);
    }

    /// Accessor names match DataFusion's exact signatures
    /// (see `SortExec::input`, `SortExec::expr`, `SortExec::fetch`).
    #[test]
    fn accessor_method_names_match_datafusion() {
        let input = employee_source();
        let lex = key("a", 0, false, false);
        let plan = SortExec::new(lex.clone(), Arc::clone(&input))
            .with_fetch(Some(7))
            .with_preserve_partitioning(false);

        // Same names as DataFusion — `input()`, `expr()`, `fetch()`,
        // `preserve_partitioning()`.
        let _: &Arc<dyn ExecutionPlan> = plan.input();
        assert_eq!(plan.expr().len(), lex.len());
        assert_eq!(plan.fetch(), Some(7));
        assert!(!plan.preserve_partitioning());
    }

    /// `with_new_children` carries forward expr / fetch /
    /// preserve_partitioning. Mirrors DataFusion
    /// `<SortExec as ExecutionPlan>::with_new_children`.
    #[test]
    fn with_new_children_preserves_config() {
        let plan = SortExec::new(key("a", 0, true, false), employee_source())
            .with_fetch(Some(3))
            .with_preserve_partitioning(false);
        let arc: Arc<dyn ExecutionPlan> = Arc::new(plan.clone());
        let rebuilt = arc.with_new_children(vec![employee_source()]).unwrap();
        let downcast = rebuilt.as_any().downcast_ref::<SortExec>().unwrap();
        assert_eq!(downcast.fetch(), Some(3));
        assert!(!downcast.preserve_partitioning());
        assert_eq!(downcast.expr().len(), 1);
    }

    #[test]
    fn with_new_children_rejects_wrong_arity() {
        let arc: Arc<dyn ExecutionPlan> =
            Arc::new(SortExec::new(key("a", 0, false, false), employee_source()));
        assert!(arc.with_new_children(vec![]).is_err());
    }
}
