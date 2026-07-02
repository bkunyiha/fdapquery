//!
//! Hash equi-join. Builds a hash table from the **right** (build) side keyed by
//! the right join columns, then probes it with each **left** (probe) row.
//!
//! ## Strict mirror of DataFusion's `HashJoinExec`
//! Struct field names (`left`, `right`, `on`, `filter`, `join_type`,
//! `join_schema`, `mode`, `projection`, `null_equality`, `null_aware`,
//! `column_indices`), constructor signature
//! (`try_new(left, right, on, filter, join_type: &JoinType, projection,
//! partition_mode, null_equality, null_aware)`), accessor names
//! (`left()`, `right()`, `on()`, `filter()`, `join_type()`, `join_schema()`,
//! `partition_mode()`, `null_equality()`), and the `DisplayAs::fmt_as`
//! `Default`/`Verbose` output
//! (`"HashJoinExec: mode={Mode:?}, join_type={JoinType:?}, on=[(l, r), …]{…}"`)
//! match `datafusion/physical-plan/src/joins/hash_join/exec.rs`
//! byte-for-byte.
//!
//! ## Documented runtime divergences
//! 1. **`PartitionMode::Auto` resolves to `Partitioned` at construction.**
//!    fdapquery's planner has no statistics surface to pick CollectLeft vs.
//!    Partitioned; the enum variant exists for serializer/planner parity but
//!    `try_new` normalises any `Auto` to `Partitioned` so downstream
//!    execution code never has to handle the indeterminate case.
//! 2. **`null_aware` is accepted but unused.** The bool is stored and surfaces
//!    in DisplayAs output (as the `", null_aware"` suffix) for byte parity
//!    with DataFusion. The execution path doesn't yet emit a null-aware
//!    anti-join column. Setting `null_aware=true` does not change join output.
//! 3. **`JoinType` variants beyond `Inner`/`Left`/`Right` panic at execute
//!    time** with a clear "not yet implemented" message. The API surface
//!    accepts all 10 variants (so callers can construct a `Full` or
//!    `LeftSemi` plan and round-trip it through the planner / serializer),
//!    but only the three classical variants actually run.
//! 4. **`filter`, `projection`, `column_indices`** are carried through Display
//!    and accessors but don't yet alter execution. The classical equi-join
//!    body uses `on` for keying and emits the full concatenated row.
//!    `column_indices` is left as an empty `Vec` until DataFusion's
//!    `build_join_schema` is mirrored alongside `JoinFilter` consumption.
//!
//! ## Implementation notes
//! - **Join keys / rows are `Vec<ScalarValue>`.** The hash table is keyed by
//!   `crate::row_key::RowKey` — the same float-aware key helper
//!   `AggregateExec` uses for group keys.
//!   String columns surface as `ScalarValue::Utf8`, so no extra normalization
//!   is needed.
//! - **`right_columns_to_exclude`** (now a private field, set by the planner
//!   via [`HashJoinExec::with_right_columns_to_exclude`]) drops duplicate
//!   join-key columns from the right side of the combined row (so an
//!   `id = id` join doesn't emit `id` twice).
//! - **Eager, not lazy.** The build side must be fully materialized first
//!   anyway; the join collects all output batches.
//! - **Right join** re-scans the left side to find which right keys matched,
//!   then emits the unmatched right rows with nulls on the left.

use crate::PhysicalExpr;
use crate::physical_plan::ExecutionPlan;
use crate::plan_properties::PlanProperties;
use crate::row_key::RowKey;
use crate::stream::{RecordBatchStreamAdapter, SendableRecordBatchStream};
use arrow_array::ArrayRef;
use async_stream::try_stream;
use fdapquery_common::{ArrowVectorBuilder, FdapQueryError, Result, ScalarValue};
use fdapquery_datatypes::{RecordBatch, Schema, record_batch};
use fdapquery_execution::TaskContext;
use fdapquery_expr::{JoinSide, JoinType, NullEquality};
use fdapquery_physical_expr::Column;
use futures::StreamExt;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

// ============================================================================
// Companion types — strict-mirror of DataFusion's `joins/` module surface.
// ============================================================================

/// Partitioning mode for a hash join. Strict mirror of
/// `datafusion::physical_plan::joins::PartitionMode`.
///
/// fdapquery does not yet have a statistics-driven optimizer that can resolve
/// `Auto`, so `HashJoinExec::try_new` normalises `Auto` → `Partitioned` at
/// construction (see the module-level "Documented runtime divergences"). The
/// variant remains in the enum so callers — planner, serializer, distributed —
/// can mirror DataFusion's match arms verbatim.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PartitionMode {
    /// Left/right children are partitioned using the left and right keys.
    Partitioned,
    /// Left side will collected into one partition.
    CollectLeft,
    /// Optimizer decides which `PartitionMode` is optimal based on
    /// statistics. fdapquery currently treats this as `Partitioned` at
    /// construction time.
    Auto,
}

/// Information about the index and placement (left or right) of the columns
/// used to build the join output schema. Strict mirror of
/// `datafusion::physical_plan::joins::utils::ColumnIndex`.
#[derive(Debug, Clone, PartialEq)]
pub struct ColumnIndex {
    /// Index of the column in the side it comes from
    pub index: usize,
    /// Whether the column is from the left or right (or neither, for Mark)
    pub side: JoinSide,
}

/// Filter applied before join output. Strict mirror of
/// `datafusion::physical_plan::joins::join_filter::JoinFilter`. Fields are
/// `pub(crate)` so downstream operators can experiment with custom joins via
/// the same surface DataFusion exposes.
#[derive(Debug, Clone)]
pub struct JoinFilter {
    /// Filter expression
    pub(crate) expression: Arc<dyn PhysicalExpr>,
    /// Column indices required to construct the intermediate batch for
    /// filtering
    pub(crate) column_indices: Vec<ColumnIndex>,
    /// Physical schema of the intermediate batch
    pub(crate) schema: Schema,
}

impl JoinFilter {
    /// Create a new `JoinFilter`. Argument order matches DataFusion's
    /// `JoinFilter::new(expression, column_indices, schema)`.
    pub fn new(
        expression: Arc<dyn PhysicalExpr>,
        column_indices: Vec<ColumnIndex>,
        schema: Schema,
    ) -> Self {
        Self {
            expression,
            column_indices,
            schema,
        }
    }

    /// Filter expression
    pub fn expression(&self) -> &Arc<dyn PhysicalExpr> {
        &self.expression
    }

    /// Column indices required to construct the intermediate batch
    pub fn column_indices(&self) -> &[ColumnIndex] {
        &self.column_indices
    }

    /// Physical schema of the intermediate batch
    pub fn schema(&self) -> &Schema {
        &self.schema
    }
}

/// The on-clause of a join, as vector of (left, right) expression pairs.
/// Strict mirror of `datafusion::physical_plan::joins::JoinOn`.
pub type JoinOn = Vec<(Arc<dyn PhysicalExpr>, Arc<dyn PhysicalExpr>)>;

/// Reference for [`JoinOn`]. Strict mirror of
/// `datafusion::physical_plan::joins::JoinOnRef`.
pub type JoinOnRef<'a> = &'a [(Arc<dyn PhysicalExpr>, Arc<dyn PhysicalExpr>)];

// ============================================================================
// HashJoinExec
// ============================================================================

/// Hash join physical operator. Strict mirror of
/// `datafusion::physical_plan::joins::HashJoinExec`.
///
/// The execution body only handles `Inner` / `Left` / `Right` today; the
/// remaining 7 `JoinType` variants error at `execute` time. See the
/// module-level docs for the full divergence list.
#[derive(Debug)]
pub struct HashJoinExec {
    /// left (build) side which gets hashed
    pub left: Arc<dyn ExecutionPlan>,
    /// right (probe) side which are filtered by the hash table
    pub right: Arc<dyn ExecutionPlan>,
    /// Set of equijoin columns from the relations: `(left_col, right_col)`
    pub on: JoinOn,
    /// Filters which are applied while finding matching rows
    pub filter: Option<JoinFilter>,
    /// How the join is performed (`OUTER`, `INNER`, etc.)
    pub join_type: JoinType,
    /// The schema after join. If `projection` is set, this is not the output
    /// schema — DataFusion mirrors this caveat.
    join_schema: Schema,
    /// Partitioning mode to use
    pub mode: PartitionMode,
    /// The projection indices of the columns in the output schema of join
    pub projection: Option<Vec<usize>>,
    /// Information of index and left / right placement of columns. Populated
    /// when the planner builds the join schema; fdapquery's planner does not
    /// yet build it, so this is typically empty until #114/#117 land.
    column_indices: Vec<ColumnIndex>,
    /// The equality null-handling behaviour of the join algorithm
    pub null_equality: NullEquality,
    /// Flag to indicate if this is a null-aware anti join (surfaces in Display
    /// for byte parity with DataFusion; not yet honoured at execute time)
    pub null_aware: bool,
    /// Right columns to exclude from the combined output row — used to drop
    /// duplicate join-key columns. Not in DataFusion's public field set
    /// (DataFusion's `build_join_schema` + `column_indices` handle this), but
    /// fdapquery's planner produces it directly; carried as a private field
    /// to keep the execution body unchanged.
    right_columns_to_exclude: HashSet<usize>,
    properties: PlanProperties,
}

impl HashJoinExec {
    /// Try to create a new [`HashJoinExec`]. Argument order matches
    /// DataFusion's `HashJoinExec::try_new(left, right, on, filter,
    /// join_type, projection, partition_mode, null_equality, null_aware)`.
    ///
    /// `PartitionMode::Auto` is normalised to `PartitionMode::Partitioned`
    /// because fdapquery has no statistics surface to pick CollectLeft vs.
    /// Partitioned at planning time — see the module-level "Documented
    /// runtime divergences".
    ///
    /// `join_schema` is computed externally and supplied via
    /// [`HashJoinExec::with_join_schema`] — DataFusion derives it from
    /// `build_join_schema(left.schema(), right.schema(), join_type)`, which
    /// fdapquery's planner already has in hand. `try_new` defaults the
    /// schema to the left input's schema; callers (the planner) override it.
    /// `right_columns_to_exclude` similarly defaults to empty here and is
    /// supplied by the planner via
    /// [`HashJoinExec::with_right_columns_to_exclude`].
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        left: Arc<dyn ExecutionPlan>,
        right: Arc<dyn ExecutionPlan>,
        on: JoinOn,
        filter: Option<JoinFilter>,
        join_type: &JoinType,
        projection: Option<Vec<usize>>,
        partition_mode: PartitionMode,
        null_equality: NullEquality,
        null_aware: bool,
    ) -> Result<Self> {
        let mode = match partition_mode {
            PartitionMode::Auto => PartitionMode::Partitioned,
            other => other,
        };
        let join_schema = left.schema();
        let properties = PlanProperties::single_partition_unknown();
        Ok(Self {
            left,
            right,
            on,
            filter,
            join_type: *join_type,
            join_schema,
            mode,
            projection,
            column_indices: Vec::new(),
            null_equality,
            null_aware,
            right_columns_to_exclude: HashSet::new(),
            properties,
        })
    }

    /// Override the join output schema. The planner builds the join schema
    /// in advance and passes it in via this setter so that the
    /// DataFusion-shaped `try_new` stays a strict mirror of the upstream
    /// signature.
    pub fn with_join_schema(mut self, schema: Schema) -> Self {
        self.join_schema = schema;
        self
    }

    /// Override the column-index metadata. DataFusion populates this in its
    /// `try_new` via `build_join_schema`; fdapquery's planner supplies it via
    /// this setter so the strict-mirror `try_new` signature stays exact.
    pub fn with_column_indices(mut self, column_indices: Vec<ColumnIndex>) -> Self {
        self.column_indices = column_indices;
        self
    }

    /// Override the right-column exclude set. Carried as a private field so
    /// the execution body can drop duplicate join-key columns without
    /// rebuilding the schema mid-execute.
    pub fn with_right_columns_to_exclude(mut self, excluded: HashSet<usize>) -> Self {
        self.right_columns_to_exclude = excluded;
        self
    }

    /// left (build) side which gets hashed
    pub fn left(&self) -> &Arc<dyn ExecutionPlan> {
        &self.left
    }

    /// right (probe) side which are filtered by the hash table
    pub fn right(&self) -> &Arc<dyn ExecutionPlan> {
        &self.right
    }

    /// Set of common columns used to join on
    pub fn on(&self) -> JoinOnRef<'_> {
        &self.on
    }

    /// Filters applied before join output
    pub fn filter(&self) -> Option<&JoinFilter> {
        self.filter.as_ref()
    }

    /// How the join is performed
    pub fn join_type(&self) -> &JoinType {
        &self.join_type
    }

    /// The schema after join. If there is a projection set, this is not the
    /// same as the output schema.
    pub fn join_schema(&self) -> &Schema {
        &self.join_schema
    }

    /// The partitioning mode of this hash join
    pub fn partition_mode(&self) -> &PartitionMode {
        &self.mode
    }

    /// The null-equality behaviour of this hash join
    pub fn null_equality(&self) -> NullEquality {
        self.null_equality
    }

    /// Information of index and left / right placement of columns
    pub fn column_indices(&self) -> &[ColumnIndex] {
        &self.column_indices
    }

    /// True iff a projection is set on the join output
    pub fn contains_projection(&self) -> bool {
        self.projection.is_some()
    }
}

/// Extract a column index from a join-on physical expression. fdapquery's
/// equi-join body works in column-index space, so the `on` clause must be a
/// pair of [`Column`] expressions. Anything richer requires the planner to
/// project first (DataFusion lifts this restriction via its
/// `equijoin_column_indices` helper, which fdapquery will mirror once #117
/// tightens the `PhysicalExpr::evaluate` selection argument).
fn on_index(expr: &Arc<dyn PhysicalExpr>, side: &'static str) -> Result<usize> {
    if let Some(c) = expr.as_any().downcast_ref::<Column>() {
        Ok(c.index)
    } else {
        Err(FdapQueryError::Internal(format!(
            "HashJoinExec: {side} join key must be a Column expression, got: {expr}"
        )))
    }
}

/// Concatenate a left row with a right row, dropping the right columns listed
/// in `right_columns_to_exclude` (the duplicate join keys). Free function so
/// the generator body can call it without holding `&self`.
fn combine_rows(
    left_row: &[ScalarValue],
    right_row: &[ScalarValue],
    right_columns_to_exclude: &HashSet<usize>,
) -> Vec<ScalarValue> {
    let mut result: Vec<ScalarValue> = left_row.to_vec();
    for (i, value) in right_row.iter().enumerate() {
        if !right_columns_to_exclude.contains(&i) {
            result.push(value.clone());
        }
    }
    result
}

/// Build an output batch from assembled rows, typed by the output schema.
fn create_batch(rows: &[Vec<ScalarValue>], schema: &Schema) -> Result<RecordBatch> {
    let mut builders: Vec<ArrowVectorBuilder> = schema
        .fields()
        .iter()
        .map(|f| ArrowVectorBuilder::new(f.data_type(), rows.len()))
        .collect();
    for row in rows {
        for (col, value) in row.iter().enumerate() {
            builders[col].append_value(value);
        }
    }
    let columns: Vec<ArrayRef> = builders.into_iter().map(|b| b.build()).collect();
    record_batch::create(schema, columns)
}

/// Wrap each column of `batch` once, so rows can be read by index without
/// re-wrapping the arrays per row.
fn columns_of(batch: &RecordBatch) -> Vec<ArrayRef> {
    (0..batch.num_columns())
        .map(|i| batch.column(i).clone())
        .collect()
}

/// The join key for one row: the values of the given key columns.
fn key_of(cols: &[ArrayRef], keys: &[usize], row: usize) -> Result<RowKey> {
    Ok(RowKey(
        keys.iter()
            .map(|&k| ScalarValue::try_from_array(&cols[k], row))
            .collect::<Result<Vec<_>>>()?,
    ))
}

/// Every column value for one row.
fn full_row(cols: &[ArrayRef], row: usize) -> Result<Vec<ScalarValue>> {
    cols.iter()
        .map(|c| ScalarValue::try_from_array(c, row))
        .collect()
}

impl ExecutionPlan for HashJoinExec {
    fn name(&self) -> &'static str {
        "HashJoinExec"
    }

    fn schema(&self) -> Schema {
        self.join_schema.clone()
    }

    fn properties(&self) -> &PlanProperties {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.left, &self.right]
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    /// Rebuild this join with new left and right inputs. Arity 2: a hash
    /// join has two inputs in `[left, right]` order. Ordering matters:
    /// swapping left and right changes the hash table's key columns and
    /// would silently produce a different join.
    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if children.len() != 2 {
            return Err(FdapQueryError::Internal(format!(
                "HashJoinExec::with_new_children expected 2 children (left, right), got {}",
                children.len()
            )));
        }
        let mut iter = children.into_iter();
        let left = iter.next().unwrap();
        let right = iter.next().unwrap();
        let mut rebuilt = HashJoinExec::try_new(
            left,
            right,
            self.on.clone(),
            self.filter.clone(),
            &self.join_type,
            self.projection.clone(),
            self.mode,
            self.null_equality,
            self.null_aware,
        )?;
        rebuilt.join_schema = self.join_schema.clone();
        rebuilt.column_indices.clone_from(&self.column_indices);
        rebuilt
            .right_columns_to_exclude
            .clone_from(&self.right_columns_to_exclude);
        Ok(Arc::new(rebuilt))
    }

    fn execute(
        &self,
        partition: usize,
        ctx: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        if partition != 0 {
            return Err(FdapQueryError::Internal(format!(
                "HashJoinExec has 1 output partition; partition {partition} is out of range"
            )));
        }
        // The classical equi-join body only handles three variants today.
        match self.join_type {
            JoinType::Inner | JoinType::Left | JoinType::Right => {}
            other => {
                return Err(FdapQueryError::Internal(format!(
                    "HashJoinExec: JoinType::{other:?} is not yet implemented; \
                     supported variants are Inner, Left, Right"
                )));
            }
        }

        // Lower `on` to column-index pairs. DataFusion's
        // `equijoin_column_indices` does the same job; fdapquery requires
        // each side of every pair to be a `Column` until #117 lands.
        let mut left_keys: Vec<usize> = Vec::with_capacity(self.on.len());
        let mut right_keys: Vec<usize> = Vec::with_capacity(self.on.len());
        for (l, r) in &self.on {
            left_keys.push(on_index(l, "left")?);
            right_keys.push(on_index(r, "right")?);
        }

        // Clone everything the generator needs — it runs detached from `self`.
        let left = Arc::clone(&self.left);
        let right = Arc::clone(&self.right);
        let join_type = self.join_type;
        let schema = self.join_schema.clone();
        let right_columns_to_exclude = self.right_columns_to_exclude.clone();
        let right_field_count = self.right.schema().fields().len();
        let left_field_count = self.left.schema().fields().len();
        let arrow_schema = Arc::new(self.join_schema.clone());
        let ctx_for_probe = Arc::clone(&ctx);
        let ctx_for_unmatched = Arc::clone(&ctx);

        let stream = try_stream! {
            // --- Build phase: load the right side into a hash table. ---
            let mut hash_table: HashMap<RowKey, Vec<Vec<ScalarValue>>> = HashMap::new();
            let right_stream = right.execute(0, Arc::clone(&ctx))?;
            let mut right_pinned = std::pin::pin!(right_stream);
            while let Some(batch_res) = right_pinned.next().await {
                let batch = batch_res?;
                let cols = columns_of(&batch);
                for row in 0..batch.num_rows() {
                    let key = key_of(&cols, &right_keys, row)?;
                    hash_table
                        .entry(key)
                        .or_default()
                        .push(full_row(&cols, row)?);
                }
            }

            // --- Probe phase: emit one batch per left input batch. ---
            let left_stream = left.execute(0, ctx_for_probe)?;
            let mut left_pinned = std::pin::pin!(left_stream);
            while let Some(left_batch_res) = left_pinned.next().await {
                let left_batch = left_batch_res?;
                let cols = columns_of(&left_batch);
                let mut output_rows: Vec<Vec<ScalarValue>> = Vec::new();
                for row in 0..left_batch.num_rows() {
                    let probe_key = key_of(&cols, &left_keys, row)?;
                    let left_row = full_row(&cols, row)?;
                    let matched = hash_table.get(&probe_key);
                    match join_type {
                        JoinType::Inner | JoinType::Right => {
                            if let Some(rows) = matched {
                                for right_row in rows {
                                    output_rows.push(combine_rows(
                                        &left_row,
                                        right_row,
                                        &right_columns_to_exclude,
                                    ));
                                }
                            }
                        }
                        JoinType::Left => {
                            if let Some(rows) = matched {
                                for right_row in rows {
                                    output_rows.push(combine_rows(
                                        &left_row,
                                        right_row,
                                        &right_columns_to_exclude,
                                    ));
                                }
                            } else {
                                // No match: left row with nulls for the right columns.
                                let null_right = vec![ScalarValue::Null; right_field_count];
                                output_rows.push(combine_rows(
                                    &left_row,
                                    &null_right,
                                    &right_columns_to_exclude,
                                ));
                            }
                        }
                        _ => unreachable!("guarded above"),
                    }
                }
                if !output_rows.is_empty() {
                    yield create_batch(&output_rows, &schema)?;
                }
            }

            // --- Right join: re-scan left to find which right keys matched,
            // then emit the unmatched right rows with nulls on the left. ---
            if matches!(join_type, JoinType::Right) {
                let mut matched_keys: HashSet<RowKey> = HashSet::new();
                let left_stream_2 = left.execute(0, ctx_for_unmatched)?;
                let mut left_pinned_2 = std::pin::pin!(left_stream_2);
                while let Some(left_batch_res) = left_pinned_2.next().await {
                    let left_batch = left_batch_res?;
                    let cols = columns_of(&left_batch);
                    for row in 0..left_batch.num_rows() {
                        let probe_key = key_of(&cols, &left_keys, row)?;
                        if hash_table.contains_key(&probe_key) {
                            matched_keys.insert(probe_key);
                        }
                    }
                }
                let mut unmatched: Vec<Vec<ScalarValue>> = Vec::new();
                for (key, rows) in &hash_table {
                    if !matched_keys.contains(key) {
                        let null_left = vec![ScalarValue::Null; left_field_count];
                        for right_row in rows {
                            unmatched.push(combine_rows(
                                &null_left,
                                right_row,
                                &right_columns_to_exclude,
                            ));
                        }
                    }
                }
                if !unmatched.is_empty() {
                    yield create_batch(&unmatched, &schema)?;
                }
            }
        };

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            arrow_schema,
            stream,
        )))
    }
}

impl crate::display::DisplayAs for HashJoinExec {
    fn fmt_as(
        &self,
        t: crate::display::DisplayFormatType,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        match t {
            crate::display::DisplayFormatType::Default
            | crate::display::DisplayFormatType::Verbose => {
                // Strict mirror of DataFusion's
                // `joins/hash_join/exec.rs::DisplayAs::fmt_as` Default arm:
                //   write!(
                //       f,
                //       "HashJoinExec: mode={:?}, join_type={:?}, on=[{}]{}{}{}{}{}",
                //       self.mode, self.join_type, on,
                //       display_filter, display_projections,
                //       display_null_equality, display_fetch, display_null_aware,
                //   )
                // fdapquery has no `fetch` field on `HashJoinExec` (DataFusion
                // added it for streaming joins), so `display_fetch` is always
                // empty — matches DataFusion's path when `self.fetch == None`.
                let display_filter = self.filter.as_ref().map_or_else(
                    String::new,
                    |jf| format!(", filter={}", jf.expression()),
                );
                let display_projections = if self.contains_projection() {
                    format!(
                        ", projection=[{}]",
                        self.projection
                            .as_ref()
                            .unwrap()
                            .iter()
                            .map(|index| format!(
                                "{}@{}",
                                self.join_schema.fields().get(*index).unwrap().name(),
                                index
                            ))
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                } else {
                    String::new()
                };
                let display_null_equality = if self.null_equality == NullEquality::NullEqualsNull {
                    ", NullsEqual: true"
                } else {
                    ""
                };
                let display_fetch = ""; // fdapquery has no fetch on HashJoinExec
                let display_null_aware = if self.null_aware { ", null_aware" } else { "" };
                let on = self
                    .on
                    .iter()
                    .map(|(c1, c2)| format!("({c1}, {c2})"))
                    .collect::<Vec<String>>()
                    .join(", ");
                write!(
                    f,
                    "HashJoinExec: mode={:?}, join_type={:?}, on=[{}]{}{}{}{}{}",
                    self.mode,
                    self.join_type,
                    on,
                    display_filter,
                    display_projections,
                    display_null_equality,
                    display_fetch,
                    display_null_aware,
                )
            }
            crate::display::DisplayFormatType::TreeRender => {
                // Mirror of DataFusion's TreeRender arm. We don't have
                // `fmt_sql` (a SQL pretty-printer for PhysicalExpr) yet, so
                // the on-pairs render via `Display` of the expression, which
                // matches fdapquery's `Column::Display` (`#i`) — the same
                // convention used in `FilterExec` and `ProjectionExec`
                // tree-render output.
                let on = self
                    .on
                    .iter()
                    .map(|(c1, c2)| format!("({c1} = {c2})"))
                    .collect::<Vec<String>>()
                    .join(", ");

                if self.join_type != JoinType::Inner {
                    writeln!(f, "join_type={:?}", self.join_type)?;
                }

                writeln!(f, "on={on}")?;

                if self.null_equality == NullEquality::NullEqualsNull {
                    writeln!(f, "NullsEqual: true")?;
                }

                if self.null_aware {
                    writeln!(f, "null_aware")?;
                }

                if let Some(filter) = self.filter.as_ref() {
                    writeln!(f, "filter={}", filter.expression())?;
                }

                Ok(())
            }
        }
    }
}

impl std::fmt::Display for HashJoinExec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        <Self as crate::display::DisplayAs>::fmt_as(
            self,
            crate::display::DisplayFormatType::Default,
            f,
        )
    }
}

#[cfg(test)]
mod tests {
    //! Join tests. These drive `HashJoinExec` directly via a tiny in-memory
    //! `PhysicalPlan`; the physical planner that normally builds a join lives
    //! in the `fdapquery` crate's `physical_planner` module.
    use super::*;
    use arrow_array::{ArrayRef, Int64Array, StringArray};
    use arrow_schema::{Field as ArrowField, Schema as ArrowSchema};
    use fdapquery_datatypes::Field;
    use fdapquery_physical_expr::Column;
    use futures::TryStreamExt;
    use std::sync::Arc;

    /// An `ExecutionPlan` that simply replays preset batches.
    #[derive(Debug)]
    struct VecExec {
        schema: Schema,
        batches: Vec<RecordBatch>,
        properties: PlanProperties,
    }

    impl VecExec {
        fn new(schema: Schema, batches: Vec<RecordBatch>) -> Self {
            Self {
                schema,
                batches,
                properties: PlanProperties::single_partition_unknown(),
            }
        }
    }

    impl ExecutionPlan for VecExec {
        fn name(&self) -> &'static str {
            "VecExec"
        }
        fn schema(&self) -> Schema {
            self.schema.clone()
        }
        fn properties(&self) -> &PlanProperties {
            &self.properties
        }
        fn execute(
            &self,
            _partition: usize,
            _ctx: Arc<TaskContext>,
        ) -> Result<SendableRecordBatchStream> {
            let arrow_schema = Arc::new(self.schema.clone());
            let inner = futures::stream::iter(self.batches.clone().into_iter().map(Ok));
            Ok(Box::pin(RecordBatchStreamAdapter::new(arrow_schema, inner)))
        }
        fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
            vec![]
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
        fn with_new_children(
            self: Arc<Self>,
            children: Vec<Arc<dyn ExecutionPlan>>,
        ) -> Result<Arc<dyn ExecutionPlan>> {
            assert!(children.is_empty());
            Ok(self)
        }
    }

    impl crate::display::DisplayAs for VecExec {
        fn fmt_as(
            &self,
            _t: crate::display::DisplayFormatType,
            f: &mut std::fmt::Formatter<'_>,
        ) -> std::fmt::Result {
            write!(f, "VecExec")
        }
    }

    impl std::fmt::Display for VecExec {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            <Self as crate::display::DisplayAs>::fmt_as(
                self,
                crate::display::DisplayFormatType::Default,
                f,
            )
        }
    }

    /// left: (id: Int64, name: Utf8) = (1,a),(2,b),(3,c)
    fn left_exec() -> VecExec {
        let schema = Schema::new(vec![
            Field::new("id", arrow_schema::DataType::Int64, true),
            Field::new("name", arrow_schema::DataType::Utf8, true),
        ]);
        let arrow = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("id", arrow_schema::DataType::Int64, true),
            ArrowField::new("name", arrow_schema::DataType::Utf8, true),
        ]));
        let id: ArrayRef = Arc::new(Int64Array::from(vec![1, 2, 3]));
        let name: ArrayRef = Arc::new(StringArray::from(vec!["a", "b", "c"]));
        VecExec::new(
            schema,
            vec![RecordBatch::try_new(arrow, vec![id, name]).unwrap()],
        )
    }

    /// right: (id: Int64, dept: Utf8) = (1,eng),(2,sales)
    fn right_exec() -> VecExec {
        let schema = Schema::new(vec![
            Field::new("id", arrow_schema::DataType::Int64, true),
            Field::new("dept", arrow_schema::DataType::Utf8, true),
        ]);
        let arrow = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("id", arrow_schema::DataType::Int64, true),
            ArrowField::new("dept", arrow_schema::DataType::Utf8, true),
        ]));
        let id: ArrayRef = Arc::new(Int64Array::from(vec![1, 2]));
        let dept: ArrayRef = Arc::new(StringArray::from(vec!["eng", "sales"]));
        VecExec::new(
            schema,
            vec![RecordBatch::try_new(arrow, vec![id, dept]).unwrap()],
        )
    }

    /// Output schema: id, name, dept (the right `id` is excluded as a duplicate key).
    fn out_schema() -> Schema {
        Schema::new(vec![
            Field::new("id", arrow_schema::DataType::Int64, true),
            Field::new("name", arrow_schema::DataType::Utf8, true),
            Field::new("dept", arrow_schema::DataType::Utf8, true),
        ])
    }

    type Row = (Option<i64>, Option<String>, Option<String>);

    fn collect_rows(batches: &[RecordBatch]) -> Vec<Row> {
        let mut out: Vec<Row> = Vec::new();
        for b in batches {
            let c0 = b.column(0).clone();
            let c1 = b.column(1).clone();
            let c2 = b.column(2).clone();
            for i in 0..b.num_rows() {
                let id = match ScalarValue::try_from_array(&c0, i).unwrap() {
                    ScalarValue::Int64(n) => Some(n),
                    ScalarValue::Null => None,
                    o => panic!("id: {o:?}"),
                };
                let name = match ScalarValue::try_from_array(&c1, i).unwrap() {
                    ScalarValue::Utf8(s) => Some(s),
                    ScalarValue::Null => None,
                    o => panic!("name: {o:?}"),
                };
                let dept = match ScalarValue::try_from_array(&c2, i).unwrap() {
                    ScalarValue::Utf8(s) => Some(s),
                    ScalarValue::Null => None,
                    o => panic!("dept: {o:?}"),
                };
                out.push((id, name, dept));
            }
        }
        out
    }

    fn test_ctx() -> Arc<TaskContext> {
        Arc::new(TaskContext::default_test())
    }

    /// `on` clause for `id = id` (left col 0 = right col 0).
    fn on_id_id() -> JoinOn {
        vec![(
            Arc::new(Column::new("id", 0)) as Arc<dyn PhysicalExpr>,
            Arc::new(Column::new("id", 0)) as Arc<dyn PhysicalExpr>,
        )]
    }

    fn build_inner_join() -> HashJoinExec {
        HashJoinExec::try_new(
            Arc::new(left_exec()),
            Arc::new(right_exec()),
            on_id_id(),
            None,
            &JoinType::Inner,
            None,
            PartitionMode::Partitioned,
            NullEquality::NullEqualsNothing,
            false,
        )
        .unwrap()
        .with_join_schema(out_schema())
        .with_right_columns_to_exclude(HashSet::from([0]))
    }

    fn build_left_join() -> HashJoinExec {
        HashJoinExec::try_new(
            Arc::new(left_exec()),
            Arc::new(right_exec()),
            on_id_id(),
            None,
            &JoinType::Left,
            None,
            PartitionMode::Partitioned,
            NullEquality::NullEqualsNothing,
            false,
        )
        .unwrap()
        .with_join_schema(out_schema())
        .with_right_columns_to_exclude(HashSet::from([0]))
    }

    #[tokio::test]
    async fn inner_join_on_id() {
        let join = build_inner_join();
        let mut rows = collect_rows(
            &join
                .execute(0, test_ctx())
                .unwrap()
                .try_collect::<Vec<_>>()
                .await
                .unwrap(),
        );
        rows.sort();
        assert_eq!(
            rows,
            vec![
                (Some(1), Some("a".to_string()), Some("eng".to_string())),
                (Some(2), Some("b".to_string()), Some("sales".to_string())),
            ]
        );
    }

    #[tokio::test]
    async fn left_join_keeps_unmatched_left() {
        let join = build_left_join();
        let mut rows = collect_rows(
            &join
                .execute(0, test_ctx())
                .unwrap()
                .try_collect::<Vec<_>>()
                .await
                .unwrap(),
        );
        rows.sort();
        assert_eq!(
            rows,
            vec![
                (Some(1), Some("a".to_string()), Some("eng".to_string())),
                (Some(2), Some("b".to_string()), Some("sales".to_string())),
                (Some(3), Some("c".to_string()), None), // id=3 has no right match
            ]
        );
    }

    /// Byte-for-byte mirror of DataFusion's `DisplayFormatType::Default`
    /// output for `HashJoinExec`:
    /// `"HashJoinExec: mode={mode:?}, join_type={join_type:?}, on=[{on}]{…}"`.
    /// Source:
    /// `datafusion::physical_plan::joins::hash_join::exec::HashJoinExec::fmt_as`.
    #[test]
    fn display_default_matches_datafusion() {
        // Config 1: Inner join, single on-clause, no filter / projection.
        // `Column` now displays as `{name}@{index}`
        // (mirrors DataFusion), so the on-pair expected text is
        // `(id@0, id@0)` for the `id`-on-`id` clause.
        let join = build_inner_join();
        assert_eq!(
            format!("{join}"),
            "HashJoinExec: mode=Partitioned, join_type=Inner, on=[(id@0, id@0)]"
        );

        // Config 2: Left join with multiple on-clauses + CollectLeft mode.
        // The second pair uses a synthetic `col2` name on the right side
        // because the index (2) is past the end of `right_exec`'s schema —
        // the test exercises Display only, never executes, so the name is
        // free.
        let on_multi: JoinOn = vec![
            (
                Arc::new(Column::new("id", 0)) as Arc<dyn PhysicalExpr>,
                Arc::new(Column::new("id", 0)) as Arc<dyn PhysicalExpr>,
            ),
            (
                Arc::new(Column::new("name", 1)) as Arc<dyn PhysicalExpr>,
                Arc::new(Column::new("col2", 2)) as Arc<dyn PhysicalExpr>,
            ),
        ];
        let join_multi = HashJoinExec::try_new(
            Arc::new(left_exec()),
            Arc::new(right_exec()),
            on_multi,
            None,
            &JoinType::Left,
            None,
            PartitionMode::CollectLeft,
            NullEquality::NullEqualsNothing,
            false,
        )
        .unwrap()
        .with_join_schema(out_schema());
        assert_eq!(
            format!("{join_multi}"),
            "HashJoinExec: mode=CollectLeft, join_type=Left, on=[(id@0, id@0), (name@1, col2@2)]"
        );

        // Config 3: Full join with NullEquality::NullEqualsNull and
        // null_aware=true — exercises the trailing display flags. We
        // construct the plan but do NOT execute it (Full panics at execute
        // until the planner emits it).
        let join_full = HashJoinExec::try_new(
            Arc::new(left_exec()),
            Arc::new(right_exec()),
            on_id_id(),
            None,
            &JoinType::Full,
            None,
            PartitionMode::Partitioned,
            NullEquality::NullEqualsNull,
            true,
        )
        .unwrap()
        .with_join_schema(out_schema());
        assert_eq!(
            format!("{join_full}"),
            "HashJoinExec: mode=Partitioned, join_type=Full, on=[(id@0, id@0)], NullsEqual: true, null_aware"
        );
    }

    /// Drive the full `displayable(plan).indent(false)` pipeline — the
    /// path EXPLAIN uses. The operator's first line through the tree
    /// walker must match DataFusion's exact string.
    #[test]
    fn displayable_indent_default_first_line() {
        let plan: Arc<dyn ExecutionPlan> = Arc::new(build_inner_join());
        let rendered = format!(
            "{}",
            crate::display::displayable(plan.as_ref()).indent(false)
        );
        let first_line = rendered.lines().next().unwrap();
        assert_eq!(
            first_line,
            "HashJoinExec: mode=Partitioned, join_type=Inner, on=[(id@0, id@0)]"
        );
    }

    /// Compile-time confirmation that the accessor surface matches
    /// DataFusion's `HashJoinExec`. Each `let _:` line forces the compiler to
    /// check the method exists with the exact name and signature shape.
    #[test]
    fn accessor_method_names_match_datafusion() {
        let _: fn(&HashJoinExec) -> &Arc<dyn ExecutionPlan> = HashJoinExec::left;
        let _: fn(&HashJoinExec) -> &Arc<dyn ExecutionPlan> = HashJoinExec::right;
        let _: fn(&HashJoinExec) -> JoinOnRef<'_> = HashJoinExec::on;
        let _: fn(&HashJoinExec) -> Option<&JoinFilter> = HashJoinExec::filter;
        let _: fn(&HashJoinExec) -> &JoinType = HashJoinExec::join_type;
        let _: fn(&HashJoinExec) -> &Schema = HashJoinExec::join_schema;
        let _: fn(&HashJoinExec) -> &PartitionMode = HashJoinExec::partition_mode;
        let _: fn(&HashJoinExec) -> NullEquality = HashJoinExec::null_equality;
        let _: fn(&HashJoinExec) -> &[ColumnIndex] = HashJoinExec::column_indices;
    }
}
