//!
//! Hash equi-join. Builds a hash table from the **right** (build) side keyed by
//! the right join columns, then probes it with each **left** (probe) row. Supports
//! `Inner`, `Left`, and `Right` joins (the three variants of `fdapquery_logical_plan::JoinType`).
//!
//! ## Implementation notes
//! - **Join keys / rows are `Vec<ScalarValue>`.** The hash table is keyed by
//!   [`crate::row_key::RowKey`] — the same float-aware key helper
//!   `HashAggregateExec` uses for group keys (§4.6 asked for a shared helper).
//!   String columns surface as `ScalarValue::Utf8`, so no extra normalization
//!   is needed.
//! - **`rightColumnsToExclude`** drops duplicate join-key columns from the right
//!   side of the combined row (so an `id = id` join doesn't emit `id` twice).
//! - **Eager, not lazy.** The build side must be fully materialized first anyway;
//!   the join collects all output batches and returns `outputs.into_iter()`.
//! - **Right join** re-scans the left side to find which right keys matched, then
//!   emits the unmatched right rows with nulls on the left.

use crate::physical_plan::ExecutionPlan;
use crate::plan_properties::PlanProperties;
use crate::row_key::RowKey;
use crate::stream::{RecordBatchStreamAdapter, SendableRecordBatchStream};
use crate::task_context::TaskContext;
use async_stream::try_stream;
use fdapquery_datatypes::{
    ArrowFieldVector, ArrowVectorBuilder, ColumnVector, FdapQueryError, RecordBatch, Result,
    ScalarValue, Schema, record_batch,
};
use fdapquery_logical_plan::JoinType;
use futures::StreamExt;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// Hash join physical operator.
pub struct HashJoinExec {
    pub left: Arc<dyn ExecutionPlan>,
    pub right: Arc<dyn ExecutionPlan>,
    pub join_type: JoinType,
    pub left_keys: Vec<usize>,
    pub right_keys: Vec<usize>,
    pub schema: Schema,
    pub right_columns_to_exclude: HashSet<usize>,
    properties: PlanProperties,
}

impl HashJoinExec {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        left: Arc<dyn ExecutionPlan>,
        right: Arc<dyn ExecutionPlan>,
        join_type: JoinType,
        left_keys: Vec<usize>,
        right_keys: Vec<usize>,
        schema: Schema,
        right_columns_to_exclude: HashSet<usize>,
    ) -> Self {
        let properties = PlanProperties::single_partition_unknown();
        Self {
            left,
            right,
            join_type,
            left_keys,
            right_keys,
            schema,
            right_columns_to_exclude,
            properties,
        }
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
        .fields
        .iter()
        .map(|f| ArrowVectorBuilder::new(&f.data_type, rows.len()))
        .collect();
    for row in rows {
        for (col, value) in row.iter().enumerate() {
            builders[col].append_value(value);
        }
    }
    let columns: Vec<Box<dyn ColumnVector>> = builders
        .into_iter()
        .map(|b| Box::new(b.build()) as Box<dyn ColumnVector>)
        .collect();
    record_batch::create(schema, columns)
}

/// Wrap each column of `batch` once, so rows can be read by index without
/// re-wrapping the arrays per row.
fn columns_of(batch: &RecordBatch) -> Vec<ArrowFieldVector> {
    (0..batch.num_columns())
        .map(|i| record_batch::field(batch, i))
        .collect()
}

/// The join key for one row: the values of the given key columns.
fn key_of(cols: &[ArrowFieldVector], keys: &[usize], row: usize) -> Result<RowKey> {
    Ok(RowKey(
        keys.iter()
            .map(|&k| cols[k].get_value(row))
            .collect::<Result<Vec<_>>>()?,
    ))
}

/// Every column value for one row.
fn full_row(cols: &[ArrowFieldVector], row: usize) -> Result<Vec<ScalarValue>> {
    cols.iter().map(|c| c.get_value(row)).collect()
}

impl ExecutionPlan for HashJoinExec {
    fn name(&self) -> &str {
        "HashJoinExec"
    }

    fn schema(&self) -> Schema {
        self.schema.clone()
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
        Ok(Arc::new(HashJoinExec::new(
            left,
            right,
            self.join_type.clone(),
            self.left_keys.clone(),
            self.right_keys.clone(),
            self.schema.clone(),
            self.right_columns_to_exclude.clone(),
        )))
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
        // Clone everything the generator needs — it runs detached from `self`.
        let left = Arc::clone(&self.left);
        let right = Arc::clone(&self.right);
        let join_type = self.join_type.clone();
        let left_keys = self.left_keys.clone();
        let right_keys = self.right_keys.clone();
        let schema = self.schema.clone();
        let right_columns_to_exclude = self.right_columns_to_exclude.clone();
        let right_field_count = self.right.schema().fields.len();
        let left_field_count = self.left.schema().fields.len();
        let arrow_schema = Arc::new(self.schema.to_arrow());
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
                    }
                }
                if !output_rows.is_empty() {
                    yield create_batch(&output_rows, &schema)?;
                }
            }

            // --- Right join: re-scan left to find which right keys matched,
            // then emit the unmatched right rows with nulls on the left.
            // Session 7 did this with a second left.execute(); we keep the
            // same shape. A buffering optimisation that avoided the
            // re-execute lives in the deferred-to-later-session list. ---
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

        Ok(Box::pin(RecordBatchStreamAdapter::new(arrow_schema, stream)))
    }
}

impl std::fmt::Display for HashJoinExec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "HashJoinExec: joinType={}, leftKeys={:?}, rightKeys={:?}",
            self.join_type, self.left_keys, self.right_keys
        )
    }
}

#[cfg(test)]
mod tests {
    //! Join tests. These drive `HashJoinExec` directly via a tiny in-memory
    //! `PhysicalPlan`; the `query-planner` that normally builds a join is
    //! covered in module 7.
    use super::*;
    use arrow_array::{ArrayRef, Int64Array, StringArray};
    use arrow_schema::{Field as ArrowField, Schema as ArrowSchema};
    use fdapquery_datatypes::Field;
    use fdapquery_datatypes::arrow_types::{INT64_TYPE, STRING_TYPE};
    use futures::TryStreamExt;
    use std::sync::Arc;

    /// An `ExecutionPlan` that simply replays preset batches.
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
        fn name(&self) -> &str {
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
            let arrow_schema = Arc::new(self.schema.to_arrow());
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

    impl std::fmt::Display for VecExec {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "VecExec")
        }
    }

    /// left: (id: Int64, name: Utf8) = (1,a),(2,b),(3,c)
    fn left_exec() -> VecExec {
        let schema = Schema::new(vec![
            Field::new("id", INT64_TYPE),
            Field::new("name", STRING_TYPE),
        ]);
        let arrow = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("id", INT64_TYPE, true),
            ArrowField::new("name", STRING_TYPE, true),
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
            Field::new("id", INT64_TYPE),
            Field::new("dept", STRING_TYPE),
        ]);
        let arrow = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("id", INT64_TYPE, true),
            ArrowField::new("dept", STRING_TYPE, true),
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
            Field::new("id", INT64_TYPE),
            Field::new("name", STRING_TYPE),
            Field::new("dept", STRING_TYPE),
        ])
    }

    type Row = (Option<i64>, Option<String>, Option<String>);

    fn collect_rows(batches: Vec<RecordBatch>) -> Vec<Row> {
        let mut out: Vec<Row> = Vec::new();
        for b in &batches {
            let c0 = record_batch::field(b, 0);
            let c1 = record_batch::field(b, 1);
            let c2 = record_batch::field(b, 2);
            for i in 0..b.num_rows() {
                let id = match c0.get_value(i).unwrap() {
                    ScalarValue::Int64(n) => Some(n),
                    ScalarValue::Null => None,
                    o => panic!("id: {o:?}"),
                };
                let name = match c1.get_value(i).unwrap() {
                    ScalarValue::Utf8(s) => Some(s),
                    ScalarValue::Null => None,
                    o => panic!("name: {o:?}"),
                };
                let dept = match c2.get_value(i).unwrap() {
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

    #[tokio::test]
    async fn inner_join_on_id() {
        let join = HashJoinExec::new(
            Arc::new(left_exec()),
            Arc::new(right_exec()),
            JoinType::Inner,
            vec![0],
            vec![0],
            out_schema(),
            HashSet::from([0]),
        );
        let mut rows = collect_rows(
            join.execute(0, test_ctx())
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
        let join = HashJoinExec::new(
            Arc::new(left_exec()),
            Arc::new(right_exec()),
            JoinType::Left,
            vec![0],
            vec![0],
            out_schema(),
            HashSet::from([0]),
        );
        let mut rows = collect_rows(
            join.execute(0, test_ctx())
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
}
