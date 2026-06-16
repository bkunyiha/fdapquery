//!
//! Hash equi-join. Builds a hash table from the **right** (build) side keyed by
//! the right join columns, then probes it with each **left** (probe) row. Supports
//! `Inner`, `Left`, and `Right` joins (the three variants of `logical_plan::JoinType`).
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

use crate::executor_context::ExecutorContext;
use crate::physical_plan::PhysicalPlan;
use crate::row_key::RowKey;
use datatypes::{
    ArrowFieldVector, ArrowVectorBuilder, ColumnVector, RecordBatch, ScalarValue, Schema,
    record_batch,
};
use logical_plan::JoinType;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// Hash join physical operator.
pub struct HashJoinExec {
    pub left: Arc<dyn PhysicalPlan>,
    pub right: Arc<dyn PhysicalPlan>,
    pub join_type: JoinType,
    pub left_keys: Vec<usize>,
    pub right_keys: Vec<usize>,
    pub schema: Schema,
    pub right_columns_to_exclude: HashSet<usize>,
}

impl HashJoinExec {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        left: Arc<dyn PhysicalPlan>,
        right: Arc<dyn PhysicalPlan>,
        join_type: JoinType,
        left_keys: Vec<usize>,
        right_keys: Vec<usize>,
        schema: Schema,
        right_columns_to_exclude: HashSet<usize>,
    ) -> Self {
        Self {
            left,
            right,
            join_type,
            left_keys,
            right_keys,
            schema,
            right_columns_to_exclude,
        }
    }

    /// Concatenate a left row with a right row, dropping the right columns listed
    /// in `right_columns_to_exclude` (the duplicate join keys).
    fn combine_rows(
        &self,
        left_row: &[ScalarValue],
        right_row: &[ScalarValue],
    ) -> Vec<ScalarValue> {
        let mut result: Vec<ScalarValue> = left_row.to_vec();
        for (i, value) in right_row.iter().enumerate() {
            if !self.right_columns_to_exclude.contains(&i) {
                result.push(value.clone());
            }
        }
        result
    }

    /// Build an output batch from assembled rows, typed by the output schema.
    fn create_batch(&self, rows: &[Vec<ScalarValue>]) -> RecordBatch {
        let mut builders: Vec<ArrowVectorBuilder> = self
            .schema
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
        record_batch::create(&self.schema, columns)
    }

    /// Wrap each column of `batch` once, so rows can be read by index without
    /// re-wrapping the arrays per row.
    fn columns_of(batch: &RecordBatch) -> Vec<ArrowFieldVector> {
        (0..batch.num_columns())
            .map(|i| record_batch::field(batch, i))
            .collect()
    }

    /// The join key for one row: the values of the given key columns.
    fn key_of(cols: &[ArrowFieldVector], keys: &[usize], row: usize) -> RowKey {
        RowKey(keys.iter().map(|&k| cols[k].get_value(row)).collect())
    }

    /// Every column value for one row.
    fn full_row(cols: &[ArrowFieldVector], row: usize) -> Vec<ScalarValue> {
        cols.iter().map(|c| c.get_value(row)).collect()
    }
}

impl PhysicalPlan for HashJoinExec {
    fn schema(&self) -> Schema {
        self.schema.clone()
    }

    fn children(&self) -> Vec<&Arc<dyn PhysicalPlan>> {
        vec![&self.left, &self.right]
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    /// Rebuild this join with new left and right inputs. See the trait-level
    /// `PhysicalPlan::with_new_children` doc for the general rewrite pattern.
    ///
    /// Arity 2: a hash join has two inputs, conventionally `[left, right]` in
    /// `children()` order. The incoming `children` vec is therefore length 2.
    /// We drain it via `into_iter()` and take each element in order — the
    /// first is the left (probe) side, the second is the right (build) side.
    /// Ordering matters: swapping left and right changes the hash table's
    /// key columns and would silently produce a different join.
    ///
    /// Both new inputs are taken by owned move (no Arc clones). All non-input
    /// fields (`join_type`, key indices, output schema, exclude set) are
    /// reused — they're properties of the join definition, not the children.
    ///
    /// DataFusion equivalently writes `children[0].clone()` / `children[1].clone()`,
    /// which trades two atomic refcount bumps for terseness. Both are correct.
    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn PhysicalPlan>>,
    ) -> Arc<dyn PhysicalPlan> {
        assert_eq!(
            children.len(),
            2,
            "HashJoinExec expects exactly 2 children (left, right)"
        );
        let mut iter = children.into_iter();
        let left = iter.next().unwrap();
        let right = iter.next().unwrap();
        Arc::new(HashJoinExec::new(
            left,
            right,
            self.join_type.clone(),
            self.left_keys.clone(),
            self.right_keys.clone(),
            self.schema.clone(),
            self.right_columns_to_exclude.clone(),
        ))
    }

    fn execute(&self, ctx: &ExecutorContext) -> Box<dyn Iterator<Item = RecordBatch>> {
        // Join doesn't read ctx itself; threads it through to both children
        // so shuffle-bearing inputs find their executor state.
        let mut hash_table: HashMap<RowKey, Vec<Vec<ScalarValue>>> = HashMap::new();
        for batch in self.right.execute(ctx) {
            let cols = Self::columns_of(&batch);
            for row in 0..batch.num_rows() {
                let key = Self::key_of(&cols, &self.right_keys, row);
                hash_table
                    .entry(key)
                    .or_default()
                    .push(Self::full_row(&cols, row));
            }
        }

        let right_field_count = self.right.schema().fields.len();
        let mut outputs: Vec<RecordBatch> = Vec::new();

        // --- Probe phase: find matches for each left row. ---
        for left_batch in self.left.execute(ctx) {
            let cols = Self::columns_of(&left_batch);
            let mut output_rows: Vec<Vec<ScalarValue>> = Vec::new();
            for row in 0..left_batch.num_rows() {
                let probe_key = Self::key_of(&cols, &self.left_keys, row);
                let left_row = Self::full_row(&cols, row);
                let matched = hash_table.get(&probe_key);
                match self.join_type {
                    JoinType::Inner | JoinType::Right => {
                        if let Some(rows) = matched {
                            for right_row in rows {
                                output_rows.push(self.combine_rows(&left_row, right_row));
                            }
                        }
                    }
                    JoinType::Left => {
                        if let Some(rows) = matched {
                            for right_row in rows {
                                output_rows.push(self.combine_rows(&left_row, right_row));
                            }
                        } else {
                            // No match: left row with nulls for the right columns.
                            let null_right = vec![ScalarValue::Null; right_field_count];
                            output_rows.push(self.combine_rows(&left_row, &null_right));
                        }
                    }
                }
            }
            if !output_rows.is_empty() {
                outputs.push(self.create_batch(&output_rows));
            }
        }

        // --- Right join: emit unmatched right rows with nulls on the left. ---
        if matches!(self.join_type, JoinType::Right) {
            let mut matched_keys: HashSet<RowKey> = HashSet::new();
            for left_batch in self.left.execute(ctx) {
                let cols = Self::columns_of(&left_batch);
                for row in 0..left_batch.num_rows() {
                    let probe_key = Self::key_of(&cols, &self.left_keys, row);
                    if hash_table.contains_key(&probe_key) {
                        matched_keys.insert(probe_key);
                    }
                }
            }
            let left_field_count = self.left.schema().fields.len();
            let mut unmatched: Vec<Vec<ScalarValue>> = Vec::new();
            for (key, rows) in &hash_table {
                if !matched_keys.contains(key) {
                    let null_left = vec![ScalarValue::Null; left_field_count];
                    for right_row in rows {
                        unmatched.push(self.combine_rows(&null_left, right_row));
                    }
                }
            }
            if !unmatched.is_empty() {
                outputs.push(self.create_batch(&unmatched));
            }
        }

        Box::new(outputs.into_iter())
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
    use datatypes::Field;
    use datatypes::arrow_types::{INT64_TYPE, STRING_TYPE};
    use std::sync::Arc;

    /// A `PhysicalPlan` that simply replays preset batches.
    struct VecExec {
        schema: Schema,
        batches: Vec<RecordBatch>,
    }

    impl PhysicalPlan for VecExec {
        fn schema(&self) -> Schema {
            self.schema.clone()
        }
        fn execute(&self, _ctx: &ExecutorContext) -> Box<dyn Iterator<Item = RecordBatch>> {
            Box::new(self.batches.clone().into_iter())
        }
        fn children(&self) -> Vec<&Arc<dyn PhysicalPlan>> {
            vec![]
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
        fn with_new_children(
            self: Arc<Self>,
            children: Vec<Arc<dyn PhysicalPlan>>,
        ) -> Arc<dyn PhysicalPlan> {
            assert!(children.is_empty());
            self
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
        VecExec {
            schema,
            batches: vec![RecordBatch::try_new(arrow, vec![id, name]).unwrap()],
        }
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
        VecExec {
            schema,
            batches: vec![RecordBatch::try_new(arrow, vec![id, dept]).unwrap()],
        }
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
                let id = match c0.get_value(i) {
                    ScalarValue::Int64(n) => Some(n),
                    ScalarValue::Null => None,
                    o => panic!("id: {o:?}"),
                };
                let name = match c1.get_value(i) {
                    ScalarValue::Utf8(s) => Some(s),
                    ScalarValue::Null => None,
                    o => panic!("name: {o:?}"),
                };
                let dept = match c2.get_value(i) {
                    ScalarValue::Utf8(s) => Some(s),
                    ScalarValue::Null => None,
                    o => panic!("dept: {o:?}"),
                };
                out.push((id, name, dept));
            }
        }
        out
    }

    #[test]
    fn inner_join_on_id() {
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
            join.execute(&ExecutorContext::new(
                "test",
                "localhost",
                0,
                "/tmp/rquery-test-ignored",
            ))
            .collect(),
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

    #[test]
    fn left_join_keeps_unmatched_left() {
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
            join.execute(&ExecutorContext::new(
                "test",
                "localhost",
                0,
                "/tmp/rquery-test-ignored",
            ))
            .collect(),
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
