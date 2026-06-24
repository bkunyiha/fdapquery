//!
//! Group-by hash aggregation — the trickiest operator in the module (ARCHITECTURE
//! §4.6). It maintains a hash map keyed by the group-by values; each input row is
//! folded into that key's per-aggregate [`Accumulator`]s. When the input is
//! exhausted it emits one output row per key: the group values followed by each
//! aggregate's result. Aggregation is *blocking* — it must see every input row
//! before it can emit anything — so `execute` consumes the whole input eagerly and
//! returns a single output batch.
//!
//! ## The group key
//! [`GroupKey`] wraps `Vec<ScalarValue>` with `Hash`/`Eq` impls. Floats are hashed
//! and compared **by bit pattern**, so the two agree and `NaN` keys group together.
//! `ScalarValue` itself is left unchanged (it stays `PartialEq`-only, since float
//! `Eq`/`Hash` is meaningful only in this grouping context).
//!
//! ## Modes
//! Single-node `Complete` (the default, and the only mode used until the
//! `distributed` module 15) calls `accumulate` + `final_value`. `Final` merges
//! incoming partial state; `Partial` would emit intermediate state. AVG's
//! intermediate state is compound ([`AccumulatorValue::AvgState`]) and cannot sit
//! in a scalar output column, so a `Partial` AVG output panics until the
//! distributed module supplies the intermediate-state schema.

use crate::aggregate_expression::AggregateExpression;
use crate::aggregate_mode::AggregateMode;
use crate::expressions::{Accumulator, AccumulatorValue, Expression};
use crate::physical_plan::ExecutionPlan;
use crate::plan_properties::PlanProperties;
use crate::stream::{RecordBatchStreamAdapter, SendableRecordBatchStream};
use crate::task_context::TaskContext;
use async_stream::try_stream;
use fdapquery_datatypes::{
    ArrowVectorBuilder, ColumnVector, FdapQueryError, Result, ScalarValue, Schema, record_batch,
};
use futures::StreamExt;
use std::collections::HashMap;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

/// Group-by hash aggregation.
pub struct HashAggregateExec {
    pub input: Arc<dyn ExecutionPlan>,
    pub group_expr: Vec<Arc<dyn Expression>>,
    pub aggregate_expr: Vec<Arc<dyn AggregateExpression>>,
    pub schema: Schema,
    pub mode: AggregateMode,
    properties: PlanProperties,
}

impl HashAggregateExec {
    /// Single-node (`Complete`) aggregation — the common case.
    pub fn new(
        input: Arc<dyn ExecutionPlan>,
        group_expr: Vec<Arc<dyn Expression>>,
        aggregate_expr: Vec<Arc<dyn AggregateExpression>>,
        schema: Schema,
    ) -> Self {
        Self::new_with_mode(
            input,
            group_expr,
            aggregate_expr,
            schema,
            AggregateMode::Complete,
        )
    }

    /// Construct with an explicit [`AggregateMode`] (for distributed execution).
    pub fn new_with_mode(
        input: Arc<dyn ExecutionPlan>,
        group_expr: Vec<Arc<dyn Expression>>,
        aggregate_expr: Vec<Arc<dyn AggregateExpression>>,
        schema: Schema,
        mode: AggregateMode,
    ) -> Self {
        let properties = PlanProperties::single_partition_unknown();
        Self {
            input,
            group_expr,
            aggregate_expr,
            schema,
            mode,
            properties,
        }
    }
}

impl ExecutionPlan for HashAggregateExec {
    fn name(&self) -> &str {
        "HashAggregateExec"
    }

    fn schema(&self) -> Schema {
        self.schema.clone()
    }

    fn properties(&self) -> &PlanProperties {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    /// Override the `as_any` hook so `ParallelContext` can downcast and
    /// recover the concrete aggregate for its partial/final split.
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    /// Rebuild this aggregate with a new input child. Arity 1. We use
    /// `new_with_mode` (not `new`) so the `mode` (Complete / Partial /
    /// Final) is preserved through the rewrite.
    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if children.len() != 1 {
            return Err(FdapQueryError::Internal(format!(
                "HashAggregateExec::with_new_children expected 1 child, got {}",
                children.len()
            )));
        }
        Ok(Arc::new(HashAggregateExec::new_with_mode(
            children.into_iter().next().unwrap(),
            self.group_expr.clone(),
            self.aggregate_expr.clone(),
            self.schema.clone(),
            self.mode,
        )))
    }

    fn execute(
        &self,
        partition: usize,
        ctx: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        if partition != 0 {
            return Err(FdapQueryError::Internal(format!(
                "HashAggregateExec has 1 output partition; partition {partition} is out of range"
            )));
        }
        // Capture everything the generator body needs by clone — the
        // generator runs detached from `self`, so it can't hold &self.
        let group_expr = self.group_expr.clone();
        let aggregate_expr = self.aggregate_expr.clone();
        let schema = self.schema.clone();
        let mode = self.mode;
        let n_group = self.group_expr.len();

        let input_stream = self.input.execute(0, Arc::clone(&ctx))?;
        let arrow_schema = Arc::new(self.schema.to_arrow());

        // The aggregator is blocking: it must see every input batch before
        // it can emit its single output batch. The `try_stream!` macro lets
        // us express that as a sequential body that `.await`s on input and
        // `yield`s the result.
        let stream = try_stream! {
            let mut map: HashMap<GroupKey, Vec<Box<dyn Accumulator>>> = HashMap::new();
            let mut input = std::pin::pin!(input_stream);
            while let Some(batch_res) = input.next().await {
                let batch = batch_res?;
                // Evaluate the group-by and aggregate-input expressions once per batch.
                let group_keys: Vec<Box<dyn ColumnVector>> = group_expr
                    .iter()
                    .map(|e| e.evaluate(&batch))
                    .collect::<Result<Vec<_>>>()?;
                let aggr_inputs: Vec<Box<dyn ColumnVector>> = aggregate_expr
                    .iter()
                    .map(|a| a.input_expression().evaluate(&batch))
                    .collect::<Result<Vec<_>>>()?;

                for row in 0..batch.num_rows() {
                    let key = GroupKey(
                        group_keys
                            .iter()
                            .map(|c| c.get_value(row))
                            .collect::<Result<Vec<_>>>()?,
                    );
                    let accumulators = map.entry(key).or_insert_with(|| {
                        aggregate_expr
                            .iter()
                            .map(|a| a.create_accumulator())
                            .collect()
                    });
                    for (i, acc) in accumulators.iter_mut().enumerate() {
                        let value = aggr_inputs[i].get_value(row)?;
                        match mode {
                            // FINAL merges incoming partial state; other modes accumulate raw values.
                            AggregateMode::Final => {
                                acc.merge(&AccumulatorValue::Scalar(value))?;
                            }
                            _ => acc.accumulate(&value)?,
                        }
                    }
                }
            }

            // Build the output batch: one row per group key.
            let mut builders: Vec<ArrowVectorBuilder> = schema
                .fields
                .iter()
                .map(|f| ArrowVectorBuilder::new(&f.data_type, map.len()))
                .collect();

            for (key, accumulators) in &map {
                for (i, group_value) in key.0.iter().enumerate() {
                    builders[i].append_value(group_value);
                }
                for (i, acc) in accumulators.iter().enumerate() {
                    // Inner-Result trick: wrap each match arm so the whole
                    // match evaluates to `Result<ScalarValue>` that we apply
                    // `?` to, avoiding an `unreachable!()` after the AVG-state
                    // error path.
                    let output: ScalarValue = match mode {
                        AggregateMode::Partial => match acc.intermediate_value()? {
                            AccumulatorValue::Scalar(s) => Ok(s),
                            AccumulatorValue::AvgState { .. } => Err(FdapQueryError::NotImplemented(
                                "HashAggregateExec PARTIAL output of AVG intermediate state \
                                 requires the distributed module"
                                    .into(),
                            )),
                        }?,
                        _ => acc.final_value()?,
                    };
                    builders[n_group + i].append_value(&output);
                }
            }

            let columns: Vec<Box<dyn ColumnVector>> = builders
                .into_iter()
                .map(|b| Box::new(b.build()) as Box<dyn ColumnVector>)
                .collect();
            let batch = record_batch::create(&schema, columns)?;
            yield batch;
        };

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            arrow_schema,
            stream,
        )))
    }
}

impl fmt::Display for HashAggregateExec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let group: Vec<String> = self.group_expr.iter().map(|e| e.to_string()).collect();
        let aggr: Vec<String> = self.aggregate_expr.iter().map(|e| e.to_string()).collect();
        write!(
            f,
            "HashAggregateExec: groupExpr=[{}], aggrExpr=[{}], mode={:?}",
            group.join(", "),
            aggr.join(", "),
            self.mode
        )
    }
}

/// Hash-map key for one group: the tuple of group-by values for a row.
/// equivalent. Floats are hashed/compared by bit pattern so `Hash` and `Eq` agree.
#[derive(Clone)]
struct GroupKey(Vec<ScalarValue>);

impl PartialEq for GroupKey {
    fn eq(&self, other: &Self) -> bool {
        self.0.len() == other.0.len()
            && self
                .0
                .iter()
                .zip(&other.0)
                .all(|(a, b)| scalar_key_eq(a, b))
    }
}

impl Eq for GroupKey {}

impl Hash for GroupKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for v in &self.0 {
            hash_scalar(v, state);
        }
    }
}

/// Equality used for group keys: bit-equality for floats (so `NaN == NaN`, to agree
/// with [`hash_scalar`]); the derived `PartialEq` for everything else.
fn scalar_key_eq(a: &ScalarValue, b: &ScalarValue) -> bool {
    use ScalarValue::*;
    match (a, b) {
        (Float32(x), Float32(y)) => x.to_bits() == y.to_bits(),
        (Float64(x), Float64(y)) => x.to_bits() == y.to_bits(),
        _ => a == b,
    }
}

/// Hash one scalar: the variant discriminant plus the value's bytes (floats by bit
/// pattern, so equal-keyed floats hash equally).
fn hash_scalar<H: Hasher>(v: &ScalarValue, state: &mut H) {
    use ScalarValue::*;
    std::mem::discriminant(v).hash(state);
    match v {
        Null => {}
        Boolean(b) => b.hash(state),
        Int8(n) => n.hash(state),
        Int16(n) => n.hash(state),
        Int32(n) => n.hash(state),
        Int64(n) => n.hash(state),
        UInt8(n) => n.hash(state),
        UInt16(n) => n.hash(state),
        UInt32(n) => n.hash(state),
        UInt64(n) => n.hash(state),
        Float32(f) => f.to_bits().hash(state),
        Float64(f) => f.to_bits().hash(state),
        Utf8(s) => s.hash(state),
        Binary(b) => b.hash(state),
        Date32(d) => d.hash(state),
    }
}

#[cfg(test)]
mod tests {
    //! Accumulator tests plus a group-by integration test over `employee.csv`
    //! (the §4.6 snapshot check). The accumulators are driven directly and the
    //! integration test builds the physical plan by hand (the `query-planner`
    //! that normally assembles it is covered in module 7).
    use super::*;
    use crate::column_expression::ColumnExpression;
    use crate::count_expression::CountExpression;
    use crate::max_expression::MaxExpression;
    use crate::min_expression::MinExpression;
    use crate::scan_exec::ScanExec;
    use crate::sum_expression::SumExpression;
    use fdapquery_datasource::{CsvDataSource, DataSource};
    use fdapquery_datatypes::Field;
    use fdapquery_datatypes::arrow_types::{INT32_TYPE, INT64_TYPE, STRING_TYPE};
    use futures::TryStreamExt;

    // ---- Accumulators driven directly. ----

    #[test]
    fn min_accumulator() {
        let mut a = MinExpression::new(Arc::new(ColumnExpression::new(0))).create_accumulator();
        for v in [10, 14, 4] {
            a.accumulate(&ScalarValue::Int32(v)).unwrap();
        }
        assert_eq!(a.final_value().unwrap(), ScalarValue::Int32(4));
    }

    #[test]
    fn max_accumulator() {
        let mut a = MaxExpression::new(Arc::new(ColumnExpression::new(0))).create_accumulator();
        for v in [10, 14, 4] {
            a.accumulate(&ScalarValue::Int32(v)).unwrap();
        }
        assert_eq!(a.final_value().unwrap(), ScalarValue::Int32(14));
    }

    #[test]
    fn sum_accumulator() {
        let mut a = SumExpression::new(Arc::new(ColumnExpression::new(0))).create_accumulator();
        for v in [10, 14, 4] {
            a.accumulate(&ScalarValue::Int32(v)).unwrap();
        }
        assert_eq!(a.final_value().unwrap(), ScalarValue::Int32(28));
    }

    // ---- Integration: GROUP BY state, MIN/MAX/COUNT(salary) over employee.csv. ----

    #[tokio::test]
    async fn group_by_state_min_max_count() {
        let ds: Arc<dyn DataSource> = Arc::new(CsvDataSource::new(
            "../testdata/employee.csv",
            None,
            true,
            1024,
        ));
        let all: Vec<String> = ds.schema().fields.iter().map(|f| f.name.clone()).collect();
        let scan = ScanExec::new(Arc::clone(&ds), all).unwrap();

        // Output: state, MIN(salary), MAX(salary), COUNT(salary).
        let out_schema = Schema::new(vec![
            Field::new("state", STRING_TYPE),
            Field::new("min_salary", INT64_TYPE),
            Field::new("max_salary", INT64_TYPE),
            Field::new("count_salary", INT32_TYPE),
        ]);
        // employee.csv columns: 0=id 1=first_name 2=last_name 3=state 4=job_title 5=salary
        let agg = HashAggregateExec::new(
            Arc::new(scan),
            vec![Arc::new(ColumnExpression::new(3))],
            vec![
                Arc::new(MinExpression::new(Arc::new(ColumnExpression::new(5)))),
                Arc::new(MaxExpression::new(Arc::new(ColumnExpression::new(5)))),
                Arc::new(CountExpression::new(Arc::new(ColumnExpression::new(5)))),
            ],
            out_schema,
        );

        let ctx = Arc::new(TaskContext::default_test());
        let batches: Vec<_> = agg
            .execute(0, ctx)
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        assert_eq!(batches.len(), 1);
        let batch = &batches[0];
        assert_eq!(batch.num_rows(), 3); // groups: CA, CO, and the null-state row

        let states = record_batch::field(batch, 0);
        let mins = record_batch::field(batch, 1);
        let maxs = record_batch::field(batch, 2);
        let counts = record_batch::field(batch, 3);

        let mut got: HashMap<Option<String>, (i64, i64, i32)> = HashMap::new();
        for i in 0..batch.num_rows() {
            let state = match states.get_value(i).unwrap() {
                ScalarValue::Utf8(s) => Some(s),
                ScalarValue::Null => None,
                other => panic!("unexpected state value: {other:?}"),
            };
            let mn = match mins.get_value(i).unwrap() {
                ScalarValue::Int64(n) => n,
                o => panic!("min: {o:?}"),
            };
            let mx = match maxs.get_value(i).unwrap() {
                ScalarValue::Int64(n) => n,
                o => panic!("max: {o:?}"),
            };
            let c = match counts.get_value(i).unwrap() {
                ScalarValue::Int32(n) => n,
                o => panic!("count: {o:?}"),
            };
            got.insert(state, (mn, mx, c));
        }

        assert_eq!(got.get(&Some("CA".to_string())), Some(&(12000, 12000, 1)));
        assert_eq!(got.get(&Some("CO".to_string())), Some(&(10000, 11500, 2)));
        assert_eq!(got.get(&None), Some(&(11500, 11500, 1)));
    }
}
