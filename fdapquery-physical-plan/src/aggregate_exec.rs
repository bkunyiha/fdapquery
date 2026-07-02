//!
//! Group-by hash aggregation — the trickiest operator in the module (ARCHITECTURE
//! §4.6). It maintains a hash map keyed by the group-by values; each input row is
//! folded into that key's per-aggregate [`Accumulator`]s. When the input is
//! exhausted it emits one output row per key: the group values followed by each
//! aggregate's result. Aggregation is *blocking* — it must see every input row
//! before it can emit anything — so `execute` consumes the whole input eagerly and
//! returns a single output batch.
//!
//! ## Strict mirror of DataFusion
//! Struct field names (`mode`, `group_by`, `aggr_expr`, `filter_expr`, `input`,
//! `schema`, `input_schema`), constructor signature (`try_new(mode, group_by,
//! aggr_expr, filter_expr, input, input_schema)`), accessor names (`mode()`,
//! `group_expr()`, `aggr_expr()`, `filter_expr()`, `input()`, `input_schema()`),
//! and the `DisplayAs::fmt_as` `Default`/`Verbose` output
//! (`"AggregateExec: mode={Mode:?}, gby=[…], aggr=[…]"`) match
//! `datafusion/physical-plan/src/aggregates/mod.rs` byte-for-byte. fdapquery's
//! `aggr_expr` element type is the trait object `Arc<dyn AggregateExpr>`
//! (vs. DataFusion's concrete `Arc<AggregateFunctionExpr>`).
//!
//! ## The group key
//! `GroupKey` wraps `Vec<ScalarValue>` with `Hash`/`Eq` impls. Floats are hashed
//! and compared **by bit pattern**, so the two agree and `NaN` keys group together.
//! `ScalarValue` itself is left unchanged (it stays `PartialEq`-only, since float
//! `Eq`/`Hash` is meaningful only in this grouping context).
//!
//! ## Modes
//! Single-node `Single` (the default, and the only mode used until the
//! `fdapquery-distributed` two-stage aggregate path) calls `accumulate` +
//! `final_value`. `Final` merges
//! incoming partial state; `Partial` would emit intermediate state. AVG's
//! intermediate state is compound ([`AccumulatorValue::AvgState`]) and cannot sit
//! in a scalar output column, so a `Partial` AVG output panics until the
//! distributed module supplies the intermediate-state schema.

use crate::AggregateExpr;
use crate::AggregateMode;
use crate::aggregates::PhysicalGroupBy;
use crate::physical_plan::ExecutionPlan;
use crate::plan_properties::PlanProperties;
use crate::stream::{RecordBatchStreamAdapter, SendableRecordBatchStream};
use crate::{Accumulator, AccumulatorValue, PhysicalExpr};
use arrow_array::ArrayRef;
use async_stream::try_stream;
use fdapquery_common::{ArrowVectorBuilder, FdapQueryError, Result, ScalarValue};
use fdapquery_datatypes::{Schema, record_batch};
use fdapquery_execution::TaskContext;
use futures::StreamExt;
use std::collections::HashMap;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

/// Hash aggregate execution plan
#[derive(Debug)]
pub struct AggregateExec {
    /// Aggregation mode (full, partial)
    mode: AggregateMode,
    /// Group by expressions
    group_by: Arc<PhysicalGroupBy>,
    /// Aggregate expressions
    aggr_expr: Vec<Arc<dyn AggregateExpr>>,
    /// FILTER (WHERE clause) expression for each aggregate expression
    filter_expr: Vec<Option<Arc<dyn PhysicalExpr>>>,
    /// Input plan, could be a partial aggregate or the input to the aggregate
    pub input: Arc<dyn ExecutionPlan>,
    /// Schema after the aggregate is applied. Contains the group by columns followed by the
    /// aggregate outputs.
    schema: Schema,
    /// Input schema before any aggregation is applied. For partial aggregate this will be the
    /// same as input.schema() but for the final aggregate it will be the same as the input
    /// to the partial aggregate, i.e., partial and final aggregates have same `input_schema`.
    pub input_schema: Schema,
    properties: PlanProperties,
}

impl AggregateExec {
    /// Create a new hash aggregate execution plan.
    ///
    /// Strict mirror of DataFusion's `AggregateExec::try_new` (parameter order,
    /// names, types). fdapquery diverges from DataFusion only in
    /// `Result`/`SchemaRef` types: `FdapQueryError` instead of
    /// `DataFusionError`, and `Schema` (cloneable) instead of `Arc<Schema>`.
    /// The output `schema` argument is also passed explicitly — DataFusion
    /// builds it internally via `create_schema(&input.schema(), &group_by,
    /// &aggr_expr, mode)`, but fdapquery's planner already has it in hand.
    pub fn try_new(
        mode: AggregateMode,
        group_by: impl Into<Arc<PhysicalGroupBy>>,
        aggr_expr: Vec<Arc<dyn AggregateExpr>>,
        filter_expr: Vec<Option<Arc<dyn PhysicalExpr>>>,
        input: Arc<dyn ExecutionPlan>,
        input_schema: Schema,
        schema: Schema,
    ) -> Result<Self> {
        let group_by = group_by.into();
        let properties = PlanProperties::single_partition_unknown();
        Ok(Self {
            mode,
            group_by,
            aggr_expr,
            filter_expr,
            input,
            schema,
            input_schema,
            properties,
        })
    }

    /// Aggregation mode (full, partial)
    pub fn mode(&self) -> &AggregateMode {
        &self.mode
    }

    /// Grouping expressions
    pub fn group_expr(&self) -> &PhysicalGroupBy {
        &self.group_by
    }

    /// Aggregate expressions
    pub fn aggr_expr(&self) -> &[Arc<dyn AggregateExpr>] {
        &self.aggr_expr
    }

    /// FILTER (WHERE clause) expression for each aggregate expression
    pub fn filter_expr(&self) -> &[Option<Arc<dyn PhysicalExpr>>] {
        &self.filter_expr
    }

    /// Input plan
    pub fn input(&self) -> &Arc<dyn ExecutionPlan> {
        &self.input
    }

    /// Get the input schema before any aggregates are applied
    pub fn input_schema(&self) -> Schema {
        self.input_schema.clone()
    }
}

impl ExecutionPlan for AggregateExec {
    fn name(&self) -> &'static str {
        "AggregateExec"
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
    /// `try_new` so the mode (Single / Partial / Final) is preserved through
    /// the rewrite.
    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if children.len() != 1 {
            return Err(FdapQueryError::Internal(format!(
                "AggregateExec::with_new_children expected 1 child, got {}",
                children.len()
            )));
        }
        Ok(Arc::new(AggregateExec::try_new(
            self.mode,
            Arc::clone(&self.group_by),
            self.aggr_expr.clone(),
            self.filter_expr.clone(),
            children.into_iter().next().unwrap(),
            self.input_schema.clone(),
            self.schema.clone(),
        )?))
    }

    fn execute(
        &self,
        partition: usize,
        ctx: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        if partition != 0 {
            return Err(FdapQueryError::Internal(format!(
                "AggregateExec has 1 output partition; partition {partition} is out of range"
            )));
        }
        // Capture everything the generator body needs by clone — the
        // generator runs detached from `self`, so it can't hold &self.
        let group_expr: Vec<Arc<dyn PhysicalExpr>> = self.group_by.input_exprs();
        let aggregate_expr = self.aggr_expr.clone();
        let schema = self.schema.clone();
        let mode = self.mode;
        let n_group = group_expr.len();

        let input_stream = self.input.execute(0, Arc::clone(&ctx))?;
        let arrow_schema = Arc::new(self.schema.clone());

        // The aggregator is blocking: it must see every input batch before
        // it can emit its single output batch. The `try_stream!` macro lets
        // us express that as a sequential body that `.await`s on input and
        // `yield`s the result.
        let stream = try_stream! {
            let mut map: HashMap<GroupKey, Vec<Box<dyn Accumulator>>> = HashMap::new();
            let mut input = std::pin::pin!(input_stream);
            while let Some(batch_res) = input.next().await {
                let batch = batch_res?;
                let num_rows = batch.num_rows();
                // Evaluate the group-by and aggregate-input expressions once per batch
                // and materialize each result to an `ArrayRef` so we can read cells
                // by index via `ScalarValue::try_from_array`.
                let group_keys: Vec<ArrayRef> = group_expr
                    .iter()
                    .map(|e| e.evaluate(&batch)?.into_array(num_rows))
                    .collect::<Result<Vec<_>>>()?;
                let aggr_inputs: Vec<ArrayRef> = aggregate_expr
                    .iter()
                    .map(|a| a.input_expression().evaluate(&batch)?.into_array(num_rows))
                    .collect::<Result<Vec<_>>>()?;

                for row in 0..batch.num_rows() {
                    let key = GroupKey(
                        group_keys
                            .iter()
                            .map(|c| ScalarValue::try_from_array(c, row))
                            .collect::<Result<Vec<_>>>()?,
                    );
                    let accumulators = map.entry(key).or_insert_with(|| {
                        aggregate_expr
                            .iter()
                            .map(|a| a.create_accumulator())
                            .collect()
                    });
                    for (i, acc) in accumulators.iter_mut().enumerate() {
                        let value = ScalarValue::try_from_array(&aggr_inputs[i], row)?;
                        match mode {
                            // FINAL / FinalPartitioned merge incoming partial state.
                            // Other modes accumulate raw values.
                            AggregateMode::Final | AggregateMode::FinalPartitioned => {
                                acc.merge(&AccumulatorValue::Scalar(value))?;
                            }
                            _ => acc.accumulate(&value)?,
                        }
                    }
                }
            }

            // Build the output batch: one row per group key.
            let mut builders: Vec<ArrowVectorBuilder> = schema
                .fields()
                .iter()
                .map(|f| ArrowVectorBuilder::new(f.data_type(), map.len()))
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
                        AggregateMode::Partial | AggregateMode::PartialReduce => {
                            match acc.intermediate_value()? {
                                AccumulatorValue::Scalar(s) => Ok(s),
                                AccumulatorValue::AvgState { .. } => Err(FdapQueryError::NotImplemented(
                                    "AggregateExec PARTIAL output of AVG intermediate state \
                                     requires the distributed module"
                                        .into(),
                                )),
                            }?
                        }
                        _ => acc.final_value()?,
                    };
                    builders[n_group + i].append_value(&output);
                }
            }

            let columns: Vec<ArrayRef> = builders.into_iter().map(|b| b.build()).collect();
            let batch = record_batch::create(&schema, columns)?;
            yield batch;
        };

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            arrow_schema,
            stream,
        )))
    }
}

impl crate::display::DisplayAs for AggregateExec {
    fn fmt_as(
        &self,
        t: crate::display::DisplayFormatType,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        // Helper that mirrors DataFusion's `format_expr_with_alias`: if the
        // expression's `to_string()` already matches the alias, just print
        // the expression; otherwise emit `"{expr} as {alias}"`. For the
        // simple `GROUP BY a, b, …` path the planner constructs each pair as
        // `(expr, expr.to_string())` so the alias collapses out.
        let format_expr_with_alias = |(e, alias): &(Arc<dyn PhysicalExpr>, String)| -> String {
            let e = e.to_string();
            if &e == alias {
                e
            } else {
                format!("{e} as {alias}")
            }
        };

        match t {
            crate::display::DisplayFormatType::Default
            | crate::display::DisplayFormatType::Verbose => {
                write!(f, "AggregateExec: mode={:?}", self.mode)?;
                let g: Vec<String> = if self.group_by.is_single() {
                    self.group_by
                        .expr
                        .iter()
                        .map(format_expr_with_alias)
                        .collect()
                } else {
                    self.group_by
                        .groups
                        .iter()
                        .map(|group| {
                            let terms = group
                                .iter()
                                .enumerate()
                                .map(|(idx, is_null)| {
                                    if *is_null {
                                        format_expr_with_alias(&self.group_by.null_expr[idx])
                                    } else {
                                        format_expr_with_alias(&self.group_by.expr[idx])
                                    }
                                })
                                .collect::<Vec<String>>()
                                .join(", ");
                            format!("({terms})")
                        })
                        .collect()
                };
                write!(f, ", gby=[{}]", g.join(", "))?;

                let a: Vec<String> = self.aggr_expr.iter().map(|agg| agg.to_string()).collect();
                write!(f, ", aggr=[{}]", a.join(", "))?;
                Ok(())
            }
            crate::display::DisplayFormatType::TreeRender => {
                let g: Vec<String> = if self.group_by.is_single() {
                    self.group_by
                        .expr
                        .iter()
                        .map(format_expr_with_alias)
                        .collect()
                } else {
                    self.group_by
                        .groups
                        .iter()
                        .map(|group| {
                            let terms = group
                                .iter()
                                .enumerate()
                                .map(|(idx, is_null)| {
                                    if *is_null {
                                        format_expr_with_alias(&self.group_by.null_expr[idx])
                                    } else {
                                        format_expr_with_alias(&self.group_by.expr[idx])
                                    }
                                })
                                .collect::<Vec<String>>()
                                .join(", ");
                            format!("({terms})")
                        })
                        .collect()
                };
                let a: Vec<String> = self.aggr_expr.iter().map(|agg| agg.to_string()).collect();
                writeln!(f, "mode={:?}", self.mode)?;
                if !g.is_empty() {
                    writeln!(f, "group_by={}", g.join(", "))?;
                }
                if !a.is_empty() {
                    writeln!(f, "aggr={}", a.join(", "))?;
                }
                Ok(())
            }
        }
    }
}

impl fmt::Display for AggregateExec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        <Self as crate::display::DisplayAs>::fmt_as(
            self,
            crate::display::DisplayFormatType::Default,
            f,
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
    use ScalarValue::{Float32, Float64};
    match (a, b) {
        (Float32(x), Float32(y)) => x.to_bits() == y.to_bits(),
        (Float64(x), Float64(y)) => x.to_bits() == y.to_bits(),
        _ => a == b,
    }
}

/// Hash one scalar: the variant discriminant plus the value's bytes (floats by bit
/// pattern, so equal-keyed floats hash equally).
fn hash_scalar<H: Hasher>(v: &ScalarValue, state: &mut H) {
    use ScalarValue::{Null, Boolean, Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64, Float32, Float64, Utf8, Binary, Date32};
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
    //! integration test builds the physical plan by hand (the physical planner
    //! that normally assembles it lives in the `fdapquery` crate's
    //! `physical_planner` module).
    use super::*;
    use crate::Column;
    use crate::CountExpr;
    use crate::MaxExpr;
    use crate::MinExpr;
    use crate::SumExpr;
    use crate::test_util::employee_source;
    use fdapquery_datatypes::Field;
    use futures::TryStreamExt;

    /// Build a single-group `PhysicalGroupBy` matching the simple `GROUP BY a, b, …`
    /// shape — the only shape fdapquery's planner emits today. Each pair's alias
    /// is the expression's own `to_string()`, so the DataFusion-style `expr as alias`
    /// rendering collapses to just `expr` (e.g. `#3`).
    fn simple_group_by(exprs: Vec<Arc<dyn PhysicalExpr>>) -> PhysicalGroupBy {
        let pairs = exprs
            .into_iter()
            .map(|e| {
                let alias = e.to_string();
                (e, alias)
            })
            .collect();
        PhysicalGroupBy::new_single(pairs)
    }

    /// Sugar for the test-only `Single`-mode constructor: no FILTER expressions,
    /// `input_schema = input.schema()`.
    fn single_mode_aggregate(
        input: Arc<dyn ExecutionPlan>,
        group_exprs: Vec<Arc<dyn PhysicalExpr>>,
        aggr_exprs: Vec<Arc<dyn AggregateExpr>>,
        out_schema: Schema,
    ) -> AggregateExec {
        let n = aggr_exprs.len();
        let input_schema = input.schema();
        AggregateExec::try_new(
            AggregateMode::Single,
            simple_group_by(group_exprs),
            aggr_exprs,
            vec![None; n],
            input,
            input_schema,
            out_schema,
        )
        .unwrap()
    }

    // ---- Accumulators driven directly. ----

    #[test]
    fn min_accumulator() {
        let mut a = MinExpr::new(Arc::new(Column::new("a", 0))).create_accumulator();
        for v in [10, 14, 4] {
            a.accumulate(&ScalarValue::Int32(v)).unwrap();
        }
        assert_eq!(a.final_value().unwrap(), ScalarValue::Int32(4));
    }

    #[test]
    fn max_accumulator() {
        let mut a = MaxExpr::new(Arc::new(Column::new("a", 0))).create_accumulator();
        for v in [10, 14, 4] {
            a.accumulate(&ScalarValue::Int32(v)).unwrap();
        }
        assert_eq!(a.final_value().unwrap(), ScalarValue::Int32(14));
    }

    #[test]
    fn sum_accumulator() {
        let mut a = SumExpr::new(Arc::new(Column::new("a", 0))).create_accumulator();
        for v in [10, 14, 4] {
            a.accumulate(&ScalarValue::Int32(v)).unwrap();
        }
        assert_eq!(a.final_value().unwrap(), ScalarValue::Int32(28));
    }

    // ---- Integration: GROUP BY state, MIN/MAX/COUNT(salary) over employee.csv. ----

    #[tokio::test]
    async fn group_by_state_min_max_count() {
        // Output: state, MIN(salary), MAX(salary), COUNT(salary).
        let out_schema = Schema::new(vec![
            Field::new("state", arrow_schema::DataType::Utf8, true),
            Field::new("min_salary", arrow_schema::DataType::Int64, true),
            Field::new("max_salary", arrow_schema::DataType::Int64, true),
            Field::new("count_salary", arrow_schema::DataType::Int32, true),
        ]);
        // employee fixture columns: 0=id 1=first_name 2=last_name 3=state 4=job_title 5=salary
        let agg = single_mode_aggregate(
            employee_source(),
            vec![Arc::new(Column::new("state", 3))],
            vec![
                Arc::new(MinExpr::new(Arc::new(Column::new("salary", 5)))),
                Arc::new(MaxExpr::new(Arc::new(Column::new("salary", 5)))),
                Arc::new(CountExpr::new(Arc::new(Column::new("salary", 5)))),
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

        let states = batch.column(0).clone();
        let mins = batch.column(1).clone();
        let maxs = batch.column(2).clone();
        let counts = batch.column(3).clone();

        let mut got: HashMap<Option<String>, (i64, i64, i32)> = HashMap::new();
        for i in 0..batch.num_rows() {
            let state = match ScalarValue::try_from_array(&states, i).unwrap() {
                ScalarValue::Utf8(s) => Some(s),
                ScalarValue::Null => None,
                other => panic!("unexpected state value: {other:?}"),
            };
            let mn = match ScalarValue::try_from_array(&mins, i).unwrap() {
                ScalarValue::Int64(n) => n,
                o => panic!("min: {o:?}"),
            };
            let mx = match ScalarValue::try_from_array(&maxs, i).unwrap() {
                ScalarValue::Int64(n) => n,
                o => panic!("max: {o:?}"),
            };
            let c = match ScalarValue::try_from_array(&counts, i).unwrap() {
                ScalarValue::Int32(n) => n,
                o => panic!("count: {o:?}"),
            };
            got.insert(state, (mn, mx, c));
        }

        assert_eq!(got.get(&Some("CA".to_string())), Some(&(12000, 12000, 1)));
        assert_eq!(got.get(&Some("CO".to_string())), Some(&(10000, 11500, 2)));
        assert_eq!(got.get(&None), Some(&(11500, 11500, 1)));
    }

    /// Smoke check that accessor method names match DataFusion's: `mode()`,
    /// `group_expr()`, `aggr_expr()`, `filter_expr()`, `input()`,
    /// `input_schema()`. If any of these is renamed away from the DataFusion
    /// surface, this test stops compiling.
    #[test]
    fn accessor_method_names_match_datafusion() {
        let out_schema = Schema::new(vec![
            Field::new("state", arrow_schema::DataType::Utf8, true),
            Field::new("min_salary", arrow_schema::DataType::Int64, true),
        ]);
        let agg = single_mode_aggregate(
            employee_source(),
            vec![Arc::new(Column::new("state", 3))],
            vec![Arc::new(MinExpr::new(Arc::new(Column::new("salary", 5))))],
            out_schema,
        );
        // Each call below proves the method exists with the DataFusion name.
        assert_eq!(*agg.mode(), AggregateMode::Single);
        assert_eq!(agg.group_expr().expr().len(), 1);
        assert_eq!(agg.aggr_expr().len(), 1);
        assert_eq!(agg.filter_expr().len(), 1);
        assert!(agg.filter_expr().iter().all(|f| f.is_none()));
        assert_eq!(agg.input().name(), "TestSourceExec");
        assert_eq!(agg.input_schema().fields().len(), 6);
    }

    /// Byte-for-byte mirror of DataFusion's `DisplayFormatType::Default`
    /// output: `"AggregateExec: mode={Mode:?}, gby=[…], aggr=[…]"`.
    /// Source: `datafusion::physical_plan::aggregates::AggregateExec::fmt_as`,
    /// `datafusion/physical-plan/src/aggregates/mod.rs` lines 1551–1620.
    #[test]
    fn display_default_matches_datafusion() {
        // Case 1: single group, single aggregate, mode=Single.
        let out_schema = Schema::new(vec![
            Field::new("state", arrow_schema::DataType::Utf8, true),
            Field::new("min_salary", arrow_schema::DataType::Int64, true),
        ]);
        let agg = single_mode_aggregate(
            employee_source(),
            vec![Arc::new(Column::new("state", 3))],
            vec![Arc::new(MinExpr::new(Arc::new(Column::new("salary", 5))))],
            out_schema,
        );
        assert_eq!(
            format!("{agg}"),
            "AggregateExec: mode=Single, gby=[state@3], aggr=[MIN(salary@5)]"
        );

        // Case 2: multi-group, multi-aggregate, mode=Single.
        let out_schema = Schema::new(vec![
            Field::new("state", arrow_schema::DataType::Utf8, true),
            Field::new("job_title", arrow_schema::DataType::Utf8, true),
            Field::new("min_salary", arrow_schema::DataType::Int64, true),
            Field::new("max_salary", arrow_schema::DataType::Int64, true),
        ]);
        let agg = single_mode_aggregate(
            employee_source(),
            vec![
                Arc::new(Column::new("state", 3)),
                Arc::new(Column::new("job_title", 4)),
            ],
            vec![
                Arc::new(MinExpr::new(Arc::new(Column::new("salary", 5)))),
                Arc::new(MaxExpr::new(Arc::new(Column::new("salary", 5)))),
            ],
            out_schema,
        );
        assert_eq!(
            format!("{agg}"),
            "AggregateExec: mode=Single, gby=[state@3, job_title@4], aggr=[MIN(salary@5), MAX(salary@5)]"
        );

        // Case 3: no group, single aggregate, mode=Single.
        let out_schema = Schema::new(vec![Field::new(
            "max_salary",
            arrow_schema::DataType::Int64,
            true,
        )]);
        let agg = single_mode_aggregate(
            employee_source(),
            vec![],
            vec![Arc::new(MaxExpr::new(Arc::new(Column::new("salary", 5))))],
            out_schema,
        );
        assert_eq!(
            format!("{agg}"),
            "AggregateExec: mode=Single, gby=[], aggr=[MAX(salary@5)]"
        );
    }

    /// Exercise the full `displayable(plan).indent(false)` pipeline — the
    /// path EXPLAIN uses. Confirms that the operator's first line through
    /// the tree walker matches DataFusion's exact string, including the
    /// trailing newline emitted by the walker.
    #[test]
    fn displayable_indent_default_first_line() {
        let out_schema = Schema::new(vec![
            Field::new("state", arrow_schema::DataType::Utf8, true),
            Field::new("min_salary", arrow_schema::DataType::Int64, true),
        ]);
        let plan: Arc<dyn ExecutionPlan> = Arc::new(single_mode_aggregate(
            employee_source(),
            vec![Arc::new(Column::new("state", 3))],
            vec![Arc::new(MinExpr::new(Arc::new(Column::new("salary", 5))))],
            out_schema,
        ));
        let rendered = format!(
            "{}",
            crate::display::displayable(plan.as_ref()).indent(false)
        );
        let first_line = rendered.lines().next().unwrap();
        assert_eq!(
            first_line,
            "AggregateExec: mode=Single, gby=[state@3], aggr=[MIN(salary@5)]"
        );
    }
}
