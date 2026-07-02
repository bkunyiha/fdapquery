//! `DefaultPhysicalPlanner` converts a `LogicalPlan` into an
//! `Arc<dyn ExecutionPlan>`.
//!
//! This implementation lives in the umbrella crate, `fdapquery`, rather than
//! in `fdapquery-physical-plan` crate. That keeps `fdapquery-physical-plan` focused
//! on physical operators while letting the default planner compose catalog
//! types (`TableProvider`, `source_as_provider`) with those operators.
//!
//! This mirrors DataFusion's layout: the `PhysicalPlanner` trait lives in
//! `datafusion-physical-plan`, while `DefaultPhysicalPlanner` lives in the
//! umbrella `datafusion-core` crate.
//!
//! ## Notes
//! - **Exhaustive matches, no catch-all.** Both `LogicalPlan` and `Expr`
//!   are closed enums (§3.1), so the matches are exhaustive. Variants that
//!   genuinely have no physical counterpart use an explicit error arm with
//!   an "unsupported" message.
//! - **Aggregates.** Folded the standalone
//!   `AggregateExpr` enum into `Expr::AggregateFunction(AggregateFunction)`
//!   (mirror of DataFusion). The planner now dispatches on
//!   `AggregateFunctionKind` and the `distinct` flag inside the
//!   `AggregateFunctionParams`. The dispatch lives in a
//!   `create_aggregate_expr` helper purely for readability.
//! - **Variants with no physical mapping**: `Not`, `Modulus`, `ScalarFunction`,
//!   and an `AggregateExpr` used as a scalar expression (handled by the
//!   `Aggregate` operator, never planned standalone).
//!   collapsed every literal kind into `Expr::Literal(ScalarValue)`, which
//!   lowers to the physical `Literal { value: ScalarValue }` by cloning
//!   the inner scalar — the Arrow data type is then derived from the
//!   `ScalarValue` at evaluate time. Date literals already carry
//!   `ScalarValue::Date32(days_since_unix_epoch)`, so no extra conversion
//!   is needed at the planner seam.

use async_trait::async_trait;
use futures::future::BoxFuture;
use fdapquery_catalog::TableProvider;
use fdapquery_datatypes::{FdapQueryError, Result, Schema};
// The standalone logical-side `AggregateExpr` enum
// is gone. Aggregates are the `Expr::AggregateFunction(AggregateFunction)`
// variant of the broad `Expr` enum (mirror of DataFusion). The
// `LogicalAggregateExpr` alias hack that used to disambiguate the
// logical enum from the physical-side `AggregateExpr` trait is no
// longer needed — only `AggregateExpr` (the physical trait) appears
// in this file.
use fdapquery_expr::{AggregateFunctionKind, Expr, LogicalPlan, NullEquality};
// Items from the physical-plan crate. The trait `PhysicalPlanner` stays
// in `fdapquery-physical-plan`; only `DefaultPhysicalPlanner` (the
// concrete impl) lives here in the umbrella so it can reach catalog +
// physical-plan + datasource without forcing those deps on every
// physical-plan consumer.
// The 12 sibling binary types (`AddExpr`,
// `SubtractExpr`, `MultiplyExpr`, `DivideExpr`, `AndExpr`, `OrExpr`,
// `EqExpr`, `NeqExpr`, `LtExpr`, `LtEqExpr`, `GtExpr`, `GtEqExpr`)
// collapsed into a unified [`BinaryExpr`] parameterised by
// [`Operator`].
// `ScanExec` is gone. The
// `TableScan` arm now builds whatever `TableProvider::scan(projection)`
// returns — a `DataSourceExec` from `fdapquery-datasource` wrapping the
// per-format `*Config: DataSource`. Strict mirror of DataFusion.
use fdapquery_physical_plan::{
    AggregateExec, AggregateExpr, AggregateMode, AvgExpr, BinaryExpr, CastExpr, Column, CountExpr,
    DateAddIntervalExpr, DateSubtractIntervalExpr, ExecutionPlan, FilterExec, GlobalLimitExec,
    HashJoinExec, JoinOn, Literal, MaxExpr, MinExpr, PartitionMode, PhysicalExpr, PhysicalGroupBy,
    PhysicalPlanner, ProjectionExec, SumExpr,
};
use std::collections::HashSet;
use std::sync::Arc;

/// Default `PhysicalPlanner` impl — the all-of-fdapquery logical-to-physical
/// lowering. Renamed from `QueryPlanner` to match
/// DataFusion's `datafusion::physical_planner::DefaultPhysicalPlanner`.
#[derive(Default)]
pub struct DefaultPhysicalPlanner;

#[async_trait]
impl PhysicalPlanner for DefaultPhysicalPlanner {
    /// Trait-dispatch entry point. Delegates to the inherent
    /// `DefaultPhysicalPlanner::create_physical_plan` so consumers
    /// holding the concrete type don't pay a vtable indirection.
    ///
    /// Matches DataFusion's delegation pattern in
    /// `<DefaultPhysicalPlanner as datafusion::physical_planner::PhysicalPlanner>::create_physical_plan`.
    /// Rust resolves
    /// `self.create_physical_plan(plan)` to the inherent method
    /// because inherent methods have priority over trait methods in
    /// name lookup — no infinite recursion.
    async fn create_physical_plan(&self, plan: &LogicalPlan) -> Result<Arc<dyn ExecutionPlan>> {
        self.create_physical_plan(plan).await
    }
}

impl DefaultPhysicalPlanner {
    pub fn new() -> Self {
        DefaultPhysicalPlanner
    }

    /// Create a physical plan from a logical plan.
    ///
    /// Returns `Arc<dyn ExecutionPlan>` (not `Box`) — matches
    /// DataFusion's `ExecutionPlan` shape, lets the planner Arc-share
    /// subtrees, and lets `DistributedPlanner` rewrite plans via
    /// `with_new_children`.
    ///
    /// Async because the `LogicalPlan::TableScan` arm awaits
    /// `TableProvider::scan(projection)`. Recursive calls into the
    /// other arms therefore also `.await`. To keep the recursive
    /// `async fn` dyn-safe (avoiding the infinite-size `impl Future`
    /// problem on direct self-recursion in async fn), the recursive
    /// body is wrapped in a single `Box::pin(async move { ... })` —
    /// same shape DataFusion uses for `DefaultPhysicalPlanner::create_physical_plan`.
    pub fn create_physical_plan<'a>(
        &'a self,
        plan: &'a LogicalPlan,
    ) -> BoxFuture<'a, Result<Arc<dyn ExecutionPlan>>> {
        Box::pin(async move {
            Ok(match plan {
                LogicalPlan::TableScan(s) => {
                    // Logical plans keep only a `TableSource`: a small,
                    // logical-side view of a table with enough information for
                    // planning and optimization, such as its schema.
                    //
                    // Physical scans are created by `TableProvider::scan`,
                    // which lives on the catalog/provider side. At
                    // registration time, a `TableProvider` is wrapped so the
                    // logical plan can store it as a `TableSource`; 
                    // see fdapquery-catalog::default_table_source::provider_as_source
                    //   pub fn provider_as_source(table_provider: Arc<dyn TableProvider>) -> Arc<dyn TableSource> {
                    //      Arc::new(DefaultTableSource::new(table_provider))
                    //   }
                    // Here we recover that original `TableProvider` and ask it to build
                    // recover that original `TableProvider` and ask it to build
                    // the executable scan node.
                    //
                    // The logical `TableScan` stores projected columns by name.
                    // `TableProvider::scan` expects column indexes, so convert
                    // names to positions in the provider schema before calling
                    // the async scan method.
                    let provider: Arc<dyn TableProvider> = fdapquery_catalog::source_as_provider(&s.data_source)?;
                    let projection_indices: Option<Vec<usize>> = if s.projection.is_empty() {
                        None
                    } else {
                        let source_schema = provider.schema();
                        let indices: Vec<usize> = s
                            .projection
                            .iter()
                            .map(|name| {
                                source_schema
                                    .fields()
                                    .iter()
                                    .position(|f| f.name() == name)
                                    .ok_or_else(|| {
                                        FdapQueryError::SchemaError(format!(
                                            "DefaultPhysicalPlanner: projection column '{name}' \
                                         not in source schema"
                                        ))
                                    })
                            })
                            .collect::<Result<Vec<usize>>>()?;
                        Some(indices)
                    };
                    provider.scan(projection_indices.as_ref()).await?
                }
                LogicalPlan::Filter(s) => {
                    let input = self.create_physical_plan(&s.input).await?;
                    let filter_expr = self.create_physical_expr(&s.expr, &s.input)?;
                    Arc::new(FilterExec::new(filter_expr, input))
                }
                LogicalPlan::Projection(p) => {
                    let input = self.create_physical_plan(&p.input).await?;
                    let projection_expr: Vec<Arc<dyn PhysicalExpr>> = p
                        .expr
                        .iter()
                        .map(|e| self.create_physical_expr(e, &p.input))
                        .collect::<Result<Vec<_>>>()?;
                    let projection_schema = Schema::new(
                        p.expr
                            .iter()
                            .map(|e| e.to_field(&p.input))
                            .collect::<Result<Vec<_>>>()?,
                    );
                    Arc::new(ProjectionExec::new(
                        projection_expr,
                        input,
                        projection_schema,
                    ))
                }
                LogicalPlan::Aggregate(a) => {
                    let input = self.create_physical_plan(&a.input).await?;
                    let input_schema = input.schema();
                    let group_expr: Vec<Arc<dyn PhysicalExpr>> = a
                        .group_expr
                        .iter()
                        .map(|e| self.create_physical_expr(e, &a.input))
                        .collect::<Result<Vec<_>>>()?;
                    let aggregate_expr: Vec<Arc<dyn AggregateExpr>> = a
                        .aggregate_expr
                        .iter()
                        .map(|agg| self.create_aggregate_expr(agg, &a.input))
                        .collect::<Result<Vec<_>>>()?;
                    // Simple `GROUP BY a, b, …` shape: each pair's alias is the
                    // expression's own `to_string()`, matching DataFusion's
                    // `PhysicalGroupBy::new_single` contract when the planner
                    // hasn't assigned an explicit alias.
                    let group_pairs: Vec<(Arc<dyn PhysicalExpr>, String)> = group_expr
                        .into_iter()
                        .map(|e| {
                            let alias = e.to_string();
                            (e, alias)
                        })
                        .collect();
                    let group_by = PhysicalGroupBy::new_single(group_pairs);
                    let n_aggrs = aggregate_expr.len();
                    Arc::new(AggregateExec::try_new(
                        AggregateMode::Single,
                        group_by,
                        aggregate_expr,
                        vec![None; n_aggrs],
                        input,
                        input_schema,
                        plan.schema()?,
                    )?)
                }
                LogicalPlan::Limit(l) => {
                    // fdapquery's logical `Limit` currently carries a single
                    // `limit` value (no separate skip). Mirror DataFusion's
                    // wire-up by passing `skip=0, fetch=Some(limit)` — the
                    // physical operator now has the full skip/fetch surface even
                    // though the logical layer doesn't expose it yet.
                    let input = self.create_physical_plan(&l.input).await?;
                    Arc::new(GlobalLimitExec::new(input, 0, Some(l.limit as usize)))
                }
                LogicalPlan::Join(j) => {
                    let left_plan = self.create_physical_plan(&j.left).await?;
                    let right_plan = self.create_physical_plan(&j.right).await?;
                    let left_schema = j.left.schema()?;
                    let right_schema = j.right.schema()?;

                    // Resolve join-key column names to indices in each input schema.
                    let left_keys: Vec<usize> =
                        j.on.iter()
                            .map(|(left_col, _)| {
                                left_schema
                                    .fields()
                                    .iter()
                                    .position(|f| f.name() == left_col)
                                    .ok_or_else(|| {
                                        FdapQueryError::SchemaError(format!(
                                            "no column named '{left_col}' in left input"
                                        ))
                                    })
                            })
                            .collect::<Result<Vec<_>>>()?;
                    let right_keys: Vec<usize> =
                        j.on.iter()
                            .map(|(_, right_col)| {
                                right_schema
                                    .fields()
                                    .iter()
                                    .position(|f| f.name() == right_col)
                                    .ok_or_else(|| {
                                        FdapQueryError::SchemaError(format!(
                                            "no column named '{right_col}' in right input"
                                        ))
                                    })
                            })
                            .collect::<Result<Vec<_>>>()?;

                    // Right columns to exclude: duplicate join keys with the same name
                    // on both sides (so the joined row doesn't carry the key twice).
                    let duplicate_key_names: HashSet<String> =
                        j.on.iter()
                            .filter(|(l, r)| l == r)
                            .map(|(_, r)| r.clone())
                            .collect();
                    let right_columns_to_exclude: HashSet<usize> = right_schema
                        .fields()
                        .iter()
                        .enumerate()
                        .filter_map(|(i, f)| {
                            if duplicate_key_names.contains(f.name()) {
                                Some(i)
                            } else {
                                None
                            }
                        })
                        .collect();

                    // Lift the column-index pairs into the DataFusion-shape
                    // `JoinOn` (pairs of `Arc<dyn PhysicalExpr>` columns). The
                    // execution body in `HashJoinExec::execute` lowers this back
                    // to index pairs — the round trip exists so the public
                    // `on` field surfaces the same type as DataFusion's
                    // `joins::JoinOn`.
                    let on: JoinOn = left_keys
                        .iter()
                        .copied()
                        .zip(right_keys.iter().copied())
                        .map(|(l, r)| {
                            let l_name = left_schema.fields()[l].name();
                            let r_name = right_schema.fields()[r].name();
                            (
                                Arc::new(Column::new(l_name, l)) as Arc<dyn PhysicalExpr>,
                                Arc::new(Column::new(r_name, r)) as Arc<dyn PhysicalExpr>,
                            )
                        })
                        .collect();
                    let join_schema = plan.schema()?;
                    Arc::new(
                        HashJoinExec::try_new(
                            left_plan,
                            right_plan,
                            on,
                            // No filter / projection / null_aware yet; the planner
                            // doesn't emit them. `null_equality` defaults to the
                            // SQL-standard `NullEqualsNothing`. `partition_mode`
                            // defaults to `Partitioned` — fdapquery has no
                            // statistics surface to pick CollectLeft.
                            None,
                            &j.join_type,
                            None,
                            PartitionMode::Partitioned,
                            NullEquality::NullEqualsNothing,
                            false,
                        )?
                        .with_join_schema(join_schema)
                        .with_right_columns_to_exclude(right_columns_to_exclude),
                    )
                }
            })
        })
    }

    /// Build the physical aggregate expression for one logical
    /// `Expr::AggregateFunction(...)`. The signature
    /// takes `&Expr` and pattern-matches the `AggregateFunction`
    /// variant; non-aggregate inputs surface as an internal error
    /// (the `Aggregate` plan invariant guarantees every element of
    /// `aggregate_expr` is an `Expr::AggregateFunction(...)`).
    fn create_aggregate_expr(
        &self,
        agg: &Expr,
        input: &LogicalPlan,
    ) -> Result<Arc<dyn AggregateExpr>> {
        let af = match agg {
            Expr::AggregateFunction(af) => af,
            other => {
                return Err(FdapQueryError::Internal(format!(
                    "create_aggregate_expr: expected Expr::AggregateFunction, found {other:?}"
                )));
            }
        };
        // fdapquery's built-in aggregates are all single-argument. The
        // SQL planner enforces this; DataFusion's UDAF surface allows
        // multi-argument aggregates but neither side is wired yet.
        let arg = af.params.args.first().ok_or_else(|| {
            FdapQueryError::Internal(
                "create_aggregate_expr: AggregateFunction has no argument expressions".into(),
            )
        })?;
        let phys_arg = self.create_physical_expr(arg, input)?;
        // DISTINCT has no physical operator wired today. Once a physical
        // distinct-count operator lands, this guard becomes the
        // dispatch site for the DISTINCT path.
        if af.params.distinct {
            return Err(FdapQueryError::NotImplemented(
                "COUNT(DISTINCT ...) is not supported".into(),
            ));
        }
        Ok(match af.func {
            AggregateFunctionKind::Max => Arc::new(MaxExpr::new(phys_arg)),
            AggregateFunctionKind::Min => Arc::new(MinExpr::new(phys_arg)),
            AggregateFunctionKind::Sum => Arc::new(SumExpr::new(phys_arg)),
            AggregateFunctionKind::Avg => Arc::new(AvgExpr::new(phys_arg)),
            AggregateFunctionKind::Count => Arc::new(CountExpr::new(phys_arg)),
        })
    }

    /// Create a physical expression from a logical expression.
    #[allow(clippy::self_only_used_in_recursion)] // public trait method; cannot be an associated function
    pub fn create_physical_expr(
        &self,
        expr: &Expr,
        input: &LogicalPlan,
    ) -> Result<Arc<dyn PhysicalExpr>> {
        Ok(match expr {
            // The logical side also collapsed: a single
            // `Expr::Literal(ScalarValue)` now mirrors DataFusion's
            // `Expr::Literal(ScalarValue)` byte-for-byte, and lowers to the
            // physical `Literal { value: ScalarValue }` (#136) by simply
            // cloning the inner scalar. The Arrow data type is derived
            // from the `ScalarValue` at `evaluate` time.
            Expr::Literal(scalar) => Arc::new(Literal::new(scalar.clone())),
            Expr::DateSubtractInterval { date, interval } => {
                Arc::new(DateSubtractIntervalExpr::new(
                    self.create_physical_expr(date, input)?,
                    self.create_physical_expr(interval, input)?,
                ))
            }
            Expr::DateAddInterval { date, interval } => Arc::new(DateAddIntervalExpr::new(
                self.create_physical_expr(date, input)?,
                self.create_physical_expr(interval, input)?,
            )),
            Expr::ColumnIndex(i) => {
                // Look up the column name from the input schema so the
                // resulting `Column` carries `{name}@{index}` for display.
                let schema = input.schema()?;
                let name = schema.fields()[*i].name();
                Arc::new(Column::new(name, *i))
            }
            Expr::Column(name) => {
                let i = input
                    .schema()?
                    .fields()
                    .iter()
                    .position(|f| f.name() == name)
                    .ok_or_else(|| {
                        FdapQueryError::SchemaError(format!("no column named '{name}'"))
                    })?;
                Arc::new(Column::new(name, i))
            }
            // An alias has no physical expression — it only renamed the column
            // during planning. Plan the inner expression directly.
            Expr::Alias { expr, .. } => self.create_physical_expr(expr, input)?,
            Expr::Cast { expr, data_type } => Arc::new(CastExpr::new(
                self.create_physical_expr(expr, input)?,
                data_type.clone(),
            )),
            // Single binary arm. Both sides lower to
            // physical expressions and we hand the operator straight
            // through to the unified [`BinaryExpr`], which supports
            // `Operator::Modulo` directly.
            Expr::BinaryExpr { left, op, right } => Arc::new(BinaryExpr::new(
                self.create_physical_expr(left, input)?,
                *op,
                self.create_physical_expr(right, input)?,
            )),

            // --- Variants with no physical counterpart. ---
            Expr::Not(_) => {
                return Err(FdapQueryError::NotImplemented(
                    "NOT is not supported".into(),
                ));
            }
            Expr::ScalarFunction { name, .. } => {
                return Err(FdapQueryError::NotImplemented(format!(
                    "scalar function '{name}' is not supported"
                )));
            }
            Expr::AggregateFunction(_) => {
                return Err(FdapQueryError::Internal(
                    "an aggregate cannot be planned as a scalar expression; \
                     aggregates are lowered by the Aggregate operator"
                        .into(),
                ));
            }
        })
    }
}

#[cfg(test)]
mod tests {
    //! Tests for `plan aggregate query`. Asserts the plan *structure*: the
    //! root is a `AggregateExec` over a single `DataSourceExec` leaf, with
    //! the expected resolved column indices in its `Display`.
    use super::*;
    use fdapquery_catalog::{InMemoryDataSource, provider_as_source};
    use fdapquery_datatypes::{Field, Schema};
    use fdapquery_expr::{DataFrame, LogicalPlan, TableScan, col, max};
    use fdapquery_optimizer::Optimizer;
    use std::sync::Arc;

    #[tokio::test]
    async fn plan_aggregate_query() {
        let schema = Schema::new(vec![
            Field::new("passenger_count", arrow::datatypes::DataType::UInt32, true),
            Field::new("max_fare", arrow::datatypes::DataType::Float64, true),
        ]);
        let data_source = provider_as_source(Arc::new(InMemoryDataSource::new(schema, vec![])));
        let df = DataFrame::new(LogicalPlan::TableScan(
            TableScan::new("", data_source, vec![]).unwrap(),
        ));

        // SELECT passenger_count, MAX(max_fare) ... GROUP BY passenger_count
        let plan = df
            .aggregate(vec![col("passenger_count")], vec![max(col("max_fare"))])
            .logical_plan()
            .clone();

        // Optimize (ProjectionPushDown trims the scan to [max_fare, passenger_count]).
        let optimized = Optimizer::new().optimize(&plan).unwrap();

        let planner = DefaultPhysicalPlanner::new();
        let physical = planner.create_physical_plan(&optimized).await.unwrap();

        // Root is an AggregateExec; the optimizer's sorted pushdown puts
        // max_fare at index 0 and passenger_count at index 1, so the
        // group key is `passenger_count@1` and the MAX argument is
        // `max_fare@0`. Tree-dump goes through the
        // `displayable(plan).indent(verbose)` builder — same idiom as
        // DataFusion. `physical.as_ref()` coerces
        // `Arc<dyn ExecutionPlan>` to `&dyn ExecutionPlan` for the
        // free function.
        // The indent walker mirrors DataFusion's
        // byte-for-byte: two ASCII spaces per nesting level, one line
        // per node, terminated by `\n`. The root sits at depth 0 (zero
        // leading spaces) and its sole child (a `DataSourceExec` leaf) is
        // at depth 1 (two leading spaces).
        // Column Display mirrors DataFusion's `{name}@{index}` form.
        let pretty = format!(
            "{}",
            fdapquery_physical_plan::displayable(physical.as_ref()).indent(false)
        );
        let mut lines = pretty.lines();
        let root_line = lines.next().expect("at least one line");
        assert_eq!(
            root_line,
            "AggregateExec: mode=Single, gby=[passenger_count@1], aggr=[MAX(max_fare@0)]",
            "unexpected root line"
        );
        let child_line = lines.next().expect("child line");
        // Two-space prefix proves the indent contract. The leaf is
        // `DataSourceExec`.
        assert!(
            child_line.starts_with("  DataSourceExec:"),
            "expected two-space-indented DataSourceExec child, got: {child_line:?}"
        );
        assert!(
            child_line.contains("max_fare") && child_line.contains("passenger_count"),
            "scan child must show pushed-down projection: {child_line}"
        );
        assert_eq!(lines.next(), None, "tree must have exactly two nodes");
        // Trailing newline — every line in the indent walker ends with `\n`.
        assert!(
            pretty.ends_with('\n'),
            "indent output must end with a newline"
        );

        // One child, a DataSourceExec leaf, with the pushed-down projection.
        let children = physical.children();
        assert_eq!(children.len(), 1);
        assert!(children[0].children().is_empty());

        // Output schema is the aggregate's: [passenger_count, MAX(max_fare)].
        assert_eq!(physical.schema().fields().len(), 2);
    }
}
