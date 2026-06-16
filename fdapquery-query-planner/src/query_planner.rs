//! Translates a `LogicalPlan` (what to compute) into a `PhysicalPlan` (how to
//! run it) — the seam between the planner and the executor. Two `match`
//! functions: `create_physical_plan` over the six `LogicalPlan` operators and
//! `create_physical_expr` over the `LogicalExpr` variants. The most boring
//! module in the workspace, by design.
//!
//! ## Notes
//! - **Exhaustive matches, no catch-all.** Both `LogicalPlan` and `LogicalExpr`
//!   are closed enums (§3.1), so the matches are exhaustive. Variants that
//!   genuinely have no physical counterpart use an explicit `panic!` arm with
//!   an "unsupported" message.
//! - **Aggregates.** `AggregateExpr` is a closed enum, so the dispatch is
//!   exhaustive with no catch-all. `CountDistinct` keeps an explicit `panic!`
//!   arm because it has no physical operator. The dispatch lives in a
//!   `create_aggregate_expr` helper purely for readability.
//! - **Variants with no physical mapping**: `Not`, `Modulus`, `ScalarFunction`,
//!   `LiteralFloat` (no physical float-literal expression), and an
//!   `AggregateExpr` used as a scalar expression (handled by the `Aggregate`
//!   operator, never planned standalone). `LiteralDate` IS lowered here —
//!   `NaiveDate::num_days_from_ce()` converted to days-since-Unix-epoch.

use datatypes::Schema;
use logical_plan::{AggregateExpr, LogicalExpr, LogicalPlan};
use physical_plan::{
    AddExpression, AggregateExpression, AndExpression, AvgExpression, CastExpression,
    ColumnExpression, CountExpression, DateAddIntervalExpression, DateSubtractIntervalExpression,
    DivideExpression, EqExpression, Expression, GtEqExpression, GtExpression, HashAggregateExec,
    HashJoinExec, LimitExec, LiteralDateExpression, LiteralDoubleExpression,
    LiteralIntervalDaysExpression, LiteralLongExpression, LiteralStringExpression, LtEqExpression,
    LtExpression, MaxExpression, MinExpression, MultiplyExpression, NeqExpression, OrExpression,
    PhysicalPlan, ProjectionExec, ScanExec, SelectionExec, SubtractExpression, SumExpression,
};
use std::collections::HashSet;
use std::sync::Arc;

/// Convert a `chrono::NaiveDate` to "days since the Unix epoch" (1970-01-01).
/// Used to lower a logical `LiteralDate(NaiveDate)` into a physical
/// `LiteralDateExpression { days_since_epoch: i32 }`.
fn days_since_unix_epoch(date: chrono::NaiveDate) -> i32 {
    let epoch = chrono::NaiveDate::from_ymd_opt(1970, 1, 1).expect("1970-01-01 is a valid date");
    (date - epoch).num_days() as i32
}

/// Creates a physical query plan from a logical query plan.
#[derive(Default)]
pub struct QueryPlanner;

impl QueryPlanner {
    pub fn new() -> Self {
        QueryPlanner
    }

    /// Create a physical plan from a logical plan.
    ///
    /// Returns `Arc<dyn PhysicalPlan>` (not `Box`) — matches DataFusion's
    /// `ExecutionPlan` shape, lets the planner Arc-share subtrees, and lets
    /// `DistributedPlanner` rewrite plans via `with_new_children`.
    pub fn create_physical_plan(&self, plan: &LogicalPlan) -> Arc<dyn PhysicalPlan> {
        match plan {
            LogicalPlan::Scan(s) => Arc::new(ScanExec::new(
                Arc::clone(&s.data_source),
                s.projection.clone(),
            )),
            LogicalPlan::Selection(s) => {
                let input = self.create_physical_plan(&s.input);
                let filter_expr = self.create_physical_expr(&s.expr, &s.input);
                Arc::new(SelectionExec::new(input, filter_expr))
            }
            LogicalPlan::Projection(p) => {
                let input = self.create_physical_plan(&p.input);
                let projection_expr: Vec<Arc<dyn Expression>> = p
                    .expr
                    .iter()
                    .map(|e| self.create_physical_expr(e, &p.input))
                    .collect();
                let projection_schema =
                    Schema::new(p.expr.iter().map(|e| e.to_field(&p.input)).collect());
                Arc::new(ProjectionExec::new(
                    input,
                    projection_schema,
                    projection_expr,
                ))
            }
            LogicalPlan::Aggregate(a) => {
                let input = self.create_physical_plan(&a.input);
                let group_expr: Vec<Arc<dyn Expression>> = a
                    .group_expr
                    .iter()
                    .map(|e| self.create_physical_expr(e, &a.input))
                    .collect();
                let aggregate_expr: Vec<Arc<dyn AggregateExpression>> = a
                    .aggregate_expr
                    .iter()
                    .map(|agg| self.create_aggregate_expr(agg, &a.input))
                    .collect();
                Arc::new(HashAggregateExec::new(
                    input,
                    group_expr,
                    aggregate_expr,
                    plan.schema(),
                ))
            }
            LogicalPlan::Limit(l) => {
                let input = self.create_physical_plan(&l.input);
                Arc::new(LimitExec::new(input, l.limit as usize))
            }
            LogicalPlan::Join(j) => {
                let left_plan = self.create_physical_plan(&j.left);
                let right_plan = self.create_physical_plan(&j.right);
                let left_schema = j.left.schema();
                let right_schema = j.right.schema();

                // Resolve join-key column names to indices in each input schema.
                let left_keys: Vec<usize> = j
                    .on
                    .iter()
                    .map(|(left_col, _)| {
                        left_schema
                            .fields
                            .iter()
                            .position(|f| &f.name == left_col)
                            .unwrap_or_else(|| panic!("No column named '{left_col}' in left input"))
                    })
                    .collect();
                let right_keys: Vec<usize> =
                    j.on.iter()
                        .map(|(_, right_col)| {
                            right_schema
                                .fields
                                .iter()
                                .position(|f| &f.name == right_col)
                                .unwrap_or_else(|| {
                                    panic!("No column named '{right_col}' in right input")
                                })
                        })
                        .collect();

                // Right columns to exclude: duplicate join keys with the same name
                // on both sides (so the joined row doesn't carry the key twice).
                let duplicate_key_names: HashSet<String> =
                    j.on.iter()
                        .filter(|(l, r)| l == r)
                        .map(|(_, r)| r.clone())
                        .collect();
                let right_columns_to_exclude: HashSet<usize> = right_schema
                    .fields
                    .iter()
                    .enumerate()
                    .filter_map(|(i, f)| {
                        if duplicate_key_names.contains(&f.name) {
                            Some(i)
                        } else {
                            None
                        }
                    })
                    .collect();

                Arc::new(HashJoinExec::new(
                    left_plan,
                    right_plan,
                    j.join_type.clone(),
                    left_keys,
                    right_keys,
                    plan.schema(),
                    right_columns_to_exclude,
                ))
            }
        }
    }

    /// Build the physical aggregate expression for one logical `AggregateExpr`.
    /// `Aggregate` branch.
    fn create_aggregate_expr(
        &self,
        agg: &AggregateExpr,
        input: &LogicalPlan,
    ) -> Arc<dyn AggregateExpression> {
        match agg {
            AggregateExpr::Max(e) => {
                Arc::new(MaxExpression::new(self.create_physical_expr(e, input)))
            }
            AggregateExpr::Min(e) => {
                Arc::new(MinExpression::new(self.create_physical_expr(e, input)))
            }
            AggregateExpr::Sum(e) => {
                Arc::new(SumExpression::new(self.create_physical_expr(e, input)))
            }
            AggregateExpr::Avg(e) => {
                Arc::new(AvgExpression::new(self.create_physical_expr(e, input)))
            }
            AggregateExpr::Count(e) => {
                Arc::new(CountExpression::new(self.create_physical_expr(e, input)))
            }
            AggregateExpr::CountDistinct(_) => {
                panic!("Unsupported aggregate function: COUNT(DISTINCT ...)")
            }
        }
    }

    /// Create a physical expression from a logical expression.
    pub fn create_physical_expr(
        &self,
        expr: &LogicalExpr,
        input: &LogicalPlan,
    ) -> Arc<dyn Expression> {
        match expr {
            LogicalExpr::LiteralLong(n) => Arc::new(LiteralLongExpression::new(*n)),
            LogicalExpr::LiteralDouble(n) => Arc::new(LiteralDoubleExpression::new(*n)),
            LogicalExpr::LiteralString(s) => Arc::new(LiteralStringExpression::new(s.clone())),
            // Lower a `LiteralDate` to a date literal expressed as days since
            // the Unix epoch.
            LogicalExpr::LiteralDate(date) => {
                Arc::new(LiteralDateExpression::new(days_since_unix_epoch(*date)))
            }
            LogicalExpr::LiteralIntervalDays(days) => {
                Arc::new(LiteralIntervalDaysExpression::new(*days))
            }
            LogicalExpr::DateSubtractInterval { date, interval } => {
                Arc::new(DateSubtractIntervalExpression::new(
                    self.create_physical_expr(date, input),
                    self.create_physical_expr(interval, input),
                ))
            }
            LogicalExpr::DateAddInterval { date, interval } => {
                Arc::new(DateAddIntervalExpression::new(
                    self.create_physical_expr(date, input),
                    self.create_physical_expr(interval, input),
                ))
            }
            LogicalExpr::ColumnIndex(i) => Arc::new(ColumnExpression::new(*i)),
            LogicalExpr::Column(name) => {
                let i = input
                    .schema()
                    .fields
                    .iter()
                    .position(|f| &f.name == name)
                    .unwrap_or_else(|| panic!("No column named '{name}'"));
                Arc::new(ColumnExpression::new(i))
            }
            // An alias has no physical expression — it only renamed the column
            // during planning. Plan the inner expression directly.
            LogicalExpr::Alias { expr, .. } => self.create_physical_expr(expr, input),
            LogicalExpr::Cast { expr, data_type } => Arc::new(CastExpression::new(
                self.create_physical_expr(expr, input),
                data_type.clone(),
            )),
            // Binary expressions: plan both sides, then pick the operator.
            LogicalExpr::Eq { l, r } => Arc::new(EqExpression::new(
                self.create_physical_expr(l, input),
                self.create_physical_expr(r, input),
            )),
            LogicalExpr::Neq { l, r } => Arc::new(NeqExpression::new(
                self.create_physical_expr(l, input),
                self.create_physical_expr(r, input),
            )),
            LogicalExpr::Gt { l, r } => Arc::new(GtExpression::new(
                self.create_physical_expr(l, input),
                self.create_physical_expr(r, input),
            )),
            LogicalExpr::GtEq { l, r } => Arc::new(GtEqExpression::new(
                self.create_physical_expr(l, input),
                self.create_physical_expr(r, input),
            )),
            LogicalExpr::Lt { l, r } => Arc::new(LtExpression::new(
                self.create_physical_expr(l, input),
                self.create_physical_expr(r, input),
            )),
            LogicalExpr::LtEq { l, r } => Arc::new(LtEqExpression::new(
                self.create_physical_expr(l, input),
                self.create_physical_expr(r, input),
            )),
            LogicalExpr::And { l, r } => Arc::new(AndExpression::new(
                self.create_physical_expr(l, input),
                self.create_physical_expr(r, input),
            )),
            LogicalExpr::Or { l, r } => Arc::new(OrExpression::new(
                self.create_physical_expr(l, input),
                self.create_physical_expr(r, input),
            )),
            LogicalExpr::Add { l, r } => Arc::new(AddExpression::new(
                self.create_physical_expr(l, input),
                self.create_physical_expr(r, input),
            )),
            LogicalExpr::Subtract { l, r } => Arc::new(SubtractExpression::new(
                self.create_physical_expr(l, input),
                self.create_physical_expr(r, input),
            )),
            LogicalExpr::Multiply { l, r } => Arc::new(MultiplyExpression::new(
                self.create_physical_expr(l, input),
                self.create_physical_expr(r, input),
            )),
            LogicalExpr::Divide { l, r } => Arc::new(DivideExpression::new(
                self.create_physical_expr(l, input),
                self.create_physical_expr(r, input),
            )),

            // --- Variants with no physical counterpart. ---
            LogicalExpr::LiteralFloat(_) => {
                panic!("LiteralFloat has no physical expression; use LiteralDouble")
            }
            LogicalExpr::Not(_) => panic!("Unsupported logical expression: NOT"),
            LogicalExpr::Modulus { .. } => panic!("Unsupported binary expression: modulus"),
            LogicalExpr::ScalarFunction { name, .. } => {
                panic!("Unsupported logical expression: scalar function '{name}'")
            }
            LogicalExpr::AggregateExpr(_) => panic!(
                "an aggregate cannot be planned as a scalar expression; \
                 aggregates are lowered by the Aggregate operator"
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    //! Tests for `plan aggregate query`. Asserts the plan *structure*: the
    //! root is a `HashAggregateExec` over a single `ScanExec` leaf, with the
    //! expected resolved column indices in its `Display`.
    use super::*;
    use datasource::InMemoryDataSource;
    use datatypes::arrow_types::{DOUBLE_TYPE, UINT32_TYPE};
    use datatypes::{Field, Schema};
    use logical_plan::{DataFrame, LogicalPlan, Scan, col, max};
    use optimizer::Optimizer;
    use std::sync::Arc;

    #[test]
    fn plan_aggregate_query() {
        let schema = Schema::new(vec![
            Field::new("passenger_count", UINT32_TYPE),
            Field::new("max_fare", DOUBLE_TYPE),
        ]);
        let data_source = Arc::new(InMemoryDataSource::new(schema, vec![]));
        let df = DataFrame::new(LogicalPlan::Scan(Scan::new("", data_source, vec![])));

        // SELECT passenger_count, MAX(max_fare) ... GROUP BY passenger_count
        let plan = df
            .aggregate(vec![col("passenger_count")], vec![max(col("max_fare"))])
            .logical_plan()
            .clone();

        // Optimize (ProjectionPushDown trims the scan to [max_fare, passenger_count]).
        let optimized = Optimizer::new().optimize(&plan);

        let planner = QueryPlanner::new();
        let physical = planner.create_physical_plan(&optimized);

        // Root is a HashAggregateExec; the optimizer's sorted pushdown puts
        // max_fare at index 0 and passenger_count at index 1, so the group key is
        // #1 and the MAX argument is #0. (`format` is the free fn — `pretty()` is
        // gated `where Self: Sized` and isn't callable on `Arc<dyn PhysicalPlan>`.)
        let pretty = physical_plan::format(physical.as_ref());
        assert!(
            pretty.starts_with(
                "HashAggregateExec: groupExpr=[#1], aggrExpr=[MAX(#0)], mode=Complete"
            ),
            "unexpected root line: {pretty}"
        );

        // One child, a ScanExec leaf, with the pushed-down projection.
        let children = physical.children();
        assert_eq!(children.len(), 1);
        assert!(children[0].children().is_empty());
        let scan_line = children[0].to_string();
        assert!(
            scan_line.starts_with("ScanExec:"),
            "expected ScanExec: {scan_line}"
        );
        assert!(scan_line.contains("max_fare") && scan_line.contains("passenger_count"));

        // Output schema is the aggregate's: [passenger_count, MAX(max_fare)].
        assert_eq!(physical.schema().fields.len(), 2);
    }
}
