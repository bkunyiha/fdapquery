//! Pushes the set of referenced columns down to the `TableScan`, so the data source
//! reads only the columns the rest of the query actually needs.
//!
//! `LogicalPlan` is a closed `enum`, so the `match` is exhaustive over all
//! six operators and needs no catch-all (per §3.1).

use fdapquery_datatypes::Result;
use fdapquery_expr::{Aggregate, Filter, Join, Limit, LogicalPlan, Projection, TableScan};
use std::collections::HashSet;
use std::sync::Arc;

use crate::optimizer::{
    OptimizerConfig, OptimizerRule, aggregate_inner, extract_columns, extract_columns_list,
};

/// The one optimisation rule so far.
pub struct ProjectionPushDownRule;

impl OptimizerRule for ProjectionPushDownRule {
    fn name(&self) -> &str {
        "projection_push_down"
    }

    /// Session 15d-1 #100 — migrated to the `try_optimize` shape.
    /// The rule always fires (even if it leaves the plan
    /// structurally unchanged, e.g. a single `TableScan`), so it
    /// returns `Ok(Some(_))` whenever `push_down` succeeds.
    fn try_optimize(
        &self,
        plan: &LogicalPlan,
        _config: &OptimizerConfig,
    ) -> Result<Option<LogicalPlan>> {
        push_down(plan, &mut HashSet::new()).map(Some)
    }
}

/// Rewrite `plan`, accumulating referenced column names on the way down and
/// trimming the `TableScan`'s projection at the leaf.
fn push_down(plan: &LogicalPlan, column_names: &mut HashSet<String>) -> Result<LogicalPlan> {
    let rewritten = match plan {
        LogicalPlan::Projection(p) => {
            extract_columns_list(&p.expr, &p.input, column_names)?;
            let input = push_down(&p.input, column_names)?;
            LogicalPlan::Projection(Projection::new(input, p.expr.clone()))
        }
        LogicalPlan::Filter(s) => {
            extract_columns(&s.expr, &s.input, column_names)?;
            let input = push_down(&s.input, column_names)?;
            LogicalPlan::Filter(Filter::new(input, s.expr.clone()))
        }
        LogicalPlan::Aggregate(a) => {
            extract_columns_list(&a.group_expr, &a.input, column_names)?;
            // Collect the columns referenced by each aggregate's *argument*
            // expression.
            for agg in &a.aggregate_expr {
                extract_columns(aggregate_inner(agg), &a.input, column_names)?;
            }
            let input = push_down(&a.input, column_names)?;
            LogicalPlan::Aggregate(Aggregate::new(
                input,
                a.group_expr.clone(),
                a.aggregate_expr.clone(),
            ))
        }
        LogicalPlan::Limit(l) => {
            let input = push_down(&l.input, column_names)?;
            LogicalPlan::Limit(Limit::new(input, l.limit))
        }
        LogicalPlan::Join(j) => {
            // If nothing has been requested yet (the join is at the root),
            // request every column from both sides.
            if column_names.is_empty() {
                let left_schema = j.left.schema()?;
                for f in left_schema.fields().iter() {
                    column_names.insert(f.name().clone());
                }
                let right_schema = j.right.schema()?;
                for f in right_schema.fields().iter() {
                    column_names.insert(f.name().clone());
                }
            }
            // The join keys are always required.
            for (left_col, right_col) in &j.on {
                column_names.insert(left_col.clone());
                column_names.insert(right_col.clone());
            }
            let left = push_down(&j.left, column_names)?;
            let right = push_down(&j.right, column_names)?;
            LogicalPlan::Join(Join::new(left, right, j.join_type.clone(), j.on.clone()))
        }
        LogicalPlan::TableScan(s) => {
            // Keep only the source columns that were actually requested, sorted.
            let schema = s.data_source.schema();
            let mut pushdown: Vec<String> = schema
                .fields()
                .iter()
                .map(|f| f.name().clone())
                .filter(|name| column_names.contains(name))
                .collect();
            pushdown.sort();
            LogicalPlan::TableScan(TableScan::new(
                s.path.clone(),
                Arc::clone(&s.data_source),
                pushdown,
            )?)
        }
    };
    Ok(rewritten)
}

#[cfg(test)]
mod tests {
    use super::*;
    use fdapquery_catalog::CsvDataSource;
    use fdapquery_expr::{DataFrame, col, count, format, lit_string, max, min};
    use std::sync::Arc;

    /// `employee` table scanned with no projection yet.
    fn csv() -> DataFrame {
        let path = "../testdata/employee.csv";
        let scan = TableScan::new(
            "employee",
            Arc::new(CsvDataSource::new(path, None, true, 1024)),
            vec![],
        )
        .unwrap();
        DataFrame::new(LogicalPlan::TableScan(scan))
    }

    #[test]
    fn projection_push_down() {
        let df = csv().project(vec![col("id"), col("first_name"), col("last_name")]);
        let optimized = ProjectionPushDownRule
            .try_optimize(df.logical_plan(), &OptimizerConfig)
            .unwrap()
            .expect("rule should fire");
        let expected = "Projection: #id, #first_name, #last_name\n\
                        \tTableScan: employee; projection=[first_name, id, last_name]\n";
        assert_eq!(optimized.pretty(), expected);
    }

    #[test]
    fn projection_push_down_with_selection() {
        let df = csv()
            .filter(col("state").eq(lit_string("CO")))
            .project(vec![col("id"), col("first_name"), col("last_name")]);
        let optimized = ProjectionPushDownRule
            .try_optimize(df.logical_plan(), &OptimizerConfig)
            .unwrap()
            .expect("rule should fire");
        let expected = "Projection: #id, #first_name, #last_name\n\
                        \tFilter: #state = 'CO'\n\
                        \t\tTableScan: employee; projection=[first_name, id, last_name, state]\n";
        assert_eq!(optimized.pretty(), expected);
    }

    #[test]
    fn projection_push_down_with_aggregate_query() {
        let df = csv().aggregate(
            vec![col("state")],
            vec![min(col("salary")), max(col("salary")), count(col("salary"))],
        );
        let optimized = ProjectionPushDownRule
            .try_optimize(df.logical_plan(), &OptimizerConfig)
            .unwrap()
            .expect("rule should fire");
        assert_eq!(
            format(&optimized),
            "Aggregate: groupExpr=[#state], aggregateExpr=[MIN(#salary), MAX(#salary), COUNT(#salary)]\n\
             \tTableScan: employee; projection=[salary, state]\n"
        );
    }
}
