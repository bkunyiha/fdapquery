//! Pushes the set of referenced columns down to the `Scan`, so the data source
//! reads only the columns the rest of the query actually needs.
//!
//! `LogicalPlan` is a closed `enum`, so the `match` is exhaustive over all
//! six operators and needs no catch-all (per §3.1).

use logical_plan::{Aggregate, Join, Limit, LogicalPlan, Projection, Scan, Selection};
use std::collections::HashSet;
use std::sync::Arc;

use crate::optimizer::{OptimizerRule, aggregate_inner, extract_columns, extract_columns_list};

/// The one optimisation rule so far.
pub struct ProjectionPushDownRule;

impl OptimizerRule for ProjectionPushDownRule {
    fn optimize(&self, plan: &LogicalPlan) -> LogicalPlan {
        push_down(plan, &mut HashSet::new())
    }
}

/// Rewrite `plan`, accumulating referenced column names on the way down and
/// trimming the `Scan`'s projection at the leaf.
fn push_down(plan: &LogicalPlan, column_names: &mut HashSet<String>) -> LogicalPlan {
    match plan {
        LogicalPlan::Projection(p) => {
            extract_columns_list(&p.expr, &p.input, column_names);
            let input = push_down(&p.input, column_names);
            LogicalPlan::Projection(Projection::new(input, p.expr.clone()))
        }
        LogicalPlan::Selection(s) => {
            extract_columns(&s.expr, &s.input, column_names);
            let input = push_down(&s.input, column_names);
            LogicalPlan::Selection(Selection::new(input, s.expr.clone()))
        }
        LogicalPlan::Aggregate(a) => {
            extract_columns_list(&a.group_expr, &a.input, column_names);
            // Collect the columns referenced by each aggregate's *argument*
            // expression.
            for agg in &a.aggregate_expr {
                extract_columns(aggregate_inner(agg), &a.input, column_names);
            }
            let input = push_down(&a.input, column_names);
            LogicalPlan::Aggregate(Aggregate::new(
                input,
                a.group_expr.clone(),
                a.aggregate_expr.clone(),
            ))
        }
        LogicalPlan::Limit(l) => {
            let input = push_down(&l.input, column_names);
            LogicalPlan::Limit(Limit::new(input, l.limit))
        }
        LogicalPlan::Join(j) => {
            // If nothing has been requested yet (the join is at the root),
            // request every column from both sides.
            if column_names.is_empty() {
                for f in j.left.schema().fields {
                    column_names.insert(f.name);
                }
                for f in j.right.schema().fields {
                    column_names.insert(f.name);
                }
            }
            // The join keys are always required.
            for (left_col, right_col) in &j.on {
                column_names.insert(left_col.clone());
                column_names.insert(right_col.clone());
            }
            let left = push_down(&j.left, column_names);
            let right = push_down(&j.right, column_names);
            LogicalPlan::Join(Join::new(left, right, j.join_type.clone(), j.on.clone()))
        }
        LogicalPlan::Scan(s) => {
            // Keep only the source columns that were actually requested, sorted.
            let schema = s.data_source.schema();
            let mut pushdown: Vec<String> = schema
                .fields
                .iter()
                .map(|f| f.name.clone())
                .filter(|name| column_names.contains(name))
                .collect();
            pushdown.sort();
            LogicalPlan::Scan(Scan::new(
                s.path.clone(),
                Arc::clone(&s.data_source),
                pushdown,
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use datasource::CsvDataSource;
    use logical_plan::{DataFrame, col, count, format, lit_string, max, min};
    use std::sync::Arc;

    /// `employee` table scanned with no projection yet.
    fn csv() -> DataFrame {
        let path = "../testdata/employee.csv";
        let scan = Scan::new(
            "employee",
            Arc::new(CsvDataSource::new(path, None, true, 1024)),
            vec![],
        );
        DataFrame::new(LogicalPlan::Scan(scan))
    }

    #[test]
    fn projection_push_down() {
        let df = csv().project(vec![col("id"), col("first_name"), col("last_name")]);
        let optimized = ProjectionPushDownRule.optimize(df.logical_plan());
        let expected = "Projection: #id, #first_name, #last_name\n\
                        \tScan: employee; projection=[first_name, id, last_name]\n";
        assert_eq!(optimized.pretty(), expected);
    }

    #[test]
    fn projection_push_down_with_selection() {
        let df = csv()
            .filter(col("state").eq(lit_string("CO")))
            .project(vec![col("id"), col("first_name"), col("last_name")]);
        let optimized = ProjectionPushDownRule.optimize(df.logical_plan());
        let expected = "Projection: #id, #first_name, #last_name\n\
                        \tSelection: #state = 'CO'\n\
                        \t\tScan: employee; projection=[first_name, id, last_name, state]\n";
        assert_eq!(optimized.pretty(), expected);
    }

    #[test]
    fn projection_push_down_with_aggregate_query() {
        let df = csv().aggregate(
            vec![col("state")],
            vec![min(col("salary")), max(col("salary")), count(col("salary"))],
        );
        let optimized = ProjectionPushDownRule.optimize(df.logical_plan());
        assert_eq!(
            format(&optimized),
            "Aggregate: groupExpr=[#state], aggregateExpr=[MIN(#salary), MAX(#salary), COUNT(#salary)]\n\
             \tScan: employee; projection=[salary, state]\n"
        );
    }
}
