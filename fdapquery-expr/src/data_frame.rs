//!
//! A fluent, `self`-consuming builder that wraps a `LogicalPlan`: each
//! transformation takes the current frame by value and returns a new one
//! wrapping the extended plan.

use crate::aggregate::Aggregate;
use crate::expressions::AggregateExpr;
use crate::filter::Filter;
use crate::join::{Join, JoinType};
use crate::limit::Limit;
use crate::logical_expr::Expr;
use crate::logical_plan::LogicalPlan;
use crate::projection::Projection;
use fdapquery_datatypes::{Result, Schema};

/// Fluent builder over a [`LogicalPlan`].
#[derive(Clone)]
pub struct DataFrame {
    plan: LogicalPlan,
}

impl DataFrame {
    /// Wrap an existing plan.
    pub fn new(plan: LogicalPlan) -> Self {
        Self { plan }
    }

    /// Apply a projection.
    pub fn project(self, expr: Vec<Expr>) -> DataFrame {
        DataFrame {
            plan: LogicalPlan::Projection(Projection::new(self.plan, expr)),
        }
    }

    /// Apply a filter.
    pub fn filter(self, expr: Expr) -> DataFrame {
        DataFrame {
            plan: LogicalPlan::Filter(Filter::new(self.plan, expr)),
        }
    }

    /// Aggregate.
    pub fn aggregate(self, group_by: Vec<Expr>, aggregate_expr: Vec<AggregateExpr>) -> DataFrame {
        DataFrame {
            plan: LogicalPlan::Aggregate(Aggregate::new(self.plan, group_by, aggregate_expr)),
        }
    }

    /// Limit the number of rows.
    pub fn limit(self, n: i32) -> DataFrame {
        DataFrame {
            plan: LogicalPlan::Limit(Limit::new(self.plan, n)),
        }
    }

    /// Join with another DataFrame.
    pub fn join(
        self,
        right: DataFrame,
        join_type: JoinType,
        on: Vec<(String, String)>,
    ) -> DataFrame {
        DataFrame {
            plan: LogicalPlan::Join(Join::new(
                self.plan,
                right.into_logical_plan(),
                join_type,
                on,
            )),
        }
    }

    /// Schema of the data this DataFrame will produce.
    pub fn schema(&self) -> Result<Schema> {
        self.plan.schema()
    }

    /// Borrow the underlying logical plan.
    pub fn logical_plan(&self) -> &LogicalPlan {
        &self.plan
    }

    /// Consume the DataFrame and return the underlying logical plan.
    pub fn into_logical_plan(self) -> LogicalPlan {
        self.plan
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expressions::{col, count, lit_double, lit_long, lit_string, max, min};
    use crate::logical_plan::{LogicalPlan, format};
    use crate::scan::TableScan;
    use fdapquery_catalog::CsvDataSource;
    use std::sync::Arc;

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
    fn build_data_frame() {
        let df = csv()
            .filter(col("state").eq(lit_string("CO")))
            .project(vec![col("id"), col("first_name"), col("last_name")]);

        let expected = "Projection: #id, #first_name, #last_name\n\
                        \tFilter: #state = 'CO'\n\
                        \t\tTableScan: employee; projection=None\n";

        assert_eq!(format(df.logical_plan()), expected);
    }

    #[test]
    fn multiplier_and_alias() {
        let df = csv()
            .filter(col("state").eq(lit_string("CO")))
            .project(vec![
                col("id"),
                col("first_name"),
                col("last_name"),
                col("salary"),
                col("salary").mult(lit_double(0.1)).alias("bonus"),
            ])
            .filter(col("bonus").gt(lit_long(1000)));

        let expected = "Filter: #bonus > 1000\n\
                        \tProjection: #id, #first_name, #last_name, #salary, #salary * 0.1 as bonus\n\
                        \t\tFilter: #state = 'CO'\n\
                        \t\t\tTableScan: employee; projection=None\n";

        assert_eq!(format(df.logical_plan()), expected);
    }

    #[test]
    fn aggregate_query() {
        let df = csv().aggregate(
            vec![col("state")],
            vec![min(col("salary")), max(col("salary")), count(col("salary"))],
        );

        assert_eq!(
            format(df.logical_plan()),
            "Aggregate: groupExpr=[#state], aggregateExpr=[MIN(#salary), MAX(#salary), COUNT(#salary)]\n\
             \tTableScan: employee; projection=None\n"
        );
    }
}
