//!
//! A fluent, `self`-consuming builder that wraps a `LogicalPlan`: each
//! transformation takes the current frame by value and returns a new one
//! wrapping the extended plan.

use crate::aggregate::Aggregate;
use crate::expressions::AggregateExpr;
use crate::join::{Join, JoinType};
use crate::limit::Limit;
use crate::logical_expr::LogicalExpr;
use crate::logical_plan::LogicalPlan;
use crate::projection::Projection;
use crate::selection::Selection;
use datatypes::Schema;

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
    pub fn project(self, expr: Vec<LogicalExpr>) -> DataFrame {
        DataFrame {
            plan: LogicalPlan::Projection(Projection::new(self.plan, expr)),
        }
    }

    /// Apply a filter.
    pub fn filter(self, expr: LogicalExpr) -> DataFrame {
        DataFrame {
            plan: LogicalPlan::Selection(Selection::new(self.plan, expr)),
        }
    }

    /// Aggregate.
    pub fn aggregate(
        self,
        group_by: Vec<LogicalExpr>,
        aggregate_expr: Vec<AggregateExpr>,
    ) -> DataFrame {
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
    pub fn schema(&self) -> Schema {
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
    use crate::scan::Scan;
    use datasource::CsvDataSource;
    use std::sync::Arc;

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
    fn build_data_frame() {
        let df = csv()
            .filter(col("state").eq(lit_string("CO")))
            .project(vec![col("id"), col("first_name"), col("last_name")]);

        let expected = "Projection: #id, #first_name, #last_name\n\
                        \tSelection: #state = 'CO'\n\
                        \t\tScan: employee; projection=None\n";

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

        let expected = "Selection: #bonus > 1000\n\
                        \tProjection: #id, #first_name, #last_name, #salary, #salary * 0.1 as bonus\n\
                        \t\tSelection: #state = 'CO'\n\
                        \t\t\tScan: employee; projection=None\n";

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
             \tScan: employee; projection=None\n"
        );
    }
}
