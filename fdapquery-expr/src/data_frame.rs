//!
//! A fluent, `self`-consuming builder that wraps a `LogicalPlan`: each
//! transformation takes the current frame by value and returns a new one
//! wrapping the extended plan.

use crate::aggregate::Aggregate;
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

    /// Aggregate. Each element of `aggregate_expr` must be
    /// `Expr::AggregateFunction(...)` — mirrors DataFusion's
    /// `LogicalPlan::Aggregate` invariant.
    pub fn aggregate(self, group_by: Vec<Expr>, aggregate_expr: Vec<Expr>) -> DataFrame {
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
    use crate::expr_fn::lit;
    use crate::expressions::{col, count, max, min};
    use crate::logical_plan::{LogicalPlan, format};
    use crate::scan::TableScan;
    use crate::table_source::TableSource;
    use arrow_schema::DataType;
    use fdapquery_datatypes::{Field, Schema};
    use std::sync::Arc;

    /// Minimal `TableSource` mock for logical-plan tests.
    ///
    /// fdapquery-expr deliberately does not depend on fdapquery-catalog
    /// (The two-trait split). Tests that need a
    /// `TableSource` build this in-file mock instead of pulling
    /// `CsvDataSource` in as a dev-dep. Mirrors DataFusion's
    /// `datafusion_expr::test::test_table_source` pattern.
    #[derive(Debug)]
    struct MockTableSource {
        schema: Schema,
    }

    impl MockTableSource {
        fn employee() -> Arc<dyn TableSource> {
            Arc::new(Self {
                schema: Schema::new(vec![
                    Field::new("id", DataType::Int64, true),
                    Field::new("first_name", DataType::Utf8, true),
                    Field::new("last_name", DataType::Utf8, true),
                    Field::new("state", DataType::Utf8, true),
                    Field::new("job_title", DataType::Utf8, true),
                    Field::new("salary", DataType::Int64, true),
                ]),
            })
        }
    }

    impl TableSource for MockTableSource {
        fn schema(&self) -> Schema {
            self.schema.clone()
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    fn csv() -> DataFrame {
        let scan = TableScan::new("employee", MockTableSource::employee(), vec![]).unwrap();
        DataFrame::new(LogicalPlan::TableScan(scan))
    }

    #[test]
    fn build_data_frame() {
        let df = csv().filter(col("state").eq(lit("CO"))).project(vec![
            col("id"),
            col("first_name"),
            col("last_name"),
        ]);

        // `Expr::Literal(ScalarValue::Utf8("CO"))` displays
        // bare `CO` (mirrors DataFusion's `ScalarValue::Display`), not the
        // old quoted `'CO'`.
        let expected = "Projection: #id, #first_name, #last_name\n\
                        \tFilter: #state = CO\n\
                        \t\tTableScan: employee; projection=None\n";

        assert_eq!(format(df.logical_plan()), expected);
    }

    #[test]
    fn multiplier_and_alias() {
        let df = csv()
            .filter(col("state").eq(lit("CO")))
            .project(vec![
                col("id"),
                col("first_name"),
                col("last_name"),
                col("salary"),
                col("salary").mult(lit(0.1_f64)).alias("bonus"),
            ])
            .filter(col("bonus").gt(lit(1000_i64)));

        let expected = "Filter: #bonus > 1000\n\
                        \tProjection: #id, #first_name, #last_name, #salary, #salary * 0.1 as bonus\n\
                        \t\tFilter: #state = CO\n\
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
