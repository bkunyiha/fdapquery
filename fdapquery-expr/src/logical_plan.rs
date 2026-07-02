//!
//! The six logical operator variants — `TableScan`, `Projection`, `Filter`,
//! `Aggregate`, `Join`, `Limit` — are collected into the `LogicalPlan` enum
//! below; `schema` / `children` / `Display` dispatch to the per-operator
//! structs that live in their own files. The free function [`format()`]
//! produces an indented tree rendering.

use crate::aggregate::Aggregate;
use crate::filter::Filter;
use crate::join::Join;
use crate::limit::Limit;
use crate::projection::Projection;
use crate::scan::TableScan;
use fdapquery_datatypes::{Result, Schema};
use std::fmt;

/// A logical plan: a data transformation or action that returns a relation.
#[derive(Clone)]
pub enum LogicalPlan {
    TableScan(TableScan),
    Projection(Projection),
    Filter(Filter),
    Aggregate(Aggregate),
    Join(Join),
    Limit(Limit),
}

impl LogicalPlan {
    /// Schema of the data produced by this plan.
    pub fn schema(&self) -> Result<Schema> {
        match self {
            LogicalPlan::TableScan(p) => Ok(p.schema()), // TableScan caches its schema at construction
            LogicalPlan::Projection(p) => p.schema(),
            LogicalPlan::Filter(p) => p.schema(),
            LogicalPlan::Aggregate(p) => p.schema(),
            LogicalPlan::Join(p) => p.schema(),
            LogicalPlan::Limit(p) => p.schema(),
        }
    }

    /// Inputs of this plan (for walking the tree).
    pub fn children(&self) -> Vec<&LogicalPlan> {
        match self {
            LogicalPlan::TableScan(p) => p.children(),
            LogicalPlan::Projection(p) => p.children(),
            LogicalPlan::Filter(p) => p.children(),
            LogicalPlan::Aggregate(p) => p.children(),
            LogicalPlan::Join(p) => p.children(),
            LogicalPlan::Limit(p) => p.children(),
        }
    }

    /// Human-readable, indented tree form.
    pub fn pretty(&self) -> String {
        format(self)
    }
}

impl fmt::Display for LogicalPlan {
    /// Single-line description of this node; the tree form is produced by
    /// [`format()`].
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LogicalPlan::TableScan(p) => write!(f, "{p}"),
            LogicalPlan::Projection(p) => write!(f, "{p}"),
            LogicalPlan::Filter(p) => write!(f, "{p}"),
            LogicalPlan::Aggregate(p) => write!(f, "{p}"),
            LogicalPlan::Join(p) => write!(f, "{p}"),
            LogicalPlan::Limit(p) => write!(f, "{p}"),
        }
    }
}

/// Format a logical plan in human-readable form.
pub fn format(plan: &LogicalPlan) -> String {
    format_indent(plan, 0)
}

fn format_indent(plan: &LogicalPlan, indent: usize) -> String {
    let mut b = String::new();
    for _ in 0..indent {
        b.push('\t');
    }
    b.push_str(&plan.to_string());
    b.push('\n');
    for child in plan.children() {
        b.push_str(&format_indent(child, indent + 1));
    }
    b
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aggregate::Aggregate;
    use crate::expr_fn::lit;
    use crate::expressions::{cast, col, max};
    use crate::filter::Filter;
    use crate::projection::Projection;
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

    fn employee_scan() -> TableScan {
        TableScan::new("employee", MockTableSource::employee(), vec![]).unwrap()
    }

    // `Expr::Literal(ScalarValue::Utf8("CO"))` displays
    // bare `CO` (mirrors DataFusion's `ScalarValue::Display`), not the old
    // quoted `'CO'`.

    #[test]
    fn build_logical_plan_manually() {
        let scan = LogicalPlan::TableScan(employee_scan());
        let filter = LogicalPlan::Filter(Filter::new(scan, col("state").eq(lit("CO"))));
        let plan = LogicalPlan::Projection(Projection::new(
            filter,
            vec![col("id"), col("first_name"), col("last_name")],
        ));

        assert_eq!(
            format(&plan),
            "Projection: #id, #first_name, #last_name\n\
             \tFilter: #state = CO\n\
             \t\tTableScan: employee; projection=None\n"
        );
    }

    #[test]
    fn build_logical_plan_nested() {
        let plan = LogicalPlan::Projection(Projection::new(
            LogicalPlan::Filter(Filter::new(
                LogicalPlan::TableScan(employee_scan()),
                col("state").eq(lit("CO")),
            )),
            vec![col("id"), col("first_name"), col("last_name")],
        ));

        assert_eq!(
            format(&plan),
            "Projection: #id, #first_name, #last_name\n\
             \tFilter: #state = CO\n\
             \t\tTableScan: employee; projection=None\n"
        );
    }

    #[test]
    fn build_aggregate_plan() {
        let scan = LogicalPlan::TableScan(employee_scan());
        let group_expr = vec![col("state")];
        let aggregate_expr = vec![max(cast(col("salary"), arrow_schema::DataType::Int32))];
        let plan = LogicalPlan::Aggregate(Aggregate::new(scan, group_expr, aggregate_expr));

        // arrow-rs's `DataType::Int32` renders as `Int32` via `Debug`, so the
        // cast prints `CAST(#salary AS Int32)`.
        assert_eq!(
            format(&plan),
            "Aggregate: groupExpr=[#state], aggregateExpr=[MAX(CAST(#salary AS Int32))]\n\
             \tTableScan: employee; projection=None\n"
        );
    }
}
