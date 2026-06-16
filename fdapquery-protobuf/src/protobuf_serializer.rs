//! `LogicalPlan` → `pb::LogicalPlanNode`, `LogicalExpr` → `pb::LogicalExprNode`.
//! Used by `client` and `distributed` (modules 13–15) to send logical plans
//! over the wire.
//!
//! ## Shape — free functions, no `Serializer` struct
//! Following DataFusion's `datafusion-proto` pattern: free `serialize_X`
//! functions, no struct. Each non-trivial conversion is a function whose name
//! includes the type being converted (`serialize_logical_plan` vs
//! `serialize_logical_expr` vs `serialize_logical_aggregate_expr`).
//!
//! ## Notes
//! - Concrete data-source dispatch uses `ds.as_any().downcast_ref::<…>()`
//!   — the standard Rust idiom (also used by DataFusion's `TableProvider`).
//! - **`LiteralDate`** serialises via the `literal_date` field in the
//!   `.proto` (days since the Unix epoch).
//! - **Aggregate expressions** serialise as
//!   `LogicalExprNode { ExprType::AggregateExpr(...) }`, symmetric with the
//!   deserializer's `AggregateExprNode` arm.

use crate::pb;
use datasource::{CsvDataSource, ParquetDataSource};
use logical_plan::{AggregateExpr, JoinType, LogicalExpr, LogicalPlan};

/// Convert a `LogicalPlan` to its `pb::LogicalPlanNode` form.
pub fn serialize_logical_plan(plan: &LogicalPlan) -> pb::LogicalPlanNode {
    match plan {
        LogicalPlan::Scan(scan) => {
            // Concrete data-source dispatch via `as_any().downcast_ref::<...>()`.
            let projection = Some(pb::ProjectionColumns {
                columns: scan.projection.clone(),
            });
            let any = scan.data_source.as_any();
            if let Some(csv) = any.downcast_ref::<CsvDataSource>() {
                // `has_header` must be propagated to the proto so the
                // deserialiser reconstructs a `CsvDataSource` with the same
                // header-handling configuration. Default-constructing this
                // field (the earlier shape) hard-coded `has_header = false`,
                // which caused the header line of any CSV to be read as a
                // data row on the receiving side — surfacing as an off-by-one
                // row count in the flight-server integration test.
                // The proto field is named `has_header` (singular); the Rust
                // field is `has_headers` (plural). The mapping is correct
                // because the deserialiser (protobuf_deserializer.rs:50)
                // reads `csv.has_header` and passes it as the `has_headers`
                // ctor arg.
                pb::LogicalPlanNode {
                    csv_scan: Some(pb::CsvTableScanNode {
                        path: scan.path.clone(),
                        projection,
                        has_header: csv.has_headers,
                        ..Default::default()
                    }),
                    ..Default::default()
                }
            } else if any.is::<ParquetDataSource>() {
                pb::LogicalPlanNode {
                    parquet_scan: Some(pb::ParquetTableScanNode {
                        path: scan.path.clone(),
                        projection,
                        ..Default::default()
                    }),
                    ..Default::default()
                }
            } else {
                panic!("Unsupported datasource used in scan")
            }
        }
        LogicalPlan::Projection(p) => pb::LogicalPlanNode {
            input: Some(Box::new(serialize_logical_plan(&p.input))),
            projection: Some(pb::ProjectionNode {
                expr: p.expr.iter().map(serialize_logical_expr).collect(),
            }),
            ..Default::default()
        },
        LogicalPlan::Selection(s) => pb::LogicalPlanNode {
            input: Some(Box::new(serialize_logical_plan(&s.input))),
            selection: Some(pb::SelectionNode {
                expr: Some(serialize_logical_expr(&s.expr)),
            }),
            ..Default::default()
        },
        LogicalPlan::Limit(l) => pb::LogicalPlanNode {
            input: Some(Box::new(serialize_logical_plan(&l.input))),
            limit: Some(pb::LimitNode {
                limit: l.limit as u32,
            }),
            ..Default::default()
        },
        LogicalPlan::Aggregate(a) => pb::LogicalPlanNode {
            input: Some(Box::new(serialize_logical_plan(&a.input))),
            aggregate: Some(pb::AggregateNode {
                group_expr: a.group_expr.iter().map(serialize_logical_expr).collect(),
                aggr_expr: a
                    .aggregate_expr
                    .iter()
                    .map(serialize_logical_aggregate_expr)
                    .collect(),
            }),
            ..Default::default()
        },
        LogicalPlan::Join(j) => pb::LogicalPlanNode {
            join: Some(Box::new(pb::JoinNode {
                left: Some(Box::new(serialize_logical_plan(&j.left))),
                right: Some(Box::new(serialize_logical_plan(&j.right))),
                join_type: join_type_to_proto(&j.join_type) as i32,
                left_join_column: j.on.iter().map(|(l, _)| l.clone()).collect(),
                right_join_column: j.on.iter().map(|(_, r)| r.clone()).collect(),
            })),
            ..Default::default()
        },
        // NOTE: the match is exhaustive over all current `LogicalPlan`
        // variants. If a new variant is added, this `match` will fail to
        // compile — which is the intent (force the writer to add a
        // serializer arm rather than panic at runtime).
    }
}

/// Convert a `LogicalExpr` to its `pb::LogicalExprNode` form.
pub fn serialize_logical_expr(expr: &LogicalExpr) -> pb::LogicalExprNode {
    use pb::logical_expr_node::ExprType;
    let expr_type = match expr {
        LogicalExpr::Column(name) => ExprType::ColumnName(name.clone()),
        LogicalExpr::LiteralString(s) => ExprType::LiteralString(s.clone()),
        LogicalExpr::LiteralFloat(n) => ExprType::LiteralF32(*n),
        LogicalExpr::LiteralDouble(n) => ExprType::LiteralF64(*n),
        LogicalExpr::LiteralLong(n) => ExprType::LiteralInt64(*n),
        LogicalExpr::LiteralDate(d) => ExprType::LiteralDate(days_since_unix_epoch(*d)),
        // Boolean and comparison binary operators.
        LogicalExpr::Eq { l, r } => binary_op_variant("eq", l, r),
        LogicalExpr::Neq { l, r } => binary_op_variant("neq", l, r),
        LogicalExpr::Lt { l, r } => binary_op_variant("lt", l, r),
        LogicalExpr::LtEq { l, r } => binary_op_variant("lteq", l, r),
        LogicalExpr::Gt { l, r } => binary_op_variant("gt", l, r),
        LogicalExpr::GtEq { l, r } => binary_op_variant("gteq", l, r),
        LogicalExpr::And { l, r } => binary_op_variant("and", l, r),
        LogicalExpr::Or { l, r } => binary_op_variant("or", l, r),
        other => panic!("Cannot serialize logical expression to protobuf: {other:?}"),
    };
    pb::LogicalExprNode {
        expr_type: Some(expr_type),
    }
}

/// Shared builder for the eight boolean / comparison binary operators.
fn binary_op_variant(
    op: &str,
    l: &LogicalExpr,
    r: &LogicalExpr,
) -> pb::logical_expr_node::ExprType {
    pb::logical_expr_node::ExprType::BinaryExpr(Box::new(pb::BinaryExprNode {
        l: Some(Box::new(serialize_logical_expr(l))),
        r: Some(Box::new(serialize_logical_expr(r))),
        op: op.to_string(),
    }))
}

/// Convert an `AggregateExpr` to a `pb::LogicalExprNode` wrapping the
/// `AggregateExpr` oneof variant. Symmetric with the deserializer's
/// `AggregateExprNode` arm.
pub fn serialize_logical_aggregate_expr(ae: &AggregateExpr) -> pb::LogicalExprNode {
    let (fn_proto, inner) = match ae {
        AggregateExpr::Sum(e) => (pb::AggregateFunction::Sum, e),
        AggregateExpr::Min(e) => (pb::AggregateFunction::Min, e),
        AggregateExpr::Max(e) => (pb::AggregateFunction::Max, e),
        AggregateExpr::Avg(e) => (pb::AggregateFunction::Avg, e),
        AggregateExpr::Count(e) => (pb::AggregateFunction::Count, e),
        AggregateExpr::CountDistinct(e) => (pb::AggregateFunction::CountDistinct, e),
    };
    pb::LogicalExprNode {
        expr_type: Some(pb::logical_expr_node::ExprType::AggregateExpr(Box::new(
            pb::AggregateExprNode {
                aggr_function: fn_proto as i32,
                expr: Some(Box::new(serialize_logical_expr(inner))),
            },
        ))),
    }
}

/// `logical_plan::JoinType` does not implement `Copy` (and the helper only
/// reads it, so there's no reason to take ownership).
fn join_type_to_proto(jt: &JoinType) -> pb::JoinType {
    match jt {
        JoinType::Inner => pb::JoinType::Inner,
        JoinType::Left => pb::JoinType::Left,
        JoinType::Right => pb::JoinType::Right,
    }
}

/// `chrono::NaiveDate` → days since the Unix epoch (1970-01-01). Same helper
/// shape as `query_planner::days_since_unix_epoch`; duplicated here to avoid
/// pulling the entire `query-planner` crate into `protobuf`'s deps just for one
/// trivial date conversion.
fn days_since_unix_epoch(date: chrono::NaiveDate) -> i32 {
    let epoch = chrono::NaiveDate::from_ymd_opt(1970, 1, 1).expect("1970-01-01 is a valid date");
    (date - epoch).num_days() as i32
}

#[cfg(test)]
mod tests {
    //! Roundtrip test: build a small `csv → filter → project` logical plan,
    //! serialise it to protobuf, deserialise it back, and assert the
    //! round-tripped plan re-formats to the same text.
    use super::serialize_logical_plan;
    use crate::deserialize_logical_plan;
    use datasource::CsvDataSource;
    use logical_plan::{DataFrame, LogicalPlan, Scan, col, format, lit_string};
    use std::sync::Arc;

    /// In-repo employee fixture from the workspace-shared `testdata/`
    /// directory, also used by the execution-module tests.
    const EMPLOYEE_CSV: &str = "../testdata/employee.csv";

    fn csv_df() -> DataFrame {
        let csv = CsvDataSource::new(EMPLOYEE_CSV, None, true, 1024);
        DataFrame::new(LogicalPlan::Scan(Scan::new(
            EMPLOYEE_CSV,
            Arc::new(csv),
            vec![],
        )))
    }

    fn roundtrip(df: DataFrame) -> LogicalPlan {
        let proto = serialize_logical_plan(df.logical_plan());
        deserialize_logical_plan(&proto)
    }

    #[test]
    fn convert_plan_to_protobuf() {
        let df = csv_df()
            .filter(col("state").eq(lit_string("CO")))
            .project(vec![col("id"), col("first_name"), col("last_name")]);
        let logical_plan = roundtrip(df);

        let expected = "Projection: #id, #first_name, #last_name\n\
                        \tSelection: #state = 'CO'\n\
                        \t\tScan: ../testdata/employee.csv; projection=None\n";
        assert_eq!(format(&logical_plan), expected);
    }
}
