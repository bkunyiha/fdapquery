//! `LogicalPlan` → `protobuf::LogicalPlanNode`, `Expr` → `protobuf::LogicalExprNode`.
//! Used by the `fdapquery-flight-client`, `fdapquery-flight-server`, and
//! `fdapquery-distributed` crates to send logical plans over the wire.
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
//! - **Literal expressions** serialise via `scalar_to_proto_expr_type`,
//!   which dispatches on the `ScalarValue` inside `Expr::Literal` to pick
//!   the matching wire field (`literal_string` / `literal_int64` /
//!   `literal_f64` / `literal_f32` / `literal_date`). Date32 ScalarValues
//!   already carry "days since the Unix epoch", so no extra conversion is
//!   needed at the seam.
//! - **Aggregate expressions** serialise as
//!   `LogicalExprNode { ExprType::AggregateExpr(...) }`, symmetric with the
//!   deserializer's `AggregateExprNode` arm. Folded the
//!   logical-side `AggregateExpr` enum into `Expr::AggregateFunction`;
//!   the wire-format `AggregateExprNode` shape is unchanged (the proto
//!   rename is part of #110/#111).

use crate::protobuf;
use fdapquery_catalog::{CsvDataSource, ParquetDataSource, source_as_provider};
use fdapquery_common::ScalarValue;
use fdapquery_expr::{AggregateFunctionKind, Expr, JoinType, LogicalPlan, Operator};

/// Convert a `LogicalPlan` to its `protobuf::LogicalPlanNode` form.
pub fn serialize_logical_plan(plan: &LogicalPlan) -> protobuf::LogicalPlanNode {
    match plan {
        LogicalPlan::TableScan(scan) => {
            // Concrete data-source dispatch via `as_any().downcast_ref::<...>()`.
            // `TableScan::data_source` is now
            // `Arc<dyn TableSource>` (the logical-side trait). Recover the
            // underlying `Arc<dyn TableProvider>` via `source_as_provider`
            // (unwraps the `DefaultTableSource` adapter) so the concrete
            // CSV / Parquet downcast still resolves to the wire-format
            // source type. Verbatim DataFusion's serializer pattern at the
            // logical seam.
            let projection = Some(protobuf::ProjectionColumns {
                columns: scan.projection.clone(),
            });
            let provider = source_as_provider(&scan.data_source)
                .expect("serialize_logical_plan: TableScan source must wrap a TableProvider");
            let any = provider.as_any();
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
                // because the deserialiser
                // (`fdapquery_proto::protobuf_deserializer::deserialize_logical_plan`)
                // reads `csv.has_header` and passes it as the `has_headers`
                // ctor arg.
                protobuf::LogicalPlanNode {
                    csv_scan: Some(protobuf::CsvTableScanNode {
                        path: scan.path.clone(),
                        projection,
                        has_header: csv.has_headers,
                        ..Default::default()
                    }),
                    ..Default::default()
                }
            } else if any.is::<ParquetDataSource>() {
                protobuf::LogicalPlanNode {
                    parquet_scan: Some(protobuf::ParquetTableScanNode {
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
        LogicalPlan::Projection(p) => protobuf::LogicalPlanNode {
            input: Some(Box::new(serialize_logical_plan(&p.input))),
            projection: Some(protobuf::ProjectionNode {
                expr: p.expr.iter().map(serialize_logical_expr).collect(),
            }),
            ..Default::default()
        },
        LogicalPlan::Filter(s) => protobuf::LogicalPlanNode {
            input: Some(Box::new(serialize_logical_plan(&s.input))),
            // Wire field name `selection` — Stable.
            selection: Some(protobuf::FilterNode {
                expr: Some(serialize_logical_expr(&s.expr)),
            }),
            ..Default::default()
        },
        LogicalPlan::Limit(l) => protobuf::LogicalPlanNode {
            input: Some(Box::new(serialize_logical_plan(&l.input))),
            limit: Some(protobuf::LimitNode {
                limit: l.limit as u32,
            }),
            ..Default::default()
        },
        LogicalPlan::Aggregate(a) => protobuf::LogicalPlanNode {
            input: Some(Box::new(serialize_logical_plan(&a.input))),
            aggregate: Some(protobuf::AggregateNode {
                group_expr: a.group_expr.iter().map(serialize_logical_expr).collect(),
                aggr_expr: a
                    .aggregate_expr
                    .iter()
                    .map(serialize_logical_aggregate_expr)
                    .collect(),
            }),
            ..Default::default()
        },
        LogicalPlan::Join(j) => protobuf::LogicalPlanNode {
            join: Some(Box::new(protobuf::JoinNode {
                left: Some(Box::new(serialize_logical_plan(&j.left))),
                right: Some(Box::new(serialize_logical_plan(&j.right))),
                join_type: join_type_to_proto(j.join_type) as i32,
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

/// Convert a `Expr` to its `protobuf::LogicalExprNode` form.
pub fn serialize_logical_expr(expr: &Expr) -> protobuf::LogicalExprNode {
    use protobuf::logical_expr_node::ExprType;
    let expr_type = match expr {
        Expr::Column(name) => ExprType::ColumnName(name.clone()),
        // `Expr::Literal(ScalarValue)` collapsed the five
        // sibling logical literal variants. The proto wire format still has
        // distinct `LiteralString` / `LiteralInt64` / `LiteralF32` /
        // `LiteralF64` / `LiteralDate` fields (proto tasks #110/#111 are
        // where that gets widened), so the serializer dispatches on the
        // inner `ScalarValue` to pick the matching wire field.
        Expr::Literal(scalar) => scalar_to_proto_expr_type(scalar),
        // Single binary arm dispatches on the
        // [`Operator`] enum. The wire format keeps its string-typed `op`
        // field (the proto rename is part of #110/#111), so we map each
        // operator to its wire string here.
        Expr::BinaryExpr { left, op, right } => {
            binary_op_variant(operator_wire_name(*op), left, right)
        }
        other => panic!("Cannot serialize logical expression to protobuf: {other:?}"),
    };
    protobuf::LogicalExprNode {
        expr_type: Some(expr_type),
    }
}

/// Shared builder for the unified [`Expr::BinaryExpr`] wire encoding.
fn binary_op_variant(op: &str, l: &Expr, r: &Expr) -> protobuf::logical_expr_node::ExprType {
    protobuf::logical_expr_node::ExprType::BinaryExpr(Box::new(protobuf::BinaryExprNode {
        l: Some(Box::new(serialize_logical_expr(l))),
        r: Some(Box::new(serialize_logical_expr(r))),
        op: op.to_string(),
    }))
}

/// Map an [`Operator`] to the wire-format op string used by the
/// existing `BinaryExprNode { op: string }` field. Operators not in this
/// set panic — the engine doesn't emit them today, and proto tasks
/// #110/#111 will widen the wire format to a typed `Operator` enum.
fn operator_wire_name(op: Operator) -> &'static str {
    match op {
        Operator::Eq => "eq",
        Operator::NotEq => "neq",
        Operator::Lt => "lt",
        Operator::LtEq => "lteq",
        Operator::Gt => "gt",
        Operator::GtEq => "gteq",
        Operator::And => "and",
        Operator::Or => "or",
        Operator::Plus => "add",
        Operator::Minus => "subtract",
        Operator::Multiply => "multiply",
        Operator::Divide => "divide",
        Operator::Modulo => "modulus",
        other => panic!(
            "operator_wire_name: Operator::{other:?} has no wire-format \
             encoding in rquery.proto's BinaryExprNode.op string (proto \
             tasks #110/#111)"
        ),
    }
}

/// Convert an aggregate `Expr::AggregateFunction(...)` to a
/// `protobuf::LogicalExprNode` wrapping the `AggregateExpr` oneof
/// variant. Symmetric with the deserializer's `AggregateExprNode` arm.
///
/// The input is now an `&Expr` (must be the
/// `AggregateFunction` variant); the wire format is unchanged. DISTINCT
/// dispatch maps to the existing `protobuf::AggregateFunction::CountDistinct`
/// enum value (the wire-format catalogue is unchanged pending
/// proto tasks #110/#111).
pub fn serialize_logical_aggregate_expr(ae: &Expr) -> protobuf::LogicalExprNode {
    let af = match ae {
        Expr::AggregateFunction(af) => af,
        other => panic!(
            "serialize_logical_aggregate_expr: expected Expr::AggregateFunction, \
             found {other:?}"
        ),
    };
    let inner = af.params.args.first().unwrap_or_else(|| {
        panic!("serialize_logical_aggregate_expr: AggregateFunction has no arguments")
    });
    let fn_proto = match (af.func, af.params.distinct) {
        (AggregateFunctionKind::Sum, _) => protobuf::AggregateFunction::Sum,
        (AggregateFunctionKind::Min, _) => protobuf::AggregateFunction::Min,
        (AggregateFunctionKind::Max, _) => protobuf::AggregateFunction::Max,
        (AggregateFunctionKind::Avg, _) => protobuf::AggregateFunction::Avg,
        (AggregateFunctionKind::Count, false) => protobuf::AggregateFunction::Count,
        (AggregateFunctionKind::Count, true) => protobuf::AggregateFunction::CountDistinct,
    };
    protobuf::LogicalExprNode {
        expr_type: Some(protobuf::logical_expr_node::ExprType::AggregateExpr(
            Box::new(protobuf::AggregateExprNode {
                aggr_function: fn_proto as i32,
                expr: Some(Box::new(serialize_logical_expr(inner))),
            }),
        )),
    }
}

/// Lower a `fdapquery_expr::JoinType` to the wire-format enum
/// `protobuf::JoinType`. The wire format currently encodes only the three
/// classical variants (`Inner` / `Left` / `Right`) — see
/// `proto/rquery.proto` `enum JoinType`. The other seven variants
/// (`Full`, `LeftSemi`, `RightSemi`, `LeftAnti`, `RightAnti`,
/// `LeftMark`, `RightMark`) are part of `fdapquery_expr::JoinType` for
/// strict API parity with `datafusion_common::JoinType`, but until
/// proto tasks #110/#111 widen the wire format we panic here so that
/// any attempt to serialise an extended-variant plan fails loudly at
/// the seam (rather than silently masquerading as Inner).
///
/// The signature takes `JoinType` by value (the type is `Copy`).
fn join_type_to_proto(jt: JoinType) -> protobuf::JoinType {
    match jt {
        JoinType::Inner => protobuf::JoinType::Inner,
        JoinType::Left => protobuf::JoinType::Left,
        JoinType::Right => protobuf::JoinType::Right,
        JoinType::Full
        | JoinType::LeftSemi
        | JoinType::RightSemi
        | JoinType::LeftAnti
        | JoinType::RightAnti
        | JoinType::LeftMark
        | JoinType::RightMark => {
            panic!(
                "join_type_to_proto: rquery.proto JoinType has no wire-format \
                 encoding for {jt:?} (proto tasks #110/#111 widen the wire format)"
            )
        }
    }
}

/// Dispatch a logical `ScalarValue` into the matching `protobuf::LogicalExprNode`
/// oneof variant. Collapsed the five sibling logical
/// `Expr::Literal*` variants into `Expr::Literal(ScalarValue)`; the wire
/// format still distinguishes string / integer / float / date (until proto
/// tasks #110/#111 unify it), so we dispatch on the ScalarValue here.
///
/// Only the variants the engine currently emits are covered. Other variants
/// — `Null`, the narrower integer widths, `UInt*`, `Binary`, `Boolean` —
/// panic because the planner never produces them; reaching this arm signals
/// an engine bug, not a malformed input.
fn scalar_to_proto_expr_type(scalar: &ScalarValue) -> protobuf::logical_expr_node::ExprType {
    use protobuf::logical_expr_node::ExprType;
    match scalar {
        ScalarValue::Utf8(s) => ExprType::LiteralString(s.clone()),
        ScalarValue::Float32(n) => ExprType::LiteralF32(*n),
        ScalarValue::Float64(n) => ExprType::LiteralF64(*n),
        ScalarValue::Int64(n) => ExprType::LiteralInt64(*n),
        ScalarValue::Date32(d) => ExprType::LiteralDate(*d),
        other => panic!(
            "scalar_to_proto_expr_type: ScalarValue variant {other:?} has no \
             wire-format encoding in rquery.proto yet (proto tasks #110/#111)"
        ),
    }
}

#[cfg(test)]
mod tests {
    //! Roundtrip test: build a small `csv → filter → project` logical plan,
    //! serialise it to protobuf, deserialise it back, and assert the
    //! round-tripped plan re-formats to the same text.
    use super::serialize_logical_plan;
    use crate::deserialize_logical_plan;
    use fdapquery_catalog::{CsvDataSource, provider_as_source};
    use fdapquery_expr::{DataFrame, LogicalPlan, TableScan, col, format, lit};
    use std::sync::Arc;

    /// In-repo employee fixture from the workspace-shared `testdata/`
    /// directory, also used by the execution-module tests.
    const EMPLOYEE_CSV: &str = "../testdata/employee.csv";

    fn csv_df() -> DataFrame {
        let csv = CsvDataSource::new(EMPLOYEE_CSV, None, true, 1024);
        DataFrame::new(LogicalPlan::TableScan(
            TableScan::new(EMPLOYEE_CSV, provider_as_source(Arc::new(csv)), vec![]).unwrap(),
        ))
    }

    fn roundtrip(df: &DataFrame) -> LogicalPlan {
        let proto = serialize_logical_plan(df.logical_plan());
        deserialize_logical_plan(&proto)
    }

    #[test]
    fn convert_plan_to_protobuf() {
        let df = csv_df().filter(col("state").eq(lit("CO"))).project(vec![
            col("id"),
            col("first_name"),
            col("last_name"),
        ]);
        let logical_plan = roundtrip(&df);

        // `Expr::Literal(ScalarValue::Utf8("CO"))` displays
        // bare `CO` (mirrors DataFusion's `ScalarValue::Display`), not the
        // old quoted `'CO'`.
        let expected = "Projection: #id, #first_name, #last_name\n\
                        \tFilter: #state = CO\n\
                        \t\tTableScan: ../testdata/employee.csv; projection=None\n";
        assert_eq!(format(&logical_plan), expected);
    }
}
