//! `protobuf::LogicalPlanNode` → `LogicalPlan`, `protobuf::LogicalExprNode` → `Expr`,
//! plus the action/schema/field helpers. Inverse of
//! [`crate::protobuf_serializer`].
//!
//! ## Shape — free functions, no `Deserializer` struct
//! Same shape as the serializer side (see `protobuf_serializer.rs` module
//! doc): no stateful struct, just free `deserialize_X` functions per
//! DataFusion's `datafusion-proto` convention. Each non-trivial conversion
//! is a function whose name includes the type being produced
//! (`deserialize_logical_plan` vs `deserialize_logical_expr` vs
//! `deserialize_schema`).
//!
//! ## Notes
//! - Each `protobuf::LogicalPlanNode` variant is dispatched via a chain of
//!   `if let Some(_) = &node.<field>` arms. prost emits each message-typed
//!   plan field as `Option<T>`, so the "is this variant set?" check is
//!   `node.x.is_some()`.
//! - Collapsed the logical-side five sibling literal
//!   variants into a single `Expr::Literal(ScalarValue)`, mirroring
//!   DataFusion. Each `protobuf::Literal*` wire field deserialises to a
//!   `ScalarValue::*` of the matching width — `LiteralInt8/16/32/64` and
//!   `LiteralUint8/16/32/64` all widen into `ScalarValue::Int64`, and the
//!   `literal_date` arm yields `ScalarValue::Date32(days_since_unix_epoch)`
//!   directly (no `chrono::NaiveDate` round trip — Date32 already encodes
//!   days since the Unix epoch).
//! - `IsNull` / `IsNotNull` / `Not` arms are unimplemented; their logical-plan
//!   variants don't exist yet, so the arms `panic!` with a clear message
//!   rather than guess at semantics.

use crate::protobuf;
use fdapquery_catalog::{CsvDataSource, ParquetDataSource, provider_as_source};
use fdapquery_common::ScalarValue;
use fdapquery_datatypes::{Field, Schema};
use fdapquery_expr::{
    Aggregate, AggregateFunction, AggregateFunctionKind, Expr, Filter, Limit, LogicalPlan,
    Operator, Projection, TableScan,
};
// JoinNode is not deserialised here. If/when that's added, re-import `JoinType`.
use fdapquery_physical_plan::{Action, QueryAction};
use std::sync::Arc;

use arrow_schema::DataType;

/// `protobuf::LogicalPlanNode` → `LogicalPlan`.
pub fn deserialize_logical_plan(node: &protobuf::LogicalPlanNode) -> LogicalPlan {
    if let Some(csv) = &node.csv_scan {
        // The schema field is set by the serializer only for Parquet — for
        // CSV the proto schema is left unset, so we pass `None` and let
        // `CsvDataSource` re-infer from the file.
        let ds = CsvDataSource::new(&csv.path, None, csv.has_header, 1024);
        // `TableScan` holds `Arc<dyn TableSource>`,
        // so wrap the provider with the `DefaultTableSource` adapter
        // via `provider_as_source`.
        LogicalPlan::TableScan(
            TableScan::new(
                &csv.path,
                provider_as_source(Arc::new(ds)),
                csv.projection
                    .as_ref()
                    .map(|p| p.columns.clone())
                    .unwrap_or_default(),
            )
            .expect("deserialize_logical_plan: CSV scan construction"),
        )
    } else if let Some(parquet) = &node.parquet_scan {
        let ds = ParquetDataSource::new(&parquet.path);
        LogicalPlan::TableScan(
            TableScan::new(
                &parquet.path,
                provider_as_source(Arc::new(ds)),
                parquet
                    .projection
                    .as_ref()
                    .map(|p| p.columns.clone())
                    .unwrap_or_default(),
            )
            .expect("deserialize_logical_plan: Parquet scan construction"),
        )
    // Wire field name `selection` is fixed by `FilterNode selection = 21;`
    // in rquery.proto. Stable across the Rust-side rename.
    } else if let Some(sel) = &node.selection {
        let input = deserialize_plan_input(node);
        let expr = deserialize_logical_expr(sel.expr.as_ref().expect("FilterNode.expr unset"));
        LogicalPlan::Filter(Filter::new(input, expr))
    } else if let Some(proj) = &node.projection {
        let input = deserialize_plan_input(node);
        let expr = proj.expr.iter().map(deserialize_logical_expr).collect();
        LogicalPlan::Projection(Projection::new(input, expr))
    } else if let Some(lim) = &node.limit {
        let input = deserialize_plan_input(node);
        LogicalPlan::Limit(Limit::new(input, lim.limit as i32))
    } else if let Some(agg) = &node.aggregate {
        let input = deserialize_plan_input(node);
        let group_expr = agg
            .group_expr
            .iter()
            .map(deserialize_logical_expr)
            .collect();
        // Each `aggr_expr` is a LogicalExprNode whose oneof is the
        // AggregateExpr variant; the deserialised `Expr` must be the
        // `Expr::AggregateFunction(...)` variant (#115 — fold of the
        // standalone `AggregateExpr` enum). The `Aggregate` plan's
        // `aggregate_expr` slot is `Vec<Expr>`.
        let aggregate_expr = agg
            .aggr_expr
            .iter()
            .map(|e| match deserialize_logical_expr(e) {
                expr @ Expr::AggregateFunction(_) => expr,
                other => panic!(
                    "AggregateNode.aggr_expr did not deserialise to an \
                     Expr::AggregateFunction: {other:?}"
                ),
            })
            .collect();
        LogicalPlan::Aggregate(Aggregate::new(input, group_expr, aggregate_expr))
    } else {
        panic!("Failed to parse logical operator: no recognised plan field set")
    }
}

/// Helper: pull the recursive `LogicalPlanNode.input` field (which prost
/// generates as `Option<Box<LogicalPlanNode>>`) and deserialise it. Panics
/// with a clear message if `input` is unset on a node that expects one
/// (Projection / Filter / Limit / Aggregate).
fn deserialize_plan_input(node: &protobuf::LogicalPlanNode) -> LogicalPlan {
    let inner = node
        .input
        .as_deref()
        .expect("LogicalPlanNode.input unset on a non-leaf plan");
    deserialize_logical_plan(inner)
}

/// `protobuf::LogicalExprNode` → `Expr`.
//
// The integer-literal arms bind values of distinct concrete types
// (`i8`/`i16`/`i32`/`i64` and the unsigned siblings) and each widens into
// `ScalarValue::Int64` with the appropriate conversion. They cannot be
// merged via `|` because the binding types differ, even though the bodies
// read identically after the typed conversions are applied.
#[allow(clippy::match_same_arms)]
pub fn deserialize_logical_expr(node: &protobuf::LogicalExprNode) -> Expr {
    use protobuf::logical_expr_node::ExprType;
    match node.expr_type.as_ref() {
        Some(ExprType::ColumnName(name)) => Expr::Column(name.clone()),
        Some(ExprType::LiteralString(s)) => Expr::Literal(ScalarValue::Utf8(s.clone())),
        // All integer literals widen into `ScalarValue::Int64`. The
        // `ScalarValue` layer is the strict-mirror DataFusion shape.
        Some(ExprType::LiteralInt8(n)) => Expr::Literal(ScalarValue::Int64(i64::from(*n))),
        Some(ExprType::LiteralInt16(n)) => Expr::Literal(ScalarValue::Int64(i64::from(*n))),
        Some(ExprType::LiteralInt32(n)) => Expr::Literal(ScalarValue::Int64(i64::from(*n))),
        Some(ExprType::LiteralInt64(n)) => Expr::Literal(ScalarValue::Int64(*n)),
        Some(ExprType::LiteralUint8(n)) => Expr::Literal(ScalarValue::Int64(i64::from(*n))),
        Some(ExprType::LiteralUint16(n)) => Expr::Literal(ScalarValue::Int64(i64::from(*n))),
        Some(ExprType::LiteralUint32(n)) => Expr::Literal(ScalarValue::Int64(i64::from(*n))),
        Some(ExprType::LiteralUint64(n)) => Expr::Literal(ScalarValue::Int64(*n as i64)),
        Some(ExprType::LiteralF32(n)) => Expr::Literal(ScalarValue::Float32(*n)),
        Some(ExprType::LiteralF64(n)) => Expr::Literal(ScalarValue::Float64(*n)),
        // The wire format already encodes days since the Unix epoch — same
        // representation `ScalarValue::Date32` uses — so no conversion
        // through `chrono::NaiveDate` is needed here.
        Some(ExprType::LiteralDate(days)) => Expr::Literal(ScalarValue::Date32(*days)),
        Some(ExprType::Alias(a)) => {
            let expr = deserialize_logical_expr(a.expr.as_deref().expect("AliasNode.expr unset"));
            Expr::Alias {
                expr: Box::new(expr),
                alias: a.alias.clone(),
            }
        }
        Some(ExprType::BinaryExpr(b)) => {
            let left = Box::new(deserialize_logical_expr(
                b.l.as_deref().expect("BinaryExprNode.l unset"),
            ));
            let right = Box::new(deserialize_logical_expr(
                b.r.as_deref().expect("BinaryExprNode.r unset"),
            ));
            // Single binary arm; map the wire-format
            // op string to the unified [`Operator`] enum.
            let op = match b.op.as_str() {
                "eq" => Operator::Eq,
                "neq" => Operator::NotEq,
                "lt" => Operator::Lt,
                "lteq" => Operator::LtEq,
                "gt" => Operator::Gt,
                "gteq" => Operator::GtEq,
                "and" => Operator::And,
                "or" => Operator::Or,
                "add" => Operator::Plus,
                "subtract" => Operator::Minus,
                "multiply" => Operator::Multiply,
                "divide" => Operator::Divide,
                "modulus" => Operator::Modulo,
                other => panic!("Unsupported binary operator: '{other}'"),
            };
            Expr::BinaryExpr { left, op, right }
        }
        Some(ExprType::AggregateExpr(a)) => {
            // The wire format's
            // `protobuf::AggregateFunction::CountDistinct` enum value
            // now lowers to `(Count, distinct=true)` on the in-memory
            // side (DISTINCT is a struct field, not a separate variant —
            // mirror of DataFusion's `AggregateFunctionParams::distinct`).
            let inner =
                deserialize_logical_expr(a.expr.as_deref().expect("AggregateExprNode.expr unset"));
            let fn_kind =
                protobuf::AggregateFunction::try_from(a.aggr_function).unwrap_or_else(|_| {
                    panic!("Unknown AggregateFunction enum value: {}", a.aggr_function)
                });
            let (kind, distinct) = match fn_kind {
                protobuf::AggregateFunction::Min => (AggregateFunctionKind::Min, false),
                protobuf::AggregateFunction::Max => (AggregateFunctionKind::Max, false),
                protobuf::AggregateFunction::Sum => (AggregateFunctionKind::Sum, false),
                protobuf::AggregateFunction::Avg => (AggregateFunctionKind::Avg, false),
                protobuf::AggregateFunction::Count => (AggregateFunctionKind::Count, false),
                protobuf::AggregateFunction::CountDistinct => (AggregateFunctionKind::Count, true),
            };
            Expr::AggregateFunction(AggregateFunction::new(
                kind,
                vec![inner],
                distinct,
                None,
                Vec::new(),
                None,
            ))
        }
        // The underlying logical-plan variants don't exist yet.
        Some(ExprType::IsNullExpr(_)) => {
            todo!("IsNull is not yet implemented in logical-plan")
        }
        Some(ExprType::IsNotNullExpr(_)) => {
            todo!("IsNotNull is not yet implemented in logical-plan")
        }
        Some(ExprType::NotExpr(_)) => {
            todo!("Not is not yet implemented as a logical expression")
        }
        None => panic!("Found null expr enum when deserialising protobuf logical expression"),
    }
}

/// `protobuf::Schema` → `fdapquery_datatypes::Schema`.
pub fn deserialize_schema(schema: &protobuf::Schema) -> Schema {
    let fields: Vec<Field> = schema.columns.iter().map(deserialize_field).collect();
    Schema::new(fields)
}

/// `protobuf::Field` → `fdapquery_datatypes::Field`.
pub fn deserialize_field(field: &protobuf::Field) -> Field {
    Field::new(&field.name, from_proto_arrow_type(field.arrow_type), true)
}

/// `protobuf::Action` → `Box<dyn Action>` (wrapping a `QueryAction`).
pub fn deserialize_action(action: &protobuf::Action) -> Box<dyn Action> {
    if let Some(query) = action.query.as_ref() {
        Box::new(QueryAction::new(deserialize_logical_plan(query)))
    } else {
        unimplemented!("Action is not implemented: {action:?}")
    }
}

/// `protobuf::ArrowType` (i32) → `arrow_schema::DataType`. Panics on enum values
/// the engine does not yet support.
fn from_proto_arrow_type(arrow_type: i32) -> DataType {
    let at = protobuf::ArrowType::try_from(arrow_type).unwrap_or_else(|_| {
        panic!("Cannot deserialize Arrow data type enum from protobuf: {arrow_type}")
    });
    match at {
        protobuf::ArrowType::Bool => arrow_schema::DataType::Boolean,
        protobuf::ArrowType::Int8 => arrow_schema::DataType::Int8,
        protobuf::ArrowType::Int16 => arrow_schema::DataType::Int16,
        protobuf::ArrowType::Int32 => arrow_schema::DataType::Int32,
        protobuf::ArrowType::Int64 => arrow_schema::DataType::Int64,
        protobuf::ArrowType::Uint8 => arrow_schema::DataType::UInt8,
        protobuf::ArrowType::Uint16 => arrow_schema::DataType::UInt16,
        protobuf::ArrowType::Uint32 => arrow_schema::DataType::UInt32,
        protobuf::ArrowType::Uint64 => arrow_schema::DataType::UInt64,
        protobuf::ArrowType::Float => arrow_schema::DataType::Float32,
        protobuf::ArrowType::Double => arrow_schema::DataType::Float64,
        protobuf::ArrowType::Utf8 => arrow_schema::DataType::Utf8,
        protobuf::ArrowType::Date32 => arrow_schema::DataType::Date32,
        other => panic!("Cannot deserialize Arrow type from protobuf: {other:?}"),
    }
}
