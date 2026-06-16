//! `pb::LogicalPlanNode` → `LogicalPlan`, `pb::LogicalExprNode` → `LogicalExpr`,
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
//! - Each `pb::LogicalPlanNode` variant is dispatched via a chain of
//!   `if let Some(_) = &node.<field>` arms. prost emits each message-typed
//!   plan field as `Option<T>`, so the "is this variant set?" check is
//!   `node.x.is_some()`.
//! - `LiteralInt8/16/32/64` and `LiteralUint8/16/32/64` all collapse into
//!   `LogicalExpr::LiteralLong(i64)` — that's what `lit_long` is for.
//! - The `literal_date` arm reverses the days-since-epoch encoding back
//!   into a `chrono::NaiveDate`.
//! - `IsNull` / `IsNotNull` / `Not` arms are unimplemented; their logical-plan
//!   variants don't exist yet, so the arms `panic!` with a clear message
//!   rather than guess at semantics.

use crate::pb;
use datasource::{CsvDataSource, ParquetDataSource};
use datatypes::{Field, Schema, arrow_types};
use logical_plan::{
    Aggregate, AggregateExpr, Limit, LogicalExpr, LogicalPlan, Projection, Scan, Selection,
};
// JoinNode is not deserialised here. If/when that's added, re-import `JoinType`.
use physical_plan::{Action, QueryAction};
use std::sync::Arc;

use arrow_schema::DataType;

/// `pb::LogicalPlanNode` → `LogicalPlan`.
pub fn deserialize_logical_plan(node: &pb::LogicalPlanNode) -> LogicalPlan {
    if let Some(csv) = &node.csv_scan {
        // The schema field is set by the serializer only for Parquet — for
        // CSV the proto schema is left unset, so we pass `None` and let
        // `CsvDataSource` re-infer from the file.
        let ds = CsvDataSource::new(&csv.path, None, csv.has_header, 1024);
        LogicalPlan::Scan(Scan::new(
            &csv.path,
            Arc::new(ds),
            csv.projection
                .as_ref()
                .map(|p| p.columns.clone())
                .unwrap_or_default(),
        ))
    } else if let Some(parquet) = &node.parquet_scan {
        let ds = ParquetDataSource::new(&parquet.path);
        LogicalPlan::Scan(Scan::new(
            &parquet.path,
            Arc::new(ds),
            parquet
                .projection
                .as_ref()
                .map(|p| p.columns.clone())
                .unwrap_or_default(),
        ))
    } else if let Some(sel) = &node.selection {
        let input = deserialize_plan_input(node);
        let expr = deserialize_logical_expr(sel.expr.as_ref().expect("SelectionNode.expr unset"));
        LogicalPlan::Selection(Selection::new(input, expr))
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
        // AggregateExpr variant; deserialise then unwrap the
        // `LogicalExpr::AggregateExpr(Box<AggregateExpr>)` wrapper.
        let aggregate_expr = agg
            .aggr_expr
            .iter()
            .map(|e| match deserialize_logical_expr(e) {
                LogicalExpr::AggregateExpr(ae) => *ae,
                other => panic!(
                    "AggregateNode.aggr_expr did not deserialise to an \
                     AggregateExpr: {other:?}"
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
/// (Projection / Selection / Limit / Aggregate).
fn deserialize_plan_input(node: &pb::LogicalPlanNode) -> LogicalPlan {
    let inner = node
        .input
        .as_deref()
        .expect("LogicalPlanNode.input unset on a non-leaf plan");
    deserialize_logical_plan(inner)
}

/// `pb::LogicalExprNode` → `LogicalExpr`.
pub fn deserialize_logical_expr(node: &pb::LogicalExprNode) -> LogicalExpr {
    use pb::logical_expr_node::ExprType;
    match node.expr_type.as_ref() {
        Some(ExprType::ColumnName(name)) => LogicalExpr::Column(name.clone()),
        Some(ExprType::LiteralString(s)) => LogicalExpr::LiteralString(s.clone()),
        // All integer literals collapse into `LiteralLong(i64)`.
        Some(ExprType::LiteralInt8(n)) => LogicalExpr::LiteralLong(*n as i64),
        Some(ExprType::LiteralInt16(n)) => LogicalExpr::LiteralLong(*n as i64),
        Some(ExprType::LiteralInt32(n)) => LogicalExpr::LiteralLong(*n as i64),
        Some(ExprType::LiteralInt64(n)) => LogicalExpr::LiteralLong(*n),
        Some(ExprType::LiteralUint8(n)) => LogicalExpr::LiteralLong(*n as i64),
        Some(ExprType::LiteralUint16(n)) => LogicalExpr::LiteralLong(*n as i64),
        Some(ExprType::LiteralUint32(n)) => LogicalExpr::LiteralLong(*n as i64),
        Some(ExprType::LiteralUint64(n)) => LogicalExpr::LiteralLong(*n as i64),
        Some(ExprType::LiteralF32(n)) => LogicalExpr::LiteralFloat(*n),
        Some(ExprType::LiteralF64(n)) => LogicalExpr::LiteralDouble(*n),
        // Added by the Rust port; reverses the days-since-epoch encoding.
        Some(ExprType::LiteralDate(days)) => LogicalExpr::LiteralDate(naive_date_from_days(*days)),
        Some(ExprType::Alias(a)) => {
            let expr = deserialize_logical_expr(a.expr.as_deref().expect("AliasNode.expr unset"));
            LogicalExpr::Alias {
                expr: Box::new(expr),
                alias: a.alias.clone(),
            }
        }
        Some(ExprType::BinaryExpr(b)) => {
            let l = Box::new(deserialize_logical_expr(
                b.l.as_deref().expect("BinaryExprNode.l unset"),
            ));
            let r = Box::new(deserialize_logical_expr(
                b.r.as_deref().expect("BinaryExprNode.r unset"),
            ));
            match b.op.as_str() {
                "eq" => LogicalExpr::Eq { l, r },
                "neq" => LogicalExpr::Neq { l, r },
                "lt" => LogicalExpr::Lt { l, r },
                "lteq" => LogicalExpr::LtEq { l, r },
                "gt" => LogicalExpr::Gt { l, r },
                "gteq" => LogicalExpr::GtEq { l, r },
                "and" => LogicalExpr::And { l, r },
                "or" => LogicalExpr::Or { l, r },
                "add" => LogicalExpr::Add { l, r },
                "subtract" => LogicalExpr::Subtract { l, r },
                "multiply" => LogicalExpr::Multiply { l, r },
                "divide" => LogicalExpr::Divide { l, r },
                other => panic!("Unsupported binary operator: '{other}'"),
            }
        }
        Some(ExprType::AggregateExpr(a)) => {
            let inner =
                deserialize_logical_expr(a.expr.as_deref().expect("AggregateExprNode.expr unset"));
            let fn_kind = pb::AggregateFunction::try_from(a.aggr_function).unwrap_or_else(|_| {
                panic!("Unknown AggregateFunction enum value: {}", a.aggr_function)
            });
            let agg = match fn_kind {
                pb::AggregateFunction::Min => AggregateExpr::Min(inner),
                pb::AggregateFunction::Max => AggregateExpr::Max(inner),
                pb::AggregateFunction::Sum => AggregateExpr::Sum(inner),
                pb::AggregateFunction::Avg => AggregateExpr::Avg(inner),
                pb::AggregateFunction::Count => AggregateExpr::Count(inner),
                pb::AggregateFunction::CountDistinct => AggregateExpr::CountDistinct(inner),
            };
            LogicalExpr::AggregateExpr(Box::new(agg))
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

/// `pb::Schema` → `datatypes::Schema`.
pub fn deserialize_schema(schema: &pb::Schema) -> Schema {
    let fields = schema.columns.iter().map(deserialize_field).collect();
    Schema::new(fields)
}

/// `pb::Field` → `datatypes::Field`.
pub fn deserialize_field(field: &pb::Field) -> Field {
    Field::new(&field.name, from_proto_arrow_type(field.arrow_type))
}

/// `pb::Action` → `Box<dyn Action>` (wrapping a `QueryAction`).
pub fn deserialize_action(action: &pb::Action) -> Box<dyn Action> {
    if let Some(query) = action.query.as_ref() {
        Box::new(QueryAction::new(deserialize_logical_plan(query)))
    } else {
        unimplemented!("Action is not implemented: {action:?}")
    }
}

/// `pb::ArrowType` (i32) → `arrow_schema::DataType`. Panics on enum values
/// the engine does not yet support.
fn from_proto_arrow_type(arrow_type: i32) -> DataType {
    let at = pb::ArrowType::try_from(arrow_type).unwrap_or_else(|_| {
        panic!("Cannot deserialize Arrow data type enum from protobuf: {arrow_type}")
    });
    match at {
        pb::ArrowType::Bool => arrow_types::BOOLEAN_TYPE,
        pb::ArrowType::Int8 => arrow_types::INT8_TYPE,
        pb::ArrowType::Int16 => arrow_types::INT16_TYPE,
        pb::ArrowType::Int32 => arrow_types::INT32_TYPE,
        pb::ArrowType::Int64 => arrow_types::INT64_TYPE,
        pb::ArrowType::Uint8 => arrow_types::UINT8_TYPE,
        pb::ArrowType::Uint16 => arrow_types::UINT16_TYPE,
        pb::ArrowType::Uint32 => arrow_types::UINT32_TYPE,
        pb::ArrowType::Uint64 => arrow_types::UINT64_TYPE,
        pb::ArrowType::Float => arrow_types::FLOAT_TYPE,
        pb::ArrowType::Double => arrow_types::DOUBLE_TYPE,
        pb::ArrowType::Utf8 => arrow_types::STRING_TYPE,
        pb::ArrowType::Date32 => arrow_types::DATE_DAY_TYPE,
        other => panic!("Cannot deserialize Arrow type from protobuf: {other:?}"),
    }
}

/// Days-since-Unix-epoch → `chrono::NaiveDate`. Inverse of the helper in
/// `protobuf_serializer.rs`; same shape as `query_planner::days_since_unix_epoch`.
fn naive_date_from_days(days: i32) -> chrono::NaiveDate {
    let epoch = chrono::NaiveDate::from_ymd_opt(1970, 1, 1).expect("1970-01-01 is a valid date");
    epoch + chrono::Duration::days(days as i64)
}
