//! `pb::PhysicalPlanNode` → `Arc<dyn PhysicalPlan>`,
//! `pb::PhysicalExprNode` → `Arc<dyn Expression>`, and the inverses of every
//! conversion in `physical_plan_serializer.rs`.
//!
//! ## Shape — free functions, no `Deserializer` struct
//! Same DataFusion-aligned shape as the serializer side: no stateful struct,
//! free `deserialize_X` functions. `Schema` / `Field` deserialization is
//! shared with the logical-plan deserializer (`crate::deserialize_schema` /
//! `crate::deserialize_field`) — both produce the same domain types from the
//! same `pb::*` messages.
//!
//! ## Notes
//! - **No downcast plumbing needed.** The deserializer builds concrete types
//!   from proto messages and returns boxed trait objects — no
//!   `as_X`-style branching on existing values.
//! - **Schema is required for `ScanExecNode`.** The serializer always emits
//!   `schema` for scans; the deserializer unwraps it accordingly. For CSV
//!   scans the materialised `Schema` is passed to `CsvDataSource::new(...)`
//!   so the source uses the wire schema rather than re-inferring from the
//!   file.
//! - **`ShuffleLocation` is `physical_plan::ShuffleLocation`** (the 6-field
//!   one matching the proto), not the 4-field `datatypes::ShuffleLocation`.
//! - **Orphan rule note.** The deserializer's leaf conversions (e.g.,
//!   `pb::ShuffleLocation` → `physical_plan::ShuffleLocation`) cannot be
//!   written as `impl From<&pb::T> for T` because the target types live in
//!   foreign crates and the orphan rule rejects the impl. They stay as free
//!   `deserialize_X` functions. The asymmetry with the serializer side
//!   (where the target `pb::*` types are local and `impl From` works) is a
//!   direct consequence of the orphan rule, not a stylistic choice.

use crate::pb;
use datasource::{CsvDataSource, DataSource, ParquetDataSource};
use datatypes::arrow_types;
use physical_plan::{
    AddExpression, AggregateExpression, AggregateMode, AndExpression, AvgExpression,
    CastExpression, ColumnExpression, CountExpression, DivideExpression, EqExpression, Expression,
    GtEqExpression, GtExpression, HashAggregateExec, LiteralDateExpression,
    LiteralDoubleExpression, LiteralLongExpression, LiteralStringExpression, LtEqExpression,
    LtExpression, MaxExpression, MinExpression, MultiplyExpression, NeqExpression, OrExpression,
    PhysicalPlan, ProjectionExec, ScanExec, SelectionExec, ShuffleLocation, ShuffleReaderExec,
    ShuffleWriterExec, SubtractExpression, SumExpression, Task,
};

use arrow_schema::DataType;
use std::sync::Arc;

/// `pb::PhysicalPlanNode` → `Arc<dyn PhysicalPlan>`.
pub fn deserialize_physical_plan(node: &pb::PhysicalPlanNode) -> Arc<dyn PhysicalPlan> {
    use pb::physical_plan_node::PlanType;
    match node.plan_type.as_ref() {
        Some(PlanType::Scan(scan)) => {
            let schema =
                crate::deserialize_schema(scan.schema.as_ref().expect("ScanExecNode.schema unset"));
            let ds: Arc<dyn DataSource> = match scan.file_format.as_str() {
                "csv" => Arc::new(CsvDataSource::new(&scan.path, Some(schema), true, 1024)),
                "parquet" => Arc::new(ParquetDataSource::new(&scan.path)),
                other => panic!("Unsupported file format: {other:?}"),
            };
            Arc::new(ScanExec::new(ds, scan.projection.clone()))
        }
        Some(PlanType::Projection(proj)) => {
            let input = deserialize_physical_plan(
                proj.input
                    .as_deref()
                    .expect("ProjectionExecNode.input unset"),
            );
            let schema = crate::deserialize_schema(
                proj.schema
                    .as_ref()
                    .expect("ProjectionExecNode.schema unset"),
            );
            let expr = proj.expr.iter().map(deserialize_physical_expr).collect();
            Arc::new(ProjectionExec::new(input, schema, expr))
        }
        Some(PlanType::Selection(sel)) => {
            let input = deserialize_physical_plan(
                sel.input.as_deref().expect("SelectionExecNode.input unset"),
            );
            let expr =
                deserialize_physical_expr(sel.expr.as_ref().expect("SelectionExecNode.expr unset"));
            Arc::new(SelectionExec::new(input, expr))
        }
        Some(PlanType::HashAggregate(agg)) => {
            let input = deserialize_physical_plan(
                agg.input
                    .as_deref()
                    .expect("HashAggregateExecNode.input unset"),
            );
            let group_expr = agg
                .group_expr
                .iter()
                .map(deserialize_physical_expr)
                .collect();
            let aggregate_expr = agg
                .aggregate_expr
                .iter()
                .map(deserialize_physical_aggr_expr)
                .collect();
            let schema = crate::deserialize_schema(
                agg.schema
                    .as_ref()
                    .expect("HashAggregateExecNode.schema unset"),
            );
            let mode = aggregate_mode_from_proto(agg.mode);
            Arc::new(HashAggregateExec::new_with_mode(
                input,
                group_expr,
                aggregate_expr,
                schema,
                mode,
            ))
        }
        Some(PlanType::ShuffleWriter(sw)) => {
            let input = deserialize_physical_plan(
                sw.input
                    .as_deref()
                    .expect("ShuffleWriterExecNode.input unset"),
            );
            let partition_expr = sw
                .partition_expr
                .iter()
                .map(deserialize_physical_expr)
                .collect();
            Arc::new(ShuffleWriterExec::new(
                input,
                partition_expr,
                sw.job_uuid.clone(),
                sw.stage_id,
                sw.partition_count,
            ))
        }
        Some(PlanType::ShuffleReader(sr)) => {
            let schema = crate::deserialize_schema(
                sr.schema
                    .as_ref()
                    .expect("ShuffleReaderExecNode.schema unset"),
            );
            let locations = sr
                .shuffle_locations
                .iter()
                .map(deserialize_shuffle_location)
                .collect();
            Arc::new(ShuffleReaderExec::new(schema, locations))
        }
        None => panic!("Failed to parse physical plan node: plan_type unset"),
    }
}

/// `pb::PhysicalExprNode` → `Arc<dyn Expression>`.
pub fn deserialize_physical_expr(node: &pb::PhysicalExprNode) -> Arc<dyn Expression> {
    use pb::physical_expr_node::ExprType;
    match node.expr_type.as_ref() {
        Some(ExprType::Column(i)) => Arc::new(ColumnExpression::new(*i as usize)),
        Some(ExprType::LiteralString(s)) => Arc::new(LiteralStringExpression::new(s.clone())),
        Some(ExprType::LiteralLong(n)) => Arc::new(LiteralLongExpression::new(*n)),
        Some(ExprType::LiteralDouble(n)) => Arc::new(LiteralDoubleExpression::new(*n)),
        Some(ExprType::LiteralDate(days)) => Arc::new(LiteralDateExpression::new(*days)),
        Some(ExprType::BinaryExpr(b)) => {
            let l =
                deserialize_physical_expr(b.l.as_deref().expect("PhysicalBinaryExprNode.l unset"));
            let r =
                deserialize_physical_expr(b.r.as_deref().expect("PhysicalBinaryExprNode.r unset"));
            match b.op.as_str() {
                "eq" => Arc::new(EqExpression::new(l, r)),
                "neq" => Arc::new(NeqExpression::new(l, r)),
                "lt" => Arc::new(LtExpression::new(l, r)),
                "lteq" => Arc::new(LtEqExpression::new(l, r)),
                "gt" => Arc::new(GtExpression::new(l, r)),
                "gteq" => Arc::new(GtEqExpression::new(l, r)),
                "and" => Arc::new(AndExpression::new(l, r)),
                "or" => Arc::new(OrExpression::new(l, r)),
                "add" => Arc::new(AddExpression::new(l, r)),
                "subtract" => Arc::new(SubtractExpression::new(l, r)),
                "multiply" => Arc::new(MultiplyExpression::new(l, r)),
                "divide" => Arc::new(DivideExpression::new(l, r)),
                other => panic!("Unsupported binary operator: '{other}'"),
            }
        }
        Some(ExprType::CastExpr(c)) => {
            let expr = deserialize_physical_expr(
                c.expr.as_deref().expect("PhysicalCastExprNode.expr unset"),
            );
            let dt = from_proto_arrow_type(c.arrow_type);
            Arc::new(CastExpression::new(expr, dt))
        }
        None => panic!("Physical expression type not set in protobuf"),
    }
}

/// `pb::PhysicalAggregateExprNode` → `Arc<dyn AggregateExpression>`.
pub fn deserialize_physical_aggr_expr(
    node: &pb::PhysicalAggregateExprNode,
) -> Arc<dyn AggregateExpression> {
    let input = deserialize_physical_expr(
        node.input_expr
            .as_ref()
            .expect("PhysicalAggregateExprNode.input_expr unset"),
    );
    let fn_kind = pb::AggregateFunction::try_from(node.aggr_function).unwrap_or_else(|_| {
        panic!(
            "Unknown AggregateFunction enum value: {}",
            node.aggr_function
        )
    });
    match fn_kind {
        pb::AggregateFunction::Sum => Arc::new(SumExpression::new(input)),
        pb::AggregateFunction::Min => Arc::new(MinExpression::new(input)),
        pb::AggregateFunction::Max => Arc::new(MaxExpression::new(input)),
        pb::AggregateFunction::Avg => Arc::new(AvgExpression::new(input)),
        pb::AggregateFunction::Count => Arc::new(CountExpression::new(input)),
        other => panic!("Unsupported aggregate function: {other:?}"),
    }
}

/// `pb::ShuffleLocation` → `physical_plan::ShuffleLocation`.
///
/// Stays a free function (rather than `impl From<&pb::ShuffleLocation> for
/// physical_plan::ShuffleLocation`) because the target type is in a foreign
/// crate and the orphan rule rejects the impl. See the module doc.
pub fn deserialize_shuffle_location(loc: &pb::ShuffleLocation) -> ShuffleLocation {
    ShuffleLocation::new(
        &loc.job_uuid,
        loc.stage_id,
        loc.partition_id,
        &loc.executor_id,
        &loc.executor_host,
        loc.executor_port,
    )
}

/// `pb::TaskInfo` → `Task`.
///
/// `Task::plan` is `Arc<dyn PhysicalPlan>`, matching what
/// [`deserialize_physical_plan`] now returns — no conversion needed.
pub fn deserialize_task(task: &pb::TaskInfo) -> Task {
    Task::new(
        &task.job_uuid,
        task.stage_id,
        task.task_id,
        task.partition_id,
        deserialize_physical_plan(task.plan.as_ref().expect("TaskInfo.plan unset")),
    )
}

// ---------------------------------------------------------------------------
// Private helpers.
// ---------------------------------------------------------------------------

/// `pb::AggregateMode` (i32) → our `AggregateMode`. Inverse of
/// `physical_plan_serializer::aggregate_mode_to_proto`. Defaults to `Complete`
/// for any unknown enum value.
fn aggregate_mode_from_proto(mode: i32) -> AggregateMode {
    match pb::AggregateMode::try_from(mode) {
        Ok(pb::AggregateMode::Complete) => AggregateMode::Complete,
        Ok(pb::AggregateMode::Partial) => AggregateMode::Partial,
        Ok(pb::AggregateMode::Final) => AggregateMode::Final,
        Err(_) => AggregateMode::Complete,
    }
}

/// `pb::ArrowType` (i32) → `arrow_schema::DataType`. Same shape as
/// `protobuf_deserializer::from_proto_arrow_type`; duplicated here so the
/// `CastExpression` arm doesn't need to reach across files. The two
/// definitions are deliberately identical.
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
