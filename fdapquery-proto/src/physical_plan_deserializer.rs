//! `protobuf::PhysicalPlanNode` → `Arc<dyn PhysicalPlan>`,
//! `protobuf::PhysicalExprNode` → `Arc<dyn PhysicalExpr>`, and the inverses of every
//! conversion in `physical_plan_serializer.rs`.
//!
//! ## Shape — free functions, no `Deserializer` struct
//! Same DataFusion-aligned shape as the serializer side: no stateful struct,
//! free `deserialize_X` functions. `Schema` / `Field` deserialization is
//! shared with the logical-plan deserializer (`crate::deserialize_schema` /
//! `crate::deserialize_field`) — both produce the same domain types from the
//! same `protobuf::*` messages.
//!
//! ## Notes
//! - **No downcast plumbing needed.** The deserializer builds concrete types
//!   from proto messages and returns boxed trait objects — no
//!   `as_X`-style branching on existing values.
//! - **Schema is required for `DataSourceExecNode`.** The serializer always emits
//!   `schema` for scans; the deserializer unwraps it accordingly. For CSV
//!   scans the materialised `Schema` is passed to `CsvDataSource::new(...)`
//!   so the source uses the wire schema rather than re-inferring from the
//!   file.
//! - **`ShuffleLocation` is `fdapquery_physical_plan::ShuffleLocation`** (the 6-field
//!   one matching the proto), not the 4-field `fdapquery_datatypes::ShuffleLocation`.
//! - **Orphan rule note.** The deserializer's leaf conversions (e.g.,
//!   `protobuf::ShuffleLocation` → `fdapquery_physical_plan::ShuffleLocation`) cannot be
//!   written as `impl From<&protobuf::T> for T` because the target types live in
//!   foreign crates and the orphan rule rejects the impl. They stay as free
//!   `deserialize_X` functions. The asymmetry with the serializer side
//!   (where the target `protobuf::*` types are local and `impl From` works) is a
//!   direct consequence of the orphan rule, not a stylistic choice.

use crate::protobuf;
use fdapquery_catalog::{CsvDataSourceConfig, ParquetDataSourceConfig};
use fdapquery_common::ScalarValue;
use fdapquery_datasource::DataSourceExec;
use fdapquery_expr::Operator;
// The 12 sibling binary types collapsed into the
// unified [`BinaryExpr`] parameterised by [`Operator`].
use fdapquery_physical_plan::{
    AggregateExec, AggregateExpr, AggregateMode, AvgExpr, BinaryExpr, CastExpr, Column, CountExpr,
    ExecutionPlan, FilterExec, Literal, MaxExpr, MinExpr, PhysicalExpr, PhysicalGroupBy,
    ProjectionExec, ShuffleLocation, ShuffleReaderExec, ShuffleWriterExec, SumExpr, Task,
};

use arrow_schema::DataType;
use std::sync::Arc;

/// `protobuf::PhysicalPlanNode` → `Arc<dyn ExecutionPlan>`.
pub fn deserialize_physical_plan(node: &protobuf::PhysicalPlanNode) -> Arc<dyn ExecutionPlan> {
    use protobuf::physical_plan_node::PlanType;
    match node.plan_type.as_ref() {
        // Wire variant name `Scan` is generated from the .proto field
        // `DataSourceExecNode scan = 1;`. The wire tag stayed `Scan`
        // across the Rust-side renames (`Scan` → `TableScan`,
        // `ScanExec` → `DataSourceExec`); the message name became
        // `DataSourceExecNode` alongside the proto renames
        // `SelectionNode → FilterNode` and
        // `SelectionExecNode → FilterExecNode`.
        Some(PlanType::Scan(scan)) => {
            let full_schema = crate::deserialize_schema(
                scan.schema
                    .as_ref()
                    .expect("DataSourceExecNode.schema unset"),
            );
            // The wire format carries column NAMES; resolve them to
            // indices against the full source schema. An empty list
            // means "no projection" (`None`).
            let projection_indices: Option<Vec<usize>> = if scan.projection.is_empty() {
                None
            } else {
                let indices: Vec<usize> = scan
                    .projection
                    .iter()
                    .map(|name| {
                        full_schema
                            .fields()
                            .iter()
                            .position(|f| f.name() == name)
                            .unwrap_or_else(|| {
                                panic!(
                                    "DataSourceExecNode: projection column '{name}' \
                                     not in source schema"
                                )
                            })
                    })
                    .collect();
                Some(indices)
            };
            // Build the per-format `*Config: DataSource` directly
            // (sync) and wrap in `DataSourceExec`. The deserializer is
            // sync; the async `TableProvider::scan` planning surface
            // is not used here because we already have the full schema
            // on the wire and don't need to re-infer.
            match scan.file_format.as_str() {
                "csv" => {
                    let config = CsvDataSourceConfig::new_for_proto(
                        scan.path.clone(),
                        full_schema,
                        projection_indices,
                        true,
                        1024,
                        b',',
                    )
                    .expect("DataSourceExecNode (csv): invalid projection over source schema");
                    Arc::new(DataSourceExec::new(Arc::new(config)))
                }
                "parquet" => {
                    let config = ParquetDataSourceConfig::new_for_proto(
                        scan.path.clone(),
                        full_schema,
                        projection_indices,
                    )
                    .expect("DataSourceExecNode (parquet): invalid projection over source schema");
                    Arc::new(DataSourceExec::new(Arc::new(config)))
                }
                other => panic!("Unsupported file format: {other:?}"),
            }
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
            Arc::new(ProjectionExec::new(expr, input, schema))
        }
        // Wire variant name `Selection` is generated from the .proto
        // field `FilterExecNode selection = 3;`. Stable across the
        // Rust-side `Selection` → `Filter` rename.
        Some(PlanType::Selection(sel)) => {
            let input = deserialize_physical_plan(
                sel.input.as_deref().expect("FilterExecNode.input unset"),
            );
            let expr =
                deserialize_physical_expr(sel.expr.as_ref().expect("FilterExecNode.expr unset"));
            Arc::new(FilterExec::new(expr, input))
        }
        Some(PlanType::HashAggregate(agg)) => {
            // Strict-mirror refactor. The wire format still
            // carries a flat `group_expr` list; we wrap it into the
            // DataFusion-style `PhysicalGroupBy::new_single` shape on the way
            // in. The simple `GROUP BY a, b, …` case (the only case
            // fdapquery's planner emits today) uses each expression's own
            // `to_string()` as its alias, which collapses out in display.
            let input = deserialize_physical_plan(
                agg.input
                    .as_deref()
                    .expect("HashAggregateExecNode.input unset"),
            );
            let group_exprs: Vec<Arc<dyn PhysicalExpr>> = agg
                .group_expr
                .iter()
                .map(deserialize_physical_expr)
                .collect();
            let aggregate_expr: Vec<Arc<dyn AggregateExpr>> = agg
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
            let group_pairs: Vec<(Arc<dyn PhysicalExpr>, String)> = group_exprs
                .into_iter()
                .map(|e| {
                    let alias = e.to_string();
                    (e, alias)
                })
                .collect();
            let group_by = PhysicalGroupBy::new_single(group_pairs);
            let input_schema = input.schema();
            let n_aggrs = aggregate_expr.len();
            Arc::new(
                AggregateExec::try_new(
                    mode,
                    group_by,
                    aggregate_expr,
                    vec![None; n_aggrs],
                    input,
                    input_schema,
                    schema,
                )
                .expect("AggregateExec::try_new failed during deserialize"),
            )
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

/// `protobuf::PhysicalExprNode` → `Arc<dyn PhysicalExpr>`.
pub fn deserialize_physical_expr(node: &protobuf::PhysicalExprNode) -> Arc<dyn PhysicalExpr> {
    use protobuf::physical_expr_node::ExprType;
    match node.expr_type.as_ref() {
        // `Column` now round-trips through a
        // `PhysicalColumn { name, index }` sub-message that carries the
        // source-schema column name alongside the index. Closes the
        // empty-string-name gap noted: mirrors
        // DataFusion's `datafusion.proto: PhysicalColumn` exactly.
        Some(ExprType::Column(c)) => Arc::new(Column::new(&c.name, c.index as usize)),
        // The five sibling literal types collapsed into a
        // single `Literal { value: ScalarValue }`. Each wire variant
        // materializes the corresponding `ScalarValue` variant. The wire
        // format keeps its five distinct variants (renaming to a single
        // wire variant is #110/#111 work).
        Some(ExprType::LiteralString(s)) => Arc::new(Literal::new(ScalarValue::Utf8(s.clone()))),
        Some(ExprType::LiteralLong(n)) => Arc::new(Literal::new(ScalarValue::Int64(*n))),
        Some(ExprType::LiteralDouble(n)) => Arc::new(Literal::new(ScalarValue::Float64(*n))),
        Some(ExprType::LiteralDate(days)) => Arc::new(Literal::new(ScalarValue::Date32(*days))),
        Some(ExprType::BinaryExpr(b)) => {
            let l =
                deserialize_physical_expr(b.l.as_deref().expect("PhysicalBinaryExprNode.l unset"));
            let r =
                deserialize_physical_expr(b.r.as_deref().expect("PhysicalBinaryExprNode.r unset"));
            // Map the wire-format op string to the
            // unified [`Operator`] enum and build a single
            // [`BinaryExpr`].
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
            Arc::new(BinaryExpr::new(l, op, r))
        }
        Some(ExprType::CastExpr(c)) => {
            let expr = deserialize_physical_expr(
                c.expr.as_deref().expect("PhysicalCastExprNode.expr unset"),
            );
            let dt = from_proto_arrow_type(c.arrow_type);
            Arc::new(CastExpr::new(expr, dt))
        }
        None => panic!("Physical expression type not set in protobuf"),
    }
}

/// `protobuf::PhysicalAggregateExprNode` → `Arc<dyn AggregateExpr>`.
pub fn deserialize_physical_aggr_expr(
    node: &protobuf::PhysicalAggregateExprNode,
) -> Arc<dyn AggregateExpr> {
    let input = deserialize_physical_expr(
        node.input_expr
            .as_ref()
            .expect("PhysicalAggregateExprNode.input_expr unset"),
    );
    let fn_kind = protobuf::AggregateFunction::try_from(node.aggr_function).unwrap_or_else(|_| {
        panic!(
            "Unknown AggregateFunction enum value: {}",
            node.aggr_function
        )
    });
    // The wildcard arm intentionally catches any future variant
    // (`AggregateFunction::Unknown` or new ones added to the proto)
    // and panics loudly so the wire-format extension is not silently
    // dropped on the read side.
    #[allow(clippy::match_wildcard_for_single_variants)]
    match fn_kind {
        protobuf::AggregateFunction::Sum => Arc::new(SumExpr::new(input)),
        protobuf::AggregateFunction::Min => Arc::new(MinExpr::new(input)),
        protobuf::AggregateFunction::Max => Arc::new(MaxExpr::new(input)),
        protobuf::AggregateFunction::Avg => Arc::new(AvgExpr::new(input)),
        protobuf::AggregateFunction::Count => Arc::new(CountExpr::new(input)),
        other => panic!("Unsupported aggregate function: {other:?}"),
    }
}

/// `protobuf::ShuffleLocation` → `fdapquery_physical_plan::ShuffleLocation`.
///
/// Stays a free function (rather than `impl From<&protobuf::ShuffleLocation> for
/// fdapquery_physical_plan::ShuffleLocation`) because the target type is in a foreign
/// crate and the orphan rule rejects the impl. See the module doc.
pub fn deserialize_shuffle_location(loc: &protobuf::ShuffleLocation) -> ShuffleLocation {
    ShuffleLocation::new(
        &loc.job_uuid,
        loc.stage_id,
        loc.partition_id,
        &loc.executor_id,
        &loc.executor_host,
        loc.executor_port,
    )
}

/// `protobuf::TaskInfo` → `Task`.
///
/// `Task::plan` is `Arc<dyn ExecutionPlan>`, matching what
/// [`deserialize_physical_plan`] now returns — no conversion needed.
pub fn deserialize_task(task: &protobuf::TaskInfo) -> Task {
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

/// `protobuf::AggregateMode` (i32) → our `AggregateMode`. Inverse of
/// `physical_plan_serializer::aggregate_mode_to_proto`. Defaults to `Single`
/// for any unknown enum value.
///
/// The wire format pre-dates the rename to DataFusion's
/// six-variant `AggregateMode`; the `COMPLETE` wire value maps to the new
/// `Single` Rust variant. The matching proto rename is tracked by #110 / #111.
fn aggregate_mode_from_proto(mode: i32) -> AggregateMode {
    match protobuf::AggregateMode::try_from(mode) {
        Ok(protobuf::AggregateMode::Complete) | Err(_) => AggregateMode::Single,
        Ok(protobuf::AggregateMode::Partial) => AggregateMode::Partial,
        Ok(protobuf::AggregateMode::Final) => AggregateMode::Final,
    }
}

/// `protobuf::ArrowType` (i32) → `arrow_schema::DataType`. Same shape as
/// `protobuf_deserializer::from_proto_arrow_type`; duplicated here so the
/// `CastExpr` arm doesn't need to reach across files. The two
/// definitions are deliberately identical.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physical_plan_serializer::serialize_physical_expr;

    /// `Column` round-trips through a
    /// `PhysicalColumn { name, index }` sub-message, mirroring DataFusion's
    /// `datafusion.proto: PhysicalColumn` shape. This test exercises the
    /// full serialize → deserialize loop and asserts both fields survive.
    #[test]
    fn column_name_survives_proto_round_trip() {
        let original = Column::new("salary", 5);
        let pb_node = serialize_physical_expr(&original);
        let restored = deserialize_physical_expr(&pb_node);
        let restored_col = restored
            .as_any()
            .downcast_ref::<Column>()
            .expect("restored expr should be a Column");
        assert_eq!(restored_col.name(), "salary");
        assert_eq!(restored_col.index(), 5);
    }

    /// The unified `Literal { value: ScalarValue }`
    /// round-trips through each of the four wire literal variants
    /// (`LiteralLong`, `LiteralDouble`, `LiteralString`, `LiteralDate`).
    /// Display format byte-for-byte mirrors DataFusion's `Literal` /
    /// `ScalarValue` display.
    #[test]
    fn literal_round_trip_preserves_value_and_display() {
        let cases: Vec<(ScalarValue, &str)> = vec![
            (ScalarValue::Int64(42), "42"),
            (ScalarValue::Float64(1.5), "1.5"),
            (ScalarValue::Utf8("CO".into()), "CO"),
            (ScalarValue::Date32(18750), "2021-05-03"),
        ];
        for (scalar, expected_display) in cases {
            let original = Literal::new(scalar.clone());
            let pb_node = serialize_physical_expr(&original);
            let restored = deserialize_physical_expr(&pb_node);
            let restored_lit = restored
                .as_any()
                .downcast_ref::<Literal>()
                .expect("restored expr should be a Literal");
            assert_eq!(restored_lit.value(), &scalar);
            assert_eq!(format!("{restored_lit}"), expected_display);
        }
    }
}
