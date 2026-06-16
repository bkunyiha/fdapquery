//! `PhysicalPlan` → `pb::PhysicalPlanNode`, `Expression` → `pb::PhysicalExprNode`,
//! `AggregateExpression` → `pb::PhysicalAggregateExprNode`, `Schema` / `Field` →
//! their proto equivalents, plus `ShuffleLocation` and `Task` for distributed
//! task dispatch. Used by `flight-server` and `distributed` (modules 13–15).
//!
//! ## Shape — free functions + `From` impls, no `Serializer` struct
//! Same DataFusion-aligned shape as the logical-plan side. Three patterns:
//!
//! 1. **Non-trivial tree-walking conversions are free functions** that include
//!    the type being converted in the name: `serialize_physical_plan`,
//!    `serialize_physical_expr`, `serialize_physical_aggr_expr`, `serialize_task`.
//! 2. **Leaf conversions** that map a single domain type to a single proto
//!    message use standard `From` impls so call sites read as
//!    `schema.into()` / `field.into()` / `loc.into()`. These cover
//!    `Schema → pb::Schema`, `Field → pb::Field`, `ShuffleLocation →
//!    pb::ShuffleLocation`.
//! 3. **Enum mappings** that don't justify a trait impl stay as private helper
//!    functions: `data_type_to_proto`, `aggregate_mode_to_proto`.
//!
//! Equivalent of DataFusion's `datafusion/proto/src/physical_plan/to_proto.rs`,
//! down to the verb (`serialize_*`).
//!
//! ## Notes
//! - Concrete-operator dispatch uses `plan.as_any().downcast_ref::<XExec>()`
//!   (same pattern DataFusion uses for `ExecutionPlan` / `PhysicalExpr`).
//!   Two family-narrowing accessors — `Expression::as_boolean_expression`
//!   and `as_math_expression` — return `&dyn BooleanExpression` /
//!   `&dyn MathExpression` so the serializer can read
//!   `left()/right()/op_name()` uniformly across all 8 boolean and 5 math
//!   operators without per-operator dispatch. Pure leaf dispatches (column,
//!   literals, cast, the five aggregates, CSV vs Parquet data sources) all
//!   go through `as_any`.
//! - Boolean and math binary operators share a single arm because both
//!   serialise to `pb::PhysicalBinaryExprNode { l, r, op }`; each family's
//!   `op_name()` method (alongside the downcast accessors) supplies the
//!   wire-format op string ("eq", "add", etc.).
//! - **`ShuffleLocation` is `physical_plan::ShuffleLocation`** (6 fields,
//!   matches the proto exactly), not the older 4-field
//!   `datatypes::ShuffleLocation` that's also in the workspace.
//!   Cleaning up that duplicate type is a separate follow-up.

use crate::pb;
use datasource::DataSource;
use datatypes::{Field, Schema};
use physical_plan::{
    AggregateExpression, AggregateMode, Expression, PhysicalPlan, ShuffleLocation, Task,
};

use arrow_schema::DataType;

/// `&dyn PhysicalPlan` → `pb::PhysicalPlanNode`.
pub fn serialize_physical_plan(plan: &dyn PhysicalPlan) -> pb::PhysicalPlanNode {
    use pb::physical_plan_node::PlanType;
    let any = plan.as_any();

    if let Some(scan) = any.downcast_ref::<physical_plan::ScanExec>() {
        let (path, file_format) = data_source_path_and_format(scan.ds.as_ref());
        return pb::PhysicalPlanNode {
            plan_type: Some(PlanType::Scan(pb::ScanExecNode {
                path,
                // **Important**: send the FULL (pre-projection) data-source schema,
                // not the projected output schema. `plan.schema()` is the
                // projected one (`ds.schema().select(&projection)`); the
                // receiver needs the full schema to construct a
                // `CsvDataSource` that knows how many columns the file has,
                // and then applies `projection` in `scan(...)`. Sending the
                // projected schema caused arrow's CSV reader to expect a
                // 2-column file when the file actually has 6 columns —
                // surfaced as "incorrect number of fields for line 1" in
                // client/tests/distributed_integration_test.rs.
                schema: Some((&scan.ds.schema()).into()),
                projection: scan.projection.clone(),
                file_format,
            })),
        };
    }
    if let Some(proj) = any.downcast_ref::<physical_plan::ProjectionExec>() {
        return pb::PhysicalPlanNode {
            plan_type: Some(PlanType::Projection(Box::new(pb::ProjectionExecNode {
                input: Some(Box::new(serialize_physical_plan(proj.input.as_ref()))),
                schema: Some((&proj.schema).into()),
                expr: proj
                    .expr
                    .iter()
                    .map(|e| serialize_physical_expr(e.as_ref()))
                    .collect(),
            }))),
        };
    }
    if let Some(sel) = any.downcast_ref::<physical_plan::SelectionExec>() {
        return pb::PhysicalPlanNode {
            plan_type: Some(PlanType::Selection(Box::new(pb::SelectionExecNode {
                input: Some(Box::new(serialize_physical_plan(sel.input.as_ref()))),
                expr: Some(serialize_physical_expr(sel.expr.as_ref())),
            }))),
        };
    }
    if let Some(agg) = any.downcast_ref::<physical_plan::HashAggregateExec>() {
        return pb::PhysicalPlanNode {
            plan_type: Some(PlanType::HashAggregate(Box::new(
                pb::HashAggregateExecNode {
                    input: Some(Box::new(serialize_physical_plan(agg.input.as_ref()))),
                    group_expr: agg
                        .group_expr
                        .iter()
                        .map(|e| serialize_physical_expr(e.as_ref()))
                        .collect(),
                    aggregate_expr: agg
                        .aggregate_expr
                        .iter()
                        .map(|a| serialize_physical_aggr_expr(a.as_ref()))
                        .collect(),
                    schema: Some((&agg.schema).into()),
                    mode: aggregate_mode_to_proto(&agg.mode) as i32,
                },
            ))),
        };
    }
    if let Some(sw) = any.downcast_ref::<physical_plan::ShuffleWriterExec>() {
        return pb::PhysicalPlanNode {
            plan_type: Some(PlanType::ShuffleWriter(Box::new(
                pb::ShuffleWriterExecNode {
                    input: Some(Box::new(serialize_physical_plan(sw.input.as_ref()))),
                    partition_expr: sw
                        .partition_expr
                        .iter()
                        .map(|e| serialize_physical_expr(e.as_ref()))
                        .collect(),
                    job_uuid: sw.job_uuid.clone(),
                    stage_id: sw.stage_id,
                    partition_count: sw.partition_count,
                },
            ))),
        };
    }
    if let Some(sr) = any.downcast_ref::<physical_plan::ShuffleReaderExec>() {
        return pb::PhysicalPlanNode {
            plan_type: Some(PlanType::ShuffleReader(pb::ShuffleReaderExecNode {
                schema: Some((&sr.shuffle_schema).into()),
                shuffle_locations: sr.shuffle_locations.iter().map(Into::into).collect(),
            })),
        };
    }
    panic!("Cannot serialize physical operator to protobuf: {}", plan)
}

/// `&dyn Expression` → `pb::PhysicalExprNode`.
pub fn serialize_physical_expr(expr: &dyn Expression) -> pb::PhysicalExprNode {
    use pb::physical_expr_node::ExprType;
    let any = expr.as_any();
    let expr_type = if let Some(c) = any.downcast_ref::<physical_plan::ColumnExpression>() {
        ExprType::Column(c.i as i32)
    } else if let Some(s) = any.downcast_ref::<physical_plan::LiteralStringExpression>() {
        ExprType::LiteralString(s.value.clone())
    } else if let Some(n) = any.downcast_ref::<physical_plan::LiteralLongExpression>() {
        ExprType::LiteralLong(n.value)
    } else if let Some(n) = any.downcast_ref::<physical_plan::LiteralDoubleExpression>() {
        ExprType::LiteralDouble(n.value)
    } else if let Some(d) = any.downcast_ref::<physical_plan::LiteralDateExpression>() {
        ExprType::LiteralDate(d.days_since_epoch)
    } else if let Some(be) = expr.as_boolean_expression() {
        // Family-narrowing: `as_boolean_expression` returns `&dyn BooleanExpression`
        // so we can read `left()/right()/op_name()` uniformly across all 8 ops
        // without enumerating each concrete type here.
        ExprType::BinaryExpr(Box::new(pb::PhysicalBinaryExprNode {
            l: Some(Box::new(serialize_physical_expr(be.left().as_ref()))),
            r: Some(Box::new(serialize_physical_expr(be.right().as_ref()))),
            op: be.op_name().to_string(),
        }))
    } else if let Some(me) = expr.as_math_expression() {
        // Same family-narrowing pattern for the 5 math ops.
        ExprType::BinaryExpr(Box::new(pb::PhysicalBinaryExprNode {
            l: Some(Box::new(serialize_physical_expr(me.left().as_ref()))),
            r: Some(Box::new(serialize_physical_expr(me.right().as_ref()))),
            op: me.op_name().to_string(),
        }))
    } else if let Some(c) = any.downcast_ref::<physical_plan::CastExpression>() {
        ExprType::CastExpr(Box::new(pb::PhysicalCastExprNode {
            expr: Some(Box::new(serialize_physical_expr(c.expr.as_ref()))),
            arrow_type: data_type_to_proto(&c.data_type) as i32,
        }))
    } else {
        panic!("Cannot serialize physical expression to protobuf: {}", expr)
    };
    pb::PhysicalExprNode {
        expr_type: Some(expr_type),
    }
}

/// `&dyn AggregateExpression` → `pb::PhysicalAggregateExprNode`.
pub fn serialize_physical_aggr_expr(
    expr: &dyn AggregateExpression,
) -> pb::PhysicalAggregateExprNode {
    let any = expr.as_any();
    let fn_kind = if any.is::<physical_plan::SumExpression>() {
        pb::AggregateFunction::Sum
    } else if any.is::<physical_plan::MinExpression>() {
        pb::AggregateFunction::Min
    } else if any.is::<physical_plan::MaxExpression>() {
        pb::AggregateFunction::Max
    } else if any.is::<physical_plan::AvgExpression>() {
        pb::AggregateFunction::Avg
    } else if any.is::<physical_plan::CountExpression>() {
        pb::AggregateFunction::Count
    } else {
        panic!(
            "Cannot serialize aggregate expression to protobuf: {}",
            expr
        )
    };
    let input = expr.input_expression();
    pb::PhysicalAggregateExprNode {
        aggr_function: fn_kind as i32,
        input_expr: Some(serialize_physical_expr(input.as_ref())),
    }
}

/// `&Task` → `pb::TaskInfo`.
pub fn serialize_task(task: &Task) -> pb::TaskInfo {
    pb::TaskInfo {
        job_uuid: task.job_uuid.clone(),
        stage_id: task.stage_id,
        task_id: task.task_id,
        partition_id: task.partition_id,
        plan: Some(serialize_physical_plan(task.plan.as_ref())),
    }
}

// ---------------------------------------------------------------------------
// Leaf conversions — standard `From` impls so call sites read as
// `schema.into()` / `field.into()` / `loc.into()`. DataFusion's pattern: where
// the conversion is type-to-type with no extra context, prefer a trait impl
// over a named function so the call site is uniform with the rest of the
// type-conversion machinery in Rust.
// ---------------------------------------------------------------------------

/// `&Schema` → `pb::Schema`.
impl From<&Schema> for pb::Schema {
    fn from(schema: &Schema) -> Self {
        pb::Schema {
            columns: schema.fields.iter().map(Into::into).collect(),
        }
    }
}

/// `&Field` → `pb::Field`.
impl From<&Field> for pb::Field {
    fn from(field: &Field) -> Self {
        pb::Field {
            name: field.name.clone(),
            arrow_type: data_type_to_proto(&field.data_type) as i32,
            nullable: true,
            children: vec![],
        }
    }
}

/// `&ShuffleLocation` → `pb::ShuffleLocation`.
/// Uses the 6-field `physical_plan::ShuffleLocation` (there is also a
/// 4-field `datatypes::ShuffleLocation` left over from earlier porting;
/// the physical_plan one is the production type and matches the proto
/// exactly).
impl From<&ShuffleLocation> for pb::ShuffleLocation {
    fn from(loc: &ShuffleLocation) -> Self {
        pb::ShuffleLocation {
            job_uuid: loc.job_uuid.clone(),
            stage_id: loc.stage_id,
            partition_id: loc.partition_id,
            executor_id: loc.executor_id.clone(),
            executor_host: loc.executor_host.clone(),
            executor_port: loc.executor_port,
        }
    }
}

// ---------------------------------------------------------------------------
// Private helpers.
// ---------------------------------------------------------------------------

/// Extract `(path, file_format)` from a `&dyn DataSource`, branching on CSV
/// vs. Parquet via `DataSource::as_any` + `downcast_ref` — the same idiom
/// DataFusion uses for `TableProvider`.
fn data_source_path_and_format(ds: &dyn DataSource) -> (String, String) {
    let any = ds.as_any();
    if let Some(csv) = any.downcast_ref::<datasource::CsvDataSource>() {
        (csv.filename.clone(), "csv".to_string())
    } else if let Some(parquet) = any.downcast_ref::<datasource::ParquetDataSource>() {
        (parquet.filename.clone(), "parquet".to_string())
    } else {
        panic!("Unsupported data-source type for protobuf serialisation")
    }
}

/// Map our `AggregateMode` → the proto enum. Same shape on both sides.
fn aggregate_mode_to_proto(m: &AggregateMode) -> pb::AggregateMode {
    match m {
        AggregateMode::Complete => pb::AggregateMode::Complete,
        AggregateMode::Partial => pb::AggregateMode::Partial,
        AggregateMode::Final => pb::AggregateMode::Final,
    }
}

/// Map `arrow_schema::DataType` → the proto `ArrowType` enum. The reverse of
/// `physical_plan_deserializer::from_proto_arrow_type`; symmetric coverage.
fn data_type_to_proto(dt: &DataType) -> pb::ArrowType {
    match dt {
        DataType::Boolean => pb::ArrowType::Bool,
        DataType::Int8 => pb::ArrowType::Int8,
        DataType::Int16 => pb::ArrowType::Int16,
        DataType::Int32 => pb::ArrowType::Int32,
        DataType::Int64 => pb::ArrowType::Int64,
        DataType::UInt8 => pb::ArrowType::Uint8,
        DataType::UInt16 => pb::ArrowType::Uint16,
        DataType::UInt32 => pb::ArrowType::Uint32,
        DataType::UInt64 => pb::ArrowType::Uint64,
        DataType::Float32 => pb::ArrowType::Float,
        DataType::Float64 => pb::ArrowType::Double,
        DataType::Utf8 => pb::ArrowType::Utf8,
        DataType::Date32 => pb::ArrowType::Date32,
        other => panic!("Cannot serialize Arrow type to protobuf: {other:?}"),
    }
}
