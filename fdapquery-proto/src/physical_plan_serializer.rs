//! `PhysicalPlan` → `protobuf::PhysicalPlanNode`, `PhysicalExpr` → `protobuf::PhysicalExprNode`,
//! `AggregateExpr` → `protobuf::PhysicalAggregateExprNode`, `Schema` / `Field` →
//! their proto equivalents, plus `ShuffleLocation` and `Task` for distributed
//! task dispatch. Used by the `fdapquery-flight-server`,
//! `fdapquery-flight-client`, and `fdapquery-distributed` crates.
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
//!    `Schema → protobuf::Schema`, `Field → protobuf::Field`, `ShuffleLocation →
//!    protobuf::ShuffleLocation`.
//! 3. **Enum mappings** that don't justify a trait impl stay as private helper
//!    functions: `data_type_to_proto`, `aggregate_mode_to_proto`.
//!
//! Equivalent of DataFusion's `datafusion/proto/src/physical_plan/to_proto.rs`,
//! down to the verb (`serialize_*`).
//!
//! ## DataSourceExec
//!
//! The leaf scan operator is now `DataSourceExec` (in
//! `fdapquery-datasource`) wrapping per-format `*Config: DataSource`
//! structs (in `fdapquery-catalog`). The serializer downcasts the plan
//! to `DataSourceExec`, then downcasts the inner `Arc<dyn DataSource>`
//! to `CsvDataSourceConfig` / `ParquetDataSourceConfig` to recover the
//! wire fields (filename, projection, format). The wire-format message
//! (`DataSourceExecNode`) was renamed from `ScanExecNode` alongside the
//! `FilterNode` / `FilterExecNode` proto renames.

use crate::protobuf;
use fdapquery_catalog::{CsvDataSourceConfig, ParquetDataSourceConfig};
use fdapquery_common::ScalarValue;
use fdapquery_datasource::{DataSource, DataSourceExec};
use fdapquery_datatypes::{Field, Schema};
use fdapquery_physical_plan::{
    AggregateExpr, AggregateMode, ExecutionPlan, PhysicalExpr, ShuffleLocation, Task,
};

use arrow_schema::DataType;

/// `&dyn ExecutionPlan` → `protobuf::PhysicalPlanNode`.
pub fn serialize_physical_plan(plan: &dyn ExecutionPlan) -> protobuf::PhysicalPlanNode {
    use protobuf::physical_plan_node::PlanType;
    let any = plan.as_any();

    if let Some(ds_exec) = any.downcast_ref::<DataSourceExec>() {
        // The scan leaf is `DataSourceExec` wrapping a per-format
        // `*Config: DataSource`. Branch on the inner concrete type to
        // recover the wire fields.
        let (path, full_schema, projection_names, file_format) =
            data_source_wire_fields(ds_exec.source().as_ref());
        return protobuf::PhysicalPlanNode {
            // Wire variant name `Scan` — Stable.
            plan_type: Some(PlanType::Scan(protobuf::DataSourceExecNode {
                path,
                // **Important**: send the FULL (pre-projection) data-source schema,
                // not the projected output schema. The receiver materialises a
                // `CsvDataSource` / `ParquetDataSource` with the full schema and
                // applies the projection inside `*Config::open`. Sending the
                // projected schema caused arrow's CSV reader to expect a
                // 2-column file when the file actually has 6 columns —
                // surfaced as "incorrect number of fields for line 1" in
                // client/tests/distributed_integration_test.rs.
                schema: Some((&full_schema).into()),
                projection: projection_names,
                file_format,
            })),
        };
    }
    if let Some(proj) = any.downcast_ref::<fdapquery_physical_plan::ProjectionExec>() {
        let projected_schema = proj.schema();
        return protobuf::PhysicalPlanNode {
            plan_type: Some(PlanType::Projection(Box::new(
                protobuf::ProjectionExecNode {
                    input: Some(Box::new(serialize_physical_plan(proj.input().as_ref()))),
                    schema: Some((&projected_schema).into()),
                    expr: proj
                        .expr()
                        .iter()
                        .map(|e| serialize_physical_expr(e.as_ref()))
                        .collect(),
                },
            ))),
        };
    }
    if let Some(sel) = any.downcast_ref::<fdapquery_physical_plan::FilterExec>() {
        return protobuf::PhysicalPlanNode {
            // Wire variant name `Selection` — Stable.
            plan_type: Some(PlanType::Selection(Box::new(protobuf::FilterExecNode {
                input: Some(Box::new(serialize_physical_plan(sel.input().as_ref()))),
                expr: Some(serialize_physical_expr(sel.predicate().as_ref())),
            }))),
        };
    }
    if let Some(agg) = any.downcast_ref::<fdapquery_physical_plan::AggregateExec>() {
        // `AggregateExec` is now strict-mirrored to
        // DataFusion: `group_by` (a `PhysicalGroupBy` struct) replaces the
        // old flat `group_expr` field, and the output `schema` is private.
        // The wire format still carries the simple flat group-expression
        // list because the planner only emits the `new_single` shape today
        // (no grouping sets); the matching renames on the proto side are
        // tracked by #110, #111. We pull the simple-shape group expressions
        // out via `PhysicalGroupBy::input_exprs()`.
        return protobuf::PhysicalPlanNode {
            plan_type: Some(PlanType::HashAggregate(Box::new(
                protobuf::HashAggregateExecNode {
                    input: Some(Box::new(serialize_physical_plan(agg.input().as_ref()))),
                    group_expr: agg
                        .group_expr()
                        .input_exprs()
                        .iter()
                        .map(|e| serialize_physical_expr(e.as_ref()))
                        .collect(),
                    aggregate_expr: agg
                        .aggr_expr()
                        .iter()
                        .map(|a| serialize_physical_aggr_expr(a.as_ref()))
                        .collect(),
                    // `agg.schema()` returns `Schema` by value (mirrors
                    // `ExecutionPlan::schema`'s pre-strict-mirror signature);
                    // the temporary lives long enough for `(&Schema).into()`.
                    schema: Some({
                        let s = agg.schema();
                        (&s).into()
                    }),
                    mode: aggregate_mode_to_proto(*agg.mode()) as i32,
                },
            ))),
        };
    }
    if let Some(sw) = any.downcast_ref::<fdapquery_physical_plan::ShuffleWriterExec>() {
        return protobuf::PhysicalPlanNode {
            plan_type: Some(PlanType::ShuffleWriter(Box::new(
                protobuf::ShuffleWriterExecNode {
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
    if let Some(sr) = any.downcast_ref::<fdapquery_physical_plan::ShuffleReaderExec>() {
        return protobuf::PhysicalPlanNode {
            plan_type: Some(PlanType::ShuffleReader(protobuf::ShuffleReaderExecNode {
                schema: Some((&sr.shuffle_schema).into()),
                shuffle_locations: sr.shuffle_locations.iter().map(Into::into).collect(),
            })),
        };
    }
    panic!("Cannot serialize physical operator to protobuf: {plan}")
}

/// `&dyn PhysicalExpr` → `protobuf::PhysicalExprNode`.
pub fn serialize_physical_expr(expr: &dyn PhysicalExpr) -> protobuf::PhysicalExprNode {
    use protobuf::physical_expr_node::ExprType;
    let any = expr.as_any();
    let expr_type = if let Some(c) = any.downcast_ref::<fdapquery_physical_plan::Column>() {
        // `Column` serializes to a `PhysicalColumn` sub-message carrying
        // both the source-schema column **name** and the **index**,
        // matching DataFusion's `datafusion.proto: PhysicalColumn
        // { name, index }` wire shape exactly.
        ExprType::Column(protobuf::PhysicalColumn {
            name: c.name().to_string(),
            index: c.index() as u32,
        })
    } else if let Some(l) = any.downcast_ref::<fdapquery_physical_plan::Literal>() {
        // The unified `Literal { value: ScalarValue }`
        // replaces the five sibling types. The wire format keeps its five
        // distinct literal variants (renaming to a single `Literal` wire
        // variant is #110/#111 work), so we branch on the inner
        // `ScalarValue` and emit the matching wire variant.
        match l.value() {
            ScalarValue::Int64(n) => ExprType::LiteralLong(*n),
            ScalarValue::Float64(n) => ExprType::LiteralDouble(*n),
            ScalarValue::Utf8(s) => ExprType::LiteralString(s.clone()),
            ScalarValue::Date32(d) => ExprType::LiteralDate(*d),
            other => panic!(
                "Cannot serialize Literal({other:?}) to protobuf: \
                 wire format only supports Int64, Float64, Utf8, and Date32"
            ),
        }
    } else if let Some(binary) = serialize_binary_op(expr) {
        // The family-narrowing `as_boolean_expression`
        // / `as_math_expression` accessors are gone; each of the 12
        // concrete binary types is now enumerated by `as_any`
        // downcast in `serialize_binary_op` below, with the wire op
        // string hardcoded per arm.
        ExprType::BinaryExpr(Box::new(binary))
    } else if let Some(c) = any.downcast_ref::<fdapquery_physical_plan::CastExpr>() {
        ExprType::CastExpr(Box::new(protobuf::PhysicalCastExprNode {
            expr: Some(Box::new(serialize_physical_expr(c.expr.as_ref()))),
            arrow_type: data_type_to_proto(&c.data_type) as i32,
        }))
    } else {
        panic!("Cannot serialize physical expression to protobuf: {expr}")
    };
    protobuf::PhysicalExprNode {
        expr_type: Some(expr_type),
    }
}

/// `&dyn PhysicalExpr` → `protobuf::PhysicalBinaryExprNode` if `expr` is the
/// unified [`BinaryExpr`], else `None`.
///
/// The 12 sibling concrete types collapsed into a
/// single `BinaryExpr { left, op, right }`. Serialisation is a single
/// `as_any().downcast_ref::<BinaryExpr>()` followed by mapping
/// `BinaryExpr::op()` to the wire-format op string. The wire format
/// retains its string-typed `op` field; proto tasks #110/#111 widen
/// it to a typed `Operator` enum.
fn serialize_binary_op(expr: &dyn PhysicalExpr) -> Option<protobuf::PhysicalBinaryExprNode> {
    use fdapquery_expr::Operator;
    use fdapquery_physical_plan::BinaryExpr;

    let any = expr.as_any();
    let b = any.downcast_ref::<BinaryExpr>()?;
    let op = match b.op() {
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
            "serialize_binary_op: Operator::{other:?} has no wire-format \
             encoding in rquery.proto's PhysicalBinaryExprNode.op string \
             (proto tasks #110/#111)"
        ),
    };
    Some(protobuf::PhysicalBinaryExprNode {
        l: Some(Box::new(serialize_physical_expr(b.left().as_ref()))),
        r: Some(Box::new(serialize_physical_expr(b.right().as_ref()))),
        op: op.to_string(),
    })
}

/// `&dyn AggregateExpr` → `protobuf::PhysicalAggregateExprNode`.
pub fn serialize_physical_aggr_expr(
    expr: &dyn AggregateExpr,
) -> protobuf::PhysicalAggregateExprNode {
    let any = expr.as_any();
    let fn_kind = if any.is::<fdapquery_physical_plan::SumExpr>() {
        protobuf::AggregateFunction::Sum
    } else if any.is::<fdapquery_physical_plan::MinExpr>() {
        protobuf::AggregateFunction::Min
    } else if any.is::<fdapquery_physical_plan::MaxExpr>() {
        protobuf::AggregateFunction::Max
    } else if any.is::<fdapquery_physical_plan::AvgExpr>() {
        protobuf::AggregateFunction::Avg
    } else if any.is::<fdapquery_physical_plan::CountExpr>() {
        protobuf::AggregateFunction::Count
    } else {
        panic!("Cannot serialize aggregate expression to protobuf: {expr}")
    };
    let input = expr.input_expression();
    protobuf::PhysicalAggregateExprNode {
        aggr_function: fn_kind as i32,
        input_expr: Some(serialize_physical_expr(input.as_ref())),
    }
}

/// `&Task` → `protobuf::TaskInfo`.
pub fn serialize_task(task: &Task) -> protobuf::TaskInfo {
    protobuf::TaskInfo {
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

/// `&Schema` → `protobuf::Schema`.
impl From<&Schema> for protobuf::Schema {
    fn from(schema: &Schema) -> Self {
        protobuf::Schema {
            columns: schema.fields().iter().map(|f| f.as_ref().into()).collect(),
        }
    }
}

/// `&Field` → `protobuf::Field`.
impl From<&Field> for protobuf::Field {
    fn from(field: &Field) -> Self {
        protobuf::Field {
            name: field.name().clone(),
            arrow_type: data_type_to_proto(field.data_type()) as i32,
            nullable: true,
            children: vec![],
        }
    }
}

/// `&ShuffleLocation` → `protobuf::ShuffleLocation`.
/// Uses the 6-field `fdapquery_physical_plan::ShuffleLocation` (there is also a
/// 4-field `fdapquery_datatypes::ShuffleLocation` left over from earlier porting;
/// the physical_plan one is the production type and matches the proto
/// exactly).
impl From<&ShuffleLocation> for protobuf::ShuffleLocation {
    fn from(loc: &ShuffleLocation) -> Self {
        protobuf::ShuffleLocation {
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

/// Extract `(path, full_schema, projection_column_names, file_format)` from
/// a `&dyn DataSource` by branching on the per-format `*Config` concrete
/// type via `as_any` + `downcast_ref` — the same idiom DataFusion uses
/// for per-source wire serialisation. Returns the column **names**
/// corresponding to the projection indices the config holds, so the
/// wire format (which is name-based today) stays unchanged.
fn data_source_wire_fields(ds: &dyn DataSource) -> (String, Schema, Vec<String>, String) {
    let any = ds.as_any();
    if let Some(csv) = any.downcast_ref::<CsvDataSourceConfig>() {
        let projection_names = projection_indices_to_names(csv.full_schema(), csv.projection());
        return (
            csv.filename().to_string(),
            csv.full_schema().clone(),
            projection_names,
            "csv".to_string(),
        );
    }
    if let Some(parquet) = any.downcast_ref::<ParquetDataSourceConfig>() {
        let projection_names =
            projection_indices_to_names(parquet.full_schema(), parquet.projection());
        return (
            parquet.filename().to_string(),
            parquet.full_schema().clone(),
            projection_names,
            "parquet".to_string(),
        );
    }
    panic!(
        "Unsupported DataSource type for protobuf serialisation: \
         only CsvDataSourceConfig and ParquetDataSourceConfig have wire encodings"
    )
}

/// Map an optional list of column indices into the full schema to a
/// list of column **names**. `None` means "all columns" and round-trips
/// as the empty list (the receiver treats `repeated string projection`
/// of length zero as "no projection"). Same convention as the previous
/// `ScanExec` wire encoding.
fn projection_indices_to_names(
    full_schema: &Schema,
    projection: Option<&Vec<usize>>,
) -> Vec<String> {
    match projection {
        None => Vec::new(),
        Some(indices) => indices
            .iter()
            .map(|i| full_schema.fields()[*i].name().clone())
            .collect(),
    }
}

/// Map our `AggregateMode` → the proto enum.
///
/// The proto wire format pre-dates the rename to DataFusion's
/// six-variant `AggregateMode`. Until #110 / #111 (the matching proto
/// renames) ship, the wire enum stays `COMPLETE` / `PARTIAL` / `FINAL` and
/// we map: `Single` ↔ `Complete`. The partitioned variants
/// (`FinalPartitioned`, `SinglePartitioned`) and `PartialReduce` are not
/// produced by any current code path; if a serializer ever sees them it
/// indicates a planner extension landed without bumping the wire format,
/// so we surface that loudly rather than silently degrade.
fn aggregate_mode_to_proto(m: AggregateMode) -> protobuf::AggregateMode {
    match m {
        AggregateMode::Single => protobuf::AggregateMode::Complete,
        AggregateMode::Partial => protobuf::AggregateMode::Partial,
        AggregateMode::Final => protobuf::AggregateMode::Final,
        AggregateMode::FinalPartitioned
        | AggregateMode::SinglePartitioned
        | AggregateMode::PartialReduce => panic!(
            "AggregateMode::{m:?} has no proto wire-format mapping; \
             extend rquery.proto's AggregateMode enum before emitting it"
        ),
    }
}

/// Map `arrow_schema::DataType` → the proto `ArrowType` enum. The reverse of
/// `physical_plan_deserializer::from_proto_arrow_type`; symmetric coverage.
fn data_type_to_proto(dt: &DataType) -> protobuf::ArrowType {
    match dt {
        DataType::Boolean => protobuf::ArrowType::Bool,
        DataType::Int8 => protobuf::ArrowType::Int8,
        DataType::Int16 => protobuf::ArrowType::Int16,
        DataType::Int32 => protobuf::ArrowType::Int32,
        DataType::Int64 => protobuf::ArrowType::Int64,
        DataType::UInt8 => protobuf::ArrowType::Uint8,
        DataType::UInt16 => protobuf::ArrowType::Uint16,
        DataType::UInt32 => protobuf::ArrowType::Uint32,
        DataType::UInt64 => protobuf::ArrowType::Uint64,
        DataType::Float32 => protobuf::ArrowType::Float,
        DataType::Float64 => protobuf::ArrowType::Double,
        DataType::Utf8 => protobuf::ArrowType::Utf8,
        DataType::Date32 => protobuf::ArrowType::Date32,
        other => panic!("Cannot serialize Arrow type to protobuf: {other:?}"),
    }
}
