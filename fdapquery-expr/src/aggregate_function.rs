//! `AggregateFunction` struct + `AggregateFunctionKind` enum.
//!
//! Strict mirror of DataFusion's `AggregateFunction` and
//! `AggregateFunctionParams` structs in
//! `datafusion_expr::expr` (`datafusion_expr::expr::AggregateFunction`
//! and `datafusion_expr::expr::AggregateFunctionParams`). The single
//! struct shape is
//! the wire-format peer of DataFusion's `Expr::AggregateFunction(...)`
//! variant — adopted in place of fdapquery's older
//! standalone `AggregateExpr` enum, which collided with the physical-side
//! `AggregateExpr` trait and forced a `LogicalAggregateExpr` alias hack.
//!
//! ## Divergence from DataFusion: `func: AggregateFunctionKind`
//!
//! DataFusion's `AggregateFunction { func: Arc<AggregateUDF>, params }`
//! carries an `Arc<AggregateUDF>` for the function identity (see
//! `datafusion_expr::expr::AggregateFunction::func`). fdapquery does not
//! yet have an `AggregateUDF` infrastructure — that surface arrives with
//! the `SessionStateBuilder` port that introduces the UDF/UDAF/UDWF
//! registries. For the duration of this gap, the `func` field holds the
//! pre-UDF kind enum `AggregateFunctionKind`, which is the exact shape
//! DataFusion used before the UDF migration. Once the registries land,
//! the migration is a single field swap (`func: AggregateFunctionKind` →
//! `func: Arc<AggregateUDF>`) with no impact on the rest of the struct
//! or any call site that goes through `AggregateFunction::new`.
//!
//! Every other field (`args`, `distinct`, `filter`, `order_by`,
//! `null_treatment`) matches DataFusion byte-for-byte by name and type.

use crate::logical_expr::Expr;
use crate::sort::{NullTreatment, Sort};
use std::fmt;

/// Pre-UDF aggregate-function identifier. Same role as DataFusion's
/// `Arc<AggregateUDF>` — names the aggregate kind without inlining its
/// implementation. The variant names match the names of the corresponding
/// DataFusion built-in `AggregateUDF`s (`min`, `max`, `sum`, `avg`,
/// `count`) so the surface display strings (`MIN(expr)` / `MAX(expr)` /
/// `SUM(expr)` / `AVG(expr)` / `COUNT(expr)` / `COUNT(DISTINCT expr)`)
/// match DataFusion's `display_name` output byte-for-byte.
///
/// COUNT(DISTINCT …) is NOT a separate kind. The DISTINCT flag rides on
/// `AggregateFunctionParams::distinct` — the same shape DataFusion uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AggregateFunctionKind {
    Min,
    Max,
    Sum,
    Avg,
    Count,
}

impl AggregateFunctionKind {
    /// The uppercase SQL name of this aggregate, as DataFusion prints in
    /// `Expr::Display`.
    pub fn name(&self) -> &'static str {
        match self {
            AggregateFunctionKind::Min => "MIN",
            AggregateFunctionKind::Max => "MAX",
            AggregateFunctionKind::Sum => "SUM",
            AggregateFunctionKind::Avg => "AVG",
            AggregateFunctionKind::Count => "COUNT",
        }
    }
}

impl fmt::Display for AggregateFunctionKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Aggregate function expression — strict mirror of DataFusion's
/// `datafusion_expr::expr::AggregateFunction` struct.
#[derive(Debug, Clone, PartialEq)]
pub struct AggregateFunction {
    /// Name of the function (in DataFusion: `Arc<AggregateUDF>`; here:
    /// `AggregateFunctionKind` — see module doc).
    pub func: AggregateFunctionKind,
    pub params: AggregateFunctionParams,
}

/// Aggregate-function call parameters — strict mirror of DataFusion's
/// `datafusion_expr::expr::AggregateFunctionParams` struct.
/// Every field matches DataFusion
/// by name and type.
#[derive(Debug, Clone, PartialEq)]
pub struct AggregateFunctionParams {
    pub args: Vec<Expr>,
    /// Whether this is a DISTINCT aggregation or not
    pub distinct: bool,
    /// Optional filter
    pub filter: Option<Box<Expr>>,
    /// Optional ordering
    pub order_by: Vec<Sort>,
    pub null_treatment: Option<NullTreatment>,
}

impl AggregateFunction {
    /// Constructor mirroring DataFusion's
    /// `datafusion_expr::expr::AggregateFunction::new_udf`.
    /// Same field layout, with
    /// `func: AggregateFunctionKind` standing in for
    /// `func: Arc<AggregateUDF>` until #120.
    pub fn new(
        func: AggregateFunctionKind,
        args: Vec<Expr>,
        distinct: bool,
        filter: Option<Box<Expr>>,
        order_by: Vec<Sort>,
        null_treatment: Option<NullTreatment>,
    ) -> Self {
        Self {
            func,
            params: AggregateFunctionParams {
                args,
                distinct,
                filter,
                order_by,
                null_treatment,
            },
        }
    }
}
