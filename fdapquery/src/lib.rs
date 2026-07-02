//! # fdapquery
//!
//! Umbrella crate for the fdapquery query engine. Re-exports the
//! user-facing types from every foundation crate so consumers can
//! write `use fdapquery::prelude::*;` instead of hand-importing
//! from 10+ crates.
//!
//! Mirrors DataFusion's `datafusion` umbrella crate. Per-crate
//! modules (`fdapquery::execution`, `fdapquery::physical_plan`, …)
//! match `datafusion::execution`, `datafusion::physical_plan`, etc.

// In-crate modules — the user-facing high-level API.
// moved these here from `fdapquery-execution` to break the
// physical-plan ↔ execution dep cycle. Matches DataFusion's
// pattern of hosting `SessionContext` in the umbrella crate.
pub mod parallel_context;
// The concrete `DefaultPhysicalPlanner` lives in the umbrella crate so
// it can compose `fdapquery-catalog` (for `source_as_provider` /
// `TableProvider`) + `fdapquery-physical-plan` + `fdapquery-datasource`
// without forcing those deps on every physical-plan consumer. Matches
// DataFusion's split: the `PhysicalPlanner` trait stays in
// `datafusion-physical-plan`, the concrete `DefaultPhysicalPlanner`
// lives in `datafusion-core`.
pub mod physical_planner;
pub mod session_context;
// `SessionState`, `SessionStateBuilder`, the
// outer `QueryPlanner` trait, and `DefaultQueryPlanner` all live in
// the umbrella because DataFusion places them in `datafusion-core`
// alongside `SessionContext` (the `QueryPlanner` trait's method
// takes `&SessionState`, so they co-locate to avoid a dep cycle).
pub mod session_state;

// Per-crate module re-exports — matches DataFusion's pattern.
pub use fdapquery_catalog as catalog;
pub use fdapquery_common as common;
pub use fdapquery_datatypes as datatypes;

// `fdapquery::execution` is a real curated module,
// not a flat alias of `fdapquery_execution`. It re-exports everything
// from `fdapquery_execution` AND adds a `context` submodule that hosts
// `SessionContext` / `ParallelContext`. Result: the canonical path
// becomes `fdapquery::execution::context::SessionContext`, matching
// `datafusion::execution::context::SessionContext` exactly. Top-level
// re-exports (`fdapquery::SessionContext` and the prelude) stay for
// ergonomics.
pub mod execution {
    pub use fdapquery_execution::*;

    /// `SessionContext` and `ParallelContext` re-exports — the
    /// high-level user-facing entry points. Mirrors
    /// `datafusion::execution::context`.
    pub mod context {
        pub use crate::parallel_context::ParallelContext;
        pub use crate::session_context::SessionContext;
    }
}
// DataFusion calls this `logical_expr` (the
// `expr` name inside the crate is reserved for the `Expr`-construction
// submodule `logical_expr::expr_fn`).
pub use fdapquery_expr as logical_expr;
pub use fdapquery_functions as functions;
pub use fdapquery_functions_aggregate as functions_aggregate;
pub use fdapquery_optimizer as optimizer;
pub use fdapquery_physical_expr as physical_expr;
// Strict mirror of DataFusion's
// `datafusion::physical_optimizer` re-export of the
// `datafusion-physical-optimizer` crate. v0.1 hosts the
// `PhysicalOptimizerRule` trait and `PhysicalOptimizer` driver;
// concrete rules land as follow-up tasks.
pub use fdapquery_physical_optimizer as physical_optimizer;
pub use fdapquery_physical_plan as physical_plan;
pub use fdapquery_sql as sql;

// Top-level convenience re-exports — the most common types.
pub use fdapquery_catalog::{CsvDataSource, InMemoryDataSource, ParquetDataSource, TableProvider};
// `ScalarValue` lives in `fdapquery-common` now
// (matches DataFusion's `datafusion_common::ScalarValue`). Import the
// canonical path even though datatypes still re-exports it transitionally.
pub use fdapquery_common::ScalarValue;
pub use fdapquery_datatypes::{FdapQueryError, Field, RecordBatch, Result, Schema};
// `SessionContext` / `ParallelContext` now live in this crate.
// Top-level re-exports for ergonomics.
pub use fdapquery_expr::{DataFrame, LogicalPlan};
pub use fdapquery_physical_plan::{ColumnarValue, ExecutionPlan, SendableRecordBatchStream};
pub use parallel_context::ParallelContext;
// `DefaultPhysicalPlanner` lives in this crate (see the `physical_planner`
// module above). Re-export at the umbrella root for ergonomics.
pub use physical_planner::DefaultPhysicalPlanner;
pub use session_context::SessionContext;
// Outer `QueryPlanner` trait, `DefaultQueryPlanner`,
// `SessionState`, and `SessionStateBuilder`. Mirror of
// `datafusion::execution::session_state::{SessionState, SessionStateBuilder}`
// + `datafusion::execution::context::QueryPlanner`.
pub use session_state::{
    Analyzer, DefaultQueryPlanner, EmptySerializerRegistry, PhysicalOptimizer, QueryPlanner,
    SerializerRegistry, SessionState, SessionStateBuilder,
};

pub mod prelude {
    //! Conventional `use fdapquery::prelude::*;` import surface.
    //!
    //! Byte-for-byte mirror of `datafusion::prelude`: the canonical
    //! short list of types plus the two `Expr`-builder helpers
    //! `col` and `lit`. The rquery-era DSL free helpers
    //! (`min`/`max`/`sum`/`count`/`avg`/`add`/`sub`/…) are
    //! intentionally absent — dropped them so the
    //! prelude is a 1:1 mirror of DataFusion's. Consumers building
    //! aggregate or binary `Expr` values use the long-form
    //! constructors in `fdapquery_expr::Expr` or the operator
    //! overloads on `Expr`.
    //!
    //! `lit(value)` is the generic literal factory — strict mirror of
    //! `datafusion_expr::lit<T: Literal>(value: T) -> Expr`. It accepts
    //! any value implementing `fdapquery_expr::Literal` (`&str`,
    //! `String`, `i64`, `i32`, `f64`, `f32`, `bool`, `chrono::NaiveDate`)
    //! and produces an `Expr::Literal(ScalarValue::*)` of the matching
    //! variant.

    pub use crate::{
        CsvDataSource, DataFrame, DefaultPhysicalPlanner, ExecutionPlan, FdapQueryError, Field,
        InMemoryDataSource, LogicalPlan, ParallelContext, ParquetDataSource, RecordBatch, Result,
        ScalarValue, Schema, SendableRecordBatchStream, SessionContext, TableProvider,
    };

    // The canonical `Expr`-introduction helpers — strict mirror of
    // `datafusion::prelude::{col, lit}`.
    pub use fdapquery_expr::{col, lit};
}
