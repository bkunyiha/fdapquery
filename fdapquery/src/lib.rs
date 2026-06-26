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

// In-crate modules — the user-facing high-level API. Session 15c
// moved these here from `fdapquery-execution` to break the
// physical-plan ↔ execution dep cycle. Matches DataFusion's
// pattern of hosting `SessionContext` in the umbrella crate.
pub mod parallel_context;
pub mod session_context;

// Per-crate module re-exports — matches DataFusion's pattern.
pub use fdapquery_catalog as catalog;
pub use fdapquery_common as common;
pub use fdapquery_datatypes as datatypes;

// Session 15d-1 #98 — `fdapquery::execution` is a real curated module,
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
// Session 15d-1 #93 — DataFusion calls this `logical_expr` (the
// `expr` name inside the crate is reserved for the `Expr`-construction
// submodule `logical_expr::expr_fn`).
pub use fdapquery_expr as logical_expr;
pub use fdapquery_functions as functions;
pub use fdapquery_functions_aggregate as functions_aggregate;
pub use fdapquery_optimizer as optimizer;
pub use fdapquery_physical_expr as physical_expr;
pub use fdapquery_physical_plan as physical_plan;
pub use fdapquery_sql as sql;

// Top-level convenience re-exports — the most common types.
pub use fdapquery_catalog::{CsvDataSource, InMemoryDataSource, ParquetDataSource, TableProvider};
// Session 15d-1 #108 — `ScalarValue` lives in `fdapquery-common` now
// (matches DataFusion's `datafusion_common::ScalarValue`). Import the
// canonical path even though datatypes still re-exports it transitionally.
pub use fdapquery_common::ScalarValue;
pub use fdapquery_datatypes::{FdapQueryError, Field, RecordBatch, Result, Schema};
// `SessionContext` / `ParallelContext` now live in this crate
// (Session 15c). Top-level re-exports for ergonomics.
pub use fdapquery_expr::{DataFrame, LogicalPlan};
pub use fdapquery_physical_plan::{
    DefaultPhysicalPlanner, ExecutionPlan, SendableRecordBatchStream,
};
pub use parallel_context::ParallelContext;
pub use session_context::SessionContext;

pub mod prelude {
    //! Conventional `use fdapquery::prelude::*;` import surface.
    //! Includes everything a consumer needs to build, register
    //! tables on, and execute a query against an `SessionContext`
    //! or `ParallelContext`. Mirrors `datafusion::prelude`.

    pub use crate::{
        CsvDataSource, DataFrame, DefaultPhysicalPlanner, ExecutionPlan, FdapQueryError, Field,
        InMemoryDataSource, LogicalPlan, ParallelContext, ParquetDataSource, RecordBatch, Result,
        ScalarValue, Schema, SendableRecordBatchStream, SessionContext, TableProvider,
    };

    // The DataFrame-building DSL free functions — `col(...)`,
    // `lit_*(...)`, and the aggregate builders `sum`/`min`/`max`/
    // `avg`/`count`/`count_distinct` plus the `cast` cast-builder.
    pub use fdapquery_expr::{
        avg, cast, col, count, count_distinct, lit_date, lit_double, lit_float, lit_long,
        lit_string, max, min, sum,
    };
}
