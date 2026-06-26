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

// Per-crate module re-exports — matches DataFusion's pattern.
pub use fdapquery_catalog as catalog;
pub use fdapquery_common as common;
pub use fdapquery_datatypes as datatypes;
pub use fdapquery_execution as execution;
pub use fdapquery_expr as expr;
pub use fdapquery_functions as functions;
pub use fdapquery_functions_aggregate as functions_aggregate;
pub use fdapquery_optimizer as optimizer;
pub use fdapquery_physical_expr as physical_expr;
pub use fdapquery_physical_plan as physical_plan;
pub use fdapquery_sql as sql;

// Top-level convenience re-exports — the most common types.
pub use fdapquery_catalog::{CsvDataSource, InMemoryDataSource, ParquetDataSource, TableProvider};
pub use fdapquery_datatypes::{FdapQueryError, Field, RecordBatch, Result, ScalarValue, Schema};
pub use fdapquery_execution::{ExecutionContext, ParallelContext};
pub use fdapquery_expr::{DataFrame, LogicalPlan};
pub use fdapquery_physical_plan::{ExecutionPlan, QueryPlanner, SendableRecordBatchStream};

pub mod prelude {
    //! Conventional `use fdapquery::prelude::*;` import surface.
    //! Includes everything a consumer needs to build, register
    //! tables on, and execute a query against an `ExecutionContext`
    //! or `ParallelContext`. Mirrors `datafusion::prelude`.

    pub use crate::{
        CsvDataSource, DataFrame, ExecutionContext, ExecutionPlan, FdapQueryError, Field,
        InMemoryDataSource, LogicalPlan, ParallelContext, ParquetDataSource, QueryPlanner,
        RecordBatch, Result, ScalarValue, Schema, SendableRecordBatchStream, TableProvider,
    };

    // The DataFrame-building DSL free functions — `col(...)`,
    // `lit_*(...)`, and the aggregate builders `sum`/`min`/`max`/
    // `avg`/`count`/`count_distinct` plus the `cast` cast-builder.
    pub use fdapquery_expr::{
        avg, cast, col, count, count_distinct, lit_date, lit_double, lit_float, lit_long,
        lit_string, max, min, sum,
    };
}
