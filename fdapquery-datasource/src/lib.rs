//! `fdapquery-datasource` — DataFusion-mirror datasource crate.
//!
//! Mirrors `datafusion-datasource`. Hosts the `DataSource` trait (the
//! datasource protocol) and `DataSourceExec` (the operator that wraps
//! a `DataSource` and implements `ExecutionPlan`).
//!
//! Dependency direction: `fdapquery-datasource` depends on
//! `fdapquery-physical-plan` (for `ExecutionPlan`). It does NOT depend
//! on `fdapquery-catalog` — `TableProvider` lives in catalog and
//! catalog can import `DataSourceExec` from here when needed.

pub mod data_source;
pub mod data_source_exec;

pub use data_source::DataSource;
pub use data_source_exec::DataSourceExec;
