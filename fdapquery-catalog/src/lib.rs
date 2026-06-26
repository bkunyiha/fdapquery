//! # datasource
//!
//! `DataSource` trait and concrete implementations: CSV, Parquet, and an
//! in-memory variant.
//!
//! ## Design
//! - `DataSource` is a trait; concrete impls are zero-state structs that hold
//!   paths/handles.
//! - `CsvDataSource` uses arrow-rs's `arrow::csv::ReaderBuilder` for parsing.
//! - `ParquetDataSource` uses arrow-rs's
//!   `parquet::arrow::ParquetRecordBatchReaderBuilder` for reading.
//! - Schema inference in CSV uses arrow-rs's typed inference (Int64, Float64,
//!   Boolean, Utf8).

// ==============================================================
// Per-file modules.
// ==============================================================
pub mod csv_data_source;
pub mod in_memory_data_source;
pub mod parquet_data_source;
// `TableProvider` (Session 13b) — replaces the pre-13b `DataSource`
// trait. Lives here, not in `fdapquery-physical-plan`, to avoid a
// catalog → physical-plan → expr → catalog cycle.
pub mod table_provider;

// ==============================================================
// Re-exports for convenient downstream `use catalog::*;` ergonomics.
// ==============================================================
pub use csv_data_source::CsvDataSource;
pub use in_memory_data_source::InMemoryDataSource;
pub use parquet_data_source::ParquetDataSource;
pub use table_provider::{SendableRecordBatchStream, TableProvider};
