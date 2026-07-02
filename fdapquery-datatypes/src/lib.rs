//! # datatypes
//!
//! Core data types — the substrate every other crate builds on.
//!
//! ## Design
//! - `RecordBatch` is the arrow-rs type, re-exported here.
//! - `Schema` and `Field` are direct arrow-rs re-exports (no wrapper structs).
//! - `ScalarValue` lives in `fdapquery_common` (mirrors
//!   `datafusion_common::ScalarValue`); consumers import it from there.
//! - `ArrowVectorBuilder` lives in `fdapquery_common` alongside `ScalarValue`
//!
//! The `ColumnVector` trait and its rquery-DNA wrappers
//! (`ArrowFieldVector`, `LiteralValueVector`) were dropped. Code that used to
//! traffic in `Box<dyn ColumnVector>` now uses `arrow_array::ArrayRef`
//! directly. `PhysicalExpr::evaluate` now returns
//! `fdapquery_physical_expr::ColumnarValue` (`Array | Scalar`), mirroring
//! DataFusion's `ColumnarValue`.

// ==============================================================
// Per-file modules.
// ==============================================================
pub mod record_batch;
pub mod schema;
pub mod shuffle_id;
pub mod shuffle_location;

// ==============================================================
// Re-exports for convenient downstream `use datatypes::*;` ergonomics.
// ==============================================================
pub use fdapquery_common::{FdapQueryError, Result};

pub use record_batch::RecordBatch;
// Promote `to_csv` so consumers can write
// `use fdapquery_datatypes::to_csv;` instead of reaching through
// `record_batch::to_csv`.
pub use record_batch::to_csv;
// `Schema`/`Field` are direct arrow re-exports — no extension traits,
// no SchemaConverter shims. Matches DataFusion exactly.
pub use schema::{Field, Schema, SchemaRef};
pub use shuffle_id::ShuffleId;
pub use shuffle_location::ShuffleLocation;
