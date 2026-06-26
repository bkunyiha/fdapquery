//! # datatypes
//!
//! Core data types — the substrate every other crate builds on.
//!
//! ## Design
//! - `ColumnVector` is a trait; concrete impls wrap arrow-rs arrays.
//! - `Schema` and `Field` are `#[derive(Clone, Debug, PartialEq)]` structs.
//! - [`ScalarValue`] is a typed enum used by physical operators in place of an
//!   `Any?`-style erased value.
//! - Concurrency uses Rayon. `RecordBatch` is the arrow-rs type, re-exported
//!   here. Vector building goes through [`ArrowVectorBuilder`].

// ==============================================================
// Per-file modules.
// ==============================================================
pub mod arrow_field_vector;
// Session 15d-1 #95 — `arrow_types` constants module removed.
// Consumers use `arrow_schema::DataType::Float64` etc. directly, matching
// DataFusion's pattern. The orphaned `arrow_types.rs` file on disk should
// be `rm`ed in a follow-up (sandbox couldn't delete due to permissions).
pub mod arrow_vector_builder;
pub mod column_vector;
pub mod literal_value_vector;
pub mod record_batch;
// Session 15d-1 #108 — `ScalarValue` moved to `fdapquery-common`. The
// re-export below keeps existing `use fdapquery_datatypes::ScalarValue`
// (single + multi-import) call sites working as a transitional
// forwarding; canonical path is `fdapquery_common::ScalarValue`.
pub mod schema;
pub mod shuffle_id;
pub mod shuffle_location;

// ==============================================================
// Re-exports for convenient downstream `use datatypes::*;` ergonomics.
// ==============================================================
pub use fdapquery_common::{FdapQueryError, Result};

pub use arrow_field_vector::ArrowFieldVector;
pub use arrow_vector_builder::ArrowVectorBuilder;
pub use column_vector::ColumnVector;
pub use literal_value_vector::LiteralValueVector;
pub use record_batch::RecordBatch;
// Session 15d-1 #94 — promote `to_csv` so consumers can write
// `use fdapquery_datatypes::to_csv;` instead of reaching through
// `record_batch::to_csv`.
pub use record_batch::to_csv;
// Session 15d-1 #108 — transitional re-export; canonical home is
// `fdapquery_common::ScalarValue`.
pub use fdapquery_common::ScalarValue;
// `Schema`/`Field` are direct arrow re-exports — no extension traits,
// no SchemaConverter shims. Matches DataFusion exactly.
pub use schema::{Field, Schema, SchemaRef};
pub use shuffle_id::ShuffleId;
pub use shuffle_location::ShuffleLocation;
