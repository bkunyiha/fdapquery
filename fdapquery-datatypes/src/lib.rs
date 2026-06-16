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
pub mod arrow_types;
pub mod arrow_vector_builder;
pub mod column_vector;
pub mod literal_value_vector;
pub mod record_batch;
pub mod scalar_value;
pub mod schema;
pub mod shuffle_id;
pub mod shuffle_location;

// ==============================================================
// Re-exports for convenient downstream `use datatypes::*;` ergonomics.
// ==============================================================
pub use arrow_field_vector::ArrowFieldVector;
pub use arrow_vector_builder::ArrowVectorBuilder;
pub use column_vector::ColumnVector;
pub use literal_value_vector::LiteralValueVector;
pub use record_batch::RecordBatch;
pub use scalar_value::ScalarValue;
pub use schema::{Field, Schema, SchemaConverter};
pub use shuffle_id::ShuffleId;
pub use shuffle_location::ShuffleLocation;
