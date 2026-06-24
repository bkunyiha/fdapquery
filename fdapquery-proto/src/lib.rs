//! # protobuf
//!
//! Protocol Buffers schemas + serialise / deserialise helpers for
//! `LogicalPlan`, `LogicalExpr`, `PhysicalPlan`, `Expression`,
//! `AggregateExpression`, `Action`, `Schema`, `Field`, `ShuffleLocation`, and
//! `Task`.
//!
//! Serialisers and deserialisers are DataFusion-style free
//! `serialize_*` / `deserialize_*` functions (see each file's module doc for
//! the rationale). Wire format used by the `flight-server`, `client`, and
//! `distributed` crates.
//!
//! The `.proto` schema lives at `../proto/rquery.proto`. Code generation runs
//! at build time via `build.rs` using `tonic-build` (which wraps `prost-build`)
//! and **requires `protoc` to be installed** on the host:
//!
//! ```text
//! brew install protobuf         # macOS
//! apt install protobuf-compiler # Debian/Ubuntu
//! ```
//!
//! The generated Rust types land in `OUT_DIR` and are re-exposed through the
//! `pb` module below.

// =============================================================================
// Generated prost types from `../proto/rquery.proto`.
//
// The `.proto`'s `package rquery.protobuf;` line tells prost to emit a single
// file named `rquery.protobuf.rs` in `OUT_DIR`. We include it under the local
// `pb` module so the rest of the crate writes `pb::LogicalPlanNode` rather than
// the longer qualified `rquery.protobuf.LogicalPlanNode`.
// =============================================================================
pub mod pb {
    include!(concat!(env!("OUT_DIR"), "/rquery.protobuf.rs"));
}

// =============================================================================
// Per-file modules.
// =============================================================================
pub mod physical_plan_deserializer;
pub mod physical_plan_serializer;
pub mod protobuf_deserializer;
pub mod protobuf_serializer;

// =============================================================================
// Re-exports for ergonomic `use protobuf::*;`.
// =============================================================================
// Physical-plan ser/de: free functions per DataFusion convention (see
// file-level docs). Serializer-side leaf conversions
// (`Schema`/`Field`/`ShuffleLocation`) live in `physical_plan_serializer.rs`
// as `impl From<&T> for pb::T` and are used at call sites as `.into()` — no
// explicit re-export needed. Deserializer-side leaf conversions stay as
// free functions because the orphan rule rejects `impl From<&pb::T> for T`
// when `T` is in a foreign crate.
pub use physical_plan_deserializer::{
    deserialize_physical_aggr_expr, deserialize_physical_expr, deserialize_physical_plan,
    deserialize_shuffle_location, deserialize_task,
};
pub use physical_plan_serializer::{
    serialize_physical_aggr_expr, serialize_physical_expr, serialize_physical_plan, serialize_task,
};
// Logical-plan ser/de: free functions per DataFusion convention (see file-level docs).
pub use protobuf_deserializer::{
    deserialize_action, deserialize_field, deserialize_logical_expr, deserialize_logical_plan,
    deserialize_schema,
};
pub use protobuf_serializer::{
    serialize_logical_aggregate_expr, serialize_logical_expr, serialize_logical_plan,
};
