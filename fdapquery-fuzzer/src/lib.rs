//! # fuzzer
//!
//! Random SQL / logical plan generator. Used for differential testing against
//! a reference engine, and to generate small random `RecordBatch`es for
//! integration tests.

pub mod fuzzer;

// Re-export the public surface so callers can write `fuzzer::Fuzzer` rather
// than `fuzzer::fuzzer::Fuzzer`.
pub use fuzzer::{EnhancedRandom, Fuzzer};
