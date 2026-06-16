//! Random `LogicalPlan` / `RecordBatch` / `LogicalExpr` generator for differential
//! testing. A `Fuzzer` is seeded deterministically so the same Rust run reproduces
//! the same sequence of batches and plans.
//!
//! ## Notes
//! - **RNG.** `rand::rngs::StdRng::seed_from_u64(0)` (workspace `rand` dep). The
//!   RNG is deterministic across Rust runs; the fuzzer is used for differential
//!   testing inside Rust.
//! - **Value type.** Random scalars are typed as [`ScalarValue`].
//!   `create_values` returns `Vec<ScalarValue>` and the `columns` parameter
//!   of the batch constructors is `Vec<Vec<ScalarValue>>`.
//! - **Two batch constructors.** Rust can't overload by argument type, so the
//!   two batch constructors carry distinct names:
//!   - [`Fuzzer::create_random_record_batch`] generates `n` rows from scratch.
//!   - [`Fuzzer::create_record_batch`] builds a batch from explicit per-column
//!     data. The shorter name goes to the from-columns version because the
//!     dependent tests call it most often.
//! - **One RNG, two access modes.** `Fuzzer` holds a single [`EnhancedRandom`]
//!   that owns the underlying `StdRng` and exposes both the biased helpers
//!   (`next_byte`, `next_double`, …) and a raw [`EnhancedRandom::rng`] accessor
//!   for the call sites that need `gen_range(..)` directly.
//! - **Per-type column builders.** [`datatypes::ArrowVectorBuilder::append_value`]
//!   performs per-type variant dispatch internally (see `arrow_vector_builder.rs`),
//!   so the batch-construction loop is one line per column.
//! - **`self.rng.random::<T>()` / `random_range(..)`.** As of rand 0.9 the
//!   value/range helpers are named `random()` and `random_range()` (the old
//!   `gen()` / `gen_range()` names — `gen` being a reserved keyword in the
//!   Rust 2024 edition — are gone). In rand 0.10 these helpers live on the
//!   [`rand::RngExt`] extension trait rather than `Rng`, so that is what the
//!   call sites import.

use datatypes::{ArrowVectorBuilder, ColumnVector, RecordBatch, ScalarValue, Schema, record_batch};
use logical_plan::{DataFrame, LogicalExpr};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

use arrow_schema::DataType;

/// Character pool for [`EnhancedRandom::next_string`] — `a-z`, `A-Z`, `0-9`.
const CHAR_POOL: &[u8] = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";

// ---------------------------------------------------------------------------
// Fuzzer
// ---------------------------------------------------------------------------

/// Random generator for column values, record batches, expressions, and plans.
pub struct Fuzzer {
    rng: EnhancedRandom,
}

impl Default for Fuzzer {
    fn default() -> Self {
        Self::new()
    }
}

impl Fuzzer {
    /// Construct a fuzzer seeded with `0`. Deterministic across Rust runs.
    pub fn new() -> Self {
        Self {
            rng: EnhancedRandom::new(0),
        }
    }

    /// Generate `n` random scalar values of the given Arrow type. Panics on
    /// unsupported types.
    pub fn create_values(&mut self, data_type: &DataType, n: usize) -> Vec<ScalarValue> {
        (0..n)
            .map(|_| match data_type {
                DataType::Int8 => ScalarValue::Int8(self.rng.next_byte()),
                DataType::Int16 => ScalarValue::Int16(self.rng.next_short()),
                DataType::Int32 => ScalarValue::Int32(self.rng.next_int()),
                DataType::Int64 => ScalarValue::Int64(self.rng.next_long()),
                DataType::Float32 => ScalarValue::Float32(self.rng.next_float()),
                DataType::Float64 => ScalarValue::Float64(self.rng.next_double()),
                DataType::Utf8 => {
                    let len = self.rng.rng().random_range(0..64);
                    ScalarValue::Utf8(self.rng.next_string(len))
                }
                other => panic!("Fuzzer::create_values: unsupported data type: {other:?}"),
            })
            .collect()
    }

    /// Build a `RecordBatch` containing `n` rows of random data for every field
    /// in `schema`. See the module note for why this name differs from
    /// [`Self::create_record_batch`].
    pub fn create_random_record_batch(&mut self, schema: &Schema, n: usize) -> RecordBatch {
        let columns: Vec<Vec<ScalarValue>> = schema
            .fields
            .iter()
            .map(|f| self.create_values(&f.data_type, n))
            .collect();
        self.create_record_batch(schema, columns)
    }

    /// Build a `RecordBatch` from explicit per-column data. Panics if `columns`
    /// is empty or if any column's length disagrees with the first column's
    /// length (`columns[0].len()` is taken as the row count).
    pub fn create_record_batch(
        &self,
        schema: &Schema,
        columns: Vec<Vec<ScalarValue>>,
    ) -> RecordBatch {
        assert!(
            !columns.is_empty(),
            "Fuzzer::create_record_batch: columns is empty"
        );
        let row_count = columns[0].len();
        let field_vectors: Vec<Box<dyn ColumnVector>> = schema
            .fields
            .iter()
            .zip(columns)
            .map(|(field, col)| {
                assert_eq!(
                    col.len(),
                    row_count,
                    "Fuzzer::create_record_batch: column length mismatch \
                     ({} != {row_count}) for field {:?}",
                    col.len(),
                    field.name,
                );
                let mut builder = ArrowVectorBuilder::new(&field.data_type, row_count);
                for v in &col {
                    builder.append_value(v);
                }
                builder.set_value_count(row_count);
                Box::new(builder.build()) as Box<dyn ColumnVector>
            })
            .collect();
        record_batch::create(schema, field_vectors)
    }

    /// Recursively build a random logical plan tree of `project` / `filter`
    /// operators on top of `input`.
    pub fn create_plan(
        &mut self,
        input: &DataFrame,
        depth: usize,
        max_depth: usize,
        max_expr_depth: usize,
    ) -> DataFrame {
        if depth == max_depth {
            // `DataFrame` wraps an owned `LogicalPlan`; clone so the caller
            // gets a fresh tree.
            return input.clone();
        }
        // Build the child plan first, then layer either a projection or a
        // filter on top of it.
        let child = self.create_plan(input, depth + 1, max_depth, max_expr_depth);
        match self.rng.rng().random_range(0..2) {
            0 => {
                let expr_count = self.rng.rng().random_range(1..5);
                let exprs: Vec<LogicalExpr> = (0..expr_count)
                    .map(|_| self.create_expression(&child, 0, max_expr_depth))
                    .collect();
                child.project(exprs)
            }
            _ => {
                // Note: the filter predicate is generated against `input`'s
                // schema (not `child`'s), so column indices reference the
                // original schema. This is intentional and matches the
                // historical generator behaviour.
                let pred = self.create_expression(input, 0, max_expr_depth);
                child.filter(pred)
            }
        }
    }

    /// Recursively build a random binary expression tree. Leaves at
    /// `depth == max_depth` are one of `ColumnIndex` / `LiteralDouble` /
    /// `LiteralLong` / `LiteralString`; internal nodes are one of the eight
    /// binary operators (`Eq` / `Neq` / `Lt` / `LtEq` / `Gt` / `GtEq` / `And` /
    /// `Or`).
    pub fn create_expression(
        &mut self,
        input: &DataFrame,
        depth: usize,
        max_depth: usize,
    ) -> LogicalExpr {
        if depth == max_depth {
            // Leaf node: pick a random literal or column reference.
            let fields_len = input.schema().fields.len();
            return match self.rng.rng().random_range(0..4) {
                0 => LogicalExpr::ColumnIndex(self.rng.rng().random_range(0..fields_len)),
                1 => LogicalExpr::LiteralDouble(self.rng.next_double()),
                2 => LogicalExpr::LiteralLong(self.rng.next_long()),
                _ => {
                    let len = self.rng.rng().random_range(0..64);
                    LogicalExpr::LiteralString(self.rng.next_string(len))
                }
            };
        }
        // Internal node: binary op over two recursive subtrees.
        let l = Box::new(self.create_expression(input, depth + 1, max_depth));
        let r = Box::new(self.create_expression(input, depth + 1, max_depth));
        match self.rng.rng().random_range(0..8) {
            0 => LogicalExpr::Eq { l, r },
            1 => LogicalExpr::Neq { l, r },
            2 => LogicalExpr::Lt { l, r },
            3 => LogicalExpr::LtEq { l, r },
            4 => LogicalExpr::Gt { l, r },
            5 => LogicalExpr::GtEq { l, r },
            6 => LogicalExpr::And { l, r },
            _ => LogicalExpr::Or { l, r },
        }
    }
}

// ---------------------------------------------------------------------------
// EnhancedRandom
// ---------------------------------------------------------------------------

/// RNG that biases toward edge-case values for each integer / floating-point
/// type (`MIN` / `MAX` / `0` / `±Inf` / `NaN`).
///
/// Owns the underlying [`StdRng`] so that `Fuzzer` keeps a single source of
/// randomness; callers that need raw `random_range(..)`-style calls reach through
/// [`Self::rng`].
pub struct EnhancedRandom {
    rng: StdRng,
}

impl EnhancedRandom {
    /// Construct seeded with the given value.
    pub fn new(seed: u64) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// Borrow the underlying RNG for direct `random_range(..)` / `random()` calls.
    /// See the module note for why both biased and raw APIs share one field.
    pub fn rng(&mut self) -> &mut StdRng {
        &mut self.rng
    }

    /// Random `i8` biased toward extremes.
    pub fn next_byte(&mut self) -> i8 {
        match self.rng.random_range(0..5) {
            0 => i8::MIN,
            1 => i8::MAX,
            // Two arms collapse to `0` to bias output toward zero.
            2 | 3 => 0,
            _ => self.rng.random::<i32>() as i8,
        }
    }

    /// Random `i16` biased toward extremes.
    pub fn next_short(&mut self) -> i16 {
        match self.rng.random_range(0..5) {
            0 => i16::MIN,
            1 => i16::MAX,
            2 | 3 => 0,
            _ => self.rng.random::<i32>() as i16,
        }
    }

    /// Random `i32` biased toward extremes.
    pub fn next_int(&mut self) -> i32 {
        match self.rng.random_range(0..5) {
            0 => i32::MIN,
            1 => i32::MAX,
            2 | 3 => 0,
            _ => self.rng.random::<i32>(),
        }
    }

    /// Random `i64` biased toward extremes.
    pub fn next_long(&mut self) -> i64 {
        match self.rng.random_range(0..5) {
            0 => i64::MIN,
            1 => i64::MAX,
            2 | 3 => 0,
            _ => self.rng.random::<i64>(),
        }
    }

    /// Random `f32` biased toward extremes including `±Inf` and `NaN`. Uses
    /// `f32::MIN_POSITIVE` (smallest positive value), not `f32::MIN` (most
    /// negative).
    pub fn next_float(&mut self) -> f32 {
        match self.rng.random_range(0..8) {
            0 => f32::MIN_POSITIVE,
            1 => f32::MAX,
            2 => f32::INFINITY,
            3 => f32::NEG_INFINITY,
            4 => f32::NAN,
            5 => -0.0_f32,
            6 => 0.0_f32,
            _ => self.rng.random::<f32>(),
        }
    }

    /// Random `f64` biased toward extremes including `±Inf` and `NaN`. See the
    /// note on `next_float` regarding `MIN_POSITIVE`.
    pub fn next_double(&mut self) -> f64 {
        match self.rng.random_range(0..8) {
            0 => f64::MIN_POSITIVE,
            1 => f64::MAX,
            2 => f64::INFINITY,
            3 => f64::NEG_INFINITY,
            4 => f64::NAN,
            5 => -0.0_f64,
            6 => 0.0_f64,
            _ => self.rng.random::<f64>(),
        }
    }

    /// Random ASCII alphanumeric string of length `len`. `len == 0` returns
    /// the empty string. Bytes are pulled from [`CHAR_POOL`] (`a-z` + `A-Z` +
    /// `0-9`, 62 chars).
    pub fn next_string(&mut self, len: usize) -> String {
        (0..len)
            .map(|_| CHAR_POOL[self.rng.random_range(0..CHAR_POOL.len())] as char)
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    //! The test loops 50 times and asserts that `create_plan` returns without
    //! panicking — i.e. random plan generation is stable across runs.
    use super::*;
    use datasource::CsvDataSource;
    use logical_plan::{LogicalPlan, Scan};
    use std::sync::Arc;

    #[test]
    fn fuzzer_example() {
        let path = "../testdata/employee.csv";
        let csv = CsvDataSource::new(path, None, true, 10);
        let input = DataFrame::new(LogicalPlan::Scan(Scan::new(
            "employee.csv",
            Arc::new(csv),
            vec![],
        )));
        let mut fuzzer = Fuzzer::new();
        for _ in 0..50 {
            // `_plan` is discarded; the test only checks that generation does
            // not panic on a depth-6 plan tree with depth-1 expressions.
            let _plan = fuzzer.create_plan(&input, 0, 6, 1);
        }
    }
}
