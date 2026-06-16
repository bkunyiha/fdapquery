//!
//! Runs the same `SELECT state, SUM(CAST(salary AS double)) FROM employee
//! GROUP BY state` query two ways — through a sequential `ExecutionContext`
//! and through a 4-worker `ParallelContext` — then verifies the two outputs
//! match by comparing them as `state → sum` maps. Prints each side's wall-clock
//! timing.
//!
//! Operates on the in-repo `testdata/employee.csv` (no external download
//! required, unlike `nyc_taxi`), so this binary is the friendliest one to run
//! to see the engine end-to-end.

use std::collections::HashMap;
use std::time::Instant;

use datatypes::{ArrowFieldVector, ColumnVector, RecordBatch, ScalarValue};
use execution::{ExecutionContext, ParallelContext};

/// In-repo employee fixture used by the existing execution-module tests.
const EMPLOYEE_CSV: &str = "../testdata/employee.csv";

fn main() {
    env_logger::init();

    let sql = "SELECT state, SUM(CAST(salary AS double)) FROM employee GROUP BY state";

    println!("=== Parallel Execution Example ===\n");
    println!("Query: {sql}\n");

    // ---- Sequential execution ----
    println!("--- Sequential Execution ---");
    let mut seq_ctx = ExecutionContext::new(HashMap::new());
    seq_ctx.register_csv("employee", EMPLOYEE_CSV);
    let seq_df = seq_ctx.sql(sql);
    let seq_start = Instant::now();
    let seq_results: Vec<RecordBatch> = seq_ctx.execute_data_frame(&seq_df).collect();
    let seq_time = seq_start.elapsed().as_millis();
    println!("Sequential execution completed in {seq_time}ms");
    print_results(&seq_results);
    println!();

    // ---- Parallel execution (4 workers) ----
    println!("--- Parallel Execution (4 workers) ---");
    let mut par_ctx = ParallelContext::with_parallelism(4, HashMap::new());
    par_ctx.register_csv("employee", EMPLOYEE_CSV);
    let par_df = par_ctx.sql(sql);
    let par_start = Instant::now();
    let par_results: Vec<RecordBatch> = par_ctx.execute_data_frame(&par_df).collect();
    let par_time = par_start.elapsed().as_millis();
    println!("Parallel execution completed in {par_time}ms");
    print_results(&par_results);
    println!();

    // ---- Verification: same (state, sum) set both ways ----
    println!("--- Verification ---");
    let seq_map = extract_results(&seq_results);
    let par_map = extract_results(&par_results);
    if seq_map == par_map {
        println!("Results match between sequential and parallel execution");
    } else {
        println!("WARNING: Results differ!");
        println!("Sequential: {seq_map:?}");
        println!("Parallel:   {par_map:?}");
    }

    println!("\n=== Example Complete ===");
}

/// Print every `(state, sum)` row in the result batches.
fn print_results(batches: &[RecordBatch]) {
    for batch in batches {
        // Wrap each column once with ArrowFieldVector so we can use the
        // `ColumnVector::get_value(row)` API — same pattern `to_csv` uses.
        let state_col = ArrowFieldVector::new(batch.column(0).clone());
        let sum_col = ArrowFieldVector::new(batch.column(1).clone());
        for row in 0..batch.num_rows() {
            let key = scalar_to_string(&state_col.get_value(row));
            let value = sum_col.get_value(row);
            println!("  {key}: {value:?}");
        }
    }
}

/// Collect `(state, sum)` pairs into a `HashMap` for set-equality
/// comparison — the aggregate's row order is non-deterministic
/// (HashMap-driven), so the only sensible equality is "same set of
/// `(key, value)` pairs."
fn extract_results(batches: &[RecordBatch]) -> HashMap<String, ScalarValue> {
    let mut out = HashMap::new();
    for batch in batches {
        let state_col = ArrowFieldVector::new(batch.column(0).clone());
        let sum_col = ArrowFieldVector::new(batch.column(1).clone());
        for row in 0..batch.num_rows() {
            let key = scalar_to_string(&state_col.get_value(row));
            let value = sum_col.get_value(row);
            out.insert(key, value);
        }
    }
    out
}

/// Stringify a `ScalarValue` for use as a `HashMap` key. `Utf8` borrows
/// its `String`, `Binary` decodes the bytes, `Null` renders as the literal
/// `"null"` (matches `to_csv`'s null rendering).
fn scalar_to_string(v: &ScalarValue) -> String {
    match v {
        ScalarValue::Utf8(s) => s.clone(),
        ScalarValue::Binary(b) => String::from_utf8_lossy(b).into_owned(),
        ScalarValue::Null => "null".to_string(),
        other => format!("{other:?}"),
    }
}
