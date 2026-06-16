//! Fans out 12 monthly CSV queries — one per month of the NYC yellow-taxi 2019
//! data — runs them in parallel, then re-aggregates the 12 result vectors into
//! a single final result. The fan-out uses **rayon**.
//!
//! ## Where the input files live
//! The directory path is **hardcoded**. Twelve files are expected at
//! `${PATH}/yellow_tripdata_2019-{01..12}.csv`. Without them, the per-month
//! query panics inside `CsvDataSource`.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use datasource::InMemoryDataSource;
use datatypes::record_batch::to_csv;
use datatypes::{RecordBatch, SchemaConverter};
use execution::ExecutionContext;
use rayon::prelude::*;

/// Hardcoded directory holding `yellow_tripdata_2019-{01..12}.csv`.
const PATH: &str = "/mnt/nyctaxi/csv/yellow/2019";

/// SQL each per-month worker runs. The aliased `max_fare` flows into the final
/// re-aggregation below.
const PER_MONTH_SQL: &str = "SELECT passenger_count, \
                             MAX(CAST(fare_amount AS double)) AS max_fare \
                             FROM tripdata \
                             GROUP BY passenger_count";

/// SQL for the final re-aggregation over the 12 per-month partial results.
const FINAL_SQL: &str = "SELECT passenger_count, MAX(max_fare) \
                         FROM tripdata GROUP BY passenger_count";

fn main() {
    env_logger::init();

    let start = Instant::now();

    // -----------------------------------------------------------------------
    // Fan-out: 12 months × per-month query, run in parallel via rayon
    // (see ARCHITECTURE §3.9).
    // -----------------------------------------------------------------------
    let results: Vec<RecordBatch> = (1u32..=12)
        .into_par_iter()
        .flat_map(|month| {
            let part_start = Instant::now();
            let batches = execute_query(PATH, month, PER_MONTH_SQL);
            println!(
                "Query against month {month} took {} ms",
                part_start.elapsed().as_millis()
            );
            batches
        })
        .collect();

    let duration = start.elapsed().as_millis();
    println!("Collected {} batches in {duration} ms", results.len());

    let first = results.first().expect("no result batches collected");
    println!("{:?}", first.schema());

    // -----------------------------------------------------------------------
    // Re-aggregate the 12 per-month partials. Register the collected batches
    // as an InMemoryDataSource and run the FINAL_SQL through a fresh context.
    //
    // `RecordBatch::schema()` returns an `Arc<arrow_schema::Schema>`;
    // `InMemoryDataSource::new` wants a `datatypes::Schema`, so we round-trip
    // through `SchemaConverter::from_arrow`.
    // -----------------------------------------------------------------------
    let final_schema = SchemaConverter::from_arrow(&first.schema());
    let in_memory: Arc<dyn datasource::DataSource> =
        Arc::new(InMemoryDataSource::new(final_schema, results));

    let mut ctx = ExecutionContext::new(HashMap::new());
    ctx.register_data_source("tripdata", in_memory);

    let df = ctx.sql(FINAL_SQL);
    for batch in ctx.execute_data_frame(&df) {
        // `println!("{batch:?}")` would dump arrow-rs's verbose Debug;
        // `to_csv` gives a more readable row-per-line view.
        print!("{}", to_csv(&batch));
    }
}

/// Per-month worker. Builds a fresh `ExecutionContext`, registers
/// `yellow_tripdata_2019-{MM}.csv` under the table name `tripdata`, runs `sql`,
/// and returns the collected batches.
///
/// Each worker owns its own table registry and CSV reader — no shared
/// mutable state across rayon workers, which keeps the closure trivially
/// `Send` without any wrapping.
fn execute_query(path: &str, month: u32, sql: &str) -> Vec<RecordBatch> {
    let filename = format!("{path}/yellow_tripdata_2019-{month:02}.csv");
    let mut ctx = ExecutionContext::new(HashMap::new());
    ctx.register_csv("tripdata", &filename);
    let df = ctx.sql(sql);
    ctx.execute_data_frame(&df).collect()
}
