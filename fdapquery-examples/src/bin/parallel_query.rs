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

use fdapquery::SessionContext;
use fdapquery_catalog::InMemoryDataSource;
use fdapquery_datatypes::RecordBatch;
use fdapquery_datatypes::record_batch::to_csv;
use futures::TryStreamExt;
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

#[tokio::main]
async fn main() {
    env_logger::init();

    let start = Instant::now();

    // -----------------------------------------------------------------------
    // Fan-out: 12 months × per-month query, run in parallel via rayon
    // (see ARCHITECTURE §3.9). Each rayon worker bridges async-to-sync
    // via `futures::executor::block_on(stream.try_collect())` since
    // rayon threads don't have a tokio runtime — same pattern as
    // `ParallelContext::execute_parallel_aggregate` and
    // `ShuffleWriterExec::write_shuffle`.
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
    // `Schema` IS `arrow_schema::Schema` — `RecordBatch::schema()` returns
    // `Arc<Schema>`, so just clone the inner value.
    // -----------------------------------------------------------------------
    let final_schema = first.schema().as_ref().clone();
    let in_memory: Arc<dyn fdapquery_catalog::TableProvider> =
        Arc::new(InMemoryDataSource::new(final_schema, results));

    let mut ctx = SessionContext::new(HashMap::new());
    ctx.register_data_source("tripdata", in_memory);

    let df = ctx.sql(FINAL_SQL).expect("parallel_query: final sql plan");
    let stream = ctx
        .execute_data_frame(&df)
        .await
        .expect("parallel_query: final execute");
    let batches: Vec<RecordBatch> = stream
        .try_collect()
        .await
        .expect("parallel_query: drain final stream");
    for batch in batches {
        // `println!("{batch:?}")` would dump arrow-rs's verbose Debug;
        // `to_csv` gives a more readable row-per-line view.
        print!(
            "{}",
            to_csv(&batch).expect("parallel_query: to_csv over result batch")
        );
    }
}

/// Per-month worker. Builds a fresh `SessionContext`, registers
/// `yellow_tripdata_2019-{MM}.csv` under the table name `tripdata`, runs
/// `sql`, and drives the async stream to completion via
/// `futures::executor::block_on` — rayon workers don't run on a tokio
/// runtime.
///
/// Each worker owns its own table registry and CSV reader — no shared
/// mutable state across rayon workers, which keeps the closure trivially
/// `Send` without any wrapping.
fn execute_query(path: &str, month: u32, sql: &str) -> Vec<RecordBatch> {
    let filename = format!("{path}/yellow_tripdata_2019-{month:02}.csv");
    let mut ctx = SessionContext::new(HashMap::new());
    ctx.register_csv("tripdata", &filename);
    let df = ctx.sql(sql).expect("parallel_query: per-month sql plan");
    // Rayon workers `block_on` the async planner just like they do the stream.
    let stream = futures::executor::block_on(ctx.execute_data_frame(&df))
        .expect("parallel_query: per-month execute");
    futures::executor::block_on(stream.try_collect())
        .expect("parallel_query: drain per-month stream")
}
