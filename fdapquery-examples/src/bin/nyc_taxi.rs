//!
//! Reads a single month of the NYC yellow-taxi trip data and runs:
//!
//! ```sql
//! SELECT passenger_count, MAX(CAST(fare_amount AS float))
//!   FROM tripdata
//!   GROUP BY passenger_count
//! ```
//!
//! Prints the logical plan, the optimized plan, every result batch, and the
//! wall-clock time. Float formatting follows Rust's default `f32::to_string`.
//!
//! ## Where the input file lives
//! The path is **hardcoded** to a specific 2019-01 yellow-taxi file. Obtain
//! the file once with:
//!
//! ```text
//! wget https://s3.amazonaws.com/nyc-tlc/trip+data/yellow_tripdata_2019-01.csv
//! ```
//!
//! and place / symlink it at the path below. Without the file, the binary
//! panics with the `CsvDataSource` "file not found" error.

use std::collections::HashMap;
use std::time::Instant;

use fdapquery::SessionContext;
use fdapquery_datatypes::RecordBatch;
use fdapquery_datatypes::record_batch::to_csv;
use fdapquery_expr::{cast, col, format, max};
use fdapquery_optimizer::Optimizer;
use futures::TryStreamExt;

/// Hardcoded NYC yellow-taxi 2019-01 path; see the module-doc for how to
/// obtain the file.
const NYC_TAXI_CSV: &str = "/mnt/nyctaxi/csv/year=2019/yellow_tripdata_2019-01.csv";

#[tokio::main]
async fn main() {
    env_logger::init();

    let ctx = SessionContext::new(HashMap::new());

    let start = Instant::now();

    // SELECT passenger_count, MAX(CAST(fare_amount AS float)) GROUP BY passenger_count
    let df = ctx.csv(NYC_TAXI_CSV).aggregate(
        vec![col("passenger_count")],
        vec![max(cast(
            col("fare_amount"),
            arrow_schema::DataType::Float32,
        ))],
    );

    println!("Logical Plan:\t{}", format(df.logical_plan()));

    // Print the optimized plan separately so a reader can see what
    // `ProjectionPushDown` (and other rules) do to the logical tree.
    // `SessionContext::execute()` will re-run `Optimizer::optimize` internally;
    // the optimizer is idempotent, so the second pass is a no-op shape-wise.
    let optimized_plan = Optimizer::new()
        .optimize(df.logical_plan())
        .expect("nyc_taxi: optimize");
    println!("Optimized Plan:\t{}", format(&optimized_plan));

    let stream = ctx
        .execute(df.logical_plan())
        .await
        .expect("nyc_taxi: execute");
    let batches: Vec<RecordBatch> = stream.try_collect().await.expect("nyc_taxi: drain stream");
    for batch in batches {
        // Print each batch's schema (arrow-rs `Schema`'s `Debug` form) and
        // its CSV rendering.
        println!("{:?}", batch.schema());
        println!(
            "{}",
            to_csv(&batch).expect("nyc_taxi: to_csv over result batch")
        );
    }

    println!("Query took {} ms", start.elapsed().as_millis());
}
