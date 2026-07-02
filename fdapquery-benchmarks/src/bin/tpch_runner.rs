//! Runs an arbitrary TPC-H SQL query against a directory of TPC-H Parquet
//! files.
//!
//! ## Usage
//!
//! ```text
//! cargo run --release --bin tpch_runner -- <query.sql> <tpch_data_dir>
//! ```
//!
//! `<query.sql>` is a path to a SQL file (e.g. `benchmarks/queries/q1.sql`).
//! `<tpch_data_dir>` must contain the eight TPC-H tables as Parquet files:
//! `customer.parquet`, `lineitem.parquet`, `nation.parquet`, `orders.parquet`,
//! `part.parquet`, `partsupp.parquet`, `region.parquet`, `supplier.parquet`.
//!
//! With the `LiteralDate` lowering in place (`query-planner` →
//! `chrono::NaiveDate` → days-since-epoch), Q1's
//! `date '1998-12-01' - interval '68 days'` predicate plans correctly
//! through the engine. Whether it executes end-to-end depends on
//! `DateSubtractIntervalExpr` at the physical layer.

use std::collections::HashMap;
use std::fs;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

use fdapquery::SessionContext;
use fdapquery_catalog::ParquetDataSource;
use fdapquery_catalog::TableProvider;
use fdapquery_datatypes::RecordBatch;
use fdapquery_datatypes::record_batch::to_csv;
use futures::TryStreamExt;

/// The eight TPC-H tables.
const TPCH_TABLES: &[&str] = &[
    "customer", "lineitem", "nation", "orders", "part", "partsupp", "region", "supplier",
];

#[tokio::main]
async fn main() -> ExitCode {
    env_logger::init();

    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 2 {
        eprintln!("Usage: tpch_runner <sql-file> <data-dir>");
        eprintln!();
        eprintln!("Arguments:");
        eprintln!("  sql-file  Path to SQL file containing the query");
        eprintln!("  data-dir  Path to directory containing TPC-H parquet files");
        return ExitCode::from(1);
    }
    let sql_file = &args[0];
    let data_dir = &args[1];

    // Read the SQL query from the file.
    let sql = fs::read_to_string(sql_file)
        .unwrap_or_else(|e| panic!("cannot read SQL file '{sql_file}': {e}"));
    println!("Executing query from {sql_file}:");
    println!("{sql}");
    println!();

    // Register the eight TPC-H tables as ParquetDataSource scans.
    let mut ctx = SessionContext::new(HashMap::new());
    for table in TPCH_TABLES {
        let path = format!("{data_dir}/{table}.parquet");
        let source: Arc<dyn TableProvider> = Arc::new(ParquetDataSource::new(path));
        ctx.register_data_source(table, source);
    }

    // Execute and time via `Instant::now()` + `elapsed()`.
    let df = ctx.sql(&sql).expect("tpch_runner: sql plan");
    let start = Instant::now();
    let stream = ctx
        .execute_data_frame(&df)
        .await
        .expect("tpch_runner: execute");
    let batches: Vec<RecordBatch> = stream
        .try_collect()
        .await
        .expect("tpch_runner: drain stream");
    for batch in batches {
        // Same shape as `nyc_taxi`: log schema at INFO, print CSV row data
        // unconditionally. Mirrors DataFusion tpch/run.rs's pattern where
        // pretty-formatted result output rides on `log::info!` and the
        // primary result output rides on `println!`.
        log::info!("output schema: {:?}", batch.schema());
        print!(
            "{}",
            to_csv(&batch).expect("tpch_runner: to_csv over result batch")
        );
    }
    let time = start.elapsed().as_millis();

    println!();
    println!("Query executed in {time} ms");
    ExitCode::SUCCESS
}
