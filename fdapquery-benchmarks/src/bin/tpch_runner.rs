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
//! `DateSubtractIntervalExpression` at the physical layer.

use std::collections::HashMap;
use std::fs;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

use datasource::{DataSource, ParquetDataSource};
use datatypes::RecordBatch;
use datatypes::record_batch::to_csv;
use execution::ExecutionContext;

/// The eight TPC-H tables.
const TPCH_TABLES: &[&str] = &[
    "customer", "lineitem", "nation", "orders", "part", "partsupp", "region", "supplier",
];

fn main() -> ExitCode {
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
    let mut ctx = ExecutionContext::new(HashMap::new());
    for table in TPCH_TABLES {
        let path = format!("{data_dir}/{table}.parquet");
        let source: Arc<dyn DataSource> = Arc::new(ParquetDataSource::new(path));
        ctx.register_data_source(table, source);
    }

    // Execute and time via `Instant::now()` + `elapsed()`.
    let df = ctx.sql(&sql);
    let start = Instant::now();
    let results: Box<dyn Iterator<Item = RecordBatch>> = ctx.execute_data_frame(&df);
    for batch in results {
        // Same shape as `nyc_taxi`: print schema then CSV row data.
        println!("{:?}", batch.schema());
        print!("{}", to_csv(&batch));
    }
    let time = start.elapsed().as_millis();

    println!();
    println!("Query executed in {time} ms");
    ExitCode::SUCCESS
}
