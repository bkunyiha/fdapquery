//! End-to-end integration test that drives a real distributed
//! `SELECT state, SUM(salary) FROM employee GROUP BY state` query
//! through `SessionContext::standalone()` — mirror of Ballista's
//! standalone test path.
//!
//! Post-Session-20d, this test uses zero-arg
//! `SessionContext::standalone()` which internally spawns the
//! in-process Flight server and connects a `FlightExecutorClient`.
//! All the hand-rolled `spawn_in_process_server` +
//! `FlightExecutorClient::connect` machinery collapsed into that
//! one constructor call.

#![cfg(feature = "standalone")]

use fdapquery::SessionContext;
use fdapquery_datatypes::RecordBatch;
use fdapquery_flight_client::SessionContextExt;
use futures::TryStreamExt;

const EMPLOYEE_CSV: &str = "../testdata/employee.csv";

/// **The Phase 1 payoff test.** Run a real
/// `SELECT state, SUM(salary) FROM employee GROUP BY state`
/// distributed query end-to-end via `SessionContext::standalone()`.
/// The constructor spawns the Flight server + connects a client
/// internally; the test only writes what the user would write.
#[tokio::test]
async fn distributed_aggregate_query_end_to_end_via_flight() {
    let mut ctx = SessionContext::standalone()
        .await
        .expect("SessionContext::standalone");
    ctx.register_csv("employee", EMPLOYEE_CSV);

    // `SessionContext::sql` is sync (returns `DataFrame`);
    // `execute_data_frame` drives the plan through the query
    // planner and returns a stream. The scheduler awaits every
    // Flight call on the current tokio runtime; the resulting
    // `SendableRecordBatchStream` is drained via `try_collect`.
    let df = ctx
        .sql("SELECT state, SUM(salary) FROM employee GROUP BY state")
        .expect("sql plan");
    let stream = ctx
        .execute_data_frame(&df)
        .await
        .expect("execute_data_frame");
    let results: Vec<RecordBatch> = stream.try_collect().await.expect("drain stream");

    // Sanity check: at least one output batch and total row count
    // matches the number of distinct states in employee.csv.
    // employee.csv has 4 data rows with states: CA, CO, CO, "" (empty)
    // → 3 distinct groups → 3 output rows.
    let total_output_rows: usize = results.iter().map(|b| b.num_rows()).sum();
    assert!(
        !results.is_empty(),
        "expected at least one result batch; got {} batches",
        results.len()
    );
    assert_eq!(
        total_output_rows, 3,
        "expected 3 output rows (one per distinct state); got {total_output_rows}",
    );

    // Verify the schema looks right: 2 columns (state, sum).
    for batch in &results {
        assert_eq!(
            batch.num_columns(),
            2,
            "expected 2 output columns (state, sum); got {}",
            batch.num_columns()
        );
    }
}
