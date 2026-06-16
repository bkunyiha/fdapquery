//!
//! Interactive Flight client: same API shape as
//! [`execution::ExecutionContext`] and [`distributed::DistributedContext`]
//! (`register_csv` / `register` / `sql` / `execute`), but the execution
//! goes over the wire via an `arrow_flight::FlightServiceClient` instead of
//! running locally or through the distributed scheduler.
//!
//! ## Where this fits in the workspace
//!
//! ```text
//!   ExecutionContext       — single-process, runs the plan locally
//!   DistributedContext<C>  — distributed, routes via Scheduler<C>
//!   Context (this file)    — interactive Flight, routes via a single Client
//! ```
//!
//! All three expose the same surface: register tables, submit SQL, get
//! `RecordBatch`es back. A reader switching between them should find the
//! method shapes identical and only the *backing transport* different.

use crate::client::Client;
use crate::endpoint::Endpoint;
use anyhow::Result;
use datasource::CsvDataSource;
use datatypes::RecordBatch;
use logical_plan::{DataFrame, LogicalPlan, Scan};
use protobuf::{pb, serialize_logical_plan};
use sql::{PrattParser, SqlExpr, SqlParser, SqlPlanner, SqlTokenizer};
use std::collections::HashMap;
use std::sync::Arc;

/// CSV batch size for tables registered through `register_csv`. Matches the
/// workspace's other contexts (`distributed::DistributedContext`,
/// `execution::ExecutionContext`).
const CSV_BATCH_SIZE: usize = 1024;

/// Interactive client-side context for executing queries via a single
/// Flight server.
///
/// Holds the table registry (so `Context::sql` can resolve table names) and
/// the [`Client`] that ships logical plans over the wire to the server's
/// `do_get` handler.
pub struct Context {
    tables: HashMap<String, DataFrame>,
    client: Client,
}

impl Context {
    /// Construct a context, connecting to the Flight server at `endpoint`.
    ///
    /// Same "not from inside a tokio runtime" caveat as [`Client::new`] —
    /// the connect step `block_on`s on a fresh runtime. Callers from inside
    /// an async context should construct the `Context` on a separate thread.
    pub fn new(endpoint: Endpoint) -> Result<Self> {
        Ok(Self {
            tables: HashMap::new(),
            client: Client::new(endpoint)?,
        })
    }

    /// Register a CSV file as a table.
    ///
    /// Mirrors `DistributedContext::register_csv` line-for-line — same
    /// `CsvDataSource::new(...)` construction, same `Scan` node, same
    /// `register(...)` delegation. The two contexts diverge only at
    /// `sql`/`execute`: one routes through a `Scheduler`, the other
    /// through a `Client`.
    pub fn register_csv(&mut self, table_name: &str, path: &str, has_header: bool) {
        let ds = CsvDataSource::new(path, None, has_header, CSV_BATCH_SIZE);
        let df = DataFrame::new(LogicalPlan::Scan(Scan::new(path, Arc::new(ds), vec![])));
        self.register(table_name, df);
    }

    /// Register a `DataFrame` as a table.
    pub fn register(&mut self, table_name: &str, df: DataFrame) {
        self.tables.insert(table_name.to_string(), df);
    }

    /// Parse + execute a SQL query via the Flight server.
    ///
    /// Identical parse pipeline to `DistributedContext::sql`: Pratt-parse
    /// the SQL, lower to `DataFrame` via `SqlPlanner`, take its logical
    /// plan. The execution step then delegates to [`Self::execute`].
    pub fn sql(&self, sql: &str) -> Result<Vec<RecordBatch>> {
        let tokens = SqlTokenizer::new(sql).tokenize();
        let parsed = SqlParser::new(tokens).parse(0);
        let select = match parsed {
            Some(SqlExpr::Select(select)) => *select,
            other => anyhow::bail!("Expected a SELECT statement, found {other:?}"),
        };
        let df = SqlPlanner::new().create_data_frame(&select, &self.tables);
        self.execute(df.logical_plan())
    }

    /// Execute a logical plan via the Flight server.
    ///
    /// The wire shape:
    /// 1. Serialise the [`LogicalPlan`] to a [`pb::LogicalPlanNode`] via
    ///    [`protobuf::serialize_logical_plan`].
    /// 2. Wrap it in a [`pb::Action`] (the protobuf message the
    ///    `flight-server`'s `do_get` handler expects in its `Ticket` body).
    /// 3. Encode via `prost::Message::encode_to_vec`.
    /// 4. Hand the bytes to [`Client::do_get`], which makes the gRPC
    ///    call, decodes the `Streaming<FlightData>` response back into
    ///    `RecordBatch`es via `FlightRecordBatchStream`, and returns the
    ///    collected vector.
    pub fn execute(&self, plan: &LogicalPlan) -> Result<Vec<RecordBatch>> {
        let plan_node: pb::LogicalPlanNode = serialize_logical_plan(plan);
        let action = pb::Action {
            query: Some(plan_node),
            task: None,
            settings: vec![],
        };
        let body: Vec<u8> = prost::Message::encode_to_vec(&action);
        self.client.do_get(body)
    }

    /// How many tables are currently registered. Useful for tests and
    /// callers that want to introspect the context state before submitting
    /// a query.
    pub fn table_count(&self) -> usize {
        self.tables.len()
    }
}

#[cfg(test)]
mod tests {
    //! Tests that don't require a running Flight server. The full
    //! parse → serialise → wire → decode round-trip is exercised by the
    //! flight-server integration test.

    use super::*;

    /// Verifies the constructor surfaces a connection error rather than
    /// panicking when the server isn't reachable. Same shape as
    /// `Client::tests::connect_to_closed_port_returns_error`.
    #[test]
    fn new_with_unreachable_endpoint_returns_error() {
        let result = Context::new(Endpoint::new("127.0.0.1", 1));
        assert!(
            result.is_err(),
            "Context::new should propagate connect failure as Err"
        );
    }
}
