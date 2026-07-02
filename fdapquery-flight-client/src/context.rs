//!
//! Interactive Flight client: same `register_csv` / `register` / `sql` /
//! `execute` shape as `fdapquery::SessionContext`, but the execution
//! goes over the wire via an `arrow_flight::FlightServiceClient` instead
//! of running locally.
//!
//! ## Where this fits in the workspace
//!
//! ```text
//!   SessionContext                      — single-process, runs the plan locally
//!   SessionContext (via
//!     SessionContextExt::standalone)    — distributed, routes via Scheduler<C>
//!   Context (this file)                 — interactive Flight, routes via a single Client
//! ```
//!
//! The distributed variant is `SessionContext` extended with the
//! [`SessionContextExt`](fdapquery_distributed::SessionContextExt) trait
//! (mirror of Ballista's `SessionContextExt` at
//! `ballista/client/src/extension.rs`). It installs a
//! `DistributedQueryPlanner` on the session's `SessionState`, so every
//! query routes through the in-process scheduler transparently.
//!
//! All three expose the same surface: register tables, submit SQL, get
//! `RecordBatch`es back. A reader switching between them should find the
//! method shapes identical and only the *backing transport* different.

use crate::client::Client;
use crate::endpoint::Endpoint;
use anyhow::Result;
use fdapquery_catalog::{CsvDataSource, provider_as_source};
use fdapquery_datatypes::RecordBatch;
use fdapquery_expr::{DataFrame, LogicalPlan, TableScan};
use fdapquery_proto::{protobuf, serialize_logical_plan};
use fdapquery_sql::SqlToRel;
use fdapquery_sql::sqlparser::dialect::GenericDialect;
use fdapquery_sql::sqlparser::parser::Parser;
use std::collections::HashMap;
use std::sync::Arc;

/// CSV batch size for tables registered through `register_csv`. Matches
/// [`fdapquery::SessionContext`]'s `register_csv` default so a query run
/// against the interactive client and a query run against a session
/// produced by [`fdapquery_distributed::SessionContextExt::standalone`]
/// see the same batch shape.
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
    /// Async — call from within a tokio runtime and `.await`. Mirrors
    /// [`Client::connect`]; no internal runtime ownership.
    pub async fn connect(endpoint: Endpoint) -> Result<Self> {
        Ok(Self {
            tables: HashMap::new(),
            client: Client::connect(endpoint).await?,
        })
    }

    /// Register a CSV file as a table.
    ///
    /// Same `CsvDataSource::new(...)` construction, same `TableScan` node,
    /// same `register(...)` delegation as [`fdapquery::SessionContext::register_csv`].
    /// This context diverges from `SessionContext` only at `sql` / `execute`:
    /// where `SessionContext` runs the plan through a `QueryPlanner`, this
    /// one ships the plan over the wire to a Flight server via
    /// [`Client::do_get`].
    pub fn register_csv(&mut self, table_name: &str, path: &str, has_header: bool) {
        let ds = CsvDataSource::new(path, None, has_header, CSV_BATCH_SIZE);
        // Wrap the provider as a `TableSource` for
        // the logical plan; the planner unwraps it at the seam.
        let scan = TableScan::new(path, provider_as_source(Arc::new(ds)), vec![])
            .expect("Context::register_csv: scan construction");
        let df = DataFrame::new(LogicalPlan::TableScan(scan));
        self.register(table_name, df);
    }

    /// Register a `DataFrame` as a table.
    pub fn register(&mut self, table_name: &str, df: DataFrame) {
        self.tables.insert(table_name.to_string(), df);
    }

    /// Parse + execute a SQL query via the Flight server. Async — call
    /// from within a tokio runtime and `.await`.
    ///
    /// Parses with `sqlparser` (same crate DataFusion uses), lowers via
    /// [`SqlToRel`], and delegates execution to [`Self::execute`].
    pub async fn sql(&self, sql: &str) -> Result<Vec<RecordBatch>> {
        let dialect = GenericDialect {};
        let mut statements = Parser::parse_sql(&dialect, sql)
            .map_err(|e| anyhow::anyhow!("parse: {e}"))?;
        if statements.len() > 1 {
            anyhow::bail!("multiple SQL statements per call are not supported at v0.1");
        }
        let statement = statements
            .pop()
            .ok_or_else(|| anyhow::anyhow!("empty SQL input"))?;
        let df = SqlToRel::new(&self.tables)
            .sql_statement_to_plan(&statement)
            .map_err(|e| anyhow::anyhow!("plan: {e}"))?;
        self.execute(df.logical_plan()).await
    }

    /// Execute a logical plan via the Flight server. Async — call from
    /// within a tokio runtime and `.await`.
    ///
    /// The wire shape:
    /// 1. Serialise the [`LogicalPlan`] to a [`protobuf::LogicalPlanNode`] via
    ///    [`fdapquery_proto::serialize_logical_plan`].
    /// 2. Wrap it in a [`protobuf::Action`] (the protobuf message the
    ///    `flight-server`'s `do_get` handler expects in its `Ticket` body).
    /// 3. Encode via `prost::Message::encode_to_vec`.
    /// 4. Hand the bytes to [`Client::do_get`], which makes the gRPC
    ///    call, decodes the `Streaming<FlightData>` response back into
    ///    `RecordBatch`es via `FlightRecordBatchStream`, and returns the
    ///    collected vector.
    pub async fn execute(&self, plan: &LogicalPlan) -> Result<Vec<RecordBatch>> {
        let plan_node: protobuf::LogicalPlanNode = serialize_logical_plan(plan);
        let action = protobuf::Action {
            query: Some(plan_node),
            task: None,
            settings: vec![],
        };
        let body: Vec<u8> = prost::Message::encode_to_vec(&action);
        self.client.do_get(body).await
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
    #[tokio::test]
    async fn connect_with_unreachable_endpoint_returns_error() {
        let result = Context::connect(Endpoint::new("127.0.0.1", 1)).await;
        assert!(
            result.is_err(),
            "Context::connect should propagate connect failure as Err"
        );
    }
}
