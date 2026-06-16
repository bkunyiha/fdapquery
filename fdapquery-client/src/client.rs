//!
//! Synchronous Flight client. Wraps an `arrow_flight::FlightServiceClient`
//! over a tonic `Channel`, plus a dedicated tokio runtime that drives the
//! async tonic calls. Sync on the outside (`Client::new`, future
//! `Client::do_action` / `Client::do_get`), async on the inside.
//!
//! ## Async ↔ sync layering
//!
//! tonic is async-only — `FlightServiceClient::do_action(...)`,
//! `do_get(...)`, etc. all return futures. But the synchronous
//! `distributed::ExecutorClient` trait (used by `Scheduler::execute_stage`
//! to dispatch tasks) needs synchronous `execute_task(...)` /
//! `execute_final_task(...)` methods. So `Client` owns a tokio runtime and
//! every method internally `block_on`s its async work. This is the inverse
//! of `flight-server`'s `spawn_blocking` pattern (async caller → sync
//! callee).
//!
//! ## Runtime ownership — one per Client (Phase 1 simplification)
//!
//! Each `Client` builds its own multi-thread tokio runtime with a single
//! worker thread. For 3 executors that's 3 runtimes — wasteful but
//! observably bounded (each idle runtime costs ~one parked thread). A
//! Phase-2 optimisation would share one runtime across all `Client`s in a
//! `FlightExecutorClient`; for the teaching port we accept the small
//! overhead in exchange for trivially clear ownership.

use crate::endpoint::Endpoint;
use anyhow::{Result, anyhow};
use arrow_flight::decode::FlightRecordBatchStream;
use arrow_flight::error::FlightError;
use arrow_flight::flight_service_client::FlightServiceClient;
use arrow_flight::{Action, Ticket};
use datatypes::RecordBatch;
use futures::StreamExt;
use tokio::runtime::Runtime;
use tonic::Request;
use tonic::transport::Channel;

/// A synchronous Flight client connected to one Flight server.
///
/// Construct with [`Client::new`]; subsequent `do_action` / `do_get` methods
/// drive the gRPC calls through the owned tokio runtime.
pub struct Client {
    /// Owned tokio runtime. Calls into the Flight server `block_on` futures
    /// against this runtime. See module-level note on ownership.
    runtime: Runtime,
    /// The tonic transport channel for this server. `Channel` is `Clone` and
    /// shares the underlying HTTP/2 connection across clones — we keep one
    /// copy here and clone it into per-method `FlightServiceClient`
    /// instances (the recommended tonic pattern).
    channel: Channel,
    /// The endpoint we connected to. Held for diagnostic / `Debug` output;
    /// not used by the gRPC machinery (`channel` carries all the wiring).
    endpoint: Endpoint,
}

impl Client {
    /// Construct a client by connecting to `endpoint`.
    ///
    /// Builds a single-worker tokio runtime, then drives `Channel::connect`
    /// on it. Returns an error if either the runtime can't be built (rare —
    /// would indicate a fatal system-level failure) or the connection
    /// can't be established (more common — server not up, wrong port,
    /// transient network).
    ///
    /// **Not safe to call from inside an existing tokio runtime** —
    /// `runtime.block_on` panics if a runtime is already active on this
    /// thread. Callers from within an async context should construct the
    /// `Client` on a separate thread (`std::thread::spawn(|| Client::new(...))`)
    /// or pre-build a `Channel` and use a (future) `Client::from_channel`
    /// constructor.
    pub fn new(endpoint: Endpoint) -> Result<Self> {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .enable_all()
            .thread_name(format!("client-{}-{}", endpoint.host, endpoint.port))
            .build()?;
        let url = endpoint.url();
        let channel = runtime.block_on(async move {
            Channel::from_shared(url)?
                .connect()
                .await
                .map_err(anyhow::Error::from)
        })?;
        Ok(Self {
            runtime,
            channel,
            endpoint,
        })
    }

    /// The endpoint this client is connected to. Useful for tracing and
    /// for `FlightExecutorClient`'s "which executor did this client
    /// belong to" lookups.
    pub fn endpoint(&self) -> &Endpoint {
        &self.endpoint
    }

    /// Send a `do_action` request to the connected Flight server and return
    /// the **first result body** as bytes.
    ///
    /// The Flight `do_action` RPC returns a server-streaming response
    /// (`stream Result`). For our two real action handlers
    /// (`"execute_task"` and any future actions), the server emits exactly
    /// one `Result`, so we wait for the first message and return its body.
    /// If the server returns no messages at all (would only happen if a
    /// custom handler broke the contract), we surface that as an error
    /// rather than silently returning empty bytes.
    ///
    /// `body` is the protobuf payload — typically `pb::TaskInfo` encoded
    /// via `prost::Message::encode_to_vec(&task_info)`. The returned
    /// `Vec<u8>` is the response payload — typically a `pb::TaskResult`
    /// the caller decodes via `prost::Message::decode(&bytes)`.
    pub fn do_action(&self, action_type: impl Into<String>, body: Vec<u8>) -> Result<Vec<u8>> {
        let action = Action {
            r#type: action_type.into(),
            body: body.into(),
        };
        let channel = self.channel.clone();
        self.runtime.block_on(async move {
            let mut client = FlightServiceClient::new(channel);
            let response = client.do_action(Request::new(action)).await?;
            let mut stream = response.into_inner();
            let first = stream
                .message()
                .await?
                .ok_or_else(|| anyhow!("do_action response stream was empty"))?;
            Ok(first.body.to_vec())
        })
    }

    /// Send a `do_get` request to the connected Flight server, decode the
    /// `FlightData` response stream back into `RecordBatch`es, and return
    /// the full collected vector.
    ///
    /// `ticket_body` is the protobuf payload that goes inside the Flight
    /// `Ticket` message — typically a `pb::Action` (with `Action.query`
    /// = `pb::LogicalPlanNode`) encoded via
    /// `prost::Message::encode_to_vec`. The server runs the plan, streams
    /// the result batches as `FlightData` messages, and this helper
    /// reassembles them.
    ///
    /// The decode path mirrors `flight-server`'s integration test —
    /// `FlightRecordBatchStream::new_from_flight_data` pipes
    /// `FlightData → RecordBatch`, mapping any `tonic::Status` errors from
    /// the wire into `FlightError::Tonic`.
    pub fn do_get(&self, ticket_body: Vec<u8>) -> Result<Vec<RecordBatch>> {
        let ticket = Ticket {
            ticket: ticket_body.into(),
        };
        let channel = self.channel.clone();
        self.runtime.block_on(async move {
            let mut client = FlightServiceClient::new(channel);
            let response = client.do_get(Request::new(ticket)).await?;
            // Map the inbound Streaming<FlightData>'s `Status` errors into
            // `FlightError::Tonic` so `FlightRecordBatchStream` can consume it.
            let flight_data_stream = response
                .into_inner()
                .map(|r| r.map_err(|status| FlightError::Tonic(Box::new(status))));
            let mut record_batch_stream =
                FlightRecordBatchStream::new_from_flight_data(flight_data_stream);
            let mut batches: Vec<RecordBatch> = Vec::new();
            while let Some(batch_result) = record_batch_stream.next().await {
                batches.push(batch_result?);
            }
            Ok(batches)
        })
    }
}

impl std::fmt::Debug for Client {
    /// Custom `Debug` because `Runtime` doesn't implement `Debug` and
    /// `Channel`'s Debug output isn't useful. Print just the endpoint.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Client")
            .field("endpoint", &self.endpoint)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// We can't easily test successful connection without a real Flight server
    /// running — that's covered by the flight-server integration test. What
    /// we *can* test here is that `new` fails (rather than panics) when the
    /// endpoint is unreachable.
    ///
    /// Port 1 is privileged and almost certainly closed — `Channel::connect`
    /// returns a transport error. We verify the error propagates rather than
    /// panicking so callers can recover cleanly.
    #[test]
    fn connect_to_closed_port_returns_error() {
        let ep = Endpoint::new("127.0.0.1", 1);
        let result = Client::new(ep);
        assert!(result.is_err(), "connecting to port 1 should fail");
    }
}
