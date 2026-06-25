//!
//! Async Flight client. Wraps an `arrow_flight::FlightServiceClient` over a
//! tonic `Channel`. Every method is `async fn`; callers drive the futures
//! on the active tokio runtime via `.await`. Mirrors DataFusion's
//! `BallistaClient` shape: a thin wrapper over the generated tonic client
//! with no internal runtime ownership.
//!
//! ## Async-native — no per-Client runtime
//!
//! tonic is async-only — `FlightServiceClient::do_action(...)`,
//! `do_get(...)`, etc. all return futures. The
//! `fdapquery_distributed::ExecutorClient` trait is also `async` (the
//! scheduler awaits each `execute_task` call on the caller's tokio
//! runtime), so the entire dispatch path is async end-to-end. `Client`
//! holds only the `Channel` and an `Endpoint` — no `Runtime` —
//! eliminating the `block_on` bridge the Phase A client carried.

use crate::endpoint::Endpoint;
use anyhow::{Result, anyhow};
use arrow_flight::decode::FlightRecordBatchStream;
use arrow_flight::error::FlightError;
use arrow_flight::flight_service_client::FlightServiceClient;
use arrow_flight::{Action, Ticket};
use fdapquery_datatypes::RecordBatch;
use futures::StreamExt;
use tonic::Request;
use tonic::transport::Channel;

/// An async Flight client connected to one Flight server.
///
/// Construct with [`Client::connect`] from within a tokio runtime;
/// subsequent `do_action` / `do_get` methods are `async fn` that the
/// caller awaits on the active runtime.
pub struct Client {
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
    /// Construct a client by connecting to `endpoint`. Async — call from
    /// within a tokio runtime and `.await`.
    ///
    /// Returns an error if the connection can't be established — server
    /// not up, wrong port, transient network. No internal runtime
    /// ownership, no `block_on` — the caller's runtime drives the
    /// `Channel::connect` future directly.
    pub async fn connect(endpoint: Endpoint) -> Result<Self> {
        let url = endpoint.url();
        let channel = Channel::from_shared(url)?
            .connect()
            .await
            .map_err(anyhow::Error::from)?;
        Ok(Self { channel, endpoint })
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
    pub async fn do_action(
        &self,
        action_type: impl Into<String>,
        body: Vec<u8>,
    ) -> Result<Vec<u8>> {
        let action = Action {
            r#type: action_type.into(),
            body: body.into(),
        };
        let channel = self.channel.clone();
        let mut client = FlightServiceClient::new(channel);
        let response = client.do_action(Request::new(action)).await?;
        let mut stream = response.into_inner();
        let first = stream
            .message()
            .await?
            .ok_or_else(|| anyhow!("do_action response stream was empty"))?;
        Ok(first.body.to_vec())
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
    pub async fn do_get(&self, ticket_body: Vec<u8>) -> Result<Vec<RecordBatch>> {
        let ticket = Ticket {
            ticket: ticket_body.into(),
        };
        let channel = self.channel.clone();
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
    }
}

impl std::fmt::Debug for Client {
    /// Custom `Debug` because `Channel`'s Debug output isn't useful.
    /// Print just the endpoint.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Client")
            .field("endpoint", &self.endpoint)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// We can't easily test successful connection without a real Flight
    /// server running — that's covered by the flight-server integration
    /// test. What we *can* test here is that `connect` fails (rather than
    /// panics) when the endpoint is unreachable.
    ///
    /// Port 1 is privileged and almost certainly closed —
    /// `Channel::connect` returns a transport error. We verify the
    /// error propagates rather than panicking so callers can recover
    /// cleanly.
    #[tokio::test]
    async fn connect_to_closed_port_returns_error() {
        let ep = Endpoint::new("127.0.0.1", 1);
        let result = Client::connect(ep).await;
        assert!(result.is_err(), "connecting to port 1 should fail");
    }
}
