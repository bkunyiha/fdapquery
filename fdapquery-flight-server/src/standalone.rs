//! [`spawn_in_process_flight_server`] — public helper used by
//! `fdapquery_flight_client::SessionContextExt::standalone()` to
//! bring up an in-process Flight server for the standalone-cluster
//! path.
//!
//! Mirror of Ballista's `ballista_executor::new_standalone_executor`
//! at `ballista/executor/src/standalone.rs`. Same pattern: spawn a
//! background thread with its own tokio runtime, bind a random TCP
//! port, hand the bound address back via `mpsc::channel`. This lets
//! the caller's runtime drive the client without competing for
//! workers with the server's request handlers.
//!
//! Session 20d moved this logic here from two hand-rolled copies in
//! `fdapquery-examples/src/bin/distributed_flight_example.rs` and
//! `fdapquery-flight-client/tests/distributed_integration_test.rs`
//! (both had ~40-line versions of the same code).

use crate::fdap_query_flight_producer::FdapQueryFlightProducer;
use fdapquery_datatypes::{FdapQueryError, Result};
use fdapquery_distributed::DistributedConfig;
use fdapquery_physical_plan::{SessionConfig, TaskContext};
use std::net::SocketAddr;
use std::sync::{Arc, mpsc};

/// Spawn an in-process Flight server on a random TCP port.
///
/// Runs in a background OS thread with its own tokio runtime so the
/// caller's runtime is free to drive the client without competing
/// for workers with the server's request handlers.
///
/// Blocks until the server has bound its socket; returns the bound
/// [`SocketAddr`] to the caller. Any bind or spawn failure surfaces
/// as `FdapQueryError::Internal`.
///
/// The spawned thread runs the server indefinitely — it has no
/// shutdown hook in v0.1. Callers (tests, examples, `standalone()`)
/// let the OS reclaim the port when the process exits. If a
/// long-running standalone use case emerges, add a shutdown channel
/// alongside the address one.
pub fn spawn_in_process_flight_server(
    executor_id: &str,
    config: &DistributedConfig,
) -> Result<SocketAddr> {
    let executor_id_owned = executor_id.to_string();
    let runtime = Arc::new(config.build_runtime_env());
    let (tx, rx) = mpsc::channel::<Result<SocketAddr>>();

    std::thread::spawn(move || {
        let tokio_runtime = match tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
        {
            Ok(rt) => rt,
            Err(e) => {
                let _ = tx.send(Err(FdapQueryError::Internal(format!(
                    "standalone: tokio runtime: {e}"
                ))));
                return;
            }
        };

        tokio_runtime.block_on(async move {
            let listener = match tokio::net::TcpListener::bind("127.0.0.1:0").await {
                Ok(l) => l,
                Err(e) => {
                    let _ = tx.send(Err(FdapQueryError::Internal(format!(
                        "standalone: bind: {e}"
                    ))));
                    return;
                }
            };
            let addr = match listener.local_addr() {
                Ok(a) => a,
                Err(e) => {
                    let _ = tx.send(Err(FdapQueryError::Internal(format!(
                        "standalone: local_addr: {e}"
                    ))));
                    return;
                }
            };
            // Signal the caller BEFORE serving — a race here would
            // let the caller connect before the socket is
            // listening. The `bind` above already reserved the port
            // in the kernel, so it's safe to send the address now.
            if tx.send(Ok(addr)).is_err() {
                // Receiver dropped; nothing to serve.
                return;
            }

            let ctx = Arc::new(TaskContext::new(
                executor_id_owned,
                addr.ip().to_string(),
                addr.port(),
                SessionConfig::new(),
                runtime,
            ));
            let producer = FdapQueryFlightProducer::new(ctx);

            // `serve_with_incoming` runs until the incoming stream
            // ends. For a `TcpListener`-backed stream this is
            // effectively "forever."
            let _ = tonic::transport::Server::builder()
                .add_service(
                    arrow_flight::flight_service_server::FlightServiceServer::new(producer),
                )
                .serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(listener))
                .await;
        });
    });

    rx.recv().map_err(|e| {
        FdapQueryError::Internal(format!(
            "standalone: failed to receive bound address: {e}"
        ))
    })?
}
