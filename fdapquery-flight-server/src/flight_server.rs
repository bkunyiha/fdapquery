//! Contains the [`serve`] function — the library-side entry point that wires
//! up a tonic gRPC server around an [`FdapQueryFlightProducer`] and binds it to a
//! TCP address. The runnable `main()` lives in `src/bin/flight_server.rs`,
//! which calls [`serve`].
//!
//! ## Why this split (lib + bin)
//! Keeping the bind/serve loop in the library lets integration tests start
//! the server on a random port (`0.0.0.0:0`) without going through the
//! binary. The binary itself stays a thin shim around `serve`.

use crate::fdap_query_flight_producer::FdapQueryFlightProducer;
use arrow_flight::flight_service_server::FlightServiceServer;
use fdapquery_physical_plan::TaskContext;
use std::net::SocketAddr;
use std::sync::Arc;
use tonic::transport::Server;
use tracing::info;

/// Bind a tonic gRPC server with the [`FdapQueryFlightProducer`] service on
/// `addr` and run it until shutdown.
///
/// `ctx` is the per-executor `Arc<TaskContext>` — built once by the
/// caller (the bin in `src/bin/flight_server.rs` or an integration
/// test) and handed to the producer which holds it for the server's
/// lifetime. Operators receive `Arc::clone(&ctx)` on every
/// `execute(partition, ctx)` call.
///
/// Returns a `tonic::transport::Error` if the bind fails or the server
/// loop exits with an error. Callers are responsible for the tokio runtime.
pub async fn serve(addr: SocketAddr, ctx: Arc<TaskContext>) -> Result<(), tonic::transport::Error> {
    let producer = FdapQueryFlightProducer::new(ctx);
    info!("Flight server listening on {}", addr);
    Server::builder()
        .add_service(FlightServiceServer::new(producer))
        .serve(addr)
        .await
}
