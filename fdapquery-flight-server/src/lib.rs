//! # flight-server
//!
//! Arrow Flight server that exposes the query engine over gRPC. This is the
//! transport that turns the [`fdapquery_distributed::ExecutorClient`] abstraction into a
//! real, network-addressable service: the scheduler in another process (or
//! another machine) makes a tonic call here, an [`FdapQueryFlightProducer`]
//! method runs the requested distributed task or streams a final-stage
//! result, and the reply goes back over gRPC.
//!
//! ## What this crate provides
//!
//! - [`fdap_query_flight_producer::FdapQueryFlightProducer`] — the
//!   [`arrow_flight::flight_service_server::FlightService`] implementation:
//!   `do_action("execute_task")` runs intermediate-stage `ShuffleWriterExec`
//!   tasks and returns shuffle locations; `do_get` streams `RecordBatch`es
//!   for either a distributed final task (`protobuf::Action.task` set) or an
//!   interactive logical plan (`protobuf::Action.query` set).
//! - [`flight_server::serve`] — the thin `serve(addr, ctx)` wrapper that
//!   boots a `tonic::transport::Server` with the producer.
//!
//! The bin in `src/bin/flight_server.rs` constructs one
//! `Arc<fdapquery_physical_plan::TaskContext>` at startup (carrying the
//! executor id, host, port, the `SessionConfig`, and the `RuntimeEnv`
//! that owns the `ShuffleManager`) and hands it to the producer for
//! the lifetime of the process.

pub mod fdap_query_flight_producer;
pub mod flight_server;

// Top-level re-exports so external consumers
// can write the short form `use fdapquery_flight_server::{FdapQueryFlightProducer, serve};`
// instead of reaching through submodule paths.
pub use fdap_query_flight_producer::FdapQueryFlightProducer;
pub use flight_server::serve;
