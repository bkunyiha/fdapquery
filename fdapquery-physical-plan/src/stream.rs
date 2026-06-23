//! Async stream surface for record batches: the canonical type alias
//! `SendableRecordBatchStream`, the `RecordBatchStream` trait (a `Stream`
//! that also carries the output schema), and `RecordBatchStreamAdapter`
//! (lifts an arbitrary `Stream<Item = Result<RecordBatch>>` into the
//! schema-aware form).
//!
//! Same shape as DataFusion's `datafusion-execution::stream` module.
//! Phase C migrates these to `fdapquery-execution` (they only live here
//! in Phase B because the `ExecutionPlan` trait that references them is
//! in this crate, and `fdapquery-execution` already depends on
//! `fdapquery-physical-plan` — moving them across the dependency edge
//! is a Phase C reshape).

use arrow_schema::SchemaRef;
use fdapquery_datatypes::{RecordBatch, Result};
use futures::Stream;
use pin_project_lite::pin_project;
use std::pin::Pin;
use std::task::{Context, Poll};

/// A `Stream` of `RecordBatch`es that also exposes its output schema.
///
/// The `schema()` accessor is mandatory because batch streams (unlike
/// plain `Iterator`s) are consumed downstream by code that needs to
/// know the schema **before** the first batch arrives —
/// `FlightDataEncoder` builds its Flight schema header from this, the
/// optimiser inspects column types, etc. A bare `Stream<Item =
/// Result<RecordBatch>>` cannot carry that information; this trait
/// pins it as a contract.
pub trait RecordBatchStream: Stream<Item = Result<RecordBatch>> {
    /// The schema of the batches this stream yields.
    fn schema(&self) -> SchemaRef;
}

/// The canonical async stream type for record batches.
///
/// `Pin<Box<…>>` lets the stream cross `await` points and be stored in
/// struct fields without lifetime grief; `Send` lets it move between
/// tokio worker threads. Matches DataFusion's
/// `datafusion-execution::SendableRecordBatchStream` exactly.
pub type SendableRecordBatchStream = Pin<Box<dyn RecordBatchStream + Send>>;

pin_project! {
    /// Wraps an arbitrary `Stream<Item = Result<RecordBatch>>` with a
    /// `SchemaRef` accessor, satisfying the `RecordBatchStream`
    /// contract. Operators reuse generic stream combinators
    /// (`StreamExt::map`, `::filter`, `::take`, `::flatten`, …) and
    /// hand the result back as a `SendableRecordBatchStream` by
    /// wrapping in this adapter.
    pub struct RecordBatchStreamAdapter<S> {
        schema: SchemaRef,
        #[pin]
        stream: S,
    }
}

impl<S> RecordBatchStreamAdapter<S> {
    /// Construct an adapter from a schema and an inner stream.
    pub fn new(schema: SchemaRef, stream: S) -> Self {
        Self { schema, stream }
    }
}

impl<S> Stream for RecordBatchStreamAdapter<S>
where
    S: Stream<Item = Result<RecordBatch>>,
{
    type Item = Result<RecordBatch>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.project().stream.poll_next(cx)
    }
}

impl<S> RecordBatchStream for RecordBatchStreamAdapter<S>
where
    S: Stream<Item = Result<RecordBatch>>,
{
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_schema::{DataType, Field, Schema};
    use futures::StreamExt;
    use std::sync::Arc;

    /// `RecordBatchStreamAdapter` over an empty stream still reports
    /// the supplied schema.
    #[tokio::test]
    async fn adapter_schema_is_returned_unchanged() {
        let schema: SchemaRef = Arc::new(Schema::new(vec![Field::new(
            "a",
            DataType::Int32,
            false,
        )]));
        let inner = futures::stream::empty::<Result<RecordBatch>>();
        let adapter = RecordBatchStreamAdapter::new(Arc::clone(&schema), inner);
        assert_eq!(adapter.schema().fields().len(), 1);
        assert_eq!(adapter.schema().field(0).name(), "a");
    }

    /// An adapter over a small iterator-derived stream yields the
    /// underlying batches in order.
    #[tokio::test]
    async fn adapter_yields_inner_batches_in_order() {
        use arrow_array::{Int32Array, RecordBatch as ArrowBatch};
        let schema: SchemaRef = Arc::new(Schema::new(vec![Field::new(
            "a",
            DataType::Int32,
            false,
        )]));
        let batch_a =
            ArrowBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int32Array::from(vec![1]))])
                .unwrap();
        let batch_b =
            ArrowBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int32Array::from(vec![2]))])
                .unwrap();
        let inner = futures::stream::iter(vec![Ok(batch_a), Ok(batch_b)]);
        let adapter = RecordBatchStreamAdapter::new(Arc::clone(&schema), inner);
        let collected: Vec<_> = adapter.collect().await;
        assert_eq!(collected.len(), 2);
        assert!(collected.iter().all(|r| r.is_ok()));
    }
}
