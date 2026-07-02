//! Async stream surface for record batches: the canonical type alias
//! `SendableRecordBatchStream`, the `RecordBatchStream` trait (a `Stream`
//! that also carries the output schema), `RecordBatchStreamAdapter`
//! (lifts an arbitrary `Stream<Item = Result<RecordBatch>>` into the
//! schema-aware form), and two small concrete streams —
//! `EmptyRecordBatchStream` (yields no batches) and `MemoryStream`
//! (replays a pre-loaded `Vec<RecordBatch>`).
//!
//! Same shape as DataFusion's `datafusion-execution::stream` and
//! `datafusion-physical-plan::stream` / `memory` modules. Migrates
//! these to `fdapquery-execution` (they only live here
//! because the `ExecutionPlan` trait that references them is
//! in this crate, and `fdapquery-execution` already depends on
//! `fdapquery-physical-plan` — moving them across the dependency edge
//! is a reshape).

use arrow_schema::SchemaRef;
use fdapquery_datatypes::{RecordBatch, Result};
use futures::Stream;
use pin_project_lite::pin_project;
use std::pin::Pin;
use std::sync::Arc;
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

/// A `RecordBatchStream` that yields no batches. Carries a `SchemaRef`
/// so callers can inspect the output type without seeing any data.
///
/// Mirror of `datafusion::physical_plan::stream::EmptyRecordBatchStream`.
/// Used as a schema-preserving placeholder (e.g. when a query produces
/// zero rows but the caller still needs a schema to build headers).
pub struct EmptyRecordBatchStream {
    /// Schema wrapped by Arc.
    schema: SchemaRef,
}

impl EmptyRecordBatchStream {
    /// Create an empty `RecordBatchStream` over the given schema.
    pub fn new(schema: SchemaRef) -> Self {
        Self { schema }
    }
}

impl Stream for EmptyRecordBatchStream {
    type Item = Result<RecordBatch>;

    fn poll_next(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Poll::Ready(None)
    }
}

impl RecordBatchStream for EmptyRecordBatchStream {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }
}

/// A `RecordBatchStream` that replays a pre-loaded `Vec<RecordBatch>`.
///
/// Mirror of `datafusion::physical_plan::memory::MemoryStream`. Used as
/// the runtime yield of `MemoryExec`-style leaf operators (in fdapquery:
/// `fdapquery_physical_plan::MemoryExec`) and anywhere a caller needs
/// to hand an already-materialised batch collection off as a stream.
///
/// If `projection` is `Some(indices)`, each batch is projected down to
/// the requested column indices via `RecordBatch::project` and the
/// reported schema is the projected schema.
///
/// The DataFusion original also carries an optional `MemoryReservation`
/// (freed on drop, tied to the datafusion `MemoryPool`) and an optional
/// `fetch` limit. fdapquery has no `MemoryPool` port yet — that
/// arrives with the runtime work in Phase 3 — so those fields are
/// omitted for now. The public constructor signature already matches
/// DataFusion's `try_new(data, schema, projection)`; adding
/// `with_reservation` / `with_fetch` is a pure addition when the
/// underlying types land.
pub struct MemoryStream {
    /// The batches to replay, in yield order.
    data: Vec<RecordBatch>,
    /// The schema reported by [`RecordBatchStream::schema`]. Already
    /// projected if `projection` is `Some(_)`.
    schema: SchemaRef,
    /// Optional column-index projection applied to every batch.
    projection: Option<Vec<usize>>,
    /// Next index into `data` to yield.
    index: usize,
}

impl MemoryStream {
    /// Create a stream over `data` with the given output schema and
    /// optional projection.
    ///
    /// `schema` MUST be the *output* schema: if `projection` is
    /// `Some(indices)`, callers pass the already-projected schema (the
    /// schema of the columns at those indices), matching DataFusion's
    /// contract. Projection index bounds are validated eagerly against
    /// the first batch (if any) so callers see the error at construction
    /// time rather than on first poll.
    pub fn try_new(
        data: Vec<RecordBatch>,
        schema: SchemaRef,
        projection: Option<Vec<usize>>,
    ) -> Result<Self> {
        // Eagerly validate projection indices against the first batch's
        // column count so an out-of-bounds projection is reported at
        // construction (matches DataFusion's expectations — the arrow
        // `project` call on the first `poll_next` would otherwise
        // surface the same error, but callers of `try_new` expect
        // structural errors upfront).
        if let (Some(indices), Some(first)) = (projection.as_ref(), data.first()) {
            let ncols = first.num_columns();
            for &i in indices {
                if i >= ncols {
                    return Err(fdapquery_datatypes::FdapQueryError::ArrowError(
                        arrow_schema::ArrowError::SchemaError(format!(
                            "MemoryStream projection index {i} out of bounds for {ncols} columns"
                        )),
                    ));
                }
            }
        }
        Ok(Self {
            data,
            schema,
            projection,
            index: 0,
        })
    }
}

impl Stream for MemoryStream {
    type Item = Result<RecordBatch>;

    fn poll_next(
        mut self: Pin<&mut Self>,
        _cx: &mut Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        if self.index >= self.data.len() {
            return Poll::Ready(None);
        }
        self.index += 1;
        let batch = &self.data[self.index - 1];
        // Return just the columns requested.
        let batch = match self.projection.as_ref() {
            Some(columns) => match batch.project(columns) {
                Ok(b) => b,
                Err(e) => return Poll::Ready(Some(Err(e.into()))),
            },
            None => batch.clone(),
        };
        Poll::Ready(Some(Ok(batch)))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.data.len().saturating_sub(self.index);
        (remaining, Some(remaining))
    }
}

impl RecordBatchStream for MemoryStream {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
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
        let schema: SchemaRef = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, true)]));
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
        let schema: SchemaRef = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, true)]));
        let batch_a = ArrowBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int32Array::from(vec![1]))],
        )
        .unwrap();
        let batch_b = ArrowBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int32Array::from(vec![2]))],
        )
        .unwrap();
        let inner = futures::stream::iter(vec![Ok(batch_a), Ok(batch_b)]);
        let adapter = RecordBatchStreamAdapter::new(Arc::clone(&schema), inner);
        let collected: Vec<_> = adapter.collect().await;
        assert_eq!(collected.len(), 2);
        assert!(collected.iter().all(|r| r.is_ok()));
    }

    /// Helper: build a 3-column schema (Int32 / Int32 / Utf8).
    fn three_col_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, true),
            Field::new("b", DataType::Int32, true),
            Field::new("c", DataType::Utf8, true),
        ]))
    }

    /// Helper: build a `RecordBatch` matching [`three_col_schema`] with
    /// the supplied `a` / `b` / `c` values.
    fn three_col_batch(
        schema: &SchemaRef,
        a: &[i32],
        b: &[i32],
        c: &[&str],
    ) -> RecordBatch {
        use arrow_array::{Int32Array, StringArray};
        RecordBatch::try_new(
            Arc::clone(schema),
            vec![
                Arc::new(Int32Array::from(a.to_vec())),
                Arc::new(Int32Array::from(b.to_vec())),
                Arc::new(StringArray::from(c.to_vec())),
            ],
        )
        .unwrap()
    }

    /// `EmptyRecordBatchStream` reports the supplied schema and yields
    /// `Poll::Ready(None)` on the very first poll.
    #[tokio::test]
    async fn empty_record_batch_stream_reports_schema_and_no_batches() {
        let schema: SchemaRef =
            Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, true)]));
        let mut stream = EmptyRecordBatchStream::new(Arc::clone(&schema));
        assert_eq!(stream.schema().fields().len(), 1);
        assert_eq!(stream.schema().field(0).name(), "a");
        assert!(stream.next().await.is_none());
    }

    /// `MemoryStream` yields every pre-loaded batch in order and reports
    /// `None` on the poll after the last batch.
    #[tokio::test]
    async fn memory_stream_yields_batches_in_order() {
        use futures::TryStreamExt;
        let schema = three_col_schema();
        let b1 = three_col_batch(&schema, &[1], &[10], &["x"]);
        let b2 = three_col_batch(&schema, &[2], &[20], &["y"]);
        let b3 = three_col_batch(&schema, &[3], &[30], &["z"]);
        let ms = MemoryStream::try_new(
            vec![b1.clone(), b2.clone(), b3.clone()],
            Arc::clone(&schema),
            None,
        )
        .unwrap();
        let collected: Vec<RecordBatch> = ms.try_collect().await.unwrap();
        assert_eq!(collected.len(), 3);
        assert_eq!(collected[0], b1);
        assert_eq!(collected[1], b2);
        assert_eq!(collected[2], b3);

        // A fresh stream is polled to `None` after the last batch — the
        // stream cannot yield a fourth batch even under repeated polls.
        let mut ms2 = MemoryStream::try_new(vec![b1.clone()], Arc::clone(&schema), None)
            .unwrap();
        assert!(ms2.next().await.is_some());
        assert!(ms2.next().await.is_none());
        assert!(ms2.next().await.is_none());
    }

    /// `MemoryStream` with a projection yields only the selected columns
    /// and reports the projected schema.
    #[tokio::test]
    async fn memory_stream_applies_projection() {
        use futures::TryStreamExt;
        let full_schema = three_col_schema();
        let batch = three_col_batch(&full_schema, &[1, 2], &[10, 20], &["x", "y"]);

        // Project columns 0 (a: Int32) and 2 (c: Utf8), skipping b.
        let projection = vec![0usize, 2];
        let projected_schema: SchemaRef =
            Arc::new(full_schema.project(&projection).unwrap());
        let ms = MemoryStream::try_new(
            vec![batch.clone()],
            Arc::clone(&projected_schema),
            Some(projection.clone()),
        )
        .unwrap();
        assert_eq!(ms.schema().fields().len(), 2);
        assert_eq!(ms.schema().field(0).name(), "a");
        assert_eq!(ms.schema().field(1).name(), "c");

        let collected: Vec<RecordBatch> = ms.try_collect().await.unwrap();
        assert_eq!(collected.len(), 1);
        assert_eq!(collected[0].num_columns(), 2);
        assert_eq!(collected[0].schema().field(0).name(), "a");
        assert_eq!(collected[0].schema().field(1).name(), "c");
    }

    /// `MemoryStream::try_new` rejects an out-of-bounds projection index
    /// at construction rather than deferring the error to `poll_next`.
    #[tokio::test]
    async fn memory_stream_try_new_rejects_out_of_bounds_projection() {
        let schema = three_col_schema();
        let batch = three_col_batch(&schema, &[1], &[10], &["x"]);
        // Column index 42 does not exist — expect an error.
        let err = MemoryStream::try_new(vec![batch], Arc::clone(&schema), Some(vec![42]))
            .err()
            .expect("expected out-of-bounds projection to fail try_new");
        // The concrete variant is `ArrowError(SchemaError)` — check the
        // rendered form so the test doesn't couple to the enum's Debug
        // layout.
        let msg = format!("{err}");
        assert!(
            msg.contains("out of bounds") || msg.contains("42"),
            "unexpected error rendering: {msg}"
        );
    }
}
