//! Memory-size estimation for [`RecordBatch`].
//!
//! Strict mirror of DataFusion's
//! `datafusion/common/src/utils/memory.rs` — specifically the
//! [`get_record_batch_memory_size`] entry point and its
//! [`RecordBatchMemoryCounter`] companion that DataFusion's metrics layer
//! (`physical-expr-common::metrics::baseline::RecordOutput`) depends on
//! to populate `output_bytes`.
//!
//! Divergences from upstream:
//!
//! - DataFusion uses `datafusion_common::HashSet`, an alias for
//!   `hashbrown::HashSet`. fdapquery uses `std::collections::HashSet`
//!   here to avoid pulling `hashbrown` into the workspace; semantics are
//!   identical because the set is only used to dedupe `NonZero<usize>`
//!   buffer pointers and only `insert` is called.
//! - The `estimate_memory_size` helper and its DataFusionError-returning
//!   plumbing are not mirrored — fdapquery doesn't yet need the
//!   hashtable-sizing path; only `get_record_batch_memory_size` is
//!   required by the `RecordOutput` impls. Add the helper here when
//!   the first call site appears.

use arrow_array::Array;
use arrow_array::RecordBatch;
use arrow_data::ArrayData;
use std::collections::HashSet;
use std::num::NonZero;

/// Calculate total used memory of this batch.
///
/// This function is used to estimate the physical memory usage of the
/// `RecordBatch`. It only counts the memory of large data `Buffer`s, and
/// ignores metadata like types and pointers.
/// The implementation will add up all unique `Buffer`'s memory size, due
/// to:
/// - The data pointer inside `Buffer` are memory regions returned by
///   global memory allocator, those regions can't have overlap.
/// - The actual used range of `ArrayRef`s inside `RecordBatch` can have
///   overlap or reuse the same `Buffer`. For example: taking a slice from
///   `Array`.
///
/// Example:
/// For a `RecordBatch` with two columns: `col1` and `col2`, two columns
/// are pointing to a sub-region of the same buffer.
///
/// {xxxxxxxxxxxxxxxxxxx} <--- buffer
///       ^    ^  ^    ^
///       |    |  |    |
/// col1->{    }  |    |
/// col2--------->{    }
///
/// In the above case, `get_record_batch_memory_size` will return the
/// size of the buffer, instead of the sum of `col1` and `col2`'s actual
/// memory size.
///
/// Note: Current `RecordBatch.get_array_memory_size()` will double count
/// the buffer memory size if multiple arrays within the batch are
/// sharing the same `Buffer`. This method provides temporary fix until
/// the issue is resolved:
/// <https://github.com/apache/arrow-rs/issues/6439>
pub fn get_record_batch_memory_size(batch: &RecordBatch) -> usize {
    RecordBatchMemoryCounter::new().count_batch(batch)
}

/// Tracks the memory used by a sequence of [`RecordBatch`]es that may
/// share underlying buffers, counting each buffer exactly once.
///
/// Use this instead of [`get_record_batch_memory_size`] to account for
/// the total memory of a sequence of batches, e.g. when buffering the
/// batches of an input stream. Such batches can share buffers (for
/// example, operators like aggregates emit one large batch as multiple
/// zero-copy slices), and calling [`get_record_batch_memory_size`] per
/// batch counts the shared buffers once per batch, while this counter
/// counts them exactly once. A batch's buffers are kept alive by the
/// batch even when only a sub-range is referenced, so counting unique
/// buffers in full reflects the memory the batches actually retain.
#[derive(Debug, Default)]
pub struct RecordBatchMemoryCounter {
    /// Start addresses of `Buffer`s that have already been counted
    /// (instead of actual used data region's pointer represented by
    /// current `Array`)
    counted_buffers: HashSet<NonZero<usize>>,
    /// Total memory of all unique buffers counted so far
    memory_usage: usize,
}

impl RecordBatchMemoryCounter {
    pub fn new() -> Self {
        Self::default()
    }

    /// Count `batch`, returning the memory used by its buffers that have
    /// not been counted before.
    pub fn count_batch(&mut self, batch: &RecordBatch) -> usize {
        let mut total_size = 0;

        for array in batch.columns() {
            let array_data = array.to_data();
            count_array_data_memory_size(&array_data, &mut self.counted_buffers, &mut total_size);
        }

        self.memory_usage += total_size;
        total_size
    }

    /// Total memory of the unique buffers of all batches counted so far.
    pub fn memory_usage(&self) -> usize {
        self.memory_usage
    }
}

/// Count the memory usage of `array_data` and its children recursively.
fn count_array_data_memory_size(
    array_data: &ArrayData,
    counted_buffers: &mut HashSet<NonZero<usize>>,
    total_size: &mut usize,
) {
    // Count memory usage for `array_data`
    for buffer in array_data.buffers() {
        if counted_buffers.insert(buffer.data_ptr().addr()) {
            *total_size += buffer.capacity();
        } // Otherwise the buffer's memory is already counted
    }

    if let Some(null_buffer) = array_data.nulls()
        && counted_buffers.insert(null_buffer.inner().inner().data_ptr().addr())
    {
        *total_size += null_buffer.inner().inner().capacity();
    }

    // Count all children `ArrayData` recursively
    for child in array_data.child_data() {
        count_array_data_memory_size(child, counted_buffers, total_size);
    }
}

#[cfg(test)]
mod record_batch_tests {
    use super::*;
    use arrow_array::{Float64Array, Int32Array, ListArray};
    use arrow_schema::{DataType, Field, Schema};
    use std::sync::Arc;

    #[test]
    fn test_get_record_batch_memory_size() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("ints", DataType::Int32, true),
            Field::new("float64", DataType::Float64, false),
        ]));

        let int_array = Int32Array::from(vec![Some(1), Some(2), Some(3), Some(4), Some(5)]);
        let float64_array = Float64Array::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(int_array), Arc::new(float64_array)])
                .unwrap();

        let size = get_record_batch_memory_size(&batch);
        assert_eq!(size, 60);
    }

    #[test]
    fn test_get_record_batch_memory_size_with_null() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("ints", DataType::Int32, true),
            Field::new("float64", DataType::Float64, false),
        ]));

        let int_array = Int32Array::from(vec![None, Some(2), Some(3)]);
        let float64_array = Float64Array::from(vec![1.0, 2.0, 3.0]);

        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(int_array), Arc::new(float64_array)])
                .unwrap();

        let size = get_record_batch_memory_size(&batch);
        assert_eq!(size, 100);
    }

    #[test]
    fn test_get_record_batch_memory_size_empty() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "ints",
            DataType::Int32,
            false,
        )]));

        let int_array: Int32Array = Int32Array::from(vec![] as Vec<i32>);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(int_array)]).unwrap();

        let size = get_record_batch_memory_size(&batch);
        assert_eq!(size, 0, "Empty batch should have 0 memory size");
    }

    #[test]
    fn test_get_record_batch_memory_size_shared_buffer() {
        let original = Int32Array::from(vec![1, 2, 3, 4, 5]);
        let slice1 = original.slice(0, 3);
        let slice2 = original.slice(2, 3);

        let schema_origin = Arc::new(Schema::new(vec![Field::new(
            "origin_col",
            DataType::Int32,
            false,
        )]));
        let batch_origin = RecordBatch::try_new(schema_origin, vec![Arc::new(original)]).unwrap();

        let schema = Arc::new(Schema::new(vec![
            Field::new("slice1", DataType::Int32, false),
            Field::new("slice2", DataType::Int32, false),
        ]));

        let batch_sliced =
            RecordBatch::try_new(schema, vec![Arc::new(slice1), Arc::new(slice2)]).unwrap();

        let size_origin = get_record_batch_memory_size(&batch_origin);
        let size_sliced = get_record_batch_memory_size(&batch_sliced);

        assert_eq!(size_origin, size_sliced);
    }

    #[test]
    fn test_record_batch_memory_counter_buffer_shared_across_batches() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "ints",
            DataType::Int32,
            false,
        )]));

        let int_array = Int32Array::from(vec![1, 2, 3, 4, 5, 6]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(int_array)]).unwrap();
        let slices = [batch.slice(0, 2), batch.slice(2, 2), batch.slice(4, 2)];

        // Counting each slice individually counts the shared buffer once per slice
        let summed: usize = slices.iter().map(get_record_batch_memory_size).sum();
        assert_eq!(summed, 3 * get_record_batch_memory_size(&batch));

        // A counter shared across the batches counts it exactly once
        let mut counter = RecordBatchMemoryCounter::new();
        let deduped: usize = slices.iter().map(|slice| counter.count_batch(slice)).sum();
        assert_eq!(deduped, get_record_batch_memory_size(&batch));
        assert_eq!(counter.memory_usage(), get_record_batch_memory_size(&batch));
    }

    #[test]
    fn test_get_record_batch_memory_size_nested_array() {
        use arrow_array::types::Int32Type;

        let schema = Arc::new(Schema::new(vec![
            Field::new(
                "nested_int",
                DataType::List(Arc::new(Field::new_list_field(DataType::Int32, true))),
                false,
            ),
            Field::new(
                "nested_int2",
                DataType::List(Arc::new(Field::new_list_field(DataType::Int32, true))),
                false,
            ),
        ]));

        let int_list_array = ListArray::from_iter_primitive::<Int32Type, _, _>(vec![Some(vec![
            Some(1),
            Some(2),
            Some(3),
        ])]);

        let int_list_array2 = ListArray::from_iter_primitive::<Int32Type, _, _>(vec![Some(vec![
            Some(4),
            Some(5),
            Some(6),
        ])]);

        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(int_list_array), Arc::new(int_list_array2)],
        )
        .unwrap();

        let size = get_record_batch_memory_size(&batch);
        assert_eq!(size, 8208);
    }
}
