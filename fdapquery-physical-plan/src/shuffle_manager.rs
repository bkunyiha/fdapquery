//!
//! Manages local shuffle storage for distributed execution, laid out as
//! `{base_dir}/{job_uuid}/{stage_id}/partition_{partition_id}.arrow`.
//!
//! ## Implementation — Arrow IPC file format
//! `write_partition` / `read_partition` use the `arrow-ipc` crate's
//! `FileWriter` / `FileReader` to serialise/deserialise `RecordBatch` data to
//! the Arrow IPC random-access file format.
//!
//! ## Status of the callers
//! `ShuffleWriterExec::execute` and `ShuffleReaderExec::execute` are still
//! stubbed with `unimplemented!()` — they need an executor context
//! (`ExecutorContext`-style type carrying the per-executor `ShuffleManager`
//! instance and identity) that lives in `flight-server` (module 13). The
//! manager itself is fully functional and unit-tested standalone.

use arrow_ipc::reader::FileReader;
use arrow_ipc::writer::FileWriter;
use datatypes::RecordBatch;
use std::fs::{File, create_dir_all};
use std::io::BufReader;
use std::path::PathBuf;

/// Local shuffle-file storage manager.
pub struct ShuffleManager {
    pub base_dir: String,
}

impl Default for ShuffleManager {
    fn default() -> Self {
        // Default shuffle-spill directory.
        Self::new("/tmp/rquery-shuffle")
    }
}

impl ShuffleManager {
    pub fn new(base_dir: impl Into<String>) -> Self {
        Self {
            base_dir: base_dir.into(),
        }
    }

    /// Write a partition's batches to a local Arrow IPC file.
    ///
    /// No-op on an empty batch list. The partition directory is created if it
    /// doesn't exist. All batches are assumed to share the schema of `batches[0]`.
    pub fn write_partition(
        &self,
        job_uuid: &str,
        stage_id: i32,
        partition_id: i32,
        batches: &[RecordBatch],
    ) {
        if batches.is_empty() {
            return;
        }
        let dir = self.partition_dir(job_uuid, stage_id);
        create_dir_all(&dir).unwrap_or_else(|e| {
            panic!("failed to create shuffle directory {}: {e}", dir.display())
        });
        let file_path = dir.join(format!("partition_{partition_id}.arrow"));
        let file = File::create(&file_path).unwrap_or_else(|e| {
            panic!("failed to create shuffle file {}: {e}", file_path.display())
        });

        // The schema is taken from the first batch.
        let schema = batches[0].schema();
        let mut writer =
            FileWriter::try_new(file, &schema).expect("failed to initialise Arrow IPC FileWriter");
        for batch in batches {
            writer
                .write(batch)
                .expect("failed to write batch to Arrow IPC file");
        }
        writer.finish().expect("failed to finalise Arrow IPC file");
    }

    /// Read a partition's batches from local Arrow IPC storage.
    ///
    /// Returns a boxed iterator (matching the rest of the `PhysicalPlan`
    /// shape). Each yielded batch is fully materialised before the next is
    /// pulled — the iterator drains the on-disk file lazily.
    pub fn read_partition(
        &self,
        job_uuid: &str,
        stage_id: i32,
        partition_id: i32,
    ) -> Box<dyn Iterator<Item = RecordBatch>> {
        let file_path = self.get_partition_file(job_uuid, stage_id, partition_id);
        if !file_path.exists() {
            panic!("Shuffle file not found: {}", file_path.display());
        }
        let file = File::open(&file_path)
            .unwrap_or_else(|e| panic!("failed to open shuffle file {}: {e}", file_path.display()));
        // BufReader because FileReader does many small reads while parsing the
        // IPC footer/dictionaries.
        let reader = FileReader::try_new(BufReader::new(file), None)
            .expect("failed to initialise Arrow IPC FileReader");
        // FileReader yields `Result<RecordBatch>`; unwrap each batch (matches
        // the panic-style error handling used everywhere else in rquery).
        Box::new(reader.map(|r| r.expect("failed to read shuffle batch")))
    }

    /// Path of a shuffle partition file.
    pub fn get_partition_file(&self, job_uuid: &str, stage_id: i32, partition_id: i32) -> PathBuf {
        self.partition_dir(job_uuid, stage_id)
            .join(format!("partition_{partition_id}.arrow"))
    }

    /// Directory for one job/stage.
    fn partition_dir(&self, job_uuid: &str, stage_id: i32) -> PathBuf {
        PathBuf::from(&self.base_dir)
            .join(job_uuid)
            .join(stage_id.to_string())
    }

    /// Remove all shuffle data for a job.
    pub fn cleanup_job(&self, job_uuid: &str) {
        let dir = PathBuf::from(&self.base_dir).join(job_uuid);
        if dir.exists() {
            let _ = std::fs::remove_dir_all(&dir);
        }
    }

    /// Remove all shuffle data.
    pub fn cleanup_all(&self) {
        let dir = PathBuf::from(&self.base_dir);
        if dir.exists() {
            let _ = std::fs::remove_dir_all(&dir);
        }
    }
}

#[cfg(test)]
mod tests {
    //! Standalone round-trip test for `ShuffleManager`. The manager is fully
    //! self-contained and worth proving correct independently of the Flight
    //! executor that exercises it end-to-end.
    use super::*;
    use arrow_array::{Int32Array, StringArray};
    use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
    use std::sync::Arc;

    /// Use a per-test directory under `/tmp/` keyed by nanoseconds so parallel
    /// `cargo test` runs don't collide on disk.
    fn temp_dir(tag: &str) -> String {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        format!("/tmp/rquery-shuffle-test-{tag}-{nanos}")
    }

    #[test]
    fn write_then_read_round_trips_batches() {
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("id", DataType::Int32, false),
            ArrowField::new("name", DataType::Utf8, false),
        ]));
        let batch1 = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3])),
                Arc::new(StringArray::from(vec!["a", "b", "c"])),
            ],
        )
        .unwrap();
        let batch2 = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![4, 5])),
                Arc::new(StringArray::from(vec!["d", "e"])),
            ],
        )
        .unwrap();

        let base = temp_dir("round-trip");
        let mgr = ShuffleManager::new(&base);
        let job_uuid = "test-job-A";
        let stage_id = 0;
        let partition_id = 2;

        mgr.write_partition(
            job_uuid,
            stage_id,
            partition_id,
            &[batch1.clone(), batch2.clone()],
        );
        let read_back: Vec<RecordBatch> = mgr
            .read_partition(job_uuid, stage_id, partition_id)
            .collect();

        assert_eq!(read_back.len(), 2);
        assert_eq!(read_back[0], batch1);
        assert_eq!(read_back[1], batch2);

        mgr.cleanup_all();
    }

    #[test]
    fn write_partition_with_empty_batches_is_a_noop() {
        let base = temp_dir("empty");
        let mgr = ShuffleManager::new(&base);
        mgr.write_partition("test-job-B", 0, 0, &[]);
        // No file should have been created — `read_partition` panics on missing files.
        assert!(!mgr.get_partition_file("test-job-B", 0, 0).exists());
        mgr.cleanup_all();
    }
}
