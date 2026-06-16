//! Identifies a single shuffle output by job + stage + partition. The protobuf
//! message comment below shows the equivalent wire representation.

/*
 message ShuffleId {
   string job_uuid = 1;
   uint32 stage_id = 2;
   uint32 partition_id = 4;
 }
*/

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ShuffleId {
    pub job_uuid: String,
    pub stage_id: i32,
    pub partition_id: i32,
}

impl ShuffleId {
    pub fn new(job_uuid: String, stage_id: i32, partition_id: i32) -> Self {
        Self {
            job_uuid,
            stage_id,
            partition_id,
        }
    }
}
