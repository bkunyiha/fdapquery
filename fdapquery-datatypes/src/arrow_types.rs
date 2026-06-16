//!
//! Named Arrow `DataType` constants — the type vocabulary the engine uses
//! throughout. Exposed as a module of `pub const` values backed by
//! `arrow_schema::DataType`. Usage: `arrow_types::INT32_TYPE` (and similar).
//!
//! Note on naming: constants follow Rust convention (`INT32_TYPE`,
//! SCREAMING_SNAKE_CASE) rather than the PascalCase the rest of the
//! workspace uses for types. The module-of-constants form was chosen
//! because `DataType` variants are `const`-constructible, so a unit struct
//! would add no value.

use arrow_schema::{DataType, IntervalUnit, TimeUnit};

pub const BOOLEAN_TYPE: DataType = DataType::Boolean;

pub const INT8_TYPE: DataType = DataType::Int8;
pub const INT16_TYPE: DataType = DataType::Int16;
pub const INT32_TYPE: DataType = DataType::Int32;
pub const INT64_TYPE: DataType = DataType::Int64;

pub const UINT8_TYPE: DataType = DataType::UInt8;
pub const UINT16_TYPE: DataType = DataType::UInt16;
pub const UINT32_TYPE: DataType = DataType::UInt32;
pub const UINT64_TYPE: DataType = DataType::UInt64;

pub const FLOAT_TYPE: DataType = DataType::Float32;
pub const DOUBLE_TYPE: DataType = DataType::Float64;

pub const STRING_TYPE: DataType = DataType::Utf8;
pub const BINARY_TYPE: DataType = DataType::Binary;

pub const DATE_DAY_TYPE: DataType = DataType::Date32;

// `IntervalUnit` is not const-constructible inside a const context in some arrow
// versions; expose this as a `pub fn` returning the value if `const` evaluation
// fails to compile, but in arrow 55 the variant *is* const-friendly.
pub const INTERVAL_DAY_TIME_TYPE: DataType = DataType::Interval(IntervalUnit::DayTime);

// Convenience — gives downstream crates a single import path for every
// commonly-used `DataType`.
pub const TIMESTAMP_MICRO_TYPE: DataType = DataType::Timestamp(TimeUnit::Microsecond, None);
