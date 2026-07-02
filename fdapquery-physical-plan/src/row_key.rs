//! A hashable key built from a row's values, shared by `AggregateExec`
//! (group keys) and `HashJoinExec` (join keys).
//!
//! Floats are hashed and compared **by bit pattern**, so `Hash` and `Eq`
//! agree (and `NaN` keys group together). See ARCHITECTURE §4.6 for the
//! rationale for sharing one implementation across both operators.

// `ScalarValue` canonical home is `fdapquery-common`
// but this crate doesn't list it as a dep. Use the transitional
// re-export from `fdapquery-datatypes`. (Migrate to direct import
// when physical-plan adds `fdapquery-common` to Cargo.toml or when
// the wrappers in 15d-2 reshape the dep graph.)
use fdapquery_common::ScalarValue;
use std::hash::{Hash, Hasher};

/// A row's key values, usable as a `HashMap`/`HashSet` key.
#[derive(Clone, Debug)]
pub(crate) struct RowKey(pub Vec<ScalarValue>);

impl PartialEq for RowKey {
    fn eq(&self, other: &Self) -> bool {
        self.0.len() == other.0.len() && self.0.iter().zip(&other.0).all(|(a, b)| scalar_eq(a, b))
    }
}

impl Eq for RowKey {}

impl Hash for RowKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for v in &self.0 {
            hash_scalar(v, state);
        }
    }
}

/// Equality for key scalars: bit-equality for floats (so `NaN == NaN`, agreeing
/// with [`hash_scalar`]); the derived `PartialEq` for everything else.
fn scalar_eq(a: &ScalarValue, b: &ScalarValue) -> bool {
    use ScalarValue::{Float32, Float64};
    match (a, b) {
        (Float32(x), Float32(y)) => x.to_bits() == y.to_bits(),
        (Float64(x), Float64(y)) => x.to_bits() == y.to_bits(),
        _ => a == b,
    }
}

/// Hash one scalar: the variant discriminant plus the value's bytes (floats by
/// bit pattern, so equal-keyed floats hash equally).
fn hash_scalar<H: Hasher>(v: &ScalarValue, state: &mut H) {
    use ScalarValue::{Null, Boolean, Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64, Float32, Float64, Utf8, Binary, Date32};
    std::mem::discriminant(v).hash(state);
    match v {
        Null => {}
        Boolean(b) => b.hash(state),
        Int8(n) => n.hash(state),
        Int16(n) => n.hash(state),
        Int32(n) => n.hash(state),
        Int64(n) => n.hash(state),
        UInt8(n) => n.hash(state),
        UInt16(n) => n.hash(state),
        UInt32(n) => n.hash(state),
        UInt64(n) => n.hash(state),
        Float32(f) => f.to_bits().hash(state),
        Float64(f) => f.to_bits().hash(state),
        Utf8(s) => s.hash(state),
        Binary(b) => b.hash(state),
        Date32(d) => d.hash(state),
    }
}
