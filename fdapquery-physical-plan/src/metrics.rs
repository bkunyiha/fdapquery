// Strict-mirror discipline: every shape here matches DataFusion's metrics
// module, including struct fields whose names start with the struct's name
// (`metric_type: MetricType`, `metric_category: MetricCategory`) and match
// arms that look textually duplicated but split the variants by metric
// kind for byte-equivalent rendering with DataFusion. Merging the arms or
// renaming the fields would diverge from DataFusion's source.
#![allow(clippy::struct_field_names, clippy::match_same_arms)]

//! Operator-level metrics — strict mirror of
//! `datafusion::physical_plan::metrics` (and the underlying
//! `datafusion-physical-expr-common::metrics`).
//!
//! Reads the canonical implementation from DataFusion 54.0.0:
//!   * <https://github.com/apache/datafusion/blob/54.0.0/datafusion/physical-expr-common/src/metrics/mod.rs>
//!   * <https://github.com/apache/datafusion/blob/54.0.0/datafusion/physical-expr-common/src/metrics/value.rs>
//!   * <https://github.com/apache/datafusion/blob/54.0.0/datafusion/physical-expr-common/src/metrics/custom.rs>
//!   * <https://github.com/apache/datafusion/blob/54.0.0/datafusion/physical-expr-common/src/metrics/builder.rs>
//!   * <https://github.com/apache/datafusion/blob/54.0.0/datafusion/physical-expr-common/src/metrics/baseline.rs>
//!   * <https://github.com/apache/datafusion/blob/54.0.0/datafusion/physical-plan/src/metrics.rs>
//!
//! ## Shape mirrored
//!
//! - [`MetricValue`] is an **enum** with every variant DataFusion exposes
//!   in `value.rs`: `OutputRows`, `ElapsedCompute`, `SpillCount`,
//!   `SpilledBytes`, `OutputBytes`, `OutputBatches`, `SpilledRows`,
//!   `CurrentMemoryUsage`, `Count`, `Gauge`, `PeakMemoryUsage`, `Time`,
//!   `StartTimestamp`, `EndTimestamp`, `PruningMetrics`, `Ratio`, and
//!   `Custom { name, value: Arc<dyn CustomMetricValue> }`.
//! - [`CustomMetricValue`] is the dyn-trait DataFusion exposes in
//!   `custom.rs` for application-defined metric values.
//! - [`Metric`] wraps a [`MetricValue`] with optional `partition`,
//!   `metric_type`, `metric_category`, and labels.
//! - [`Label`] / [`LabelValue`] mirror DataFusion's static/Arc-string
//!   sharing layout in `mod.rs`.
//! - [`MetricsSet`] stores `Arc<Metric>` values and exposes the full
//!   chain DataFusion's display path consumes.
//! - [`MetricBuilder`], [`BaselineMetrics`], [`SpillMetrics`],
//!   [`SplitMetrics`], and [`ExecutionPlanMetricsSet`] mirror the
//!   construction/aggregation surfaces.
//!
//! ## Divergences (documented; minimal)
//!
//! - DataFusion uses `parking_lot::Mutex<Option<DateTime<Utc>>>` for
//!   [`Timestamp`]; fdapquery uses `std::sync::Mutex<Option<DateTime<Utc>>>`
//!   to avoid pulling `parking_lot` into the workspace.  `DateTime<Utc>`
//!   is `Copy`, so the poisoning paths cannot leave a half-modified state.
//! - [`BaselineMetrics::output_rows_skew_metric`] is included as a strict
//!   mirror; the helper `output_rows_skew_score` is private and matches
//!   DataFusion's algorithm byte-for-byte.
//!
//! ## Aggregation semantics
//!
//! Per-variant aggregation matches DataFusion's `MetricValue::aggregate`
//! in `value.rs`:
//!
//! - Counter variants (`OutputRows`, `SpillCount`, `SpilledBytes`,
//!   `OutputBytes`, `OutputBatches`, `SpilledRows`, `Count { … }`) ADD.
//! - Gauge variants (`CurrentMemoryUsage`, `Gauge { … }`,
//!   `PeakMemoryUsage { … }`) ADD.
//! - Time variants (`ElapsedCompute`, `Time { … }`) ADD durations.
//! - `StartTimestamp` aggregates by MIN; `EndTimestamp` aggregates by MAX.
//! - `PruningMetrics` adds component-wise; `Ratio` defers to the
//!   configured merge strategy; `Custom` defers to
//!   [`CustomMetricValue::aggregate`].
//!
//! Cross-variant aggregation panics with `"Mismatched metric types. Can
//! not aggregate …"` — same message as DataFusion.

use arrow_array::RecordBatch;
use chrono::{DateTime, Utc};
use fdapquery_common::Result;
use fdapquery_common::utils::memory::get_record_batch_memory_size;
use std::any::Any;
use std::borrow::{Borrow, Cow};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt::{self, Debug, Display};
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use std::task::Poll;
use std::time::Duration;

// =============================================================================
// MetricCategory / MetricType (unchanged from previous shape).
// =============================================================================

/// Categorisation of an operator-level metric for display filtering.
/// Mirrors `datafusion::physical_plan::metrics::MetricCategory`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum MetricCategory {
    /// Row-count metrics.  Deterministic.
    Rows,
    /// Byte-count metrics.  Deterministic.
    Bytes,
    /// Wall-clock timing metrics.  Non-deterministic.
    Timing,
    /// Catch-all category for metrics that do not fit rows/bytes/timing —
    /// or that were registered without an explicit category. Filtering by
    /// category treats any [`Metric`] with `metric_category == None` as
    /// `Uncategorized`, so this variant is the bucket a caller must
    /// include in [`MetricsSet::filter_by_categories`] to keep timestamps,
    /// ratios, and other miscellaneous metrics.
    ///
    /// Strict mirror of DataFusion's `MetricCategory::Uncategorized`.
    Uncategorized,
}

/// Coarse partition of metrics by "noisiness".  Mirrors
/// `datafusion::physical_plan::metrics::MetricType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MetricType {
    /// Always-on, user-facing summary metrics.
    Summary,
    /// Implementation-detail metrics, useful when developing or tuning.
    Dev,
}

// =============================================================================
// Count / Gauge / Time / Timestamp — atomic primitives the enum variants wrap.
// =============================================================================

/// A counter to record things such as number of input or output rows.
///
/// Note `clone`ing counters update the same underlying metrics.  Mirrors
/// DataFusion's `Count`.
#[derive(Debug, Clone)]
pub struct Count {
    value: Arc<AtomicUsize>,
}

impl Default for Count {
    fn default() -> Self {
        Self::new()
    }
}

impl Count {
    pub fn new() -> Self {
        Self {
            value: Arc::new(AtomicUsize::new(0)),
        }
    }
    pub fn add(&self, n: usize) {
        self.value.fetch_add(n, AtomicOrdering::Relaxed);
    }
    pub fn value(&self) -> usize {
        self.value.load(AtomicOrdering::Relaxed)
    }
}

impl PartialEq for Count {
    fn eq(&self, other: &Self) -> bool {
        self.value() == other.value()
    }
}

impl Display for Count {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", human_readable_count(self.value()))
    }
}

/// A metric that holds a single instantaneous numeric value that can go
/// up, down, or be set to an absolute reading — the analogue of a
/// Prometheus gauge. Operators reach for this to expose things like
/// "current memory usage in bytes" or "queue depth" via
/// [`MetricValue::Gauge`] / [`MetricValue::CurrentMemoryUsage`], where
/// the latest sample is what matters and history is not tracked. Contrast
/// with [`Count`], which is monotonic. Cloning shares the underlying
/// atomic, so a [`Gauge`] handed to worker threads updates the same slot
/// the metrics set will read.
///
/// Strict mirror of DataFusion's `Gauge`.
#[derive(Debug, Clone)]
pub struct Gauge {
    value: Arc<AtomicUsize>,
}

impl Default for Gauge {
    fn default() -> Self {
        Self::new()
    }
}

impl Gauge {
    pub fn new() -> Self {
        Self {
            value: Arc::new(AtomicUsize::new(0)),
        }
    }
    pub fn add(&self, n: usize) {
        self.value.fetch_add(n, AtomicOrdering::Relaxed);
    }
    pub fn sub(&self, n: usize) {
        self.value.fetch_sub(n, AtomicOrdering::Relaxed);
    }
    pub fn set_max(&self, n: usize) {
        self.value.fetch_max(n, AtomicOrdering::Relaxed);
    }
    pub fn set(&self, n: usize) -> usize {
        self.value.swap(n, AtomicOrdering::Relaxed)
    }
    pub fn value(&self) -> usize {
        self.value.load(AtomicOrdering::Relaxed)
    }
}

impl PartialEq for Gauge {
    fn eq(&self, other: &Self) -> bool {
        self.value() == other.value()
    }
}

impl Display for Gauge {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.value())
    }
}

/// Accumulator for a duration measured across many separate intervals —
/// the metric that backs `elapsed_compute` and every operator-defined
/// timing metric. Sample it repeatedly by calling [`Time::timer`] to
/// obtain a [`ScopedTimerGuard`] whose lifetime records one interval,
/// or by directly calling [`Time::add_elapsed`] / [`Time::add_duration`].
/// Any non-zero duration is rounded up to at least 1 ns so a measured
/// event is distinguishable from "no event recorded". Cloning shares the
/// underlying atomic so timers spawned on worker threads all fold into
/// the same total.
///
/// Strict mirror of DataFusion's `Time`.
#[derive(Debug, Clone)]
pub struct Time {
    nanos: Arc<AtomicUsize>,
}

impl Default for Time {
    fn default() -> Self {
        Self::new()
    }
}

impl Time {
    pub fn new() -> Self {
        Self {
            nanos: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Add elapsed nanoseconds since `start` to self.
    pub fn add_elapsed(&self, start: std::time::Instant) {
        self.add_duration(start.elapsed());
    }

    pub fn add_duration(&self, duration: Duration) {
        // Mirrors DataFusion: round any non-zero duration up to ≥1 nanosecond
        // so a measured event is distinguishable from "no event recorded".
        let more_nanos = duration.as_nanos() as usize;
        self.nanos
            .fetch_add(more_nanos.max(1), AtomicOrdering::Relaxed);
    }
    pub fn add(&self, other: &Time) {
        self.add_duration(Duration::from_nanos(other.value() as u64));
    }
    pub fn value(&self) -> usize {
        self.nanos.load(AtomicOrdering::Relaxed)
    }

    /// Start a scoped timer. The returned [`ScopedTimerGuard`] records
    /// the interval from `Instant::now()` until it is dropped (or until
    /// [`ScopedTimerGuard::stop`] / [`ScopedTimerGuard::done`] is called)
    /// into this [`Time`]. This is the idiomatic way to measure a block
    /// of work: `let _t = time.timer();` at the top of a function or
    /// `while let Some(_) = ...` loop body, and the elapsed time is
    /// folded in when the guard falls out of scope.
    ///
    /// Strict mirror of DataFusion's `Time::timer`.
    pub fn timer(&self) -> ScopedTimerGuard<'_> {
        ScopedTimerGuard {
            inner: self,
            start: Some(std::time::Instant::now()),
        }
    }

    /// Return a scoped guard that adds the elapsed time between the
    /// given instant and its drop / call to `stop` to this metric.
    pub fn timer_with(&self, now: std::time::Instant) -> ScopedTimerGuard<'_> {
        ScopedTimerGuard {
            inner: self,
            start: Some(now),
        }
    }
}

impl PartialEq for Time {
    fn eq(&self, other: &Self) -> bool {
        self.value() == other.value()
    }
}

impl Display for Time {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", human_readable_duration(self.value() as u64))
    }
}

/// RAII handle produced by [`Time::timer`] that folds the interval from
/// its creation to its `Drop` into a borrowed [`Time`] metric. Used to
/// wrap a scope of work — a function body, a loop iteration, a poll of
/// a stream — so that the elapsed time is recorded exactly once, even
/// on early returns or `?` unwinding. Call [`ScopedTimerGuard::stop`]
/// to record and reset without dropping (for repeated measurement in
/// one scope) or [`ScopedTimerGuard::done`] to record and consume the
/// guard explicitly.
///
/// Strict mirror of DataFusion's `ScopedTimerGuard`.
pub struct ScopedTimerGuard<'a> {
    inner: &'a Time,
    start: Option<std::time::Instant>,
}

impl ScopedTimerGuard<'_> {
    /// Stop the timer and record the time taken.
    pub fn stop(&mut self) {
        if let Some(start) = self.start.take() {
            self.inner.add_elapsed(start);
        }
    }

    /// Restart the timer recording from the current time.
    pub fn restart(&mut self) {
        self.start = Some(std::time::Instant::now());
    }

    /// Stop the timer, record the time taken, and consume self.
    pub fn done(mut self) {
        self.stop();
    }

    /// Stop the timer and record the time taken since `end_time`.
    pub fn stop_with(&mut self, end_time: std::time::Instant) {
        if let Some(start) = self.start.take() {
            let elapsed = end_time - start;
            self.inner.add_duration(elapsed);
        }
    }

    /// Stop the timer, record the time taken since `end_time`, and consume self.
    pub fn done_with(mut self, end_time: std::time::Instant) {
        self.stop_with(end_time);
    }
}

impl Drop for ScopedTimerGuard<'_> {
    fn drop(&mut self) {
        self.stop();
    }
}

/// A shared, mutable slot holding an optional wall-clock instant. Backs
/// the [`MetricValue::StartTimestamp`] and [`MetricValue::EndTimestamp`]
/// variants: an operator records the current time via
/// [`Timestamp::record`] when it begins and again when it finishes,
/// giving `EXPLAIN ANALYZE` output the interval during which each
/// partition ran. `None` means the moment has not been captured yet.
/// Cross-partition aggregation takes the MIN across `StartTimestamp`s
/// and the MAX across `EndTimestamp`s via [`Timestamp::update_to_min`]
/// and [`Timestamp::update_to_max`].
///
/// fdapquery uses `std::sync::Mutex` here instead of DataFusion's
/// `parking_lot::Mutex` to avoid pulling `parking_lot` into the workspace.
/// `DateTime<Utc>` is `Copy`, so the poisoning paths cannot leave a
/// half-modified state.
///
/// Strict mirror of DataFusion's `Timestamp`.
#[derive(Debug, Clone)]
pub struct Timestamp {
    timestamp: Arc<Mutex<Option<DateTime<Utc>>>>,
}

impl Default for Timestamp {
    fn default() -> Self {
        Self::new()
    }
}

impl Timestamp {
    /// Create a new timestamp with value `None`.
    pub fn new() -> Self {
        Self {
            timestamp: Arc::new(Mutex::new(None)),
        }
    }
    /// Sets the timestamp's value to the current time.  Mirrors
    /// DataFusion's `Timestamp::record`.
    pub fn record(&self) {
        self.set(Utc::now());
    }
    /// Sets the timestamp's value to a specified time.  Mirrors
    /// DataFusion's `Timestamp::set`.
    pub fn set(&self, now: DateTime<Utc>) {
        *self.timestamp.lock().unwrap() = Some(now);
    }
    /// Return the timestamp's value at the last time `record()` was
    /// called.  Returns `None` if `record()` has not been called.
    pub fn value(&self) -> Option<DateTime<Utc>> {
        *self.timestamp.lock().unwrap()
    }
    /// Sets the value of this timestamp to the minimum of this and other.
    pub fn update_to_min(&self, other: &Timestamp) {
        let min = match (self.value(), other.value()) {
            (None, None) => None,
            (Some(v), None) => Some(v),
            (None, Some(v)) => Some(v),
            (Some(v1), Some(v2)) => Some(if v1 < v2 { v1 } else { v2 }),
        };
        *self.timestamp.lock().unwrap() = min;
    }
    /// Sets the value of this timestamp to the maximum of this and other.
    pub fn update_to_max(&self, other: &Timestamp) {
        let max = match (self.value(), other.value()) {
            (None, None) => None,
            (Some(v), None) => Some(v),
            (None, Some(v)) => Some(v),
            (Some(v1), Some(v2)) => Some(if v1 < v2 { v2 } else { v1 }),
        };
        *self.timestamp.lock().unwrap() = max;
    }
}

impl PartialEq for Timestamp {
    fn eq(&self, other: &Self) -> bool {
        self.value() == other.value()
    }
}

impl Display for Timestamp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.value() {
            None => write!(f, "NONE"),
            Some(v) => write!(f, "{v}"),
        }
    }
}

// =============================================================================
// PruningMetrics — pruned/matched/fully_matched triple.  Mirrors
// DataFusion's `PruningMetrics`.
// =============================================================================

#[derive(Debug, Clone)]
pub struct PruningMetrics {
    pruned: Arc<AtomicUsize>,
    matched: Arc<AtomicUsize>,
    fully_matched: Arc<AtomicUsize>,
}

impl Default for PruningMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl PruningMetrics {
    pub fn new() -> Self {
        Self {
            pruned: Arc::new(AtomicUsize::new(0)),
            matched: Arc::new(AtomicUsize::new(0)),
            fully_matched: Arc::new(AtomicUsize::new(0)),
        }
    }
    pub fn add_pruned(&self, n: usize) {
        self.pruned.fetch_add(n, AtomicOrdering::Relaxed);
    }
    pub fn add_matched(&self, n: usize) {
        self.matched.fetch_add(n, AtomicOrdering::Relaxed);
    }
    pub fn add_fully_matched(&self, n: usize) {
        self.fully_matched.fetch_add(n, AtomicOrdering::Relaxed);
    }
    pub fn subtract_matched(&self, n: usize) {
        self.matched.fetch_sub(n, AtomicOrdering::Relaxed);
    }
    pub fn pruned(&self) -> usize {
        self.pruned.load(AtomicOrdering::Relaxed)
    }
    pub fn matched(&self) -> usize {
        self.matched.load(AtomicOrdering::Relaxed)
    }
    pub fn fully_matched(&self) -> usize {
        self.fully_matched.load(AtomicOrdering::Relaxed)
    }
}

impl Display for PruningMetrics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let matched = self.matched();
        let total = self.pruned() + matched;
        let fully = self.fully_matched();
        if fully != 0 {
            write!(
                f,
                "{} total → {} matched -> {} fully matched",
                human_readable_count(total),
                human_readable_count(matched),
                human_readable_count(fully)
            )
        } else {
            write!(
                f,
                "{} total → {} matched",
                human_readable_count(total),
                human_readable_count(matched)
            )
        }
    }
}

// =============================================================================
// RatioMetrics + RatioMergeStrategy — a `(part, total)` pair displayed as
// a percentage plus raw counts.  Used for scores such as "fraction of
// files pruned" and the output-rows skew metric, where the reader wants
// both the ratio and the underlying numerator/denominator.  Cross-
// partition merging follows the [`RatioMergeStrategy`] the metric was
// created with (see [`RatioMetrics::merge`]).  Strict mirror of
// DataFusion's `RatioMetrics`.
// =============================================================================

#[derive(Debug, Clone, Default)]
pub enum RatioMergeStrategy {
    #[default]
    AddPartAddTotal,
    AddPartSetTotal,
    SetPartAddTotal,
}

#[derive(Debug, Clone, Default)]
pub struct RatioMetrics {
    part: Arc<AtomicUsize>,
    total: Arc<AtomicUsize>,
    merge_strategy: RatioMergeStrategy,
    /// Ratios are displayed as `1% (1/100)`; this controls the latter part.
    display_raw_values: bool,
}

impl RatioMetrics {
    pub fn new() -> Self {
        Self {
            part: Arc::new(AtomicUsize::new(0)),
            total: Arc::new(AtomicUsize::new(0)),
            merge_strategy: RatioMergeStrategy::AddPartAddTotal,
            display_raw_values: true,
        }
    }
    pub fn with_merge_strategy(mut self, strategy: RatioMergeStrategy) -> Self {
        self.merge_strategy = strategy;
        self
    }
    pub fn with_display_raw_values(mut self, display_raw_values: bool) -> Self {
        self.display_raw_values = display_raw_values;
        self
    }
    pub fn add_part(&self, n: usize) {
        self.part.fetch_add(n, AtomicOrdering::Relaxed);
    }
    pub fn add_total(&self, n: usize) {
        self.total.fetch_add(n, AtomicOrdering::Relaxed);
    }
    pub fn set_part(&self, n: usize) {
        self.part.store(n, AtomicOrdering::Relaxed);
    }
    pub fn set_total(&self, n: usize) {
        self.total.store(n, AtomicOrdering::Relaxed);
    }
    pub fn part(&self) -> usize {
        self.part.load(AtomicOrdering::Relaxed)
    }
    pub fn total(&self) -> usize {
        self.total.load(AtomicOrdering::Relaxed)
    }
    pub fn merge_strategy(&self) -> &RatioMergeStrategy {
        &self.merge_strategy
    }
    pub fn display_raw_values(&self) -> bool {
        self.display_raw_values
    }
    pub fn merge(&self, other: &Self) {
        match self.merge_strategy {
            RatioMergeStrategy::AddPartAddTotal => {
                self.add_part(other.part());
                self.add_total(other.total());
            }
            RatioMergeStrategy::AddPartSetTotal => {
                self.add_part(other.part());
                self.set_total(other.total());
            }
            RatioMergeStrategy::SetPartAddTotal => {
                self.set_part(other.part());
                self.add_total(other.total());
            }
        }
    }
}

impl PartialEq for RatioMetrics {
    fn eq(&self, other: &Self) -> bool {
        self.part() == other.part()
            && self.total() == other.total()
            && self.display_raw_values == other.display_raw_values
    }
}

impl Display for RatioMetrics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let part = self.part();
        let total = self.total();
        if total == 0 {
            write!(f, "N/A")?;
        } else {
            let basis_points =
                (((part as u128 * 10_000) + (total as u128 / 2)) / total as u128) as usize;
            let whole = basis_points / 100;
            let frac = basis_points % 100;
            if frac == 0 {
                write!(f, "{whole}%")?;
            } else if frac % 10 == 0 {
                write!(f, "{whole}.{}%", frac / 10)?;
            } else {
                write!(f, "{whole}.{frac:02}%")?;
            }
        }

        if !self.display_raw_values {
            return Ok(());
        }

        if total == 0 {
            if part == 0 {
                write!(f, " (0/0)")
            } else {
                write!(f, " ({}/0)", human_readable_count(part))
            }
        } else {
            write!(
                f,
                " ({}/{})",
                human_readable_count(part),
                human_readable_count(total)
            )
        }
    }
}

// =============================================================================
// CustomMetricValue — strict mirror of
// `datafusion-physical-expr-common::metrics::custom::CustomMetricValue`.
// =============================================================================

/// A trait for implementing custom metric values.
///
/// Mirrors `datafusion_physical_expr_common::metrics::CustomMetricValue`
/// in `custom.rs`.  Application- or operator-specific metric types
/// implement this trait so they can be aggregated and displayed alongside
/// standard metrics.
pub trait CustomMetricValue: Display + Debug + Send + Sync {
    /// Returns a new, zero-initialised version of this metric value.
    fn new_empty(&self) -> Arc<dyn CustomMetricValue>;

    /// Merges another metric value into this one.  The type of `other`
    /// could be of a different custom type as long as it's aggregatable
    /// into self.
    fn aggregate(&self, other: Arc<dyn CustomMetricValue + 'static>);

    /// Returns this value as a `Any` to support dynamic downcasting.
    fn as_any(&self) -> &dyn Any;

    /// Optionally returns a numeric representation of the value, if
    /// meaningful.  Otherwise defaults to zero.
    fn as_usize(&self) -> usize {
        0
    }

    /// Compares this value with another custom value.
    fn is_eq(&self, other: &Arc<dyn CustomMetricValue>) -> bool;
}

// =============================================================================
// MetricValue — the **enum**.  Strict mirror of DataFusion's
// `MetricValue` in `physical-expr-common/src/metrics/value.rs`.
// =============================================================================

/// The typed payload carried by a [`Metric`] — the closed set of metric
/// shapes an operator can register. Each variant pairs a semantic name
/// (`"output_rows"`, `"elapsed_compute"`, …) with the atomic primitive
/// it wraps ([`Count`], [`Gauge`], [`Time`], [`Timestamp`],
/// [`PruningMetrics`], [`RatioMetrics`], or an
/// application-defined [`CustomMetricValue`]). Variants split into
/// well-known kinds (e.g. `OutputRows`, `ElapsedCompute`) whose display
/// name is fixed by DataFusion, and named kinds (`Count { name, .. }`,
/// `Time { name, .. }`, …) that the operator labels itself. Aggregation
/// across partitions is variant-specific — counters sum, timestamps
/// take min/max, custom values defer to their trait impl (see
/// [`MetricValue::aggregate`]).
///
/// Strict mirror of DataFusion's enum
/// `datafusion_physical_expr_common::metrics::MetricValue`.
#[derive(Debug, Clone)]
pub enum MetricValue {
    /// Number of output rows produced: "output_rows" metric.
    OutputRows(Count),
    /// Elapsed Compute Time: wall-clock time spent on CPU-intensive work.
    ElapsedCompute(Time),
    /// Number of spills produced.
    SpillCount(Count),
    /// Total size of spilled bytes.
    SpilledBytes(Count),
    /// Total size of output bytes.
    OutputBytes(Count),
    /// Total number of output batches produced.
    OutputBatches(Count),
    /// Total size of spilled rows.
    SpilledRows(Count),
    /// Current memory used.
    CurrentMemoryUsage(Gauge),
    /// Operator-defined counter.
    Count {
        name: Cow<'static, str>,
        count: Count,
    },
    /// Operator-defined gauge.
    Gauge {
        name: Cow<'static, str>,
        gauge: Gauge,
    },
    /// Operator-defined peak memory usage in bytes.
    PeakMemoryUsage {
        name: Cow<'static, str>,
        gauge: Gauge,
    },
    /// Operator-defined timing.
    Time { name: Cow<'static, str>, time: Time },
    /// Execution start timestamp.
    StartTimestamp(Timestamp),
    /// Execution end timestamp.
    EndTimestamp(Timestamp),
    /// Metrics related to scan pruning.
    PruningMetrics {
        name: Cow<'static, str>,
        pruning_metrics: PruningMetrics,
    },
    /// Metrics that should be displayed as a ratio.
    Ratio {
        name: Cow<'static, str>,
        ratio_metrics: RatioMetrics,
    },
    /// User-defined / extensible metric value.
    Custom {
        /// The provided name of this metric.
        name: Cow<'static, str>,
        /// A custom implementation of the metric value.
        value: Arc<dyn CustomMetricValue>,
    },
}

// PartialEq is implemented by hand rather than derived because the
// `Custom` variant carries an `Arc<dyn CustomMetricValue>`, which cannot
// derive `PartialEq`. Non-`Custom` variants compare their name + inner
// primitive; `Custom` defers to [`CustomMetricValue::is_eq`] so extension
// types decide their own equality semantics. All other cross-variant
// combinations are `false`.
//
// Strict mirror of DataFusion's manual
// `impl PartialEq for datafusion_physical_expr_common::metrics::MetricValue`.
impl PartialEq for MetricValue {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::OutputRows(a), Self::OutputRows(b)) => a == b,
            (Self::ElapsedCompute(a), Self::ElapsedCompute(b)) => a == b,
            (Self::SpillCount(a), Self::SpillCount(b)) => a == b,
            (Self::SpilledBytes(a), Self::SpilledBytes(b)) => a == b,
            (Self::OutputBytes(a), Self::OutputBytes(b)) => a == b,
            (Self::OutputBatches(a), Self::OutputBatches(b)) => a == b,
            (Self::SpilledRows(a), Self::SpilledRows(b)) => a == b,
            (Self::CurrentMemoryUsage(a), Self::CurrentMemoryUsage(b)) => a == b,
            (
                Self::Count {
                    name: n1,
                    count: c1,
                },
                Self::Count {
                    name: n2,
                    count: c2,
                },
            ) => n1 == n2 && c1 == c2,
            (
                Self::Gauge {
                    name: n1,
                    gauge: g1,
                },
                Self::Gauge {
                    name: n2,
                    gauge: g2,
                },
            )
            | (
                Self::PeakMemoryUsage {
                    name: n1,
                    gauge: g1,
                },
                Self::PeakMemoryUsage {
                    name: n2,
                    gauge: g2,
                },
            ) => n1 == n2 && g1 == g2,
            (Self::Time { name: n1, time: t1 }, Self::Time { name: n2, time: t2 }) => {
                n1 == n2 && t1 == t2
            }
            (Self::StartTimestamp(a), Self::StartTimestamp(b)) => a == b,
            (Self::EndTimestamp(a), Self::EndTimestamp(b)) => a == b,
            (
                Self::PruningMetrics {
                    name: n1,
                    pruning_metrics: p1,
                },
                Self::PruningMetrics {
                    name: n2,
                    pruning_metrics: p2,
                },
            ) => n1 == n2 && p1.pruned() == p2.pruned() && p1.matched() == p2.matched(),
            (
                Self::Ratio {
                    name: n1,
                    ratio_metrics: r1,
                },
                Self::Ratio {
                    name: n2,
                    ratio_metrics: r2,
                },
            ) => n1 == n2 && r1 == r2,
            (
                Self::Custom {
                    name: n1,
                    value: v1,
                },
                Self::Custom {
                    name: n2,
                    value: v2,
                },
            ) => n1 == n2 && v1.is_eq(v2),
            _ => false,
        }
    }
}

impl MetricValue {
    /// Return the display name of this variant (e.g. `"output_rows"` for
    /// [`MetricValue::OutputRows`], `"elapsed_compute"` for
    /// [`MetricValue::ElapsedCompute`], or the caller-supplied `name`
    /// stored in [`MetricValue::Count`], [`MetricValue::Time`], and other
    /// operator-defined variants). Used by the `EXPLAIN ANALYZE` printer
    /// via [`Metric`]'s `Display` impl and by
    /// [`MetricsSet::aggregate_by_name`] / [`MetricsSet::sum_by_name`] to
    /// key metrics with the same semantic role together across partitions.
    ///
    /// Strict mirror of DataFusion's `MetricValue::name`.
    pub fn name(&self) -> &str {
        match self {
            Self::OutputRows(_) => "output_rows",
            Self::SpillCount(_) => "spill_count",
            Self::SpilledBytes(_) => "spilled_bytes",
            Self::OutputBytes(_) => "output_bytes",
            Self::OutputBatches(_) => "output_batches",
            Self::SpilledRows(_) => "spilled_rows",
            Self::CurrentMemoryUsage(_) => "mem_used",
            Self::ElapsedCompute(_) => "elapsed_compute",
            Self::Count { name, .. } => name.borrow(),
            Self::Gauge { name, .. } | Self::PeakMemoryUsage { name, .. } => name.borrow(),
            Self::Time { name, .. } => name.borrow(),
            Self::StartTimestamp(_) => "start_timestamp",
            Self::EndTimestamp(_) => "end_timestamp",
            Self::PruningMetrics { name, .. } => name.borrow(),
            Self::Ratio { name, .. } => name.borrow(),
            Self::Custom { name, .. } => name.borrow(),
        }
    }

    /// Project this metric down to a single `usize` for numeric
    /// consumers — counters and gauges yield their current value, timing
    /// variants yield accumulated nanoseconds, and timestamps yield
    /// nanoseconds since the Unix epoch (0 if unset). Composite variants
    /// ([`MetricValue::PruningMetrics`], [`MetricValue::Ratio`]) return
    /// 0 because they cannot be flattened losslessly; custom values defer
    /// to [`CustomMetricValue::as_usize`]. Callers such as
    /// [`MetricsSet::output_rows`] and [`MetricsSet::elapsed_compute`]
    /// use this to expose a single scalar to programmatic consumers of
    /// the metrics set.
    ///
    /// Strict mirror of DataFusion's `MetricValue::as_usize`.
    pub fn as_usize(&self) -> usize {
        match self {
            Self::OutputRows(c) => c.value(),
            Self::SpillCount(c) => c.value(),
            Self::SpilledBytes(c) => c.value(),
            Self::OutputBytes(c) => c.value(),
            Self::OutputBatches(c) => c.value(),
            Self::SpilledRows(c) => c.value(),
            Self::CurrentMemoryUsage(g) => g.value(),
            Self::ElapsedCompute(t) => t.value(),
            Self::Count { count, .. } => count.value(),
            Self::Gauge { gauge, .. } | Self::PeakMemoryUsage { gauge, .. } => gauge.value(),
            Self::Time { time, .. } => time.value(),
            Self::StartTimestamp(ts) | Self::EndTimestamp(ts) => ts
                .value()
                .and_then(|t| t.timestamp_nanos_opt())
                .map_or(0, |n| n as usize),
            Self::PruningMetrics { .. } => 0,
            Self::Ratio { .. } => 0,
            Self::Custom { value, .. } => value.as_usize(),
        }
    }

    /// Build a fresh, zero-initialised value of the same variant (with
    /// the same `name`, merge strategy, or display flags where relevant)
    /// so callers have an accumulator to aggregate other partitions'
    /// values into. Used by [`MetricsSet::sum`] and
    /// [`MetricsSet::aggregate_by_name`] to seed a target before folding
    /// each per-partition [`Metric`] in with [`MetricValue::aggregate`].
    /// For [`MetricValue::Custom`] this defers to
    /// [`CustomMetricValue::new_empty`] so extension types stay in
    /// control of their own zero.
    ///
    /// Strict mirror of DataFusion's `MetricValue::new_empty`.
    pub fn new_empty(&self) -> Self {
        match self {
            Self::OutputRows(_) => Self::OutputRows(Count::new()),
            Self::SpillCount(_) => Self::SpillCount(Count::new()),
            Self::SpilledBytes(_) => Self::SpilledBytes(Count::new()),
            Self::OutputBytes(_) => Self::OutputBytes(Count::new()),
            Self::OutputBatches(_) => Self::OutputBatches(Count::new()),
            Self::SpilledRows(_) => Self::SpilledRows(Count::new()),
            Self::CurrentMemoryUsage(_) => Self::CurrentMemoryUsage(Gauge::new()),
            Self::ElapsedCompute(_) => Self::ElapsedCompute(Time::new()),
            Self::Count { name, .. } => Self::Count {
                name: name.clone(),
                count: Count::new(),
            },
            Self::Gauge { name, .. } => Self::Gauge {
                name: name.clone(),
                gauge: Gauge::new(),
            },
            Self::PeakMemoryUsage { name, .. } => Self::PeakMemoryUsage {
                name: name.clone(),
                gauge: Gauge::new(),
            },
            Self::Time { name, .. } => Self::Time {
                name: name.clone(),
                time: Time::new(),
            },
            Self::StartTimestamp(_) => Self::StartTimestamp(Timestamp::new()),
            Self::EndTimestamp(_) => Self::EndTimestamp(Timestamp::new()),
            Self::PruningMetrics { name, .. } => Self::PruningMetrics {
                name: name.clone(),
                pruning_metrics: PruningMetrics::new(),
            },
            Self::Ratio {
                name,
                ratio_metrics,
            } => {
                let merge_strategy = ratio_metrics.merge_strategy.clone();
                Self::Ratio {
                    name: name.clone(),
                    ratio_metrics: RatioMetrics::new()
                        .with_merge_strategy(merge_strategy)
                        .with_display_raw_values(ratio_metrics.display_raw_values),
                }
            }
            Self::Custom { name, value } => Self::Custom {
                name: name.clone(),
                value: value.new_empty(),
            },
        }
    }

    /// Aggregate the value of `other` into `self`.  Panics on mismatched
    /// variants, with DataFusion's exact panic message.  Mirrors
    /// DataFusion's `MetricValue::aggregate`.
    pub fn aggregate(&mut self, other: &Self) {
        match (self, other) {
            // Counter-like variants: sum.
            (Self::OutputRows(c), Self::OutputRows(o))
            | (Self::SpillCount(c), Self::SpillCount(o))
            | (Self::SpilledBytes(c), Self::SpilledBytes(o))
            | (Self::OutputBytes(c), Self::OutputBytes(o))
            | (Self::OutputBatches(c), Self::OutputBatches(o))
            | (Self::SpilledRows(c), Self::SpilledRows(o))
            | (Self::Count { count: c, .. }, Self::Count { count: o, .. }) => c.add(o.value()),

            // Gauge-like variants: sum.
            (Self::CurrentMemoryUsage(g), Self::CurrentMemoryUsage(o))
            | (Self::Gauge { gauge: g, .. }, Self::Gauge { gauge: o, .. })
            | (Self::PeakMemoryUsage { gauge: g, .. }, Self::PeakMemoryUsage { gauge: o, .. }) => {
                g.add(o.value());
            }

            // Time-like variants: add durations.
            (Self::ElapsedCompute(t), Self::ElapsedCompute(o))
            | (Self::Time { time: t, .. }, Self::Time { time: o, .. }) => t.add(o),

            // Timestamp variants: min for start, max for end.
            (Self::StartTimestamp(t), Self::StartTimestamp(o)) => t.update_to_min(o),
            (Self::EndTimestamp(t), Self::EndTimestamp(o)) => t.update_to_max(o),

            // Pruning metrics: add component-wise.
            (
                Self::PruningMetrics {
                    pruning_metrics: p, ..
                },
                Self::PruningMetrics {
                    pruning_metrics: o, ..
                },
            ) => {
                p.add_pruned(o.pruned());
                p.add_matched(o.matched());
                p.add_fully_matched(o.fully_matched());
            }

            // Ratio metrics: defer to the merge strategy.
            (
                Self::Ratio {
                    ratio_metrics: r, ..
                },
                Self::Ratio {
                    ratio_metrics: o, ..
                },
            ) => r.merge(o),

            // Custom metrics: defer to the `CustomMetricValue::aggregate` impl.
            (
                Self::Custom { value: v, .. },
                Self::Custom {
                    value: other_value, ..
                },
            ) => {
                v.aggregate(Arc::clone(other_value));
            }

            // Mismatched.
            m @ (_, _) => panic!(
                "Mismatched metric types. Can not aggregate {:?} with value {:?}",
                m.0, m.1
            ),
        }
    }

    /// Return the display ordering key for this metric — lower values
    /// print first in `EXPLAIN ANALYZE`. The ordering puts high-signal
    /// summary metrics first (`output_rows`, `elapsed_compute`, output
    /// byte/batch counts), then pruning stats, then spill and memory
    /// metrics, then generic operator-defined counters and gauges, and
    /// finally timing/ratio/timestamp/custom metrics — matching
    /// DataFusion's canonical ordering so equivalent plans produce
    /// byte-equivalent output. Used by [`MetricsSet::sorted_for_display`].
    ///
    /// Strict mirror of DataFusion's `MetricValue::display_sort_key`.
    pub fn display_sort_key(&self) -> u8 {
        match self {
            Self::OutputRows(_) => 0,
            Self::ElapsedCompute(_) => 1,
            Self::OutputBytes(_) => 2,
            Self::OutputBatches(_) => 3,
            Self::PruningMetrics { name, .. } => match name.as_ref() {
                "files_ranges_pruned_statistics" => 4,
                "row_groups_pruned_statistics" => 5,
                "row_groups_pruned_bloom_filter" => 6,
                "page_index_pages_pruned" => 7,
                "page_index_rows_pruned" => 8,
                _ => 9,
            },
            Self::SpillCount(_) => 10,
            Self::SpilledBytes(_) => 11,
            Self::SpilledRows(_) => 12,
            Self::CurrentMemoryUsage(_) => 13,
            Self::Count { name, .. } => match name.as_ref() {
                "page_index_pages_skipped_by_fully_matched" => 8,
                _ => 14,
            },
            Self::PeakMemoryUsage { .. } => 13,
            Self::Gauge { .. } => 15,
            Self::Time { .. } => 16,
            Self::Ratio { .. } => 17,
            Self::StartTimestamp(_) => 18,
            Self::EndTimestamp(_) => 19,
            Self::Custom { .. } => 20,
        }
    }

    /// Return `true` iff this is a [`MetricValue::StartTimestamp`] or
    /// [`MetricValue::EndTimestamp`]. Used by
    /// [`MetricsSet::timestamps_removed`] to strip wall-clock start/end
    /// markers from a display set — those are useful for tracing but
    /// noisy when comparing two runs, since they differ by the actual
    /// clock time even when everything else about the plan matched.
    ///
    /// Strict mirror of DataFusion's `MetricValue::is_timestamp`.
    pub fn is_timestamp(&self) -> bool {
        matches!(self, Self::StartTimestamp(_) | Self::EndTimestamp(_))
    }
}

impl Display for MetricValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutputRows(c)
            | Self::OutputBatches(c)
            | Self::SpillCount(c)
            | Self::SpilledRows(c)
            | Self::Count { count: c, .. } => write!(f, "{c}"),
            Self::SpilledBytes(c) | Self::OutputBytes(c) => {
                write!(f, "{}", human_readable_size(c.value()))
            }
            Self::CurrentMemoryUsage(g) => {
                write!(f, "{}", human_readable_size(g.value()))
            }
            Self::PeakMemoryUsage { gauge: g, .. } => {
                write!(f, "{}", human_readable_size(g.value()))
            }
            Self::Gauge { gauge: g, .. } => {
                write!(f, "{}", human_readable_count(g.value()))
            }
            Self::ElapsedCompute(t) | Self::Time { time: t, .. } => {
                if t.value() > 0 {
                    write!(f, "{t}")
                } else {
                    write!(f, "NOT RECORDED")
                }
            }
            Self::StartTimestamp(ts) | Self::EndTimestamp(ts) => write!(f, "{ts}"),
            Self::PruningMetrics {
                pruning_metrics, ..
            } => write!(f, "{pruning_metrics}"),
            Self::Ratio { ratio_metrics, .. } => write!(f, "{ratio_metrics}"),
            Self::Custom { value, .. } => write!(f, "{value}"),
        }
    }
}

// =============================================================================
// Label + LabelValue.  Strict mirror of DataFusion's `Label` / `LabelValue`
// in `physical-expr-common/src/metrics/mod.rs`.
// =============================================================================

/// A `name=value` string pair attached to a [`Metric`] to disambiguate
/// otherwise-identical entries — for example marking one `output_rows`
/// as `side=left` and another as `side=right` in a hash-join operator,
/// or tagging a spill counter with `stage=partitioning`. Labels appear
/// alongside the automatic `partition=N` label in `EXPLAIN ANALYZE`
/// output as `metric_name{k1=v1, k2=v2}=value`. Attach labels via
/// [`MetricBuilder::with_label`] / [`MetricBuilder::with_new_label`] or
/// directly with [`Metric::with_label`].
///
/// Strict mirror of DataFusion's `Label`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Label {
    name: LabelValue,
    value: LabelValue,
}

impl Label {
    /// Create a new [`Label`].
    pub fn new(name: impl Into<LabelValue>, value: impl Into<LabelValue>) -> Self {
        let name = name.into();
        let value = value.into();
        Self { name, value }
    }

    /// Returns the name of this label.
    pub fn name(&self) -> &str {
        self.name.as_str()
    }

    /// Returns the value of this label.
    pub fn value(&self) -> &str {
        self.value.as_str()
    }
}

impl Display for Label {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}={}", self.name, self.value)
    }
}

/// The shared-string type used for [`Label`] names and values. A
/// [`LabelValue`] stores either a `&'static str` (no allocation) or an
/// `Arc<str>` (single allocation shared across every clone). This
/// matters because operators typically build the same handful of
/// labels once per partition — using literal names like `"side"` and
/// `"stage"` stays allocation-free, and dynamic values from the query
/// plan are refcounted rather than duplicated per metric. Construct via
/// the `From` impls for `&'static str`, `String`, `Arc<str>`, or
/// `Cow<'static, str>`.
///
/// Strict mirror of DataFusion's `LabelValue`.
#[derive(Clone)]
pub struct LabelValue(LabelValueInner);

#[derive(Clone)]
enum LabelValueInner {
    Static(&'static str),
    Shared(Arc<str>),
}

impl LabelValue {
    /// Return this label value as a string slice.
    pub fn as_str(&self) -> &str {
        match &self.0 {
            LabelValueInner::Static(value) => value,
            LabelValueInner::Shared(value) => value.as_ref(),
        }
    }
}

impl From<&'static str> for LabelValue {
    fn from(value: &'static str) -> Self {
        Self(LabelValueInner::Static(value))
    }
}

impl From<String> for LabelValue {
    fn from(value: String) -> Self {
        Self(LabelValueInner::Shared(Arc::from(value)))
    }
}

impl From<Arc<str>> for LabelValue {
    fn from(value: Arc<str>) -> Self {
        Self(LabelValueInner::Shared(value))
    }
}

impl From<Cow<'static, str>> for LabelValue {
    fn from(value: Cow<'static, str>) -> Self {
        match value {
            Cow::Borrowed(v) => v.into(),
            Cow::Owned(v) => v.into(),
        }
    }
}

impl PartialEq for LabelValue {
    fn eq(&self, other: &Self) -> bool {
        self.as_str() == other.as_str()
    }
}

impl Eq for LabelValue {}

impl Hash for LabelValue {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_str().hash(state);
    }
}

impl Debug for LabelValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Debug::fmt(self.as_str(), f)
    }
}

impl Display for LabelValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Display::fmt(self.as_str(), f)
    }
}

// =============================================================================
// Metric — wraps a MetricValue with partition + type + category + labels.
// Strict mirror of DataFusion's `Metric`.
// =============================================================================

/// Something that tracks a value of interest during execution.
///
/// Mirrors `datafusion_physical_expr_common::metrics::Metric`.
#[derive(Debug)]
pub struct Metric {
    value: MetricValue,
    labels: Vec<Label>,
    partition: Option<usize>,
    metric_type: MetricType,
    metric_category: Option<MetricCategory>,
}

impl Metric {
    pub fn new(value: MetricValue, partition: Option<usize>) -> Self {
        Self {
            value,
            labels: Vec::new(),
            partition,
            metric_type: MetricType::Dev,
            metric_category: None,
        }
    }

    /// Construct a [`Metric`] with an initial batch of [`Label`]s in one
    /// call — the shortcut used by [`MetricBuilder::build`], which has
    /// already collected labels via its fluent API before it emits the
    /// metric. Prefer this over `Metric::new(...).with_label(...)`
    /// chaining when the label set is already known up front. The
    /// resulting metric defaults to [`MetricType::Dev`] with no
    /// category; use [`Metric::with_type`] / [`Metric::with_category`]
    /// to override.
    ///
    /// Strict mirror of DataFusion's `Metric::new_with_labels`.
    pub fn new_with_labels(
        value: MetricValue,
        partition: Option<usize>,
        labels: Vec<Label>,
    ) -> Self {
        Self {
            value,
            labels,
            partition,
            metric_type: MetricType::Dev,
            metric_category: None,
        }
    }

    pub fn with_type(mut self, metric_type: MetricType) -> Self {
        self.metric_type = metric_type;
        self
    }

    pub fn with_category(mut self, category: MetricCategory) -> Self {
        self.metric_category = Some(category);
        self
    }

    /// Append a [`Label`] to this metric and return `self`, enabling
    /// fluent chained construction: `Metric::new(v, Some(0)).with_label(l)`.
    /// Labels are additive — each call pushes to the existing vector, so
    /// the caller controls the order in which they appear in the
    /// `{k=v, k=v}` clause of the `Display` output.
    ///
    /// Strict mirror of DataFusion's `Metric::with_label`.
    pub fn with_label(mut self, label: Label) -> Self {
        self.labels.push(label);
        self
    }

    pub fn labels(&self) -> &[Label] {
        &self.labels
    }

    pub fn value(&self) -> &MetricValue {
        &self.value
    }

    pub fn value_mut(&mut self) -> &mut MetricValue {
        &mut self.value
    }

    pub fn partition(&self) -> Option<usize> {
        self.partition
    }

    pub fn metric_type(&self) -> MetricType {
        self.metric_type
    }

    pub fn metric_category(&self) -> Option<MetricCategory> {
        self.metric_category
    }
}

impl Display for Metric {
    /// Same Display shape as DataFusion's `Metric`:
    /// `name{partition=…, k=v, …}=value`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.value.name())?;
        let mut iter = self
            .partition
            .iter()
            .map(|p| Label::new("partition", p.to_string()))
            .chain(self.labels().iter().cloned())
            .peekable();
        if iter.peek().is_some() {
            write!(f, "{{")?;
            let mut first = true;
            for label in iter {
                if first {
                    first = false;
                } else {
                    write!(f, ", ")?;
                }
                write!(f, "{label}")?;
            }
            write!(f, "}}")?;
        }
        write!(f, "={}", self.value)
    }
}

// =============================================================================
// MetricsSet — `Vec<Arc<Metric>>` with the full filter/aggregate/sort
// chain.  Strict mirror of DataFusion's `MetricsSet`.
// =============================================================================

/// A snapshot of the metrics for a particular execution plan.  Mirrors
/// `datafusion_physical_expr_common::metrics::MetricsSet`.
#[derive(Default, Debug, Clone)]
pub struct MetricsSet {
    metrics: Vec<Arc<Metric>>,
}

impl MetricsSet {
    pub fn new() -> Self {
        MetricsSet::default()
    }

    pub fn push(&mut self, metric: Arc<Metric>) {
        self.metrics.push(metric);
    }

    pub fn iter(&self) -> impl Iterator<Item = &Arc<Metric>> {
        self.metrics.iter()
    }

    pub fn is_empty(&self) -> bool {
        self.metrics.is_empty()
    }

    /// Convenience: total output rows across partitions.  Mirrors
    /// DataFusion's `MetricsSet::output_rows`.
    pub fn output_rows(&self) -> Option<usize> {
        self.sum(|m| matches!(m.value(), MetricValue::OutputRows(_)))
            .map(|v| v.as_usize())
    }

    /// Convenience: total spill count.  Mirrors DataFusion.
    pub fn spill_count(&self) -> Option<usize> {
        self.sum(|m| matches!(m.value(), MetricValue::SpillCount(_)))
            .map(|v| v.as_usize())
    }

    /// Convenience: total spilled bytes.  Mirrors DataFusion.
    pub fn spilled_bytes(&self) -> Option<usize> {
        self.sum(|m| matches!(m.value(), MetricValue::SpilledBytes(_)))
            .map(|v| v.as_usize())
    }

    /// Convenience: total spilled rows.  Mirrors DataFusion.
    pub fn spilled_rows(&self) -> Option<usize> {
        self.sum(|m| matches!(m.value(), MetricValue::SpilledRows(_)))
            .map(|v| v.as_usize())
    }

    /// Convenience: total elapsed compute time (nanoseconds).  Mirrors
    /// DataFusion.
    pub fn elapsed_compute(&self) -> Option<usize> {
        self.sum(|m| matches!(m.value(), MetricValue::ElapsedCompute(_)))
            .map(|v| v.as_usize())
    }

    /// Aggregate every [`Metric`] matching predicate `f` into a single
    /// [`MetricValue`], returning `None` if no metric matched. This is
    /// the primitive underlying every convenience roll-up on
    /// [`MetricsSet`] ([`MetricsSet::output_rows`],
    /// [`MetricsSet::spill_count`], [`MetricsSet::sum_by_name`], …): it
    /// seeds an accumulator via [`MetricValue::new_empty`] from the
    /// first match, then folds subsequent matches in with
    /// [`MetricValue::aggregate`]. Reach for it directly when writing a
    /// custom summary — e.g. summing all `Count { name, .. }` entries
    /// whose label satisfies some predicate.
    ///
    /// Strict mirror of DataFusion's `MetricsSet::sum`.
    pub fn sum<F>(&self, mut f: F) -> Option<MetricValue>
    where
        F: FnMut(&Metric) -> bool,
    {
        let mut iter = self.metrics.iter().filter(|m| f(m.as_ref())).peekable();
        let mut accum = match iter.peek() {
            None => return None,
            Some(m) => m.value().new_empty(),
        };
        iter.for_each(|m| accum.aggregate(m.value()));
        Some(accum)
    }

    /// Sum every operator-named metric ([`MetricValue::Count`],
    /// [`MetricValue::Time`], [`MetricValue::Gauge`],
    /// [`MetricValue::PeakMemoryUsage`], [`MetricValue::PruningMetrics`],
    /// [`MetricValue::Ratio`]) whose caller-supplied `name` matches
    /// `metric_name`. Intentionally does NOT match the well-known
    /// built-in variants (`OutputRows`, `ElapsedCompute`, …) — those
    /// have their own convenience roll-ups. Use this to fetch a total
    /// for an operator-specific counter like `"spill_buffer_size"` or
    /// `"row_groups_pruned_statistics"` across all partitions.
    ///
    /// Strict mirror of DataFusion's `MetricsSet::sum_by_name`.
    pub fn sum_by_name(&self, metric_name: &str) -> Option<MetricValue> {
        self.sum(|m| match m.value() {
            MetricValue::Count { name, .. } => name == metric_name,
            MetricValue::Time { name, .. } => name == metric_name,
            MetricValue::OutputRows(_) => false,
            MetricValue::ElapsedCompute(_) => false,
            MetricValue::SpillCount(_) => false,
            MetricValue::SpilledBytes(_) => false,
            MetricValue::OutputBytes(_) => false,
            MetricValue::OutputBatches(_) => false,
            MetricValue::SpilledRows(_) => false,
            MetricValue::CurrentMemoryUsage(_) => false,
            MetricValue::Gauge { name, .. } => name == metric_name,
            MetricValue::PeakMemoryUsage { name, .. } => name == metric_name,
            MetricValue::StartTimestamp(_) => false,
            MetricValue::EndTimestamp(_) => false,
            MetricValue::PruningMetrics { name, .. } => name == metric_name,
            MetricValue::Ratio { name, .. } => name == metric_name,
            MetricValue::Custom { .. } => false,
        })
    }

    /// Aggregate by metric name.  Same `name` → entries are summed into
    /// the first-seen metadata bundle.  Resulting metrics carry
    /// `partition = None`.
    pub fn aggregate_by_name(&self) -> Self {
        let mut map: HashMap<String, Metric> = HashMap::new();
        for metric in &self.metrics {
            let key = metric.value().name().to_string();
            if let Some(accum) = map.get_mut(&key) {
                accum.value_mut().aggregate(metric.value());
            } else {
                let partition = None;
                let mut accum = Metric::new(metric.value().new_empty(), partition)
                    .with_type(metric.metric_type());
                if let Some(cat) = metric.metric_category() {
                    accum = accum.with_category(cat);
                }
                accum.value_mut().aggregate(metric.value());
                map.insert(key, accum);
            }
        }
        let new_metrics = map.into_values().map(Arc::new).collect();
        Self {
            metrics: new_metrics,
        }
    }

    /// Return a copy of this set sorted for `EXPLAIN ANALYZE` output,
    /// keyed first on [`MetricValue::display_sort_key`] (so `output_rows`
    /// leads, elapsed time follows, and custom metrics trail) and then
    /// alphabetically on [`MetricValue::name`] to break ties among
    /// same-category metrics. Callers typically pipeline this after
    /// [`MetricsSet::aggregate_by_name`] to produce the deterministic
    /// per-operator summary the plan printer expects.
    ///
    /// Strict mirror of DataFusion's `MetricsSet::sorted_for_display`.
    pub fn sorted_for_display(mut self) -> Self {
        self.metrics.sort_by(|a, b| {
            match a
                .value()
                .display_sort_key()
                .cmp(&b.value().display_sort_key())
            {
                Ordering::Equal => a.value().name().cmp(b.value().name()),
                other => other,
            }
        });
        self
    }

    /// Return a copy of this set with every
    /// [`MetricValue::StartTimestamp`] / [`MetricValue::EndTimestamp`]
    /// entry filtered out. Use before rendering `EXPLAIN ANALYZE` output
    /// meant to be compared across runs — wall-clock start/end times
    /// diverge run-to-run and add noise even when the plan and workload
    /// are otherwise identical. The interval between them is still
    /// reflected in `elapsed_compute`.
    ///
    /// Strict mirror of DataFusion's `MetricsSet::timestamps_removed`.
    pub fn timestamps_removed(self) -> Self {
        let metrics = self
            .metrics
            .into_iter()
            .filter(|m| !m.value().is_timestamp())
            .collect();
        Self { metrics }
    }

    /// Return a copy of this set containing only metrics whose
    /// [`MetricType`] is in `allowed`. Used to gate summary vs.
    /// developer-only metrics: rendering a user-facing plan usually
    /// filters to `[MetricType::Summary]` to hide implementation-detail
    /// metrics, while a diagnostic dump keeps both. An empty `allowed`
    /// slice returns an empty set (matches DataFusion — the identity
    /// case is "no types allowed", not "no filter").
    ///
    /// Strict mirror of DataFusion's `MetricsSet::filter_by_metric_types`.
    pub fn filter_by_metric_types(self, allowed: &[MetricType]) -> Self {
        if allowed.is_empty() {
            return Self { metrics: vec![] };
        }
        let metrics = self
            .metrics
            .into_iter()
            .filter(|m| allowed.contains(&m.metric_type()))
            .collect();
        Self { metrics }
    }

    /// Return a copy of this set containing only metrics whose
    /// [`MetricCategory`] is in `allowed`. Metrics registered without a
    /// declared category are treated as
    /// [`MetricCategory::Uncategorized`] for the purpose of this check,
    /// so callers who want to keep them must include that variant
    /// explicitly. Use to slice a metrics set for category-specific
    /// dashboards — e.g. `[MetricCategory::Rows]` for a row-count
    /// summary or `[MetricCategory::Timing]` for a wall-clock view. An
    /// empty `allowed` slice returns an empty set.
    ///
    /// Strict mirror of DataFusion's `MetricsSet::filter_by_categories`.
    pub fn filter_by_categories(self, allowed: &[MetricCategory]) -> Self {
        if allowed.is_empty() {
            return Self { metrics: vec![] };
        }
        let metrics = self
            .metrics
            .into_iter()
            .filter(|m| {
                let cat = m.metric_category().unwrap_or(MetricCategory::Uncategorized);
                allowed.contains(&cat)
            })
            .collect();
        Self { metrics }
    }
}

impl Display for MetricsSet {
    /// Comma-separated list of `Metric`s — mirrors DataFusion's
    /// `MetricsSet::fmt`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut first = true;
        for m in &self.metrics {
            if first {
                first = false;
            } else {
                write!(f, ", ")?;
            }
            write!(f, "{m}")?;
        }
        Ok(())
    }
}

impl IntoIterator for MetricsSet {
    type Item = Arc<Metric>;
    type IntoIter = std::vec::IntoIter<Arc<Metric>>;
    fn into_iter(self) -> Self::IntoIter {
        self.metrics.into_iter()
    }
}

impl<'a> IntoIterator for &'a MetricsSet {
    type Item = &'a Arc<Metric>;
    type IntoIter = std::slice::Iter<'a, Arc<Metric>>;
    fn into_iter(self) -> Self::IntoIter {
        self.metrics.iter()
    }
}

impl Extend<Arc<Metric>> for MetricsSet {
    fn extend<I: IntoIterator<Item = Arc<Metric>>>(&mut self, iter: I) {
        self.metrics.extend(iter);
    }
}

impl FromIterator<Arc<Metric>> for MetricsSet {
    fn from_iter<T: IntoIterator<Item = Arc<Metric>>>(iter: T) -> Self {
        Self {
            metrics: iter.into_iter().collect(),
        }
    }
}

// =============================================================================
// ExecutionPlanMetricsSet — mutable, shared handle operators use to record
// metrics during execution.  Strict mirror of DataFusion's
// `ExecutionPlanMetricsSet` in `physical-expr-common::metrics::mod`.
// =============================================================================

/// A set of [`Metric`]s for an individual operator.
///
/// Mirrors `datafusion_physical_expr_common::metrics::ExecutionPlanMetricsSet`.
/// Each `clone()` of this structure shares the same underlying metrics set.
#[derive(Default, Debug, Clone)]
pub struct ExecutionPlanMetricsSet {
    inner: Arc<Mutex<MetricsSet>>,
}

impl ExecutionPlanMetricsSet {
    /// Create a new empty shared metrics set.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(MetricsSet::new())),
        }
    }

    /// Add the specified metric to the underlying metric set.
    pub fn register(&self, metric: Arc<Metric>) {
        self.inner.lock().unwrap().push(metric);
    }

    /// Return a clone of the inner [`MetricsSet`].
    pub fn clone_inner(&self) -> MetricsSet {
        let guard = self.inner.lock().unwrap();
        (*guard).clone()
    }
}

impl From<MetricsSet> for ExecutionPlanMetricsSet {
    fn from(metrics: MetricsSet) -> Self {
        Self {
            inner: Arc::new(Mutex::new(metrics)),
        }
    }
}

// =============================================================================
// MetricBuilder — fluent constructor for metrics.  Strict mirror of
// DataFusion's `MetricBuilder` in `physical-expr-common::metrics::builder`.
// =============================================================================

/// Structure for constructing metrics, counters, timers, etc.
///
/// Mirrors `datafusion_physical_expr_common::metrics::MetricBuilder`.
#[derive(Clone)]
pub struct MetricBuilder<'a> {
    metrics: &'a ExecutionPlanMetricsSet,
    partition: Option<usize>,
    labels: Vec<Label>,
    metric_type: MetricType,
    metric_category: Option<MetricCategory>,
}

impl<'a> MetricBuilder<'a> {
    /// Create a new `MetricBuilder` that will register the result of
    /// `build()` with the `metrics`.
    pub fn new(metrics: &'a ExecutionPlanMetricsSet) -> Self {
        Self {
            metrics,
            partition: None,
            labels: vec![],
            metric_type: MetricType::Dev,
            metric_category: None,
        }
    }

    pub fn with_label(mut self, label: Label) -> Self {
        self.labels.push(label);
        self
    }

    pub fn with_type(mut self, metric_type: MetricType) -> Self {
        self.metric_type = metric_type;
        self
    }

    pub fn with_category(mut self, category: MetricCategory) -> Self {
        self.metric_category = Some(category);
        self
    }

    pub fn with_new_label(
        self,
        name: impl Into<Cow<'static, str>>,
        value: impl Into<Cow<'static, str>>,
    ) -> Self {
        self.with_label(Label::new(
            LabelValue::from(name.into()),
            LabelValue::from(value.into()),
        ))
    }

    pub fn with_partition(mut self, partition: usize) -> Self {
        self.partition = Some(partition);
        self
    }

    /// Consume self and register a metric of the specified value.
    pub fn build(self, value: MetricValue) {
        let Self {
            labels,
            partition,
            metrics,
            metric_type,
            metric_category,
        } = self;
        let mut metric = Metric::new_with_labels(value, partition, labels).with_type(metric_type);
        if let Some(category) = metric_category {
            metric = metric.with_category(category);
        }
        metrics.register(Arc::new(metric));
    }

    pub fn output_rows(self, partition: usize) -> Count {
        let count = Count::new();
        self.with_category(MetricCategory::Rows)
            .with_partition(partition)
            .build(MetricValue::OutputRows(count.clone()));
        count
    }

    pub fn spill_count(self, partition: usize) -> Count {
        let count = Count::new();
        self.with_category(MetricCategory::Rows)
            .with_partition(partition)
            .build(MetricValue::SpillCount(count.clone()));
        count
    }

    pub fn spilled_bytes(self, partition: usize) -> Count {
        let count = Count::new();
        self.with_category(MetricCategory::Bytes)
            .with_partition(partition)
            .build(MetricValue::SpilledBytes(count.clone()));
        count
    }

    pub fn spilled_rows(self, partition: usize) -> Count {
        let count = Count::new();
        self.with_category(MetricCategory::Rows)
            .with_partition(partition)
            .build(MetricValue::SpilledRows(count.clone()));
        count
    }

    pub fn output_bytes(self, partition: usize) -> Count {
        let count = Count::new();
        self.with_category(MetricCategory::Bytes)
            .with_partition(partition)
            .build(MetricValue::OutputBytes(count.clone()));
        count
    }

    pub fn output_batches(self, partition: usize) -> Count {
        let count = Count::new();
        self.with_category(MetricCategory::Rows)
            .with_partition(partition)
            .build(MetricValue::OutputBatches(count.clone()));
        count
    }

    pub fn mem_used(self, partition: usize) -> Gauge {
        let gauge = Gauge::new();
        self.with_category(MetricCategory::Bytes)
            .with_partition(partition)
            .build(MetricValue::CurrentMemoryUsage(gauge.clone()));
        gauge
    }

    pub fn counter(self, counter_name: impl Into<Cow<'static, str>>, partition: usize) -> Count {
        self.with_partition(partition).global_counter(counter_name)
    }

    pub fn gauge(self, gauge_name: impl Into<Cow<'static, str>>, partition: usize) -> Gauge {
        self.with_partition(partition).global_gauge(gauge_name)
    }

    pub fn global_counter(self, counter_name: impl Into<Cow<'static, str>>) -> Count {
        let count = Count::new();
        self.build(MetricValue::Count {
            name: counter_name.into(),
            count: count.clone(),
        });
        count
    }

    pub fn global_gauge(self, gauge_name: impl Into<Cow<'static, str>>) -> Gauge {
        let gauge = Gauge::new();
        self.build(MetricValue::Gauge {
            name: gauge_name.into(),
            gauge: gauge.clone(),
        });
        gauge
    }

    pub fn peak_memory_usage(
        self,
        gauge_name: impl Into<Cow<'static, str>>,
        partition: usize,
    ) -> Gauge {
        let gauge = Gauge::new();
        self.with_category(MetricCategory::Bytes)
            .with_partition(partition)
            .build(MetricValue::PeakMemoryUsage {
                name: gauge_name.into(),
                gauge: gauge.clone(),
            });
        gauge
    }

    pub fn elapsed_compute(self, partition: usize) -> Time {
        let time = Time::new();
        self.with_category(MetricCategory::Timing)
            .with_partition(partition)
            .build(MetricValue::ElapsedCompute(time.clone()));
        time
    }

    pub fn subset_time(self, subset_name: impl Into<Cow<'static, str>>, partition: usize) -> Time {
        let time = Time::new();
        self.with_category(MetricCategory::Timing)
            .with_partition(partition)
            .build(MetricValue::Time {
                name: subset_name.into(),
                time: time.clone(),
            });
        time
    }

    pub fn start_timestamp(self, partition: usize) -> Timestamp {
        let timestamp = Timestamp::new();
        self.with_category(MetricCategory::Timing)
            .with_partition(partition)
            .build(MetricValue::StartTimestamp(timestamp.clone()));
        timestamp
    }

    pub fn end_timestamp(self, partition: usize) -> Timestamp {
        let timestamp = Timestamp::new();
        self.with_category(MetricCategory::Timing)
            .with_partition(partition)
            .build(MetricValue::EndTimestamp(timestamp.clone()));
        timestamp
    }

    pub fn pruning_metrics(
        self,
        name: impl Into<Cow<'static, str>>,
        partition: usize,
    ) -> PruningMetrics {
        let pruning_metrics = PruningMetrics::new();
        self.with_category(MetricCategory::Rows)
            .with_partition(partition)
            .build(MetricValue::PruningMetrics {
                name: name.into(),
                pruning_metrics: pruning_metrics.clone(),
            });
        pruning_metrics
    }

    pub fn ratio_metrics(
        self,
        name: impl Into<Cow<'static, str>>,
        partition: usize,
    ) -> RatioMetrics {
        self.ratio_metrics_with_strategy(name, partition, RatioMergeStrategy::default())
    }

    pub fn ratio_metrics_with_strategy(
        self,
        name: impl Into<Cow<'static, str>>,
        partition: usize,
        merge_strategy: RatioMergeStrategy,
    ) -> RatioMetrics {
        let ratio_metrics = RatioMetrics::new().with_merge_strategy(merge_strategy);
        self.with_category(MetricCategory::Rows)
            .with_partition(partition)
            .build(MetricValue::Ratio {
                name: name.into(),
                ratio_metrics: ratio_metrics.clone(),
            });
        ratio_metrics
    }
}

// =============================================================================
// BaselineMetrics + SpillMetrics + SplitMetrics + RecordOutput.  Strict
// mirror of DataFusion's
// `physical-expr-common::metrics::baseline`.
// =============================================================================

/// Bundle of the standard "baseline" metrics every physical operator
/// records: `start_timestamp`, `end_timestamp`, `elapsed_compute`,
/// `output_rows`, `output_bytes`, and `output_batches`. Constructing one
/// via [`BaselineMetrics::new`] registers all six against an
/// [`ExecutionPlanMetricsSet`] and records the start timestamp; the
/// operator then updates them during execution (typically through
/// [`BaselineMetrics::record_poll`], the `RecordOutput` trait impls, or
/// a [`ScopedTimerGuard`] returned from `elapsed_compute().timer()`).
/// The end timestamp is captured when [`BaselineMetrics::done`] is
/// called or, as a safety net, when the `BaselineMetrics` is dropped.
///
/// Strict mirror of DataFusion's `BaselineMetrics`.
#[derive(Debug, Clone)]
pub struct BaselineMetrics {
    /// `end_time` is set when `BaselineMetrics::done()` is called.
    end_time: Timestamp,

    /// Amount of time the operator was actively trying to use the CPU.
    elapsed_compute: Time,

    /// Output rows: the total output rows.
    output_rows: Count,

    /// Memory usage of all output batches.
    ///
    /// Note: This value may be overestimated. If multiple output
    /// `RecordBatch` instances share underlying memory buffers, their
    /// sizes will be counted multiple times.
    /// Issue: <https://github.com/apache/datafusion/issues/16841>
    output_bytes: Count,

    /// Output batches: the total output batch count.
    output_batches: Count,
}

impl BaselineMetrics {
    /// Create a new BaselineMetric structure and set `start_time` to now.
    pub fn new(metrics: &ExecutionPlanMetricsSet, partition: usize) -> Self {
        let start_time = MetricBuilder::new(metrics).start_timestamp(partition);
        start_time.record();

        Self {
            end_time: MetricBuilder::new(metrics)
                .with_type(MetricType::Summary)
                .end_timestamp(partition),
            elapsed_compute: MetricBuilder::new(metrics)
                .with_type(MetricType::Summary)
                .elapsed_compute(partition),
            output_rows: MetricBuilder::new(metrics)
                .with_type(MetricType::Summary)
                .output_rows(partition),
            output_bytes: MetricBuilder::new(metrics)
                .with_type(MetricType::Summary)
                .output_bytes(partition),
            output_batches: MetricBuilder::new(metrics)
                .with_type(MetricType::Dev)
                .output_batches(partition),
        }
    }

    /// Returns a [`BaselineMetrics`] that updates the same
    /// `elapsed_compute` ignoring all other metrics.  Mirrors
    /// DataFusion's `BaselineMetrics::intermediate`.
    pub fn intermediate(&self) -> BaselineMetrics {
        Self {
            end_time: Timestamp::default(),
            elapsed_compute: self.elapsed_compute.clone(),
            output_rows: Count::default(),
            output_bytes: Count::default(),
            output_batches: Count::default(),
        }
    }

    /// Returns the metric for cpu time spent in this operator.
    pub fn elapsed_compute(&self) -> &Time {
        &self.elapsed_compute
    }

    /// Returns the metric for the total number of output rows.
    pub fn output_rows(&self) -> &Count {
        &self.output_rows
    }

    /// Returns the metric for the total number of output batches.
    pub fn output_batches(&self) -> &Count {
        &self.output_batches
    }

    /// Records the fact that this operator's execution is complete
    /// (recording the `end_time` metric).
    pub fn done(&self) {
        self.end_time.record();
    }

    /// Record that some number of rows have been produced as output.
    pub fn record_output(&self, num_rows: usize) {
        self.output_rows.add(num_rows);
    }

    /// If not previously recorded `done()`, record it.
    pub fn try_done(&self) {
        if self.end_time.value().is_none() {
            self.end_time.record();
        }
    }

    /// Process a poll result of a stream producing output for an operator.
    ///
    /// Note: this method only updates `output_rows` and `end_time` metrics.
    /// Remember to update `elapsed_compute` and other metrics manually.
    ///
    /// Strict mirror of DataFusion's `BaselineMetrics::record_poll` in
    /// `datafusion_physical_expr_common::metrics::BaselineMetrics::record_poll`.
    pub fn record_poll(
        &self,
        poll: Poll<Option<Result<RecordBatch>>>,
    ) -> Poll<Option<Result<RecordBatch>>> {
        if let Poll::Ready(maybe_batch) = &poll {
            match maybe_batch {
                Some(Ok(batch)) => {
                    batch.record_output(self);
                }
                Some(Err(_)) => self.done(),
                None => self.done(),
            }
        }
        poll
    }

    /// Returns a derived metric that summarizes how unevenly
    /// `output_rows` are distributed across partitions.  Mirrors
    /// DataFusion's `BaselineMetrics::output_rows_skew_metric` in
    /// `datafusion_physical_expr_common::metrics::BaselineMetrics::output_rows_skew_metric`.
    pub fn output_rows_skew_metric(metrics: &MetricsSet) -> Option<Arc<Metric>> {
        use std::collections::BTreeMap;

        let output_rows = metrics
            .iter()
            .filter_map(|metric| match (metric.partition(), metric.value()) {
                (Some(partition), MetricValue::OutputRows(count)) => {
                    Some((partition, count.value() as u128))
                }
                _ => None,
            })
            .fold(
                BTreeMap::<usize, u128>::new(),
                |mut output_rows, (partition, rows)| {
                    *output_rows.entry(partition).or_default() += rows;
                    output_rows
                },
            )
            .into_values()
            .collect::<Vec<_>>();

        if output_rows.is_empty() {
            return None;
        }

        let ratio_metrics = RatioMetrics::new().with_display_raw_values(false);
        if let Some(score) = output_rows_skew_score(&output_rows) {
            ratio_metrics.set_part((score * 10_000.0).round() as usize);
            ratio_metrics.set_total(10_000);
        }

        Some(Arc::new(
            Metric::new(
                MetricValue::Ratio {
                    name: Cow::Borrowed(OUTPUT_ROWS_SKEW_METRIC_NAME),
                    ratio_metrics,
                },
                None,
            )
            .with_type(MetricType::Dev),
        ))
    }
}

const OUTPUT_ROWS_SKEW_METRIC_NAME: &str = "output_rows_skew";

/// See [`BaselineMetrics::output_rows_skew_metric`] for the algorithm.
/// Strict mirror of DataFusion's `output_rows_skew_score` in
/// `datafusion_physical_expr_common::metrics::baseline::output_rows_skew_score`.
fn output_rows_skew_score(output_rows: &[u128]) -> Option<f64> {
    if output_rows.is_empty() {
        return None;
    }

    let partition_count = output_rows.len();
    if partition_count == 1 {
        return Some(0.0);
    }

    let (total_rows, sum_of_squares) =
        output_rows
            .iter()
            .fold((0.0_f64, 0.0_f64), |(total_rows, sum_of_squares), rows| {
                let rows = *rows as f64;
                (total_rows + rows, sum_of_squares + rows.powi(2))
            });
    if total_rows == 0.0 {
        return None;
    }
    if sum_of_squares == 0.0 {
        return None;
    }

    let effective_parallelism = total_rows.powi(2) / sum_of_squares;
    let balanced_score = (effective_parallelism - 1.0) / (partition_count as f64 - 1.0);

    Some((1.0 - balanced_score).clamp(0.0, 1.0))
}

impl Drop for BaselineMetrics {
    fn drop(&mut self) {
        self.try_done();
    }
}

/// Bundle of the standard spill-tracking counters an operator that can
/// spill to disk (sort, hash join, hash aggregate, …) records:
/// `spill_count` (how many times the operator flushed to disk),
/// `spilled_bytes` (total bytes written), and `spilled_rows` (total
/// rows). Constructing one via [`SpillMetrics::new`] registers all
/// three against an [`ExecutionPlanMetricsSet`] with the appropriate
/// [`MetricCategory`]. Operators call `.add(n)` on the individual
/// counters as work happens; cross-partition roll-ups aggregate as
/// counters.
///
/// Strict mirror of DataFusion's `SpillMetrics`.
#[derive(Debug, Clone)]
pub struct SpillMetrics {
    /// Count of spills during the execution of the operator.
    pub spill_file_count: Count,

    /// Total bytes actually written to disk during the execution of the
    /// operator.
    pub spilled_bytes: Count,

    /// Total spilled rows during the execution of the operator.
    pub spilled_rows: Count,
}

impl SpillMetrics {
    pub fn new(metrics: &ExecutionPlanMetricsSet, partition: usize) -> Self {
        Self {
            spill_file_count: MetricBuilder::new(metrics).spill_count(partition),
            spilled_bytes: MetricBuilder::new(metrics).spilled_bytes(partition),
            spilled_rows: MetricBuilder::new(metrics).spilled_rows(partition),
        }
    }
}

/// Counter bundle for operators that break oversized input
/// `RecordBatch`es into smaller pieces (for example when honoring a
/// downstream batch-size preference or a memory budget). Exposes a
/// single `batches_split` [`Count`] registered against an
/// [`ExecutionPlanMetricsSet`] under [`MetricCategory::Rows`]; the
/// operator increments it once per split so the `EXPLAIN ANALYZE`
/// reader can tell whether splitting was a hot path or a rarity.
///
/// Strict mirror of DataFusion's `SplitMetrics`.
#[derive(Debug, Clone)]
pub struct SplitMetrics {
    /// Number of times an input `RecordBatch` was split.
    pub batches_split: Count,
}

impl SplitMetrics {
    pub fn new(metrics: &ExecutionPlanMetricsSet, partition: usize) -> Self {
        Self {
            batches_split: MetricBuilder::new(metrics)
                .with_category(MetricCategory::Rows)
                .counter("batches_split", partition),
        }
    }
}

/// Extension trait that folds a produced value (a raw `usize` row
/// count, a `RecordBatch`, an `Option<RecordBatch>`, or a
/// `Result<RecordBatch>`) into a [`BaselineMetrics`] and returns the
/// value unchanged. Lets operator code write
/// `Poll::Ready(Some(Ok(batch.record_output(&self.baseline))))`
/// inline in a `Stream::poll_next` impl instead of unpacking the batch,
/// updating row/byte/batch counts by hand, and repacking it. The impls
/// for `Option` and `Result` variants no-op on the `None` / `Err` case.
///
/// Strict mirror of DataFusion's `RecordOutput`.
pub trait RecordOutput {
    fn record_output(self, bm: &BaselineMetrics) -> Self;
}

impl RecordOutput for usize {
    fn record_output(self, bm: &BaselineMetrics) -> Self {
        bm.record_output(self);
        self
    }
}

impl RecordOutput for RecordBatch {
    fn record_output(self, bm: &BaselineMetrics) -> Self {
        bm.record_output(self.num_rows());
        let n_bytes = get_record_batch_memory_size(&self);
        bm.output_bytes.add(n_bytes);
        bm.output_batches.add(1);
        self
    }
}

impl RecordOutput for &RecordBatch {
    fn record_output(self, bm: &BaselineMetrics) -> Self {
        bm.record_output(self.num_rows());
        let n_bytes = get_record_batch_memory_size(self);
        bm.output_bytes.add(n_bytes);
        bm.output_batches.add(1);
        self
    }
}

impl RecordOutput for Option<&RecordBatch> {
    fn record_output(self, bm: &BaselineMetrics) -> Self {
        if let Some(record_batch) = &self {
            record_batch.record_output(bm);
        }
        self
    }
}

impl RecordOutput for Option<RecordBatch> {
    fn record_output(self, bm: &BaselineMetrics) -> Self {
        if let Some(record_batch) = &self {
            record_batch.record_output(bm);
        }
        self
    }
}

// fdapquery's `Result` is `Result<T, FdapQueryError>` (vs DataFusion's
// `Result<T, DataFusionError>`); the impl is otherwise byte-for-byte.
impl RecordOutput for Result<RecordBatch> {
    fn record_output(self, bm: &BaselineMetrics) -> Self {
        if let Ok(record_batch) = &self {
            record_batch.record_output(bm);
        }
        self
    }
}

// =============================================================================
// human-readable formatters — strict mirror of
// `datafusion_common::display::human_readable`.
// =============================================================================

/// Powers-of-two byte-size constants (`KB`, `MB`, `GB`, `TB`) used by
/// [`human_readable_size`] to pick a display unit. Callers that need
/// the same thresholds elsewhere (e.g. sizing a buffer against `MB`)
/// can pull them from here rather than redefining them, keeping the
/// definitions consistent with the display path.
///
/// Strict mirror of DataFusion's `units`.
pub mod units {
    pub const TB: u64 = 1 << 40;
    pub const GB: u64 = 1 << 30;
    pub const MB: u64 = 1 << 20;
    pub const KB: u64 = 1 << 10;
}

/// Render a byte count using the largest binary unit whose value is at
/// least 2 (e.g. `4194304` → `"4.0 MB"`, `1023` → `"1023.0 B"`). Used
/// by the `Display` impls for [`MetricValue::SpilledBytes`],
/// [`MetricValue::OutputBytes`], [`MetricValue::CurrentMemoryUsage`],
/// and [`MetricValue::PeakMemoryUsage`] so `EXPLAIN ANALYZE` output
/// stays compact and comparable. Output is always to one decimal
/// place; the "2× threshold" avoids things like `"1.0 KB"` for what
/// is actually `1024` bytes.
///
/// Strict mirror of DataFusion's `human_readable_size`.
pub fn human_readable_size(size: usize) -> String {
    use units::*;
    let size = size as u64;
    let (value, unit) = if size >= 2 * TB {
        (size as f64 / TB as f64, "TB")
    } else if size >= 2 * GB {
        (size as f64 / GB as f64, "GB")
    } else if size >= 2 * MB {
        (size as f64 / MB as f64, "MB")
    } else if size >= 2 * KB {
        (size as f64 / KB as f64, "KB")
    } else {
        (size as f64, "B")
    };
    format!("{value:.1} {unit}")
}

/// Render an integer count using SI suffixes (`K`, `M`, `B`, `T`) once
/// it crosses one thousand; below 1_000 the raw integer is emitted with
/// no suffix. Used by [`Count`]'s `Display` impl (and thus by the
/// row/batch/spill counters in [`MetricValue`]) so `EXPLAIN ANALYZE`
/// output collapses `1234567` into `"1.23 M"`. Uses base-1000 (decimal)
/// throughout, unlike [`human_readable_size`] which uses base-1024, so
/// rows and bytes read naturally in their respective conventions.
///
/// Strict mirror of DataFusion's `human_readable_count`.
pub fn human_readable_count(count: usize) -> String {
    let count = count as u64;
    let (value, unit) = if count >= 1_000_000_000_000 {
        (count as f64 / 1_000_000_000_000.0, " T")
    } else if count >= 1_000_000_000 {
        (count as f64 / 1_000_000_000.0, " B")
    } else if count >= 1_000_000 {
        (count as f64 / 1_000_000.0, " M")
    } else if count >= 1_000 {
        (count as f64 / 1_000.0, " K")
    } else {
        return count.to_string();
    };
    if value >= 100.0 {
        format!("{value:.1}{unit}")
    } else {
        format!("{value:.2}{unit}")
    }
}

/// Render a nanosecond duration in the largest appropriate unit — `ns`
/// below 1 μs, `µs` below 1 ms, `ms` below 1 s, and `s` above. Used by
/// [`Time`]'s `Display` impl (and thus by [`MetricValue::ElapsedCompute`]
/// and [`MetricValue::Time`]) so `EXPLAIN ANALYZE` output prints
/// `"3.24 ms"` rather than `3240000`. Above 1 μs the output uses two
/// decimal places; sub-microsecond values print as bare integer
/// nanoseconds because further decimals would exceed clock resolution.
///
/// Strict mirror of DataFusion's `human_readable_duration`.
pub fn human_readable_duration(nanos: u64) -> String {
    const NANOS_PER_SEC: f64 = 1_000_000_000.0;
    const NANOS_PER_MILLI: f64 = 1_000_000.0;
    const NANOS_PER_MICRO: f64 = 1_000.0;
    let n = nanos as f64;
    if nanos >= 1_000_000_000 {
        format!("{:.2}s", n / NANOS_PER_SEC)
    } else if nanos >= 1_000_000 {
        format!("{:.2}ms", n / NANOS_PER_MILLI)
    } else if nanos >= 1_000 {
        format!("{:.2}µs", n / NANOS_PER_MICRO)
    } else {
        format!("{nanos}ns")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn arc(m: Metric) -> Arc<Metric> {
        Arc::new(m)
    }

    fn sample_set() -> MetricsSet {
        let mut s = MetricsSet::new();

        // Two OutputRows entries from different partitions.
        let c1 = Count::new();
        c1.add(10);
        s.push(arc(Metric::new(MetricValue::OutputRows(c1), Some(0))
            .with_type(MetricType::Summary)
            .with_category(MetricCategory::Rows)));
        let c2 = Count::new();
        c2.add(20);
        s.push(arc(Metric::new(MetricValue::OutputRows(c2), Some(1))
            .with_type(MetricType::Summary)
            .with_category(MetricCategory::Rows)));

        // One ElapsedCompute entry.
        let t = Time::new();
        t.add_duration(Duration::from_nanos(100));
        s.push(arc(Metric::new(MetricValue::ElapsedCompute(t), None)
            .with_type(MetricType::Summary)
            .with_category(MetricCategory::Timing)));

        // One Dev-tagged custom counter.
        let g = Count::new();
        g.add(4096);
        s.push(arc(Metric::new(
            MetricValue::Count {
                name: Cow::Borrowed("spill_buffer_size"),
                count: g,
            },
            None,
        )
        .with_type(MetricType::Dev)
        .with_category(MetricCategory::Bytes)));

        // Two timestamps (uncategorised).
        let st = Timestamp::new();
        st.set(Utc.timestamp_nanos(1_700_000_000_000_000_000));
        s.push(arc(
            Metric::new(MetricValue::StartTimestamp(st), None).with_type(MetricType::Summary)
        ));
        let et = Timestamp::new();
        et.set(Utc.timestamp_nanos(1_700_000_500_000_000_000));
        s.push(arc(
            Metric::new(MetricValue::EndTimestamp(et), None).with_type(MetricType::Summary)
        ));

        s
    }

    #[test]
    fn filter_by_metric_types_keeps_only_listed_types() {
        let s = sample_set().filter_by_metric_types(&[MetricType::Summary]);
        // 2× output_rows + elapsed_compute + 2× timestamps = 5.
        assert_eq!(s.iter().count(), 5);

        let s = sample_set().filter_by_metric_types(&[MetricType::Dev]);
        assert_eq!(s.iter().count(), 1);
        assert_eq!(s.iter().next().unwrap().value().name(), "spill_buffer_size");

        assert!(sample_set().filter_by_metric_types(&[]).is_empty());
    }

    #[test]
    fn filter_by_categories_routes_uncategorised_through_uncategorized_bucket() {
        // Rows category alone keeps the 2 output_rows entries.
        let s = sample_set().filter_by_categories(&[MetricCategory::Rows]);
        assert_eq!(s.iter().count(), 2);

        // Uncategorized alone keeps the 2 timestamps.
        let s = sample_set().filter_by_categories(&[MetricCategory::Uncategorized]);
        assert_eq!(s.iter().count(), 2);

        // Empty allow-list drops everything.
        assert!(sample_set().filter_by_categories(&[]).is_empty());
    }

    #[test]
    fn aggregate_by_name_sums_counters_and_takes_min_max_of_timestamps() {
        let s = sample_set().aggregate_by_name();
        // 6 inputs, 5 distinct names.
        assert_eq!(s.iter().count(), 5);
        let row = s
            .iter()
            .find(|m| m.value().name() == "output_rows")
            .unwrap();
        assert_eq!(row.value().as_usize(), 30); // 10 + 20
        assert!(row.partition().is_none());

        let elapsed = s
            .iter()
            .find(|m| m.value().name() == "elapsed_compute")
            .unwrap();
        assert_eq!(elapsed.value().as_usize(), 100);
    }

    #[test]
    fn timestamps_removed_drops_start_and_end() {
        let s = sample_set().timestamps_removed();
        assert_eq!(s.iter().count(), 4);
        assert!(s.iter().all(|m| !m.value().is_timestamp()));
    }

    #[test]
    fn sorted_for_display_uses_display_sort_key() {
        let s = sample_set().aggregate_by_name().sorted_for_display();
        let names: Vec<&str> = s.iter().map(|m| m.value().name()).collect();
        assert_eq!(
            names,
            vec![
                "output_rows",
                "elapsed_compute",
                "spill_buffer_size",
                "start_timestamp",
                "end_timestamp",
            ]
        );
    }

    #[test]
    fn metric_display_includes_partition_label() {
        let c = Count::new();
        c.add(42);
        let metric = Metric::new(MetricValue::OutputRows(c), Some(1))
            .with_label(Label::new("region", "us-west"));
        assert_eq!(
            metric.to_string(),
            "output_rows{partition=1, region=us-west}=42"
        );
    }

    #[test]
    fn metric_value_display_uses_human_readable_for_bytes() {
        let c = Count::new();
        c.add(units::MB as usize * 2);
        let v = MetricValue::SpilledBytes(c);
        assert_eq!(v.to_string(), "2.0 MB");
    }

    #[test]
    #[should_panic(expected = "Mismatched metric types. Can not aggregate")]
    fn aggregate_across_variants_panics() {
        let c = Count::new();
        c.add(1);
        let t = Time::new();
        t.add_duration(Duration::from_nanos(10));
        let mut a = MetricValue::OutputRows(c);
        let b = MetricValue::ElapsedCompute(t);
        a.aggregate(&b);
    }

    #[test]
    fn full_pipeline_chain_matches_display_path() {
        let s = sample_set()
            .filter_by_metric_types(&[MetricType::Summary, MetricType::Dev])
            .aggregate_by_name()
            .sorted_for_display()
            .timestamps_removed();

        assert!(s.iter().all(|m| !m.value().is_timestamp()));
        let rendered = s.to_string();
        assert!(rendered.contains("output_rows="));
        assert!(rendered.contains("elapsed_compute="));
        assert!(rendered.contains("spill_buffer_size="));
    }

    // -----------------------------------------------------------------
    // CustomMetricValue tests — strict mirror of DataFusion's
    // `value.rs::tests::{test_custom_metric, test_custom_metric_with_mismatching_names,
    // test_display_custom_metric}`.
    // -----------------------------------------------------------------

    #[derive(Debug, Default)]
    struct CustomCounter {
        count: AtomicUsize,
    }

    impl Display for CustomCounter {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "count: {}", self.count.load(AtomicOrdering::Relaxed))
        }
    }

    impl CustomMetricValue for CustomCounter {
        fn new_empty(&self) -> Arc<dyn CustomMetricValue> {
            Arc::new(CustomCounter::default())
        }

        fn aggregate(&self, other: Arc<dyn CustomMetricValue + 'static>) {
            let other = other.as_any().downcast_ref::<Self>().unwrap();
            self.count.fetch_add(
                other.count.load(AtomicOrdering::Relaxed),
                AtomicOrdering::Relaxed,
            );
        }

        fn as_any(&self) -> &dyn Any {
            self
        }

        fn is_eq(&self, other: &Arc<dyn CustomMetricValue>) -> bool {
            let Some(other) = other.as_any().downcast_ref::<Self>() else {
                return false;
            };
            self.count.load(AtomicOrdering::Relaxed) == other.count.load(AtomicOrdering::Relaxed)
        }
    }

    fn new_custom_counter(name: &'static str, value: usize) -> MetricValue {
        let custom_counter = CustomCounter::default();
        custom_counter
            .count
            .fetch_add(value, AtomicOrdering::Relaxed);
        MetricValue::Custom {
            name: Cow::Borrowed(name),
            value: Arc::new(custom_counter),
        }
    }

    #[test]
    fn test_custom_metric_with_mismatching_names() {
        let mut custom_val = new_custom_counter("Hi", 1);
        let other_custom_val = new_custom_counter("Hello", 1);

        // Not equal since the name differs.
        assert!(other_custom_val != custom_val);

        // Should work even though the name differs.
        custom_val.aggregate(&other_custom_val);

        let expected_val = new_custom_counter("Hi", 2);
        assert!(expected_val == custom_val);
    }

    #[test]
    fn test_custom_metric() {
        let mut custom_val = new_custom_counter("hi", 11);
        let other_custom_val = new_custom_counter("hi", 20);

        custom_val.aggregate(&other_custom_val);

        assert!(custom_val != other_custom_val);

        if let MetricValue::Custom { value, .. } = custom_val {
            let counter = value
                .as_any()
                .downcast_ref::<CustomCounter>()
                .expect("Expected CustomCounter");
            assert_eq!(counter.count.load(AtomicOrdering::Relaxed), 31);
        } else {
            panic!("Unexpected value");
        }
    }

    #[test]
    fn test_display_custom_metric() {
        let custom_val = new_custom_counter("hi", 11);
        assert_eq!(custom_val.to_string(), "count: 11");
    }

    #[test]
    fn test_custom_metric_display_sort_key() {
        let v = new_custom_counter("hi", 0);
        assert_eq!(v.display_sort_key(), 20);
    }

    // -----------------------------------------------------------------
    // Timestamp test — mirrors DataFusion's `test_display_timestamp`.
    // -----------------------------------------------------------------

    #[test]
    fn test_display_timestamp() {
        let timestamp = Timestamp::new();
        let values = vec![
            MetricValue::StartTimestamp(timestamp.clone()),
            MetricValue::EndTimestamp(timestamp.clone()),
        ];

        for value in &values {
            assert_eq!("NONE", value.to_string(), "value {value:?}");
        }

        timestamp.set(Utc.timestamp_nanos(1_431_648_000_000_000));
        for value in &values {
            assert_eq!(
                "1970-01-17 13:40:48 UTC",
                value.to_string(),
                "value {value:?}"
            );
        }
    }

    // -----------------------------------------------------------------
    // Label / LabelValue tests — mirror DataFusion's
    // `test_label_owned_and_borrowed_values_are_equal`.
    // -----------------------------------------------------------------

    #[test]
    fn test_label_owned_and_borrowed_values_are_equal() {
        let borrowed = Label::new("foo", "bar");
        let owned = Label::new("foo".to_string(), "bar".to_string());
        let shared = Label::new("foo", Arc::<str>::from("bar"));

        assert_eq!(borrowed, owned);
        assert_eq!(borrowed, shared);
        assert_eq!(borrowed.to_string(), owned.to_string());
        assert_eq!(borrowed.to_string(), shared.to_string());
    }

    // -----------------------------------------------------------------
    // MetricBuilder + ExecutionPlanMetricsSet tests — mirror
    // DataFusion's `test_output_rows` / `test_elapsed_compute` / `test_sum`.
    // -----------------------------------------------------------------

    #[test]
    fn test_output_rows_via_builder() {
        let metrics = ExecutionPlanMetricsSet::new();
        assert!(metrics.clone_inner().output_rows().is_none());

        let partition = 1;
        let output_rows = MetricBuilder::new(&metrics).output_rows(partition);
        output_rows.add(13);

        let output_rows = MetricBuilder::new(&metrics).output_rows(partition + 1);
        output_rows.add(7);
        assert_eq!(metrics.clone_inner().output_rows().unwrap(), 20);
    }

    #[test]
    fn test_elapsed_compute_via_builder() {
        let metrics = ExecutionPlanMetricsSet::new();
        assert!(metrics.clone_inner().elapsed_compute().is_none());

        let partition = 1;
        let elapsed_compute = MetricBuilder::new(&metrics).elapsed_compute(partition);
        elapsed_compute.add_duration(Duration::from_nanos(1234));

        let elapsed_compute = MetricBuilder::new(&metrics).elapsed_compute(partition + 1);
        elapsed_compute.add_duration(Duration::from_nanos(6));
        assert_eq!(metrics.clone_inner().elapsed_compute().unwrap(), 1240);
    }

    #[test]
    fn test_sum_via_builder() {
        let metrics = ExecutionPlanMetricsSet::new();

        let count1 = MetricBuilder::new(&metrics)
            .with_new_label("foo", "bar")
            .counter("my_counter", 1);
        count1.add(1);

        let count2 = MetricBuilder::new(&metrics).counter("my_counter", 2);
        count2.add(2);

        let metrics = metrics.clone_inner();
        assert!(metrics.sum(|_| false).is_none());

        let expected_count = Count::new();
        expected_count.add(3);
        let expected_sum = MetricValue::Count {
            name: "my_counter".into(),
            count: expected_count,
        };

        assert_eq!(metrics.sum(|_| true), Some(expected_sum));
    }

    // -----------------------------------------------------------------
    // BaselineMetrics test — mirrors DataFusion's record_poll path
    // without the arrow types.
    // -----------------------------------------------------------------

    #[test]
    fn test_baseline_metrics_done_records_end_time() {
        let metrics = ExecutionPlanMetricsSet::new();
        let bm = BaselineMetrics::new(&metrics, 0);
        bm.record_output(7);
        bm.done();

        let snap = metrics.clone_inner();
        assert_eq!(snap.output_rows(), Some(7));
        // end_timestamp should now have been recorded
        let has_end = snap.iter().any(|m| match m.value() {
            MetricValue::EndTimestamp(ts) => ts.value().is_some(),
            _ => false,
        });
        assert!(has_end, "BaselineMetrics::done must record end_timestamp");
    }

    // -----------------------------------------------------------------
    // record_poll / RecordOutput-for-RecordBatch tests — mirror
    // DataFusion's `BaselineMetrics::record_poll` semantics from
    // `datafusion_physical_expr_common::metrics::BaselineMetrics::record_poll`. DataFusion has
    // no upstream tests for this path; these are end-to-end checks of
    // the strict mirror.
    // -----------------------------------------------------------------

    fn end_timestamp_recorded(snap: &MetricsSet) -> bool {
        snap.iter().any(|m| match m.value() {
            MetricValue::EndTimestamp(ts) => ts.value().is_some(),
            _ => false,
        })
    }

    fn output_batches_count(snap: &MetricsSet) -> usize {
        snap.iter()
            .filter_map(|m| match m.value() {
                MetricValue::OutputBatches(c) => Some(c.value()),
                _ => None,
            })
            .sum()
    }

    fn output_bytes_count(snap: &MetricsSet) -> usize {
        snap.iter()
            .filter_map(|m| match m.value() {
                MetricValue::OutputBytes(c) => Some(c.value()),
                _ => None,
            })
            .sum()
    }

    fn make_batch() -> RecordBatch {
        use arrow_array::Int32Array;
        use arrow_schema::{DataType, Field, Schema};
        let schema = Arc::new(Schema::new(vec![Field::new(
            "ints",
            DataType::Int32,
            false,
        )]));
        let arr = Int32Array::from(vec![1, 2, 3, 4, 5]);
        RecordBatch::try_new(schema, vec![Arc::new(arr)]).unwrap()
    }

    #[test]
    fn record_output_for_record_batch_updates_rows_bytes_and_batches() {
        let metrics = ExecutionPlanMetricsSet::new();
        let bm = BaselineMetrics::new(&metrics, 0);

        let batch = make_batch();
        let returned = batch.clone().record_output(&bm);
        assert_eq!(returned.num_rows(), batch.num_rows());

        let snap = metrics.clone_inner();
        assert_eq!(snap.output_rows(), Some(5));
        assert_eq!(output_batches_count(&snap), 1);
        assert!(
            output_bytes_count(&snap) > 0,
            "output_bytes should track buffer memory"
        );
    }

    #[test]
    fn record_poll_ready_some_ok_records_batch() {
        let metrics = ExecutionPlanMetricsSet::new();
        let bm = BaselineMetrics::new(&metrics, 0);

        let batch = make_batch();
        let poll: Poll<Option<Result<RecordBatch>>> = Poll::Ready(Some(Ok(batch)));
        let returned = bm.record_poll(poll);
        assert!(matches!(returned, Poll::Ready(Some(Ok(_)))));

        let snap = metrics.clone_inner();
        assert_eq!(snap.output_rows(), Some(5));
        assert_eq!(output_batches_count(&snap), 1);
        // Stream not yet finished — done() must NOT have fired.
        assert!(!end_timestamp_recorded(&snap));
    }

    #[test]
    fn record_poll_ready_none_calls_done() {
        let metrics = ExecutionPlanMetricsSet::new();
        let bm = BaselineMetrics::new(&metrics, 0);

        let poll: Poll<Option<Result<RecordBatch>>> = Poll::Ready(None);
        let returned = bm.record_poll(poll);
        assert!(matches!(returned, Poll::Ready(None)));

        let snap = metrics.clone_inner();
        assert!(end_timestamp_recorded(&snap));
        assert_eq!(snap.output_rows(), Some(0));
    }

    #[test]
    fn record_poll_ready_some_err_calls_done() {
        let metrics = ExecutionPlanMetricsSet::new();
        let bm = BaselineMetrics::new(&metrics, 0);

        let poll: Poll<Option<Result<RecordBatch>>> = Poll::Ready(Some(Err(
            fdapquery_common::FdapQueryError::Execution("boom".into()),
        )));
        let returned = bm.record_poll(poll);
        assert!(matches!(returned, Poll::Ready(Some(Err(_)))));

        let snap = metrics.clone_inner();
        assert!(end_timestamp_recorded(&snap));
    }

    #[test]
    fn record_poll_pending_is_passthrough() {
        let metrics = ExecutionPlanMetricsSet::new();
        let bm = BaselineMetrics::new(&metrics, 0);

        let poll: Poll<Option<Result<RecordBatch>>> = Poll::Pending;
        let returned = bm.record_poll(poll);
        assert!(matches!(returned, Poll::Pending));

        let snap = metrics.clone_inner();
        assert_eq!(snap.output_rows(), Some(0));
        assert!(!end_timestamp_recorded(&snap));
    }
}
