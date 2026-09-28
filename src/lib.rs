use polars::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;
use rayon::prelude::*;
use std::collections::HashMap;

#[derive(Debug, Clone)]
struct MeasurementValues {
    aperf: u64,
    mperf: u64,
    unhalted_core_cycles: u64,
    time: u64,
}

#[derive(Debug, Clone)]
struct Measurement {
    pid: u32,
    pid_start: u64,
    exec_count: u64,
    cpu_core: u32,
    interval_type: u32, // 0 = start, 1 = end
    time: u64,          // microseconds since epoch
    start_values: MeasurementValues,
    end_values: MeasurementValues,
}

#[derive(Debug, Clone)]
struct ProcessedInterval {
    pid: u32,
    pid_start: u64,
    start_time: u64,
    end_time: u64,
    cpu_core: u32,
    delta_aperf: u64,
    delta_mperf: u64,
    delta_unhalted_core_cycles: u64,
    execution_time: f64, // microseconds
    shared_state: String,
}

/// State for tracking active tasks during processing
struct ProcessingState {
    active_1: Option<Measurement>,
    active_2: Option<Measurement>,
    output: Vec<ProcessedInterval>,
}

impl ProcessingState {
    fn new() -> Self {
        Self {
            active_1: None,
            active_2: None,
            output: Vec::new(),
        }
    }

    fn push_metrics(&mut self, metrics: ProcessedInterval) {
        self.output.push(metrics);
    }

    fn push_optional_metrics(&mut self, metrics: Option<ProcessedInterval>) {
        if let Some(m) = metrics {
            self.output.push(m);
        }
    }
}

/// Converts a Polars DataFrame to a Vec<Measurement>
fn dataframe_to_measurements(mut df: DataFrame) -> Result<Vec<Measurement>, String> {
    // Ensure the DataFrame is rechunked for better performance
    df.rechunk_mut();

    // Extract columns as strongly-typed series
    let pid_series = df
        .column("pid")
        .map_err(|e| e.to_string())?
        .u32()
        .map_err(|e| e.to_string())?;

    let pid_start_series = df
        .column("pid_start")
        .map_err(|e| e.to_string())?
        .u64()
        .map_err(|e| e.to_string())?;

    let exec_count_series = df
        .column("exec_count")
        .map_err(|e| e.to_string())?
        .u64()
        .map_err(|e| e.to_string())?;

    let cpu_core_series = df
        .column("cpu_core")
        .map_err(|e| e.to_string())?
        .u32()
        .map_err(|e| e.to_string())?;

    let time_start_series = df
        .column("time_start")
        .map_err(|e| e.to_string())?
        .u64()
        .map_err(|e| e.to_string())?;

    let aperf_start_series = df
        .column("aperf_start")
        .map_err(|e| e.to_string())?
        .u64()
        .map_err(|e| e.to_string())?;

    let mperf_start_series = df
        .column("mperf_start")
        .map_err(|e| e.to_string())?
        .u64()
        .map_err(|e| e.to_string())?;

    let ucc_start_series = df
        .column("unhalted_core_cycles_start")
        .map_err(|e| e.to_string())?
        .u64()
        .map_err(|e| e.to_string())?;

    let time_end_series = df
        .column("time_end")
        .map_err(|e| e.to_string())?
        .u64()
        .map_err(|e| e.to_string())?;

    let aperf_end_series = df
        .column("aperf_end")
        .map_err(|e| e.to_string())?
        .u64()
        .map_err(|e| e.to_string())?;

    let mperf_end_series = df
        .column("mperf_end")
        .map_err(|e| e.to_string())?
        .u64()
        .map_err(|e| e.to_string())?;

    let ucc_end_series = df
        .column("unhalted_core_cycles_end")
        .map_err(|e| e.to_string())?
        .u64()
        .map_err(|e| e.to_string())?;

    let mut data = Vec::with_capacity(df.height());

    // remove all values that:
    // - Case-1: have a time difference of zero between start and end
    // - Case-2: have zero values in any aperf, mperf, or ucc values
    // - Case-3: have no end or no start values
    for i in 0..df.height() {
        // The ? operator unwraps Ok(value) or returns early with Err(error)
        // ID
        let pid = pid_series
            .get(i)
            .ok_or_else(|| "Failed to get pid value".to_string())?;
        let pid_start = pid_start_series
            .get(i)
            .ok_or_else(|| "Failed to get pid_start value".to_string())?;

        let exec_count = exec_count_series
            .get(i)
            .ok_or_else(|| "Failed to get exec_count value".to_string())?;
        let cpu_core = cpu_core_series
            .get(i)
            .ok_or_else(|| "Failed to get cpu_core value".to_string())?;
        // start
        let start_time = time_start_series
            .get(i)
            .ok_or_else(|| "Failed to get time_start value".to_string())?;
        let aperf_start = aperf_start_series
            .get(i)
            .ok_or_else(|| "Failed to get aperf_start value".to_string())?;
        let mperf_start = mperf_start_series
            .get(i)
            .ok_or_else(|| "Failed to get mperf_start value".to_string())?;
        let unhalted_core_cycles_start = ucc_start_series
            .get(i)
            .ok_or_else(|| "Failed to get unhalted_core_cycles_start value".to_string())?;
        // end
        let end_time = time_end_series
            .get(i)
            .ok_or_else(|| "Failed to get time_end value".to_string())?;
        let aperf_end = aperf_end_series
            .get(i)
            .ok_or_else(|| "Failed to get aperf_end value".to_string())?;
        let mperf_end = mperf_end_series
            .get(i)
            .ok_or_else(|| "Failed to get mperf_end value".to_string())?;
        let unhalted_core_cycles_end = ucc_end_series
            .get(i)
            .ok_or_else(|| "Failed to get unhalted_core_cycles_end value".to_string())?;

        // Skip if time difference is zero or any value is zero, handles case-1, case-2, and case-3
        if end_time == start_time
            || aperf_start == 0
            || mperf_start == 0
            || unhalted_core_cycles_start == 0
            || aperf_end == 0
            || mperf_end == 0
            || unhalted_core_cycles_end == 0
        {
            continue;
        }

        let start_values = MeasurementValues {
            aperf: aperf_start,
            mperf: mperf_start,
            unhalted_core_cycles: unhalted_core_cycles_start,
            time: start_time,
        };

        let end_values = MeasurementValues {
            aperf: aperf_end,
            mperf: mperf_end,
            unhalted_core_cycles: unhalted_core_cycles_end,
            time: end_time,
        };

        // Create the Measurement struct
        let measurement_start = Measurement {
            pid: pid,
            pid_start: pid_start,
            exec_count: exec_count,
            cpu_core: cpu_core,
            interval_type: 0, // 0 for start
            time: start_time,
            start_values: start_values.clone(),
            end_values: end_values.clone(),
        };

        let measurement_end = Measurement {
            pid: pid,
            pid_start: pid_start,
            exec_count: exec_count,
            cpu_core: cpu_core,
            interval_type: 1, // 1 for end
            time: end_time,
            start_values: start_values.clone(),
            end_values: end_values.clone(),
        };

        data.push(measurement_start);
        data.push(measurement_end);
    }

    Ok(data)
}

/// Converts a Vec<ProcessedInterval> to a Polars DataFrame
fn processed_intervals_to_dataframe(
    intervals: Vec<ProcessedInterval>,
) -> Result<DataFrame, String> {
    DataFrame::new(vec![
        Column::new(
            "pid".into(),
            intervals.iter().map(|i| i.pid).collect::<Vec<u32>>(),
        ),
        Column::new(
            "pid_start".into(),
            intervals.iter().map(|i| i.pid_start).collect::<Vec<u64>>(),
        ),
        Column::new(
            "start_time".into(),
            intervals.iter().map(|i| i.start_time).collect::<Vec<u64>>(),
        ),
        Column::new(
            "end_time".into(),
            intervals.iter().map(|i| i.end_time).collect::<Vec<u64>>(),
        ),
        Column::new(
            "cpu_core".into(),
            intervals.iter().map(|i| i.cpu_core).collect::<Vec<u32>>(),
        ),
        Column::new(
            "delta_aperf".into(),
            intervals
                .iter()
                .map(|i| i.delta_aperf)
                .collect::<Vec<u64>>(),
        ),
        Column::new(
            "delta_mperf".into(),
            intervals
                .iter()
                .map(|i| i.delta_mperf)
                .collect::<Vec<u64>>(),
        ),
        Column::new(
            "delta_unhalted_core_cycles".into(),
            intervals
                .iter()
                .map(|i| i.delta_unhalted_core_cycles)
                .collect::<Vec<u64>>(),
        ),
        Column::new(
            "execution_time".into(),
            intervals
                .iter()
                .map(|i| i.execution_time)
                .collect::<Vec<f64>>(),
        ),
        Column::new(
            "shared_state".into(),
            intervals
                .iter()
                .map(|i| i.shared_state.as_str())
                .collect::<Vec<&str>>(),
        ),
    ])
    .map_err(|e| e.to_string())
}

/// Helper function to safely calculate difference with weight
fn safe_diff(end_val: u64, start_val: u64, weight: f64) -> u64 {
    if end_val >= start_val {
        ((end_val - start_val) as f64 * weight) as u64
    } else {
        0
    }
}

/// Linear interpolation of performance counter values for a task at an intermediate time point
fn linear_interpolate_sibling_values(
    active: &Measurement,
    measurement: &Measurement,
) -> Result<(Option<Measurement>, Option<Measurement>), String> {
    let time_diff = active.end_values.time - active.start_values.time;
    let time_factor = (measurement.time - active.start_values.time) as f64 / time_diff as f64;

    // Calculate interpolated values
    let aperf_intermediate = active.start_values.aperf
        + ((active.end_values.aperf as f64 - active.start_values.aperf as f64) * time_factor)
            as u64;
    let mperf_intermediate = active.start_values.mperf
        + ((active.end_values.mperf as f64 - active.start_values.mperf as f64) * time_factor)
            as u64;
    let unhalted_core_cycles_intermediate = active.start_values.unhalted_core_cycles
        + ((active.end_values.unhalted_core_cycles as f64
            - active.start_values.unhalted_core_cycles as f64)
            * time_factor) as u64;

    let active_intermediate_start = Measurement {
        pid: active.pid,
        pid_start: active.pid_start,
        exec_count: active.exec_count,
        cpu_core: active.cpu_core,
        interval_type: active.interval_type,
        time: measurement.time,
        start_values: MeasurementValues {
            aperf: aperf_intermediate,
            mperf: mperf_intermediate,
            unhalted_core_cycles: unhalted_core_cycles_intermediate,
            // The next cut interpolates from here, so the base time has to move with the values.
            time: measurement.time,
        },
        end_values: active.end_values.clone(),
    };

    let active_intermediate_end = Measurement {
        interval_type: 1, // Change to end
        ..active_intermediate_start.clone()
    };

    // Check if values are unchanged
    if active_intermediate_start.start_values.aperf == active.end_values.aperf
        && active_intermediate_start.start_values.mperf == active.end_values.mperf
        && active_intermediate_start.start_values.unhalted_core_cycles
            == active.end_values.unhalted_core_cycles
    {
        return Ok((None, None));
    }

    Ok((
        Some(active_intermediate_start),
        Some(active_intermediate_end),
    ))
}

/// Calculate split metrics for an active task at an intermediate point
fn calc_split(
    active: &Measurement,
    measurement: &Measurement,
    interval_type: &str,
    weight: (f64, f64, f64),
) -> Result<
    (
        Option<Measurement>,
        Option<Measurement>,
        Option<ProcessedInterval>,
    ),
    String,
> {
    let (active_intermediate_start, active_intermediate_end) =
        linear_interpolate_sibling_values(active, measurement).map_err(|e| e.to_string())?;

    if active_intermediate_start.is_none() || active_intermediate_end.is_none() {
        return Ok((None, None, None));
    }

    match (active_intermediate_start, active_intermediate_end) {
        (Some(active_intermediate_start), Some(active_intermediate_end)) => {
            let active_metrics = ProcessedInterval {
                pid: active.pid,
                pid_start: active.pid_start,
                start_time: active.time,
                end_time: active_intermediate_end.time,
                cpu_core: active.cpu_core,
                delta_aperf: safe_diff(
                    active_intermediate_start.start_values.aperf,
                    active.start_values.aperf,
                    weight.0,
                ),
                delta_mperf: safe_diff(
                    active_intermediate_start.start_values.mperf,
                    active.start_values.mperf,
                    weight.1,
                ),
                delta_unhalted_core_cycles: safe_diff(
                    active_intermediate_start.start_values.unhalted_core_cycles,
                    active.start_values.unhalted_core_cycles,
                    weight.2,
                ),
                execution_time: (measurement.time - active.time) as f64,
                shared_state: interval_type.to_string(),
            };

            Ok((
                Some(active_intermediate_start),
                Some(active_intermediate_end),
                Some(active_metrics),
            ))
        }
        _ => Ok((None, None, None)),
    }
}

/// Calculate end metrics for a task when it's descheduled
fn calc_out_end(
    active: &Measurement,
    measurement: &Measurement,
    interval_type: &str,
    weight: (f64, f64, f64),
) -> ProcessedInterval {
    // active is the active task that is being descheduled
    // measurement is the measurement at the time of descheduling
    ProcessedInterval {
        pid: active.pid,
        pid_start: active.pid_start,
        start_time: active.time,
        end_time: measurement.time,
        cpu_core: active.cpu_core,
        delta_aperf: safe_diff(
            measurement.end_values.aperf,
            active.start_values.aperf,
            weight.0,
        ),
        delta_mperf: safe_diff(
            measurement.end_values.mperf,
            active.start_values.mperf,
            weight.1,
        ),
        delta_unhalted_core_cycles: safe_diff(
            measurement.end_values.unhalted_core_cycles,
            active.start_values.unhalted_core_cycles,
            weight.2,
        ),
        execution_time: (measurement.time - active.time) as f64,
        shared_state: interval_type.to_string(),
    }
}

/// Handle a task start event
fn handle_task_start(state: &mut ProcessingState, measurement: &Measurement) -> Result<(), String> {
    if state.active_1.is_none() {
        state.active_1 = Some(measurement.clone());
        if let Some(ref active_2_ref) = state.active_2 {
            if active_2_ref.time < measurement.time {
                let (active_intermediate_start, _, active_metrics) =
                    calc_split(active_2_ref, measurement, "alone", (1.0, 1.0, 1.0))?;
                state.push_optional_metrics(active_metrics);
                state.active_2 = active_intermediate_start;
            }
        }
    } else if state.active_2.is_none() {
        state.active_2 = Some(measurement.clone());
        if let Some(ref active_1_ref) = state.active_1 {
            if active_1_ref.time < measurement.time {
                let (active_intermediate_start, _, active_metrics) =
                    calc_split(active_1_ref, measurement, "alone", (1.0, 1.0, 1.0))?;
                state.push_optional_metrics(active_metrics);
                state.active_1 = active_intermediate_start;
            }
        }
    }
    Ok(())
}

/// Handle a task end event when only one task is active
fn handle_single_task_end(
    state: &mut ProcessingState,
    measurement: &Measurement,
    is_active_1: bool,
) {
    if is_active_1 {
        if let Some(ref active_1_ref) = state.active_1 {
            let active_metrics = calc_out_end(active_1_ref, measurement, "alone", (1.0, 1.0, 1.0));
            state.push_metrics(active_metrics);
            state.active_1 = None;
        }
    } else {
        if let Some(ref active_2_ref) = state.active_2 {
            let active_metrics = calc_out_end(active_2_ref, measurement, "alone", (1.0, 1.0, 1.0));
            state.push_metrics(active_metrics);
            state.active_2 = None;
        }
    }
}

/// Handle a task end event when both tasks are active
fn handle_both_tasks_end(
    state: &mut ProcessingState,
    measurement: &Measurement,
    ht_factor: f64,
) -> Result<(), String> {
    let (active_1_ref, active_2_ref) = match (&state.active_1, &state.active_2) {
        (Some(a1), Some(a2)) => (a1, a2),
        _ => return Err("Expected both tasks to be active".to_string()),
    };

    if active_1_ref.pid == measurement.pid && active_1_ref.exec_count == measurement.exec_count {
        // active_1 is terminated
        let active_metrics_1 =
            calc_out_end(active_1_ref, measurement, "overlap", (1.0, 1.0, ht_factor));

        let (active_intermediate_start, _, active_metrics_2) =
            calc_split(active_2_ref, measurement, "overlap", (1.0, 1.0, ht_factor))?;

        state.push_optional_metrics(active_metrics_2);
        state.active_2 = active_intermediate_start;

        state.push_metrics(active_metrics_1);
        state.active_1 = None;
    } else if active_2_ref.pid == measurement.pid
        && active_2_ref.exec_count == measurement.exec_count
    {
        // active_2 is terminated
        let active_metrics_2 =
            calc_out_end(active_2_ref, measurement, "overlap", (1.0, 1.0, ht_factor));

        let (active_intermediate_start, _, active_metrics_1) =
            calc_split(active_1_ref, measurement, "overlap", (1.0, 1.0, ht_factor))?;

        state.push_optional_metrics(active_metrics_1);
        state.active_1 = active_intermediate_start;

        state.push_metrics(active_metrics_2);
        state.active_2 = None;
    } else {
        return Err(format!(
            "Measurement {:?} does not match any active task. Active tasks: {:?}, {:?}",
            measurement, active_1_ref, active_2_ref
        ));
    }
    Ok(())
}

/// Handle a task end event
fn handle_task_end(
    state: &mut ProcessingState,
    measurement: &Measurement,
    ht_factor: f64,
) -> Result<(), String> {
    match (&state.active_1, &state.active_2) {
        (Some(_), None) => {
            // active_1 is terminated
            handle_single_task_end(state, measurement, true);
        }
        (None, Some(_)) => {
            // active_2 is terminated
            handle_single_task_end(state, measurement, false);
        }
        (Some(_), Some(_)) => {
            // Both tasks are active - handle both
            // This requires checking which task is being terminated
            // and calculating the split metrics accordingly
            handle_both_tasks_end(state, measurement, ht_factor)?;
        }
        (None, None) => {
            // No active tasks - skip this measurement
        }
    }
    Ok(())
}

/// Main function to process monitoring data
fn process_monitoring_data(
    raw_data: Vec<Measurement>,
    ht_factor: f64,
) -> Result<Vec<ProcessedInterval>, String> {
    if raw_data.is_empty() {
        return Ok(Vec::new());
    }

    let mut state = ProcessingState::new();

    for measurement in raw_data.iter() {
        if measurement.interval_type == 0 {
            // A new task starts
            handle_task_start(&mut state, measurement)?;
        } else if measurement.interval_type == 1 {
            // An existing task is terminated
            handle_task_end(&mut state, measurement, ht_factor)?;
        }
    }

    Ok(state.output)
}

fn prepare_one_df(df: DataFrame, ht_factor: f64) -> Result<DataFrame, String> {
    if df.height() == 0 {
        return Ok(DataFrame::default());
    }

    let mut data = dataframe_to_measurements(df)?;

    data.sort_unstable_by(|a, b| {
        a.time
            .cmp(&b.time)
            .then(b.interval_type.cmp(&a.interval_type))
    });

    let processed = process_monitoring_data(data, ht_factor)?;
    processed_intervals_to_dataframe(processed)
}

#[pyfunction]
fn prepare_monitoring_data(
    py: Python<'_>,
    py_df: PyDataFrame,
    ht_factor: f64,
) -> PyResult<PyDataFrame> {
    // Convert to a Rust DataFrame (zero-copy)
    let df: DataFrame = py_df.into();

    let result: Result<DataFrame, String> = py.detach(|| prepare_one_df(df, ht_factor));

    match result {
        Ok(out_df) => Ok(PyDataFrame(out_df)),
        Err(msg) => Err(PyRuntimeError::new_err(msg)),
    }
}
// #[pyfunction]
// fn prepare_monitoring_data_for_tuples(
//     py: Python<'_>,
//     py_df: PyDataFrame,
//     core_pairs: Vec<(i32, i32)>,
//     ht_factor: f64,
// ) -> PyResult<Py<PyDict>> {
//     let df: DataFrame = py_df.into();

//     // Detach: return Rust-native data only (no Python objects inside)
//     let result: Result<Vec<((i32, i32), DataFrame)>, String> = py.detach(move || {
//         if df.height() == 0 {
//             return Ok(core_pairs
//                 .iter()
//                 .copied()
//                 .map(|pair| (pair, DataFrame::default()))
//                 .collect());
//         }

//         // Partition once by cpu_core
//         let parts = df
//             .partition_by(["cpu_core"], true)
//             .map_err(|e| e.to_string())?;

//         // cpu_core -> partition DF
//         let mut by_core: HashMap<i32, DataFrame> = HashMap::with_capacity(parts.len());
//         for p in parts {
//             if p.height() == 0 {
//                 continue;
//             }
//             let core_val = p
//                 .column("cpu_core").map_err(|e| e.to_string())?
//                 .u32().map_err(|e| e.to_string())?
//                 .get(0)
//                 .ok_or_else(|| "cpu_core partition had no first value".to_string())?;

//             by_core.insert(core_val as i32, p);
//         }

//         // One output DF per tuple
//         let mut out: Vec<((i32, i32), DataFrame)> = Vec::with_capacity(core_pairs.len());

//         for (a, b) in core_pairs.iter().copied() {
//             let mut combined: Option<DataFrame> = None;

//             for core in [a, b] {
//                 if core < 0 {
//                     continue;
//                 }
//                 if let Some(part) = by_core.get(&core) {
//                     if combined.is_none() {
//                         combined = Some(part.clone());
//                     } else {
//                         combined
//                             .as_mut()
//                             .unwrap()
//                             .vstack_mut(part)
//                             .map_err(|e| e.to_string())?;
//                     }
//                 }
//             }

//             let tuple_df = combined.unwrap_or_else(DataFrame::default);
//             let prepared = prepare_one_df(tuple_df, ht_factor)?;
//             out.push(((a, b), prepared));
//         }

//         Ok(out)
//     });

//     // Attached again: build the Python dict
//     match result {
//         Ok(pairs) => {
//             // New PyO3 API returns Bound<'py, PyDict>
//             let dict = PyDict::new(py);

//             for ((a, b), df) in pairs {
//                 // key: (a, b), value: Polars DataFrame
//                 dict.set_item((a, b), PyDataFrame(df))?;
//             }

//             // Convert Bound<'py, PyDict> -> Py<PyDict> for returning from #[pyfunction]
//             Ok(dict.unbind())
//         }
//         Err(msg) => Err(PyRuntimeError::new_err(msg)),
//     }
// }

#[pyfunction]
fn prepare_monitoring_data_for_tuples(
    py: Python<'_>,
    py_df: PyDataFrame,
    core_pairs: Vec<(i32, i32)>,
    ht_factor: f64,
) -> PyResult<PyDataFrame> {
    let df: DataFrame = py_df.into();

    // Detached: return a single Rust DataFrame (no Python objects)
    let result: Result<DataFrame, String> = py.detach(move || {
        if df.height() == 0 || core_pairs.is_empty() {
            return Ok(DataFrame::default());
        }

        // Partition once by cpu_core
        let parts = df
            .partition_by(["cpu_core"], true)
            .map_err(|e| e.to_string())?;

        // cpu_core -> partition DF
        let mut by_core: HashMap<i32, DataFrame> = HashMap::with_capacity(parts.len());
        for p in parts {
            if p.height() == 0 {
                continue;
            }
            let core_val = p
                .column("cpu_core")
                .map_err(|e| e.to_string())?
                .u32()
                .map_err(|e| e.to_string())?
                .get(0)
                .ok_or_else(|| "cpu_core partition had no first value".to_string())?;

            by_core.insert(core_val as i32, p);
        }

        // --- Per-tuple computation ---
        // Sequential version (minimal dependencies):
        // let mut prepared_dfs: Vec<DataFrame> = Vec::with_capacity(core_pairs.len());

        // for (a, b) in core_pairs.iter().copied() {
        //     let mut combined: Option<DataFrame> = None;

        //     for core in [a, b] {
        //         if core < 0 {
        //             continue;
        //         }
        //         if let Some(part) = by_core.get(&core) {
        //             if combined.is_none() {
        //                 combined = Some(part.clone());
        //             } else {
        //                 combined
        //                     .as_mut()
        //                     .unwrap()
        //                     .vstack_mut(part)
        //                     .map_err(|e| e.to_string())?;
        //             }
        //         }
        //     }

        //     let tuple_df = combined.unwrap_or_else(DataFrame::default);
        //     let prepared = prepare_one_df(tuple_df, ht_factor)?;
        //     prepared_dfs.push(prepared);
        // }

        // --- Parallel version (optional): replace the sequential block above with this ---
        let prepared_dfs: Vec<DataFrame> = core_pairs
            .par_iter()
            .copied()
            .map(|(a, b)| -> Result<DataFrame, String> {
                let mut combined: Option<DataFrame> = None;

                for core in [a, b] {
                    if core < 0 {
                        continue;
                    }
                    if let Some(part) = by_core.get(&core) {
                        if combined.is_none() {
                            combined = Some(part.clone());
                        } else {
                            combined
                                .as_mut()
                                .unwrap()
                                .vstack_mut(part)
                                .map_err(|e| e.to_string())?;
                        }
                    }
                }

                let tuple_df = combined.unwrap_or_else(DataFrame::default);
                prepare_one_df(tuple_df, ht_factor)
            })
            .collect::<Result<Vec<_>, String>>()?;

        // --- Concatenate all prepared frames into one ---
        // Filter out schema-less empties (DataFrame::default()) to avoid schema mismatch errors
        let mut iter = prepared_dfs.into_iter().filter(|d| d.width() > 0);

        let mut out = match iter.next() {
            Some(df0) => df0,
            None => return Ok(DataFrame::default()),
        };

        for df in iter {
            out.vstack_mut(&df).map_err(|e| e.to_string())?;
        }

        Ok(out)
    });

    // Attached again: convert Rust result to Python error/return
    match result {
        Ok(out_df) => Ok(PyDataFrame(out_df)),
        Err(msg) => Err(PyRuntimeError::new_err(msg)),
    }
}

#[pymodule]
fn metrion_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(prepare_monitoring_data, m)?)?;
    m.add_function(wrap_pyfunction!(prepare_monitoring_data_for_tuples, m)?)?;
    Ok(())
}
