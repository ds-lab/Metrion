#!/usr/bin/env python3
"""Plot run-to-run EffiBench energy/performance trends per model.

The script scans an artifacts folder that contains subdirectories with
``*_energy_results.json`` files. It groups files by model name and run number,
keeps only numbered runs, finds the set of problems present in all runs of a
group, intersects this set with the canonical baseline file, and then
aggregates the requested metrics per run:

- If ``*_stats`` are present, use per-problem ``mean`` for the line and
    aggregate uncertainty as root-sum-square of per-problem ``stdev`` values.
- If ``*_stats`` are missing, fall back to plotting the scalar values without
    error bars.
- ``--problem-aggregation {sum,mean}`` controls whether the plot aggregates
    by summing common problems (default) or averaging per problem.
- ``--results-file`` allows plotting directly from one results JSON file
    without requiring numbered artifact subdirectories.

For each model group, a 4-panel matplotlib figure is written where the x-axis
is the run number. The first three panels always include a canonical baseline.
The ``cpu_core_pid_work`` panel is plotted when that metric is available in all
runs and the canonical file for a common problem subset. When stdev is
available, coefficient of variation (CV = stdev / mean) is overlaid as a
secondary y-axis in each panel.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import cast

# Use a writable matplotlib cache location when HOME config is read-only.
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = str(Path(tempfile.gettempdir()) / "matplotlib")

import matplotlib.pyplot as plt

RUN_RESULTS_RE = re.compile(r"^(?P<base>.+?)_(?P<run>\d+)_energy_results\.json$")
ENERGY_RESULTS_GLOB = "*_energy_results.json"


@dataclass(frozen=True)
class Metrics:
    energy_j: float
    unhalted_cycles: float
    kernel_time_ms: float
    cpu_core_pid_work: float | None = None
    energy_j_stdev: float | None = None
    unhalted_cycles_stdev: float | None = None
    kernel_time_ms_stdev: float | None = None
    cpu_core_pid_work_stdev: float | None = None
    runs_total: int | None = None


ProblemKey = int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("results_sadra/energy_batch_artifacts_20260414_210505"),
        help="Folder with numbered run artifact directories.",
    )
    parser.add_argument(
        "--results-file",
        type=Path,
        default=None,
        help=("Direct path to one *_energy_results.json file. When set, --artifacts-dir discovery is skipped."),
    )
    parser.add_argument(
        "--canonical-file",
        type=Path,
        default=Path("results_sadra/EffiLearner/results/canonical_EffiBench_energy_results.json"),
        help="Canonical baseline energy results JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for plots and CSV summary. Defaults to <artifacts-dir>/plots_run_metrics.",
    )
    parser.add_argument(
        "--min-runs",
        type=int,
        default=2,
        help="Minimum number of runs required per model group.",
    )
    parser.add_argument(
        "--problem-aggregation",
        choices=("sum", "mean"),
        default="sum",
        help=("How to aggregate metrics across common problems: 'sum' (default) or 'mean' per problem."),
    )
    return parser.parse_args()


def _as_problem_key(entry: dict) -> ProblemKey | None:
    # Newer artifacts may omit task_name but still provide problem_idx.
    # Use a stable numeric key that is shared with canonical results.
    problem_idx = entry.get("problem_idx")
    if problem_idx is None:
        problem_idx = entry.get("source_entry_index")
    if problem_idx is None:
        return None

    try:
        idx = int(problem_idx)
    except (TypeError, ValueError):
        return None

    return idx


def _as_metrics(entry: dict) -> Metrics | None:
    energy = entry.get("energy")
    perf = entry.get("performance")
    if not isinstance(energy, dict) or not isinstance(perf, dict):
        return None

    energy_stats = entry.get("energy_stats")
    perf_stats = entry.get("performance_stats")

    def _extract_metric(
        value_dict: dict,
        stats_dict: dict | None,
        key: str,
    ) -> tuple[float | None, float | None, int | None]:
        value: float | None = None
        stdev: float | None = None
        n: int | None = None

        if isinstance(stats_dict, dict):
            stat_obj = stats_dict.get(key)
            if isinstance(stat_obj, dict):
                mean_raw = stat_obj.get("mean")
                if mean_raw is not None:
                    try:
                        value = float(mean_raw)
                    except (TypeError, ValueError):
                        value = None

                stdev_raw = stat_obj.get("stdev")
                if stdev_raw is not None:
                    try:
                        stdev = float(stdev_raw)
                    except (TypeError, ValueError):
                        stdev = None

                n_raw = stat_obj.get("n")
                if n_raw is not None:
                    try:
                        n = int(n_raw)
                    except (TypeError, ValueError):
                        n = None

        if value is None:
            raw = value_dict.get(key)
            if raw is not None:
                try:
                    value = float(raw)
                except (TypeError, ValueError):
                    value = None

        return value, stdev, n

    try:
        runs_total = entry.get("runs_total")
        if runs_total is not None:
            runs_total = int(runs_total)

        energy_j, energy_j_stdev, energy_n = _extract_metric(energy, energy_stats, "cpu_total_energy_j")
        unhalted_cycles, unhalted_cycles_stdev, cycles_n = _extract_metric(
            perf,
            perf_stats,
            "unhalted_cycles",
        )
        kernel_time_ms, kernel_time_ms_stdev, kernel_n = _extract_metric(perf, perf_stats, "kernel_time_ms")
        cpu_core_pid_work, cpu_core_pid_work_stdev, work_n = _extract_metric(
            perf,
            perf_stats,
            "cpu_core_pid_work",
        )

        if energy_j is None or unhalted_cycles is None or kernel_time_ms is None:
            return None

        if runs_total is None:
            n_candidates = [n for n in (energy_n, cycles_n, kernel_n, work_n) if isinstance(n, int) and n > 0]
            if n_candidates:
                runs_total = Counter(n_candidates).most_common(1)[0][0]

        return Metrics(
            energy_j=energy_j,
            unhalted_cycles=unhalted_cycles,
            kernel_time_ms=kernel_time_ms,
            cpu_core_pid_work=cpu_core_pid_work,
            energy_j_stdev=energy_j_stdev,
            unhalted_cycles_stdev=unhalted_cycles_stdev,
            kernel_time_ms_stdev=kernel_time_ms_stdev,
            cpu_core_pid_work_stdev=cpu_core_pid_work_stdev,
            runs_total=runs_total,
        )
    except (KeyError, TypeError, ValueError):
        return None


def load_metrics_by_problem(json_file: Path) -> dict[ProblemKey, Metrics]:
    with json_file.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, list):
        raise ValueError(f"Expected list JSON in {json_file}, got {type(data).__name__}")

    metrics_by_problem: dict[ProblemKey, Metrics] = {}
    for entry in data:
        if not isinstance(entry, dict):
            continue

        key = _as_problem_key(entry)
        metrics = _as_metrics(entry)
        if key is None or metrics is None:
            continue

        # Keep first occurrence if duplicates appear.
        metrics_by_problem.setdefault(key, metrics)

    return metrics_by_problem


def discover_numbered_runs(artifacts_dir: Path) -> dict[str, dict[int, Path]]:
    grouped: dict[str, dict[int, Path]] = {}

    for subdir in sorted(artifacts_dir.iterdir()):
        if not subdir.is_dir():
            continue

        for result_file in sorted(subdir.glob(ENERGY_RESULTS_GLOB)):
            match = RUN_RESULTS_RE.match(result_file.name)
            if not match:
                continue

            base_name = match.group("base")
            run_no = int(match.group("run"))

            run_map = grouped.setdefault(base_name, {})
            if run_no in run_map:
                print(
                    "Warning: duplicate run",
                    run_no,
                    "for",
                    base_name,
                    "-> keeping",
                    run_map[run_no],
                    "ignoring",
                    result_file,
                )
                continue

            run_map[run_no] = result_file

    return grouped


def discover_single_run_file(results_file: Path) -> dict[str, dict[int, Path]]:
    match = RUN_RESULTS_RE.match(results_file.name)
    if match:
        return {
            match.group("base"): {
                int(match.group("run")): results_file,
            }
        }

    base_name = results_file.stem
    suffix = "_energy_results"
    if base_name.endswith(suffix):
        base_name = base_name[: -len(suffix)]
    if not base_name:
        base_name = "single_run"

    return {
        base_name: {
            1: results_file,
        }
    }


def sum_metrics(
    problem_map: dict[ProblemKey, Metrics],
    keys: set[ProblemKey],
    problem_aggregation: str = "sum",
) -> Metrics:
    energy_j = 0.0
    unhalted_cycles = 0.0
    kernel_time_ms = 0.0
    energy_var = 0.0
    cycles_var = 0.0
    kernel_var = 0.0
    has_energy_std = True
    has_cycles_std = True
    has_kernel_std = True

    cpu_core_pid_work: float | None = 0.0
    cpu_core_pid_work_var = 0.0
    has_work_std = True

    for key in keys:
        m = problem_map[key]
        energy_j += m.energy_j
        unhalted_cycles += m.unhalted_cycles
        kernel_time_ms += m.kernel_time_ms

        if m.energy_j_stdev is not None:
            energy_var += m.energy_j_stdev * m.energy_j_stdev
        else:
            has_energy_std = False

        if m.unhalted_cycles_stdev is not None:
            cycles_var += m.unhalted_cycles_stdev * m.unhalted_cycles_stdev
        else:
            has_cycles_std = False

        if m.kernel_time_ms_stdev is not None:
            kernel_var += m.kernel_time_ms_stdev * m.kernel_time_ms_stdev
        else:
            has_kernel_std = False

        if m.cpu_core_pid_work is None:
            cpu_core_pid_work = None
        elif cpu_core_pid_work is not None:
            cpu_core_pid_work += m.cpu_core_pid_work

        if m.cpu_core_pid_work is None or m.cpu_core_pid_work_stdev is None:
            has_work_std = False
        else:
            cpu_core_pid_work_var += m.cpu_core_pid_work_stdev * m.cpu_core_pid_work_stdev

    scale = 1.0
    if problem_aggregation == "mean" and keys:
        scale = 1.0 / len(keys)

    cpu_core_pid_work_scaled = cpu_core_pid_work * scale if cpu_core_pid_work is not None else None

    return Metrics(
        energy_j=energy_j * scale,
        unhalted_cycles=unhalted_cycles * scale,
        kernel_time_ms=kernel_time_ms * scale,
        cpu_core_pid_work=cpu_core_pid_work_scaled,
        energy_j_stdev=(math.sqrt(energy_var) * scale) if has_energy_std and keys else None,
        unhalted_cycles_stdev=(math.sqrt(cycles_var) * scale) if has_cycles_std and keys else None,
        kernel_time_ms_stdev=(math.sqrt(kernel_var) * scale) if has_kernel_std and keys else None,
        cpu_core_pid_work_stdev=((math.sqrt(cpu_core_pid_work_var) * scale) if has_work_std and keys else None),
    )


def keys_with_cpu_core_pid_work(problem_map: dict[ProblemKey, Metrics]) -> set[ProblemKey]:
    return {key for key, metrics in problem_map.items() if metrics.cpu_core_pid_work is not None}


def _errorbar_or_line(
    ax: plt.Axes,
    x_values: list[int],
    y_values: list[float | None],
    yerr_values: list[float | None],
    *,
    color: str,
    label_base: str,
) -> None:
    if not y_values or any(val is None for val in y_values):
        return

    y_vals = [float(cast(int | float, val)) for val in y_values]

    if yerr_values and len(yerr_values) == len(y_values) and all(val is not None for val in yerr_values):
        y_err_vals = [float(cast(int | float, val)) for val in yerr_values]
        ax.errorbar(
            x_values,
            y_vals,
            yerr=y_err_vals,
            fmt="-o",
            linewidth=2.0,
            markersize=5,
            capsize=3,
            capthick=1.1,
            elinewidth=1.2,
            color=color,
            label=f"{label_base} ±1σ",
        )
        return

    ax.plot(
        x_values,
        y_vals,
        marker="o",
        linewidth=2.0,
        color=color,
        label=label_base,
    )


def _add_baseline(
    ax: plt.Axes,
    x_values: list[int],
    baseline_mean: float,
    baseline_stdev: float | None,
) -> None:
    ax.axhline(
        baseline_mean,
        linestyle="--",
        linewidth=1.8,
        color="tab:red",
        label="Canonical baseline",
    )
    if baseline_stdev is not None and x_values:
        left = min(x_values)
        right = max(x_values)
        ax.fill_between(
            [left, right],
            [baseline_mean - baseline_stdev, baseline_mean - baseline_stdev],
            [baseline_mean + baseline_stdev, baseline_mean + baseline_stdev],
            color="tab:red",
            alpha=0.12,
            linewidth=0.0,
            label="Canonical ±1σ",
        )


def _coefficient_of_variation(mean_value: float | None, stdev_value: float | None) -> float | None:
    if mean_value is None or stdev_value is None:
        return None

    denom = abs(mean_value)
    if denom <= 0.0:
        return None

    return stdev_value / denom


def _coefficient_of_variation_percent(mean_value: float | None, stdev_value: float | None) -> float | None:
    cv = _coefficient_of_variation(mean_value, stdev_value)
    if cv is None:
        return None
    return cv * 100.0


def _plot_cv_overlay(
    ax: plt.Axes,
    x_values: list[int],
    mean_values: list[float | None],
    stdev_values: list[float | None],
    *,
    baseline_mean: float | None,
    baseline_stdev: float | None,
) -> None:
    run_cv_points = [
        (x, cv * 100.0)
        for x, cv in (
            (
                x,
                _coefficient_of_variation(mean_val, stdev_val),
            )
            for x, mean_val, stdev_val in zip(x_values, mean_values, stdev_values, strict=False)
        )
        if cv is not None
    ]

    baseline_cv = _coefficient_of_variation_percent(baseline_mean, baseline_stdev)
    if not run_cv_points and baseline_cv is None:
        return

    cv_ax = ax.twinx()
    if run_cv_points:
        cv_x, cv_y = zip(*run_cv_points, strict=False)
        cv_ax.plot(
            cv_x,
            cv_y,
            linestyle=":",
            marker="s",
            markersize=3.5,
            linewidth=1.2,
            color="0.35",
            alpha=0.95,
            label="Run CV [%]",
        )

    if baseline_cv is not None:
        cv_ax.axhline(
            baseline_cv,
            linestyle=":",
            linewidth=1.1,
            color="0.55",
            alpha=0.9,
            label="Canonical CV [%]",
        )

    cv_ax.set_ylabel("CV [%]", color="0.35")
    cv_ax.tick_params(axis="y", colors="0.35")
    cv_ax.grid(False)


def infer_runs_per_problem(problem_maps: list[dict[ProblemKey, Metrics]], keys: set[ProblemKey]) -> int | None:
    counts: Counter[int] = Counter()
    for problem_map in problem_maps:
        for key in keys:
            runs_total = problem_map[key].runs_total
            if runs_total is not None and runs_total > 0:
                counts[runs_total] += 1

    if not counts:
        return None

    return counts.most_common(1)[0][0]


def _safe_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return safe.strip("_") or "model"


def _pretty_model_name(base_name: str) -> str:
    prefix = "EffiBench_vllm_"
    if base_name.startswith(prefix):
        return base_name[len(prefix) :]
    return base_name


def write_summary_csv(rows: list[dict[str, str | int | float | None]], output_file: Path) -> None:
    if not rows:
        return

    fieldnames = [
        "model_group",
        "model_label",
        "run",
        "problem_aggregation",
        "common_problems_across_runs",
        "common_problems_with_canonical",
        "common_problems_with_work_metric_runs",
        "common_problems_with_work_metric_canonical",
        "sum_energy_j",
        "sum_energy_j_stdev",
        "sum_energy_cv_pct",
        "sum_unhalted_cycles",
        "sum_unhalted_cycles_stdev",
        "sum_unhalted_cycles_cv_pct",
        "sum_kernel_time_ms",
        "sum_kernel_time_ms_stdev",
        "sum_kernel_time_ms_cv_pct",
        "sum_cpu_core_pid_work",
        "sum_cpu_core_pid_work_stdev",
        "sum_cpu_core_pid_work_cv_pct",
        "baseline_energy_j",
        "baseline_energy_j_stdev",
        "baseline_energy_cv_pct",
        "baseline_unhalted_cycles",
        "baseline_unhalted_cycles_stdev",
        "baseline_unhalted_cycles_cv_pct",
        "baseline_kernel_time_ms",
        "baseline_kernel_time_ms_stdev",
        "baseline_kernel_time_ms_cv_pct",
        "baseline_cpu_core_pid_work",
        "baseline_cpu_core_pid_work_stdev",
        "baseline_cpu_core_pid_work_cv_pct",
    ]

    with output_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    artifacts_dir = args.artifacts_dir.resolve()
    results_file = args.results_file.resolve() if args.results_file is not None else None
    canonical_file = args.canonical_file.resolve()

    if args.output_dir is None:
        if results_file is not None:
            output_dir = results_file.parent / "plots_run_metrics"
        else:
            output_dir = artifacts_dir / "plots_run_metrics"
    else:
        output_dir = args.output_dir.resolve()

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        if args.output_dir is not None:
            raise

        fallback = artifacts_dir.parent / f"{artifacts_dir.name}_plots_run_metrics"
        fallback.mkdir(parents=True, exist_ok=True)
        print(f"Warning: cannot write to {output_dir}, using fallback {fallback}")
        output_dir = fallback

    if results_file is None:
        if not artifacts_dir.exists():
            raise SystemExit(f"Artifacts directory does not exist: {artifacts_dir}")
    else:
        if not results_file.exists():
            raise SystemExit(f"Results file does not exist: {results_file}")
        if not results_file.is_file():
            raise SystemExit(f"Results path is not a file: {results_file}")

    if not canonical_file.exists():
        raise SystemExit(f"Canonical file does not exist: {canonical_file}")

    canonical_by_problem = load_metrics_by_problem(canonical_file)
    if not canonical_by_problem:
        raise SystemExit(f"No valid canonical entries found in: {canonical_file}")

    if results_file is not None:
        grouped_runs = discover_single_run_file(results_file)
    else:
        grouped_runs = discover_numbered_runs(artifacts_dir)

    if not grouped_runs:
        raise SystemExit(f"No numbered run files found under: {artifacts_dir}")

    summary_rows: list[dict[str, str | int | float | None]] = []
    plotted_groups = 0

    mode_label = "sum across problems" if args.problem_aggregation == "sum" else "mean across problems"
    series_label = "Run mean (problem-sum)" if args.problem_aggregation == "sum" else "Run mean (problem-mean)"
    energy_ylabel = "Energy [J]" if args.problem_aggregation == "sum" else "Energy per problem [J]"
    cycles_ylabel = "Unhalted core cycles" if args.problem_aggregation == "sum" else "Unhalted core cycles per problem"
    kernel_ylabel = "Kernel time [ms]" if args.problem_aggregation == "sum" else "Kernel time per problem [ms]"
    work_ylabel = "CPU core PID work" if args.problem_aggregation == "sum" else "CPU core PID work per problem"
    effective_min_runs = 1 if results_file is not None else args.min_runs

    for base_name, run_files in sorted(grouped_runs.items()):
        if len(run_files) < effective_min_runs:
            print(f"Skipping {base_name}: only {len(run_files)} run(s), requires >= {effective_min_runs}")
            continue

        run_numbers = sorted(run_files.keys())
        run_problem_maps = {run_no: load_metrics_by_problem(run_files[run_no]) for run_no in run_numbers}

        problem_sets = [set(problem_map.keys()) for problem_map in run_problem_maps.values()]
        if not problem_sets:
            print(f"Skipping {base_name}: no valid metrics found")
            continue

        common_across_runs = set.intersection(*problem_sets)
        if not common_across_runs:
            print(f"Skipping {base_name}: no common problems across runs")
            continue

        # Keep comparison fair by requiring canonical presence as well.
        common_for_plot = common_across_runs & set(canonical_by_problem.keys())
        if not common_for_plot:
            print(f"Skipping {base_name}: no overlap between run intersection and canonical file")
            continue

        runs_per_problem = infer_runs_per_problem(list(run_problem_maps.values()), common_for_plot)
        runs_per_problem_label = f"{runs_per_problem}x" if runs_per_problem is not None else "unknown"

        baseline = sum_metrics(canonical_by_problem, common_for_plot, args.problem_aggregation)

        sums_by_run = {
            run_no: sum_metrics(run_problem_maps[run_no], common_for_plot, args.problem_aggregation)
            for run_no in run_numbers
        }

        canonical_work_keys = keys_with_cpu_core_pid_work(canonical_by_problem)
        work_problem_sets = [keys_with_cpu_core_pid_work(problem_map) for problem_map in run_problem_maps.values()]
        common_work_across_runs = set.intersection(*work_problem_sets) if work_problem_sets else set()
        common_work_with_canonical = common_work_across_runs & canonical_work_keys

        work_keys_for_series = common_work_with_canonical if common_work_with_canonical else common_work_across_runs

        has_work_metric = len(work_keys_for_series) > 0
        has_work_baseline = len(common_work_with_canonical) > 0
        if has_work_metric:
            work_sums_by_run = {
                run_no: sum_metrics(run_problem_maps[run_no], work_keys_for_series, args.problem_aggregation)
                for run_no in run_numbers
            }
            cpu_core_pid_work_series = [work_sums_by_run[run_no].cpu_core_pid_work for run_no in run_numbers]
            cpu_core_pid_work_err = [work_sums_by_run[run_no].cpu_core_pid_work_stdev for run_no in run_numbers]
            baseline_work = (
                sum_metrics(canonical_by_problem, common_work_with_canonical, args.problem_aggregation)
                if has_work_baseline
                else None
            )
            baseline_cpu_core_pid_work = baseline_work.cpu_core_pid_work if baseline_work is not None else None
            baseline_cpu_core_pid_work_stdev = (
                baseline_work.cpu_core_pid_work_stdev if baseline_work is not None else None
            )
        else:
            baseline_cpu_core_pid_work = None
            baseline_cpu_core_pid_work_stdev = None
            cpu_core_pid_work_series = []
            cpu_core_pid_work_err = []

        model_label = _pretty_model_name(base_name)
        x_values = run_numbers
        energy_series = [sums_by_run[run_no].energy_j for run_no in run_numbers]
        energy_err = [sums_by_run[run_no].energy_j_stdev for run_no in run_numbers]
        cycles_series = [sums_by_run[run_no].unhalted_cycles for run_no in run_numbers]
        cycles_err = [sums_by_run[run_no].unhalted_cycles_stdev for run_no in run_numbers]
        kernel_series = [sums_by_run[run_no].kernel_time_ms for run_no in run_numbers]
        kernel_err = [sums_by_run[run_no].kernel_time_ms_stdev for run_no in run_numbers]

        fig, axes = plt.subplots(4, 1, sharex=True, figsize=(11, 12), layout="constrained")

        _errorbar_or_line(
            axes[0],
            x_values,
            energy_series,
            energy_err,
            color="tab:blue",
            label_base=series_label,
        )
        _add_baseline(axes[0], x_values, baseline.energy_j, baseline.energy_j_stdev)
        _plot_cv_overlay(
            axes[0],
            x_values,
            energy_series,
            energy_err,
            baseline_mean=baseline.energy_j,
            baseline_stdev=baseline.energy_j_stdev,
        )
        axes[0].set_ylabel(energy_ylabel)
        axes[0].set_title(f"CPU total energy ({mode_label}; mean over repeats, CV overlay)")

        _errorbar_or_line(
            axes[1],
            x_values,
            cycles_series,
            cycles_err,
            color="tab:green",
            label_base=series_label,
        )
        _add_baseline(axes[1], x_values, baseline.unhalted_cycles, baseline.unhalted_cycles_stdev)
        _plot_cv_overlay(
            axes[1],
            x_values,
            cycles_series,
            cycles_err,
            baseline_mean=baseline.unhalted_cycles,
            baseline_stdev=baseline.unhalted_cycles_stdev,
        )
        axes[1].set_ylabel(cycles_ylabel)
        axes[1].set_title(f"Unhalted core cycles ({mode_label}; mean over repeats, CV overlay)")

        _errorbar_or_line(
            axes[2],
            x_values,
            kernel_series,
            kernel_err,
            color="tab:purple",
            label_base=series_label,
        )
        _add_baseline(axes[2], x_values, baseline.kernel_time_ms, baseline.kernel_time_ms_stdev)
        _plot_cv_overlay(
            axes[2],
            x_values,
            kernel_series,
            kernel_err,
            baseline_mean=baseline.kernel_time_ms,
            baseline_stdev=baseline.kernel_time_ms_stdev,
        )
        axes[2].set_ylabel(kernel_ylabel)
        axes[2].set_title(f"Kernel time ({mode_label}; mean over repeats, CV overlay)")

        if has_work_metric:
            _errorbar_or_line(
                axes[3],
                x_values,
                cpu_core_pid_work_series,
                cpu_core_pid_work_err,
                color="tab:orange",
                label_base=series_label,
            )

            if baseline_cpu_core_pid_work is not None:
                _add_baseline(
                    axes[3],
                    x_values,
                    baseline_cpu_core_pid_work,
                    baseline_cpu_core_pid_work_stdev,
                )
            else:
                axes[3].text(
                    0.5,
                    0.9,
                    "Canonical baseline unavailable for cpu_core_pid_work",
                    transform=axes[3].transAxes,
                    ha="center",
                    va="center",
                    fontsize=9,
                )

            _plot_cv_overlay(
                axes[3],
                x_values,
                cpu_core_pid_work_series,
                cpu_core_pid_work_err,
                baseline_mean=baseline_cpu_core_pid_work,
                baseline_stdev=baseline_cpu_core_pid_work_stdev,
            )
        else:
            axes[3].text(
                0.5,
                0.5,
                "cpu_core_pid_work not available in all runs",
                transform=axes[3].transAxes,
                ha="center",
                va="center",
                fontsize=10,
            )

        axes[3].set_xlabel("Run number")
        axes[3].set_ylabel(work_ylabel)
        axes[3].set_title(f"CPU core PID work ({mode_label}; mean over repeats, CV overlay)")

        for ax in axes:
            ax.grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(loc="best")

        fig.suptitle(
            (
                f"{model_label} | runs: {len(run_numbers)} | per-problem repeats: {runs_per_problem_label}\n"
                f"problem aggregation: {mode_label}\n"
                f"common problems (runs / with canonical / work runs / work+canonical): "
                f"{len(common_across_runs)} / {len(common_for_plot)} / "
                f"{len(common_work_across_runs)} / {len(common_work_with_canonical)}"
            ),
            fontsize=11,
            x=0.01,
            ha="left",
        )

        output_png = output_dir / f"{_safe_name(model_label)}__run_metrics.png"
        fig.savefig(output_png, dpi=180)
        plt.close(fig)

        for run_no in run_numbers:
            run_sums = sums_by_run[run_no]
            summary_rows.append(
                {
                    "model_group": base_name,
                    "model_label": model_label,
                    "run": run_no,
                    "problem_aggregation": args.problem_aggregation,
                    "common_problems_across_runs": len(common_across_runs),
                    "common_problems_with_canonical": len(common_for_plot),
                    "common_problems_with_work_metric_runs": len(common_work_across_runs),
                    "common_problems_with_work_metric_canonical": len(common_work_with_canonical),
                    "sum_energy_j": run_sums.energy_j,
                    "sum_energy_j_stdev": run_sums.energy_j_stdev,
                    "sum_energy_cv_pct": _coefficient_of_variation_percent(
                        run_sums.energy_j,
                        run_sums.energy_j_stdev,
                    ),
                    "sum_unhalted_cycles": run_sums.unhalted_cycles,
                    "sum_unhalted_cycles_stdev": run_sums.unhalted_cycles_stdev,
                    "sum_unhalted_cycles_cv_pct": _coefficient_of_variation_percent(
                        run_sums.unhalted_cycles,
                        run_sums.unhalted_cycles_stdev,
                    ),
                    "sum_kernel_time_ms": run_sums.kernel_time_ms,
                    "sum_kernel_time_ms_stdev": run_sums.kernel_time_ms_stdev,
                    "sum_kernel_time_ms_cv_pct": _coefficient_of_variation_percent(
                        run_sums.kernel_time_ms,
                        run_sums.kernel_time_ms_stdev,
                    ),
                    "sum_cpu_core_pid_work": (
                        cpu_core_pid_work_series[run_numbers.index(run_no)] if has_work_metric else None
                    ),
                    "sum_cpu_core_pid_work_stdev": (
                        cpu_core_pid_work_err[run_numbers.index(run_no)] if has_work_metric else None
                    ),
                    "sum_cpu_core_pid_work_cv_pct": (
                        _coefficient_of_variation_percent(
                            cpu_core_pid_work_series[run_numbers.index(run_no)],
                            cpu_core_pid_work_err[run_numbers.index(run_no)],
                        )
                        if has_work_metric
                        else None
                    ),
                    "baseline_energy_j": baseline.energy_j,
                    "baseline_energy_j_stdev": baseline.energy_j_stdev,
                    "baseline_energy_cv_pct": _coefficient_of_variation_percent(
                        baseline.energy_j,
                        baseline.energy_j_stdev,
                    ),
                    "baseline_unhalted_cycles": baseline.unhalted_cycles,
                    "baseline_unhalted_cycles_stdev": baseline.unhalted_cycles_stdev,
                    "baseline_unhalted_cycles_cv_pct": _coefficient_of_variation_percent(
                        baseline.unhalted_cycles,
                        baseline.unhalted_cycles_stdev,
                    ),
                    "baseline_kernel_time_ms": baseline.kernel_time_ms,
                    "baseline_kernel_time_ms_stdev": baseline.kernel_time_ms_stdev,
                    "baseline_kernel_time_ms_cv_pct": _coefficient_of_variation_percent(
                        baseline.kernel_time_ms,
                        baseline.kernel_time_ms_stdev,
                    ),
                    "baseline_cpu_core_pid_work": baseline_cpu_core_pid_work,
                    "baseline_cpu_core_pid_work_stdev": baseline_cpu_core_pid_work_stdev,
                    "baseline_cpu_core_pid_work_cv_pct": _coefficient_of_variation_percent(
                        baseline_cpu_core_pid_work,
                        baseline_cpu_core_pid_work_stdev,
                    ),
                }
            )

        plotted_groups += 1
        print(
            f"Plotted {model_label}: {len(run_numbers)} runs, "
            f"{len(common_across_runs)} common across runs, "
            f"{len(common_for_plot)} with canonical"
        )

    summary_file = output_dir / "run_metrics_summary.csv"
    write_summary_csv(summary_rows, summary_file)

    if plotted_groups == 0:
        raise SystemExit("No plots generated. Check inputs and problem overlap.")

    print(f"\nGenerated {plotted_groups} plot file(s) in: {output_dir}")
    print(f"Summary CSV: {summary_file}")


if __name__ == "__main__":
    main()
