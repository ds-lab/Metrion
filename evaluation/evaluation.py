# %%
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl
from tools.scripts import parser


# %%
def summarize_active_energy_by_tool(tool_results: Mapping[str, pl.DataFrame]) -> pl.DataFrame:
    """Aggregate active energy metrics per tool.

    Parameters
    ----------
    tool_results : Mapping[str, pl.DataFrame]
        Mapping of tool identifiers to parsed result frames. Each frame is expected
        to contain energy columns that follow the :data:`results_schema_df` schema.

    Returns
    -------
    pl.DataFrame
        Frame with columns ``tool`` and the mean active energy metrics for each
        tool. ``Ground Truth`` is always the left-most entry when present.
    """
    summary_df_schema = {
        "tool": pl.Utf8,
        "pkg_energy_active_j": pl.Float64,
        "dram_energy_active_j": pl.Float64,
        "gpu_energy_active_j": pl.Float64,
    }

    summary_df = pl.DataFrame(schema=summary_df_schema)

    for tool, df in tool_results.items():
        if df.is_empty():
            continue

        mean_values = df.select(
            pl.col("pkg_energy_active_j").mean().alias("pkg_energy_active_j"),
            pl.col("dram_energy_active_j").mean().alias("dram_energy_active_j"),
            pl.col("gpu_energy_active_j").mean().alias("gpu_energy_active_j"),
        )

        row = pl.DataFrame(
            {
                "tool": [tool],
                "pkg_energy_active_j": mean_values["pkg_energy_active_j"],
                "dram_energy_active_j": mean_values["dram_energy_active_j"],
                "gpu_energy_active_j": mean_values["gpu_energy_active_j"],
            }
        )

        summary_df = pl.concat([summary_df, row], how="vertical")

    return summary_df


def calculate_accuracy(
    tool_results: Mapping[str, pl.DataFrame],
) -> pl.DataFrame:
    """Calculate accuracy metrics for each tool compared to ground truth.

    Parameters
    ----------
    tool_results : Mapping[str, pl.DataFrame]
        Mapping of tool identifiers to parsed result frames. Each frame is expected
        to contain energy columns that follow the :data:`results_schema_df` schema.
        The ground truth tool must be included in the mapping with the key
        ``"Ground Truth"``.

    Returns
    -------
    pl.DataFrame
        Frame with columns ``tool``, ``pkg_energy_active_j_mae``,
        ``dram_energy_active_j_mae``, ``gpu_energy_active_j_mae``,
        ``pkg_energy_active_j_mape``, ``dram_energy_active_j_mape``,
        ``gpu_energy_active_j_mape`` containing the mean absolute error (MAE)
        and mean absolute percentage error (MAPE) for each tool compared to the
        ground truth.

    Raises
    ------
    ValueError
        Raised when the ground truth data is missing or when no other tools are
        available for comparison.
    """
    if "Ground Truth" not in tool_results:
        raise ValueError("Ground truth data is required for accuracy calculation.")

    ground_truth_df = tool_results["Ground Truth"]

    if ground_truth_df.is_empty():
        raise ValueError("Ground truth data is empty.")

    accuracy_df_schema = {
        "tool": pl.Utf8,
        "pkg_energy_active_j_mae": pl.Float64,
        "dram_energy_active_j_mae": pl.Float64,
        "gpu_energy_active_j_mae": pl.Float64,
        "pkg_energy_active_j_mape": pl.Float64,
        "dram_energy_active_j_mape": pl.Float64,
        "gpu_energy_active_j_mape": pl.Float64,
    }

    accuracy_df = pl.DataFrame(schema=accuracy_df_schema)

    for tool, df in tool_results.items():
        if tool == "Ground Truth" or df.is_empty():
            continue

        # calculate absolute errors and percentage errors for each energy column
        mae_errors = df.select(
            (pl.col("pkg_energy_active_j").mean() - ground_truth_df["pkg_energy_active_j"].mean())
            .abs()
            .alias("pkg_energy_active_j_mae"),
            (pl.col("dram_energy_active_j").mean() - ground_truth_df["dram_energy_active_j"].mean())
            .abs()
            .alias("dram_energy_active_j_mae"),
            (pl.col("gpu_energy_active_j").mean() - ground_truth_df["gpu_energy_active_j"].mean())
            .abs()
            .alias("gpu_energy_active_j_mae"),
        )

        # avoid division by zero by adding a tiny epsilon to denominators
        mape_errors = df.select(
            (
                (pl.col("pkg_energy_active_j").mean() - ground_truth_df["pkg_energy_active_j"].mean())
                / (ground_truth_df["pkg_energy_active_j"].mean())
            )
            .abs()
            .alias("pkg_energy_active_j_mape"),
            (
                (pl.col("dram_energy_active_j").mean() - ground_truth_df["dram_energy_active_j"].mean())
                / (ground_truth_df["dram_energy_active_j"].mean())
            )
            .abs()
            .alias("dram_energy_active_j_mape"),
            (
                (pl.col("gpu_energy_active_j").mean() - ground_truth_df["gpu_energy_active_j"].mean())
                / (ground_truth_df["gpu_energy_active_j"].mean())
            )
            .abs()
            .alias("gpu_energy_active_j_mape"),
        )

        # aggregate per-tool mean values (one row per tool) to match the docstring
        pkg_mae = mae_errors["pkg_energy_active_j_mae"].mean()
        dram_mae = mae_errors["dram_energy_active_j_mae"].mean()
        gpu_mae = mae_errors["gpu_energy_active_j_mae"].mean()

        pkg_mape = mape_errors["pkg_energy_active_j_mape"].mean()
        dram_mape = mape_errors["dram_energy_active_j_mape"].mean()
        gpu_mape = mape_errors["gpu_energy_active_j_mape"].mean()

        new_row = pl.DataFrame(
            {
                "tool": [tool],
                "pkg_energy_active_j_mae": [pkg_mae],
                "dram_energy_active_j_mae": [dram_mae],
                "gpu_energy_active_j_mae": [gpu_mae],
                "pkg_energy_active_j_mape": [pkg_mape],
                "dram_energy_active_j_mape": [dram_mape],
                "gpu_energy_active_j_mape": [gpu_mape],
            }
        )

        accuracy_df = accuracy_df.vstack(new_row)

    if accuracy_df.is_empty():
        raise ValueError("No valid accuracy data available.")

    return accuracy_df


def get_results_paths(measurement_path: Path, workload_type: str) -> dict[str, Path]:
    """Get paths to measurement results for each tool.

    Parameters
    ----------
    measurement_path : Path
        Base path containing subdirectories for each tool's measurement results.

    Returns
    -------
    dict[str, Path]
        Mapping from tool identifiers to their corresponding result paths.
    """
    return {
        "Ground Truth": measurement_path / "rapl" / "results",
        "EnergAt": measurement_path / "energat" / "results" / f"{workload_type}_workload",
        "Metrion": measurement_path / "metrion" / "results" / f"{workload_type}_workload",
        "Scaphandre": measurement_path / "scaphandre" / "results" / f"{workload_type}_workload",
    }


def evaluation_pipeline(measurement_path: Path, output_path: Path, workload_type: str):
    results_paths = get_results_paths(measurement_path, workload_type)

    ground_truth_df = parser.parse_rapl_logs_workload(results_paths["Ground Truth"], workload_type)

    energat_df = parser.parse_energat_logs(results_paths["EnergAt"])
    metrion_df = parser.parse_metrion_logs(
        results_paths["Metrion"],
        "stress-ng",
        datetime(2025, 1, 1, 0, 0, 0, microsecond=0, tzinfo=UTC),
        datetime(2027, 1, 1, 0, 0, 0, microsecond=0, tzinfo=UTC),
        dialect="duckdb",
    )
    scaphandre_df = parser.parse_scaphandre_logs(results_paths["Scaphandre"], "stress-ng", step_s=0.1)

    tool_results = {
        "Ground Truth": ground_truth_df,
        "EnergAt": energat_df,
        "Metrion": metrion_df,
        "Scaphandre": scaphandre_df,
    }

    accuracy_df = calculate_accuracy(tool_results)

    return accuracy_df, tool_results


def plot_experiment(
    results_per_tool: Sequence[tuple[Any, ...]] | Mapping[str, pl.DataFrame],
    runtime_results: Sequence[tuple[Any, ...]] | Mapping[str, pl.DataFrame],
    experiment_name: str,
    output_dir: Path,
    figsize: tuple[int, int] = (10, 6),
):
    """Render grouped bar plots for package energy, DRAM energy, and runtime.

    Parameters
    ----------
    results_per_tool : Sequence[tuple[str, object, Mapping[str, pl.DataFrame]]]
        Sequence of tuples describing energy results per experiment. Each tuple
        is expected to contain exactly three elements where the first item is
        the experiment label and the last item is a mapping of tool identifiers
        to their corresponding result frames.
    runtime_results : Sequence[tuple[str, object, Mapping[str, pl.DataFrame]]]
        Sequence of tuples describing runtime results per experiment using the
        same convention as ``results_per_tool``.
    experiment_name : str
        High-level experiment descriptor used for chart titles and exported file
        names.
    output_dir : Path
        Directory where the generated figures will be stored.
    figsize : tuple[int, int], optional
        Figure size applied to each individual matplotlib figure.

    Returns
    -------
    dict[str, tuple["Figure", "Axes"]]
        Mapping from metric key (``"pkg_energy_active_j"``,
        ``"dram_energy_active_j"``, ``"runtime"``) to the matplotlib figure and
        axes used for plotting.
    """

    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    output_dir.mkdir(parents=True, exist_ok=True)

    def _slugify(text: str) -> str:
        slug_chars: list[str] = []
        last_was_sep = False
        for ch in text.lower():
            if ch.isalnum():
                slug_chars.append(ch)
                last_was_sep = False
            else:
                if not last_was_sep:
                    slug_chars.append("_")
                    last_was_sep = True
        slug = "".join(slug_chars).strip("_")
        return slug or "experiment"

    def _unpack(
        entries: Sequence[tuple[Any, ...]] | Mapping[str, pl.DataFrame],
        default_label: str,
    ) -> list[tuple[str, Mapping[str, pl.DataFrame]]]:
        unpacked: list[tuple[str, Mapping[str, pl.DataFrame]]] = []
        if isinstance(entries, Mapping):
            unpacked.append((default_label, entries))
            return unpacked
        for entry in entries:
            if not isinstance(entry, tuple):
                raise TypeError("Each entry must be a tuple")
            experiment_label_candidate = entry[0] if entry else None
            data_candidate = entry[-1] if entry else None
            if len(entry) >= 3 and isinstance(data_candidate, Mapping):
                experiment_label = experiment_label_candidate
                data_mapping = data_candidate
            elif len(entry) == 2 and isinstance(entry[1], Mapping):
                experiment_label, data_mapping = entry
            else:
                raise ValueError("Each entry must contain two or three elements")
            if not isinstance(experiment_label, str):
                raise TypeError("Experiment label must be a string")
            if not isinstance(data_mapping, Mapping):
                raise TypeError("Experiment data must be a mapping of tool to DataFrame")
            unpacked.append((experiment_label, data_mapping))
        return unpacked

    energy_entries = _unpack(results_per_tool, experiment_name)
    runtime_entries = _unpack(runtime_results, experiment_name)

    experiment_order: list[str] = []
    tool_order: list[str] = []

    energy_values: dict[str, dict[str, dict[str, float]]] = {}
    for experiment_label, tool_frames in energy_entries:
        if experiment_label not in experiment_order:
            experiment_order.append(experiment_label)
        summary_df = summarize_active_energy_by_tool(tool_frames)
        energy_by_tool: dict[str, dict[str, float]] = {}
        for row in summary_df.iter_rows(named=True):
            tool_name = str(row["tool"])
            if tool_name not in tool_order:
                tool_order.append(tool_name)
            energy_by_tool[tool_name] = {
                "pkg_energy_active_j": float(row.get("pkg_energy_active_j", 0.0) or 0.0),
                "dram_energy_active_j": float(row.get("dram_energy_active_j", 0.0) or 0.0),
            }
        energy_values[experiment_label] = energy_by_tool

    runtime_values: dict[str, dict[str, float]] = {}
    for experiment_label, tool_frames in runtime_entries:
        if experiment_label not in experiment_order:
            experiment_order.append(experiment_label)
        runtime_by_tool: dict[str, float] = {}
        for tool_name, df in tool_frames.items():
            if tool_name not in tool_order:
                tool_order.append(tool_name)
            if df is None or df.is_empty():
                runtime_by_tool[tool_name] = 0.0
                continue
            duration_series = df["duration_s"].cast(pl.Float64, strict=False)
            duration_mean_raw = duration_series.mean()
            if isinstance(duration_mean_raw, int | float):
                runtime_by_tool[tool_name] = float(duration_mean_raw)
            else:
                runtime_by_tool[tool_name] = 0.0
        runtime_values[experiment_label] = runtime_by_tool

    if not experiment_order:
        raise ValueError("No experiment data provided for plotting.")
    if not tool_order:
        raise ValueError("No tools detected in the provided experiment data.")

    slug_base = _slugify(experiment_name)

    preferred_sequence = [
        "Ground Truth",
        "Metrion",
        "EnergAt",
        "Scaphandre",
    ]
    preferred_set = set(preferred_sequence)
    ordered_tools: list[str] = [tool for tool in preferred_sequence if tool in tool_order]
    ordered_tools.extend([tool for tool in tool_order if tool not in preferred_set])
    tool_order = ordered_tools

    cmap = plt.get_cmap("Pastel1")
    if len(tool_order) <= 1:
        color_map = {tool_order[0]: cmap(0.0)} if tool_order else {}
    else:
        color_map = {tool: cmap(idx) for idx, tool in enumerate(tool_order)}

    metric_specs = [
        ("pkg_energy_active_j", "Package Energy (J)", "energy"),
        ("dram_energy_active_j", "DRAM Energy (J)", "energy"),
        ("runtime", "Runtime (s)", "runtime"),
    ]

    figures: dict[str, tuple[Figure, Axes]] = {}

    x_positions = np.arange(len(experiment_order), dtype=float)
    n_tools = max(len(tool_order), 1)
    group_width = 0.78
    bar_width = group_width / n_tools

    base_positions = np.linspace(-(n_tools - 1) / 2, (n_tools - 1) / 2, n_tools) * bar_width
    if n_tools > 2:
        gap = bar_width * 0.35
        for idx in range(2, n_tools):
            base_positions[idx] += gap
        base_positions -= base_positions.mean()

    offset_lookup = {tool: base_positions[idx] for idx, tool in enumerate(tool_order)}

    for metric_key, metric_label, source in metric_specs:
        fig, ax = plt.subplots(figsize=figsize, layout="constrained")

        for _tool_idx, tool_name in enumerate(tool_order):
            offset = offset_lookup[tool_name]
            if source == "energy":
                heights = [
                    energy_values.get(exp, {}).get(tool_name, {}).get(metric_key, 0.0) for exp in experiment_order
                ]
            else:
                heights = [runtime_values.get(exp, {}).get(tool_name, 0.0) for exp in experiment_order]

            bars = ax.bar(
                x_positions + offset,
                heights,
                bar_width,
                label=tool_name,
                color=color_map[tool_name],
                edgecolor="black",
                linewidth=0.6,
                alpha=0.9,
                zorder=3,
            )

            ax.bar_label(
                bars,
                labels=[f"{val:.1f}" for val in heights],
                padding=3,
                fontsize=14,
                rotation=90,
            )

        ax.set_xticks(x_positions)
        ax.set_xticklabels(experiment_order, rotation=0, fontsize=14)
        ax.set_ylabel(metric_label, fontsize=16)
        ax.set_xlabel("Workload Type", fontsize=16)
        ax.tick_params(axis="y", labelsize=14)
        ax.set_axisbelow(True)

        ax.legend(
            loc="upper left", ncols=min(len(tool_order), 2), frameon=False, title="Tool", fontsize=14, title_fontsize=14
        )

        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

        filename = output_dir / f"{slug_base}_{metric_key}.png"
        fig.savefig(filename, dpi=300)

        figures[metric_key] = (fig, ax)

    return figures


def runtime_evaluation(measurement_path: Path, output_path: Path, workload_type: str):
    results_paths = get_results_paths(measurement_path, workload_type)

    ground_truth_runtime_df = parser.parse_runtime_logs(
        results_paths["Ground Truth"] / f"rapl-{workload_type}-workload.csv"
    )

    energat_runtime_df = parser.parse_runtime_logs(results_paths["EnergAt"] / f"energat-{workload_type}-workload.csv")
    metrion_runtime_df = parser.parse_runtime_logs(results_paths["Metrion"] / f"metrion-{workload_type}-workload.csv")
    scaphandre_runtime_df = parser.parse_runtime_logs(
        results_paths["Scaphandre"] / f"scaphandre-{workload_type}-workload.csv"
    )

    tool_runtime_results = {
        "Ground Truth": ground_truth_runtime_df,
        "EnergAt": energat_runtime_df,
        "Metrion": metrion_runtime_df,
        "Scaphandre": scaphandre_runtime_df,
    }

    return tool_runtime_results


if __name__ == "__main__":
    measurement_path = Path(__file__).parent / "data" / "measurements"
    plot_path = Path(__file__).parent / "data" / "plots"

    # cpu workload
    plot_path_cpu = plot_path / "cpu_workload"
    cpu_accuracy_df, cpu_tool_results = evaluation_pipeline(measurement_path, plot_path_cpu, "cpu")

    cpu_runtime_results = runtime_evaluation(measurement_path, plot_path_cpu, "cpu")

    # # memory workload
    # plot_path_mem = plot_path / "memory_workload"
    # mem_accuracy_df, mem_tool_results = evaluation_pipeline(measurement_path, plot_path_mem, "memory")
    # mem_runtime_results = runtime_evaluation(measurement_path, plot_path_mem, "memory")

    # # mixed workload
    # plot_path_mixed = plot_path / "mixed_workload"
    # mixed_accuracy_df, mixed_tool_results = evaluation_pipeline(measurement_path, plot_path_mixed, "mixed")
    # mixed_runtime_results = runtime_evaluation(measurement_path, plot_path_mixed, "mixed")

    # plot_experiment(
    #     [("cpu", cpu_tool_results), ("memory", mem_tool_results), ("mixed", mixed_tool_results)],
    #     [("cpu", cpu_runtime_results), ("memory", mem_runtime_results), ("mixed", mixed_runtime_results)],
    #     "CPU Workload Experiment",
    #     plot_path_cpu,
    # )

    plot_experiment(
        [
            ("cpu", cpu_tool_results),
        ],
        [("cpu", cpu_runtime_results)],
        "CPU Workload Experiment",
        plot_path_cpu,
    )


# %%
