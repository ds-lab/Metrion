"""Dummy input for testing plot_experiment function with random data."""

import random
from pathlib import Path

import polars as pl

from evaluation import plot_experiment

# Set seed for reproducibility (remove for truly random values)
random.seed(42)


def random_energy_df():
    """Generate a DataFrame with random energy and runtime values."""
    return pl.DataFrame(
        {
            "pkg_energy_active_j": [random.uniform(1000, 15000)],
            "dram_energy_active_j": [random.uniform(0, 5000)],
            "gpu_energy_active_j": [0.0],
            "duration_s": [random.uniform(30, 180)],
        }
    )


cpu_tool_results = {
    "Ground Truth": random_energy_df(),
    "Metrion": random_energy_df(),
    "EnergAt": random_energy_df(),
    "Scaphandre": random_energy_df(),
}

memory_tool_results = {
    "Ground Truth": random_energy_df(),
    "Metrion": random_energy_df(),
    "EnergAt": random_energy_df(),
    "Scaphandre": random_energy_df(),
}

combined_tool_results = {
    "Ground Truth": random_energy_df(),
    "Metrion": random_energy_df(),
    "EnergAt": random_energy_df(),
    "Scaphandre": random_energy_df(),
}

# Sequence input format: list of tuples (experiment_label, tool_results_mapping)
results_per_tool_sequence = [
    ("CPU", cpu_tool_results),
    ("Memory", memory_tool_results),
    ("Combined", combined_tool_results),
]

runtime_results_sequence = [
    ("CPU", cpu_tool_results),
    ("Memory", memory_tool_results),
    ("Combined", combined_tool_results),
]

if __name__ == "__main__":
    output_dir = Path(__file__).parent / "data" / "plots" / "test_output"

    # Test with all three workloads
    print("Generating plots with CPU, Memory, and Combined workload data...")
    figures = plot_experiment(
        results_per_tool=results_per_tool_sequence,
        runtime_results=runtime_results_sequence,
        experiment_name="Energy Measurement Comparison",
        output_dir=output_dir,
    )
    print(f"Generated figures: {list(figures.keys())}")

    print(f"\nDone! Check the output directory: {output_dir}")
