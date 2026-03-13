# METRION

METRION is a research-grade framework for attributing energy consumption. This repository provides the core framework, the energy attribution model described in the METRION paper, and the accompanying data sources and system scanning components for Intel CPUs on Linux systems.

## Why METRION?
- Implements the paper’s energy attribution model, accounting for frequency scaling, simultaneous multithreading, multi-socket topologies, and NUMA effects.
- Ships with reusable system discovery and data source modules that translate platform-specific counters into the platform-independent data model.
- Provides a platform-independent data model and modular architecture so new data sources, storages, or host operating systems can be added without touching the attribution logic.

## Repository Layout
- `configs/` default collection settings consumed by the orchestrator.
- `metrion/` Python package with the orchestrator, discovery, monitoring, storage, and energy attribution modules.
- `src/` Rust crates compiled into the Python package via `maturin`.
- `evaluation/` evaluation automation to determine the accuracy of METRION and compare against other tools (Scaphandre, EnergAt).
- `tests/` unit and integration tests for the Python and Rust components.
- `target/`, `wheels/` build artifacts produced by Cargo and maturin (ignored by version control in normal workflows).

## Prerequisites
- A Linux system, e.g., Ubuntu 22.04 or newer with systemd (the evaluation scripts rely on `systemd-run`).
- Python 3.12 and optionally a package manager like [uv](https://docs.astral.sh/uv/getting-started/installation/).
- Rust toolchain (2024 edition) with `cargo` available in `PATH`.
- [BCC (BPF Compiler Collection)](https://github.com/iovisor/bcc/blob/master/INSTALL.md) 0.35 or newer for the eBPF monitors.
- Root privileges to access eBPF, perf, and RAPL counters (`sudo` is required for most workflows).
- Linux Headers, for Ubuntu: `sudo apt-get update && sudo apt-get install linux-headers-$(uname -r)`

## Usage Setup
1. Clone the project
	```bash
	git clone <repository-url>
	````
2. Install it via pip: 
	```bash
	pip install -e .
	```

## Development Setup with uv
1. Clone the project and create a Python 3.12 virtual environment with uv:
	```bash
	git clone <repository-url>
	cd metrion
	uv venv --python 3.12 --system-site-packages
	source .venv/bin/activate
	```
2. Install the Python dependencies (core, extras, and notebooks used in the evaluation):
	```bash
	uv sync --all-extras
	```
3. Build and install the Rust extensions into the environment:
	```bash
	maturin develop
	```
4. Enable developer tooling:
	```bash
	pre-commit install
	```

## Running the Framework
1. Review `configs/config.yaml` and adjust metric allowlists, idle power hints, or storage settings as required.
2. Launch the orchestrator with elevated privileges using the CLI. Since `sudo` typically resets the environment, locate the executable first:
	```bash
	# Identify where metrion is installed
	METRION_BIN=$(which metrion)

	# Run indefinitely (stop with Ctrl+C)
	sudo "$METRION_BIN" monitor

	# Run for a specific duration (e.g., 5 minutes)
	sudo "$METRION_BIN" monitor --timeout 5

	# Use a custom config file
	sudo "$METRION_BIN" monitor --config /path/to/config.yaml

	# Combine options
	sudo "$METRION_BIN" monitor -t 10 -c /path/to/config.yaml
	```
	Alternatively, run the orchestrator directly via Python (adjust paths as needed):
	```bash
	PYTHON_BIN=$(which python)
	sudo "$PYTHON_BIN" metrion/orchestrator.py monitor
	sudo "$PYTHON_BIN" metrion/orchestrator.py monitor --timeout 5
	```
3. METRION writes measurements to a timestamped SQLite database in the project root (e.g., `monitor_2025_12_04_08_18_42.db`) and streams logs to stdout. Use `Ctrl+C` to stop the process gracefully.

### CLI Options
| Option | Short | Description |
|--------|-------|-------------|
| `--config` | `-c` | Path to the configuration file (default: `configs/config.yaml`) |
| `--timeout` | `-t` | Timeout in minutes. If not specified, runs indefinitely until Ctrl+C |
| `--help` | | Show help message and exit |

## Energy Attribution Workflow
The analysis CLI reconstructs CPU, DRAM, and GPU energy per process from a monitoring session.

```bash
METRION_BIN=$(which metrion)
sudo "$METRION_BIN" analyze \
  --db-path monitor_2025_12_04_08_18_42.db \
  --filter-cmdline "python" \
  --start-time "2025-01-01 00:00:00" \
  --end-time "2028-01-01 00:00:00"
```

Alternatively, you can run the CLI directly:
```bash
PYTHON_BIN=$(which python)
sudo "$PYTHON_BIN" metrion/energy_attribution/energy_attribution_cli.py \
  --db-path monitor_2025_12_04_08_18_42.db \
  --filter-cmdline "python" \
  --start-time "2025-01-01 00:00:00" \
  --end-time "2026-01-01 00:00:00"
```

### Analysis CLI Options
| Option | Description |
|--------|-------------|
| `--db-path` | Path to the database file (required) |
| `--filter-cmdline` | Filter processes by command line substring |
| `--filter-pids` | Filter by comma-separated PIDs (e.g. `1234,5678`) |
| `--start-time` | Analysis window start (`YYYY-MM-DD HH:MM:SS`) |
| `--end-time` | Analysis window end (`YYYY-MM-DD HH:MM:SS`) |
| `--unit` | Energy unit: `J` (default) or `Wh` |
| `--backend` | Plotting backend: `plotly` (default) or `matplotlib` |
| `--no-plots` | Skip plot and HTML report generation (summary still printed) |
| `--export-summary` | Export per-process summary to CSV and JSON |
| `--export-raw-data` | Export raw filtered interval data to CSV per device (`raw_cpu_*.csv`, `raw_dram_*.csv`, `raw_gpu_*.csv`) |
| `--force` | Force recreation of materialised tables when queries change |
| `--type` | Database type: `duckdb` (default) or `sqlite` |

- The script expects timestamps in UTC as stored in the database.
- Generated HTML reports and exported files are written to `metrion/energy_attribution/output/` by default.

## Evaluation
The evaluation workflow reproduces the experiments from the paper and compares METRION with Scaphandre and EnergAt.

```bash
sudo evaluation/run_evaluation.sh
```

- Run the script from the repository root; it resolves workload scripts relative to the current directory.
- Each profiler run is executed inside its own transient systemd service to isolate cgroups and captures logs in `evaluation/data/logs/`.
- Energy measurements are saved in `evaluation/data/measurements/<tool>/` corresponding to the tool that produced them.

## Tips and Tricks 
Delete pycache files:
```bash
find . \( -type d -name "__pycache__" -o -type f \( -name "*.pyc" -o -name "*.pyo" \) \) -exec rm -rf {} +
```


## Citation
This software is currently under development at the University of Augsburg. If you build on it in academic work, please cite the forthcoming publication:

```bibtex
@article{weigell2025,
  author  = {Weigell, Benjamin and Hornung, Simon and Bauer, Bernhard},
  title   = {METRION: A Framework for Accurate Software Energy Measurement},
  journal = {TBD},
  year    = {2025}
}
```

## License
This project is released under the terms of the license contained in `LICENSE`.
