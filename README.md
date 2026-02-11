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
- `target/`, `wheels/` build artefacts produced by Cargo and maturin (ignored by version control in normal workflows).

## Prerequisites
- Ubuntu 22.04 or newer with systemd (the evaluation scripts rely on `systemd-run`).
- Python 3.12 and the [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager.
- Rust toolchain (2024 edition) with `cargo` available in `PATH`.
- BCC (BPF Compiler Collection) 0.35 or newer for the eBPF monitors.
- Root privileges to access eBPF, perf, and RAPL counters (`sudo` is required for most workflows).
- sudo apt-get update && sudo apt-get install linux-headers-$(uname -r)

## Environment Setup
1. Clone the project and create a Python 3.12 virtual environment with uv:
	```bash
	git clone <repository-url>
	cd metrion
	uv venv --python 3.12
	source .venv/bin/activate
	```
2. Install the Python dependencies (core, extras, and notebooks used in the evaluation):
	```bash
	uv sync --all-extras
	```
3. Expose the system-installed BCC package to the virtual environment (adjust the path if your distribution stores BCC elsewhere):
	```bash
	echo "$(ls /usr/lib/python3/dist-packages/bcc-*.egg | head -1)" > .venv/lib/python3.12/site-packages/bcc-egg.pth
	```
4. Build and install the Rust extensions into the environment:
	```bash
	maturin develop
	```
5. (Optional) Enable developer tooling:
	```bash
	pre-commit install
	```

## Running the Framework
1. Review `configs/config.yaml` and adjust metric allowlists, idle power hints, or storage settings as required.
2. Launch the orchestrator with elevated privileges using the CLI:
	```bash
	# Run indefinitely (stop with Ctrl+C)
	sudo .venv/bin/metrion

	# Run for a specific duration (e.g., 5 minutes)
	sudo .venv/bin/metrion --timeout 5

	# Use a custom config file
	sudo .venv/bin/metrion --config /path/to/config.yaml

	# Combine options
	sudo .venv/bin/metrion -t 10 -c /path/to/config.yaml
	```
	Alternatively, run the orchestrator directly via Python:
	```bash
	sudo .venv/bin/python3.12 metrion/orchestrator.py
	sudo .venv/bin/python3.12 metrion/orchestrator.py --timeout 5
	```
3. METRION writes measurements to a timestamped SQLite database in the project root (for example `monitor_2025_12_04_08_18_42.db`) and streams logs to stdout. Use `Ctrl+C` to stop the process gracefully.

### CLI Options
| Option | Short | Description |
|--------|-------|-------------|
| `--config` | `-c` | Path to the configuration file (default: `configs/config.yaml`) |
| `--timeout` | `-t` | Timeout in minutes. If not specified, runs indefinitely until Ctrl+C |
| `--help` | | Show help message and exit |

## Energy Attribution Workflow
The analysis CLI in `metrion/energy_attribution/energy_attribution_cli.py` reconstructs CPU and DRAM per process from a run.

```bash
.venv/bin/python3.12 metrion/energy_attribution/energy_attribution_cli.py \
  --db-path monitor_2025_12_04_08_18_42.db \
  --filter-cmdline "python" \
  --start-time "2025-01-01 00:00:00" \
  --end-time "2026-01-01 00:00:00" \
  --type "duckdb"
```

- The script expects timestamps in UTC as stored in the database.
- Set `--force true` to rebuild materialized views when the schema or queries change.
- Generated HTML reports are written to `metrion/energy_attribution/output/` by default.

## Benchmark Evaluation
The evaluation workflow reproduces the experiments from the paper and compares METRION with Scaphandre and EnergAt.

```bash
sudo evaluation/run_evaluation.sh
```

- Run the script from the repository root; it resolves workload scripts relative to the current directory.
- Each profiler run is executed inside its own transient systemd service to isolate cgroups and captures logs in `evaluation/data/logs/`.
- Energy measurements land in `evaluation/data/measurements/<tool>/` according to the wrapper that produced them.

## Tipps and Tricks 
Delete pycache files. 
find . \( -type d -name "__pycache__" -o -type f \( -name "*.pyc" -o -name "*.pyo" \) \) -exec rm -rf {} +


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
