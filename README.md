# METRION

METRION measures how much energy each process, thread and cgroup on a Linux machine uses. It
reads the hardware energy counters of CPU, DRAM and GPU, follows every context switch with eBPF,
and splits the measured energy among the threads that caused it. The attribution model is
described in the [METRION paper](https://arxiv.org/abs/2512.06806).

<!-- Demo video: in GitHub's web editor, drag an MP4 (at most 10 MB) onto this line. -->

## Requirements

| | |
|---|---|
| CPU | **Intel**, for CPU and DRAM energy (RAPL). Hybrid P/E-core CPUs are supported. |
| GPU | **NVIDIA** with a working driver, optional. |
| Kernel | Linux with BTF, i.e. `/sys/kernel/btf/vmlinux` exists (Ubuntu 20.10+, Debian 11+, RHEL 8.2+). |
| System | x86-64, glibc 2.28 or newer, systemd with cgroup v2. |
| Python | 3.12 |
| Optional | `perf` for the running kernel (Ubuntu: `linux-tools-$(uname -r)`) and `strace`, only for the memory counter metrics `CPU_CORE_PID_MEM_*`. They are off by default and not available on hybrid CPUs. |

Measuring needs root: eBPF, perf counters and RAPL are not open to normal users. Analyzing a run
does not.

## Installation

Download the wheel from the [releases page](../../releases) and install it as a tool, which puts
`metrion` on your PATH in an environment of its own. [uv](https://docs.astral.sh/uv/) fetches
Python 3.12 if the machine lacks it:

```bash
uv tool install ./metrion-0.2.0-cp312-cp312-manylinux_2_28_x86_64.whl
```

Or with pip, in a Python 3.12 environment:

```bash
pip install ./metrion-0.2.0-cp312-cp312-manylinux_2_28_x86_64.whl
```

The wheel brings everything it needs. BCC, kernel headers or a compiler are not required.

## Quick start

```bash
mkdir my-measurement && cd my-measurement
metrion init                              # writes ./metrion.yaml
sudo $(which metrion) calibrate           # measures the idle power into ./calibration.json
sudo $(which metrion) monitor             # measure until Ctrl+C, or add --timeout <minutes>
metrion analyze monitor_<timestamp>.db    # which processes used the energy
```

`sudo` searches only its own PATH, so name `metrion` by its full path; `metrion init` prints it
for your installation. Every command works in the current directory: `monitor` reads
`./metrion.yaml` and `./calibration.json` and writes its data next to them, stamped with the
start time, so no run overwrites another.

## What to measure

`metrion analyze` selects what it reports with at most one filter. Processes always count
together with all their children, so the energy of a shell script includes every program it
starts.

| You want the energy of | Filter | One row per |
|---|---|---|
| everything on the machine | *(none)* | process, the 20 largest |
| a program by name | `--cmdline train.py` | process |
| processes you know the PIDs of | `--pids 1234,5678` | process |
| a Docker container | `--cgroup $(docker inspect -f '{{.Id}}' web)` | cgroup |
| all Docker containers | `--cgroup docker-` | cgroup |
| a Kubernetes pod | `--cgroup pod$(kubectl get pod web -o jsonpath='{.metadata.uid}' \| tr - _)` | cgroup |
| a systemd service | `--cgroup nginx.service` | cgroup |

`--cmdline` and `--cgroup` match any part of the command line (or process name) and of the
cgroup path. The pod example is for the systemd cgroup driver, which writes the pod UID with
underscores; with the cgroupfs driver, leave out `| tr - _`. Further options:

- `--start` and `--end` narrow the analysis to a time range, in UTC (`"2026-09-30 08:18:35"`).
- `--unit Wh` shows watt-hours instead of joules.
- `--csv DIR` also writes the whole table (`table.csv`) and every selected row per time window
  and thread (`rows.csv`), in joules.

`metrion <command> --help` lists all options.

### After the run or live

Both ways give the same numbers; they differ in when and in what they keep.

**After the run** (the default). `monitor` stores the raw measurements, and `analyze` attributes
them when you ask. You can analyze a run as often as you like, with any filter.

```bash
metrion analyze monitor_2026_09_30_08_18_28.db --cmdline train.py
```

**Live.** With `live_attribution.enabled: true`, `monitor` attributes the energy while it runs and
writes one row per time window and thread to `energy_<timestamp>/`, about 40 s behind real time.
Set `storage.store_raw_measurements: false` as well to keep only this result, which is smaller by
orders of magnitude, for long runs. `analyze` reads the directory like a database:

```bash
metrion analyze energy_2026_09_30_08_18_28 --cgroup docker-
```

A live result only knows the threads that used energy. A parent process that only waits, such as
a shell running a script, is not in it, so `--pids <that shell>` does not find its children
there. Select the children by name instead, or analyze after the run.

## Reading the report

A busy loop started from a shell, selected by the shell's PID:

```text
╭─ Metrion energy report ────────────────────────────────────────────────────────────────────╮
│ Run        2026-09-30 08:18:28 → 2026-09-30 08:18:59 UTC  (31 s)                           │
│ Selection  pids 2071864  (3 processes, 3 threads)                                          │
│ Active     2026-09-30 08:18:35 → 2026-09-30 08:18:47 UTC  (when the selection used energy) │
╰────────────────────────────────────────────────────────────────────────────────────────────╯

Energy by component
╭───────────┬──────────┬─────────┬───────────┬────────────╮
│ component │ idle [J] │ dyn [J] │ total [J] │ of the run │
├───────────┼──────────┼─────────┼───────────┼────────────┤
│ CPU       │     15.2 │   144.3 │     159.5 │      69.7% │
├───────────┼──────────┼─────────┼───────────┼────────────┤
│ total     │     15.2 │   144.3 │     159.5 │      69.7% │
╰───────────┴──────────┴─────────┴───────────┴────────────╯
idle  the selection's share of what the machine draws doing nothing
dyn   what its work added on top

Energy by process, largest first

  pid       ppid      name      threads   cgroup               cpu idle [J]   cpu dyn [J]   total [J]
 ─────────────────────────────────────────────────────────────────────────────────────────────────────
  2072093   2071864   python3   1         session-4022.scope           15.2         144.3       159.5
  2071864   2071862   bash      1         session-4022.scope           <0.1          <0.1        <0.1
  2071865   2071864   sleep     1         session-4022.scope           <0.1          <0.1        <0.1
 ─────────────────────────────────────────────────────────────────────────────────────────────────────
  sum                                                                  15.2         144.3       159.5
```

- **idle** is the selection's share of the power the machine draws when nothing runs, as
  measured by `calibrate`. **dyn** is what its work added on top.
- **of the run** is the selection's share of all energy measured during the run.
- Components without energy (here DRAM, as this CPU has no DRAM counter, and the GPU) are left
  out. Columns that do not fit the terminal are dropped; `--csv` always has all of them.

## Calibration

The idle power is the part of the energy that is drawn whether anything runs or not. METRION
splits it among the running processes, so it has to know how large it is. `calibrate` measures
it once per machine:

```bash
sudo $(which metrion) calibrate
```

It lists what it will freeze and asks before it starts. For about two minutes it freezes the
logins and the desktop (`user.slice`, including the terminal it runs in), Kubernetes pods and
container runtimes, lets the machine settle, and reads the energy counters. Afterwards
everything continues where it stopped; if the measurement is killed, systemd thaws the units
anyway.

- `--skip PATTERN` leaves matching units running, `--freeze PATTERN` freezes more,
  e.g. `--freeze machine.slice` for virtual machines, which are left running by default.
- `--yes` skips the question, for scripts.
- Processes that hold a GPU are listed, not stopped. The GPU idle power includes whatever they
  keep running.

The result goes to `./calibration.json`, next to `metrion.yaml`. For a measurement in another
directory, calibrate there, or set `calibration` in its `metrion.yaml` to the existing file:

```yaml
calibration: /home/me/metrion/calibration.json
```

Calibrate again after changing the hardware. `monitor` refuses to start if the calibration does
not know a device it measures, or was measured on another machine. To skip calibration, give
the idle power in the configuration (see below).

## Configuration

`metrion init` writes the default configuration to `./metrion.yaml`, with a comment on every
setting. `monitor --config <file>` uses another file. Relative paths in it start in the directory
`metrion` runs in. Unknown keys are rejected, so a typo stops `monitor` at startup instead of
being ignored.

| Section | What it controls |
|---|---|
| `calibration` | The file with the idle power, see [Calibration](#calibration). |
| `storage` | Where the data goes: `duckdb` (one file) or `parquet` (a directory, cheapest during the run), and whether raw measurements are kept. |
| `energy_model.components` | Which components are measured (`cpu`, `dram`, `gpu`, `network`), with which metrics and idle and maximum power. |
| `live_attribution` | Attribution during the run, see [above](#after-the-run-or-live). |
| `timing` | How often sources are read and data is written; the defaults suit most runs. |
| `ebpf.ringbuf_mib` | Buffer for context switches. Raise it if the log warns about lost events. |

To leave out a component, e.g. on a machine without an NVIDIA GPU:

```yaml
energy_model:
  components:
    gpu:
      mode: exclude
```

To give the power yourself instead of calibrating, list it per device (CPU and DRAM: RAPL
package number; GPU: index), in watts:

```yaml
energy_model:
  components:
    cpu:
      mode: include
      power:
        - id: 0
          idle_power: 12.5
          max_power: 65
```

## Troubleshooting

- **`sudo: metrion: command not found`**: use the full path, `sudo $(which metrion) ...`.
- **"No calibration at ..."**: run `sudo $(which metrion) calibrate` in this directory, or
  point `calibration` in `metrion.yaml` to an existing one.
- **The log warns about a ring buffer overflow**: raise `ebpf.ringbuf_mib`.
- **Analyzing a Parquet run fails with a permission error**: the first `analyze` of a Parquet
  run writes a database into its directory, which belongs to root. Run it once with `sudo`.
- **`monitor` does not stop on Ctrl+C**: `sudo pkill -9 -f "metrion monitor"`.

## Development

Building from source needs a current Rust toolchain, `clang`, `libelf-dev` and
`zlib1g-dev` (Ubuntu names), and [uv](https://docs.astral.sh/uv/):

```bash
git clone <repository-url> && cd metrion
uv sync --all-extras     # Python dependencies, including the development tools
uv run maturin develop   # compiles the Rust extension and the eBPF programs into the package
uv run pre-commit install
uv run pytest tests/unit
```

`sudo scripts/build_wheel.sh` builds the release wheel in a manylinux container (Docker) and
writes it to `dist/`.

| Directory | Content |
|---|---|
| `metrion/` | The Python package: CLI, orchestrator, discovery, data sources, storage, attribution. |
| `src/` | The Rust extension, which loads the eBPF programs. |
| `evaluation/` | The experiments of the paper, comparing METRION with Scaphandre and EnergAt. |
| `bench/` | Benchmarks and checks of the agent's internals. |
| `tests/` | Unit and integration tests. |

### Evaluation

`sudo evaluation/run_evaluation.sh`, from the repository root, calibrates the machine and
reproduces the experiments of the paper. Each profiler runs in its own transient systemd service;
logs go to `evaluation/data/logs/`, measurements to `evaluation/data/measurements/<tool>/`.

## Citation

If you use METRION in academic work, please cite our paper
([arXiv:2512.06806](https://arxiv.org/abs/2512.06806)):

```bibtex
@misc{weigell2025metrion,
  author        = {Weigell, Benjamin and Hornung, Simon and Bauer, Bernhard},
  title         = {{METRION}: A Framework for Accurate Software Energy Measurement},
  year          = {2025},
  eprint        = {2512.06806},
  archivePrefix = {arXiv},
  primaryClass  = {cs.SE},
  doi           = {10.48550/arXiv.2512.06806},
  note          = {Accepted at the 10th IEEE/ACM International Workshop on Green and Sustainable Software (GREENS'26), ICSE 2026}
}
```

## License

Released under the terms in [LICENSE](LICENSE).
