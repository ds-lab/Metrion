# METRION performance harness

Measurement scripts for the four optimisation tracks (database write cost,
Python overhead, live energy streaming, shutdown latency). They exist so that
every proposal in the optimisation plan can be *verified on your Linux box*
before anyone changes production code.

Nothing here imports `metrion` except the shutdown watchdog, so the benchmarks
run without root, without BCC and without a GPU.

## Setup

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv sync --all-extras
# the benchmarks additionally need nothing beyond what metrion already depends on
```

Run them from inside the `bench/` directory (they import `_common` by name):

```bash
cd bench
```

## 1. Database write path — `bench_db_write.py`

Reproduces `SQLDatabaseStorage._save_measurement_df` and compares it against the
proposed variants (no PRIMARY KEY, native timestamps instead of `strftime`,
tuned pragmas, Arrow-direct insert, parquet sink).

```bash
python bench_db_write.py --rows 2000000 --batch 250000 --repeats 3
# the interesting one: what the PK index costs once the table is large
python bench_db_write.py --rows 1000000 --prefill 20000000
```

Watch the `db size` line as well as the timings — the PRIMARY KEY ART index is
what makes the file grow superlinearly.

## 2. Entity upsert path — `bench_entity_upsert.py`

Reproduces `save_entities` for `logical_entity`, which is the write that runs on
every collection cycle and grows with thread churn.

```bash
python bench_entity_upsert.py --entities 20000 --cycles 10 --repeats 3
```

Variant `1b` isolates plain SQLAlchemy/duckdb-engine `executemany` from the
`ON CONFLICT` cost, so you can tell how much of the gap is the ORM layer and how
much is the upsert semantics.

## 3. Pipeline Python overhead — `bench_python_overhead.py`

Six independent sections (`--only abcdef` to select):

| section | what it isolates |
|---|---|
| A | `util._generate_id` — one snowflake id per row, in a Python loop |
| B | the three `.unique()` passes over every raw ring-buffer batch |
| C | the six-metric fan-out and the IPC/DB volume it creates |
| D | `util.build_cgroup_id_map` — full `os.walk` of `/sys/fs/cgroup` per call (Linux only) |
| E | the per-row registry loops (`iter_rows` + per-entity lookup) |
| F | the temp-file IPC round trip between the producer and the saver process |

```bash
python bench_python_overhead.py --rows 500000 --repeats 3
python bench_python_overhead.py --only d        # cgroup walk, on the real machine
```

Section D is the one that must be run on the target host: its cost is entirely a
function of how many cgroups that machine has.

## 4. Ring-buffer consumption — `bench_ringbuf_callback.py`

Lower bound for the per-event Python callback in
`EbpfRingBufferProcess._handle_ring_event`, plus the ring-buffer bandwidth
arithmetic for the current 328-byte `sched_event_v2`.

```bash
python bench_ringbuf_callback.py --events 500000 --rate 50000
```

To get the *real* number rather than the lower bound, profile the running
monitor instead:

```bash
# how many context switches the machine actually produces
perf stat -e sched:sched_switch -a sleep 10

# where the ring-buffer process spends its time (needs py-spy)
sudo py-spy top --pid $(pgrep -f SchedSwitchBpfMonitor) --native

# drops reported by METRION itself
sudo metrion monitor 2>&1 | grep -i "missed events"
```

## 5. Shutdown — `shutdown_watchdog.py`

Runs the monitor, sends `SIGINT`, and dumps the stacks of every thread in every
Python process of the tree if it has not exited within the deadline.

```bash
sudo $(which python) shutdown_watchdog.py \
    --cmd "$(which metrion) monitor" \
    --run-seconds 60 --deadline 20 --dump-dir ./shutdown-dumps
```

The frame at the top of each dumped stack is the thing that is hanging. It also
reports orphaned `batch_*.ipc` files left in `/tmp`, which are measurements that
never reached the database.

## 6. BCC against libbpf CO-RE — `validate_bcc_vs_core.py`

Loads the BCC probes of an older commit (default `a75b6e2`, the last one with BCC,
taken from git) next to their CO-RE ports on the same tracepoints, runs a known
workload and compares the events field by field. Needs root, BCC and the built
extension; run it again after changing a `.bpf.c` file or on a new kernel.

```bash
sudo .venv/bin/python bench/validate_bcc_vs_core.py registry
sudo .venv/bin/python bench/validate_bcc_vs_core.py sched_switch
sudo .venv/bin/python bench/validate_bcc_vs_core.py tcp
# what two programs on one tracepoint differ by anyway: CO-RE against CO-RE
sudo .venv/bin/python bench/validate_bcc_vs_core.py sched_switch --control
```

Timestamps each program takes itself differ by some microseconds, and the perf
counter deltas of `sched_switch` by about 0.3 % in both modes; everything else
must match. The exit code is 1 if any run differs.

## Recording a baseline

Before changing anything, capture a baseline so later runs are comparable:

```bash
for s in bench_db_write bench_entity_upsert bench_python_overhead bench_ringbuf_callback; do
    python $s.py 2>&1 | tee "baseline_$s.txt"
done
```
