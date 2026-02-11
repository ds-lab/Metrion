#!/bin/bash

set -e
set -o pipefail

# Resolve project root once
PROJECT_ROOT="$(pwd)"   # assuming you run this from project root
DATA_DIR="$PROJECT_ROOT/evaluation/data"
PYTHON_ENV="$PROJECT_ROOT/.venv/bin/python3.12"
TARGET_UID=1016
TARGET_GID=1022

# General workload scripts
SLEEP_SCRIPT="$PROJECT_ROOT/evaluation/workloads/sleep_workload.sh"
CPU_WORKLOAD_SCRIPT="$PROJECT_ROOT/evaluation/workloads/cpu_workloads.sh"
MEMORY_WORKLOAD_SCRIPT="$PROJECT_ROOT/evaluation/workloads/memory_workload.sh"
MIXED_WORKLOAD_SCRIPT="$PROJECT_ROOT/evaluation/workloads/mixed_workload.sh"
MEASURE_ENERGY_OF_WORKLOAD="$PROJECT_ROOT/evaluation/measure_energy_of_workload.sh"

# Metrion
METRION_WRAPPER="$PROJECT_ROOT/evaluation/tools/scripts/metrion_wrapper.sh"

# Scaphandre
SCAPHANDRE_BIN="$PROJECT_ROOT/evaluation/tools/bin/scaphandre/scaphandre"
SCAPHANDRE_WRAPPER="$PROJECT_ROOT/evaluation/tools/scripts/scaphandre_wrapper.sh"

# EnergAt
ENERGAT_WRAPPER="$PROJECT_ROOT/evaluation/tools/scripts/energat_wrapper.sh"

# Data paths
LOG_PATH="$PROJECT_ROOT/evaluation/data/logs/benchmark_logs_$(date +%Y%m%d_%H%M%S).log"

# Energy files for runs
RAPL_IDLE_CSV="$DATA_DIR/measurements/rapl/results/measurements_idle.csv"
RAPL_WORKLOAD_CSV="$DATA_DIR/measurements/rapl/results/measurements_workload.csv"
SCAPHANDRE_TOTAL_ENERGY_CSV="$DATA_DIR/measurements/scaphandre/measurements_total_energy.csv"
ENERGAT_TOTAL_ENERGY_CSV="$DATA_DIR/measurements/energat/measurements_total_energy.csv"
METRION_TOTAL_ENERGY_CSV="$DATA_DIR/measurements/metrion/measurements_total_energy.csv"

MEASURE_FREEZE_OPTION="--freeze" # --no-freeze or --freeze

# Route all output through tee so it lands in both stdout and the log file

exec > >(tee -a "$LOG_PATH") 2>&1


# --- functions -------------------------------------------------------------------
_turbo_sysfs="/sys/devices/system/cpu/intel_pstate/no_turbo"
_governor_path="/sys/devices/system/cpu/cpu0/cpufreq/scaling_available_governors"

# --- TURBO ---
turbo_status() {
  if [ -f "$_turbo_sysfs" ]; then
    [[ $(cat "$_turbo_sysfs") -eq 0 ]] && echo "enabled" || echo "disabled"
  else
    echo "unsupported"
  fi
}

turbo_enable() {
  [ -f "$_turbo_sysfs" ] && echo 0 | sudo tee "$_turbo_sysfs" >/dev/null
  echo "Turbo enabled"
}

turbo_disable() {
  [ -f "$_turbo_sysfs" ] && echo 1 | sudo tee "$_turbo_sysfs" >/dev/null
  echo "Turbo disabled"
}

governor_set_all() {
  local gov="${1:-}"
  if [ -z "$gov" ]; then
    echo "Usage: governor_set_all <governor>" >&2
    return 1
  fi
  for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    [ -f "$f" ] && echo "$gov" | sudo tee "$f" >/dev/null
  done
  echo "Governor set to $gov"
}

# --- tool calls ---
run_idle_measurements() {
  echo "--- Running measure_energy_cpu_dram.sh IDLE ---"
  sudo systemd-run --unit="rapl-idle-measurement-$$" --wait --pipe "$MEASURE_ENERGY_OF_WORKLOAD" "idle" "$SLEEP_SCRIPT" "$RAPL_IDLE_CSV" "$MEASURE_FREEZE_OPTION"
  echo "Idle calculation finished. Output stored in $RAPL_IDLE_CSV"
  echo ""
}

run_workload_measurements_rapl() {
  local workload_type="${1:-cpu}"
  local workload_script="${2:-$CPU_WORKLOAD_SCRIPT}"

  echo "--- Running measure_energy_cpu_dram.sh cpu workload ($workload_type) ---"
  sudo systemd-run --unit="rapl-workload-measurement-$$" --wait --pipe \
    --setenv=LOCATION_DIR="${DATA_DIR}/measurements/rapl/results" \
    --setenv=NAME="rapl" \
    "$MEASURE_ENERGY_OF_WORKLOAD" "rapl-$workload_type" "$workload_script" "$RAPL_WORKLOAD_CSV" "$MEASURE_FREEZE_OPTION"
  echo "measure_energy_cpu_dram.sh finished. Output stored in $RAPL_WORKLOAD_CSV"
  echo ""
}

run_metrion_measurements() {
  local workload_type="${1:-cpu}"
  local workload_script="${2:-$CPU_WORKLOAD_SCRIPT}"

  echo "--- Running Metrion with workload ($workload_type) ---"
  echo "Workload script: $workload_script"

  # Use the wrapper as the workload for measure_energy.sh
  # Note: --no-freeze for outer script; FREEZE_DURING_WORKLOAD=1 freezes only during the actual workload
  sudo systemd-run --unit="Metrion-agent-$$" --wait --pipe \
    --setenv=ORCH_UNIT="Metrion-orchestrator-$$" \
    --setenv=PYTHON_ENV="$PYTHON_ENV" \
    --setenv=CPU_WORKLOAD_SCRIPT="$workload_script" \
    --setenv=INIT_WAIT_SEC=45 \
    --setenv=EXIT_WAIT_SEC=180 \
    --setenv=DATA_DIR="$DATA_DIR" \
    --setenv=PROJECT_ROOT="$PROJECT_ROOT" \
    --setenv=FREEZE_DURING_WORKLOAD=1 \
    "$MEASURE_ENERGY_OF_WORKLOAD" "metrion-$workload_type" \
    "$METRION_WRAPPER" "$METRION_TOTAL_ENERGY_CSV" --no-freeze
}

run_scaphandre_measurements() {
  local workload_type="${1:-cpu}"
  local workload_script="${2:-$CPU_WORKLOAD_SCRIPT}"

  echo "--- Running scaphandre wrapper with CPU workload ($workload_type) ---"
  echo "Workload script: $workload_script"
  echo "Scaphandre binary: $SCAPHANDRE_BIN"
  echo "Scaphandre wrapper: $SCAPHANDRE_WRAPPER"
  echo "Scaphandre total energy CSV: $SCAPHANDRE_TOTAL_ENERGY_CSV"

  sudo systemd-run --unit="scaphandre-measurement-$$" --wait --pipe \
    --setenv=SCAPHANDRE_UNIT=scaphandre-profiler \
    --setenv=SCAPHANDRE_BIN="$SCAPHANDRE_BIN" \
    --setenv=SCAPHANDRE_PROCESS_REGEX="stress-ng" \
    --setenv=SCAPHANDRE_SAMPLING_SEC=0 \
    --setenv=SCAPHANDRE_SAMPLING_NS=100000000 \
    --setenv=CPU_WORKLOAD_SCRIPT="$workload_script" \
    --setenv=INIT_WAIT_SEC=5 \
    --setenv=EXIT_WAIT_SEC=30 \
    --setenv=DATA_DIR="$DATA_DIR" \
    "$MEASURE_ENERGY_OF_WORKLOAD" "scaphandre-$workload_type" \
    "$SCAPHANDRE_WRAPPER" "$SCAPHANDRE_TOTAL_ENERGY_CSV" "$MEASURE_FREEZE_OPTION"
}

run_energat_measurements() {
  local workload_type="${1:-cpu}"
  local workload_script="${2:-$CPU_WORKLOAD_SCRIPT}"

  echo "--- Running energat wrapper with workload ($workload_type) ---"
  echo "Workload script: $workload_script"

  sudo systemd-run --unit="energat-measurement-$$" --wait --pipe \
    --setenv=ENERGAT_UNIT=energat-profiler \
    --setenv=CPU_WORKLOAD_SCRIPT="$workload_script" \
    --setenv=PROJECT_ROOT_DIR="$PROJECT_ROOT" \
    --setenv=DATA_DIR="$DATA_DIR" \
    --setenv=INIT_WAIT_SEC=5 \
    --setenv=EXIT_WAIT_SEC=30 \
    "$MEASURE_ENERGY_OF_WORKLOAD" "energat-$workload_type" \
    "$ENERGAT_WRAPPER" "$ENERGAT_TOTAL_ENERGY_CSV" "$MEASURE_FREEZE_OPTION"
}


# --- flow ------------------------------------------------------------------------
echo "===== $(date '+%Y-%m-%d %H:%M:%S') | starting benchmark run ====="

#governor_set_all "performance"
#turbo_disable

run_random_measurements() {
  workload_type="cpu"
  script="$CPU_WORKLOAD_SCRIPT"
  local funcs=(
    run_workload_measurements_rapl
    # run_energat_measurements
    run_idle_measurements
    # run_scaphandre_measurements
    run_metrion_measurements
  )
  mapfile -t shuffled < <(printf '%s\n' "${funcs[@]}" | shuf)
  local IFS=', '
  echo "Random order: ${shuffled[*]}"
  local last=$(( ${#shuffled[@]} - 1 ))
  for idx in "${!shuffled[@]}"; do
    "${shuffled[$idx]}" "$workload_type" "$script"
    if [ "$idx" -lt "$last" ]; then
      sleep 10
    fi
  done
}

for round in {1..1}; do
  echo "=== Round $round ==="
  run_random_measurements
  echo ""
done


governor_set_all "ondemand"
turbo_enable

echo "Check if any cgroup is frozen (it should not be): ALL should be 0"
grep -R . /sys/fs/cgroup/*/cgroup.freeze

sudo chown -R $TARGET_UID:$TARGET_GID $PROJECT_ROOT/evaluation/