#!/usr/bin/env bash
# metrion_workload_wrapper.sh
# Run a workload under Metrion with no CLI args.
# Reads all configuration from environment variables.

set -euo pipefail

# --- Source shared libraries ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/cgroup_freeze.sh"

# --- required env ----------------------------------------------------------------
: "${ORCH_UNIT:?ORCH_UNIT is required}"                       # e.g. metrion-metrion
: "${PYTHON_ENV:?PYTHON_ENV is required}"                     # e.g. /path/to/venv/bin/python
: "${CPU_WORKLOAD_SCRIPT:?CPU_WORKLOAD_SCRIPT is required}"   # the real workload script to run
: "${INIT_WAIT_SEC:?INIT_WAIT_SEC is required}"               # seconds to wait after starting scaphandre
: "${EXIT_WAIT_SEC:?EXIT_WAIT_SEC is required}"               # seconds to wait after workload ends
: "${DATA_DIR:?DATA_DIR is required}"                         # e.g. evaluation/data
: "${PROJECT_ROOT:?PROJECT_ROOT is required}"                 # used for cleaning old logs

# Optional: freeze cgroups during workload (default: 0 = no freeze)
FREEZE_DURING_WORKLOAD="${FREEZE_DURING_WORKLOAD:-0}"

METRION_SCRIPT="${PROJECT_ROOT%/}/metrion/orchestrator.py"
METRION_LOG_PATH="${METRION_LOG_PATH:-${DATA_DIR}/measurements/metrion/logs/metrion_$(date +%Y%m%d_%H%M%S).log}"


# --- functions -------------------------------------------------------------------
start_metrion() {
  echo "Starting metrion as transient service: $ORCH_UNIT"
  local metrion_cmd
  metrion_cmd="exec \"$PYTHON_ENV\" \"$METRION_SCRIPT\""
  if [[ -n "${METRION_LOG_PATH:-}" ]]; then
    metrion_cmd+=" >>\"$METRION_LOG_PATH\" 2>&1"
  fi

  sudo systemd-run \
    --unit="$ORCH_UNIT" \
    --property=KillMode=process \
    --property=KillSignal=SIGINT \
    --property=SendSIGHUP=no \
    --property=SendSIGKILL=yes \
    --property=TimeoutStopSec=15min \
    --same-dir \
    --collect \
    /bin/bash -lc "$metrion_cmd"
}

stop_metrion() {
  if sudo systemctl is-active --quiet "${ORCH_UNIT}"; then
    echo "Requesting graceful shutdown (SIGINT) for ${ORCH_UNIT}…"
    sudo systemctl kill -s SIGINT "${ORCH_UNIT}"

    # Wait up to 600s (tweak as you like) for a clean exit
    for _ in $(seq 1 600); do
      if ! sudo systemctl is-active --quiet "${ORCH_UNIT}"; then
        echo "metrion stopped gracefully."
        return 0
      fi
      sleep 1
    done

    echo "Graceful shutdown timed out; stopping unit (may escalate)…"
    sudo systemctl stop "${ORCH_UNIT}"
  fi
}


cleanup_all() {
  cleanup_freeze
  stop_metrion
}

trap cleanup_all EXIT INT TERM

# --- flow ------------------------------------------------------------------------
start_metrion
echo "Waiting ${INIT_WAIT_SEC}s for metrion to initialize..."
sleep "$INIT_WAIT_SEC"

# Freeze cgroups only during the actual workload
if [[ "$FREEZE_DURING_WORKLOAD" -eq 1 ]]; then
  setup_freeze_units
  freeze_and_wait
fi

echo "Starting CPU workload..."
set +e
LOCATION_DIR="${DATA_DIR}/measurements/metrion/results" NAME="metrion" bash "$CPU_WORKLOAD_SCRIPT"
WORKLOAD_STATUS=$?
set -e

echo "CPU workload finished with status $WORKLOAD_STATUS"

# Thaw immediately after workload completes
if [[ "$FREEZE_DURING_WORKLOAD" -eq 1 && "$CGROUP_FROZEN" -eq 1 ]]; then
  thaw_units
fi
sleep "$EXIT_WAIT_SEC"   # let metrion process final data; the EXIT

stop_metrion

exit $WORKLOAD_STATUS