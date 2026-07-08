#!/usr/bin/env bash
set -euo pipefail

# --- required env ----------------------------------------------------------------
: "${ENERGAT_UNIT:?ENERGAT_UNIT is required}"
: "${CPU_WORKLOAD_SCRIPT:?CPU_WORKLOAD_SCRIPT is required}"
: "${PROJECT_ROOT_DIR:?PROJECT_ROOT_DIR is required}"
: "${DATA_DIR:?DATA_DIR is required}"
: "${INIT_WAIT_SEC:?INIT_WAIT_SEC is required}"                         # seconds to wait after starting energat
: "${EXIT_WAIT_SEC:?EXIT_WAIT_SEC is required}"                         # seconds to wait after workload ends

ENERGAT_PYTHON="${ENERGAT_PYTHON:-${PROJECT_ROOT_DIR}/evaluation/tools/bin/energat/.venv/bin/python3.12}"
ENERGAT_MAIN="${ENERGAT_MAIN:-${PROJECT_ROOT_DIR}/evaluation/tools/bin/energat/energat/__main__.py}"
ENERGAT_BASELINE_FILE="${ENERGAT_BASELINE_FILE:-${PROJECT_ROOT_DIR}/evaluation/tools/bin/energat/data/baseline_power.json}"

WORKLOAD_NAME="${WORKLOAD_NAME:-stress-ng}"
REPORT_TAG="${REPORT_TAG:-${WORKLOAD_NAME}_$(date +%Y%m%d_%H%M%S)}"
ENERGAT_LOG_PATH="${ENERGAT_LOG_PATH:-${DATA_DIR}/measurements/energat/logs/energat_${REPORT_TAG}.log}"
ENERGAT_RESULTS_PATH="${ENERGAT_RESULTS_PATH:-${DATA_DIR}/measurements/energat/results/}"

WORKLOAD_PID=""

# --- functions -------------------------------------------------------------------
start_energat() {
  if [[ -z "${WORKLOAD_PID}" ]]; then
    echo "Workload PID missing; cannot launch energat." >&2
    return 1
  fi

  local energat_cmd="exec \"${ENERGAT_PYTHON}\" \"${ENERGAT_MAIN}\" --pid \"${WORKLOAD_PID}\" --output \"${ENERGAT_RESULTS_PATH}\" --basefile \"${ENERGAT_BASELINE_FILE}\" --name \"${REPORT_TAG}\""

  energat_cmd+=" >>\"${ENERGAT_LOG_PATH}\" 2>&1"
  echo "EnergAt command: ${energat_cmd}"

  echo "Starting energat unit ${ENERGAT_UNIT} for PID ${WORKLOAD_PID}..."
  sudo systemd-run \
    --unit="$ENERGAT_UNIT" \
    --property=KillMode=control-group \
    --property=KillSignal=SIGINT \
    --property=SendSIGHUP=no \
    --property=SendSIGKILL=yes \
    --property=TimeoutStopSec=10min \
    --same-dir \
    --collect \
    /bin/bash -lc "${energat_cmd}"

}

stop_energat() {
  if sudo systemctl is-active --quiet "${ENERGAT_UNIT}"; then
    echo "Stopping energat unit ${ENERGAT_UNIT}..."
    sudo systemctl stop "${ENERGAT_UNIT}"
  fi
}

trap stop_energat EXIT INT TERM


# --- flow ------------------------------------------------------------------------
echo "Starting CPU workload..."
INIT_WAIT_SEC="${INIT_WAIT_SEC}" LOCATION_DIR=${ENERGAT_RESULTS_PATH} NAME="energat" bash "${CPU_WORKLOAD_SCRIPT}" &
WORKLOAD_PID=$!
echo "Workload started with PID ${WORKLOAD_PID}"

start_energat

set +e 
wait "${WORKLOAD_PID}"
WORKLOAD_STATUS=$?
set -e
WORKLOAD_PID=""

echo "CPU workload finished with status ${WORKLOAD_STATUS}"
echo "Waiting ${EXIT_WAIT_SEC}s for energat flush..."
sleep "${EXIT_WAIT_SEC}"

stop_energat

exit "${WORKLOAD_STATUS}"