# ...existing code...
#!/usr/bin/env bash
# Run a workload under scaphandre profiling, with no CLI args.
# Reads all configuration from environment variables.

set -euo pipefail

# --- required env ----------------------------------------------------------------
: "${SCAPHANDRE_UNIT:?SCAPHANDRE_UNIT is required}"                     # e.g. scaphandre-profiler
: "${SCAPHANDRE_BIN:?SCAPHANDRE_BIN is required}"                       # e.g. /usr/local/bin/scaphandre
: "${SCAPHANDRE_PROCESS_REGEX:?SCAPHANDRE_PROCESS_REGEX is required}"   # e.g. evaluation.py
: "${SCAPHANDRE_SAMPLING_SEC:?SCAPHANDRE_SAMPLING_SEC is required}"     # -s value (seconds between samples)
: "${SCAPHANDRE_SAMPLING_NS:?SCAPHANDRE_SAMPLING_NS is required}"       # -n value (nanoseconds between samples)
: "${CPU_WORKLOAD_SCRIPT:?CPU_WORKLOAD_SCRIPT is required}"             # the real workload script to run
: "${INIT_WAIT_SEC:?INIT_WAIT_SEC is required}"                         # seconds to wait after starting scaphandre
: "${EXIT_WAIT_SEC:?EXIT_WAIT_SEC is required}"                         # seconds to wait after workload ends
: "${DATA_DIR:?DATA_DIR is required}"                                   # e.g. evaluation/data

SCAPHANDRE_REPORT_DIR="${SCAPHANDRE_REPORT_DIR:-${DATA_DIR}/measurements/scaphandre/results}"
SCAPHANDRE_REPORT_TAG="${SCAPHANDRE_REPORT_TAG:-$(date +%Y%m%d_%H%M%S)}"
SCAPHANDRE_REPORT_PATH="${SCAPHANDRE_REPORT_DIR}/energy_report_${SCAPHANDRE_REPORT_TAG}.json"

SCAPHANDRE_LOG_PATH="${SCAPHANDRE_LOG_PATH:-${DATA_DIR}/measurements/scaphandre/logs/scaphandre_${SCAPHANDRE_REPORT_TAG}.log}"


# --- functions -------------------------------------------------------------------
start_scaphandre() {
  echo "Starting scaphandre as transient service: $SCAPHANDRE_UNIT"
  local scaphandre_cmd
  scaphandre_cmd="exec \"$SCAPHANDRE_BIN\" json -s \"$SCAPHANDRE_SAMPLING_SEC\" --step-nano \"$SCAPHANDRE_SAMPLING_NS\" -f \"$SCAPHANDRE_REPORT_PATH\" --process-regex \"$SCAPHANDRE_PROCESS_REGEX\""
  echo "Scaphandre command: $scaphandre_cmd"
  if [[ -n "${SCAPHANDRE_LOG_PATH:-}" ]]; then
    scaphandre_cmd+=" >>\"$SCAPHANDRE_LOG_PATH\" 2>&1"
  fi

  sudo systemd-run \
    --unit="$SCAPHANDRE_UNIT" \
    --property=KillMode=control-group \
    --property=KillSignal=SIGINT \
    --property=SendSIGHUP=no \
    --property=SendSIGKILL=yes \
    --property=TimeoutStopSec=10min \
    --same-dir \
    --collect \
    /bin/bash -lc "$scaphandre_cmd"
}

stop_scaphandre() {
  if sudo systemctl is-active --quiet "${SCAPHANDRE_UNIT}"; then
    echo "Stopping scaphandre unit ${SCAPHANDRE_UNIT}..."
    sudo systemctl stop "${SCAPHANDRE_UNIT}"
  fi
}
trap stop_scaphandre EXIT INT TERM

# --- flow ------------------------------------------------------------------------
start_scaphandre
echo "Waiting ${INIT_WAIT_SEC}s for scaphandre to initialize..."
sleep "$INIT_WAIT_SEC"

echo "Starting CPU workload..."
set +e
LOCATION_DIR="${DATA_DIR}/measurements/scaphandre/results" NAME="scaphandre" bash "$CPU_WORKLOAD_SCRIPT"
WORKLOAD_STATUS=$?
set -e

echo "CPU workload finished with status $WORKLOAD_STATUS"
sleep "$EXIT_WAIT_SEC"   # let scaphandre process final data

stop_scaphandre

exit "$WORKLOAD_STATUS"