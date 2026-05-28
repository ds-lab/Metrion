#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/weigeben/metrion-internal}"
ARTIFACTS_ROOT="${ARTIFACTS_ROOT:-$PROJECT_ROOT/results_sadra/energy_batch_artifacts_20260414_210505}"
RUNNER_SCRIPT="${RUNNER_SCRIPT:-$PROJECT_ROOT/run_effibench_energy.py}"
CONFIG_PATH="${CONFIG_PATH:-$PROJECT_ROOT/configs/config.yaml}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"

# This artifacts batch was produced with --same-pid-runs.
SAME_PID_RUNS="${SAME_PID_RUNS:-1}"
DRY_RUN="${DRY_RUN:-0}"
STOP_ON_ERROR="${STOP_ON_ERROR:-0}"

resolve_db_for_dir() {
  local dir="$1"
  local marker="$dir/.db_marker"

  if [[ -f "$marker" ]]; then
    local db_name
    db_name="$(awk -F '=' '$1=="db_name" && $2!="<none>" {print $2; exit}' "$marker" || true)"
    if [[ -n "$db_name" ]]; then
      shopt -s nullglob
      local named_candidates=("$dir"/*__"$db_name" "$dir/$db_name")
      shopt -u nullglob
      if (( ${#named_candidates[@]} > 0 )); then
        printf "%s\n" "${named_candidates[0]}"
        return 0
      fi
    fi
  fi

  shopt -s nullglob
  local db_candidates=("$dir"/*.db)
  shopt -u nullglob
  if (( ${#db_candidates[@]} == 0 )); then
    return 1
  fi

  local newest=""
  local newest_mtime=0
  local db
  for db in "${db_candidates[@]}"; do
    local mtime
    mtime="$(stat -c %Y "$db" 2>/dev/null || echo 0)"
    if (( mtime > newest_mtime )); then
      newest_mtime="$mtime"
      newest="$db"
    fi
  done

  [[ -n "$newest" ]] || return 1
  printf "%s\n" "$newest"
}

if [[ ! -d "$ARTIFACTS_ROOT" ]]; then
  echo "[error] Artifacts root does not exist: $ARTIFACTS_ROOT"
  exit 1
fi

if [[ ! -f "$RUNNER_SCRIPT" ]]; then
  echo "[error] Missing runner script: $RUNNER_SCRIPT"
  exit 1
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "[error] Missing config file: $CONFIG_PATH"
  exit 1
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  else
    echo "[error] Python executable not found. Set PYTHON_BIN or create $PROJECT_ROOT/.venv/bin/python"
    exit 1
  fi
fi

echo "[scan] Searching for *_energy_results.pids.json under: $ARTIFACTS_ROOT"
mapfile -d '' PIDS_FILES < <(find "$ARTIFACTS_ROOT" -type f -name "*_energy_results.pids.json" -print0 | sort -z)

TOTAL="${#PIDS_FILES[@]}"
if (( TOTAL == 0 )); then
  echo "[error] No pids files found in $ARTIFACTS_ROOT"
  exit 1
fi

echo "[scan] Found $TOTAL artifact record files to re-analyze."

OK_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0
MISSING_WORK_COUNT=0

STEP=0
for PIDS_FILE in "${PIDS_FILES[@]}"; do
  STEP=$((STEP + 1))

  ARTIFACT_DIR="$(dirname "$PIDS_FILE")"
  STEM="$(basename "$PIDS_FILE" "_energy_results.pids.json")"
  INPUT_HINT="$ARTIFACT_DIR/$STEM.json"
  OUT_FILE="$ARTIFACT_DIR/${STEM}_energy_results.json"

  DB_PATH="$(resolve_db_for_dir "$ARTIFACT_DIR" || true)"
  if [[ -z "$DB_PATH" ]]; then
    echo "[skip] [$STEP/$TOTAL] No .db file found in $ARTIFACT_DIR"
    SKIP_COUNT=$((SKIP_COUNT + 1))
    continue
  fi

  CMD=(
    "$PYTHON_BIN" "$RUNNER_SCRIPT"
    "--analyze-only"
    "--input-files" "$INPUT_HINT"
    "--db-path" "$DB_PATH"
    "--config" "$CONFIG_PATH"
  )

  if (( SAME_PID_RUNS == 1 )); then
    CMD+=("--same-pid-runs")
  fi

  echo
  echo "[run] [$STEP/$TOTAL] artifact=$ARTIFACT_DIR"
  echo "[run] db=$DB_PATH"
  printf "[cmd]"
  for token in "${CMD[@]}"; do
    printf " %q" "$token"
  done
  printf "\n"

  if (( DRY_RUN == 1 )); then
    echo "[dry-run] skipped"
    OK_COUNT=$((OK_COUNT + 1))
    continue
  fi

  if "${CMD[@]}"; then
    if [[ -f "$OUT_FILE" ]] && grep -q '"cpu_core_pid_work"' "$OUT_FILE"; then
      echo "[ok] cpu_core_pid_work present in $OUT_FILE"
    else
      echo "[warn] Re-analysis completed but cpu_core_pid_work not found in $OUT_FILE"
      MISSING_WORK_COUNT=$((MISSING_WORK_COUNT + 1))
    fi
    OK_COUNT=$((OK_COUNT + 1))
  else
    echo "[fail] analyze-only command failed for $ARTIFACT_DIR"
    FAIL_COUNT=$((FAIL_COUNT + 1))
    if (( STOP_ON_ERROR == 1 )); then
      echo "[fail] STOP_ON_ERROR=1, exiting early"
      break
    fi
  fi
done

echo
echo "[done] Total artifacts      : $TOTAL"
echo "[done] Successful analyzes  : $OK_COUNT"
echo "[done] Failed analyzes      : $FAIL_COUNT"
echo "[done] Skipped (missing DB) : $SKIP_COUNT"
echo "[done] Missing work metric  : $MISSING_WORK_COUNT"

if (( FAIL_COUNT > 0 )); then
  exit 1
fi

if (( MISSING_WORK_COUNT > 0 )); then
  exit 2
fi
