#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/weigeben/metrion-internal}"
RESULTS_ROOT="${RESULTS_ROOT:-$PROJECT_ROOT/results_sadra}"
RUNNER_SCRIPT="${RUNNER_SCRIPT:-$PROJECT_ROOT/run_effibench_energy.py}"
CONFIG_PATH="${CONFIG_PATH:-$PROJECT_ROOT/configs/config.yaml}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"

# Requested run mode defaults.
RUNS="${RUNS:-1000}"
WARMUP="${WARMUP:-20}"
ENTRY_TIMEOUT="${ENTRY_TIMEOUT:-120}"
FILE_GLOB="${FILE_GLOB:-EffiBench*.json}"
DB_PATTERN="${DB_PATTERN:-monitor*.db}"
MAX_PARALLEL_ENTRIES="${MAX_PARALLEL_ENTRIES:-4}"

# Artifact handling.
ARTIFACTS_ROOT="${ARTIFACTS_ROOT:-$RESULTS_ROOT/energy_batch_artifacts_$(date -u +%Y%m%d_%H%M%S)}"
COPY_DB="${COPY_DB:-1}"
MOVE_DB="${MOVE_DB:-0}"
DRY_RUN="${DRY_RUN:-0}"
RESUME_ARTIFACTS="${RESUME_ARTIFACTS:-0}"

fmt_seconds() {
  local total="$1"
  if [[ -z "$total" || "$total" -lt 0 ]]; then
    total=0
  fi
  local h=$((total / 3600))
  local m=$(((total % 3600) / 60))
  local s=$((total % 60))
  printf "%02d:%02d:%02d" "$h" "$m" "$s"
}

cleanup() {
  if [[ -n "${SUDO_KEEPALIVE_PID:-}" ]]; then
    kill "$SUDO_KEEPALIVE_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

if [[ ! -x "$PYTHON_BIN" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  else
    echo "[error] Python executable not found. Set PYTHON_BIN or create $PROJECT_ROOT/.venv/bin/python"
    exit 1
  fi
fi

if [[ ! -f "$RUNNER_SCRIPT" ]]; then
  echo "[error] Missing runner script: $RUNNER_SCRIPT"
  exit 1
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "[error] Missing config file: $CONFIG_PATH"
  exit 1
fi

if [[ ! -d "$RESULTS_ROOT" ]]; then
  echo "[error] Results root does not exist: $RESULTS_ROOT"
  exit 1
fi

if ! [[ "$RUNS" =~ ^[0-9]+$ ]] || ((RUNS < 2)); then
  echo "[error] RUNS must be an integer >= 2 for --same-pid-runs (current: $RUNS)"
  exit 1
fi

if ((COPY_DB == 1 && MOVE_DB == 1)); then
  echo "[error] COPY_DB and MOVE_DB cannot both be 1"
  exit 1
fi

mkdir -p "$ARTIFACTS_ROOT"
SUMMARY_TSV="$ARTIFACTS_ROOT/run_summary.tsv"
START_INDEX=0

if ((RESUME_ARTIFACTS == 1)); then
  if [[ -f "$SUMMARY_TSV" ]]; then
    START_INDEX="$(awk -F '\t' 'NR>1 && $1 ~ /^[0-9]+$/ {if($1>m)m=$1} END{print m+0}' "$SUMMARY_TSV")"
  fi

  DIR_INDEX_MAX="$(find "$ARTIFACTS_ROOT" -maxdepth 1 -mindepth 1 -type d -printf '%f\n' | sed -n 's/^\([0-9][0-9]*\)_.*/\1/p' | sort -n | tail -n1)"
  DIR_INDEX_MAX="${DIR_INDEX_MAX:-0}"
  if ((10#$DIR_INDEX_MAX > START_INDEX)); then
    START_INDEX=$((10#$DIR_INDEX_MAX))
  fi

  if [[ ! -f "$SUMMARY_TSV" || ! -s "$SUMMARY_TSV" ]]; then
    printf "index\tinput_json\tstatus\texit_code\trun_seconds\tdb_artifacts\tartifact_dir\n" >"$SUMMARY_TSV"
  fi

  echo "[resume] Enabled. Existing index base: $START_INDEX"
else
  printf "index\tinput_json\tstatus\texit_code\trun_seconds\tdb_artifacts\tartifact_dir\n" >"$SUMMARY_TSV"
fi

# Refresh sudo credentials once and keep them alive for this long batch.
if command -v sudo >/dev/null 2>&1; then
  echo "[auth] Refreshing sudo credentials..."
  sudo -v
  (
    while true; do
      sudo -n true
      sleep 45
      kill -0 "$$" >/dev/null 2>&1 || exit 0
    done
  ) >/dev/null 2>&1 &
  SUDO_KEEPALIVE_PID="$!"
fi

echo "[scan] Searching for input files under: $RESULTS_ROOT"
mapfile -d '' INPUT_FILES < <(
  find "$RESULTS_ROOT" -type f -name "$FILE_GLOB" \
    ! -name "*_energy_results.json" \
    ! -name "*_energy_results.pids.json" \
    -print0 | sort -z
)

TOTAL="${#INPUT_FILES[@]}"
if ((TOTAL == 0)); then
  echo "[error] No files found matching $FILE_GLOB under $RESULTS_ROOT"
  exit 1
fi

echo "[scan] Found $TOTAL input file(s)."
echo "[scan] Artifacts root: $ARTIFACTS_ROOT"

action="copy"
if ((MOVE_DB == 1)); then
  action="move"
fi

BATCH_START_EPOCH="$(date +%s)"
OK_COUNT=0
FAIL_COUNT=0

for idx in "${!INPUT_FILES[@]}"; do
  INPUT_FILE="${INPUT_FILES[$idx]}"
  STEP=$((idx + 1))
  RUN_INDEX=$((START_INDEX + STEP))
  REL_PATH="${INPUT_FILE#"$RESULTS_ROOT"/}"
  STEM="$(basename "${INPUT_FILE%.json}")"
  SAFE_REL="$(echo "$REL_PATH" | tr '/[:space:]' '__' | tr -cd '[:alnum:]_.-')"
  RUN_STAMP="$(date -u +%Y%m%d_%H%M%S)"
  RUN_KEY="$(printf "%03d_%s_%s" "$RUN_INDEX" "$SAFE_REL" "$RUN_STAMP")"
  ARTIFACT_DIR="$ARTIFACTS_ROOT/$RUN_KEY"
  RUN_LOG_DIR="$ARTIFACT_DIR/logs"

  mkdir -p "$RUN_LOG_DIR"
  DB_MARKER="$ARTIFACT_DIR/.db_marker"
  touch "$DB_MARKER"

  RUN_START_EPOCH="$(date +%s)"
  echo
  echo "[progress] [$STEP/$TOTAL] START  $REL_PATH"

  CMD=(
    "$PYTHON_BIN" "$RUNNER_SCRIPT"
    "--input-files" "$INPUT_FILE"
    "--config" "$CONFIG_PATH"
    "--same-pid-runs"
    "--runs" "$RUNS"
    "--warmup" "$WARMUP"
    "--entry-timeout" "$ENTRY_TIMEOUT"
    "--log-dir" "$RUN_LOG_DIR"
    "--max-parallel-entries" "$MAX_PARALLEL_ENTRIES"
  )

  printf "[command]"
  for token in "${CMD[@]}"; do
    printf " %q" "$token"
  done
  printf "\n"

  EXIT_CODE=0
  if ((DRY_RUN == 1)); then
    echo "[dry-run] Command skipped."
  else
    if "${CMD[@]}"; then
      EXIT_CODE=0
    else
      EXIT_CODE=$?
    fi
  fi

  # Rename log files so names include the input stem (directory is already per-input).
  LOG_RENAMED=0
  shopt -s nullglob
  for LOG_FILE in "$RUN_LOG_DIR"/*.log; do
    LOG_BASE="$(basename "$LOG_FILE")"
    TARGET_LOG="$RUN_LOG_DIR/${STEM}__${LOG_BASE}"
    if [[ "$LOG_FILE" != "$TARGET_LOG" ]]; then
      mv -f "$LOG_FILE" "$TARGET_LOG"
    fi
    LOG_RENAMED=$((LOG_RENAMED + 1))
  done
  shopt -u nullglob

  # Copy or move DB files created/modified by this run to this input's artifact folder.
  DB_COUNT=0
  DB_NAMES=()
  mapfile -d '' NEW_DBS < <(
    find "$PROJECT_ROOT" -maxdepth 1 -type f -name "$DB_PATTERN" -newer "$DB_MARKER" -print0 | sort -z
  )
  for DB_FILE in "${NEW_DBS[@]}"; do
    DB_BASE="$(basename "$DB_FILE")"
    DB_NAMES+=("$DB_BASE")
    TARGET_DB="$ARTIFACT_DIR/${STEM}__${DB_BASE}"
    if ((MOVE_DB == 1)); then
      mv -f "$DB_FILE" "$TARGET_DB"
    elif ((COPY_DB == 1)); then
      cp -f "$DB_FILE" "$TARGET_DB"
    fi
    DB_COUNT=$((DB_COUNT + 1))
    if ((COPY_DB == 1 || MOVE_DB == 1)); then
      echo "[db] ${action}d $(basename "$DB_FILE") -> $TARGET_DB"
    fi
  done
  if ((DB_COUNT == 0)); then
    echo "[db] No DB files matching '$DB_PATTERN' detected for this run."
  fi

  {
    printf "input_json=%s\n" "$INPUT_FILE"
    printf "db_pattern=%s\n" "$DB_PATTERN"
    if ((DB_COUNT == 0)); then
      printf "db_name=<none>\n"
    else
      for DB_NAME in "${DB_NAMES[@]}"; do
        printf "db_name=%s\n" "$DB_NAME"
      done
    fi
  } >"$DB_MARKER"

  # Also snapshot generated result files to the artifact dir for easier navigation.
  RESULT_JSON="${INPUT_FILE%.json}_energy_results.json"
  RESULT_PIDS="${INPUT_FILE%.json}_energy_results.pids.json"
  if [[ -f "$RESULT_JSON" ]]; then
    cp -f "$RESULT_JSON" "$ARTIFACT_DIR/"
  fi
  if [[ -f "$RESULT_PIDS" ]]; then
    cp -f "$RESULT_PIDS" "$ARTIFACT_DIR/"
  fi

  RUN_END_EPOCH="$(date +%s)"
  RUN_SECONDS=$((RUN_END_EPOCH - RUN_START_EPOCH))
  BATCH_SECONDS=$((RUN_END_EPOCH - BATCH_START_EPOCH))
  AVG_SECONDS=$((BATCH_SECONDS / STEP))
  REMAINING=$((AVG_SECONDS * (TOTAL - STEP)))

  STATUS="OK"
  if ((EXIT_CODE == 0)); then
    OK_COUNT=$((OK_COUNT + 1))
  else
    STATUS="FAIL"
    FAIL_COUNT=$((FAIL_COUNT + 1))
  fi

  printf "%d\t%s\t%s\t%d\t%d\t%d\t%s\n" \
    "$RUN_INDEX" "$INPUT_FILE" "$STATUS" "$EXIT_CODE" "$RUN_SECONDS" "$DB_COUNT" "$ARTIFACT_DIR" >>"$SUMMARY_TSV"

  echo "[progress] [$STEP/$TOTAL] $STATUS   $REL_PATH"
  echo "[timing]   run=$(fmt_seconds "$RUN_SECONDS") elapsed=$(fmt_seconds "$BATCH_SECONDS") eta~=$(fmt_seconds "$REMAINING")"
  echo "[artifacts] $ARTIFACT_DIR (logs renamed: $LOG_RENAMED, db captured: $DB_COUNT)"
done

echo
echo "[done] Total files : $TOTAL"
echo "[done] Successful  : $OK_COUNT"
echo "[done] Failed      : $FAIL_COUNT"
echo "[done] Summary TSV : $SUMMARY_TSV"
echo "[done] Artifacts   : $ARTIFACTS_ROOT"

if ((FAIL_COUNT > 0)); then
  exit 1
fi
