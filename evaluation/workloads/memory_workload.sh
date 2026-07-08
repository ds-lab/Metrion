#!/usr/bin/env bash

# optional startup delay to align with orchestrator scheduling
INIT_WAIT_SEC="${INIT_WAIT_SEC:-0}"

sleep "$INIT_WAIT_SEC"

# resolve required environment variables (allow hyphenated names via printenv fallback)
LOCATION_DIR_VALUE="${LOCATION_DIR:-$(printenv 'location-dir' 2>/dev/null || true)}"
NAME_VALUE="${NAME:-$(printenv 'name' 2>/dev/null || true)}"
INIT_WAIT_SEC="${INIT_WAIT_SEC:-0}"

if [[ -z "$LOCATION_DIR_VALUE" ]]; then
	echo "ERROR: location-dir environment variable is required." >&2
	exit 1
fi

if [[ -z "$NAME_VALUE" ]]; then
	echo "ERROR: name environment variable is required." >&2
	exit 1
fi

mkdir -p "$LOCATION_DIR_VALUE"

CSV_FILE="${LOCATION_DIR_VALUE%/}/${NAME_VALUE}-memory-workload.csv"

if [[ ! -f "$CSV_FILE" ]]; then
	echo "name,start,stop,execution_time" >"$CSV_FILE"
fi

# array of stream operation targets for stress-ng to execute
STREAM_OPS=(200000)

ISO_TIMESTAMP_FORMAT='+%Y-%m-%dT%H:%M:%S.%3N%z'

SCRIPT_START=$(date "$ISO_TIMESTAMP_FORMAT")
START_MS=$(date +%s%3N)
for OPS in "${STREAM_OPS[@]}"; do
	echo "running stream workload with ${OPS} operations"
	sleep $INIT_WAIT_SEC
	stress-ng --stream 0 --stream-ops "$OPS" --metrics-brief
done
END_MS=$(date +%s%3N)
SCRIPT_STOP=$(date "$ISO_TIMESTAMP_FORMAT")

DURATION_MS=$((END_MS - START_MS - INIT_WAIT_SEC * 1000))
EXECUTION_SECS=$(printf '%s.%03d' $((DURATION_MS / 1000)) $((DURATION_MS % 1000)))

printf '%s,%s,%s,%s\n' "$NAME_VALUE" "$SCRIPT_START" "$SCRIPT_STOP" "$EXECUTION_SECS" >>"$CSV_FILE"

echo "Done: ran memory workload for ${EXECUTION_SECS}s"