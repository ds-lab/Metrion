#!/usr/bin/env bash
#
# measure_energy_of_workload.sh - A script to measure the energy consumption of a workload.
#
# This script freezes most system processes using cgroups, reads initial RAPL
# energy values for package and DRAM, executes a given workload script, reads
# the final RAPL values, and calculates the energy consumed. The results are
# appended to a CSV file.
#
# Usage:
# To prevent the script from freezing the terminal session it is run from,
# it is recommended to execute it via 'systemd-run'.
#
#   sudo systemd-run --unit="metrion-agent-$$" --wait \
#       ./measure_energy_of_workload.sh <workload_name> <path_to_workload_script> <output_csv_file> [--freeze|--no-freeze]
#
# Example:
#   sudo systemd-run --unit="metrion-agent-$$" --wait \
#       ./measure_energy_of_workload.sh "cpu_workload" benchmarks/workloads/cpu_workloads.sh measurements.csv --no-freeze
#

set -euo pipefail

# --- Source shared libraries ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/tools/scripts/cgroup_freeze.sh"

# --- Script Arguments ---

if [[ "$#" -lt 3 || "$#" -gt 4 ]]; then
    echo "Usage: $0 <workload_name> <workload_script> <output_csv> [--freeze|--no-freeze]"
    exit 1
fi

WORKLOAD_NAME="$1"
WORKLOAD_SCRIPT="$2"
OUTPUT_CSV="$3"
FREEZE_ARG="${4:---freeze}"

MAX_DRAM=65712999613
MAX_PKG=262143328850

case "$FREEZE_ARG" in
    --freeze)
        FREEZE_ENABLED=1
        ;;
    --no-freeze)
        FREEZE_ENABLED=0
        ;;
    *)
        echo "Error: Invalid freeze option '$FREEZE_ARG'. Use --freeze or --no-freeze."
        exit 1
        ;;
esac

if [[ ! -f "$WORKLOAD_SCRIPT" ]]; then
    echo "Error: Workload script not found at '$WORKLOAD_SCRIPT'"
    exit 1
fi

if [[ "$(id -u)" -ne 0 ]]; then
    echo "Error: This script must be run as root."
    exit 1
fi

# --- Cgroup Management (uses shared library) ---

cleanup() {
    if [[ "$FREEZE_ENABLED" -eq 1 ]]; then
        cleanup_freeze
    fi
}

trap 'cleanup' EXIT INT TERM

# --- RAPL Reading ---

BASE_RAPL_PATH=/sys/devices/virtual/powercap/intel-rapl
declare -a PKG_PATHS
declare -a DRAM_PATHS

# Find all package directories (e.g., intel-rapl:0, intel-rapl:1)
for pkg_dir in "$BASE_RAPL_PATH"/intel-rapl:[0-9]; do
    # Ensure it's a directory and the energy file is readable
    if [[ -d "$pkg_dir" && -r "${pkg_dir}/energy_uj" ]]; then
        PKG_PATHS+=("${pkg_dir}/energy_uj")

        # Look for a corresponding DRAM sub-domain within the package directory
        dram_energy_file=""
        # Loop through potential sub-domains (e.g., intel-rapl:0:0)
        for sub_dir in "${pkg_dir}"/intel-rapl:*:*; do
            # Check if it's a directory with a 'name' file that contains 'dram'
            if [[ -d "$sub_dir" && -f "${sub_dir}/name" && $(<"${sub_dir}/name") == "dram" ]]; then
                if [[ -r "${sub_dir}/energy_uj" ]]; then
                    dram_energy_file="${sub_dir}/energy_uj"
                    break # Found the dram file, no need to check other sub-domains
                fi
            fi
        done
        DRAM_PATHS+=("$dram_energy_file") # Will be empty if not found
    fi
done

if [[ ${#PKG_PATHS[@]} -eq 0 ]]; then
    echo "Error: No RAPL package domains found under $BASE_RAPL_PATH"
    exit 1
fi

echo "Found ${#PKG_PATHS[@]} RAPL package(s)."

if [[ ${#DRAM_PATHS[@]} -eq 0 ]]; then
    echo "Error: No RAPL DRAM domains found under $BASE_RAPL_PATH"
    exit 1
fi

echo "Found ${#DRAM_PATHS[@]} RAPL DRAM domain(s)."

# Reads the sum of energy values from a list of paths.
# $1: array name to read from (e.g., PKG_PATHS)
read_energy_values() {
    local -n path_array=$1
    local values=()
    for path in "${path_array[@]}"; do
        if [[ -f "$path" ]]; then
            values+=("$(<"$path")")
        fi
    done
    printf '%s\n' "${values[@]}"
}

sum_energy_values() {
    local sum=0
    for value in "$@"; do
        [[ -z "$value" ]] && continue
        sum=$((sum + value))
    done
    echo "$sum"
}
compute_wrapped_diffs() {
    local -n start_arr=$1
    local -n end_arr=$2
    local max_val=$3
    local diffs=()
    for idx in "${!start_arr[@]}"; do
        local start=${start_arr[$idx]}
        local end=${end_arr[$idx]}
        local diff
        if (( end < start )); then
            diff=$((max_val - start + end))
        else
            diff=$((end - start))
        fi
        diffs+=("$diff")
    done
    printf '%s\n' "${diffs[@]}"
}

# --- Main Execution ---

if [[ "$FREEZE_ENABLED" -eq 1 ]]; then
    setup_freeze_units
    freeze_and_wait
else
    echo "Skipping cgroup freezing (--no-freeze)"
fi

echo "Reading start RAPL values..."
start_time=$(date --iso-8601=seconds)
mapfile -t pkg_start_values < <(read_energy_values PKG_PATHS)
mapfile -t dram_start_values < <(read_energy_values DRAM_PATHS)
pkg_start=$(sum_energy_values "${pkg_start_values[@]}")
dram_start=$(sum_energy_values "${dram_start_values[@]}")

echo "Executing workload script: $WORKLOAD_SCRIPT"
echo "--- Workload Output Start ---"
# Execute the script and wait for it to complete
bash "$WORKLOAD_SCRIPT"
echo "--- Workload Output End ---"
echo "Workload script finished."

# Thaw cgroups immediately after workload completes
if [[ "$FREEZE_ENABLED" -eq 1 && "$CGROUP_FROZEN" -eq 1 ]]; then
    thaw_units
fi

echo "Reading end RAPL values..."
stop_time=$(date --iso-8601=seconds)
mapfile -t pkg_end_values < <(read_energy_values PKG_PATHS)
mapfile -t dram_end_values < <(read_energy_values DRAM_PATHS)
pkg_end=$(sum_energy_values "${pkg_end_values[@]}")
dram_end=$(sum_energy_values "${dram_end_values[@]}")

# Thawing was already done above; cleanup trap is a safety net for error cases

# --- Calculation and CSV Output ---

mapfile -t pkg_diff_values < <(compute_wrapped_diffs pkg_start_values pkg_end_values "${MAX_PKG:-0}")
mapfile -t dram_diff_values < <(compute_wrapped_diffs dram_start_values dram_end_values "${MAX_DRAM:-0}")
pkg_diff=$(sum_energy_values "${pkg_diff_values[@]}")
dram_diff=$(sum_energy_values "${dram_diff_values[@]}")

# Create CSV with header if it doesn't exist
if [[ ! -f "$OUTPUT_CSV" ]]; then
    echo "workload_name,freeze,start,stop,pkg_start_uj,pkg_end_uj,dram_start_uj,dram_end_uj,pkg_diff_uj,dram_diff_uj" > "$OUTPUT_CSV"
fi

# Append results
echo "$WORKLOAD_NAME,$FREEZE_ENABLED,$start_time,$stop_time,$pkg_start,$pkg_end,$dram_start,$dram_end,$pkg_diff,$dram_diff" >> "$OUTPUT_CSV"

echo "Results appended to $OUTPUT_CSV"
echo "Done."