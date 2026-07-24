#!/usr/bin/env bash
# freeze-idle-rapl.sh  –  freeze slices, sample RAPL, thaw
#
# Uses the RAPL files located at
#   /sys/devices/virtual/powercap/intel-rapl/intel-rapl:<n>/energy_uj
#
# $1 : dwell time in seconds  (default 30)
# $2 : log folder path        (default /var/log)
# $3…: extra units / slices   (optional)
#
# Typical detached launch from SSH:
#   sudo systemd-run --unit=rapl-idle --property=StandardOutput=file:/var/log/rapl-idle.log freeze-idle-rapl.sh 120 /var/log kubepods.slice k3s.service containerd.service
#
set -euo pipefail

# Force a locale-independent decimal point for this script's own printf/bc
# calculations. bc always emits dot-decimal output; under a comma-decimal
# LC_NUMERIC (e.g. de_DE), bash's printf "%f" silently truncates that output
# at the decimal point instead of reformatting it, corrupting every power
# value below. Scoped to this script's environment only -- does not change
# the host's system-wide locale configuration.
export LC_NUMERIC=C

IDLE_SEC="${1:-30}"
shift || true
LOG_FOLDER="${1:-/var/log}"
shift || true
EXTRA_UNITS=("$@")

# Timeout for waiting on FreezerState=frozen (seconds)
FREEZE_TIMEOUT=60

###############################################################################
# 1. Build the list of units/slices to freeze
###############################################################################
# Explicit kubepods parent slices (freeze parents first for proper ordering)
KUBEPODS_PARENTS=(
    kubepods.slice
    kubepods-burstable.slice
    kubepods-besteffort.slice
)

# Start with user.slice and kubepods parents
FREEZE=(user.slice)
for parent in "${KUBEPODS_PARENTS[@]}"; do
    # Only add if the slice exists
    if systemctl list-units --type=slice --all --no-legend | grep -q "^${parent}"; then
        FREEZE+=("$parent")
    fi
done

# List ALL slices (not just --state=active) to catch activating/deactivating/loaded slices
while read -r unit _; do
    case "$unit" in
        "-.slice"|system.slice|init.scope) ;;      # never freeze these
        *)  FREEZE+=("$unit") ;;
    esac
done < <(systemctl list-units --type=slice --all --no-legend)
FREEZE+=("${EXTRA_UNITS[@]}")

# de-duplicate while preserving order (parents first)
declare -A seen; uniq=()
for u in "${FREEZE[@]}"; do
    [[ ${seen[$u]+x} ]] && continue
    seen[$u]=1; uniq+=("$u")
done
FREEZE=("${uniq[@]}")

###############################################################################
# 2. Log file
###############################################################################
LOG="${LOG_FOLDER}/rapl-idle.log"
mkdir -p "$LOG_FOLDER"
exec >>"$LOG" 2>&1

# Change ownership to the user who invoked sudo (if applicable)
if [[ -n "${SUDO_USER:-}" ]]; then
    chown -R "${SUDO_USER}:${SUDO_USER}" "$LOG_FOLDER"
fi

echo "LOG    : $LOG"
echo "Dwell  : ${IDLE_SEC}s"
echo "Freeze : ${FREEZE[*]}"
echo "Units  : ${#FREEZE[@]} total"

# Pod count sanity check (if kubectl available)
if command -v kubectl &>/dev/null; then
    pod_count=$(kubectl get pods --all-namespaces --no-headers 2>/dev/null | wc -l || echo "N/A")
    echo "Pods   : $pod_count (kubectl)"
fi

echo "Begin  : $(date --iso=seconds)"

###############################################################################
# 3. Locate all RAPL packages and sub-domains under /sys/devices/…
###############################################################################
BASE=/sys/devices/virtual/powercap/intel-rapl
declare -a PKG_PATHS
declare -a PKG_NUMS
declare -A CPU_PATHS
declare -A GPU_PATHS
declare -A DRAM_PATHS

# Find all package domains and their sub-domains
# Only directories whose name file reads "package-N" are CPU socket packages;
# platform domains like "psys" are skipped.
while IFS= read -r -d '' pkg_path; do
    name_file="$pkg_path/name"
    [[ -f "$name_file" ]] || continue
    domain_name=$(<"$name_file")
    [[ "$domain_name" == package-* ]] || continue

    pkg_num="${domain_name#package-}"   # "package-0" → "0"
    PKG_PATHS+=("$pkg_path")
    PKG_NUMS+=("$pkg_num")

    # Dynamically find all sub-domains (cpu, gpu, dram)
    while IFS= read -r -d '' sub_path; do
        name_file="$sub_path/name"
        if [[ -f "$name_file" ]]; then
            sub_name=$(<"$name_file")
            case "$sub_name" in
                core|cores|cpu)
                    CPU_PATHS[$pkg_num]="$sub_path"
                    ;;
                uncore|gpu)
                    GPU_PATHS[$pkg_num]="$sub_path"
                    ;;
                dram)
                    DRAM_PATHS[$pkg_num]="$sub_path"
                    ;;
            esac
        fi
    done < <(find "$pkg_path" -maxdepth 1 -type d -name 'intel-rapl:*:*' -print0)
done < <(find "$BASE" -maxdepth 1 -type d -name 'intel-rapl:[0-9]*' \
         -exec test -f '{}/energy_uj' ';' -print0 | sort -z)

if [[ ${#PKG_PATHS[@]} -eq 0 ]]; then
    echo "Error  : No RAPL package domains found under $BASE"
    exit 1
fi

echo "Found  : ${#PKG_PATHS[@]} package domain(s)"
for i in "${!PKG_PATHS[@]}"; do
    pkg_num="${PKG_NUMS[i]}"

    # Show what domains are available for this package
    domains="Package"
    [[ -n "${CPU_PATHS[$pkg_num]:-}" ]] && domains+=", CPU cores"
    [[ -n "${GPU_PATHS[$pkg_num]:-}" ]] && domains+=", GPU"
    [[ -n "${DRAM_PATHS[$pkg_num]:-}" ]] && domains+=", DRAM"
    
    echo "Package: ${pkg_num} (${domains})"
done

# echoes space-separated values: pkg cpu gpu dram for each package
read_all_rapl() {
    local output=""
    for i in "${!PKG_PATHS[@]}"; do
        local pkg cpu gpu dram
        pkg_num="${PKG_NUMS[i]}"
        
        # Package total
        pkg=$(<"${PKG_PATHS[i]}/energy_uj")
        
        # CPU cores (PP0)
        if [[ -n "${CPU_PATHS[$pkg_num]:-}" ]]; then
            cpu=$(<"${CPU_PATHS[$pkg_num]}/energy_uj")
        else
            cpu="-"
        fi
        
        # GPU (PP1)
        if [[ -n "${GPU_PATHS[$pkg_num]:-}" ]]; then
            gpu=$(<"${GPU_PATHS[$pkg_num]}/energy_uj")
        else
            gpu="-"
        fi
        
        # DRAM
        if [[ -n "${DRAM_PATHS[$pkg_num]:-}" ]]; then
            dram=$(<"${DRAM_PATHS[$pkg_num]}/energy_uj")
        else
            dram="-"
        fi
        
        output+="$pkg $cpu $gpu $dram "
    done
    echo "$output"
}

###############################################################################
# 4. Helpers
###############################################################################
freeze_units(){
    for u in "${FREEZE[@]}"; do
        echo "Freezing $u"
        systemctl freeze "$u" 2>/dev/null || echo "  (skip: $u not freezable or not found)"
    done
}

thaw_units(){
    # Thaw in reverse order (children first, then parents)
    for (( i=${#FREEZE[@]}-1; i>=0; i-- )); do
        local u="${FREEZE[i]}"
        echo "Thawing  $u"
        systemctl thaw "$u" 2>/dev/null || true
    done
}

# Wait for all units to reach FreezerState=frozen
wait_for_frozen(){
    local timeout="$1"
    local start_time end_time elapsed
    start_time=$(date +%s)
    end_time=$((start_time + timeout))
    
    echo "Waiting for FreezerState=frozen (timeout: ${timeout}s)..."
    
    while true; do
        local all_frozen=true
        local still_freezing=()
        
        for u in "${FREEZE[@]}"; do
            # Get FreezerState property (returns empty if not applicable)
            local state
            state=$(systemctl show -p FreezerState --value "$u" 2>/dev/null || echo "")
            
            # Skip units that don't support freezing or are already frozen
            if [[ -z "$state" ]] || [[ "$state" == "frozen" ]]; then
                continue
            fi
            
            # Unit is still freezing
            all_frozen=false
            still_freezing+=("$u")
        done
        
        if $all_frozen; then
            echo "All units frozen."
            return 0
        fi
        
        elapsed=$(( $(date +%s) - start_time ))
        if (( $(date +%s) >= end_time )); then
            echo "WARNING: Timeout after ${elapsed}s. Still freezing: ${still_freezing[*]}"
            return 1
        fi
        
        # Brief sleep before polling again
        sleep 0.2
    done
}

###############################################################################
# 5. Main
###############################################################################
freeze_units
trap 'echo "Thaw   : $(date --iso=seconds)"; thaw_units' EXIT INT TERM

# Wait for all units to actually be frozen before measuring
if ! wait_for_frozen "$FREEZE_TIMEOUT"; then
    echo "WARNING: Proceeding with measurement despite incomplete freeze"
fi

echo "RAPL   : Starting measurement at $(date --iso=seconds)"

# Read initial energy values for all packages
read -a energy0 < <(read_all_rapl)
sleep "$IDLE_SEC"
# Read final energy values for all packages
read -a energy1 < <(read_all_rapl)

# Calculate and display power consumption for each package
for i in "${!PKG_PATHS[@]}"; do
    pkg_num="${PKG_NUMS[i]}"
    echo "=== Package $pkg_num ==="
    
    # Each package has 4 values: pkg, cpu, gpu, dram
    base_idx=$((i * 4))
    
    # Package total power
    pkg_idx=$base_idx
    # Calculates the average package power consumption (in Watts) during the idle period.
    # The formula computes the difference in energy readings for the package (in microjoules),
    # divides by the idle duration in seconds, and converts the result from microjoules to joules (by dividing by 1,000,000).
    # This yields the average power usage over the idle interval.
    pkg_power=$(bc -l <<< "scale=4;(${energy1[pkg_idx]}-${energy0[pkg_idx]})/$IDLE_SEC/1000000")
    printf 'Package Total      : %8.4f W\n' "$pkg_power"
    
    # CPU cores power (PP0)
    cpu_idx=$((base_idx + 1))
    if [[ "${energy0[cpu_idx]}" != "-" ]] && [[ "${energy1[cpu_idx]}" != "-" ]]; then
        cpu_power=$(bc -l <<< "scale=4;(${energy1[cpu_idx]}-${energy0[cpu_idx]})/$IDLE_SEC/1000000")
        printf 'CPU Cores          : %8.4f W\n' "$cpu_power"
    else
        printf 'CPU Cores          : %8s\n' "N/A"
    fi
    
    # GPU power (PP1)
    gpu_idx=$((base_idx + 2))
    if [[ "${energy0[gpu_idx]}" != "-" ]] && [[ "${energy1[gpu_idx]}" != "-" ]]; then
        gpu_power=$(bc -l <<< "scale=4;(${energy1[gpu_idx]}-${energy0[gpu_idx]})/$IDLE_SEC/1000000")
        printf 'Integrated GPU     : %8.4f W\n' "$gpu_power"
    else
        printf 'Integrated GPU     : %8s\n' "N/A"
    fi
    
    # DRAM power
    dram_idx=$((base_idx + 3))
    if [[ "${energy0[dram_idx]}" != "-" ]] && [[ "${energy1[dram_idx]}" != "-" ]]; then
        dram_power=$(bc -l <<< "scale=4;(${energy1[dram_idx]}-${energy0[dram_idx]})/$IDLE_SEC/1000000")
        printf 'DRAM               : %8.4f W\n' "$dram_power"
    else
        printf 'DRAM               : %8s\n' "N/A"
    fi
    
    # Calculate uncore power (Package - CPU - GPU)
    if [[ "${energy0[cpu_idx]}" != "-" ]] && [[ "${energy1[cpu_idx]}" != "-" ]]; then
        uncore_power="$pkg_power"
        uncore_power=$(bc -l <<< "$uncore_power - $cpu_power")
        
        if [[ "${energy0[gpu_idx]}" != "-" ]] && [[ "${energy1[gpu_idx]}" != "-" ]]; then
            uncore_power=$(bc -l <<< "$uncore_power - $gpu_power")
        fi
        
        printf 'Uncore (estimated) : %8.4f W\n' "$uncore_power"
    fi
    
    echo  # Empty line for readability between packages
done

# Calculate total system power
total_pkg_power=0
total_cpu_power=0
total_gpu_power=0
total_dram_power=0
total_uncore_power=0
dram_available=false
cpu_available=false
gpu_available=false

for i in "${!PKG_PATHS[@]}"; do
    base_idx=$((i * 4))
    pkg_idx=$base_idx
    cpu_idx=$((base_idx + 1))
    gpu_idx=$((base_idx + 2))
    dram_idx=$((base_idx + 3))
    
    # Package total
    pkg_power=$(bc -l <<< "scale=4;(${energy1[pkg_idx]}-${energy0[pkg_idx]})/$IDLE_SEC/1000000")
    total_pkg_power=$(bc -l <<< "$total_pkg_power + $pkg_power")
    
    # CPU cores
    if [[ "${energy0[cpu_idx]}" != "-" ]] && [[ "${energy1[cpu_idx]}" != "-" ]]; then
        cpu_power=$(bc -l <<< "scale=4;(${energy1[cpu_idx]}-${energy0[cpu_idx]})/$IDLE_SEC/1000000")
        total_cpu_power=$(bc -l <<< "$total_cpu_power + $cpu_power")
        cpu_available=true
    fi
    
    # GPU
    if [[ "${energy0[gpu_idx]}" != "-" ]] && [[ "${energy1[gpu_idx]}" != "-" ]]; then
        gpu_power=$(bc -l <<< "scale=4;(${energy1[gpu_idx]}-${energy0[gpu_idx]})/$IDLE_SEC/1000000")
        total_gpu_power=$(bc -l <<< "$total_gpu_power + $gpu_power")
        gpu_available=true
    fi
    
    # DRAM
    if [[ "${energy0[dram_idx]}" != "-" ]] && [[ "${energy1[dram_idx]}" != "-" ]]; then
        dram_power=$(bc -l <<< "scale=4;(${energy1[dram_idx]}-${energy0[dram_idx]})/$IDLE_SEC/1000000")
        total_dram_power=$(bc -l <<< "$total_dram_power + $dram_power")
        dram_available=true
    fi
done

echo "=== System Totals ==="
printf 'Total Package Power: %8.4f W\n' "$total_pkg_power"

if [[ "$cpu_available" == true ]]; then
    printf 'Total CPU Cores    : %8.4f W\n' "$total_cpu_power"
fi

if [[ "$gpu_available" == true ]]; then
    printf 'Total GPU          : %8.4f W\n' "$total_gpu_power"
fi

if [[ "$dram_available" == true ]]; then
    printf 'Total DRAM         : %8.4f W\n' "$total_dram_power"
fi

if [[ "$cpu_available" == true ]]; then
    total_uncore_power="$total_pkg_power"
    total_uncore_power=$(bc -l <<< "$total_uncore_power - $total_cpu_power")
    
    if [[ "$gpu_available" == true ]]; then
        total_uncore_power=$(bc -l <<< "$total_uncore_power - $total_gpu_power")
    fi
    
    printf 'Total Uncore (est.): %8.4f W\n' "$total_uncore_power"
fi

if [[ "$dram_available" == true ]]; then
    total_system_power=$(bc -l <<< "$total_pkg_power + $total_dram_power")
    printf 'Total System       : %8.4f W\n' "$total_system_power"
fi

echo "Done   : $(date --iso=seconds)"
