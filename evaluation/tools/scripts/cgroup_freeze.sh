#!/usr/bin/env bash
#
# cgroup_freeze.sh - Shared library for cgroup freezing/thawing operations
#
# Purpose:
#   Freeze background system processes during energy measurements to reduce
#   noise and isolate the workload's energy consumption.
#
# Usage:
#   source /path/to/cgroup_freeze.sh
#   setup_freeze_units   # Build the FREEZE_UNITS array
#   freeze_units         # Freeze all units
#   wait_for_frozen 60   # Wait up to 60s for frozen state
#   thaw_units           # Thaw all units
#
# Design - Why system.slice is NOT frozen:
#   This library deliberately excludes system.slice from freezing. This is
#   critical because:
#
#   1. The measurement scripts themselves run via systemd-run, which places
#      them in system.slice. Freezing system.slice would freeze the scripts.
#
#   2. The profilers (metrion, scaphandre) are started via systemd-run with
#      custom unit names (e.g., --unit="Metrion-orchestrator-$$"), placing
#      them in system.slice where they can continue collecting data.
#
#   3. The workload (stress-ng) runs as a child process of the wrapper script,
#      inheriting its cgroup (system.slice), so it can execute normally.
#
#   Execution hierarchy (all in system.slice, NOT frozen):
#
#     system.slice (NEVER frozen)
#     ├── Metrion-agent-$$ (measure_energy_of_workload.sh)
#     │   └── metrion_wrapper.sh
#     │       └── bash cpu_workloads.sh
#     │           └── stress-ng
#     └── Metrion-orchestrator-$$
#         └── python orchestrator.py
#
#   What DOES get frozen:
#     - user.slice (user sessions, terminals, desktop apps)
#     - kubepods.slice (Kubernetes pods)
#     - Other application slices
#     - Container services (k3s, containerd, docker)
#

# --- Configuration ---

# Timeout for waiting on FreezerState=frozen (seconds)
FREEZE_TIMEOUT="${FREEZE_TIMEOUT:-60}"

# Wait time in seconds after freezing cgroups before proceeding
IDLE_WAIT_SEC="${IDLE_WAIT_SEC:-5}"

# Explicit kubepods parent slices (freeze parents first for proper ordering)
KUBEPODS_PARENTS=(
    kubepods.slice
    kubepods-burstable.slice
    kubepods-besteffort.slice
)

# Array to hold units to freeze (populated by setup_freeze_units)
declare -a FREEZE_UNITS

# Track frozen state
CGROUP_FROZEN=0

# --- Functions ---

setup_freeze_units() {
    # Start with user.slice and kubepods parents
    FREEZE_UNITS=(user.slice)
    for parent in "${KUBEPODS_PARENTS[@]}"; do
        # Only add if the slice exists
        if systemctl list-units --type=slice --all --no-legend | grep -q "^${parent}"; then
            FREEZE_UNITS+=("$parent")
        fi
    done

    # List ALL slices (not just --state=active) to catch activating/deactivating/loaded slices
    while read -r unit _; do
        case "$unit" in
            # CRITICAL: system.slice must NEVER be frozen - it contains our measurement
            # scripts, profilers (metrion/scaphandre), and workloads (stress-ng).
            # See header documentation for the full execution hierarchy.
            "-.slice"|system.slice|init.scope) ;;
            *) FREEZE_UNITS+=("$unit") ;;
        esac
    done < <(systemctl list-units --type=slice --all --no-legend)

    # Add common services that might be active
    local extra_units=(k3s.service containerd.service docker.service)
    FREEZE_UNITS+=("${extra_units[@]}")

    # De-duplicate the list
    declare -A seen
    local uniq=()
    for u in "${FREEZE_UNITS[@]}"; do
        [[ -z "$u" ]] && continue
        [[ ${seen[$u]+x} ]] && continue
        seen[$u]=1
        uniq+=("$u")
    done
    FREEZE_UNITS=("${uniq[@]}")
}

freeze_units() {
    echo "--- Freezing cgroup units ---"
    for u in "${FREEZE_UNITS[@]}"; do
        echo "Freezing $u"
        systemctl freeze "$u" || echo "Warning: Could not freeze $u"
    done
    echo "-----------------------------"
    CGROUP_FROZEN=1
}

thaw_units() {
    echo "--- Thawing cgroup units ---"
    # Thaw in reverse order (children first, then parents)
    for (( i=${#FREEZE_UNITS[@]}-1; i>=0; i-- )); do
        local u="${FREEZE_UNITS[i]}"
        echo "Thawing $u"
        systemctl thaw "$u" 2>/dev/null || true
    done
    echo "----------------------------"
    CGROUP_FROZEN=0
}

# Wait for all units to reach FreezerState=frozen
wait_for_frozen() {
    local timeout="${1:-$FREEZE_TIMEOUT}"
    local start_time end_time elapsed
    start_time=$(date +%s)
    end_time=$((start_time + timeout))

    echo "Waiting for FreezerState=frozen (timeout: ${timeout}s)..."

    while true; do
        local all_frozen=true
        local still_freezing=()

        for u in "${FREEZE_UNITS[@]}"; do
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

# Convenience function: freeze, wait, and settle
freeze_and_wait() {
    freeze_units
    if ! wait_for_frozen "$FREEZE_TIMEOUT"; then
        echo "WARNING: Proceeding despite incomplete freeze"
    fi
    echo "Waiting ${IDLE_WAIT_SEC}s for system to settle..."
    sleep "$IDLE_WAIT_SEC"
}

# Cleanup function for traps - only thaws if frozen
cleanup_freeze() {
    if [[ "$CGROUP_FROZEN" -eq 1 ]]; then
        thaw_units
    fi
}
