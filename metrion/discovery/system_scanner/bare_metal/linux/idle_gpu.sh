#!/usr/bin/env bash
# idle_gpu.sh  –  evict GPU users, measure idle GPU power using pynvml

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${SCRIPT_DIR}/../../../../../.venv/bin/python3"

COOLDOWN_SEC="${1:-60}"   # wait time after killing GPU users
MEASURE_SEC="${2:-60}"    # measurement dwell time
LOG_FOLDER="${3:-/var/log}"

###############################################################################
# 1. Log file
###############################################################################
LOG="${LOG_FOLDER}/gpu-idle.log"
mkdir -p "$LOG_FOLDER"
exec >>"$LOG" 2>&1

# Change ownership to the user who invoked sudo (if applicable)
if [[ -n "${SUDO_USER:-}" ]]; then
	chown -R "${SUDO_USER}:${SUDO_USER}" "$LOG_FOLDER"
fi

echo "LOG    : $LOG"
echo "Cooldown: ${COOLDOWN_SEC}s"
echo "Measure : ${MEASURE_SEC}s"
echo "Begin   : $(date --iso=seconds)"

###############################################################################
# 2. GPU discovery and helpers (using pynvml via Python)
###############################################################################
if ! command -v nvidia-smi >/dev/null 2>&1; then
	echo "Error  : nvidia-smi not found"
	exit 1
fi

if ! "$PYTHON" -c "import pynvml" 2>/dev/null; then
	echo "Error  : pynvml Python library not found. Install with: pip install pynvml"
	exit 1
fi

mapfile -t GPU_UUIDS < <(nvidia-smi --query-gpu=uuid --format=csv,noheader 2>/dev/null)
if [[ ${#GPU_UUIDS[@]} -eq 0 ]]; then
	echo "Error  : No NVIDIA GPUs detected"
	exit 1
fi

mapfile -t GPU_META < <(nvidia-smi --query-gpu=name,pci.bus_id,power.limit --format=csv,noheader 2>/dev/null)
echo "Found  : ${#GPU_UUIDS[@]} GPU(s)"
for i in "${!GPU_UUIDS[@]}"; do
	echo "GPU[$i] : ${GPU_META[i]} | UUID=${GPU_UUIDS[i]}"
done

read_all_power(){
	# Instantaneous power draw in Watts using pynvml
	"$PYTHON" << 'PYEOF'
import pynvml
pynvml.nvmlInit()
count = pynvml.nvmlDeviceGetCount()
for i in range(count):
    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
    power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
    print(f"{power_mw / 1000.0:.6f}")
pynvml.nvmlShutdown()
PYEOF
}

read_all_energy(){
	# Total energy counter in millijoules (mJ) since driver load using pynvml
	"$PYTHON" << 'PYEOF'
import pynvml
try:
    pynvml.nvmlInit()
    count = pynvml.nvmlDeviceGetCount()
    for i in range(count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        energy_mj = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
        print(energy_mj)
    pynvml.nvmlShutdown()
except pynvml.NVMLError as e:
    # Energy counter not supported on this GPU
    exit(1)
PYEOF
}

collect_gpu_pids(){
	local out pids=()
	if out=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null); then
		while read -r pid; do [[ -n "$pid" ]] && pids+=("$pid"); done <<<"$out"
	fi
	if out=$(nvidia-smi --query-accounted-apps=pid --format=csv,noheader,nounits 2>/dev/null || true); then
		while read -r pid; do [[ -n "$pid" ]] && pids+=("$pid"); done <<<"$out"
	fi
	if out=$(nvidia-smi pmon -c 1 2>/dev/null); then
		while read -r pid; do [[ -n "$pid" ]] && pids+=("$pid"); done < <(printf "%s\n" "$out" | awk 'NF>=5 && $2 ~ /^[0-9]+$/ {print $2}')
	fi

	declare -A seen_p; local uniq_p=()
	for pid in "${pids[@]}"; do
		[[ ${seen_p[$pid]+x} ]] && continue
		seen_p[$pid]=1; uniq_p+=("$pid")
	done
	printf "%s\n" "${uniq_p[@]}"
}

kill_gpu_pids(){
	local pids=("$@")
	if [[ ${#pids[@]} -eq 0 ]]; then
		echo "Kill   : no GPU processes running"
		return
	fi
	echo "Kill   : ${#pids[@]} GPU process(es): ${pids[*]}"
	for pid in "${pids[@]}"; do
		if kill -0 "$pid" 2>/dev/null; then
			kill -TERM "$pid" 2>/dev/null || true
		fi
	done
	sleep 2
	for pid in "${pids[@]}"; do
		if kill -0 "$pid" 2>/dev/null; then
			kill -KILL "$pid" 2>/dev/null || true
		fi
	done
}

###############################################################################
# 3. Main
###############################################################################
mapfile -t gpu_pids < <(collect_gpu_pids)
kill_gpu_pids "${gpu_pids[@]}"

echo "Cool   : waiting ${COOLDOWN_SEC}s for GPUs to settle"
sleep "$COOLDOWN_SEC"

declare -a sum_power
for i in "${!GPU_UUIDS[@]}"; do
	sum_power[i]=0
done

echo "Measure: start $(date --iso=seconds)"
energy_supported=true
mapfile -t energy_start < <(read_all_energy || true)
if [[ ${#energy_start[@]} -ne ${#GPU_UUIDS[@]} ]]; then
	energy_supported=false
fi

if $energy_supported; then
	echo "Method : energy counter (total_energy_consumption)"
	sleep "$MEASURE_SEC"
	mapfile -t energy_end < <(read_all_energy || true)
	if [[ ${#energy_end[@]} -ne ${#GPU_UUIDS[@]} ]]; then
		echo "Warn   : energy counter unavailable at end; falling back to power draw"
		energy_supported=false
	else
		for i in "${!GPU_UUIDS[@]}"; do
			printf 'Raw    : GPU[%d] energy_start=%s mJ energy_end=%s mJ\n' \
				"$i" "${energy_start[i]}" "${energy_end[i]}"
		done
	fi
fi

if $energy_supported; then
	declare -a avg_power
	total_power=0
	for i in "${!GPU_UUIDS[@]}"; do
		delta_mj=$(bc -l <<< "scale=6;${energy_end[i]}-${energy_start[i]}")
		if (( $(echo "$delta_mj < 0" | bc -l) )); then
			delta_mj=0
		fi
		printf 'Delta  : GPU[%d] delta_energy=%s mJ over %s sec\n' \
			"$i" "$delta_mj" "$MEASURE_SEC"
		avg_power[i]=$(bc -l <<< "scale=6;${delta_mj}/1000/${MEASURE_SEC}")
		total_power=$(bc -l <<< "${total_power}+${avg_power[i]}")
		printf 'GPU[%d] %-60s : %8.4f W (energy)\n' "$i" "${GPU_META[i]} | UUID=${GPU_UUIDS[i]}" "${avg_power[i]}"
	done
	printf 'Total GPU Idle Power                      : %8.4f W\n' "$total_power"
else
	echo "Method : power.draw sampling"
	samples=0
	for (( t=1; t<=MEASURE_SEC; t++ )); do
		mapfile -t power_now < <(read_all_power)
		if [[ ${#power_now[@]} -ne ${#GPU_UUIDS[@]} ]]; then
			echo "Error  : power readings unavailable for all GPUs"
			exit 1
		fi
		for i in "${!GPU_UUIDS[@]}"; do
			sum_power[i]=$(bc -l <<< "scale=6;${sum_power[i]}+${power_now[i]}")
		done
		samples=$((samples+1))
		sleep 1
	done

	echo "=== GPU Idle Power (avg over ${samples}s) ==="
	total_power=0
	for i in "${!GPU_UUIDS[@]}"; do
		avg=$(bc -l <<< "scale=4;${sum_power[i]}/${samples}")
		total_power=$(bc -l <<< "${total_power}+${avg}")
		printf 'GPU[%d] %-60s : %8.4f W\n' "$i" "${GPU_META[i]} | UUID=${GPU_UUIDS[i]}" "$avg"
	done
	printf 'Total GPU Idle Power                      : %8.4f W\n' "$total_power"
fi

echo "Done   : $(date --iso=seconds)"
