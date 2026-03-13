-- 0) Per-DRAM energy intervals with counter wrap handling

CREATE TABLE IF NOT EXISTS energy_interval_dram_tmp AS
WITH energy_samples_dram AS (
    SELECT
        m.id,
        m.value,
        m.metric_id,
        m.physical_entity_id AS dram_id,
        e.parent_id AS cpu_id,
        m.logical_entity_id,
        m.start_ts,
        m."end_ts"
    FROM
        measurement AS m
        JOIN dram AS d ON m.physical_entity_id = d.id
        JOIN entity AS e ON m.physical_entity_id = e.id
    WHERE
        m.metric_id = 'DRAM_SOCKET_TOTAL_ENERGY'
        AND m.start_ts >= '{start_time}'
        AND m.start_ts <= '{end_time}'
    ORDER BY
        m.start_ts
),
energy_interval_dram AS (
    SELECT *
    FROM (
        SELECT
            dram_id,
            cpu_id,
            "start_ts" AS interval_start,
            LEAD("start_ts") OVER w AS interval_end,
            CASE
                WHEN LEAD(value) OVER w IS NULL THEN 0
                WHEN LEAD(value) OVER w >= value THEN LEAD(value) OVER w - value
                ELSE ((65712999613.0 / 1000000.0) - value) + LEAD(value) OVER w
            END AS energy_j
        FROM energy_samples_dram
        WINDOW w AS (PARTITION BY dram_id ORDER BY "start_ts")
    )
    WHERE interval_end IS NOT NULL
)
SELECT dram_id, cpu_id, interval_start, interval_end, energy_j
FROM energy_interval_dram;

--sqlite CREATE INDEX IF NOT EXISTS idx_ei_dram_end_start
--sqlite   ON energy_interval_dram_tmp (dram_id, cpu_id, interval_end, interval_start);

-- 1) Final view: active via reads, idle via PID count
CREATE TABLE IF NOT EXISTS dram_energy_per_thread_view AS
WITH
-- (A) Idle/active split per DRAM interval
dram_interval_powers AS (
    SELECT
        vi.dram_id,
        vi.cpu_id,
        vi.interval_start,
        vi.interval_end,
        vi.energy_j,
        ({date_method}(vi.interval_end) - {date_method}(vi.interval_start)) * 24.0 * 3600.0 AS duration_s,
        d.idle_power
    FROM energy_interval_dram_tmp AS vi
    JOIN dram AS d ON vi.dram_id = d.id
),
dram_interval_energy_split AS (
    SELECT
        dram_id,
        cpu_id,
        interval_start,
        interval_end,
        energy_j,
        idle_power,
        duration_s,
        (idle_power * duration_s) AS idle_energy_j_total,
        {max_func}(energy_j - (idle_power * duration_s), 0.0) AS active_energy_j_total
    FROM dram_interval_powers
),

-- (B) PIDs present on DRAM (used for IDLE equal split). Presence = has memory in the interval.
dram_memory_usage_by_thread AS (
    SELECT
        dei.dram_id,
        dei.cpu_id,
        dei.interval_start,
        dei.interval_end,
        m.logical_entity_id,
        SUM(m.value) AS total_memory_usage_by_thread
    FROM energy_interval_dram_tmp AS dei
    JOIN measurement AS m
      ON m.physical_entity_id = dei.dram_id
     AND m.metric_id = 'DRAM_SOCKET_PID_MEMORY'
     AND m.value IS NOT NULL
     AND ABS(m.value) < 1e307
     AND m.start_ts >= dei.interval_start
     AND m."end_ts" <= dei.interval_end
    GROUP BY
        dei.dram_id, dei.cpu_id, dei.interval_start, dei.interval_end, m.logical_entity_id
),
dram_threads_present_on_dram AS (
    SELECT DISTINCT
        dram_id, cpu_id, interval_start, interval_end, logical_entity_id
    FROM dram_memory_usage_by_thread
    WHERE total_memory_usage_by_thread > 0
),
active_threads_on_dram AS (
    SELECT
        dram_id, cpu_id, interval_start, interval_end,
        COUNT(DISTINCT logical_entity_id) AS thread_count
    FROM dram_threads_present_on_dram
    GROUP BY
        dram_id, cpu_id, interval_start, interval_end
),

-- (C) DRAM access counts per PID from core PMU metrics (drives ACTIVE split)
--     Local reads: cores on this CPU reading this CPU's DRAM.
dram_local_reads_by_thread AS (
    SELECT
        dei.dram_id,                -- target DRAM
        dei.cpu_id,
        dei.interval_start,
        dei.interval_end,
        m.logical_entity_id,
        SUM(m.value) AS accesses
    FROM
        energy_interval_dram_tmp dei,
        entity e,
        measurement m
    WHERE
        e.parent_id = dei.cpu_id              -- cores of this CPU
        AND m.physical_entity_id = e.id
        AND m.metric_id = 'CPU_CORE_PID_MEM_LOAD_L3_MISS_LOCAL_DRAM'
        AND m.start_ts >= dei.interval_start
        AND m."end_ts"  <= dei.interval_end
    GROUP BY
        dei.dram_id, dei.cpu_id, dei.interval_start, dei.interval_end, m.logical_entity_id
),

--     Remote reads: cores on other CPUs -> this DRAM (accurate for 2-socket)
dram_remote_reads_by_thread AS (
    SELECT
        dei.dram_id,                -- target DRAM (remote to those cores)
        dei.cpu_id,
        dei.interval_start,
        dei.interval_end,
        m.logical_entity_id,
        SUM(m.value) AS accesses
    FROM
        energy_interval_dram_tmp dei,
        entity e,
        measurement m
    WHERE
        m.metric_id = 'CPU_CORE_PID_MEM_LOAD_L3_MISS_REMOTE_DRAM'
        AND m.start_ts >= dei.interval_start
        AND m."end_ts"  <= dei.interval_end
        AND e.id = m.physical_entity_id
        AND e.parent_id <> dei.cpu_id
    GROUP BY
        dei.dram_id, dei.cpu_id, dei.interval_start, dei.interval_end, m.logical_entity_id
),
dram_accesses_all AS (
    SELECT * FROM dram_local_reads_by_thread
    UNION ALL
    SELECT * FROM dram_remote_reads_by_thread
),
dram_accesses_by_thread AS (
    SELECT
        dram_id, cpu_id, interval_start, interval_end, logical_entity_id,
        SUM(accesses) AS total_accesses
    FROM dram_accesses_all
    GROUP BY dram_id, cpu_id, interval_start, interval_end, logical_entity_id
),
dram_accesses_total_on_dram AS (
    SELECT
        dram_id, cpu_id, interval_start, interval_end,
        SUM(total_accesses) AS total_accesses_on_dram
    FROM dram_accesses_by_thread
    GROUP BY dram_id, cpu_id, interval_start, interval_end
),
dram_access_share_by_thread AS (
    SELECT
        a.dram_id, a.cpu_id, a.interval_start, a.interval_end, a.logical_entity_id,
        (a.total_accesses * 1.0 / NULLIF(t.total_accesses_on_dram, 0)) AS access_share
    FROM dram_accesses_by_thread a
    JOIN dram_accesses_total_on_dram t
      ON a.dram_id       = t.dram_id
     AND a.cpu_id        = t.cpu_id
     AND a.interval_start = t.interval_start
     AND a.interval_end   = t.interval_end
),

-- 4) Allocate per-DRAM active energy by access share; idle energy equally by PID count
dram_active_energy_per_thread_per_dram AS (
    SELECT
        s.dram_id, s.cpu_id, s.interval_start, s.interval_end, sh.logical_entity_id,
        COALESCE(sh.access_share, 0.0) * s.active_energy_j_total AS active_energy_per_dram_per_thread_j
    FROM dram_interval_energy_split s
    LEFT JOIN dram_access_share_by_thread sh
      ON s.dram_id = sh.dram_id
     AND s.cpu_id  = sh.cpu_id
     AND s.interval_start = sh.interval_start
     AND s.interval_end   = sh.interval_end
),
dram_idle_energy_per_thread_per_dram AS (
    SELECT
        s.dram_id, s.cpu_id, s.interval_start, s.interval_end, p.logical_entity_id,
        CASE WHEN ap.thread_count > 0
             THEN s.idle_energy_j_total * 1.0 / ap.thread_count
             ELSE 0.0
        END AS idle_energy_per_dram_per_thread_j
    FROM dram_interval_energy_split s
    JOIN active_threads_on_dram ap
      ON ap.dram_id = s.dram_id
     AND ap.cpu_id  = s.cpu_id
     AND ap.interval_start = s.interval_start
     AND ap.interval_end   = s.interval_end
    JOIN dram_threads_present_on_dram p
      ON p.dram_id = s.dram_id
     AND p.cpu_id  = s.cpu_id
     AND p.interval_start = s.interval_start
     AND p.interval_end   = s.interval_end
),

-- 5) Sum per-DRAM allocations to per-PID totals per interval
dram_energy_per_thread AS (
    SELECT
        COALESCE(a.logical_entity_id, i.logical_entity_id) AS logical_entity_id,
        COALESCE(a.interval_start, i.interval_start) AS interval_start,
        COALESCE(a.interval_end,   i.interval_end)   AS interval_end,
        SUM(COALESCE(a.active_energy_per_dram_per_thread_j, 0.0)) AS active_energy_j,
        SUM(COALESCE(i.idle_energy_per_dram_per_thread_j,   0.0)) AS idle_energy_j
    FROM dram_active_energy_per_thread_per_dram a
    FULL OUTER JOIN dram_idle_energy_per_thread_per_dram i
      ON a.logical_entity_id = i.logical_entity_id
     AND a.dram_id = i.dram_id
     AND a.cpu_id  = i.cpu_id
     AND a.interval_start = i.interval_start
     AND a.interval_end   = i.interval_end
    GROUP BY 1,2,3
)
SELECT
    interval_start,
    interval_end,
    logical_entity_id,
    active_energy_j + idle_energy_j AS total_energy_j,
    active_energy_j,
    idle_energy_j
FROM dram_energy_per_thread
ORDER BY interval_start, logical_entity_id;