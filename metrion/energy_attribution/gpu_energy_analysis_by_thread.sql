-- GPU Energy Attribution SQL Query
-- Creates energy intervals from GPU energy counter and attributes to processes
-- using SM (compute) utilization ratios.

-- 0) Per-GPU energy intervals with counter wrap handling
CREATE TABLE IF NOT EXISTS energy_interval_gpu_tmp AS
WITH energy_samples_gpu AS (
    SELECT
        m.id,
        m.value,
        m.metric_id,
        m.physical_entity_id AS gpu_id,
        m.logical_entity_id,
        m.start_ts,
        m.end_ts
    FROM measurement AS m
    WHERE
        m.metric_id = 'GPU_DEVICE_TOTAL_ENERGY'
        AND m.start_ts >= '{start_time}'
        AND m.start_ts <= '{end_time}'
        AND m.value IS NOT NULL
    ORDER BY m.start_ts
),
energy_interval_gpu AS (
    SELECT *
    FROM (
        SELECT
            gpu_id,
            start_ts AS interval_start,
            LEAD(start_ts) OVER w AS interval_end,
            CASE
                -- Energy counter wrap-around: 64-bit counter in mJ converted to J
                WHEN LEAD(value) OVER w >= value THEN LEAD(value) OVER w - value
                ELSE ((18446744073709551615.0 / 1000.0) - value) + LEAD(value) OVER w
            END AS energy_j
        FROM energy_samples_gpu
        WINDOW w AS (PARTITION BY gpu_id ORDER BY start_ts)
    )
    WHERE interval_end IS NOT NULL
)
SELECT gpu_id, interval_start, interval_end, energy_j
FROM energy_interval_gpu;

-- 1) Final view: attribute energy by SM utilization ratio
CREATE TABLE IF NOT EXISTS gpu_energy_per_thread_view AS
WITH
-- (A) Idle/dynamic split per GPU interval
gpu_interval_powers AS (
    SELECT
        ei.gpu_id,
        ei.interval_start,
        ei.interval_end,
        ei.energy_j,
        (julian(ei.interval_end) - julian(ei.interval_start)) * 24.0 * 3600.0 AS duration_s,
        g.idle_power
    FROM energy_interval_gpu_tmp AS ei
    JOIN gpu AS g ON ei.gpu_id = g.id
),
gpu_interval_energy_split AS (
    SELECT
        gpu_id,
        interval_start,
        interval_end,
        energy_j,
        idle_power,
        duration_s,
        -- Capped at the measured energy, so an over-estimated idle power cannot leave a
        -- negative remainder for the processes to share.
        LEAST(COALESCE(idle_power, 0) * duration_s, energy_j) AS idle_energy_j_total,
        energy_j - idle_energy_j_total AS dynamic_energy_j_total
    FROM gpu_interval_powers
),

-- (B) Per-process SM utilization within each energy interval
gpu_util_by_thread AS (
    SELECT
        ei.gpu_id,
        ei.interval_start,
        ei.interval_end,
        m.logical_entity_id,
        SUM(m.value * CASE WHEN m.end_ts = m.start_ts THEN 1.0
            ELSE (epoch(LEAST(m.end_ts, ei.interval_end)) - epoch(GREATEST(m.start_ts, ei.interval_start)))
               / (epoch(m.end_ts) - epoch(m.start_ts)) END) AS total_util,
        SUM(CASE WHEN m.end_ts = m.start_ts THEN 1.0
            ELSE (epoch(LEAST(m.end_ts, ei.interval_end)) - epoch(GREATEST(m.start_ts, ei.interval_start)))
               / (epoch(m.end_ts) - epoch(m.start_ts)) END) AS sample_count
    FROM energy_interval_gpu_tmp AS ei
    JOIN measurement AS m
      ON m.physical_entity_id = ei.gpu_id
     AND m.metric_id = 'GPU_DEVICE_PID_COMPUTE_UTILIZATION'
     AND m.value IS NOT NULL
     AND ABS(m.value) < 1e307
     AND m.start_ts < ei.interval_end
     AND m.end_ts > ei.interval_start
    GROUP BY ei.gpu_id, ei.interval_start, ei.interval_end, m.logical_entity_id
),

-- (C) Total utilization per GPU per interval (for ratio calculation)
gpu_util_total AS (
    SELECT
        gpu_id,
        interval_start,
        interval_end,
        SUM(total_util) AS total_util_all,
        SUM(sample_count) AS total_samples
    FROM gpu_util_by_thread
    GROUP BY gpu_id, interval_start, interval_end
),

-- (D) Utilization share per PID
gpu_util_share_by_thread AS (
    SELECT
        u.gpu_id,
        u.interval_start,
        u.interval_end,
        u.logical_entity_id,
        u.total_util,
        u.sample_count,
        t.total_util_all,
        t.total_samples,
        CASE
            WHEN t.total_util_all > 0 THEN (u.total_util / t.total_util_all)
            ELSE 0.0
        END AS util_share,
        CASE
            WHEN t.total_samples > 0 THEN (u.sample_count / t.total_samples)
            ELSE 0.0
        END AS time_share
    FROM gpu_util_by_thread u
    JOIN gpu_util_total t
      ON u.gpu_id = t.gpu_id
     AND u.interval_start = t.interval_start
     AND u.interval_end = t.interval_end
),

-- (E) Attribute dynamic energy by utilization, idle by time share
gpu_dynamic_energy_per_thread AS (
    SELECT
        s.gpu_id,
        s.interval_start,
        s.interval_end,
        sh.logical_entity_id,
        COALESCE(sh.util_share, 0.0) * s.dynamic_energy_j_total AS dynamic_energy_j
    FROM gpu_interval_energy_split s
    LEFT JOIN gpu_util_share_by_thread sh
      ON s.gpu_id = sh.gpu_id
     AND s.interval_start = sh.interval_start
     AND s.interval_end = sh.interval_end
),
gpu_idle_energy_per_thread AS (
    SELECT
        s.gpu_id,
        s.interval_start,
        s.interval_end,
        sh.logical_entity_id,
        COALESCE(sh.time_share, 0.0) * s.idle_energy_j_total AS idle_energy_j
    FROM gpu_interval_energy_split s
    LEFT JOIN gpu_util_share_by_thread sh
      ON s.gpu_id = sh.gpu_id
     AND s.interval_start = sh.interval_start
     AND s.interval_end = sh.interval_end
),

-- (F) Combine dynamic and idle energy per PID
gpu_energy_per_thread AS (
    SELECT
        COALESCE(d.logical_entity_id, i.logical_entity_id) AS logical_entity_id,
        COALESCE(d.gpu_id, i.gpu_id) AS gpu_id,
        COALESCE(d.interval_start, i.interval_start) AS interval_start,
        COALESCE(d.interval_end, i.interval_end) AS interval_end,
        COALESCE(d.dynamic_energy_j, 0.0) AS dynamic_energy_j,
        COALESCE(i.idle_energy_j, 0.0) AS idle_energy_j
    FROM gpu_dynamic_energy_per_thread d
    FULL OUTER JOIN gpu_idle_energy_per_thread i
      ON d.logical_entity_id = i.logical_entity_id
     AND d.gpu_id = i.gpu_id
     AND d.interval_start = i.interval_start
     AND d.interval_end = i.interval_end
)

SELECT
    interval_start,
    interval_end,
    logical_entity_id,
    gpu_id,
    dynamic_energy_j + idle_energy_j AS total_energy_j,
    dynamic_energy_j as active_energy_j,
    idle_energy_j
FROM gpu_energy_per_thread
ORDER BY interval_start, logical_entity_id;
