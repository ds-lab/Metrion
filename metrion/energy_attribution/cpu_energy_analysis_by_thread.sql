-- materialize intervals

CREATE TABLE IF NOT EXISTS
    energy_interval_pkg_tmp AS
WITH
    energy_samples AS (
        SELECT
            m.id,
            m.value,
            m.metric_id,
            m.physical_entity_id AS package_id,
            m.logical_entity_id,
            m."start_ts",
            m."end_ts",
            -- The socket reports the value its counter wraps at the constant is what runs
            -- recorded before that value was read from sysfs.
            COALESCE(c.max_energy_range_j, 262143328850.0 / 1000000.0) AS energy_wrap_j
        FROM
            measurement m
            LEFT JOIN cpu c ON c.id = m.physical_entity_id
        WHERE
            m.metric_id = 'CPU_SOCKET_PKG_ENERGY'
            AND m."start_ts" >= '{start_time}'
            AND m."start_ts" <= '{end_time}'
    ),
    energy_interval AS (
        SELECT
            *
        FROM
            (
                SELECT
                    package_id,
                    "start_ts" AS interval_start,
                    lead ("start_ts") over w AS interval_end,
                    CASE
                        WHEN lead (value) over w IS NULL THEN 0
                        WHEN lead (value) over w >= value THEN lead (value) over w - value
                        ELSE (energy_wrap_j - value) + lead (value) over w
                    END AS energy_j
                FROM
                    energy_samples
                window
                    w AS (
                        PARTITION BY
                            package_id
                        ORDER BY
                            "start_ts"
                    )
            )
        WHERE
            interval_end IS NOT NULL
    )
SELECT
    package_id,
    interval_start,
    interval_end,
    energy_j
FROM
    energy_interval;

CREATE TABLE
    IF NOT EXISTS total_energy_per_thread_view AS
WITH
    work_per_thread_per_socket AS (
        SELECT
            ei.interval_start,
            ei.interval_end,
            m.logical_entity_id,
            e.parent_id AS package_id,
            SUM(m.value * CASE WHEN m."end_ts" = m."start_ts" THEN 1.0
                ELSE (epoch(LEAST(m."end_ts", ei.interval_end)) - epoch(GREATEST(m."start_ts", ei.interval_start)))
                   / (epoch(m."end_ts") - epoch(m."start_ts")) END) AS WORK
        FROM
            energy_interval_pkg_tmp ei,
            measurement m,
            entity e
        WHERE
            m."start_ts" < ei.interval_end
            AND m."end_ts" > ei.interval_start
            AND e.id = m.physical_entity_id
            AND e.parent_id = ei.package_id
            AND m.metric_id = 'CPU_CORE_PID_WORK'
            AND m.value IS NOT NULL
            AND Abs(m.value) < 1e307
        GROUP BY
            ei.interval_start,
            ei.interval_end,
            m.logical_entity_id,
            e.parent_id
    ), -- 4 Work per socket per time interval 
    work_per_socket AS (
        SELECT
            interval_start,
            interval_end,
            package_id,
            SUM(work) AS socket_work
        FROM
            work_per_thread_per_socket
        GROUP BY
            interval_start,
            interval_end,
            package_id
    ),
    -- 5 Execution time per PID & socket per time interval
    time_per_thread AS (
        SELECT
            ei.interval_start,
            ei.interval_end,
            m.logical_entity_id,
            e.parent_id AS package_id,
            SUM(m.value * CASE WHEN m."end_ts" = m."start_ts" THEN 1.0
                ELSE (epoch(LEAST(m."end_ts", ei.interval_end)) - epoch(GREATEST(m."start_ts", ei.interval_start)))
                   / (epoch(m."end_ts") - epoch(m."start_ts")) END) / 1000 AS exec_time_s -- convert ms to s
        FROM
            energy_interval_pkg_tmp ei,
            measurement m,
            entity e
        WHERE
            m.start_ts < ei.interval_end
            AND m.end_ts > ei.interval_start
            AND e.id = m.physical_entity_id
            AND e.parent_id = ei.package_id
            AND m.metric_id = 'CPU_CORE_PID_TIME'
            AND m.value IS NOT NULL
            AND Abs(m.value) < 1e307
        GROUP BY
            ei.interval_start,
            ei.interval_end,
            m.logical_entity_id,
            e.parent_id
    ),
    -- 6 Execution time socket per time interval
    time_per_socket AS (
        SELECT
            interval_start,
            interval_end,
            package_id,
            SUM(exec_time_s) AS socket_time_s
        FROM
            time_per_thread
        GROUP BY
            interval_start,
            interval_end,
            package_id
    ),
    -- 7 Merge metrics plus idle energy per PID and socket and time interval 
    activity_per_thread AS (
        SELECT
            w.interval_start,
            w.interval_end,
            w.logical_entity_id,
            w.package_id,
            w.work,
            ei.energy_j,
            t.exec_time_s,
            -- Capped at the measured energy, so an over-estimated idle power cannot leave a
            -- negative remainder for the threads to share.
            LEAST(
                COALESCE(c.idle_power_package, 0) * (
                    julian(w.interval_end) - julian(w.interval_start)
                ) * 24 * 60 * 60,
                ei.energy_j
            ) AS idle_energy_j
        FROM
            work_per_thread_per_socket w,
            energy_interval_pkg_tmp ei,
            time_per_thread t,
            cpu c
        WHERE
            w.interval_start = ei.interval_start
            AND w.interval_end = ei.interval_end
            AND w.package_id = ei.package_id
            AND w.package_id = t.package_id
            AND w.logical_entity_id = t.logical_entity_id
            AND w.interval_start = t.interval_start
            AND w.interval_end = t.interval_end
            AND c.id = w.package_id
    ),
    -- Energy attribution per PID and socket and time interval
    energy_attribution_per_thread AS (
        SELECT
            a.interval_start,
            a.interval_end,
            a.logical_entity_id,
            a.package_id,
            (a.work / ws.socket_work) * (a.energy_j - a.idle_energy_j) AS active_energy_j,
            (a.exec_time_s / ts.socket_time_s) * a.idle_energy_j AS idle_energy_j
        FROM
            activity_per_thread a,
            work_per_socket ws,
            time_per_socket ts
        WHERE
            a.interval_start = ws.interval_start
            AND a.interval_end = ws.interval_end
            AND a.package_id = ws.package_id
            AND a.interval_start = ts.interval_start
            AND a.interval_end = ts.interval_end
            AND a.package_id = ts.package_id
    ),
    -- 8 Final total energy per PID and time interval
    total_energy_per_thread AS (
        SELECT
            eap.interval_start,
            eap.interval_end,
            eap.logical_entity_id,
            SUM(eap.active_energy_j) + SUM(eap.idle_energy_j) AS total_energy_j,
            SUM(eap.active_energy_j) AS active_energy_j,
            SUM(eap.idle_energy_j) AS idle_energy_j
        FROM
            energy_attribution_per_thread eap
        GROUP BY
            eap.interval_start,
            eap.interval_end,
            eap.logical_entity_id
    )
SELECT
    interval_start,
    interval_end,
    logical_entity_id,
    total_energy_j,
    active_energy_j,
    idle_energy_j
FROM
    total_energy_per_thread
ORDER BY
    interval_start,
    logical_entity_id;
