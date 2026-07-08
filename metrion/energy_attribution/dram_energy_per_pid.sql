SELECT
  e.interval_start,
  e.interval_end,
  le.native_pid,
  le.cmdline,
  SUM(e.total_energy_j) AS total_energy_j,
  SUM(e.active_energy_j) AS active_energy_j,
  SUM(e.idle_energy_j) AS idle_energy_j
FROM
  logical_entity le,
  dram_energy_per_thread_view e
WHERE
  le.id = e.logical_entity_id
  {filter_condition}
GROUP BY
  le.native_pid,
  le.cmdline,
  e.interval_start,
  e.interval_end
ORDER BY
  e.interval_start,
  le.native_pid
