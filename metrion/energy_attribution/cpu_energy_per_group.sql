SELECT
  e.interval_start,
  e.interval_end,
  le.cmdline,
  le.executable_name,
  le.resource_group_name,
  le.resource_group_id,
  SUM(e.total_energy_j) AS total_energy_j,
  SUM(e.active_energy_j) AS active_energy_j,
  SUM(e.idle_energy_j) AS idle_energy_j
FROM
  logical_entity le,
  total_energy_per_thread_view e
WHERE
  le.id = e.logical_entity_id
  AND le.cmdline IS NOT NULL
  {filter_condition}
GROUP BY
  le.cmdline,
  le.resource_group_name,
  le.executable_name,
  le.resource_group_id,
  e.interval_start,
  e.interval_end
ORDER BY
  e.interval_start,
  le.cmdline
