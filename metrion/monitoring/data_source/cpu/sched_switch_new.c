#include "common_bpf.h"
/*
good documentation:
- https://linux-kernel-labs.github.io/refs/heads/master/lectures/processes.html
- https://docs.huihoo.com/doxygen/linux/kernel/3.7/structtask__struct.html

*/

/* Perf event arrays
 */
BPF_PERF_ARRAY(cycles_core, NUM_CPUS);
BPF_PERF_ARRAY(aperf_core, NUM_CPUS);
BPF_PERF_ARRAY(mperf_core, NUM_CPUS);
BPF_PERF_ARRAY(load_local_perf, NUM_CPUS);
BPF_PERF_ARRAY(load_remote_perf, NUM_CPUS);
BPF_PERF_ARRAY(store_perf, NUM_CPUS);

/*
 * Performance monitoring data snapshot for a specific process at a point in
 * time
 */
struct monitoring_data_slice
{
  // start values
  u64 pid_start;                  // Start time of the thread / process
  u64 time_start;                 // Kernel timestamp in nanoseconds
  u64 aperf_start;                // Actual Performance Register
  u64 mperf_start;                // Maximum Performance Register
  u64 unhalted_core_cycles_start; // Core cycles while not halted
  u64 loads_local_start;          // Data reads from local dram
  u64 loads_remote_start;         // Data reads from remote dram
  u64 stores_start;               // Data writes
  // end values
  u64 time_end;                   // Kernel timestamp in nanoseconds
  u64 aperf_end;                  // Actual Performance Register
  u64 mperf_end;                  // Maximum Performance Register
  u64 unhalted_core_cycles_end;   // Core cycles while not halted
  u64 loads_local_end;            // Data reads from local dram
  u64 loads_remote_end;           // Data reads from remote dram
  u64 stores_end;                 // Data writes

  // meta data
  u32 pid;                        // Process ID 
  u32 tgid;                       // Thread Group ID
  u32 cpu_core;                   // CPU core number
  u32 __pad;                      // Padding
};

// Per CPU Arry for the monitoring data
BPF_PERCPU_ARRAY(monitoring_data, struct monitoring_data_slice, 1);
// Count dropped events
BPF_PERCPU_ARRAY(rb_drops, u64, 1);

// Ringbuf
BPF_RINGBUF_OUTPUT(events, 524288);

/* Process metadata storage - Key: PID, Value: pid_metadata */
BPF_HASH(pid_to_metadata, u32, struct pid_metadata, 524288);

/* Enhanced helper to collect process metadata - called on sched_switch.
 * Uses lazy initialization: skips if metadata already exists.
 * Uses start_boottime for consistency with logical_entity_registry.
 */
static __always_inline void collect_pid_metadata(u32 pid, const char *comm)
{
  // Check if metadata already exists to avoid redundant work.
  if (pid_to_metadata.lookup(&pid))
  {
    return;
  }

  struct pid_metadata metadata = {};

  struct task_struct *task = (struct task_struct *)bpf_get_current_task();
  if (!task)
  {
    return;
  }

  bpf_probe_read_kernel(&metadata.tgid, sizeof(metadata.tgid), &task->tgid);
  // Use start_boottime for consistency with logical_entity_registry (CLOCK_BOOTTIME)
  bpf_probe_read_kernel(&metadata.start_time, sizeof(metadata.start_time), &task->start_boottime);

  struct task_struct *parent = NULL;
  bpf_probe_read_kernel(&parent, sizeof(parent), &task->real_parent);
  if (parent)
  {
    bpf_probe_read_kernel(&metadata.parent_tgid, sizeof(metadata.parent_tgid), &parent->tgid);
    bpf_probe_read_kernel(&metadata.parent_tid, sizeof(metadata.parent_tid), &parent->pid);
    bpf_probe_read_kernel(&metadata.start_time_parent, sizeof(metadata.start_time_parent), &parent->start_boottime);
    struct task_struct *parent_group_leader = NULL;
    bpf_probe_read_kernel(&parent_group_leader, sizeof(parent_group_leader), &parent->group_leader);
    if (parent_group_leader)
    {
      bpf_probe_read_kernel(&metadata.start_time_parent_tgid, sizeof(metadata.start_time_parent_tgid), &parent_group_leader->start_boottime);
    }
  }

  u64 uid_gid = bpf_get_current_uid_gid();
  metadata.uid = uid_gid & 0xFFFFFFFF;
  metadata.gid = uid_gid >> 32;

  struct task_struct *group_leader = NULL;
  bpf_probe_read_kernel(&group_leader, sizeof(group_leader), &task->group_leader);
  if (group_leader)
  {
    bpf_probe_read_kernel(&metadata.start_time_tgid, sizeof(metadata.start_time_tgid), &group_leader->start_boottime);
  }
  metadata.is_kernel_thread = 0;

  metadata.cgroup_id = bpf_get_current_cgroup_id();

  bpf_probe_read_kernel_str(metadata.comm, sizeof(metadata.comm), comm);

  // Read cmdline for current task
  read_cmdline_from_task(group_leader, metadata.cmdline, comm);

  pid_to_metadata.update(&pid, &metadata);
}

TRACEPOINT_PROBE(sched, sched_switch)
{
  // if ((bpf_get_prandom_u32() & 0xF) != 0)
  //   return 0;
  // general metadata
  u64 current_time = bpf_ktime_get_ns();
  u32 cpu_core = bpf_get_smp_processor_id();

  // performance counters
  s64 cycles_tmp = cycles_core.perf_read(cpu_core);
  s64 aperf_tmp = aperf_core.perf_read(cpu_core);
  s64 mperf_tmp = mperf_core.perf_read(cpu_core);
  s64 loads_local_tmp = load_local_perf.perf_read(cpu_core);
  s64 loads_remote_tmp = load_remote_perf.perf_read(cpu_core);
  s64 stores_tmp = store_perf.perf_read(cpu_core);

  u64 cycles = (cycles_tmp >= 0) ? cycles_tmp : 0;
  u64 aperf = (aperf_tmp >= 0) ? aperf_tmp : 0;
  u64 mperf = (mperf_tmp >= 0) ? mperf_tmp : 0;
  u64 loads_local = (loads_local_tmp >= 0) ? loads_local_tmp : 0;
  u64 loads_remote = (loads_remote_tmp >= 0) ? loads_remote_tmp : 0;
  u64 stores = (stores_tmp >= 0) ? stores_tmp : 0;

  // array access 
  u32 key = 0;
  // Reuse existing monitoring data slice
  struct monitoring_data_slice *s = monitoring_data.lookup(&key);

  // --- Handle outgoing task (prev) END INTERVAL ---
  // prev is the current task
  u32 prev_pid = args->prev_pid;
  if (prev_pid > 0)
  {
    u64 pid_tgid = bpf_get_current_pid_tgid();
    struct task_struct *task = (void *)bpf_get_current_task();
    u64 task_start = 0;
    // Use start_boottime to match /proc/stat and logical_entity_registry (CLOCK_BOOTTIME)
    bpf_probe_read_kernel(&task_start, sizeof(task_start), &task->start_boottime);
    u32 tgid = pid_tgid >> 32;
    if (s && s->pid == prev_pid)
    {
      // Reserve output record and fill it directly (no memcpy; no writing END into *s)
      struct monitoring_data_slice *out =
          events.ringbuf_reserve(sizeof(struct monitoring_data_slice));
      if (!out)
      {
        u64 *d = rb_drops.lookup(&key);
        if (d) (*d)++;
        // Do NOT return: still record the next start snapshot below.
      }
      else
      {
        // START snapshot from per-CPU storage
        out->pid_start                  = task_start;
        out->time_start                 = s->time_start;
        out->aperf_start                = s->aperf_start;
        out->mperf_start                = s->mperf_start;
        out->unhalted_core_cycles_start = s->unhalted_core_cycles_start;
        out->loads_local_start          = s->loads_local_start;
        out->loads_remote_start         = s->loads_remote_start;
        out->stores_start               = s->stores_start;

        // END snapshot from current reads
        out->time_end                   = current_time;
        out->aperf_end                  = aperf;
        out->mperf_end                  = mperf;
        out->unhalted_core_cycles_end   = cycles;
        out->loads_local_end            = loads_local;
        out->loads_remote_end           = loads_remote;
        out->stores_end                 = stores;

        // Metadata
        out->pid      = prev_pid;
        out->tgid     = tgid;
        out->cpu_core = cpu_core;
        out->__pad    = 0;

        events.ringbuf_submit(out, 0);
      }
    }
    // Always collect metadata for outgoing task (prev)
    collect_pid_metadata(prev_pid, args->prev_comm);
  }

  // --- Handle incoming task (next) ---
  u32 next_pid = args->next_pid;
  if (next_pid > 0)
  {
    if (s)
    {
      s->pid      = next_pid;
      s->cpu_core = cpu_core;

      s->time_start                 = current_time;
      s->aperf_start                = aperf;
      s->mperf_start                = mperf;
      s->unhalted_core_cycles_start = cycles;
      s->loads_local_start          = loads_local;
      s->loads_remote_start         = loads_remote;
      s->stores_start               = stores;
    }
  }
  return 0;
}