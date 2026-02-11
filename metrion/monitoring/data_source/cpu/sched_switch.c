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
// BPF_PERF_ARRAY(cache_miss, NUM_CPUS);
BPF_PERF_ARRAY(load_local_perf, NUM_CPUS);
BPF_PERF_ARRAY(load_remote_perf, NUM_CPUS);
BPF_PERF_ARRAY(store_perf, NUM_CPUS);

/*
 * Interval type enumeration for start/end monitoring points
 */
enum interval_type
{
  INTERVAL_START = 0,
  INTERVAL_END = 1
};


/*
 * Performance monitoring data snapshot for a specific process at a point in
 * time
 */
struct monitoring_data_slice
{
  u64 time_start;                 // Kernel timestamp in nanoseconds
  u64 aperf_start;                // Actual Performance Register
  u64 mperf_start;                // Maximum Performance Register
  u64 unhalted_core_cycles_start; // Core cycles while not halted
  u64 cache_miss_start;           // L2 cache request misses
  u64 loads_local_start;          // Data reads from local dram
  u64 loads_remote_start;         // Data reads from remote dram
  u64 stores_start;               // Data writes
  u64 time_end;                   // Kernel timestamp in nanoseconds
  u64 aperf_end;                  // Actual Performance Register
  u64 mperf_end;                  // Maximum Performance Register
  u64 unhalted_core_cycles_end;   // Core cycles while not halted
  u64 cache_miss_end;             // L2 cache request misses
  u64 loads_local_end;            // Data reads from local dram
  u64 loads_remote_end;           // Data reads from remote dram
  u64 stores_end;                 // Data writes
};

/*
 * Composite key structure for nested hierarchy:
 * PID -> execution_count -> cpu_core -> interval_type (start/end)
 */
struct monitoring_key
{
  u64 exec_count; // Execution count for this PID on this cpu core
  u32 pid;        // Process ID
  u32 cpu_core;   // CPU core number
};

struct pid_exec_count_key
{
  u32 pid;      // Process ID
  u32 cpu_core; // CPU core number
};

/*
 * Hash maps for process metadata and execution tracking
 */

/* Process metadata storage - Key: PID, Value: pid_metadata */
BPF_HASH(pid_to_metadata, u32, struct pid_metadata, 524288);

/* Process execution counter - Key: PID, Value: execution count */
BPF_HASH(pid_exec_count, struct pid_exec_count_key, u64, 524288);

/*
 * Double-buffered hash maps for atomic snapshot reading
 * These provide the same nested structure but with separate buffers
 */

/* Primary monitoring data buffer with composite key structure */
BPF_HASH(pid_monitoring_data_buf_1, struct monitoring_key,
         struct monitoring_data_slice, 8388608);

/* Secondary monitoring data buffer with composite key structure */
BPF_HASH(pid_monitoring_data_buf_2, struct monitoring_key,
         struct monitoring_data_slice, 8388608);

BPF_HASH(pid_buffer, struct monitoring_key,
         u32, 8388608);

/* Indicating which buffer is currently being used */
BPF_ARRAY(buffer, u32, 1);
BPF_HASH(debug_counters, u32, u64);

/*Map Lengths*/
BPF_ARRAY(__pid_monitoring_data_buf_len__, u64, 2);

/* Indicate if there is still writing to a map */
BPF_ARRAY(inflight, u64, 2);
BPF_ARRAY(inserts, u64, 4);


/* Return current counter value (initialise to 0 if absent) */
static __always_inline u64 get_seq(u32 pid, u32 cpu_core)
{
  u64 zero = 0;
  struct pid_exec_count_key key = {}; // <- zero everything
  key.pid = pid;
  key.cpu_core = cpu_core;
  u64 *p = pid_exec_count.lookup_or_try_init(&key, &zero);
  return p ? *p : 0;
}

static __always_inline void ctr_inc(enum insert_ctr which)
{
  u64 *p = inserts.lookup((u32 *)&which);
  if (p)
    __sync_fetch_and_add(p, 1);
}

static __always_inline void dbg_bump(u32 id)
{
  u64 one = 1, *p = debug_counters.lookup_or_try_init(&id, &one);
  if (p)
    __sync_fetch_and_add(p, 1);
}

/* Atomically ++counter after the END record has been written */
static __always_inline void inc_seq(u32 pid, u32 cpu_core)
{
  struct pid_exec_count_key key = {};
  key.pid = pid;
  key.cpu_core = cpu_core;
  u64 *p = pid_exec_count.lookup(&key);
  if (p)
    __sync_fetch_and_add(p, 1);
}

static __always_inline void inflight_inc(u32 i)
{
  u64 *p = inflight.lookup(&i);
  if (p)
    __sync_fetch_and_add(p, 1);
}

static __always_inline void inflight_dec(u32 i)
{
  u64 *p = inflight.lookup(&i);
  if (p)
    __sync_fetch_and_add(p, (u64)-1);
}

static __always_inline void buf_len_inc(u32 buf)
{
  u64 *p = __pid_monitoring_data_buf_len__.lookup(&buf);
  if (p)
    __sync_fetch_and_add(p, 1);
}


/* Enhanced helper to collect process metadata*/
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
  bpf_probe_read_kernel(&metadata.start_time, sizeof(metadata.start_time), &task->start_time);

  struct task_struct *parent = NULL;
  bpf_probe_read_kernel(&parent, sizeof(parent), &task->real_parent);
  if (parent)
  {
    bpf_probe_read_kernel(&metadata.parent_tgid, sizeof(metadata.parent_tgid), &parent->tgid);
    bpf_probe_read_kernel(&metadata.parent_tid, sizeof(metadata.parent_tid), &parent->pid);
    bpf_probe_read_kernel(&metadata.start_time_parent, sizeof(metadata.start_time_parent), &parent->start_time);
    struct task_struct *parent_group_leader = NULL;
    bpf_probe_read_kernel(&parent_group_leader, sizeof(parent_group_leader), &parent->group_leader);
    if (parent_group_leader)
    {
      bpf_probe_read_kernel(&metadata.start_time_parent_tgid, sizeof(metadata.start_time_parent_tgid), &parent_group_leader->start_time);
    }
  }

  u64 uid_gid = bpf_get_current_uid_gid();
  metadata.uid = uid_gid & 0xFFFFFFFF;
  metadata.gid = uid_gid >> 32;

  struct task_struct *group_leader = NULL;
  bpf_probe_read_kernel(&group_leader, sizeof(group_leader), &task->group_leader);
  if (group_leader)
  {
    bpf_probe_read_kernel(&metadata.start_time_tgid, sizeof(metadata.start_time_tgid), &group_leader->start_time);
  }
  void *mm;
  // kernel task do not have memory
  bpf_probe_read_kernel(&mm, sizeof(mm), &task->mm);
  metadata.is_kernel_thread = (mm == NULL) ? 1 : 0;

  metadata.cgroup_id = bpf_get_current_cgroup_id();

  bpf_probe_read_kernel_str(metadata.comm, sizeof(metadata.comm), comm);

  // Read cmdline for current task
  read_cmdline_from_task(group_leader, metadata.cmdline, comm);

  pid_to_metadata.update(&pid, &metadata);
}

TRACEPOINT_PROBE(sched, sched_switch)
{
  u64 current_time = bpf_ktime_get_ns();
  u32 cpu_core = bpf_get_smp_processor_id();
  u32 buf_idx = 0;
  long res = 0;

  s64 cycles_tmp = cycles_core.perf_read(cpu_core);
  s64 aperf_tmp = aperf_core.perf_read(cpu_core);
  s64 mperf_tmp = mperf_core.perf_read(cpu_core);
  // s64 cache_miss_tmp = cache_miss.perf_read(cpu_core);
  s64 loads_local_tmp = load_local_perf.perf_read(cpu_core);
  s64 loads_remote_tmp = load_remote_perf.perf_read(cpu_core);
  s64 stores_tmp = store_perf.perf_read(cpu_core);

  u64 cycles = (cycles_tmp >= 0) ? cycles_tmp : 0;
  u64 aperf = (aperf_tmp >= 0) ? aperf_tmp : 0;
  u64 mperf = (mperf_tmp >= 0) ? mperf_tmp : 0;
  u64 cache_miss = 0; // (cache_miss_tmp >= 0) ? cache_miss_tmp : 0;
  u64 loads_local = (loads_local_tmp >= 0) ? loads_local_tmp : 0;
  u64 loads_remote = (loads_remote_tmp >= 0) ? loads_remote_tmp : 0;
  u64 stores = (stores_tmp >= 0) ? stores_tmp : 0;

  // --- Handle outgoing task (prev) END INTERVAL ---
  // prev is the current task
  u32 prev_pid = args->prev_pid;
  if (prev_pid > 0)
  {
    u64 seq = get_seq(prev_pid, cpu_core);
    struct monitoring_key prev_key = {};
    prev_key.pid = prev_pid;
    prev_key.exec_count = seq;
    prev_key.cpu_core = cpu_core;

    u32 *pendbuf = pid_buffer.lookup(&prev_key);
    if (pendbuf)
    {
      u32 end_buf_val = *pendbuf;
      ctr_inc(END_ATTEMPT);
      struct monitoring_data_slice *s = (end_buf_val == 0)
                                            ? pid_monitoring_data_buf_1.lookup(&prev_key)
                                            : pid_monitoring_data_buf_2.lookup(&prev_key);

      if (s)
      {
        s->time_end = current_time;
        s->aperf_end = aperf;
        s->mperf_end = mperf;
        s->unhalted_core_cycles_end = cycles;
        s->cache_miss_end = cache_miss;
        s->loads_local_end = loads_local;
        s->loads_remote_end = loads_remote;
        s->stores_end = stores;
        ctr_inc(END_OK);
      }
      else
      {
        dbg_bump(DBG_END_LOOKUP_MISS);
      }
      inc_seq(prev_pid, cpu_core);
      inflight_dec(end_buf_val);
      pid_buffer.delete(&prev_key);
    }
    else
    {
      dbg_bump(0);
    }
    // Always try to collect metadata for current task.
    collect_pid_metadata(prev_pid, args->prev_comm);
  }

  // --- Handle incoming task (next) ---
  u32 next_pid = args->next_pid;
  if (next_pid > 0)
  {
    // 1. Atomically grab the current buffer index and immediately increment its inflight counter.
    u32 current_buf_val = 0;
    u32 *pbuf = buffer.lookup(&buf_idx);
    if (pbuf)
    {
      current_buf_val = *pbuf;
    }
    if (current_buf_val != 2) // capture only data if not stopped
    {
      inflight_inc(current_buf_val);

      // 2. Prepare the data that we will attempt to write.
      u64 seq = get_seq(next_pid, cpu_core);

      struct monitoring_key next_key = {};
      next_key.pid = next_pid;
      next_key.exec_count = seq;
      next_key.cpu_core = cpu_core;

      struct monitoring_data_slice data = {
          .time_start = current_time,
          .aperf_start = aperf,
          .mperf_start = mperf,
          .unhalted_core_cycles_start = cycles,
          .cache_miss_start = cache_miss,
          .loads_local_start = loads_local,
          .loads_remote_start = loads_remote,
          .stores_start = stores,
      };

      // 3. Attempt to write the data.
      if (current_buf_val == 0)
      {
        res = pid_monitoring_data_buf_1.update(&next_key, &data);
      }
      else
      {
        res = pid_monitoring_data_buf_2.update(&next_key, &data);
      }

      // 4. Check if the write succeeded. If so, also write the hint. If not, rollback the inflight counter.
      if (res == 0)
      {
        // Only update the pid_buffer if the main data update was successful
        if (pid_buffer.update(&next_key, &current_buf_val) == 0)
        {
          ctr_inc(START_OK);
          buf_len_inc(current_buf_val);
        }
        else
        {
          // If this second update fails, we have an incomplete start.
          // We must roll back the inflight counter.
          inflight_dec(current_buf_val);
          dbg_bump(DBG_HINT_UPDATE_FAIL);
          ctr_inc(START_FAIL);
          // And delete the entry from the main buffer to avoid a leak
          if (current_buf_val == 0)
          {
            pid_monitoring_data_buf_1.delete(&next_key);
          }
          else
          {
            pid_monitoring_data_buf_2.delete(&next_key);
          }
        }
      }
      else
      {
        // Rollback: The data write failed, so we must decrement the counter we optimistically incremented.
        inflight_dec(current_buf_val);
        dbg_bump(DBG_START_UPDATE_FAIL);
        ctr_inc(START_FAIL);
      }
    }
  }
  return 0;
}