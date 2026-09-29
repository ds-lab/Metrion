#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include "common_bpf.h"

char LICENSE[] SEC("license") = "GPL";

#define PF_EXITING 0x00000004
/*
 * sched_switch_v2.c - Zero-loss entity matching via inline metadata
 *
 * Key insight: Instead of relying on a separate LogicalEntityRegistry BPF program,
 * we include full process metadata in the ring buffer event the FIRST time we see
 * a PID. This eliminates the race condition where sched_switch fires before
 * task_newtask is processed.
 *
 * Architecture:
 * - Track "seen PIDs" in a BPF hash map (pid_seen)
 * - First sched_switch for a PID: send EVENT_TYPE_WITH_METADATA
 * - Subsequent sched_switch: send EVENT_TYPE_DATA_ONLY (lightweight)
 * - Userspace creates LogicalEntity inline when it receives metadata events
 */

/* Perf event arrays, one entry per CPU; the loader sets max_entries */
struct {
  __uint(type, BPF_MAP_TYPE_PERF_EVENT_ARRAY);
  __type(key, u32);
  __type(value, u32);
} cycles_core SEC(".maps");
struct {
  __uint(type, BPF_MAP_TYPE_PERF_EVENT_ARRAY);
  __type(key, u32);
  __type(value, u32);
} aperf_core SEC(".maps");
struct {
  __uint(type, BPF_MAP_TYPE_PERF_EVENT_ARRAY);
  __type(key, u32);
  __type(value, u32);
} mperf_core SEC(".maps");
struct {
  __uint(type, BPF_MAP_TYPE_PERF_EVENT_ARRAY);
  __type(key, u32);
  __type(value, u32);
} load_local_perf SEC(".maps");
struct {
  __uint(type, BPF_MAP_TYPE_PERF_EVENT_ARRAY);
  __type(key, u32);
  __type(value, u32);
} load_remote_perf SEC(".maps");
struct {
  __uint(type, BPF_MAP_TYPE_PERF_EVENT_ARRAY);
  __type(key, u32);
  __type(value, u32);
} store_perf SEC(".maps");

/* Event type flags */
#define EVENT_TYPE_DATA_ONLY      0
#define EVENT_TYPE_WITH_METADATA  1

/*
 * Compact key: (pid, start_time_truncated) to track unique task instances
 * We truncate start_time to save space and avoid false negatives from
 * minor timing variations. Using 10ms buckets.
 */
struct pid_key {
  u32 pid;
  u32 start_bucket;  // start_time_ns / 10_000_000 (10ms buckets)
};

/* Tracks which (pid, start_bucket) combinations we've already seen. A task's key is removed at
 * its last switch, so this only holds live threads. */
struct {
  __uint(type, BPF_MAP_TYPE_HASH);
  __uint(max_entries, 65536);
  __type(key, struct pid_key);
  __type(value, u8);
} pid_seen SEC(".maps");

/*
 * Full event structure with embedded metadata for first-seen PIDs.
 * When event_type == EVENT_TYPE_DATA_ONLY, metadata fields are zeroed.
 */
struct sched_event_v2 {
  // ===== Monitoring Data (always present) =====
  u64 pid_start;                  // Thread start time (ns since boot)
  u64 time_start;                 // Kernel timestamp in nanoseconds
  u64 aperf_start;                // Actual Performance Register
  u64 mperf_start;                // Maximum Performance Register
  u64 unhalted_core_cycles_start; // Core cycles while not halted
  u64 loads_local_start;          // Data reads from local dram
  u64 loads_remote_start;         // Data reads from remote dram
  u64 stores_start;               // Data writes

  u64 time_end;                   // Kernel timestamp in nanoseconds
  u64 aperf_end;                  // Actual Performance Register
  u64 mperf_end;                  // Maximum Performance Register
  u64 unhalted_core_cycles_end;   // Core cycles while not halted
  u64 loads_local_end;            // Data reads from local dram
  u64 loads_remote_end;           // Data reads from remote dram
  u64 stores_end;                 // Data writes

  u32 pid;                        // Thread ID
  u32 tgid;                       // Thread Group ID (Process ID)
  u32 cpu_core;                   // CPU core number
  u8  event_type;                 // 0 = data only, 1 = with metadata
  u8  __pad[3];                   // Padding for alignment

  // ===== Metadata (only present when event_type == 1) =====
  u64 tgid_start;                 // TGID (process) start time
  u64 parent_tid_start;           // Parent thread start time
  u64 parent_tgid_start;          // Parent process start time
  u64 cgroup_id;                  // Control group identifier

  u32 parent_tid;                 // Parent thread ID
  u32 parent_tgid;                // Parent process ID
  u32 uid;                        // User ID
  u32 gid;                        // Group ID

  char comm[TASK_COMM_LEN];       // Process command name
  char cmdline[CMDLINE_MAX_LEN];  // Command line arguments
};

/* Per CPU array for tracking the currently running task's start snapshot */
struct start_snapshot {
  u64 pid_start;
  u64 time_start;
  u64 aperf_start;
  u64 mperf_start;
  u64 unhalted_core_cycles_start;
  u64 loads_local_start;
  u64 loads_remote_start;
  u64 stores_start;
  u32 pid;
  u32 tgid;
};

struct {
  __uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
  __uint(max_entries, 1);
  __type(key, u32);
  __type(value, struct start_snapshot);
} cpu_snapshot SEC(".maps");

struct {
  __uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
  __uint(max_entries, 1);
  __type(key, u32);
  __type(value, u64);
} rb_drops SEC(".maps");

struct {
  __uint(type, BPF_MAP_TYPE_RINGBUF);
  __uint(max_entries, 4096);  // in bytes, the loader sets it from ebpf.ringbuf_mib
} events SEC(".maps");

static __always_inline void drop_inc(void) {
  u32 k = 0;
  u64 *d = bpf_map_lookup_elem(&rb_drops, &k);
  if (d) (*d)++;
}

/*
 * Check if we've seen this (pid, start_time) before.
 * Returns 1 if first time (and marks as seen), 0 if already seen.
 */
static __always_inline int mark_first_seen(u32 pid, u64 start_ns) {
  struct pid_key key = {};
  key.pid = pid;
  key.start_bucket = (u32)(start_ns / 10000000);  // 10ms bucket

  u8 *seen = bpf_map_lookup_elem(&pid_seen, &key);
  if (seen) {
    return 0;  // Already seen
  }

  // Mark as seen
  u8 one = 1;
  bpf_map_update_elem(&pid_seen, &key, &one, BPF_ANY);
  return 1;  // First time
}

static __always_inline void forget_seen(u32 pid, u64 start_ns) {
  struct pid_key key = {};
  key.pid = pid;
  key.start_bucket = (u32)(start_ns / 10000000);
  bpf_map_delete_elem(&pid_seen, &key);
}

/*
 * Collect and embed metadata into the event.
 * Called only for first-seen PIDs.
 */
static __always_inline void collect_metadata_inline(
    struct sched_event_v2 *evt,
    struct task_struct *task,
    const char *comm
) {
  evt->event_type = EVENT_TYPE_WITH_METADATA;

  // Get TGID start time from group leader
  struct task_struct *group_leader = NULL;
  bpf_probe_read_kernel(&group_leader, sizeof(group_leader), &task->group_leader);
  if (group_leader) {
    bpf_probe_read_kernel(&evt->tgid_start, sizeof(evt->tgid_start), &group_leader->start_boottime);
  } else {
    evt->tgid_start = evt->pid_start;
  }

  // Get parent info
  struct task_struct *parent = NULL;
  bpf_probe_read_kernel(&parent, sizeof(parent), &task->real_parent);
  if (parent) {
    bpf_probe_read_kernel(&evt->parent_tid, sizeof(evt->parent_tid), &parent->pid);
    bpf_probe_read_kernel(&evt->parent_tgid, sizeof(evt->parent_tgid), &parent->tgid);
    bpf_probe_read_kernel(&evt->parent_tid_start, sizeof(evt->parent_tid_start), &parent->start_boottime);

    struct task_struct *parent_leader = NULL;
    bpf_probe_read_kernel(&parent_leader, sizeof(parent_leader), &parent->group_leader);
    if (parent_leader) {
      bpf_probe_read_kernel(&evt->parent_tgid_start, sizeof(evt->parent_tgid_start), &parent_leader->start_boottime);
    } else {
      evt->parent_tgid_start = evt->parent_tid_start;
    }
  }

  // User/Group/Cgroup
  u64 uid_gid = bpf_get_current_uid_gid();
  evt->uid = (u32)uid_gid;
  evt->gid = (u32)(uid_gid >> 32);
  evt->cgroup_id = bpf_get_current_cgroup_id();

  // Comm and cmdline
  bpf_probe_read_kernel_str(evt->comm, sizeof(evt->comm), comm);
  read_cmdline_from_task(group_leader, evt->cmdline, comm);
}

SEC("tracepoint/sched/sched_switch")
int sched_switch(struct trace_event_raw_sched_switch *args)
{
  u64 current_time = bpf_ktime_get_boot_ns();  // CLOCK_BOOTTIME: includes suspend time, matches boot_epoch_s() + offset
  u32 cpu_core = bpf_get_smp_processor_id();

  // Read performance counters
  s64 cycles_tmp = bpf_perf_event_read(&cycles_core, cpu_core);
  s64 aperf_tmp = bpf_perf_event_read(&aperf_core, cpu_core);
  s64 mperf_tmp = bpf_perf_event_read(&mperf_core, cpu_core);
  s64 loads_local_tmp = bpf_perf_event_read(&load_local_perf, cpu_core);
  s64 loads_remote_tmp = bpf_perf_event_read(&load_remote_perf, cpu_core);
  s64 stores_tmp = bpf_perf_event_read(&store_perf, cpu_core);

  u64 cycles = (cycles_tmp >= 0) ? cycles_tmp : 0;
  u64 aperf = (aperf_tmp >= 0) ? aperf_tmp : 0;
  u64 mperf = (mperf_tmp >= 0) ? mperf_tmp : 0;
  u64 loads_local = (loads_local_tmp >= 0) ? loads_local_tmp : 0;
  u64 loads_remote = (loads_remote_tmp >= 0) ? loads_remote_tmp : 0;
  u64 stores = (stores_tmp >= 0) ? stores_tmp : 0;

  u32 key = 0;
  struct start_snapshot *snap = bpf_map_lookup_elem(&cpu_snapshot, &key);

  // --- Handle outgoing task (prev) END INTERVAL ---
  // outgoing task is current task
  u32 prev_pid = args->prev_pid;
  if (prev_pid > 0 && snap && snap->pid == prev_pid)
  {
    struct task_struct *task = (struct task_struct *)bpf_get_current_task();
    u64 task_start = 0;
    bpf_probe_read_kernel(&task_start, sizeof(task_start), &task->start_boottime);

    u64 pid_tgid = bpf_get_current_pid_tgid();
    u32 tgid = (u32)(pid_tgid >> 32);

    // Check if this is a first-seen PID
    int is_first = mark_first_seen(prev_pid, task_start);

    // Reserve ring buffer space
    struct sched_event_v2 *out = bpf_ringbuf_reserve(&events, sizeof(struct sched_event_v2), 0);
    if (!out) {
      drop_inc();
      // The metadata went down with this event: forget the task so the next switch sends it again.
      if (is_first) {
        forget_seen(prev_pid, task_start);
      }
    } else {
      // Fill monitoring data (always present)
      out->pid_start = task_start;
      out->time_start = snap->time_start;
      out->aperf_start = snap->aperf_start;
      out->mperf_start = snap->mperf_start;
      out->unhalted_core_cycles_start = snap->unhalted_core_cycles_start;
      out->loads_local_start = snap->loads_local_start;
      out->loads_remote_start = snap->loads_remote_start;
      out->stores_start = snap->stores_start;

      out->time_end = current_time;
      out->aperf_end = aperf;
      out->mperf_end = mperf;
      out->unhalted_core_cycles_end = cycles;
      out->loads_local_end = loads_local;
      out->loads_remote_end = loads_remote;
      out->stores_end = stores;

      out->pid = prev_pid;
      out->tgid = tgid;
      out->cpu_core = cpu_core;
      out->event_type = EVENT_TYPE_DATA_ONLY;
      out->__pad[0] = 0;
      out->__pad[1] = 0;
      out->__pad[2] = 0;

      // Zero metadata fields by default
      out->tgid_start = 0;
      out->parent_tid_start = 0;
      out->parent_tgid_start = 0;
      out->cgroup_id = 0;
      out->parent_tid = 0;
      out->parent_tgid = 0;
      out->uid = 0;
      out->gid = 0;
      out->comm[0] = '\0';
      out->cmdline[0] = '\0';

      // If first-seen, collect and embed metadata
      if (is_first) {
        collect_metadata_inline(out, task, args->prev_comm);
      }

      bpf_ringbuf_submit(out, 0);
    }

    // A dying task switches out one last time after sched_process_exit, so its key is dropped
    // here; dropped any earlier, that last switch would mark it as seen again for good.
    u32 flags = 0;
    bpf_probe_read_kernel(&flags, sizeof(flags), &task->flags);
    if (flags & PF_EXITING) {
      forget_seen(prev_pid, task_start);
    }
  }

  // --- Handle incoming task (next) START INTERVAL ---
  u32 next_pid = args->next_pid;
  if (next_pid > 0 && snap) {
    snap->pid = next_pid;
    snap->time_start = current_time;
    snap->aperf_start = aperf;
    snap->mperf_start = mperf;
    snap->unhalted_core_cycles_start = cycles;
    snap->loads_local_start = loads_local;
    snap->loads_remote_start = loads_remote;
    snap->stores_start = stores;
  }

  return 0;
}
