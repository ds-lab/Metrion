#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include "common_bpf.h"

char LICENSE[] SEC("license") = "GPL";

#ifndef TASK_COMM_LEN
#define TASK_COMM_LEN 16
#endif

#ifndef CMDLINE_MAX_LEN
#define CMDLINE_MAX_LEN 128
#endif

#ifndef CLONE_THREAD
#define CLONE_THREAD 0x00010000
#endif

enum event_type {
    EVENT_TYPE_NEW    = 1,
    EVENT_TYPE_EXEC   = 2,
    EVENT_TYPE_RENAME = 3,
    EVENT_TYPE_EXIT   = 4,
};

struct task_event_t {
    u32 event_type;

    u32 tid;
    u32 tgid;
    u64 tid_start_ns;   // creation time for this tid in boot-ns domain
    u64 tgid_start_ns;

    u32 parent_tid;
    u32 parent_tgid;
    u64 parent_tid_start_ns;
    u64 parent_tgid_start_ns;

    u32 uid;
    u32 gid;
    u64 cgroup_id;

    u32 old_tid;  // PID before migration (for EXEC events)
    u32 __pad;    // Padding for alignment

    char comm[TASK_COMM_LEN]; // current/new comm
    char cmdline[CMDLINE_MAX_LEN]; // command line arguments
};

struct comm_t {
    char comm[TASK_COMM_LEN];
};

struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 1024 * 4096);  // 1024 pages
} events SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 262144);
    __type(key, u32);
    __type(value, u64);
} tid_start_ns SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 262144);
    __type(key, u32);
    __type(value, u32);
} tid_tgid SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 262144);
    __type(key, u32);
    __type(value, struct comm_t);
} tid_comm SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
    __uint(max_entries, 1);
    __type(key, u32);
    __type(value, u64);
} rb_drops SEC(".maps");

static __always_inline void inc_drop(void)
{
    u32 k = 0;
    u64 *v = bpf_map_lookup_elem(&rb_drops, &k);
    if (v) (*v)++;
}

static __always_inline u64 lookup_start(u32 tid)
{
    u64 *p = bpf_map_lookup_elem(&tid_start_ns, &tid);
    return p ? *p : 0;
}

static __always_inline void cache_comm(u32 tid, const char *src_comm)
{
    struct comm_t c = {};
    bpf_probe_read_str(c.comm, sizeof(c.comm), src_comm);
    bpf_map_update_elem(&tid_comm, &tid, &c, BPF_ANY);
}

/*
 * task:task_newtask
 * Fields commonly: pid, comm, clone_flags
 * See kernel trace/events/task.h. :contentReference[oaicite:7]{index=7}
 */
SEC("tracepoint/task/task_newtask")
int task_newtask(struct trace_event_raw_task_newtask *args)
{
    struct task_event_t e = {};
    e.event_type = EVENT_TYPE_NEW;

    e.tid = (u32)args->pid;

    e.tid_start_ns = bpf_ktime_get_boot_ns();

    /* creator is current */
    u64 pt = bpf_get_current_pid_tgid();
    e.parent_tid  = (u32)(pt & 0xFFFFFFFF);
    e.parent_tgid = (u32)(pt >> 32);

    /* parent start times (if available in map) */
    e.parent_tid_start_ns  = lookup_start(e.parent_tid);
    e.parent_tgid_start_ns = lookup_start(e.parent_tgid);

    /* derive child tgid */
    if (args->clone_flags & CLONE_THREAD){
        e.tgid = e.parent_tgid;
        e.tgid_start_ns = e.parent_tgid_start_ns;
    }
    else{
        e.tgid = e.tid;
        e.tgid_start_ns = e.tid_start_ns;
    }

    /* comm at fork/clone time */
    bpf_probe_read_str(e.comm, sizeof(e.comm), args->comm);

    /* Read cmdline from parent (current task during fork) */
    struct task_struct *task = (struct task_struct *)bpf_get_current_task();
    read_cmdline_from_task(task, e.cmdline, e.comm);

    /* metadata */
    u64 uid_gid = bpf_get_current_uid_gid();
    e.uid = (u32)uid_gid;
    e.gid = (u32)(uid_gid >> 32);
    e.cgroup_id = bpf_get_current_cgroup_id();

    /* update kernel caches */
    bpf_map_update_elem(&tid_start_ns, &e.tid, &e.tid_start_ns, BPF_ANY);
    bpf_map_update_elem(&tid_tgid, &e.tid, &e.tgid, BPF_ANY);
    cache_comm(e.tid, e.comm);

    int ret = bpf_ringbuf_output(&events, &e, sizeof(e), 0);
    if (ret) inc_drop();
    return 0;
}

/*
 * sched:sched_process_exec
 * Tracepoint provides filename (as __data_loc), pid, old_pid
 */
SEC("tracepoint/sched/sched_process_exec")
int sched_process_exec(struct trace_event_raw_sched_process_exec *args)
{
    struct task_event_t e = {};
    e.event_type = EVENT_TYPE_EXEC;

    u32 pid = (u32)args->pid;
    u32 old = (u32)args->old_pid;

    e.tid = pid;
    e.old_tid = old;  // Capture old PID for migration tracking

    /* TGID after exec: read from current */
    u64 pt = bpf_get_current_pid_tgid();
    e.tgid = (u32)(pt >> 32);

    /* start time: prefer pid; else migrate from old pid */
    u64 st = lookup_start(pid);
    if (!st && old && old != pid) {
        st = lookup_start(old);
        if (st) {
            bpf_map_update_elem(&tid_start_ns, &pid, &st, BPF_ANY);
            bpf_map_delete_elem(&tid_start_ns, &old);
        }
    }
    e.tid_start_ns = st;
    e.tgid_start_ns = lookup_start(e.tgid);

    /* migrate comm cache if pid swapped */
    if (old && old != pid) {
        struct comm_t *oc = bpf_map_lookup_elem(&tid_comm, &old);
        if (oc) {
            struct comm_t tmp = {};
            __builtin_memcpy(&tmp, oc, sizeof(tmp));
            bpf_map_update_elem(&tid_comm, &pid, &tmp, BPF_ANY);
        }
        bpf_map_delete_elem(&tid_comm, &old);
        bpf_map_delete_elem(&tid_tgid, &old);
    }

    /* update TGID cache */
    bpf_map_update_elem(&tid_tgid, &pid, &e.tgid, BPF_ANY);

    /* comm after exec (kernel typically sets comm to basename during exec) */
    bpf_get_current_comm(e.comm, sizeof(e.comm));
    cache_comm(e.tid, e.comm);

    /* Read cmdline after exec - this will have the new program's cmdline */
    struct task_struct *task = (struct task_struct *)bpf_get_current_task();
    read_cmdline_from_task(task, e.cmdline, e.comm);

    /* metadata update (exec might change creds) */
    u64 uid_gid = bpf_get_current_uid_gid();
    e.uid = (u32)uid_gid;
    e.gid = (u32)(uid_gid >> 32);
    e.cgroup_id = bpf_get_current_cgroup_id();

    int ret = bpf_ringbuf_output(&events, &e, sizeof(e), 0);
    if (ret) inc_drop();
    return 0;
}

/*
 * task:task_rename
 * Do NOT depend on args->oldcomm/newcomm. Maintain our own comm cache instead.
 */
SEC("tracepoint/task/task_rename")
int task_rename(struct trace_event_raw_task_rename *args)
{
    struct task_event_t e = {};
    e.event_type = EVENT_TYPE_RENAME;

    u64 pt = bpf_get_current_pid_tgid();
    e.tid  = (u32)(pt & 0xFFFFFFFF);
    e.tgid = (u32)(pt >> 32);
    e.tid_start_ns = lookup_start(e.tid);
    e.tgid_start_ns = lookup_start(e.tgid);

    /* new comm from current */
    bpf_get_current_comm(e.comm, sizeof(e.comm));
    cache_comm(e.tid, e.comm);

    /* Read cmdline */
    struct task_struct *task = (struct task_struct *)bpf_get_current_task();
    read_cmdline_from_task(task, e.cmdline, e.comm);

    /* keep tgid cache warm */
    bpf_map_update_elem(&tid_tgid, &e.tid, &e.tgid, BPF_ANY);

    /* metadata */
    u64 uid_gid = bpf_get_current_uid_gid();
    e.uid = (u32)uid_gid;
    e.gid = (u32)(uid_gid >> 32);
    e.cgroup_id = bpf_get_current_cgroup_id();

    int ret = bpf_ringbuf_output(&events, &e, sizeof(e), 0);
    if (ret) inc_drop();
    return 0;
}

/*
 * sched:sched_process_exit
 * Fires when task exits.
 */
SEC("tracepoint/sched/sched_process_exit")
int sched_process_exit(struct trace_event_raw_sched_process_exit *args)
{
    struct task_event_t e = {};
    e.event_type = EVENT_TYPE_EXIT;

    u64 pt = bpf_get_current_pid_tgid();
    e.tid  = (u32)pt;
    e.tgid = (u32)(pt >> 32);
    e.tid_start_ns = lookup_start(e.tid);
    e.tgid_start_ns = lookup_start(e.tgid);

    bpf_get_current_comm(e.comm, sizeof(e.comm));

    /* Read cmdline */
    struct task_struct *task = (struct task_struct *)bpf_get_current_task();
    read_cmdline_from_task(task, e.cmdline, e.comm);

    /* cleanup caches to avoid map growth */
    bpf_map_delete_elem(&tid_start_ns, &e.tid);
    bpf_map_delete_elem(&tid_tgid, &e.tid);
    bpf_map_delete_elem(&tid_comm, &e.tid);

    int ret = bpf_ringbuf_output(&events, &e, sizeof(e), 0);
    if (ret) inc_drop();
    return 0;
}
