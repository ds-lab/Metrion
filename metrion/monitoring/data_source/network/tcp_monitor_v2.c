#include <uapi/linux/ptrace.h>
#include <uapi/linux/ip.h>
#include <uapi/linux/ipv6.h>
#include <uapi/linux/tcp.h>
#include <uapi/linux/udp.h>
#include <net/sock.h>
#include <linux/skbuff.h>
#include <linux/netdevice.h>
#include <bcc/proto.h>

/*
 * TCP/UDP network traffic monitor v2 with inline metadata for zero-loss entity matching.
 *
 * Key improvement: Like sched_switch_v2, we embed process metadata in the first
 * network event for each PID to ensure entities are created before they're needed.
 */

#ifndef TASK_COMM_LEN
#define TASK_COMM_LEN 16
#endif

#ifndef CMDLINE_MAX_LEN
#define CMDLINE_MAX_LEN 128
#endif

/* Event type flags */
#define EVENT_TYPE_DATA_ONLY      0
#define EVENT_TYPE_WITH_METADATA  1

/*
 * Track seen PIDs to only send metadata once per unique task.
 */
struct pid_key {
  u32 pid;
  u32 start_bucket;  // start_time_ns / 10_000_000 (10ms buckets)
};

BPF_HASH(pid_seen, struct pid_key, u8, 262144);

/*
 * Network event with optional inline metadata.
 */
struct tcp_event_v2
{
  // === Core event data (always present) ===
  u64 cgroup_id;
  u64 pid_start;        // pid/tid start time in ns since boot
  u64 tgid_start;       // tgid (process) start time in ns since boot
  u64 time_ns;          // event timestamp in ns since boot
  u64 socket_id;        // socket pointer as unique identifier
  u32 pid;
  u32 tgid;
  u32 bytes;
  u32 ifindex;
  u32 daddr;
  u32 saddr;
  u16 dport;
  u16 sport;
  u16 family;
  u8 ipv6_daddr[16];
  u8 ipv6_rcv_saddr[16];
  u8 direction;      // 0 = receive, 1 = send
  u8 transport_type; // 0 = TCP, 1 = UDP
  u8 event_type;     // 0 = data only, 1 = with metadata
  u8 __pad;

  // === Inline metadata (only present when event_type == 1) ===
  u64 parent_tid_start;
  u64 parent_tgid_start;
  u32 parent_tid;
  u32 parent_tgid;
  u32 uid;
  u32 gid;
  char comm[TASK_COMM_LEN];
  char cmdline[CMDLINE_MAX_LEN];
};

// Ring buffer for events
BPF_RINGBUF_OUTPUT(events, 4096);

// Count dropped events (per-CPU for performance)
BPF_PERCPU_ARRAY(rb_drops, u64, 1);

static __always_inline void drop_inc(void)
{
  u32 key = 0;
  u64 *drops = rb_drops.lookup(&key);
  if (drops)
    (*drops)++;
}

/*
 * Check if we've seen this (pid, start_time) before.
 * Returns 1 if first time (and marks as seen), 0 if already seen.
 */
static __always_inline int mark_first_seen(u32 pid, u64 start_ns) {
  struct pid_key key = {};
  key.pid = pid;
  key.start_bucket = (u32)(start_ns / 10000000);  // 10ms bucket

  u8 *seen = pid_seen.lookup(&key);
  if (seen) {
    return 0;  // Already seen
  }

  u8 one = 1;
  pid_seen.update(&key, &one);
  return 1;  // First time
}

/*
 * Helper to read cmdline from a task's mm_struct into a buffer.
 */
static __always_inline void read_cmdline(struct task_struct *task, char *cmdline_buf)
{
  cmdline_buf[0] = '\0';

  if (!task)
    return;

  struct mm_struct *mm = NULL;
  bpf_probe_read_kernel(&mm, sizeof(mm), &task->mm);

  if (mm)
  {
    unsigned long arg_start = 0, arg_end = 0;
    bpf_probe_read_kernel(&arg_start, sizeof(arg_start), &mm->arg_start);
    bpf_probe_read_kernel(&arg_end, sizeof(arg_end), &mm->arg_end);

    if (arg_start && arg_end && arg_end > arg_start)
    {
      unsigned long arg_len = arg_end - arg_start;
      if (arg_len > 0 && arg_len <= 512)
      {
        unsigned long read_len = (arg_len < (CMDLINE_MAX_LEN - 1)) ? arg_len : (CMDLINE_MAX_LEN - 1);
        long ret = bpf_probe_read_user(cmdline_buf, read_len, (void *)arg_start);
        if (ret == 0)
        {
#pragma unroll
          for (int i = 0; i < (CMDLINE_MAX_LEN - 2); i++)
          {
            if (cmdline_buf[i] == '\0' && i > 0)
              cmdline_buf[i] = ' ';
          }
          cmdline_buf[CMDLINE_MAX_LEN - 1] = '\0';
          return;
        }
      }
    }
  }

  // Fallback: use comm
  bpf_get_current_comm(cmdline_buf, CMDLINE_MAX_LEN);
}

/* Helper to get pid/tgid start times and optionally collect inline metadata */
static __always_inline void collect_task_info(
    struct tcp_event_v2 *e,
    int collect_metadata
)
{
  struct task_struct *task = (struct task_struct *)bpf_get_current_task();
  if (!task)
  {
    e->pid_start = 0;
    e->tgid_start = 0;
    return;
  }

  // Get pid/tid start time (CLOCK_BOOTTIME domain)
  bpf_probe_read_kernel(&e->pid_start, sizeof(e->pid_start), &task->start_boottime);

  // Get tgid (process group leader) start time
  struct task_struct *group_leader = NULL;
  bpf_probe_read_kernel(&group_leader, sizeof(group_leader), &task->group_leader);
  if (group_leader)
  {
    bpf_probe_read_kernel(&e->tgid_start, sizeof(e->tgid_start), &group_leader->start_boottime);
  }
  else
  {
    e->tgid_start = e->pid_start;
  }

  if (!collect_metadata)
  {
    e->event_type = EVENT_TYPE_DATA_ONLY;
    return;
  }

  e->event_type = EVENT_TYPE_WITH_METADATA;

  // Get parent info
  struct task_struct *parent = NULL;
  bpf_probe_read_kernel(&parent, sizeof(parent), &task->real_parent);
  if (parent)
  {
    bpf_probe_read_kernel(&e->parent_tid, sizeof(e->parent_tid), &parent->pid);
    bpf_probe_read_kernel(&e->parent_tgid, sizeof(e->parent_tgid), &parent->tgid);
    bpf_probe_read_kernel(&e->parent_tid_start, sizeof(e->parent_tid_start), &parent->start_boottime);

    struct task_struct *parent_leader = NULL;
    bpf_probe_read_kernel(&parent_leader, sizeof(parent_leader), &parent->group_leader);
    if (parent_leader)
    {
      bpf_probe_read_kernel(&e->parent_tgid_start, sizeof(e->parent_tgid_start), &parent_leader->start_boottime);
    }
    else
    {
      e->parent_tgid_start = e->parent_tid_start;
    }
  }

  // User/Group
  u64 uid_gid = bpf_get_current_uid_gid();
  e->uid = (u32)uid_gid;
  e->gid = (u32)(uid_gid >> 32);

  // Comm and cmdline
  bpf_get_current_comm(e->comm, sizeof(e->comm));
  read_cmdline(group_leader, e->cmdline);
}

int trace_tcp_cleanup_rbuf(struct pt_regs *ctx)
{
  struct sock *sk = (struct sock *)PT_REGS_PARM1(ctx);
  int copied = (int)PT_REGS_PARM2(ctx);

  if (!sk || copied <= 0)
    return 0;

  u64 pid_tgid = bpf_get_current_pid_tgid();
  u32 pid = pid_tgid & 0xFFFFFFFF;
  u32 tgid = pid_tgid >> 32;

  // Get start time early to check first-seen status
  struct task_struct *task = (struct task_struct *)bpf_get_current_task();
  u64 start_ns = 0;
  if (task)
    bpf_probe_read_kernel(&start_ns, sizeof(start_ns), &task->start_boottime);

  int is_first = mark_first_seen(pid, start_ns);

  struct tcp_event_v2 *e = events.ringbuf_reserve(sizeof(struct tcp_event_v2));
  if (!e)
  {
    drop_inc();
    return 0;
  }

  e->pid = pid;
  e->tgid = tgid;
  e->cgroup_id = bpf_get_current_cgroup_id();
  e->time_ns = bpf_ktime_get_ns();
  e->socket_id = (u64)sk;
  e->bytes = copied;
  e->direction = 0; // receive
  e->transport_type = 0; // TCP
  e->__pad = 0;

  // Collect task info with optional metadata
  collect_task_info(e, is_first);

  u32 ifidx = 0;
  bpf_probe_read_kernel(&ifidx, sizeof(ifidx), &sk->__sk_common.skc_bound_dev_if);
  e->ifindex = ifidx;

  e->daddr = sk->__sk_common.skc_daddr;
  e->saddr = sk->__sk_common.skc_rcv_saddr;
  e->dport = bpf_ntohs(sk->__sk_common.skc_dport);
  e->sport = sk->__sk_common.skc_num;
  e->family = sk->__sk_common.skc_family;

  struct in6_addr ipv6_daddr = sk->__sk_common.skc_v6_daddr;
  struct in6_addr ipv6_rcv_saddr = sk->__sk_common.skc_v6_rcv_saddr;
#pragma unroll
  for (int i = 0; i < 16; i++)
  {
    e->ipv6_daddr[i] = ipv6_daddr.s6_addr[i];
    e->ipv6_rcv_saddr[i] = ipv6_rcv_saddr.s6_addr[i];
  }

  events.ringbuf_submit(e, 0);
  return 0;
}

int trace_udp_recvmsg(struct pt_regs *ctx, struct sock *sk, struct msghdr *msg, size_t len)
{
  if (!sk || len <= 0)
    return 0;

  u64 pid_tgid = bpf_get_current_pid_tgid();
  u32 pid = pid_tgid & 0xFFFFFFFF;
  u32 tgid = pid_tgid >> 32;

  struct task_struct *task = (struct task_struct *)bpf_get_current_task();
  u64 start_ns = 0;
  if (task)
    bpf_probe_read_kernel(&start_ns, sizeof(start_ns), &task->start_boottime);

  int is_first = mark_first_seen(pid, start_ns);

  struct tcp_event_v2 *e = events.ringbuf_reserve(sizeof(struct tcp_event_v2));
  if (!e)
  {
    drop_inc();
    return 0;
  }

  e->pid = pid;
  e->tgid = tgid;
  e->cgroup_id = bpf_get_current_cgroup_id();
  e->time_ns = bpf_ktime_get_ns();
  e->socket_id = (u64)sk;
  e->bytes = (u32)len;
  e->direction = 0; // receive
  e->transport_type = 1; // UDP
  e->__pad = 0;

  collect_task_info(e, is_first);

  u32 ifidx = 0;
  bpf_probe_read_kernel(&ifidx, sizeof(ifidx), &sk->__sk_common.skc_bound_dev_if);
  e->ifindex = ifidx;

  e->family = sk->__sk_common.skc_family;
  e->daddr = sk->__sk_common.skc_daddr;
  e->saddr = sk->__sk_common.skc_rcv_saddr;
  e->dport = bpf_ntohs(sk->__sk_common.skc_dport);
  e->sport = sk->__sk_common.skc_num;

  struct in6_addr ipv6_daddr = sk->__sk_common.skc_v6_daddr;
  struct in6_addr ipv6_rcv_saddr = sk->__sk_common.skc_v6_rcv_saddr;
#pragma unroll
  for (int i = 0; i < 16; i++)
  {
    e->ipv6_daddr[i] = ipv6_daddr.s6_addr[i];
    e->ipv6_rcv_saddr[i] = ipv6_rcv_saddr.s6_addr[i];
  }

  events.ringbuf_submit(e, 0);
  return 0;
}

int trace_tcp_sendmsg(struct pt_regs *ctx, struct sock *sk, struct msghdr *msg, size_t size)
{
  if (!sk || size <= 0)
    return 0;

  u64 pid_tgid = bpf_get_current_pid_tgid();
  u32 pid = pid_tgid & 0xFFFFFFFF;
  u32 tgid = pid_tgid >> 32;

  struct task_struct *task = (struct task_struct *)bpf_get_current_task();
  u64 start_ns = 0;
  if (task)
    bpf_probe_read_kernel(&start_ns, sizeof(start_ns), &task->start_boottime);

  int is_first = mark_first_seen(pid, start_ns);

  struct tcp_event_v2 *e = events.ringbuf_reserve(sizeof(struct tcp_event_v2));
  if (!e)
  {
    drop_inc();
    return 0;
  }

  e->pid = pid;
  e->tgid = tgid;
  e->cgroup_id = bpf_get_current_cgroup_id();
  e->time_ns = bpf_ktime_get_ns();
  e->socket_id = (u64)sk;
  e->bytes = size;
  e->direction = 1; // send
  e->transport_type = 0; // TCP
  e->__pad = 0;

  collect_task_info(e, is_first);

  u32 ifidx = 0;
  bpf_probe_read_kernel(&ifidx, sizeof(ifidx), &sk->__sk_common.skc_bound_dev_if);
  e->ifindex = ifidx;

  e->daddr = sk->__sk_common.skc_daddr;
  e->saddr = sk->__sk_common.skc_rcv_saddr;
  e->dport = bpf_ntohs(sk->__sk_common.skc_dport);
  e->sport = sk->__sk_common.skc_num;
  e->family = sk->__sk_common.skc_family;

  struct in6_addr ipv6_daddr = sk->__sk_common.skc_v6_daddr;
  struct in6_addr ipv6_rcv_saddr = sk->__sk_common.skc_v6_rcv_saddr;
#pragma unroll
  for (int i = 0; i < 16; i++)
  {
    e->ipv6_daddr[i] = ipv6_daddr.s6_addr[i];
    e->ipv6_rcv_saddr[i] = ipv6_rcv_saddr.s6_addr[i];
  }

  events.ringbuf_submit(e, 0);
  return 0;
}

int trace_udp_sendmsg(struct pt_regs *ctx, struct sock *sk, struct msghdr *msg, size_t len)
{
  if (!sk || len <= 0)
    return 0;

  u64 pid_tgid = bpf_get_current_pid_tgid();
  u32 pid = pid_tgid & 0xFFFFFFFF;
  u32 tgid = pid_tgid >> 32;

  struct task_struct *task = (struct task_struct *)bpf_get_current_task();
  u64 start_ns = 0;
  if (task)
    bpf_probe_read_kernel(&start_ns, sizeof(start_ns), &task->start_boottime);

  int is_first = mark_first_seen(pid, start_ns);

  struct tcp_event_v2 *e = events.ringbuf_reserve(sizeof(struct tcp_event_v2));
  if (!e)
  {
    drop_inc();
    return 0;
  }

  e->pid = pid;
  e->tgid = tgid;
  e->cgroup_id = bpf_get_current_cgroup_id();
  e->time_ns = bpf_ktime_get_ns();
  e->socket_id = (u64)sk;
  e->bytes = (u32)len;
  e->direction = 1; // send
  e->transport_type = 1; // UDP
  e->__pad = 0;

  collect_task_info(e, is_first);

  u32 ifidx = 0;
  bpf_probe_read_kernel(&ifidx, sizeof(ifidx), &sk->__sk_common.skc_bound_dev_if);
  e->ifindex = ifidx;

  e->family = sk->__sk_common.skc_family;
  e->daddr = sk->__sk_common.skc_daddr;
  e->saddr = sk->__sk_common.skc_rcv_saddr;
  e->dport = bpf_ntohs(sk->__sk_common.skc_dport);
  e->sport = sk->__sk_common.skc_num;

  struct in6_addr ipv6_daddr = sk->__sk_common.skc_v6_daddr;
  struct in6_addr ipv6_rcv_saddr = sk->__sk_common.skc_v6_rcv_saddr;
#pragma unroll
  for (int i = 0; i < 16; i++)
  {
    e->ipv6_daddr[i] = ipv6_daddr.s6_addr[i];
    e->ipv6_rcv_saddr[i] = ipv6_rcv_saddr.s6_addr[i];
  }

  events.ringbuf_submit(e, 0);
  return 0;
}

/* Clean up pid_seen on task exit to prevent memory growth */
TRACEPOINT_PROBE(sched, sched_process_exit)
{
  u64 pid_tgid = bpf_get_current_pid_tgid();
  u32 pid = (u32)pid_tgid;

  struct task_struct *task = (struct task_struct *)bpf_get_current_task();
  u64 start_ns = 0;
  bpf_probe_read_kernel(&start_ns, sizeof(start_ns), &task->start_boottime);

  struct pid_key key = {};
  key.pid = pid;
  key.start_bucket = (u32)(start_ns / 10000000);

  pid_seen.delete(&key);

  return 0;
}
