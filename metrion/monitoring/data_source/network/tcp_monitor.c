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
 * TCP/UDP network traffic monitor using eBPF ring buffer.
 * Captures send/receive events with pid start time for LogicalEntityRegistry lookup.
 */

struct tcp_event
{
  u64 cgroup_id;
  u64 pid_start_time;   // pid/tid start time in ns since boot
  u64 tgid_start_time;  // tgid (process) start time in ns since boot
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
};

// Ring buffer for events in pages
BPF_RINGBUF_OUTPUT(events, 4096);

// Count dropped events (per-CPU for performance)
BPF_PERCPU_ARRAY(rb_drops, u64, 1);

/* Helper to get pid/tgid start times from task_struct */
static __always_inline void get_pid_start_times(u64 *pid_start_time, u64 *tgid_start_time)
{
  struct task_struct *task = (struct task_struct *)bpf_get_current_task();
  if (!task)
  {
    *pid_start_time = 0;
    *tgid_start_time = 0;
    return;
  }

  // Get pid/tid start time - use start_boottime for consistency with
  // sched_switch and logical_entity_registry (CLOCK_BOOTTIME domain)
  bpf_probe_read_kernel(pid_start_time, sizeof(*pid_start_time), &task->start_boottime);

  // Get tgid (process group leader) start time
  struct task_struct *group_leader = NULL;
  bpf_probe_read_kernel(&group_leader, sizeof(group_leader), &task->group_leader);
  if (group_leader)
  {
    bpf_probe_read_kernel(tgid_start_time, sizeof(*tgid_start_time), &group_leader->start_boottime);
  }
  else
  {
    *tgid_start_time = *pid_start_time;
  }
}

static __always_inline void drop_inc(void)
{
  u32 key = 0;
  u64 *drops = rb_drops.lookup(&key);
  if (drops)
    (*drops)++;
}

int trace_tcp_cleanup_rbuf(struct pt_regs *ctx)
{
  struct sock *sk = (struct sock *)PT_REGS_PARM1(ctx);
  int copied = (int)PT_REGS_PARM2(ctx);

  if (!sk || copied <= 0)
    return 0;

  struct tcp_event *e = events.ringbuf_reserve(sizeof(struct tcp_event));
  if (!e)
  {
    drop_inc();
    return 0;
  }

  u64 pid_tgid = bpf_get_current_pid_tgid();
  e->pid = pid_tgid & 0xFFFFFFFF;
  e->tgid = pid_tgid >> 32;
  e->cgroup_id = bpf_get_current_cgroup_id();
  e->time_ns = bpf_ktime_get_ns();
  e->socket_id = (u64)sk;

  get_pid_start_times(&e->pid_start_time, &e->tgid_start_time);

  e->bytes = copied;
  e->direction = 0; // receive
  e->transport_type = 0; // TCP

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

  struct tcp_event *e = events.ringbuf_reserve(sizeof(struct tcp_event));
  if (!e)
  {
    drop_inc();
    return 0;
  }

  u64 pid_tgid = bpf_get_current_pid_tgid();
  e->pid = pid_tgid & 0xFFFFFFFF;
  e->tgid = pid_tgid >> 32;
  e->cgroup_id = bpf_get_current_cgroup_id();
  e->time_ns = bpf_ktime_get_ns();
  e->socket_id = (u64)sk;

  get_pid_start_times(&e->pid_start_time, &e->tgid_start_time);

  e->bytes = (u32)len;
  e->direction = 0; // receive
  e->transport_type = 1; // UDP

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

  struct tcp_event *e = events.ringbuf_reserve(sizeof(struct tcp_event));
  if (!e)
  {
    drop_inc();
    return 0;
  }

  u64 pid_tgid = bpf_get_current_pid_tgid();
  e->pid = pid_tgid & 0xFFFFFFFF;
  e->tgid = pid_tgid >> 32;
  e->cgroup_id = bpf_get_current_cgroup_id();
  e->time_ns = bpf_ktime_get_ns();
  e->socket_id = (u64)sk;

  get_pid_start_times(&e->pid_start_time, &e->tgid_start_time);

  e->bytes = size;
  e->direction = 1; // send
  e->transport_type = 0; // TCP

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

  struct tcp_event *e = events.ringbuf_reserve(sizeof(struct tcp_event));
  if (!e)
  {
    drop_inc();
    return 0;
  }

  u64 pid_tgid = bpf_get_current_pid_tgid();
  e->pid = pid_tgid & 0xFFFFFFFF;
  e->tgid = pid_tgid >> 32;
  e->cgroup_id = bpf_get_current_cgroup_id();
  e->time_ns = bpf_ktime_get_ns();
  e->socket_id = (u64)sk;

  get_pid_start_times(&e->pid_start_time, &e->tgid_start_time);

  e->bytes = (u32)len;
  e->direction = 1; // send
  e->transport_type = 1; // UDP

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

/* Tracepoint to capture ifindex for incoming packets */
TRACEPOINT_PROBE(net, netif_receive_skb)
{
  // ifindex capture via tracepoint removed - we capture it directly from socket now
  // This tracepoint was used for fallback when skc_bound_dev_if was 0
  // For simplicity in the ring buffer implementation, we skip this complexity
  return 0;
}

/* Tracepoint to capture ifindex for outgoing packets */
TRACEPOINT_PROBE(net, net_dev_queue)
{
  // ifindex capture via tracepoint removed - we capture it directly from socket now
  return 0;
}