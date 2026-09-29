#ifndef COMMON_BPF_H
#define COMMON_BPF_H

#define TASK_COMM_LEN 16
#define CMDLINE_MAX_LEN 128

/*
 * good documentation:
 * - https://linux-kernel-labs.github.io/refs/heads/master/lectures/processes.html
 * - https://docs.huihoo.com/doxygen/linux/kernel/3.7/structtask__struct.html
 */

/*
 * Process metadata structure containing identification and cgroup information
 */
struct pid_metadata
{
  u64 start_time;              // Process start time (boot time offset)
  u64 start_time_parent;       // Parent thread start time (boot time offset)
  u64 start_time_parent_tgid;  // Parent group leader process start time (boot time offset)
  u64 start_time_tgid;         // Thread group leader start time (boot time offset)
  u64 cgroup_id;               // Control group identifier
  u32 tgid;                    // Thread group ID (process ID)
  u32 parent_tid;              // Parent thread ID
  u32 parent_tgid;             // Parent thread group ID (process ID)
  u32 uid;                     // Real user ID
  u32 gid;                     // Real group ID
  u8 is_kernel_thread;         // Flag: 1 if kernel thread, 0 if user process
  char comm[TASK_COMM_LEN];    // Process command name (truncated)
  char cmdline[CMDLINE_MAX_LEN]; // Command line arguments (first part)
};

/*
 * Debug and insert counter enumerations
 */
enum insert_ctr
{
  END_ATTEMPT = 0, // saw pendbuf and tried to write END
  END_OK = 1,      // successfully updated END fields
  START_FAIL = 2,  // failed after data.update() or pid_buffer.update()
  START_OK = 3,    // successfully wrote START + hint
};

enum dbg_ctr
{
  DBG_START_UPDATE_FAIL = 10,
  DBG_HINT_UPDATE_FAIL = 11,
  DBG_END_LOOKUP_MISS = 12, // pendbuf present but couldn't find data entry
};

/*
 * Helper to read cmdline from a task's mm_struct into a buffer.
 * Falls back to copying comm if cmdline is unavailable.
 * Returns 1 if cmdline was successfully read, 0 otherwise.
 */
static __always_inline int read_cmdline_from_task(struct task_struct *task,
                                                  char *cmdline_buf,
                                                  const char *fallback_comm)
{
  cmdline_buf[0] = '\0';

  if (!task)
    return 0;

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
            {
              cmdline_buf[i] = ' ';
            }
          }
          cmdline_buf[CMDLINE_MAX_LEN - 1] = '\0';
          return 1;
        }
      }
    }
  }

  // Fallback: use comm if provided, otherwise read from task->comm
  if (fallback_comm)
  {
    bpf_probe_read_kernel_str(cmdline_buf, CMDLINE_MAX_LEN, fallback_comm);
  }
  else
  {
    bpf_probe_read_kernel_str(cmdline_buf, CMDLINE_MAX_LEN, task->comm);
  }
  cmdline_buf[CMDLINE_MAX_LEN - 1] = '\0';
  return 0;
}

#endif /* COMMON_BPF_H */
