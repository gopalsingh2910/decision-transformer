import numpy as np

def fcfs(env):
    ready_queue = env.ready_queue
    if not ready_queue:
        return -1
    arrival_times = [p.arrival_time for p in ready_queue]
    action = np.argmin(arrival_times)
    return action
    

def sjf(env):
    ready_queue = env.ready_queue
    if not ready_queue:
        return -1
    cpu_bursts = [env.cpu_used.get(p.pid, 0) + p.est_cpu_burst() for p in ready_queue]
    action = np.argmin(cpu_bursts)
    return action

def rr(env):
    # Round-robin should rotate by ready-queue index, not PID
    if not env.ready_queue:
        return -1

    if env.time_slice_left > 0:
        return -1

    # Find index of current process in ready_queue
    idx = 0
    if env.current_pid != -1:
        for i, p in enumerate(env.ready_queue):
            if p.pid == env.current_pid:
                idx = i
                break
    # Advance to the next process
    return (idx + 1) % len(env.ready_queue)

def srtf(env):
    ready_queue = env.ready_queue
    if not ready_queue:
        return -1
    cpu_bursts = [p.est_cpu_burst() for p in ready_queue]
    action = np.argmin(cpu_bursts)
    return action

def act(env, rule):
    if rule == "FCFS":
        return fcfs(env)
    elif rule == "SJF":
        return sjf(env)
    elif rule == "RR":
        return rr(env)
    elif rule == "SRTF":
        return srtf(env)
    return -1