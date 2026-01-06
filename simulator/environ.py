import numpy as np

from . import _cpu_ as cpu
from .workload import Workload
from .data_loader import DataLoader
from .regimes import REGIMES
from decision_transformer.param import MIN_TIMESTEPS, MAX_TIMESTEPS
from ._cpu_.param import TIMER_INTERRUPT, QUEUE_CAP
from .param import (
    WAIT_TIME_WEIGHT,
    BLOCKED_TIME_WEIGHT,
    IDLE_CPI_WEIGHT,
    CONTEXT_SWITCH_PENALTY,
    COMPLETION_REWARD
    )

class Env:
    def __init__(self, cpu, wl, dl, save=False):
        self.cpu = cpu
        self.wl = wl
        self.dl = dl
        self.save = save
        pass

    def reset(self, seed=None, regime=None):
        if seed is not None:
            np.random.seed(seed)
        self.regime = np.random.choice(list(REGIMES.values())) if regime is None else regime

        self.cpu.reset()
        self.wl.reset(self.regime)
        self.dl.reset()

        self.episode_length = np.random.randint(MIN_TIMESTEPS, MAX_TIMESTEPS + 1)
        self.episode_length_left = self.episode_length

        self.admission_queue = []
        self.ready_queue = []
        self.blocked_queue = []  # (proc, io_remaining)

        self.arrivals_last_8 = [0]*8

        self.last_run_time = {}
        self.cpu_used = {}
        self.wait_time = {}
        self.blocked_time = {}

        self.current_pid = -1
        self.time_slice_left = 0
        pass

    def est_next_cpu(self, proc):
        if hasattr(proc, "est_cpu_bursts") and proc.est_cpu_bursts:
            return int(proc.est_cpu_bursts[0])
        nxt = proc.next_cpu_burst()
        return int(nxt) if nxt is not None else 0

    def get_state(self):
        tick = self.cpu.clock.tick
        arrivals_hist = self.arrivals_last_8[-8:]

        state = [
            self.current_pid / 100.0,
            len(self.ready_queue) / QUEUE_CAP,
            len(self.blocked_queue) / QUEUE_CAP,
            float(np.mean(arrivals_hist)) / 10.0,
            sum(self.est_next_cpu(p) for p in self.ready_queue) / 100.0,
            self.time_slice_left / TIMER_INTERRUPT,
        ]

        for hist in self.arrivals_last_8:
            state.append(hist / 10.0)

        for p in self.ready_queue:
            pid = p.pid
            state.append((tick - p.arrival_time) / 1000.0)  # age
            state.append(tick - self.last_run_time.get(pid, p.arrival_time) / 1000.0)  # since last run
            state.append(self.cpu_used.get(pid, 0) / 1000.0)  # cpu used
            state.append(self.blocked_time.get(pid, 0) / 1000.0)  # blocked time
            state.append(self.wait_time.get(pid, 0) / 1000.0)  # wait time
            state.append(self.est_next_cpu(p) / 100.0) # est rem cpu
        
        for _ in range(QUEUE_CAP - len(self.ready_queue)):
            state.append(0.0)  # age
            state.append(0.0)  # since last run
            state.append(0.0)  # cpu used
            state.append(0.0)  # blocked time
            state.append(0.0)  # wait time
            state.append(0.0)  # est rem cpu
        return state

    def update_blocked(self):
        new_blocked = []
        completed = []

        for p, io_rem in self.blocked_queue:
            pid = p.pid
            self.blocked_time[pid] = self.blocked_time.get(pid, 0) + 1
            io_rem -= 1
            if io_rem <= 0:
                completed.append(p)
            else:
                new_blocked.append((p, io_rem))

        self.blocked_queue = new_blocked
        if completed:
            space = QUEUE_CAP - len(self.ready_queue)
            if space > 0:
                admitted = completed[:space]
                self.ready_queue.extend(admitted)
                overflow = completed[space:]
                # overflow go back to admission_queue to be admitted later
                if overflow:
                    self.admission_queue.extend(overflow)
            else:
                # No space: defer all completed processes
                self.admission_queue.extend(completed)
        pass

    def run_selected(self, action) -> int:
        if not self.ready_queue:
            self.current_pid = -1
            self.time_slice_left = TIMER_INTERRUPT
            return 0

        if action is None or action < 0 or action >= len(self.ready_queue):
            self.cpu.idle_time += 1
            return 0

        p = self.ready_queue[action]
        pid = p.pid
        if self.current_pid == -1 or self.current_pid != pid:
            self.time_slice_left = TIMER_INTERRUPT
            self.current_pid = pid

        if self.time_slice_left <= 0:
            self.time_slice_left = TIMER_INTERRUPT

        burst = p.next_cpu_burst()
        if burst is None:
            self.terminate(pid)
            self.cpu.idle_time += 1
            return 0

        p.cpu_bursts[0] -= 1
        self.cpu_used[pid] = self.cpu_used.get(pid, 0) + 1
        self.last_run_time[pid] = self.cpu.clock.tick
        self.time_slice_left -= 1

        if p.cpu_bursts[0] == 0:
            p.cpu_bursts = p.cpu_bursts[1:]
            if len(p.io_bursts) > 0:
                self.ready_queue.pop(action)
                io = p.io_bursts[0]
                p.io_bursts = p.io_bursts[1:]
                self.blocked_queue.append((p, io))
                self.current_pid = -1
        
        # If process has no remaining work, terminate
        if p.is_complete():
            self.terminate(pid)
            return 1
        return 0

    def step(self, action):
        '''
            reward = - a*sigma (wait_time) - b*sigma(blocked_time) - c*sigma(cpi idle time) - d*1[context switched] + d*sigma 1[completed]
        '''

        state = self.get_state()
        arrivals = self.wl.generate_jobs(self.cpu.clock.tick)
        self.arrivals_last_8.append(len(arrivals))
        self.arrivals_last_8 = self.arrivals_last_8[-8:]
        
        self.admission_queue.extend(arrivals)
        space = min(QUEUE_CAP - len(self.ready_queue), len(self.admission_queue))
        if space > 0:
            admitted = self.admission_queue[:space]
            self.ready_queue.extend(admitted)
            self.admission_queue = self.admission_queue[space:]

        job_completed = self.run_selected(action)
        self.update_blocked()

        for p in self.ready_queue:
            pid = p.pid
            if pid != self.current_pid:
                self.wait_time[pid] = self.wait_time.get(pid, 0) + 1

        self.cpu.clock.step()
        self.episode_length_left -= 1
        next_state = self.get_state()
        done = self.episode_length_left <= 0
        
        reward = -WAIT_TIME_WEIGHT * sum(self.wait_time.values()) / 1000.0 - BLOCKED_TIME_WEIGHT * sum(self.blocked_time.values()) / 1000.0 - IDLE_CPI_WEIGHT * self.cpu.idle_time / 1000.0 - CONTEXT_SWITCH_PENALTY * (1 if self.time_slice_left == TIMER_INTERRUPT else 0) + COMPLETION_REWARD * job_completed

        self.dl.load_data(
            state=state,
            action=action,
            reward=reward
        )

        if done and self.save:
            self.dl.save_episode()
        return next_state, reward, done
    
    def terminate(self, pid):
        self.ready_queue = [p for p in self.ready_queue if p.pid != pid]
        self.blocked_queue = [(p, io_rem) for (p, io_rem) in self.blocked_queue if p.pid != pid]
        if self.current_pid == pid:
            self.current_pid = -1
            self.time_slice_left = TIMER_INTERRUPT
        
        # Clean up stats
        if pid in self.last_run_time:
            del self.last_run_time[pid]
        if pid in self.cpu_used:
            del self.cpu_used[pid]
        if pid in self.wait_time:
            del self.wait_time[pid]
        if pid in self.blocked_time:
            del self.blocked_time[pid]
        pass

def make_env(pre_emptive=False, path=None):
    save = path is not None
    c = cpu.initialize(pre_emptive=pre_emptive)
    wl = Workload()
    dl = DataLoader(path=path)
    env = Env(c, wl, dl, save)
    return env
