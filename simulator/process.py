import numpy as np

class Process:
    def __init__(self, pid, arrival_time, cpu_bursts, io_bursts, est_bursts=None, meta=None):
        self.pid = pid
        self.arrival_time = arrival_time
        self.cpu_bursts = np.array(cpu_bursts)
        self.io_bursts = np.array(io_bursts)
        self.est_bursts = np.array(est_bursts)
        self.meta = meta or {}
        pass

    def next_cpu_burst(self) -> int:
        if len(self.cpu_bursts) > 0:
            return self.cpu_bursts[0]
        return None
    
    def est_cpu_burst(self) -> int:
        if self.est_bursts is not None and len(self.est_bursts) > 0:
            return np.sum(self.est_bursts)
        return None

    def is_complete(self) -> bool:
        return len(self.cpu_bursts) == 0 and len(self.io_bursts) == 0
    