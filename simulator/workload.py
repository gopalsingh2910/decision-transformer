import numpy as np

from . process import Process
from . regimes import REGIMES

class Workload:
    """
    Workload generator that samples arrivals and process burst structures per tick.

    Usage:
        wl = Workload()
        wl.reset()
        jobs = wl.generate_jobs(regime=REGIME_7, tick=clock_tick)
    """

    def __init__(self, regime=None):
        self.regime = regime
        self.on_off_state = 'OFF'  # For ON/OFF regimes
        self.pid_counter = 0
        pass

    def reset(self, regime):
        self.regime = regime
        self.on_off_state = 'OFF'
        self.pid_counter = 0
        pass

    def switch_on_off(self):
        if self.on_off_state == "OFF":
            if np.random.uniform(0, 1) <= self.regime["arrival"]["off_to_on"]:
                self.on_off_state = "ON"
        else:
            if np.random.uniform(0, 1) <= self.regime["arrival"]["on_to_off"]:
                self.on_off_state = "OFF"

    def arrival_count(self) -> int:
        a = self.regime["arrival"]
        if a["type"] == "poisson":
            return np.random.poisson(a["lambda"])
        elif a["type"] == "on_off":
            # Update ON/OFF state then sample with respective rate
            self.switch_on_off()
            lam = a["lambda_on"] if self.on_off_state == "ON" else a["lambda_off"]
            return np.random.poisson(lam)
        return 0

    def sample_bursts(self):
        ptype = "cpu" if np.random.uniform(0, 1) <= self.regime["mix"]["cpu"] else "io"
        bcfg = self.regime["bursts"][ptype]

        # Number of CPU bursts
        n_low, n_high = bcfg["cpu_burst_count_range"]
        n_bursts = int(np.random.randint(n_low, n_high + 1))

        # CPU bursts: support uniform or heavy-tailed distributions
        cpu_bursts = [1 for _ in range(n_bursts)]
        if "cpu_uniform" in bcfg and bcfg["cpu_uniform"] is not None:
            lo, hi = bcfg["cpu_uniform"]
            cpu_bursts = [int(np.random.uniform(lo, hi)) for _ in range(n_bursts)]
        elif "cpu_dist" in bcfg and bcfg["cpu_dist"] is not None:
            dist = bcfg["cpu_dist"]
            if dist.get("type") == "pareto":
                alpha = float(dist["alpha"])
                xm = int(dist["min"])  # scale parameter (minimum)
                cpu_bursts = [int(max(1, xm * (np.random.pareto(alpha) + 1))) for _ in range(n_bursts)]

        # IO bursts only for IO-bound processes and only up to len(cpu_bursts)-1
        io_bursts = []
        if bcfg.get("io_uniform") is not None and n_bursts > 1:
            ilo, ihi = bcfg["io_uniform"]
            # Between CPU bursts: there are n_bursts - 1 IO bursts
            io_bursts = [int(np.random.randint(ilo, ihi + 1)) for _ in range(n_bursts - 1)]
        return cpu_bursts, io_bursts, ptype

    def observe_bursts(self, cpu_bursts):
        noise_std = self.regime.get("obs_noise_std", 0.0)
        if noise_std > 0:
            # observed = true * exp(N(0, noise_std))
            est_bursts = [max(1, int(round(b * float(np.exp(np.random.normal(0.0, noise_std)))))) for b in cpu_bursts]
            return est_bursts
        return []

    def generate_jobs(self, tick):
        """Generate zero or more jobs at the given tick.

        Returns a list of Process objects.
        """

        jobs = []
        if self.regime['type'] not in REGIMES.keys():
            return jobs
        
        count = self.arrival_count()
        if count <= 0:
            return jobs
        
        for _ in range(count):
            cpu_bursts, io_bursts, ptype = self.sample_bursts()
            est_bursts = self.observe_bursts(cpu_bursts)
            self.pid_counter += 1
            proc = Process(
                pid=self.pid_counter,
                arrival_time=tick,
                cpu_bursts=cpu_bursts,
                io_bursts=io_bursts,
                est_bursts=est_bursts,
                meta={"type": ptype.upper()}
            )
            jobs.append(proc)
        return jobs

