REGIME_1 = "REGIME_1"   # IO-heavy baseline
REGIME_2 = "REGIME_2"   # CPU-heavy baseline
REGIME_3 = "REGIME_3"   # Mixed baseline
REGIME_4 = "REGIME_4"   # Bursty arrivals (ON/OFF)
REGIME_5 = "REGIME_5"   # IO overload
REGIME_6 = "REGIME_6"   # Sparse extreme CPU
REGIME_7 = "REGIME_7"   # Partial observability
REGIME_8 = "REGIME_8"   # Heavy-tailed CPU bursts
REGIME_9 = "REGIME_9"   # Heavy-tailed + noisy

WAIT_TIME_WEIGHT = 0.38       # strongest: keep queues short
BLOCKED_TIME_WEIGHT = 0.23    # next: reduce IO stalls
IDLE_CPI_WEIGHT = 0.08        # smaller: avoid idle cycles
CONTEXT_SWITCH_PENALTY = 0.01 # tiny: discourage thrashing without freezing
COMPLETION_REWARD = 0.30      # per-job bonus (not episode-end)