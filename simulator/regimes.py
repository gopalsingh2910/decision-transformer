from . param import (
    REGIME_1,
    REGIME_2,
    REGIME_3,
    REGIME_4,
    REGIME_5,
    REGIME_6,
    REGIME_7,
    REGIME_8,
    REGIME_9,
)

REGIMES = {
    REGIME_1: {
        "type": REGIME_1,
        "arrival": {"type": "poisson", "lambda": 0.3},
        "mix": {"io": 0.8, "cpu": 0.2},
        "bursts": {
            "io": {
                "cpu_uniform": (1, 4),
                "io_uniform": (20, 60),
                "cpu_burst_count_range": (3, 6)
            },
            "cpu": {
                "cpu_uniform": (50, 150),
                "io_uniform": None,
                "cpu_burst_count_range": (1, 1)
            }
        },
        "obs_noise_std": 0.1,
    },
    REGIME_2: {
        "type": REGIME_2,
        "arrival": {"type": "poisson", "lambda": 0.08},
        "mix": {"cpu": 0.9, "io": 0.1},
        "bursts": {
            "cpu": {
                "cpu_uniform": (200, 600),
                "io_uniform": None,
                "cpu_burst_count_range": (1, 1)
            },
            "io": {
                "cpu_uniform": (3, 8),
                "io_uniform": (30, 80),
                "cpu_burst_count_range": (1, 3)
            }
        },
        "obs_noise_std": 0.2,
    },
    REGIME_3: {
        "type": REGIME_3,
        "arrival": {"type": "poisson", "lambda": 0.5},
        "mix": {"io": 0.6, "cpu": 0.4},
        "bursts": {
            "io": {
                "cpu_uniform": (1, 6),
                "io_uniform": (15, 50),
                "cpu_burst_count_range": (2, 5)
            },
            "cpu": {
                "cpu_uniform": (100, 400),
                "io_uniform": None,
                "cpu_burst_count_range": (1, 1)
            }
        },
        "obs_noise_std": 0.25,
    },
    REGIME_4: {
        "type": REGIME_4,
        "arrival": {
            "type": "on_off",
            "off_to_on": 0.05,
            "on_to_off": 0.25,
            "lambda_off": 0.05,
            "lambda_on": 1.25,
        },
        "mix": {"io": 0.7, "cpu": 0.3},
        "bursts": {
            "io": {
                "cpu_uniform": (1, 6),
                "io_uniform": (15, 50),
                "cpu_burst_count_range": (2, 5)
            },
            "cpu": {
                "cpu_uniform": (100, 400),
                "io_uniform": None,
                "cpu_burst_count_range": (1, 1)
            }
        },
        "obs_noise_std": 0.35,
    },
    REGIME_5: {
        "type": REGIME_5,
        "arrival": {"type": "poisson", "lambda": 0.4},
        "mix": {"io": 1.0, "cpu": 0.0},
        "bursts": {
            "io": {
                "cpu_uniform": (1, 8),
                "io_uniform": (10, 30),
                "cpu_burst_count_range": (1, 3)
            }
        },
        "obs_noise_std": 0.05,
    },
    REGIME_6: {
        "type": REGIME_6,
        "arrival": {"type": "poisson", "lambda": 0.03},
        "mix": {"io": 0.0, "cpu": 1.0},
        "bursts": {
            "cpu": {
                "cpu_uniform": (600, 900),
                "io_uniform": None,
                "cpu_burst_count_range": (1, 1)
            }
        },
        "obs_noise_std": 0.3,
    },
    REGIME_7: {
        "type": REGIME_7,
        "arrival": {"type": "poisson", "lambda": 0.5},
        "mix": {"io": 0.6, "cpu": 0.4},
        "bursts": {
            "io": {
                "cpu_uniform": (1, 6),
                "io_uniform": (15, 50),
                "cpu_burst_count_range": (2, 5)
            },
            "cpu": {
                "cpu_uniform": (100, 400),
                "io_uniform": None,
                "cpu_burst_count_range": (1, 1)
            }
        },
        "obs_noise_std": 0.4,
    },
    REGIME_8: {
        "type": REGIME_8,
        "arrival": {"type": "poisson", "lambda": 0.4},
        "mix": {"cpu": 0.7, "io": 0.3},
        "bursts": {
            "cpu": {
                "cpu_dist": {
                    "type": "pareto",
                    "alpha": 1.5,
                    "min": 50
                },
                "io_uniform": None,
                "cpu_burst_count_range": (1, 1)
            },
            "io": {
                "cpu_uniform": (2, 6),
                "io_uniform": (20, 60),
                "cpu_burst_count_range": (1, 3)
            }
        },
        "obs_noise_std": 0.2,
    },
    REGIME_9: {
        "type": REGIME_9,
        "arrival": {"type": "poisson", "lambda": 0.4},
        "mix": {"cpu": 0.7, "io": 0.3},
        "bursts": {
            "cpu": {
                "cpu_dist": {
                    "type": "pareto",
                    "alpha": 1.5,
                    "min": 50
                },
                "io_uniform": None,
                "cpu_burst_count_range": (1, 1)
            },
            "io": {
                "cpu_uniform": (2, 6),
                "io_uniform": (20, 60),
                "cpu_burst_count_range": (1, 3)
            }
        },
        "obs_noise_std": 0.5,
    },
}

