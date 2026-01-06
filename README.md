# CPU Scheduler — Decision Transformer Simulation

This repository implements a CPU scheduling simulator and a Decision Transformer based agent for learning scheduling policies from generated episodes.

## Overview
- Simulator: event-driven CPU/workload simulator that models process arrivals, CPU bursts, IO bursts, queues and rewards.
- Data generation: produces episode data used to train a Decision Transformer model.
- Learning: a transformer-based policy is trained/tested on generated episodes.

## Requirements
Install dependencies from:

```
pip install -r requirements.txt
```

Key packages: `numpy`, `torch`, `matplotlib`, `tqdm`.

## Quick start
1. Generate training/testing episodes (optional — `utils.loader` will auto-generate if checkpoint missing):

```
python -c "from utils.loader import generate_data; generate_data(epoches_num=1000, path=None)"
```

2. Run the main script (loads model or trains if no checkpoint):

```
python run.py
```

Notes: `run.py` calls `utils.loader.load_model()` then `decision_transformer.run(model)`. Uncommented lines in `run.py` show how to generate data, train, save, and test.

## Project structure
- `run.py` — entry point to load model and start/run training or evaluation.
- `requirements.txt` — Python dependencies.
- `decision_transformer/` — model code and dataset utilities
  - `data_loader.py` — dataset reader and batch sampler
  - `param.py` — DT hyperparameters (sequence length, batch size, model dims)
- `simulator/` — environment and workload generator
  - `environ.py` — environment wrapper and reward logic
  - `workload.py`, `process.py`, `regimes.py` — workload/process generation
  - `_cpu_/` — CPU primitives and policy heuristics (`QUEUE_CAP`, `TIMER_INTERRUPT`, and `policy.py` rules)
- `utils/` — helpers
  - `loader.py` — data generation, model load/save, training orchestration
  - `paths.py` — default filesystem paths for data and checkpoint
  - `plot.py`, `validate.py` — utilities for results and validation
- `_data_/` — storage for `training_data` and `testing_data` and model checkpoint

## Configuration
- Data & checkpoints paths are defined in `utils.paths` (`TRAINING_DATA_PATH`, `TESTING_DATA_PATH`, `CHECKPOINT_PATH`).
- Main DT hyperparameters live in `decision_transformer/param.py` (e.g., `SEQ_LEN`, `BATCH_SIZE`, `D_MODEL`).
- CPU-related constants are in `simulator/_cpu_/param.py` (e.g., `QUEUE_CAP`, `TIMER_INTERRUPT`).

## How it works (high level)
- The simulator (`simulator.eniron` + helpers) steps a clock, admits jobs, runs selected process according to an action, updates blocked queue and collects rewards.
- `utils.loader.generate_data` runs the simulator using heuristic policies (FCFS, SJF, RR, SRTF) to produce episodes stored as pickled files.
- `decision_transformer` trains a transformer on these episodes to predict actions conditioned on returns-to-go and state history.

## Extending / Next steps
- Adjust regimes in `simulator/regimes.py` to change arrival/burst distributions.
- Tune transformer params in `decision_transformer/param.py` and training steps in `utils.loader`.
- Add tests or CI to validate training/data generation reproducibility.

## Where to look first
- To generate data: `utils/loader.py` → `generate_data`
- To run / train: `run.py` and `decision_transformer` package
- For environment details: `simulator/environ.py`

If you'd like, I can expand this README with example plots, expected outputs, or a short tutorial on training and evaluation.

![comparing the algorithms](https://github.com/gopalsingh2910/decision-transformer/blob/main/results/graph.png)
