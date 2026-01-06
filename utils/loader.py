import os
import torch
import pickle
import numpy as np

import simulator
import decision_transformer as DT

from decision_transformer import param as p
from simulator._cpu_ import policy
from .paths import TRAINING_DATA_PATH, TESTING_DATA_PATH, CHECKPOINT_PATH, trim_path

def generate_data(epoches_num=10000, path=None, renew=False):
    print(f'Generating {epoches_num} episodes of data...')
    if path is not None and not renew:
        if os.path.exists(path):
            with open(path, 'rb') as f:
                content = f.read(1)
                if content:
                    print(f'Data already exists at {trim_path(path)}.')
                    return
    elif path is None:
        print(f'Error: Path does not exists.')
        generate_data(epoches_num, TRAINING_DATA_PATH, renew=renew)
        generate_data(epoches_num, TESTING_DATA_PATH, renew=renew)
        return
    
    if path is not None and renew:
        if os.path.exists(path):
            os.remove(path)

    epoches_num = int(epoches_num)
    env = simulator.make_env(pre_emptive=True, path=path)
    for i in range(epoches_num):
        env.reset()
        done = False
        while not done:
            # Bias toward stronger heuristics for better training targets
            rule = np.random.choice(
                ['FCFS', 'SJF', 'RR', 'SRTF'],
                p=[0.4, 0.4, 0.1, 0.1]
            )
            action = policy.act(env=env, rule=rule)
            _, _, done = env.step(action)
        if (i + 1) % 50 == 0:
            print(f'Episodes {i + 1}/{epoches_num} generated')
    pass

def load_model(path=None):
    model = DT.get_instance()
    if path is None:
        path = CHECKPOINT_PATH
        print(f'Loading default model instance...')
    else:
        print(f'Loading model instance from {trim_path(path)}...')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(path, map_location=device) if os.path.exists(path) else None
    if checkpoint is not None and isinstance(checkpoint, dict) and ('model_state' in checkpoint) and ('optimizer_state' in checkpoint):
        m, o = model['model'], model['optimizer']
        result = m.load_state_dict(checkpoint["model_state"], strict=False)
        # Log any missing or unexpected keys due to architecture changes
        if hasattr(result, 'missing_keys') and result.missing_keys:
            print(f"Warning: missing model keys: {result.missing_keys}")
        if hasattr(result, 'unexpected_keys') and result.unexpected_keys:
            print(f"Warning: unexpected model keys: {result.unexpected_keys}")
        try:
            o.load_state_dict(checkpoint["optimizer_state"])
        except Exception as e:
            print(f"Warning: optimizer state load failed: {e}. Continuing with fresh optimizer.")
        print(f'Loaded model from checkpoint at {trim_path(path)}.')
        new_model = {'model': m, 'optimizer': o}
    else:
        print(f'Checkpoint at {trim_path(path)} is invalid or empty.')
        print(f'Creating new model instance...')
        generate_data(epoches_num=p.EPISODES_NUM*p.TRAINING_FRACTION, path=TRAINING_DATA_PATH)
        generate_data(epoches_num=p.EPISODES_NUM*p.TESTING_FRACTION, path=TESTING_DATA_PATH)
        new_model = DT.train(model, TRAINING_DATA_PATH, int(p.EPISODES_NUM*p.TRAINING_FRACTION))
        save_model(new_model, path=path)
        DT.test(new_model, TESTING_DATA_PATH, int(p.EPISODES_NUM*p.TESTING_FRACTION))
    return new_model

def save_model(model, path=None):
    if path is None:
        path = CHECKPOINT_PATH
        print(f'Saving model to default path {trim_path(path)}...')
    else:
        print(f'Saving model to {trim_path(path)}...')

    m, o = model['model'], model['optimizer']
    checkpoint = {
        "model_state": m.state_dict(),
        "optimizer_state": o.state_dict()
    }
    torch.save(checkpoint, path)
    print(f'Model saved to {trim_path(path)}.')
