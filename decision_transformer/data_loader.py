import os
import pickle
import struct
import numpy as np

from .param import BATCH_SIZE, SEQ_LEN

class DatasetLoader:
    def __init__(self, path, num_episodes=0):
        self.path = path
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Episodes file not found: {self.path}")

        # Prefer reading index file; fall back to one-time scan
        idx_path = self.path + "+.idx" if self.path.endswith(".pkl") else self.path + ".idx"
        if not os.path.exists(idx_path):
            idx_path = self.path + ".idx"

        self.offsets = []
        if os.path.exists(idx_path):
            with open(idx_path, 'rb') as idxf:
                data = idxf.read()
                # Each offset is stored as 8-byte little-endian unsigned integer
                for (off,) in struct.iter_unpack('<Q', data):
                    self.offsets.append(off)
        else:
            with open(self.path, 'rb') as f:
                while True:
                    try:
                        self.offsets.append(f.tell())
                        pickle.load(f)
                    except EOFError:
                        break

        self.num_episodes = len(self.offsets) if num_episodes == 0 else min(num_episodes, len(self.offsets))

    def _load_episode(self, idx):
        with open(self.path, 'rb') as f:
            f.seek(self.offsets[idx])
            return pickle.load(f)

    def sample_batch(self, shuffle=True):
        states_batch = []
        actions_batch = []
        rewards_batch = []

        # Keep sampling random episodes until we fill the batch with valid windows
        while len(states_batch) < BATCH_SIZE:
            idx = np.random.randint(0, self.num_episodes)
            episode = self._load_episode(idx)
            if not episode or len(episode) < SEQ_LEN:
                continue

            full_states, full_actions, full_rewards = zip(*episode)
            full_states = np.asarray(full_states, dtype=float)
            full_actions = np.asarray(full_actions, dtype=int)
            full_rewards = np.asarray(full_rewards, dtype=float)

            # Returns-to-go, normalized (same logic as previous implementation)
            rtg = full_rewards[::-1].cumsum()[::-1] / 10.0

            start = np.random.randint(0, len(episode) - SEQ_LEN + 1)
            states = full_states[start:start + SEQ_LEN]
            actions = full_actions[start:start + SEQ_LEN]
            rewards = rtg[start:start + SEQ_LEN]

            states_batch.append(states)
            actions_batch.append(actions)
            rewards_batch.append(rewards)

        if shuffle:
            order = np.random.permutation(len(states_batch))
            states_batch = [states_batch[i] for i in order]
            actions_batch = [actions_batch[i] for i in order]
            rewards_batch = [rewards_batch[i] for i in order]

        return {
            "rtgs": np.stack(rewards_batch, axis=0),        # (B, K)
            "states": np.stack(states_batch, axis=0),       # (B, K, state_dim)
            "actions": np.stack(actions_batch, axis=0),     # (B, K)
        }