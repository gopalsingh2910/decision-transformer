import os
import pickle
import struct

from utils.validate import validate
from decision_transformer.param import SEQ_LEN

class DataLoader:
    def __init__(self, path=None):
        self.path = path
        self.current_episode = []
        pass

    def reset(self):
        self.current_episode = []
        pass

    def load_data(self, state, action, reward):
        self.current_episode.append((state, action, reward))
        pass

    def save_episode(self):
        is_valid, error = validate([self.current_episode])
        if not is_valid:
            if os.path.exists(self.path):
                os.remove(self.path)
            raise ValueError(f"cannot save episode {error}")

        # Ensure directory exists
        dir_path = os.path.dirname(self.path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)

        if len(self.current_episode) < SEQ_LEN:
            return

        idx_path = self.path + ".idx"
        with open(self.path, "ab") as f:
            offset = f.tell()
            pickle.dump(self.current_episode, f, protocol=pickle.HIGHEST_PROTOCOL)
        # Append offset to index file as 8-byte little-endian unsigned integer
        with open(idx_path, "ab") as idxf:
            idxf.write(struct.pack("<Q", offset))
        pass
