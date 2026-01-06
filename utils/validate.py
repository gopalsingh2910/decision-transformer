import numpy as np

from decision_transformer.param import STATE_DIM

def validate(episodes):
    error = None
    for i, ep in enumerate(episodes):
        for t, transition in enumerate(ep):
            if len(transition) != 3:
                error = f"episode {i}, transition {t}: contains {len(transition)} elements, expected 3"
            state, action, reward = transition
            if not isinstance(state, (list, np.ndarray)):
                error = f"episode {i}, transition {t}: state is {type(state)}, expected list/np.ndarray "
            if len(state) != STATE_DIM:
                error = f"episode {i}, transition {t}: state dimension is {len(state)}, expected {STATE_DIM}"
            if not isinstance(action, (int, np.integer)):
                error = f"episode {i}, transition {t}: action is {type(action)}, expected int/np.integer"
            if not isinstance(reward, (int, float, np.integer, np.floating)):
                error = f"episode {i}, transition {t}: reward is {type(reward)}, expected int/np.integer or float/np.float"
            if error is not None:
                return False, error
    return True, None