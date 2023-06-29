import numpy as np


def mask(policy: np.ndarray, valid_actions: np.ndarray) -> np.ndarray:
    policy = policy * valid_actions
    if policy.sum() > 0:
        return policy / policy.sum()
    else:
        return valid_actions / valid_actions.sum()
