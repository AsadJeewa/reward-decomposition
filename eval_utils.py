import numpy as np
from scipy.stats import spearmanr


def compute_controllability(weights, returns):
    """
    Cosine preference controllability.
    Measures alignment between requested preferences and realised returns.
    """
    weights_norm = weights / (np.linalg.norm(weights, axis=1, keepdims=True) + 1e-8)
    returns_norm = returns / (np.linalg.norm(returns, axis=1, keepdims=True) + 1e-8)
    scores = np.sum(weights_norm * returns_norm, axis=1)
    return np.mean(scores)


def compute_objective_controllability(weights, returns):
    """
    Per-objective Spearman correlation between preference and return.
    """
    num_obj = returns.shape[1]
    objective_control = []
    for d in range(num_obj):
        if np.std(returns[:, d]) < 1e-8 or np.std(weights[:, d]) < 1e-8:
            objective_control.append(0.0)
        else:
            corr, _ = spearmanr(weights[:, d], returns[:, d])
            objective_control.append(corr)
    return np.array(objective_control)


def compute_local_sensitivity(weights, returns):
    """
    Average change in return per change in preference for nearest neighbour pairs.
    """
    sensitivities = []
    for i in range(len(weights)):
        distances = np.linalg.norm(weights - weights[i], axis=1)
        distances[i] = np.inf
        j = np.argmin(distances)
        dw = np.linalg.norm(weights[i] - weights[j])
        if dw < 1e-8:
            continue  # skip if weights are identical
        dr = np.linalg.norm(returns[i] - returns[j])
        sensitivities.append(dr / (dw + 1e-8))
    return np.mean(sensitivities)


def compute_all_controllability_metrics(weights, returns):
    """
    Compute all controllability metrics and return as a dict.
    """
    weights = np.array(weights)
    returns = np.array(returns)

    co = compute_controllability(weights, returns)
    obj_co = compute_objective_controllability(weights, returns)
    ls = compute_local_sensitivity(weights, returns)

    result = {
        "preference_controllability": co,
        "local_sensitivity": ls,
    }
    for d, score in enumerate(obj_co):
        result[f"objective_controllability_{d}"] = score

    return result

def normalize_returns(returns, min_r=None, max_r=None):
    """
    Normalise each objective independently to [0,1].
    returns: (N, D)
    """
    returns = np.array(returns)
    if min_r is None:
        min_r = returns.min(axis=0)
    if max_r is None:
        max_r = returns.max(axis=0)
    return (returns - min_r) / (max_r - min_r + 1e-8)