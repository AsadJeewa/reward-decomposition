import numpy as np
from morl_baselines.common.performance_indicators import hypervolume, sparsity, expected_utility

class RunningMeanStd:
    """
    Calculates a running mean and standard deviation for a data stream.
    This is a numerically stable implementation using Welford's online algorithm.
    """
    def __init__(self, reward_size = 1):
        self.mean = np.zeros(reward_size)
        self.var = np.ones(reward_size)
        self.count = 1e-4

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def normalize(self, x):
        # Clip the standard deviation to prevent division by zero
        std = np.sqrt(self.var).clip(min=1e-8)
        return np.clip((x - self.mean) / std, -10, 10)

def evaluate_agent_metrics(pareto_archive,ref_point, n_to_select=None):
    """
    Compute MORL performance metrics from Pareto archive.
    """

    if len(pareto_archive.evaluations) == 0:
        return {}

    # Get stored rewards and weights
    rewards = np.asarray(pareto_archive.evaluations)

    weights = np.asarray([
    ind[0] for ind in pareto_archive.individuals
    ])

    # Pareto filtering
    mask = np.ones(len(rewards), dtype=bool)

    for i in range(len(rewards)):
        for j in range(len(rewards)):
            if (
                np.all(rewards[j] >= rewards[i])
                and np.any(rewards[j] > rewards[i])
            ):
                mask[i] = False
                break

    front = rewards[mask]

    if len(front) == 0:
        return {}

    # Optional downsampling for expensive HV
    if n_to_select is not None and len(front) > n_to_select:
        indices = np.random.choice(
            len(front),
            n_to_select,
            replace=False
        )
        hv_points = front[indices]
    else:
        hv_points = front


    metrics = {
        "eval/hypervolume": hypervolume(
            ref_point=ref_point,
            points=hv_points
        ),

        "eval/sparsity": sparsity(front),

        "eval/eum": expected_utility(
            front,
            weights[mask]
        ),

        "eval/cardinality": len(front),
    }

    return metrics