import secrets
import uuid
import numpy as np
import torch
from ppo.agent import ContinuousAgent, DiscreteAgent
import mo_gymnasium as mo_gym
from morl_baselines.common.performance_indicators import hypervolume, sparsity, expected_utility
from morl_baselines.common.weights import equally_spaced_weights
from tqdm import tqdm
import pickle
import envs
import os
from envs.building_env import BuildingEnv_9d
from envs.utils_building import ParameterGenerator
from plot_utils import plot_preferences
from eval_utils import compute_all_controllability_metrics
import pandas as pd
from scipy.stats import spearmanr
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gymnasium.wrappers.vector import NormalizeObservation


# Pareto front calculation (robust and correct for maximization)
def pareto_front(points: np.ndarray) -> np.ndarray:
    n_points = points.shape[0]
    is_efficient = np.ones(n_points, dtype=bool)
    for i in range(n_points):
        for j in range(n_points):
            if all(points[j] >= points[i]) and any(points[j] > points[i]):
                is_efficient[i] = False
                break
    return is_efficient


def select_points_by_crowd_distance(pareto_points, n_to_select):
    """
    Selects a subset of points from a Pareto front using the crowd distance metric.

    This function identifies the N most spread-out points, which are crucial for
    getting a representative sample of the front, especially for computationally
    expensive tasks like hypervolume calculation.

    Args:
        pareto_points (np.ndarray): A NumPy array of shape (num_points, num_objectives)
                                    representing the points on the Pareto front.
        n_to_select (int): The number of points to select from the front.

    Returns:
        np.ndarray: A NumPy array of shape (n_to_select, num_objectives) containing
                    the selected points.
    """
    num_points, num_objectives = pareto_points.shape

    # If the number to select is greater than or equal to the number of points,
    # return all the points.
    if n_to_select >= num_points:
        return pareto_points

    # --- Step 1: Initialize Distances ---
    # Create an array to store the crowd distance for each point.
    crowding_distances = np.zeros(num_points)

    # --- Step 2: Loop Through Each Objective ---
    for i in range(num_objectives):
        # a. Sort points based on the current objective
        # We get the sorted indices to keep track of the original points
        sorted_indices = np.argsort(pareto_points[:, i])
        sorted_points = pareto_points[sorted_indices]

        # b. Assign infinite distance to boundary points
        # This ensures the extreme points of the front are always selected
        crowding_distances[sorted_indices[0]] = np.inf
        crowding_distances[sorted_indices[-1]] = np.inf
        
        # Get the min and max values for normalization
        min_val = sorted_points[0, i]
        max_val = sorted_points[-1, i]
        
        # Avoid division by zero if all values for an objective are the same
        if max_val == min_val:
            continue

        # c. Calculate distance for interior points
        for j in range(1, num_points - 1):
            distance = sorted_points[j + 1, i] - sorted_points[j - 1, i]
            normalized_distance = distance / (max_val - min_val)
            
            # d. Add to the total crowd distance
            crowding_distances[sorted_indices[j]] += normalized_distance

    # --- Step 3: Select the Top N Points ---
    # Sort the original indices based on the calculated crowding distances in descending order
    top_n_indices = np.argsort(crowding_distances)[::-1]

    # Select the first n_to_select indices from the sorted list
    selected_indices = top_n_indices[:n_to_select]

    # Return the corresponding points
    return pareto_points[selected_indices]

def normalize_returns(returns, max_r=None, min_r=None):
    """
    Normalise each objective independently to [0,1].
    returns: (N, D)
    """
    if min_r is None:
        min_r = returns.min(axis=0)
    if max_r is None:
        max_r = returns.max(axis=0)
    print("MIN MAX: ", returns.min(axis=0), returns.max(axis=0))

    return (returns - min_r) / (max_r - min_r + 1e-8)
# Set up vectorized env
env_id = "minecart-v0"  # or "mo-reacher-v5"
# env_id = "mo-humanoid-v5"  # or "mo-reacher-v5"
# env_id = "fruit-tree-v0"  # or "mo-reacher-v5"
reward_size = 3
num_eval_weights = 100
num_eval_episodes = 10
num_envs = num_eval_episodes
labels = [str(i) for i in range(reward_size)]  # Adjust based on the environment

if env_id == "deep-sea-treasure-v0":
    ref_point = np.array([0.0, -50.0])
elif env_id == "minecart-v0":
    ref_point = np.array([-1, -1, -200.0])
elif env_id == "mo-reacher-v5":
    ref_point = np.array([-50, -50, -50, -50]),
else:
    print("Please specify a reference point for the environment")
    exit()
gamma = 0.99
n_to_select = 2048

model_path = "runs/minecart-v0__main_ppo__2026-08-07 11.02.05.314703__1__positive/"

with open(model_path + "hparams.json", "r") as f:
    hparams = json.load(f)

training_seed = hparams["seed"]
if not os.path.exists(f"results/{env_id}"):
    os.makedirs(f"results/{env_id}", exist_ok=True)

if env_id == "building":
    # Special case for BuildingEnv_9d
    vec_envs = mo_gym.wrappers.vector.MOSyncVectorEnv(
        lambda: BuildingEnv_9d(ParameterGenerator(Building='OfficeLarge', Weather='Warm_Marine', Location='ElPaso')) 
        for _ in range(num_envs)
    )
else:
    vec_envs = mo_gym.wrappers.vector.MOSyncVectorEnv(
        [lambda: mo_gym.make(env_id, max_episode_steps = 1000) for _ in range(num_envs)]
    )


try: 
    norm_stats = pickle.load(open(model_path + "norm_stats.pkl", "rb"))
    print(norm_stats)
    mean = norm_stats.mean
    std = np.sqrt(norm_stats.var)
except:
    mean = np.zeros(vec_envs.single_observation_space.shape)
    std = np.ones(vec_envs.single_observation_space.shape)
vec_envs = mo_gym.wrappers.vector.MORecordEpisodeStatistics(vec_envs)

# Agent
if env_id == "building":
    # Special case for BuildingEnv_9d
    env_temp = BuildingEnv_9d(ParameterGenerator(Building='OfficeLarge', Weather='Warm_Marine', Location='ElPaso'))
else:
    env_temp = mo_gym.make(env_id)

if env_temp.action_space.__class__.__name__ == "Box":
    eval_agent = ContinuousAgent(env_temp, reward_size=reward_size).to("cpu")
else:
    eval_agent = DiscreteAgent(env_temp, reward_size=reward_size).to("cpu")
    
eval_agent.load_state_dict(torch.load(model_path + "main_ppo.rl_model"))
# eval_agent.eval()
n_points = 30
exp_note = "default"
right_angled = True
run_id = secrets.token_urlsafe(4)[:6]
plot_preferences(run_id=run_id, seed=training_seed, agent=eval_agent, env=env_temp, algo="d3po", n_points=30, exp_note=exp_note, right_angled=right_angled)

# Evaluation preferences: fixed, well-distributed reference directions
weights = equally_spaced_weights(
    dim=reward_size,
    n=num_eval_weights,
    seed=1000
)

# Buffers
rewards_list = []
weights_list = []

total_episodes = len(weights) * num_eval_episodes
pbar = tqdm(total=total_episodes, desc="Evaluating")

for weight_idx, weight in enumerate(weights):
    print(f"Evaluating preference {weight_idx + 1}/{num_eval_weights}")

    episode_returns = []

    obs, _ = vec_envs.reset()
    # Same preference across all parallel environments

    curr_weights = torch.tensor(
        np.tile(weight, (num_envs, 1)),
        dtype=torch.float32
    )

    env_rewards = np.zeros(
        (num_envs, reward_size),
        dtype=np.float32
    )

    gammas = np.ones((num_envs, 1))

    # Track which environments have completed

    finished = np.zeros(num_envs, dtype=bool)

    while not np.all(finished):
        obs = (obs - mean) / (std + 1e-8)

        actions, _ = eval_agent.predict(
            obs,
            curr_weights,
            deterministic=True,
            device="cpu"
        )

        next_obs, rews, dones, truncs, infos = vec_envs.step(actions)

        env_rewards += gammas * rews
        gammas *= gamma

        terminations = np.logical_or(dones, truncs)

        # Only record environments that finish this episode
        newly_finished = terminations & ~finished

        if np.any(newly_finished):
            episode_returns.extend(
                env_rewards[newly_finished]
            )
            finished[newly_finished] = True

        obs = next_obs

    # Mean return across the 10 parallel episodes
    mean_return = np.mean(
        np.asarray(episode_returns),
        axis=0
    )

    rewards_list.append(mean_return)
    weights_list.append(weight)

    pbar.update(num_eval_episodes)

pbar.close()
rewards_list = np.asarray(rewards_list)
weights_list = np.asarray(weights_list)

# Additional evaluation with extreme (one-hot) weights
print("Evaluating on extreme (one-hot) preference weights...")
extreme_rewards = []
extreme_weights = []

metrics = compute_all_controllability_metrics(weights_list, rewards_list)
print("Preference controllability:", metrics["preference_controllability"])
print("Local sensitivity:", metrics["local_sensitivity"])
print("Objective controllability:", [v for k, v in metrics.items() if k.startswith("objective_controllability")])

# rewards_list = np.vstack(rewards_list)
# weights_list = np.vstack(weights_list)

mask = pareto_front(rewards_list)
front = rewards_list[mask]
dominated = rewards_list[~mask]
# print(weights_list[mask])
# print(front)
print("Pareto front shape:", front.shape)

# Hypervolume and sparsity
# ref_point = front.min(axis=0) - 1e-6
import itertools

n_obj = front.shape[1]
pairs = list(itertools.combinations(range(reward_size), 2))

print(pairs)

for i, j in pairs:
    xlabel = labels[i] if i < len(labels) else f"Objective {i}"
    ylabel = labels[j] if j < len(labels) else f"Objective {j}"
    
    print(f"Plotting {xlabel} vs {ylabel} for Pareto front and dominated points...")
    # 1. Pareto front vs. Dominated Points
    plt.figure(figsize=(7,5))
    plt.scatter(dominated[:, i], dominated[:, j], alpha=0.4, label="Dominated", color="blue")
    plt.scatter(front[:, i], front[:, j], alpha=0.8, label="Pareto front", color="red",
                marker='o', edgecolors='k', s=60)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"Pareto Front vs. Dominated Points ({xlabel} vs. {ylabel})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"results/{env_id}/pareto_front_vs_dominated_{env_id}_({xlabel} vs. {ylabel}).png", dpi=150)
    plt.close()
    
    # 2. Pareto front only
    plt.figure(figsize=(7,5))
    plt.scatter(front[:, i], front[:, j], alpha=0.8, label="Pareto front", color="red",
                marker='o', edgecolors='k', s=60)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"Pareto Front ({xlabel} vs. {ylabel})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"results/{env_id}/pareto_front_{env_id}_({xlabel} vs. {ylabel}).png", dpi=150)
    plt.close()

filtered_front = select_points_by_crowd_distance(front, n_to_select)
hv = hypervolume(ref_point=ref_point, points=filtered_front)
sprs = sparsity(front)
print("Hypervolume of Pareto front:", hv)
print("Sparsity of Pareto front:", sprs)
print("Expected utility of Pareto front:", expected_utility(front, weights_list))

print(np.max(front, axis=0))
pickle.dump({
    "rewards": rewards_list,
    "weights": weights_list,
    "mask": mask,
    "pareto_front": front,
    "hypervolume": hv,
    "sparsity": sprs,
    "expected_utility": expected_utility(front, weights_list[mask]),
}, open(f"results/{env_id}/eval_results_{env_id}.pkl", "wb"))

data = {
    "run_id": run_id,
    "training_seed": training_seed,
    "hypervolume": hv,
    "sparsity": sprs,
    "expected_utility": expected_utility(front, weights_list[mask]),
    "preference_controllability": CO,
    "local_sensitivity": local_sensitivity
    }

# Add one column per objective
for d, score in enumerate(objective_control):
    data[f"objective_controllability_{d}"] = score
df = pd.DataFrame([data])
filepath = f"results/{env_id}/metrics_d3po_{exp_note}.csv"
df.to_csv(filepath, mode="a", index=False, header=not os.path.isfile(filepath))