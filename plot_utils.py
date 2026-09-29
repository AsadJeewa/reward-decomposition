import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr, spearmanr

def get_dst_treasure_returns(gamma):
    treasures = np.array([0.7, 8.2, 11.5, 14.0, 15.1, 16.1, 19.6, 20.3, 22.4, 23.7])
    steps = np.array([1, 3, 5, 7, 8, 9, 13, 14, 17, 19])

    treasure_returns = gamma ** (steps - 1) * treasures

    if gamma == 1.0:
        time_returns = -steps.astype(float)
    else:
        time_returns = -(1 - gamma**steps) / (1 - gamma)

    return np.column_stack((treasure_returns, time_returns))


def sample_line(n_points):
    return np.linspace(0, 1, n_points)


def sample_simplex_grid(n_points):
    """
    Samples (t, s) over a 2D simplex:
    t >= 0, s >= 0, t + s <= 1
    """
    ts = []
    ss = []

    grid = np.linspace(0, 1, n_points)

    for t in grid:
        for s in grid:
            if t + s <= 1.0:
                ts.append(t)
                ss.append(s)

    return np.array(ts), np.array(ss)


def evaluate_line(run_id, seed, agent, algo, env, gamma, n_points=50, exp_note=""):
    algo_lower = algo.lower()

    is_d3po = "d3po" in algo_lower

    ts = sample_line(n_points)

    rows = []

    for t in ts:
        w = np.zeros(env.unwrapped.reward_dim, dtype=np.float32)
        w[0] = t
        w[1] = 1.0 - t

        print(f"Evaluating {algo}: w={np.round(w, 3)}")

        obs, _ = env.reset()
        done = False

        ep_return = np.zeros(env.unwrapped.reward_dim, dtype=np.float32)
        discount = 1.0
        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            w_tensor = torch.tensor(w, dtype=torch.float32)

            with torch.no_grad():
                if is_d3po:
                    action, _ = agent.predict(
                        obs_tensor,
                        w_tensor,
                        deterministic=True,
                        device="cpu",
                    )
                    action = np.asarray(action).item()
                else:
                    action = env.action_space.sample()

            obs, vec_reward, terminated, truncated, _ = env.step(action)

            ep_return += discount * vec_reward
            discount *= gamma
            done = terminated or truncated

        rows.append(
            {
                "run_id": run_id,
                "training_seed": seed,
                "algo": algo,
                "t": t,
                **{f"w{i}": wi for i, wi in enumerate(w)},
                **{f"r{i}": ri for i, ri in enumerate(ep_return)},
            }
        )

    df = pd.DataFrame(rows)

    filepath = f"results/{env.spec.id}/pref_line_{algo}_{exp_note}.csv"
    df.to_csv(filepath, mode="a", index=False, header=not Path(filepath).is_file())

    return df


def evaluate_simplex(
    run_id, seed, agent, algo, env, n_points=10, exp_note="", right_angled=True
):
    algo_lower = algo.lower()

    is_d3po = "d3po" in algo_lower

    ts, ss = sample_simplex_grid(n_points)
    rows = []

    for t, s in zip(ts, ss):
        w = np.zeros(env.unwrapped.reward_dim, dtype=np.float32)
        w[0] = t
        w[1] = s
        w[2] = 1.0 - t - s

        print(f"Evaluating w = {w}")

        obs, _ = env.reset()
        done = False

        ep_return = np.zeros(env.unwrapped.reward_dim, dtype=np.float32)

        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            w_tensor = torch.tensor(w, dtype=torch.float32)

            with torch.no_grad():
                if is_d3po:
                    action, _ = agent.predict(
                        obs_tensor,
                        w_tensor,
                        deterministic=True,
                        device="cpu",
                    )
                else:
                    action = env.action_space.sample()

            obs, vec_reward, terminated, truncated, _ = env.step(action)

            ep_return += vec_reward
            done = terminated or truncated

        rows.append(
            {
                "run_id": run_id,
                "training_seed": seed,
                "algo": algo,
                "t": t,
                "s": s,
                **{f"w{i}": wi for i, wi in enumerate(w)},
                **{f"r{i}": ri for i, ri in enumerate(ep_return)},
            }
        )

    df = pd.DataFrame(rows)

    filepath = f"results/{env.spec.id}/pref_simplex_{algo}_{exp_note}.csv"
    df.to_csv(filepath, mode="a", index=False, header=not Path(filepath).is_file())

    return df


def plot_mean_line(weights, mean_returns, std_returns, algo, env_id, gamma, exp_note=""):
    """
    Plot mean +/- std return for each preference across training runs.
    Uses returns already produced by the main evaluation pass.
    """
    t = weights[:, 0]
    order = np.argsort(t)

    t = t[order]
    mean_returns = mean_returns[order]
    std_returns = std_returns[order]

    plt.figure(figsize=(7, 5))

    for i in range(mean_returns.shape[1]):
        plt.plot(t, mean_returns[:, i], label=f"Obj {i}")

        plt.fill_between(
            t,
            mean_returns[:, i] - std_returns[:, i],
            mean_returns[:, i] + std_returns[:, i],
            alpha=0.2,
        )

    # Ground-truth optimal return for DST
    treasure_returns = get_dst_treasure_returns(gamma)

    if mean_returns.shape[1] == 2 and "deep-sea-treasure" in env_id:
        optimal_returns = []
        optimal_treasures = []
        for ti in t:
            w = np.array([ti, 1.0 - ti])
            utilities = treasure_returns @ w
            best = np.argmax(utilities)
            optimal_returns.append(treasure_returns[best])
            optimal_treasures.append(best + 1)
        optimal_returns = np.array(optimal_returns)
        optimal_treasures = np.array(optimal_treasures)
        for i in range(2):
            plt.plot(t, optimal_returns[:, i], linestyle="--", label=f"Optimal Obj {i}")

    plt.xlabel("Treasure value weight $w_0$" if "deep-sea-treasure" in env_id else "t (w0)")
    plt.ylabel("Return")
    plt.title("Mean Preference Line")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"results/{env_id}/pref_line_{algo}_{exp_note}_mean.png")
    plt.close()
    if mean_returns.shape[1] == 2 and "deep-sea-treasure" in env_id:
        distances = np.linalg.norm(
            mean_returns[:, None, :] - treasure_returns[None, :, :],
            axis=2,
        )
        learned_treasures = np.argmin(distances, axis=1) + 1
        plt.figure(figsize=(7, 4))
        plt.step(
            t, optimal_treasures, where="mid", linewidth=2, label="Optimal partition"
        )
        plt.step(
            t, learned_treasures, where="mid", linewidth=2, label="Learned partition"
        )
        plt.xlabel("Treasure value weight $w_0$")
        plt.ylabel("Treasure regime")
        plt.yticks(range(1, 11), [f"T{i}" for i in range(1, 11)])
        plt.xlim(0, 1)
        plt.legend()
        plt.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"results/{env_id}/pref_partition_{algo}_{exp_note}_mean.png")
        plt.close()
        weights_line = np.column_stack([t, 1.0 - t])
        optimal_utility = np.max(weights_line @ treasure_returns.T, axis=1)
        learned_utility = np.sum(weights_line * mean_returns, axis=1)
        regret = optimal_utility - learned_utility
        plt.figure(figsize=(7, 4))
        plt.plot(t, regret)
        plt.xlabel("Treasure value weight $w_0$")
        plt.ylabel("Scalarised regret")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"results/{env_id}/pref_regret_{algo}_{exp_note}_mean.png")
        plt.close()


def plot_mean_simplex(
    weights, mean_returns, std_returns, algo, env_id, exp_note="", right_angled=True
):
    num_obj = mean_returns.shape[1]

    # Shared RAW RETURN scales across algorithms
    mean_vmin = [0.0, 0.0, -3.9734]
    mean_vmax = [1.0494, 1.0330, -0.2809]

    w0 = weights[:, 0]
    w1 = weights[:, 1]
    w2 = weights[:, 2]

    def _get_plot_coordinates():
        if right_angled:
            return w0, w1

        x = w1 + 0.5 * w2
        y = (np.sqrt(3) / 2.0) * w2
        return x, y

    plot_x, plot_y = _get_plot_coordinates()

    def _decorate_axis(ax):
        ax.set_aspect("equal", adjustable="box")

        if right_angled:
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 1.02)
            ax.set_xlabel("w0")
            ax.set_ylabel("w1")
        else:
            ax.set_xlim(-0.03, 1.03)
            ax.set_ylim(-0.03, np.sqrt(3) / 2.0 + 0.03)

            ax.text(0.0, -0.035, "w0=1", ha="center", va="top")
            ax.text(1.0, -0.035, "w1=1", ha="center", va="top")
            ax.text(
                0.5,
                np.sqrt(3) / 2.0 + 0.02,
                "w2=1",
                ha="center",
                va="bottom",
            )

            ax.set_xticks([])
            ax.set_yticks([])

    def _plot_simplex(values, statistic):
        fig, axes = plt.subplots(
            1,
            num_obj,
            figsize=(6 * num_obj, 5),
        )

        if num_obj == 1:
            axes = [axes]

        for i in range(num_obj):
            vals = values[:, i].copy()

            # Reverse objective 2 because it is minimised.
            cmap = "viridis_r" if i == 2 else "viridis"

            if statistic == "mean":
                plot_vmin = mean_vmin[i]
                plot_vmax = mean_vmax[i]
                cbar_label = f"Objective {i} return"
            else:
                # Standard deviations are non-negative.
                plot_vmin = 0.0
                plot_vmax = std_returns[:, i].max()
                cbar_label = f"Objective {i} return std."

                # For std, larger is not better/worse, so don't reverse.
                cmap = "viridis"

            sc = axes[i].scatter(
                plot_x,
                plot_y,
                c=vals,
                s=20,
                vmin=plot_vmin,
                vmax=plot_vmax,
                cmap=cmap,
                alpha=0.7,
            )

            axes[i].set_title(f"Objective {i} {statistic}")
            _decorate_axis(axes[i])
            plt.colorbar(sc, ax=axes[i], label=cbar_label)

        plt.tight_layout()
        plt.savefig(f"results/{env_id}/pref_simplex_{algo}_{exp_note}_{statistic}.png")
        plt.close()

    _plot_simplex(mean_returns, "mean")
    _plot_simplex(std_returns, "std")

    # Separate raw mean-return plots
    for i in range(num_obj):
        vals = mean_returns[:, i].copy()
        cmap = "viridis_r" if i == 2 else "viridis"

        fig, ax = plt.subplots(figsize=(6, 5))

        sc = ax.scatter(
            plot_x,
            plot_y,
            c=vals,
            s=20,
            vmin=mean_vmin[i],
            vmax=mean_vmax[i],
            cmap=cmap,
            alpha=0.7,
        )

        ax.set_title(f"Objective {i} Mean Return")
        _decorate_axis(ax)
        plt.colorbar(sc, ax=ax, label=f"Objective {i} return")

        plt.tight_layout()
        plt.savefig(f"results/{env_id}/pref_simplex_{algo}_{exp_note}_obj{i}_raw.png")
        plt.close()

def evaluate_preferences(
    run_id, seed, agent, env, algo, gamma, n_points=50, exp_note="", right_angled=True
):
    """
    Evaluate one training seed and return the preference/return DataFrame.
    Plotting across seeds is handled separately by plot_mean_preferences().
    """
    if env.unwrapped.reward_dim == 2:
        return evaluate_line(
            run_id, seed, agent, algo, env, gamma, n_points=n_points, exp_note=exp_note
        )

    elif env.unwrapped.reward_dim == 3:
        return evaluate_simplex(
            run_id,
            seed,
            agent,
            algo,
            env,
            n_points=n_points,
            exp_note=exp_note,
            right_angled=right_angled,
        )

    else:
        raise ValueError(
            "evaluate_preferences currently supports only 2 or 3 objectives."
        )


def plot_mean_preferences(
    weights, mean_returns, std_returns, algo, env, gamma, exp_note="", right_angled=True
):
    """
    Plot aggregate preference-response results using the single evaluation pass.
    """
    if env.unwrapped.reward_dim == 2:
        plot_mean_line(
            weights, mean_returns, std_returns, algo, env.spec.id, gamma, exp_note
        )
    elif env.unwrapped.reward_dim == 3:
        plot_mean_simplex(
            weights,
            mean_returns,
            std_returns,
            algo,
            env.spec.id,
            exp_note,
            right_angled=right_angled,
        )

    else:
        raise ValueError(
            "plot_mean_preferences currently supports only 2 or 3 objectives."
        )


def plot_correlations(env, algo, all_weights, all_returns, exp_note=""):
    num_obj = all_returns.shape[1]

    print("\n=== Correlations ===")

    for obj in range(num_obj):
        p_corr, _ = pearsonr(all_weights[:, obj], all_returns[:, obj])
        s_corr, _ = spearmanr(all_weights[:, obj], all_returns[:, obj])

        print(f"Obj {obj} | Pearson: {p_corr:.3f} | Spearman: {s_corr:.3f}")

    fig, axes = plt.subplots(1, num_obj, figsize=(5 * num_obj, 4))

    if num_obj == 1:
        axes = [axes]

    for i in range(num_obj):
        x = all_weights[:, i]
        y = all_returns[:, i]

        axes[i].scatter(x, y)

        coeffs = np.polyfit(x, y, 1)
        line = np.poly1d(coeffs)

        xs = np.linspace(x.min(), x.max(), 100)
        axes[i].plot(xs, line(xs))

        axes[i].set_xlabel(f"w[{i}]")
        axes[i].set_ylabel(f"r[{i}]")
        axes[i].set_title(f"Obj {i}")

    plt.tight_layout()
    plt.savefig(f"results/{env.spec.id}/weight_return_scatter_{algo}_{exp_note}.png")
    plt.close()