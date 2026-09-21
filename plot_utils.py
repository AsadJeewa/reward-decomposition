import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr, spearmanr


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


def evaluate_line(run_id, seed, agent, algo, env, n_points=50, exp_note=""):
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

            ep_return += vec_reward
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
    df.to_csv(
        filepath,
        mode="a",
        index=False,
        header=not Path(filepath).is_file(),
    )

    return df


def evaluate_simplex(
    run_id,
    seed,
    agent,
    algo,
    env,
    n_points=10,
    exp_note="",
    right_angled=True,
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
    df.to_csv(
        filepath,
        mode="a",
        index=False,
        header=not Path(filepath).is_file(),
    )

    return df


def plot_mean_line(weights, mean_returns, std_returns, algo, env_id, exp_note=""):
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

    treasure_returns = np.array(
        [
            [0.7, -1],
            [8.2, -3],
            [11.5, -5],
            [14.0, -7],
            [15.1, -8],
            [16.1, -9],
            [19.6, -13],
            [20.3, -14],
            [22.4, -17],
            [23.7, -19],
        ]
    )

    if mean_returns.shape[1] == 2 and "deep-sea-treasure" in env_id:
        optimal_returns = []

        for ti in t:
            w = np.array([ti, 1.0 - ti])
            utilities = treasure_returns @ w
            best = np.argmax(utilities)
            optimal_returns.append(treasure_returns[best])

        optimal_returns = np.array(optimal_returns)

        for i in range(2):
            plt.plot(
                t,
                optimal_returns[:, i],
                linestyle="--",
                label=f"Optimal Obj {i}",
            )

    plt.xlabel("t (w0)")
    plt.ylabel("Return")
    plt.title(f"{algo} Preference Line - Mean +/- Std")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"results/{env_id}/pref_line_{algo}_{exp_note}_mean.png")
    plt.close()


def plot_mean_simplex(
    weights,
    mean_returns,
    std_returns,
    algo,
    env_id,
    exp_note="",
    right_angled=True,
):
    num_obj = mean_returns.shape[1]

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

            if statistic == "mean" and i == 2 and np.all(values[:, i] < 0):
                cbar_label = f"Objective {i} -log(-return)"
                vals = -np.log(-vals + 1e-6)
            else:
                cbar_label = f"Objective {i} return"

            alpha = np.where(vals == 0, 1.0, 0.6)

            sc = axes[i].scatter(
                plot_x,
                plot_y,
                c=vals,
                s=20,
                cmap="viridis",
                alpha=alpha,
            )

            axes[i].set_title(f"Objective {i} {statistic}")
            _decorate_axis(axes[i])
            plt.colorbar(sc, ax=axes[i], label=cbar_label)

        plt.tight_layout()
        plt.savefig(
            f"results/{env_id}/pref_simplex_{algo}_{exp_note}_{statistic}.png"
        )
        plt.close()

    _plot_simplex(mean_returns, "mean")
    _plot_simplex(std_returns, "std")

    for i in range(num_obj):
        vals = mean_returns[:, i].copy()

        fig, ax = plt.subplots(figsize=(6, 5))

        alpha = np.where(vals == 0, 1.0, 0.6)

        sc = ax.scatter(
            plot_x,
            plot_y,
            c=vals,
            s=20,
            cmap="viridis",
            alpha=alpha,
        )

        ax.set_title(f"{algo} Objective {i} Mean Return")
        _decorate_axis(ax)
        plt.colorbar(sc, ax=ax, label=f"Objective {i} return")

        plt.tight_layout()
        plt.savefig(
            f"results/{env_id}/pref_simplex_{algo}_{exp_note}_obj{i}_raw.png"
        )
        plt.close()


def evaluate_preferences(
    run_id,
    seed,
    agent,
    env,
    algo,
    n_points=50,
    exp_note="",
    right_angled=True,
):
    if env.unwrapped.reward_dim == 2:
        return evaluate_line(
            run_id,
            seed,
            agent,
            algo,
            env,
            n_points=n_points,
            exp_note=exp_note,
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
    weights,
    mean_returns,
    std_returns,
    algo,
    env,
    exp_note="",
    right_angled=True,
):
    if env.unwrapped.reward_dim == 2:
        plot_mean_line(
            weights,
            mean_returns,
            std_returns,
            algo,
            env.spec.id,
            exp_note,
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

        print(
            f"Obj {obj} | Pearson: {p_corr:.3f} | "
            f"Spearman: {s_corr:.3f}"
        )

    fig, axes = plt.subplots(
        1,
        num_obj,
        figsize=(5 * num_obj, 4),
    )

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
    plt.savefig(
        f"results/{env.spec.id}/weight_return_scatter_{algo}_{exp_note}.png"
    )
    plt.close()