import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

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


def evaluate_line(agent, algo, env, n_points=50, exp_note=""):
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
                    actions, _ = agent.predict(obs_tensor, w_tensor, deterministic=True, device="cpu")

                else:
                    actions = env.action_space.sample()

            obs, vec_reward, terminated, truncated, _ = env.step(actions)

            ep_return += vec_reward
            done = terminated or truncated

        rows.append({
            "algo": algo,
            "t": t,
            **{f"w{i}": wi for i, wi in enumerate(w)},
            **{f"r{i}": ri for i, ri in enumerate(ep_return)}
        })

    df = pd.DataFrame(rows)
    df.to_csv(f"results/{env.spec.id}/pref_line_{exp_note}.csv", index=False)

    plt.figure(figsize=(7, 5))

    r_cols = [c for c in df.columns if c.startswith("r")]

    for i, r in enumerate(r_cols):
        plt.plot(df["t"], df[r], label=f"Obj {i}")

    plt.xlabel("t (w0)")
    plt.ylabel("Return")
    plt.title(f"{algo} Preference Line")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"results/{env.spec.id}/pref_line_{exp_note}.png")
    plt.close()


def evaluate_simplex(agent, algo, env, n_points=10, exp_note="", right_angled=True):
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
        i=0
        while not done:

            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            w_tensor = torch.tensor(w, dtype=torch.float32)

            with torch.no_grad():

                if is_d3po:
                    actions, _ = agent.predict(obs_tensor, w_tensor, deterministic=True, device="cpu")

                else:
                    actions = env.action_space.sample()
            obs, vec_reward, terminated, truncated, _ = env.step(actions)
            # print(action) # only 4
            i+=1

            ep_return += vec_reward
            done = terminated or truncated

        rows.append({
            "algo": algo,
            "t": t,
            "s": s,
            **{f"w{i}": wi for i, wi in enumerate(w)},
            **{f"r{i}": ri for i, ri in enumerate(ep_return)}
        })

    df = pd.DataFrame(rows)
    df.to_csv(f"results/{env.spec.id}/pref_simplex_{exp_note}.csv", index=False)
    
    r_cols = [c for c in df.columns if c.startswith("r")]
    num_obj = len(r_cols)
    print("plot")
    fig, axes = plt.subplots(
        1,
        num_obj,
        figsize=(6 * num_obj, 5),
        sharex=True,
        sharey=True,
    )

    if num_obj == 1:
        axes = [axes]

    for i, r in enumerate(r_cols):

        vals = df[r]
        alpha = np.where(vals == 0, 1.0, 0.6) # highlight 0 to be fully opaque
        if i == 2:
            vals = -np.log(-vals + 1e-6)  # only if all vals < 0

        if right_angled:
            sc = axes[i].scatter(
                df["t"],
                df["s"],
                c=vals,
                s=40,
                cmap="viridis",
                alpha=alpha,
            )

            # axes[i].set_xlim(0, 1)
            # axes[i].set_ylim(0, 1)
            axes[i].set_title(f"Objective {i}")
            axes[i].set_xlabel("t (w0)")
            axes[i].set_ylabel("s (w1)")

        else:
            w0 = ts
            w1 = ss
            w2 = 1 - ts - ss

            x = w1 + 0.5 * w2
            y = (np.sqrt(3) / 2) * w2

            sc = axes[i].scatter(
                x,
                y,
                c=df[r],
                cmap="viridis"
            )
            axes[i].text(0, -0.035, "w0=1")
            axes[i].text(1, -0.035, "w1=1")
            axes[i].text(0.5, np.sqrt(3)/2, "w2=1")

        plt.colorbar(sc, ax=axes[i])

    plt.tight_layout()
    print("save fig")
    plt.savefig(f"results/{env.spec.id}/pref_simplex_{exp_note}.png")
    plt.close()


def plot_preferences(agent, algo, env, n_points=50, exp_note="", right_angled=True):
    if env.unwrapped.reward_dim == 2:
        evaluate_line(
            agent,
            algo,
            env,
            n_points=n_points,
            exp_note=exp_note
        )
    elif env.unwrapped.reward_dim == 3:
        evaluate_simplex(
            agent,
            algo,
            env,
            n_points=n_points,
            exp_note=exp_note,
            right_angled=right_angled
        )
    else:
        raise ValueError(
            "plot_preferences currently supports only 2 or 3 objectives."
        )
    