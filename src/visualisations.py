import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap, BoundaryNorm
from sklearn.decomposition import PCA
import torch
import torch.nn.functional as F
from utils import collect_valid_states_fourrooms
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _to_coord_set(cells):
    if cells is None:
        return None

    if isinstance(cells, set):
        return {tuple(map(int, cell)) for cell in cells}

    arr = np.asarray(cells, dtype=np.int32)

    if arr.size == 0:
        return set()

    if arr.ndim == 1:
        if arr.shape[0] != 2:
            raise ValueError(f"Expected coordinate pair, got shape {arr.shape}")
        return {tuple(map(int, arr.tolist()))}

    if arr.ndim == 2 and arr.shape[1] == 2:
        return {tuple(map(int, row)) for row in arr.tolist()}

    raise ValueError(f"Could not interpret cells with shape {arr.shape} as coordinates")


def _to_coord_array(cells):
    if cells is None:
        return None

    if (
        isinstance(cells, np.ndarray)
        and cells.dtype.kind in {"i", "u"}
        and cells.ndim == 2
        and cells.shape[1] == 2
    ):
        return cells.astype(np.int32, copy=False)

    coord_set = _to_coord_set(cells)
    if coord_set is None:
        return None
    if len(coord_set) == 0:
        return np.zeros((0, 2), dtype=np.int32)

    return np.asarray(sorted(coord_set), dtype=np.int32)


def _extract_grid_spec(env):
    base = env.unwrapped if hasattr(env, "unwrapped") else env

    free_cells = _to_coord_array(getattr(base, "_free_cells", None))
    blocked_cells = _to_coord_set(getattr(base, "_blocked_cells", None))
    outer_wall_cells = _to_coord_set(getattr(base, "_outer_wall_cells", None)) or set()

    width = getattr(base, "width", None)
    height = getattr(base, "height", None)

    if width is None and hasattr(base, "grid_size"):
        width = int(base.grid_size)
    if height is None and hasattr(base, "grid_size"):
        height = int(base.grid_size)

    if free_cells is not None and (width is None or height is None) and len(free_cells) > 0:
        width = int(free_cells[:, 0].max()) + 1
        height = int(free_cells[:, 1].max()) + 1

    if free_cells is None and blocked_cells is not None and width is not None and height is not None:
        free = []
        for y in range(height):
            for x in range(width):
                if (x, y) not in blocked_cells:
                    free.append((x, y))
        free_cells = np.asarray(free, dtype=np.int32)

    if free_cells is None:
        raise RuntimeError("Env must expose _free_cells or (_blocked_cells + width/height or grid_size).")

    if blocked_cells is None:
        blocked_cells = set()
        free_set = {tuple(map(int, row)) for row in free_cells.tolist()}
        for y in range(height):
            for x in range(width):
                if (x, y) not in free_set:
                    blocked_cells.add((x, y))

    goal_pos = None
    if getattr(base, "_goal_pos", None) is not None:
        goal_pos = tuple(map(int, np.asarray(base._goal_pos, dtype=np.int32).tolist()))
    elif getattr(env, "goal_position", None) is not None:
        goal_pos = tuple(map(int, np.asarray(env.goal_position, dtype=np.int32).tolist()))

    action_names = getattr(base, "action_names", None)
    if action_names is None:
        action_names = [f"a{i}" for i in range(getattr(env.action_space, "n", 4))]

    return {
        "base": base,
        "free_cells": free_cells,
        "blocked_cells": blocked_cells,
        "outer_wall_cells": outer_wall_cells,
        "width": int(width),
        "height": int(height),
        "goal_pos": goal_pos,
        "action_names": list(action_names),
    }


def _cell_centers(obs, is_discrete):
    obs = np.asarray(obs, dtype=np.float32)
    if is_discrete:
        return obs[:, 0] + 0.5, obs[:, 1] + 0.5
    return obs[:, 0], obs[:, 1]


def _setup_grid_ax(ax, width, height, title=None):
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(0, width + 1, 1))
    ax.set_yticks(np.arange(0, height + 1, 1))
    ax.grid(color="lightgray", linewidth=0.8, alpha=0.6)
    ax.tick_params(labelsize=8, length=0)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if title is not None:
        ax.set_title(title)


def _overlay_blocked(ax, blocked_cells, facecolor="gray", alpha=0.85):
    for x, y in blocked_cells:
        ax.add_patch(
            Rectangle(
                (x, y),
                1.0,
                1.0,
                facecolor=facecolor,
                edgecolor=facecolor,
                linewidth=0.0,
                alpha=alpha,
                zorder=1,
            )
        )


def _draw_goal(ax, goal_pos, goal_size=120, color="green", edgecolors="black"):
    if goal_pos is None:
        return

    ax.scatter(
        goal_pos[0] + 0.5,
        goal_pos[1] + 0.5,
        c=color,
        marker="*",
        s=goal_size,
        edgecolors=edgecolors,
        zorder=6,
    )


def plot_policy_rollouts(
    env,
    policy_fn,
    goal_pos=None,
    eval_episodes=8,
    n_cols=4,
    is_discrete=False,
    step_point_size=10,
    start_size=55,
    end_size=45,
    goal_size=120,
    arrow_width=0.008,
    reset_options=None,
):
    spec = _extract_grid_spec(env)
    blocked_cells = spec["blocked_cells"]
    width = spec["width"]
    height = spec["height"]

    if goal_pos is None:
        goal_pos = spec["goal_pos"]

    ep_rewards = []
    ep_successes = []
    trajectories = []

    for ep_i in range(eval_episodes):
        if reset_options is None:
            obs, _ = env.reset()
        else:
            obs, _ = env.reset(options=reset_options)

        done = False
        ep_rew = 0.0
        terminated_flag = False
        trajectory = [np.asarray(obs, dtype=np.float32).copy()]

        while not done:
            action = policy_fn(obs)
            obs, rew, term, trunc, info = env.step(action)
            ep_rew += float(rew)
            trajectory.append(np.asarray(obs, dtype=np.float32).copy())
            done = term or trunc
            if term:
                terminated_flag = True

        trajectory = np.asarray(trajectory, dtype=np.float32)
        trajectories.append(trajectory)
        ep_rewards.append(ep_rew)

        if goal_pos is not None:
            if is_discrete:
                reached = np.array_equal(
                    trajectory[-1].astype(np.int32),
                    np.array(goal_pos, dtype=np.int32),
                )
            else:
                reached = bool(terminated_flag)
        else:
            reached = bool(terminated_flag)

        ep_successes.append(reached)
        print(
            f"Episode {ep_i + 1}: {len(trajectory) - 1} steps | "
            f"return: {ep_rew:.2f} | reached goal: {reached}"
        )

    success_rate = float(np.mean(ep_successes)) if ep_successes else 0.0
    avg_return = float(np.mean(ep_rewards)) if ep_rewards else 0.0
    std_return = float(np.std(ep_rewards)) if ep_rewards else 0.0

    print(f"\nSuccess rate: {success_rate:.2%} over {eval_episodes} episodes")
    print(f"Average return: {avg_return:.2f} ± {std_return:.2f}")

    n_rows = math.ceil(eval_episodes / n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.4 * n_cols, 4.4 * n_rows),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    for i, ax in enumerate(axes_flat):
        if i >= len(trajectories):
            ax.axis("off")
            continue

        _setup_grid_ax(ax, width, height)
        _overlay_blocked(ax, blocked_cells)
        _draw_goal(ax, goal_pos, goal_size=goal_size)

        traj = trajectories[i]
        xs, ys = _cell_centers(traj, is_discrete=is_discrete)

        ax.scatter(
            xs,
            ys,
            c=np.arange(len(xs)),
            cmap="Blues",
            s=step_point_size,
            zorder=4,
        )

        if len(xs) > 1:
            dx = xs[1:] - xs[:-1]
            dy = ys[1:] - ys[:-1]
            ax.quiver(
                xs[:-1],
                ys[:-1],
                dx,
                dy,
                angles="xy",
                scale_units="xy",
                scale=1,
                width=arrow_width,
                color="tab:blue",
                alpha=0.85,
                zorder=3,
            )

        ax.scatter(xs[0], ys[0], c="black", s=start_size, zorder=7, label="Start")
        ax.scatter(xs[-1], ys[-1], c="red", s=end_size, zorder=7, label="End")

        ax.set_title(
            f"Ep {i + 1} | steps={len(traj) - 1} | success={ep_successes[i]}",
            fontsize=10,
        )

        if i == 0:
            ax.legend(loc="upper right", fontsize=8)

    plt.suptitle(
        f"Policy rollouts"
        + (f" toward goal {goal_pos}" if goal_pos is not None else "")
        + f"\nSuccess rate: {success_rate:.2%} | Avg return: {avg_return:.2f} ± {std_return:.2f}",
        fontsize=14,
    )
    plt.tight_layout()
    plt.show()


def plot_q_diagnostics(
    env,
    value_fn,
    actor_fn=None,
    is_discrete=False,
    num_actions=4,
    action_names=None,
    goal_pos=None,
    eval_returns=None,
    figsize=(20, 5),
):
    spec = _extract_grid_spec(env)
    free_cells = spec["free_cells"]
    blocked_cells = spec["blocked_cells"]
    width = spec["width"]
    height = spec["height"]

    if goal_pos is None:
        goal_pos = spec["goal_pos"]
    if action_names is None:
        action_names = spec["action_names"]

    extent = [0, width, 0, height]
    fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)

    if is_discrete:
        q_best = np.full((height, width), np.nan, dtype=np.float32)
        best_action_idx = np.full((height, width), -1, dtype=np.int32)

        obs_batch = free_cells.astype(np.float32)
        q_vals = np.asarray(value_fn(obs_batch), dtype=np.float32)

        if q_vals.ndim != 2 or q_vals.shape[1] != num_actions:
            raise ValueError("For discrete envs, value_fn must return shape [N, num_actions].")

        for i, (x, y) in enumerate(free_cells):
            q_best[y, x] = q_vals[i].max()
            best_action_idx[y, x] = int(q_vals[i].argmax())

        im0 = axes[0].imshow(
            q_best,
            origin="lower",
            aspect="equal",
            extent=extent,
            cmap="viridis",
            interpolation="nearest",
        )
        _overlay_blocked(axes[0], blocked_cells, facecolor="black", alpha=1.0)
        _draw_goal(axes[0], goal_pos, goal_size=100, color="red", edgecolors="black")
        _setup_grid_ax(axes[0], width, height, "Max Q(s, a)")
        fig.colorbar(im0, ax=axes[0], shrink=0.9, label="Q value")

        colours = [
            "tab:blue", "tab:orange", "tab:green", "tab:red",
            "tab:purple", "tab:brown", "tab:pink", "tab:gray",
        ]
        action_cmap = ListedColormap(colours[:num_actions])
        norm = BoundaryNorm(np.arange(-0.5, num_actions + 0.5, 1), action_cmap.N)
        masked_actions = np.ma.masked_where(best_action_idx < 0, best_action_idx)

        im1 = axes[1].imshow(
            masked_actions,
            origin="lower",
            aspect="equal",
            extent=extent,
            cmap=action_cmap,
            norm=norm,
            interpolation="nearest",
        )
        _overlay_blocked(axes[1], blocked_cells, facecolor="black", alpha=1.0)
        _draw_goal(axes[1], goal_pos, goal_size=100, color="white", edgecolors="black")
        _setup_grid_ax(axes[1], width, height, "Greedy action argmax_a Q(s,a)")

        cbar = fig.colorbar(im1, ax=axes[1], shrink=0.9, ticks=list(range(num_actions)))
        cbar.ax.set_yticklabels(action_names)

        arrow_delta = {
            0: (0.0, 0.35),    # up
            1: (0.0, -0.35),   # down
            2: (-0.35, 0.0),   # left
            3: (0.35, 0.0),    # right
        }

        for (x, y) in free_cells:
            a = best_action_idx[y, x]
            if a not in arrow_delta:
                continue
            dx, dy = arrow_delta[a]
            axes[1].arrow(
                x + 0.5,
                y + 0.5,
                dx,
                dy,
                head_width=0.18,
                head_length=0.16,
                fc="black",
                ec="black",
                linewidth=0.5,
                alpha=0.8,
                zorder=6,
            )
    else:
        val_map = np.full((height, width), np.nan, dtype=np.float32)

        obs_batch = np.stack(
            [
                free_cells[:, 0].astype(np.float32) + 0.5,
                free_cells[:, 1].astype(np.float32) + 0.5,
            ],
            axis=1,
        )

        vals = np.asarray(value_fn(obs_batch), dtype=np.float32).reshape(-1)

        if vals.shape[0] != free_cells.shape[0]:
            raise ValueError("For continuous envs, value_fn must return shape [N] for N queried cells.")

        for i, (x, y) in enumerate(free_cells):
            val_map[y, x] = vals[i]

        im0 = axes[0].imshow(
            val_map,
            origin="lower",
            aspect="equal",
            extent=extent,
            cmap="viridis",
            interpolation="nearest",
        )
        _overlay_blocked(axes[0], blocked_cells, facecolor="black", alpha=1.0)
        _draw_goal(axes[0], goal_pos, goal_size=100, color="red", edgecolors="black")
        _setup_grid_ax(axes[0], width, height, "State value V(s) / Q(s, π(s))")
        fig.colorbar(im0, ax=axes[0], shrink=0.9, label="Value")

        if actor_fn is not None:
            acts = np.asarray(actor_fn(obs_batch), dtype=np.float32)
            if acts.ndim != 2 or acts.shape[1] < 2:
                raise ValueError("For continuous envs, actor_fn must return shape [N, act_dim>=2].")

            quiver_xs = free_cells[:, 0].astype(np.float32) + 0.5
            quiver_ys = free_cells[:, 1].astype(np.float32) + 0.5

            _setup_grid_ax(axes[1], width, height, "Greedy action direction π(s)")
            axes[1].set_facecolor("whitesmoke")
            _overlay_blocked(axes[1], blocked_cells, facecolor="gray", alpha=0.85)
            axes[1].quiver(
                quiver_xs,
                quiver_ys,
                acts[:, 0],
                acts[:, 1],
                angles="xy",
                scale_units="xy",
                scale=2.5,
                width=0.004,
                color="tab:blue",
                alpha=0.75,
                zorder=4,
            )
            _draw_goal(axes[1], goal_pos, goal_size=100, color="green", edgecolors="black")
        else:
            axes[1].text(0.5, 0.5, "No actor_fn provided", ha="center", va="center")
            axes[1].set_axis_off()

    if eval_returns is not None and len(eval_returns) > 0:
        steps, rets = zip(*eval_returns)
        axes[2].plot(steps, rets, color="steelblue", linewidth=1.5)
        axes[2].set_title("Evaluation return over training")
        axes[2].set_xlabel("Environment steps")
        axes[2].set_ylabel("Mean episodic return")
        axes[2].grid(alpha=0.25)
    else:
        axes[2].text(0.5, 0.5, "No eval returns provided", ha="center", va="center")
        axes[2].set_axis_off()

    plt.show()

def plot_full_embedding_dashboard_html(
    overall_results,
    qnet,
    mode="mean",
    normalise=True,
    annotate=False,
    title_task_3d="Task embeddings in 3D PCA space",
    title_weights="Q-network weight distributions",
    title_sa_fixed_3d="Fixed-probe SA embeddings in 3D PCA space",
    title_sa_iso="Final SA isotropy diagnostics",
    save_html="full_embedding_dashboard.html",
    sa_keywords=("sa_encoder",),
    goal_keywords=("goal_encoder",),
    bins=80,
    weights_his=None,
):

    def _to_numpy_embedding(x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        x = np.asarray(x, dtype=np.float32)
        return x

    def _ensure_keyword_iterable(x):
        if isinstance(x, str):
            return [x]
        return list(x)

    def _collect_embeddings(results, key, mode):
        vectors = []
        labels = []
        goal_names = []

        for goal, data in results.items():
            emb_list = data.get(key, [])
            if len(emb_list) == 0:
                continue

            emb_list_np = [_to_numpy_embedding(e).reshape(-1) for e in emb_list]

            if mode == "mean":
                vec = np.mean(np.stack(emb_list_np, axis=0), axis=0)
                vectors.append(vec)
                labels.append(str(goal))
                goal_names.append(str(goal))
            elif mode == "last":
                vec = emb_list_np[-1]
                vectors.append(vec)
                labels.append(str(goal))
                goal_names.append(str(goal))
            elif mode == "all":
                for i, vec in enumerate(emb_list_np):
                    vectors.append(vec)
                    labels.append(f"{goal} | idx={i}")
                    goal_names.append(str(goal))
            else:
                raise ValueError("mode must be one of: 'mean', 'last', or 'all'")

        if len(vectors) < 2:
            raise ValueError(f"Need at least 2 embeddings in '{key}' to run PCA.")

        X = np.stack(vectors, axis=0)

        if normalise:
            X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

        n_components = min(3, X.shape[0], X.shape[1])
        if n_components < 2:
            raise ValueError(f"Need embedding dimension >= 2 and at least 2 samples for '{key}' PCA.")

        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)
        explained = pca.explained_variance_ratio_

        if n_components < 3:
            X_3d = np.zeros((X_pca.shape[0], 3), dtype=X_pca.dtype)
            X_3d[:, :n_components] = X_pca
            explained_3 = np.zeros(3, dtype=np.float32)
            explained_3[:n_components] = explained
        else:
            X_3d = X_pca
            explained_3 = explained

        return X, X_3d, labels, goal_names, pca, explained, explained_3

    def _make_3d_figure(X_3d, labels, goal_names, explained_3, title):
        unique_goals = sorted(list(set(goal_names)))
        palette = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
        ]
        color_map = {g: palette[i % len(palette)] for i, g in enumerate(unique_goals)}

        fig = go.Figure()

        for goal in unique_goals:
            idxs = [i for i, g in enumerate(goal_names) if g == goal]
            xs = X_3d[idxs, 0]
            ys = X_3d[idxs, 1]
            zs = X_3d[idxs, 2]
            texts = [labels[i] for i in idxs]
            color = color_map[goal]

            for x, y, z in zip(xs, ys, zs):
                fig.add_trace(go.Scatter3d(
                    x=[0, x],
                    y=[0, y],
                    z=[0, z],
                    mode="lines",
                    line=dict(color=color, width=5),
                    opacity=0.65,
                    showlegend=False,
                    hoverinfo="skip"
                ))

            fig.add_trace(go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="markers+text" if annotate else "markers",
                text=texts if annotate else None,
                textposition="top center",
                marker=dict(size=6, color=color, opacity=0.95),
                name=goal,
                hovertext=texts,
                hovertemplate=(
                    "label=%{hovertext}<br>"
                    "PC1=%{x:.3f}<br>"
                    "PC2=%{y:.3f}<br>"
                    "PC3=%{z:.3f}<extra></extra>"
                ),
                textfont=dict(size=10)
            ))

        fig.add_trace(go.Scatter3d(
            x=[0],
            y=[0],
            z=[0],
            mode="markers",
            marker=dict(size=5, color="black", symbol="x"),
            name="origin",
            hovertemplate="origin<extra></extra>"
        ))

        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title=f"PC1 ({explained_3[0]*100:.1f}%)",
                yaxis_title=f"PC2 ({explained_3[1]*100:.1f}%)",
                zaxis_title=f"PC3 ({explained_3[2]*100:.1f}%)",
                aspectmode="cube"
            ),
            template="plotly_white",
            legend=dict(x=1.02, y=1.0),
            margin=dict(l=0, r=0, t=50, b=0)
        )
        return fig

    sa_keywords_local = [k.lower() for k in _ensure_keyword_iterable(sa_keywords)]
    goal_keywords_local = [k.lower() for k in _ensure_keyword_iterable(goal_keywords)]

        # ---------------------------
    # Window 1: weights + slider
    # ---------------------------
    if weights_his is None or len(weights_his) == 0:
        sa_weights = []
        goal_weights = []

        for name, param in qnet.named_parameters():
            if not param.requires_grad:
                continue
            vals = param.detach().cpu().numpy().ravel()
            lname = name.lower()

            if any(k in lname for k in sa_keywords_local):
                sa_weights.append(vals)
            elif any(k in lname for k in goal_keywords_local):
                goal_weights.append(vals)

        fig_w = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("SA encoder weights", "Goal encoder weights")
        )

        if len(sa_weights) > 0:
            sa_all = np.concatenate(sa_weights)
            fig_w.add_trace(
                go.Histogram(
                    x=sa_all,
                    nbinsx=bins,
                    marker_color="#1f77b4",
                    opacity=0.8,
                    name="SA encoder"
                ),
                row=1, col=1
            )

        if len(goal_weights) > 0:
            goal_all = np.concatenate(goal_weights)
            fig_w.add_trace(
                go.Histogram(
                    x=goal_all,
                    nbinsx=bins,
                    marker_color="#ff7f0e",
                    opacity=0.8,
                    name="Goal encoder"
                ),
                row=1, col=2
            )

        fig_w.update_xaxes(title_text="Weight value", row=1, col=1)
        fig_w.update_xaxes(title_text="Weight value", row=1, col=2)
        fig_w.update_yaxes(title_text="Count", row=1, col=1)
        fig_w.update_yaxes(title_text="Count", row=1, col=2)
        fig_w.update_layout(
            title=title_weights,
            template="plotly_white",
            bargap=0.05,
            showlegend=False,
            margin=dict(l=40, r=40, t=70, b=40)
        )

    else:
        fig_w = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("SA encoder weights", "Goal encoder weights")
        )

        first = weights_his[0]

        sa0 = np.asarray(first.get("sa_sample", []), dtype=np.float32)
        goal0 = np.asarray(first.get("goal_sample", []), dtype=np.float32)

        def _stats_text(snap):
            sa_stats = snap.get("sa_stats", {})
            goal_stats = snap.get("goal_stats", {})
            return (
                f"{title_weights}"
                f"<br><sup>{snap.get('stage', 'snapshot')} | "
                f"SA μ={sa_stats.get('mean', np.nan):.4f}, σ={sa_stats.get('std', np.nan):.4f} | "
                f"Goal μ={goal_stats.get('mean', np.nan):.4f}, σ={goal_stats.get('std', np.nan):.4f}</sup>"
            )

        fig_w.add_trace(
            go.Histogram(
                x=sa0,
                nbinsx=bins,
                marker_color="#1f77b4",
                opacity=0.8,
                name="SA encoder",
                histnorm=""
            ),
            row=1, col=1
        )

        fig_w.add_trace(
            go.Histogram(
                x=goal0,
                nbinsx=bins,
                marker_color="#ff7f0e",
                opacity=0.8,
                name="Goal encoder",
                histnorm=""
            ),
            row=1, col=2
        )

        frames = []
        slider_steps = []

        for i, snap in enumerate(weights_his):
            sa_vals = np.asarray(snap.get("sa_sample", []), dtype=np.float32)
            goal_vals = np.asarray(snap.get("goal_sample", []), dtype=np.float32)

            frames.append(
                go.Frame(
                    name=str(i),
                    data=[
                        go.Histogram(
                            x=sa_vals,
                            nbinsx=bins,
                            marker_color="#1f77b4",
                            opacity=0.8,
                            name="SA encoder",
                            histnorm=""
                        ),
                        go.Histogram(
                            x=goal_vals,
                            nbinsx=bins,
                            marker_color="#ff7f0e",
                            opacity=0.8,
                            name="Goal encoder",
                            histnorm=""
                        ),
                    ],
                    layout=go.Layout(
                        title=_stats_text(snap)
                    )
                )
            )

            slider_steps.append(
                {
                    "method": "animate",
                    "label": snap.get("stage", f"step_{i}"),
                    "args": [
                        [str(i)],
                        {
                            "mode": "immediate",
                            "frame": {"duration": 0, "redraw": True},
                            "transition": {"duration": 0},
                        },
                    ],
                }
            )

        fig_w.frames = frames

        fig_w.update_xaxes(title_text="Weight value", row=1, col=1)
        fig_w.update_xaxes(title_text="Weight value", row=1, col=2)
        fig_w.update_yaxes(title_text="Count", row=1, col=1)
        fig_w.update_yaxes(title_text="Count", row=1, col=2)

        fig_w.update_layout(
            title=_stats_text(first),
            template="plotly_white",
            bargap=0.05,
            showlegend=False,
            margin=dict(l=40, r=40, t=95, b=70),
            sliders=[
                {
                    "active": 0,
                    "pad": {"t": 20},
                    "currentvalue": {"prefix": "Weights snapshot: "},
                    "steps": slider_steps,
                }
            ],
            updatemenus=[
                {
                    "type": "buttons",
                    "direction": "left",
                    "x": 0.0,
                    "y": 1.18,
                    "showactive": False,
                    "buttons": [
                        {
                            "label": "Play",
                            "method": "animate",
                            "args": [
                                None,
                                {
                                    "fromcurrent": True,
                                    "transition": {"duration": 250},
                                    "frame": {"duration": 500, "redraw": True},
                                },
                            ],
                        },
                        {
                            "label": "Pause",
                            "method": "animate",
                            "args": [
                                [None],
                                {
                                    "mode": "immediate",
                                    "transition": {"duration": 0},
                                    "frame": {"duration": 0, "redraw": False},
                                },
                            ],
                        },
                    ],
                }
            ],
        )
    # ---------------------------
    # Window 2: task embeddings PCA
    # ---------------------------
    X_task, X_task_3d, labels_task, goal_names_task, pca_task, explained_task, explained_task_3 = \
        _collect_embeddings(overall_results, key="task_embeddings", mode=mode)

    fig_task_3d = _make_3d_figure(
        X_3d=X_task_3d,
        labels=labels_task,
        goal_names=goal_names_task,
        explained_3=explained_task_3,
        title=title_task_3d
    )

    # ---------------------------
    # Window 3: fixed-probe SA embeddings PCA
    # ---------------------------
    X_sa_fix, X_sa_fix_3d, labels_sa_fix, goal_names_sa_fix, pca_sa_fix, explained_sa_fix, explained_sa_fix_3 = \
        _collect_embeddings(overall_results, key="sa_fixed_probe_embeddings", mode=mode)

    fig_sa_fixed_3d = _make_3d_figure(
        X_3d=X_sa_fix_3d,
        labels=labels_sa_fix,
        goal_names=goal_names_sa_fix,
        explained_3=explained_sa_fix_3,
        title=title_sa_fixed_3d
    )

    # ---------------------------
    # Window 4: final SA isotropy diagnostics
    # ---------------------------
    all_batches = []
    for goal, data in overall_results.items():
        batches = data.get("sa_batches_final", [])
        for b in batches:
            all_batches.append(_to_numpy_embedding(b))

    if len(all_batches) == 0:
        raise ValueError("No 'sa_batches_final' found in overall_results.")

    X_iso = np.concatenate(all_batches, axis=0).astype(np.float32)   # [N, D]
    Xc = X_iso - X_iso.mean(axis=0, keepdims=True)

    cov = np.cov(Xc, rowvar=False)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.sort(eigvals)[::-1]

    trace = float(np.trace(cov))
    mean_diag = float(np.mean(np.diag(cov)))
    offdiag = cov - np.diag(np.diag(cov))
    mean_abs_offdiag = float(np.mean(np.abs(offdiag)))
    eig_ratio = float(eigvals[0] / (eigvals[-1] + 1e-8))

    Xn = Xc / (np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-8)
    gram = Xn @ Xn.T
    mask = ~np.eye(gram.shape[0], dtype=bool)
    mean_pairwise_cos = float(np.mean(gram[mask]))

    fig_iso = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Covariance heatmap", "Eigenvalue spectrum")
    )

    fig_iso.add_trace(
        go.Heatmap(
            z=cov,
            colorscale="RdBu",
            zmid=0,
            colorbar=dict(title="cov")
        ),
        row=1, col=1
    )

    fig_iso.add_trace(
        go.Bar(
            x=np.arange(1, len(eigvals) + 1),
            y=eigvals,
            marker_color="#2ca02c",
            name="eigenvalues"
        ),
        row=1, col=2
    )

    fig_iso.update_xaxes(title_text="Dimension", row=1, col=1)
    fig_iso.update_yaxes(title_text="Dimension", row=1, col=1)
    fig_iso.update_xaxes(title_text="Sorted eigenvalue index", row=1, col=2)
    fig_iso.update_yaxes(title_text="Eigenvalue", row=1, col=2)

    fig_iso.update_layout(
        title=(
            f"{title_sa_iso}"
            f"<br><sup>trace={trace:.4f}, mean_diag={mean_diag:.4f}, "
            f"mean_abs_offdiag={mean_abs_offdiag:.4f}, eig_ratio={eig_ratio:.4f}, "
            f"mean_pairwise_cos={mean_pairwise_cos:.4f}</sup>"
        ),
        template="plotly_white",
        showlegend=False,
        margin=dict(l=40, r=40, t=90, b=40)
    )

    # Explicit heights help embedded Plotly panels render reliably
    fig_w.update_layout(height=500)
    fig_task_3d.update_layout(height=500)
    fig_sa_fixed_3d.update_layout(height=500)
    fig_iso.update_layout(height=500)

    # ---------------------------
    # Save one HTML
    # ---------------------------
    save_html = Path(save_html)
    save_html.parent.mkdir(parents=True, exist_ok=True)

    html_w = fig_w.to_html(
        full_html=False,
        include_plotlyjs="cdn",
        config={"responsive": True},
        default_width="100%",
        default_height="100%",
    )

    html_task = fig_task_3d.to_html(
        full_html=False,
        include_plotlyjs=False,
        config={"responsive": True},
        default_width="100%",
        default_height="100%",
    )

    html_sa_fix = fig_sa_fixed_3d.to_html(
        full_html=False,
        include_plotlyjs=False,
        config={"responsive": True},
        default_width="100%",
        default_height="100%",
    )

    html_iso = fig_iso.to_html(
        full_html=False,
        include_plotlyjs=False,
        config={"responsive": True},
        default_width="100%",
        default_height="100%",
    )

    dashboard_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Embedding Dashboard</title>
    <style>
        html, body {{
            margin: 0;
            padding: 0;
            width: 100%;
            height: 100%;
            font-family: Arial, sans-serif;
            background: #f3f4f6;
        }}

        body {{
            overflow: hidden;
        }}

        .dashboard {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            grid-template-rows: 1fr 1fr;
            gap: 12px;
            width: 100vw;
            height: 100vh;
            padding: 12px;
            box-sizing: border-box;
        }}

        .panel {{
            background: white;
            border: 1px solid #dcdcdc;
            border-radius: 10px;
            overflow: hidden;
            min-width: 0;
            min-height: 0;
            box-shadow: 0 1px 4px rgba(0,0,0,0.06);
            display: flex;
            flex-direction: column;
        }}

        .plot-wrap {{
            flex: 1 1 auto;
            width: 100%;
            height: 100%;
            min-width: 0;
            min-height: 0;
            overflow: hidden;
        }}

        .plotly-graph-div {{
            width: 100% !important;
            height: 100% !important;
        }}
    </style>
</head>
<body>
    <div class="dashboard">
        <div class="panel">
            <div class="plot-wrap">{html_w}</div>
        </div>
        <div class="panel">
            <div class="plot-wrap">{html_task}</div>
        </div>
        <div class="panel">
            <div class="plot-wrap">{html_sa_fix}</div>
        </div>
        <div class="panel">
            <div class="plot-wrap">{html_iso}</div>
        </div>
    </div>
</body>
</html>
"""
    save_html.write_text(dashboard_html, encoding="utf-8")

    return {
        "save_html": str(save_html),
        "fig_weights": fig_w,
        "fig_task_3d": fig_task_3d,
        "fig_sa_fixed_3d": fig_sa_fixed_3d,
        "fig_sa_isotropy": fig_iso,
        "isotropy_metrics": {
            "trace": trace,
            "mean_diag": mean_diag,
            "mean_abs_offdiag": mean_abs_offdiag,
            "eig_ratio": eig_ratio,
            "mean_pairwise_cos": mean_pairwise_cos,
        },
    }

def plot_eval_results(eval_first, eval_time_first, min_steps_first, min_time_first):
    if eval_first and eval_time_first:
        xs_steps, ys_steps = zip(*eval_first)
        xs_time, ys_time = zip(*eval_time_first)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

        ax1.plot(xs_steps, ys_steps)
        if min_steps_first is not None:
            ax1.axvline(min_steps_first, color='red', linestyle='--', label='Good policy achieved')
            ax1.annotate(
                f"Good policy achieved at step {min_steps_first}",
                xy=(min_steps_first, 0.99),
                xytext=(min_steps_first + 5000, 0.5),
                arrowprops=dict(arrowstyle="->", color='red'),
                color='red',
            )
        ax1.set_xlabel("Environment steps")
        ax1.set_ylabel("Mean episodic return")
        ax1.set_title("Q = φ(s,a)ᵀψ(z) on FourRooms (discrete) – steps")
        ax1.grid(alpha=0.25)
        
        ax2.plot(xs_time, ys_time)
        if min_time_first is not None:
            ax2.axvline(min_time_first, color='red', linestyle='--', label='Good policy achieved')
            ax2.annotate(
                f"Good policy achieved at time {min_time_first:.2f}s",
                xy=(min_time_first, 0.99),
                xytext=(min_time_first + 5.0, 0.5),
                arrowprops=dict(arrowstyle="->", color='red'),
                color='red',
            )
        ax2.set_xlabel("Evaluation time (s)")
        ax2.set_ylabel("Mean episodic return")
        ax2.set_title("Q = φ(s,a)ᵀψ(z) on FourRooms (discrete) – time")
        ax2.grid(alpha=0.25)

        plt.tight_layout()
        plt.show()

def print_goal_embedding_similarity(task_embedding_memory, goal_labels=None, decimals=3):
    if len(task_embedding_memory) == 0:
        print("Goal embedding memory is empty.")
        return

    M = np.stack(task_embedding_memory, axis=0)  # [N, D]
    M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-8)
    sim = M @ M.T  # cosine similarity matrix, [N, N]

    if goal_labels is None:
        goal_labels = [f"g{i}" for i in range(len(task_embedding_memory))]

    print("\nGoal embedding cosine similarity matrix:")
    header = " " + " ".join([f"{str(label):>10s}" for label in goal_labels])
    print(header)

    for i, row in enumerate(sim):
        row_str = " ".join([f"{x:10.{decimals}f}" for x in row])
        print(f"{str(goal_labels[i]):>8s} {row_str}")

    if len(task_embedding_memory) > 1:
        upper = sim[np.triu_indices(len(task_embedding_memory), k=1)]
        print(f"\nOff-diagonal mean similarity: {upper.mean():.{decimals}f}")
        print(f"Off-diagonal min similarity: {upper.min():.{decimals}f}")
        print(f"Off-diagonal max similarity: {upper.max():.{decimals}f}")

def shared_pca_projection(emb_before, emb_after, n_components=2):
    X = np.concatenate([emb_before, emb_after], axis=0)
    pca = PCA(n_components=n_components)
    Xp = pca.fit_transform(X)
    Z_before = Xp[:emb_before.shape[0]]
    Z_after = Xp[emb_before.shape[0]:]
    return Z_before, Z_after, pca

def plot_before_after(Z_before, Z_after, label_before, label_after, title):
    plt.figure(figsize=(7, 6))
    plt.scatter(Z_before[:, 0], Z_before[:, 1], s=25, alpha=0.7, label=label_before)
    plt.scatter(Z_after[:, 0], Z_after[:, 1], s=25, alpha=0.7, label=label_after)
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()

def visualise_q_table(goal, q_network, eval_returns=None, task_embedding=None, device=None, make_env=None, reset_options=None):
    q_network.eval()

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if make_env is None:
        raise ValueError("make_env function must be provided to create the evaluation environment.")
    
    eval_env_first = make_env(goal=goal)

    base_env = eval_env_first.unwrapped if hasattr(eval_env_first, "unwrapped") else eval_env_first
    env_action_names = getattr(base_env, "action_names", ["Up", "Down", "Left", "Right"])

    if task_embedding is not None:
        if not torch.is_tensor(task_embedding):
            task_embedding = torch.tensor(task_embedding, dtype=torch.float32, device=device)
        else:
            task_embedding = task_embedding.to(device).float()

    def dqn_policy_fn(obs, q_network, goal, task_embedding=None):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            if task_embedding is None:
                goal_arr = np.array(goal, dtype=np.float32)
                goal_t = torch.tensor(goal_arr, dtype=torch.float32, device=device).unsqueeze(0)
                q_vals = q_network.q_val_for_argmax_action(obs_t, goal_t)
            else:
                q_vals = q_network.q_val_for_argmax_action_from_embedding(
                    obs_t,
                    task_embedding,
                    normalize_embedding=False,
                )

            action = int(q_vals.argmax(dim=-1).item())

        return action

    def dqn_value_fn(obs_batch, q_network, goal, task_embedding=None):
        obs_t = torch.tensor(obs_batch, dtype=torch.float32, device=device)
        B = obs_t.shape[0]

        with torch.no_grad():
            if task_embedding is None:
                goal_arr = np.array(goal, dtype=np.float32)
                goal_t = torch.tensor(goal_arr, dtype=torch.float32, device=device).unsqueeze(0)
                goal_batch = goal_t.expand(B, -1)
                q_vals = q_network.q_val_for_argmax_action(obs_t, goal_batch)
            else:
                q_vals = q_network.q_val_for_argmax_action_from_embedding(
                    obs_t,
                    task_embedding,
                    normalize_embedding=False,
                )

            q_vals = q_vals.cpu().numpy()

        return q_vals

    plot_policy_rollouts(
        env=eval_env_first,
        policy_fn=lambda obs: dqn_policy_fn(
            obs,
            q_network=q_network,
            goal=goal,
            task_embedding=task_embedding,
        ),
        goal_pos=goal,
        eval_episodes=8,
        n_cols=4,
        is_discrete=True,
        step_point_size=12,
        start_size=60,
        end_size=50,
        goal_size=130,
        arrow_width=0.01,
        reset_options=reset_options
    )

    plot_q_diagnostics(
        env=eval_env_first,
        value_fn=lambda obs_batch: dqn_value_fn(
            obs_batch,
            q_network=q_network,
            goal=goal,
            task_embedding=task_embedding,
        ),
        actor_fn=None,
        is_discrete=True,
        num_actions=eval_env_first.action_space.n,
        action_names=env_action_names,
        goal_pos=goal,
        eval_returns=eval_returns,
    )

    eval_env_first.close()
    q_network.train()

    return q_network


def visualise_embeddings(
    goal,
    q_network,
    actor=None,
    device=None,
    make_env=None,
    action_mode="auto",
    probe_actions=None,
):
    import numpy as np
    import torch
    import torch.nn.functional as F
    import matplotlib.pyplot as plt

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if make_env is None:
        raise ValueError("make_env function must be provided to create the evaluation environment.")

    eval_env_first = make_env(goal=goal)
    base_goal = np.array(goal, dtype=np.float32)
    base_goal_t = torch.tensor(base_goal, dtype=torch.float32, device=device).unsqueeze(0)

    states_base, coords_base = collect_valid_states_fourrooms(eval_env_first)
    obs_base_t = torch.tensor(states_base, dtype=torch.float32, device=device)

    q_network.eval()
    if actor is not None:
        actor.eval()

    with torch.no_grad():
        psi_z_base = q_network.encode_goal(base_goal_t).cpu().numpy()  # [1, D_goal_like]

        if action_mode == "auto":
            if hasattr(q_network, "num_actions"):
                action_mode = "all_discrete"
            elif actor is not None:
                action_mode = "actor"
            else:
                raise ValueError(
                    "Could not infer action_mode. Provide actor for continuous mode "
                    "or set action_mode explicitly."
                )

        # ---------------------------------------------------
        # Discrete mode: enumerate all actions
        # ---------------------------------------------------
        if action_mode == "all_discrete":
            if not hasattr(q_network, "num_actions"):
                raise ValueError("q_network.num_actions is required for discrete action analysis.")
            if not hasattr(q_network, "obs_dim"):
                raise ValueError("q_network.obs_dim is required for discrete action analysis.")
            if not hasattr(q_network, "rep_dim"):
                raise ValueError("q_network.rep_dim is required for discrete action analysis.")

            N = obs_base_t.shape[0]
            A = q_network.num_actions
            action_dim = getattr(q_network, "action_dim", q_network.num_actions)

            act_onehot = F.one_hot(
                torch.arange(A, device=device),
                num_classes=action_dim
            ).float()

            obs_rep = obs_base_t.unsqueeze(1).expand(-1, A, -1)
            act_rep = act_onehot.unsqueeze(0).expand(N, -1, -1)

            obs_flat = obs_rep.reshape(N * A, q_network.obs_dim)
            act_flat = act_rep.reshape(N * A, action_dim)

            phi_sa_base = q_network.encode_state_action(obs_flat, act_flat).cpu().numpy()
            phi_sa_base = phi_sa_base.reshape(N, A, -1)
            phi_sa_base_flat = phi_sa_base.reshape(N * A, -1)

            print("Embedding mode: all_discrete")
            print("states_base:", states_base.shape)
            print("coords_base:", coords_base.shape)
            print("phi(s,a) base:", phi_sa_base.shape)
            print("psi(z) / task code base:", psi_z_base.shape)

        # ---------------------------------------------------
        # Continuous mode: actor action for each state
        # ---------------------------------------------------
        elif action_mode == "actor":
            if actor is None:
                raise ValueError("actor must be provided when action_mode='actor'.")

            N = obs_base_t.shape[0]
            goal_batch = base_goal_t.expand(N, -1)

            act_base = actor(obs_base_t, goal_batch)
            phi_sa_base = q_network.encode_state_action(obs_base_t, act_base).cpu().numpy()
            phi_sa_base_flat = phi_sa_base

            print("Embedding mode: actor")
            print("states_base:", states_base.shape)
            print("coords_base:", coords_base.shape)
            print("actions from actor:", act_base.shape)
            print("phi(s, pi(s,g)) base:", phi_sa_base.shape)
            print("psi(z) / task code base:", psi_z_base.shape)

        # ---------------------------------------------------
        # Continuous mode: fixed probe actions
        # ---------------------------------------------------
        elif action_mode == "probe":
            if probe_actions is None:
                raise ValueError("probe_actions must be provided when action_mode='probe'.")

            probe_actions_t = torch.tensor(np.asarray(probe_actions), dtype=torch.float32, device=device)
            N = obs_base_t.shape[0]
            K = probe_actions_t.shape[0]

            obs_rep = obs_base_t.unsqueeze(1).expand(-1, K, -1)
            act_rep = probe_actions_t.unsqueeze(0).expand(N, -1, -1)

            obs_flat = obs_rep.reshape(N * K, obs_base_t.shape[1])
            act_flat = act_rep.reshape(N * K, probe_actions_t.shape[1])

            phi_sa_base = q_network.encode_state_action(obs_flat, act_flat).cpu().numpy()
            phi_sa_base = phi_sa_base.reshape(N, K, -1)
            phi_sa_base_flat = phi_sa_base.reshape(N * K, -1)

            print("Embedding mode: probe")
            print("states_base:", states_base.shape)
            print("coords_base:", coords_base.shape)
            print("probe_actions:", probe_actions_t.shape)
            print("phi(s,a_probe) base:", phi_sa_base.shape)
            print("psi(z) / task code base:", psi_z_base.shape)

        else:
            raise ValueError(f"Unknown action_mode: {action_mode}")

    # --------------------------------------------
    # Shared diagnostics
    # --------------------------------------------
    phisa_t = torch.tensor(phi_sa_base_flat, dtype=torch.float32)
    psi_t = torch.tensor(psi_z_base.squeeze(0), dtype=torch.float32)

    norms = phisa_t.norm(dim=-1)
    print(f"phi(s,a) norms: mean={norms.mean().item():.4f}, std={norms.std().item():.4f}")

    phisa_centered = phisa_t - phisa_t.mean(dim=0, keepdim=True)
    U, S, Vh = torch.linalg.svd(phisa_centered, full_matrices=False)
    S_np = S.cpu().numpy()
    explained = (S_np ** 2) / (S_np ** 2).sum()
    cumulative = explained.cumsum()

    print("top 5 singular values:", S_np[:5])
    print("cumulative variance (first 5 dims):", cumulative[:5])

    can_compare_cosine = (phisa_t.shape[-1] == psi_t.shape[-1])

    if can_compare_cosine:
        psi_unit = psi_t / (psi_t.norm() + 1e-8)
        phisa_unit = phisa_t / (phisa_t.norm(dim=-1, keepdim=True) + 1e-8)
        cos = phisa_unit @ psi_unit
        cos_np = cos.cpu().numpy()
        print(f"cos(phi(s,a), psi/task): mean={cos_np.mean():.4f}, std={cos_np.std():.4f}")
    else:
        cos_np = None
        print(
            f"Skipping cosine histogram: phi dim = {phisa_t.shape[-1]}, "
            f"goal/task dim = {psi_t.shape[-1]}"
        )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(cumulative, marker="o")
    axes[0].set_xlabel("Number of components")
    axes[0].set_ylabel("Cumulative variance explained")
    axes[0].set_title(f"Variance spectrum of phi(s,a) [{goal}]")
    axes[0].grid(alpha=0.3)

    if cos_np is not None:
        axes[1].hist(cos_np, bins=40, alpha=0.7)
        axes[1].set_xlabel("cos(phi(s,a), psi/task)")
        axes[1].set_ylabel("count")
        axes[1].set_title(f"Cosine histo phi(s,a) vs psi/task [{goal}]")
        axes[1].grid(alpha=0.3)
    else:
        psi_np = psi_t.cpu().numpy()
        axes[1].hist(psi_np, bins=40, alpha=0.7)
        axes[1].set_xlabel("task code value")
        axes[1].set_ylabel("count")
        axes[1].set_title(f"Task-code histogram [{goal}]")
        axes[1].grid(alpha=0.3)
        axes[1].text(
            0.5, 0.95,
            f"phi dim={phisa_t.shape[-1]}, task dim={psi_t.shape[-1]}",
            ha="center", va="top",
            transform=axes[1].transAxes, fontsize=10
        )

    plt.tight_layout()
    plt.show()

    eval_env_first.close()

    q_network.train()
    if actor is not None:
        actor.train()



def visualise_q_table_td3(
    goal,
    actor,
    q1,
    q2,
    eval_returns=None,
    task_embedding=None,
    device=None,
    make_env=None,
    reset_options=None,
):
    actor.eval()
    q1.eval()
    q2.eval()

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if make_env is None:
        raise ValueError("make_env function must be provided to create the evaluation environment.")

    eval_env_first = make_env(goal=goal)

    # Optional task embedding
    if task_embedding is not None:
        if not torch.is_tensor(task_embedding):
            task_embedding = torch.tensor(task_embedding, dtype=torch.float32, device=device)
        else:
            task_embedding = task_embedding.to(device).float()

    goal_arr = np.array(goal, dtype=np.float32)
    goal_t_single = torch.tensor(goal_arr, dtype=torch.float32, device=device).unsqueeze(0)

    def td3_policy_fn(obs):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action_t = actor(obs_t, goal_t_single)
        return action_t.squeeze(0).cpu().numpy().astype(np.float32)

    def td3_value_fn(obs_batch):
        obs_t = torch.tensor(obs_batch, dtype=torch.float32, device=device)
        B = obs_t.shape[0]
        goal_batch = goal_t_single.expand(B, -1)

        with torch.no_grad():
            act_t = actor(obs_t, goal_batch)

            if task_embedding is None:
                q1_vals = q1(obs_t, act_t, goal_batch)
                q2_vals = q2(obs_t, act_t, goal_batch)
            else:
                q1_vals = q1.forward_with_task_embedding(
                    obs_t,
                    act_t,
                    task_embedding,
                    normalize_embedding=False,
                )
                q2_vals = q2.forward_with_task_embedding(
                    obs_t,
                    act_t,
                    task_embedding,
                    normalize_embedding=False,
                )

            minq_vals = torch.minimum(q1_vals, q2_vals)

        # Return [N] to match continuous branch of plot_q_diagnostics
        return minq_vals.squeeze(-1).cpu().numpy()

    def td3_actor_fn(obs_batch):
        obs_t = torch.tensor(obs_batch, dtype=torch.float32, device=device)
        B = obs_t.shape[0]
        goal_batch = goal_t_single.expand(B, -1)

        with torch.no_grad():
            act_t = actor(obs_t, goal_batch)

        return act_t.cpu().numpy()

    # 1) Rollouts
    plot_policy_rollouts(
        env=eval_env_first,
        policy_fn=td3_policy_fn,
        goal_pos=goal,
        eval_episodes=8,
        n_cols=4,
        is_discrete=False,
        step_point_size=12,
        start_size=60,
        end_size=50,
        goal_size=130,
        arrow_width=0.01,
        reset_options=reset_options,
    )

    # 2) Q diagnostics (continuous mode)
    plot_q_diagnostics(
        env=eval_env_first,
        value_fn=td3_value_fn,
        actor_fn=td3_actor_fn,
        is_discrete=False,
        num_actions=0,
        action_names=None,
        goal_pos=goal,
        eval_returns=eval_returns,
    )

    eval_env_first.close()
    actor.train()
    q1.train()
    q2.train()

    return actor, q1, q2

def plot_full_embedding_dashboard_2d_html(
    overall_results,
    qnet,
    mode="mean",
    normalise=True,
    annotate=False,
    title_task_3d="Task embeddings in 2D PCA space",
    title_weights="Q-network weight distributions",
    title_sa_fixed_3d="Fixed-probe SA embeddings in 2D PCA space",
    title_sa_iso="Final SA isotropy diagnostics",
    save_html="full_embedding_dashboard.html",
    sa_keywords=("sa_encoder",),
    goal_keywords=("goal_encoder",),
    bins=80,
    weights_his=None,
):
    # ---------------------------
    # Helper functions
    # ---------------------------
    def _to_numpy_embedding(x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        x = np.asarray(x, dtype=np.float32)
        return x

    def _ensure_keyword_iterable(x):
        if isinstance(x, str):
            return [x]
        return list(x)

    def _collect_embeddings(results, key, mode):
        """
        Collect embeddings and project to 2D with PCA.
        Returns:
            X         : original embeddings [N, D]
            X_pca_2d  : 2D PCA projection [N, 2]
            labels    : text labels for points
            goal_names: goal labels (for coloring)
            pca       : PCA object
            explained : variance ratios for PC1, PC2
        """
        vectors = []
        labels = []
        goal_names = []

        for goal, data in results.items():
            emb_list = data.get(key, [])
            if len(emb_list) == 0:
                continue

            emb_list_np = [_to_numpy_embedding(e).reshape(-1) for e in emb_list]

            if mode == "mean":
                vec = np.mean(np.stack(emb_list_np, axis=0), axis=0)
                vectors.append(vec)
                labels.append(str(goal))
                goal_names.append(str(goal))
            elif mode == "last":
                vec = emb_list_np[-1]
                vectors.append(vec)
                labels.append(str(goal))
                goal_names.append(str(goal))
            elif mode == "all":
                for i, vec in enumerate(emb_list_np):
                    vectors.append(vec)
                    labels.append(f"{goal} | idx={i}")
                    goal_names.append(str(goal))
            else:
                raise ValueError("mode must be one of: 'mean', 'last', or 'all'")

        if len(vectors) < 2:
            raise ValueError(f"Need at least 2 embeddings in '{key}' to run PCA.")

        X = np.stack(vectors, axis=0)

        if normalise:
            X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

        n_components = min(2, X.shape[0], X.shape[1])
        if n_components < 2:
            raise ValueError(
                f"Need embedding dimension >= 2 and at least 2 samples for '{key}' PCA."
            )

        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)
        explained = pca.explained_variance_ratio_

        return X, X_pca, labels, goal_names, pca, explained

    def _make_2d_figure(X_2d, labels, goal_names, explained, title):
        """
        Make a 2D PCA scatter figure with PC1/PC2 axes.
        """
        unique_goals = sorted(list(set(goal_names)))
        palette = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
        ]
        color_map = {g: palette[i % len(palette)] for i, g in enumerate(unique_goals)}

        fig = go.Figure()

        for goal in unique_goals:
            idxs = [i for i, g in enumerate(goal_names) if g == goal]
            xs = X_2d[idxs, 0]
            ys = X_2d[idxs, 1]
            texts = [labels[i] for i in idxs]
            color = color_map[goal]

            fig.add_trace(go.Scatter(
                x=xs,
                y=ys,
                mode="markers+text" if annotate else "markers",
                text=texts if annotate else None,
                textposition="top center",
                marker=dict(size=9, color=color, opacity=0.9),
                name=goal,
                hovertext=texts,
                hovertemplate=(
                    "label=%{hovertext}<br>"
                    "PC1=%{x:.3f}<br>"
                    "PC2=%{y:.3f}<extra></extra>"
                ),
                textfont=dict(size=10)
            ))

        fig.add_trace(go.Scatter(
            x=[0],
            y=[0],
            mode="markers",
            marker=dict(size=7, color="black", symbol="x"),
            name="origin",
            hovertemplate="origin<extra></extra>"
        ))

        fig.update_layout(
            title=title,
            xaxis_title=f"PC1 ({explained[0]*100:.1f}%)",
            yaxis_title=f"PC2 ({explained[1]*100:.1f}%)",
            template="plotly_white",
            legend=dict(x=1.02, y=1.0),
            margin=dict(l=40, r=40, t=60, b=40)
        )

        return fig

    sa_keywords_local = [k.lower() for k in _ensure_keyword_iterable(sa_keywords)]
    goal_keywords_local = [k.lower() for k in _ensure_keyword_iterable(goal_keywords)]

    # ---------------------------
    # Window 1: weights + slider
    # ---------------------------
    if weights_his is None or len(weights_his) == 0:
        sa_weights = []
        goal_weights = []

        for name, param in qnet.named_parameters():
            if not param.requires_grad:
                continue
            vals = param.detach().cpu().numpy().ravel()
            lname = name.lower()

            if any(k in lname for k in sa_keywords_local):
                sa_weights.append(vals)
            elif any(k in lname for k in goal_keywords_local):
                goal_weights.append(vals)

        fig_w = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("SA encoder weights", "Goal encoder weights")
        )

        if len(sa_weights) > 0:
            sa_all = np.concatenate(sa_weights)
            fig_w.add_trace(
                go.Histogram(
                    x=sa_all,
                    nbinsx=bins,
                    marker_color="#1f77b4",
                    opacity=0.8,
                    name="SA encoder"
                ),
                row=1, col=1
            )

        if len(goal_weights) > 0:
            goal_all = np.concatenate(goal_weights)
            fig_w.add_trace(
                go.Histogram(
                    x=goal_all,
                    nbinsx=bins,
                    marker_color="#ff7f0e",
                    opacity=0.8,
                    name="Goal encoder"
                ),
                row=1, col=2
            )

        fig_w.update_xaxes(title_text="Weight value", row=1, col=1)
        fig_w.update_xaxes(title_text="Weight value", row=1, col=2)
        fig_w.update_yaxes(title_text="Count", row=1, col=1)
        fig_w.update_yaxes(title_text="Count", row=1, col=2)
        fig_w.update_layout(
            title=title_weights,
            template="plotly_white",
            bargap=0.05,
            showlegend=False,
            margin=dict(l=40, r=40, t=70, b=40)
        )

    else:
        fig_w = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("SA encoder weights", "Goal encoder weights")
        )

        first = weights_his[0]

        sa0 = np.asarray(first.get("sa_sample", []), dtype=np.float32)
        goal0 = np.asarray(first.get("goal_sample", []), dtype=np.float32)

        def _stats_text(snap):
            sa_stats = snap.get("sa_stats", {})
            goal_stats = snap.get("goal_stats", {})
            return (
                f"{title_weights}"
                f"<br><sup>{snap.get('stage', 'snapshot')} | "
                f"SA μ={sa_stats.get('mean', np.nan):.4f}, σ={sa_stats.get('std', np.nan):.4f} | "
                f"Goal μ={goal_stats.get('mean', np.nan):.4f}, σ={goal_stats.get('std', np.nan):.4f}</sup>"
            )

        fig_w.add_trace(
            go.Histogram(
                x=sa0,
                nbinsx=bins,
                marker_color="#1f77b4",
                opacity=0.8,
                name="SA encoder",
                histnorm=""
            ),
            row=1, col=1
        )

        fig_w.add_trace(
            go.Histogram(
                x=goal0,
                nbinsx=bins,
                marker_color="#ff7f0e",
                opacity=0.8,
                name="Goal encoder",
                histnorm=""
            ),
            row=1, col=2
        )

        frames = []
        slider_steps = []

        for i, snap in enumerate(weights_his):
            sa_vals = np.asarray(snap.get("sa_sample", []), dtype=np.float32)
            goal_vals = np.asarray(snap.get("goal_sample", []), dtype=np.float32)

            frames.append(
                go.Frame(
                    name=str(i),
                    data=[
                        go.Histogram(
                            x=sa_vals,
                            nbinsx=bins,
                            marker_color="#1f77b4",
                            opacity=0.8,
                            name="SA encoder",
                            histnorm=""
                        ),
                        go.Histogram(
                            x=goal_vals,
                            nbinsx=bins,
                            marker_color="#ff7f0e",
                            opacity=0.8,
                            name="Goal encoder",
                            histnorm=""
                        ),
                    ],
                    layout=go.Layout(
                        title=_stats_text(snap)
                    )
                )
            )

            slider_steps.append(
                {
                    "method": "animate",
                    "label": snap.get("stage", f"step_{i}"),
                    "args": [
                        [str(i)],
                        {
                            "mode": "immediate",
                            "frame": {"duration": 0, "redraw": True},
                            "transition": {"duration": 0},
                        },
                    ],
                }
            )

        fig_w.frames = frames

        fig_w.update_xaxes(title_text="Weight value", row=1, col=1)
        fig_w.update_xaxes(title_text="Weight value", row=1, col=2)
        fig_w.update_yaxes(title_text="Count", row=1, col=1)
        fig_w.update_yaxes(title_text="Count", row=1, col=2)

        fig_w.update_layout(
            title=_stats_text(first),
            template="plotly_white",
            bargap=0.05,
            showlegend=False,
            margin=dict(l=40, r=40, t=95, b=70),
            sliders=[
                {
                    "active": 0,
                    "pad": {"t": 20},
                    "currentvalue": {"prefix": "Weights snapshot: "},
                    "steps": slider_steps,
                }
            ],
            updatemenus=[
                {
                    "type": "buttons",
                    "direction": "left",
                    "x": 0.0,
                    "y": 1.18,
                    "showactive": False,
                    "buttons": [
                        {
                            "label": "Play",
                            "method": "animate",
                            "args": [
                                None,
                                {
                                    "fromcurrent": True,
                                    "transition": {"duration": 250},
                                    "frame": {"duration": 500, "redraw": True},
                                },
                            ],
                        },
                        {
                            "label": "Pause",
                            "method": "animate",
                            "args": [
                                [None],
                                {
                                    "mode": "immediate",
                                    "transition": {"duration": 0},
                                    "frame": {"duration": 0, "redraw": False},
                                },
                            ],
                        },
                    ],
                }
            ],
        )

    # ---------------------------
    # Window 2: task embeddings PCA (2D)
    # ---------------------------
    X_task, X_task_2d, labels_task, goal_names_task, pca_task, explained_task = \
        _collect_embeddings(overall_results, key="task_embeddings", mode=mode)

    fig_task_2d = _make_2d_figure(
        X_2d=X_task_2d,
        labels=labels_task,
        goal_names=goal_names_task,
        explained=explained_task,
        title=title_task_3d
    )

    # ---------------------------
    # Window 3: fixed-probe SA embeddings PCA (2D)
    # ---------------------------
    X_sa_fix, X_sa_fix_2d, labels_sa_fix, goal_names_sa_fix, pca_sa_fix, explained_sa_fix = \
        _collect_embeddings(overall_results, key="sa_fixed_probe_embeddings", mode=mode)

    fig_sa_fixed_2d = _make_2d_figure(
        X_2d=X_sa_fix_2d,
        labels=labels_sa_fix,
        goal_names=goal_names_sa_fix,
        explained=explained_sa_fix,
        title=title_sa_fixed_3d
    )

    # ---------------------------
    # Window 4: final SA isotropy diagnostics
    # ---------------------------
    all_batches = []
    for goal, data in overall_results.items():
        batches = data.get("sa_batches_final", [])
        for b in batches:
            all_batches.append(_to_numpy_embedding(b))

    if len(all_batches) == 0:
        raise ValueError("No 'sa_batches_final' found in overall_results.")

    X_iso = np.concatenate(all_batches, axis=0).astype(np.float32)   # [N, D]
    Xc = X_iso - X_iso.mean(axis=0, keepdims=True)

    cov = np.cov(Xc, rowvar=False)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.sort(eigvals)[::-1]

    trace = float(np.trace(cov))
    mean_diag = float(np.mean(np.diag(cov)))
    offdiag = cov - np.diag(np.diag(cov))
    mean_abs_offdiag = float(np.mean(np.abs(offdiag)))
    eig_ratio = float(eigvals[0] / (eigvals[-1] + 1e-8))

    Xn = Xc / (np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-8)
    gram = Xn @ Xn.T
    mask = ~np.eye(gram.shape[0], dtype=bool)
    mean_pairwise_cos = float(np.mean(gram[mask]))

    fig_iso = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Covariance heatmap", "Eigenvalue spectrum")
    )

    fig_iso.add_trace(
        go.Heatmap(
            z=cov,
            colorscale="RdBu",
            zmid=0,
            colorbar=dict(title="cov")
        ),
        row=1, col=1
    )

    fig_iso.add_trace(
        go.Bar(
            x=np.arange(1, len(eigvals) + 1),
            y=eigvals,
            marker_color="#2ca02c",
            name="eigenvalues"
        ),
        row=1, col=2
    )

    fig_iso.update_xaxes(title_text="Dimension", row=1, col=1)
    fig_iso.update_yaxes(title_text="Dimension", row=1, col=1)
    fig_iso.update_xaxes(title_text="Sorted eigenvalue index", row=1, col=2)
    fig_iso.update_yaxes(title_text="Eigenvalue", row=1, col=2)

    fig_iso.update_layout(
        title=(
            f"{title_sa_iso}"
            f"<br><sup>trace={trace:.4f}, mean_diag={mean_diag:.4f}, "
            f"mean_abs_offdiag={mean_abs_offdiag:.4f}, eig_ratio={eig_ratio:.4f}, "
            f"mean_pairwise_cos={mean_pairwise_cos:.4f}</sup>"
        ),
        template="plotly_white",
        showlegend=False,
        margin=dict(l=40, r=40, t=90, b=40)
    )

    # Explicit heights
    fig_w.update_layout(height=500)
    fig_task_2d.update_layout(height=500)
    fig_sa_fixed_2d.update_layout(height=500)
    fig_iso.update_layout(height=500)

    # ---------------------------
    # Save one HTML
    # ---------------------------
    save_html = Path(save_html)
    save_html.parent.mkdir(parents=True, exist_ok=True)

    html_w = fig_w.to_html(
        full_html=False,
        include_plotlyjs="cdn",
        config={"responsive": True},
        default_width="100%",
        default_height="100%",
    )

    html_task = fig_task_2d.to_html(
        full_html=False,
        include_plotlyjs=False,
        config={"responsive": True},
        default_width="100%",
        default_height="100%",
    )

    html_sa_fix = fig_sa_fixed_2d.to_html(
        full_html=False,
        include_plotlyjs=False,
        config={"responsive": True},
        default_width="100%",
        default_height="100%",
    )

    html_iso = fig_iso.to_html(
        full_html=False,
        include_plotlyjs=False,
        config={"responsive": True},
        default_width="100%",
        default_height="100%",
    )

    dashboard_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Embedding Dashboard</title>
    <style>
        html, body {{
            margin: 0;
            padding: 0;
            width: 100%;
            height: 100%;
            font-family: Arial, sans-serif;
            background: #f3f4f6;
        }}

        body {{
            overflow: hidden;
        }}

        .dashboard {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            grid-template-rows: 1fr 1fr;
            gap: 12px;
            width: 100vw;
            height: 100vh;
            padding: 12px;
            box-sizing: border-box;
        }}

        .panel {{
            background: white;
            border: 1px solid #dcdcdc;
            border-radius: 10px;
            overflow: hidden;
            min-width: 0;
            min-height: 0;
            box-shadow: 0 1px 4px rgba(0,0,0,0.06);
            display: flex;
            flex-direction: column;
        }}

        .plot-wrap {{
            flex: 1 1 auto;
            width: 100%;
            height: 100%;
            min-width: 0;
            min-height: 0;
            overflow: hidden;
        }}

        .plotly-graph-div {{
            width: 100% !important;
            height: 100% !important;
        }}
    </style>
</head>
<body>
    <div class="dashboard">
        <div class="panel">
            <div class="plot-wrap">{html_w}</div>
        </div>
        <div class="panel">
            <div class="plot-wrap">{html_task}</div>
        </div>
        <div class="panel">
            <div class="plot-wrap">{html_sa_fix}</div>
        </div>
        <div class="panel">
            <div class="plot-wrap">{html_iso}</div>
        </div>
    </div>
</body>
</html>
"""
    save_html.write_text(dashboard_html, encoding="utf-8")

    return {
        "save_html": str(save_html),
        "fig_weights": fig_w,
        "fig_task_2d": fig_task_2d,
        "fig_sa_fixed_2d": fig_sa_fixed_2d,
        "fig_sa_isotropy": fig_iso,
        "isotropy_metrics": {
            "trace": trace,
            "mean_diag": mean_diag,
            "mean_abs_offdiag": mean_abs_offdiag,
            "eig_ratio": eig_ratio,
            "mean_pairwise_cos": mean_pairwise_cos,
        },
    }