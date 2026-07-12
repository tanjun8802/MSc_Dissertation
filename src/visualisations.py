import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap, BoundaryNorm


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