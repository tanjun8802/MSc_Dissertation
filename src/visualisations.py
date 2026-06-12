import torch
import numpy as np

def plot_policy_rollouts(env, policy_fn, 
                         goal_pos=(9, 9), 
                         eval_episodes=8, 
                         n_cols=4, 
                         is_discrete=False, 
                         step_point_size=10, 
                         start_size=55, 
                         end_size=45, 
                         goal_size=120, 
                         arrow_width=0.008):

    import math
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    env_core = env.unwrapped
    blocked_cells = env_core._blocked_cells
    grid_size = env_core.grid_size

    goal_xy = np.array(goal_pos, dtype=np.int32)

    ep_rewards = []
    ep_successes = []
    trajectories = []

    for ep_i in range(eval_episodes):
        obs, _ = env.reset(options={"goal_position": goal_pos})
        done = False
        ep_rew = 0.0
        terminated_flag = False
        trajectory = [obs.copy()]

        while not done:
            action = policy_fn(obs)
            obs, rew, term, trunc, _ = env.step(action)
            ep_rew += float(rew)
            trajectory.append(obs.copy())
            done = term or trunc
            if term:
                terminated_flag = True

        trajectory = np.asarray(trajectory, dtype=np.float32)
        trajectories.append(trajectory)
        ep_rewards.append(ep_rew)

        if is_discrete:
            reached = np.array_equal(trajectory[-1].astype(np.int32), goal_xy)
        else:
            reached = terminated_flag

        ep_successes.append(reached)
        print(
            f"Episode {ep_i + 1}: {len(trajectory) - 1} steps | "
            f"return: {ep_rew:.2f} | reached goal: {reached}"
        )

    success_rate = float(np.mean(ep_successes)) if ep_successes else 0.0
    avg_return = float(np.mean(ep_rewards)) if ep_rewards else 0.0
    std_return = float(np.std(ep_rewards)) if ep_rewards else 0.0

    print(f"\nSuccess rate: {success_rate:.2%} over {eval_episodes} episodes")
    print(f"Average return: {avg_return:.2f} \u00b1 {std_return:.2f}")

    def overlay_walls(ax):
        for x, y in blocked_cells:
            ax.add_patch(
                Rectangle(
                    (x, y), 1.0, 1.0,
                    facecolor="gray",
                    edgecolor="gray",
                    linewidth=0.0,
                    alpha=0.85,
                    zorder=1,
                )
            )

    def draw_gridworld(ax):
        ax.set_xlim(0, grid_size)
        ax.set_ylim(0, grid_size)
        ax.set_aspect("equal")
        ax.set_xticks(np.arange(0, grid_size + 1, 1))
        ax.set_yticks(np.arange(0, grid_size + 1, 1))
        ax.grid(color="lightgray", linewidth=0.8, alpha=0.7)
        ax.tick_params(labelsize=8, length=0)
        overlay_walls(ax)
        ax.scatter(
            goal_pos[0] + 0.5,
            goal_pos[1] + 0.5,
            c="green",
            marker="*",
            s=goal_size,
            zorder=6,
            label="Goal",
        )

    n_rows = math.ceil(eval_episodes / n_cols)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.4 * n_cols, 4.4 * n_rows),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    for i, ax in enumerate(axes_flat):
        if i >= len(trajectories):
            ax.axis("off")
            continue

        draw_gridworld(ax)

        traj = trajectories[i]
        # For discrete envs the obs is an integer cell index; add 0.5 to plot
        # at the cell centre.  For continuous envs the obs is already a
        # floating-point position within the grid.
        if is_discrete:
            xs = traj[:, 0] + 0.5
            ys = traj[:, 1] + 0.5
        else:
            xs = traj[:, 0]
            ys = traj[:, 1]

        ax.scatter(
            xs, ys,
            c=np.arange(len(xs)),
            cmap="Blues",
            s=step_point_size,
            zorder=4,
        )

        dx = xs[1:] - xs[:-1]
        dy = ys[1:] - ys[:-1]
        ax.quiver(
            xs[:-1], ys[:-1], dx, dy,
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
        f"Policy rollouts toward goal {goal_pos}\n"
        f"Success rate: {success_rate:.2%} | "
        f"Avg return: {avg_return:.2f} \u00b1 {std_return:.2f}",
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

    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.colors import ListedColormap, BoundaryNorm

    if action_names is None:
        action_names = [f"a{i}" for i in range(num_actions)]

    env_core = env.unwrapped
    grid_size = env_core.grid_size
    blocked_cells = env_core._blocked_cells
    extent = [0, grid_size, 0, grid_size]

    def overlay_walls(ax, facecolor="black", alpha=1.0):
        for x, y in blocked_cells:
            ax.add_patch(
                Rectangle(
                    (x, y), 1.0, 1.0,
                    facecolor=facecolor,
                    alpha=alpha,
                    edgecolor=None,
                    linewidth=0.0,
                    zorder=5,
                )
            )

    def _setup_grid_ax(ax, title):
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim(0, grid_size)
        ax.set_ylim(0, grid_size)
        ax.set_xticks(np.arange(0, grid_size + 1, 1))
        ax.set_yticks(np.arange(0, grid_size + 1, 1))
        ax.grid(color="lightgray", linewidth=0.8, alpha=0.6)

    fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)

    # ------------------------------------------------------------------
    # Panel 1 – value / max-Q heatmap
    # ------------------------------------------------------------------
    if is_discrete:
        q_best = np.full((grid_size, grid_size), np.nan, dtype=np.float32)
        best_action_idx = np.full((grid_size, grid_size), -1, dtype=np.int32)

        for y in range(grid_size):
            xs_np = np.arange(grid_size, dtype=np.float32)
            obs_batch = np.stack(
                [xs_np, np.full_like(xs_np, y)], axis=1
            )                                                    # [grid_size, 2]
            q_vals = np.asarray(
                value_fn(obs_batch), dtype=np.float32
            )                                                    # [grid_size, num_actions]

            for x in range(grid_size):
                if (x, y) in blocked_cells:
                    continue
                q_best[y, x] = q_vals[x].max()
                best_action_idx[y, x] = int(q_vals[x].argmax())

        im0 = axes[0].imshow(
            q_best,
            origin="lower",
            aspect="equal",
            extent=extent,
            cmap="viridis",
            interpolation="nearest",
        )
        overlay_walls(axes[0])
        if goal_pos is not None:
            axes[0].scatter(
                goal_pos[0] + 0.5, goal_pos[1] + 0.5,
                c="red", s=100, marker="*", zorder=6, label="goal",
            )
            axes[0].legend(loc="upper right")
        _setup_grid_ax(axes[0], "Max Q(s, a) over discrete actions")
        fig.colorbar(im0, ax=axes[0], shrink=0.9, label="Q value")

    else:
        val_map = np.full((grid_size, grid_size), np.nan, dtype=np.float32)

        for y in range(grid_size):
            xs_np = np.arange(grid_size, dtype=np.float32)
            # Sample at cell centres for continuous observations
            obs_batch = np.stack(
                [xs_np + 0.5, np.full_like(xs_np, y + 0.5)], axis=1
            )                                                    # [grid_size, 2]
            vals = np.asarray(
                value_fn(obs_batch), dtype=np.float32
            )                                                    # [grid_size]

            for x in range(grid_size):
                if (x, y) in blocked_cells:
                    continue
                val_map[y, x] = vals[x]

        im0 = axes[0].imshow(
            val_map,
            origin="lower",
            aspect="equal",
            extent=extent,
            cmap="viridis",
            interpolation="nearest",
        )
        overlay_walls(axes[0])
        if goal_pos is not None:
            axes[0].scatter(
                goal_pos[0] + 0.5, goal_pos[1] + 0.5,
                c="red", s=100, marker="*", zorder=6, label="goal",
            )
            axes[0].legend(loc="upper right")
        _setup_grid_ax(axes[0], "State value V(s) / Q(s, \u03c0(s))")
        fig.colorbar(im0, ax=axes[0], shrink=0.9, label="Value")

    # ------------------------------------------------------------------
    # Panel 2 – greedy-action colour map (discrete) or quiver (continuous)
    # ------------------------------------------------------------------
    if is_discrete:
        _COLOURS = [
            "tab:blue", "tab:orange", "tab:green", "tab:red",
            "tab:purple", "tab:brown", "tab:pink", "tab:gray",
        ]
        action_cmap = ListedColormap(_COLOURS[:num_actions])
        norm = BoundaryNorm(
            np.arange(-0.5, num_actions + 0.5, 1), action_cmap.N
        )
        masked_actions = np.ma.masked_where(
            best_action_idx < 0, best_action_idx
        )
        im1 = axes[1].imshow(
            masked_actions,
            origin="lower",
            aspect="equal",
            extent=extent,
            cmap=action_cmap,
            norm=norm,
            interpolation="nearest",
        )
        overlay_walls(axes[1])
        if goal_pos is not None:
            axes[1].scatter(
                goal_pos[0] + 0.5, goal_pos[1] + 0.5,
                c="white", s=100, marker="*", edgecolors="black", zorder=6,
            )
        _setup_grid_ax(axes[1], "Greedy action argmax_a Q(s, a)")
        cbar = fig.colorbar(
            im1, ax=axes[1], shrink=0.9, ticks=list(range(num_actions))
        )
        cbar.ax.set_yticklabels(action_names)

    else:
        # Continuous: quiver plot of the greedy action direction
        if actor_fn is not None:
            quiver_xs, quiver_ys, u_vals, v_vals = [], [], [], []
            for y in range(grid_size):
                xs_np = np.arange(grid_size, dtype=np.float32)
                obs_batch = np.stack(
                    [xs_np + 0.5, np.full_like(xs_np, y + 0.5)], axis=1
                )
                acts = np.asarray(
                    actor_fn(obs_batch), dtype=np.float32
                )                                                # [grid_size, act_dim]
                for x in range(grid_size):
                    if (x, y) in blocked_cells:
                        continue
                    quiver_xs.append(x + 0.5)
                    quiver_ys.append(y + 0.5)
                    u_vals.append(float(acts[x, 0]))
                    v_vals.append(float(acts[x, 1]))

            quiver_xs = np.array(quiver_xs)
            quiver_ys = np.array(quiver_ys)
            u_vals = np.array(u_vals)
            v_vals = np.array(v_vals)

            axes[1].set_facecolor("whitesmoke")
            overlay_walls(axes[1], facecolor="gray", alpha=0.85)
            axes[1].quiver(
                quiver_xs, quiver_ys, u_vals, v_vals,
                angles="xy",
                scale_units="xy",
                scale=2.5,      # action magnitude 1 → 0.4 grid units
                width=0.004,
                color="tab:blue",
                alpha=0.75,
                zorder=4,
            )
            if goal_pos is not None:
                axes[1].scatter(
                    goal_pos[0] + 0.5, goal_pos[1] + 0.5,
                    c="green", s=100, marker="*", zorder=6, label="goal",
                )
                axes[1].legend(loc="upper right")
            _setup_grid_ax(axes[1], "Greedy action direction \u03c0(s)")
        else:
            axes[1].text(
                0.5, 0.5, "No actor_fn provided",
                ha="center", va="center",
            )
            axes[1].set_axis_off()

    # ------------------------------------------------------------------
    # Panel 3 – training curve
    # ------------------------------------------------------------------
    if eval_returns is not None and len(eval_returns) > 0:
        steps, rets = zip(*eval_returns)
        axes[2].plot(steps, rets, color="steelblue", linewidth=1.5)
        axes[2].set_title("Evaluation return over training")
        axes[2].set_xlabel("Environment steps")
        axes[2].set_ylabel("Mean episodic return")
        axes[2].grid(alpha=0.25)
    else:
        axes[2].text(
            0.5, 0.5, "No eval returns provided",
            ha="center", va="center",
        )
        axes[2].set_axis_off()

    plt.show()
