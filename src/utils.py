from dataclasses import dataclass
import torch
import numpy as np


@dataclass # typed container for batches of replay data
class ReplayBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_obs: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    episode_id: torch.Tensor
    timestep: torch.Tensor
    indices: torch.Tensor

    def __len__(self):
        return self.obs.shape[0]


class TrajectoryReplayBuffer:
    def __init__(self, capacity, obs_dim, action_dim, device="cpu"):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device

        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.terminated = np.zeros((capacity, 1), dtype=np.float32)
        self.truncated = np.zeros((capacity, 1), dtype=np.float32)

        self.episode_id = np.full((capacity,), -1, dtype=np.int64)
        self.timestep = np.full((capacity,), -1, dtype=np.int64)

        self.pos = 0
        self.size = 0
        self.full = False

        self.current_episode_id = 0
        self.episode_to_indices = {}

    def __len__(self):
        return self.size

    def add_episode(self, episode):
        ep_id = self.current_episode_id
        self.current_episode_id += 1

        ep_indices = []

        T = len(episode["obs"])
        for t in range(T):
            idx = self.pos

            if self.full:
                old_ep = self.episode_id[idx]
                if old_ep in self.episode_to_indices:
                    try:
                        self.episode_to_indices[old_ep].remove(idx)
                        if len(self.episode_to_indices[old_ep]) == 0:
                            del self.episode_to_indices[old_ep]
                    except ValueError:
                        pass

            self.obs[idx] = np.asarray(episode["obs"][t], dtype=np.float32)
            self.actions[idx] = np.asarray(episode["actions"][t], dtype=np.float32).reshape(-1)
            self.rewards[idx] = np.asarray([episode["rewards"][t]], dtype=np.float32)
            self.next_obs[idx] = np.asarray(episode["next_obs"][t], dtype=np.float32)
            self.terminated[idx] = np.asarray([episode["terminated"][t]], dtype=np.float32)
            self.truncated[idx] = np.asarray([episode["truncated"][t]], dtype=np.float32)

            self.episode_id[idx] = ep_id
            self.timestep[idx] = t

            ep_indices.append(idx)

            self.pos = (self.pos + 1) % self.capacity
            if self.size < self.capacity:
                self.size += 1
            else:
                self.full = True

        self.episode_to_indices[ep_id] = ep_indices

    def sample(self, batch_size):
        assert self.size > 0, "Buffer is empty"
        idxs = np.random.randint(0, self.size, size=batch_size)

        return ReplayBatch(
            obs=torch.tensor(self.obs[idxs], device=self.device),
            actions=torch.tensor(self.actions[idxs], device=self.device),
            rewards=torch.tensor(self.rewards[idxs], device=self.device),
            next_obs=torch.tensor(self.next_obs[idxs], device=self.device),
            terminated=torch.tensor(self.terminated[idxs], device=self.device),
            truncated=torch.tensor(self.truncated[idxs], device=self.device),
            episode_id=torch.tensor(self.episode_id[idxs], device=self.device),
            timestep=torch.tensor(self.timestep[idxs], device=self.device),
            indices=torch.tensor(idxs, device=self.device),
        )

    def sample_positive_future_goal_batch(self, batch_size, min_k=1, max_k=None, gamma=0.99, exclude_self_loops=True, atol=1e-6):
        
        assert self.size > 0, "Buffer is empty"
        assert 0.0 <= gamma < 1.0, "gamma must be in [0, 1)"

        valid_indices = []
        future_goal_indices = []

        tries = 0
        max_tries = batch_size * 200  # higher because we may reject self-loops

        while len(valid_indices) < batch_size and tries < max_tries:
            idx = np.random.randint(0, self.size)
            ep_id = self.episode_id[idx]
            t = self.timestep[idx]

            if ep_id == -1 or ep_id not in self.episode_to_indices:
                tries += 1
                continue

            ep_idxs = self.episode_to_indices[ep_id]
            ep_len = len(ep_idxs)

            if t >= ep_len - 1:
                tries += 1
                continue

            # Reject self-loop anchor transitions: obs -> next_obs does not move
            if exclude_self_loops:
                if np.allclose(self.obs[idx], self.next_obs[idx], atol=atol, rtol=0.0):
                    tries += 1
                    continue

            max_valid_k = ep_len - 1 - t
            if max_k is not None:
                max_valid_k = min(max_valid_k, max_k)

            if max_valid_k < min_k:
                tries += 1
                continue

            ks = np.arange(min_k, max_valid_k + 1, dtype=np.int64)

            if gamma == 0.0:
                probs = np.zeros_like(ks, dtype=np.float64)
                probs[0] = 1.0
            else:
                log_weights = (ks - 1) * np.log(gamma)
                log_weights -= np.max(log_weights)
                weights = np.exp(log_weights)
                probs = weights / weights.sum()

            k = np.random.choice(ks, p=probs)
            future_t = t + k
            future_idx = ep_idxs[future_t]

            valid_indices.append(idx)
            future_goal_indices.append(future_idx)
            tries += 1

        assert len(valid_indices) == batch_size, (
            f"Could only sample {len(valid_indices)} valid transitions out of "
            f"requested {batch_size}. Increase max_tries or collect more non-self-loop data."
        )

        idxs = np.array(valid_indices, dtype=np.int64)
        g_idxs = np.array(future_goal_indices, dtype=np.int64)

        batch = {
            "obs": torch.tensor(self.obs[idxs], device=self.device),
            "actions": torch.tensor(self.actions[idxs], device=self.device),
            "next_obs": torch.tensor(self.next_obs[idxs], device=self.device),
            "future_state": torch.tensor(self.obs[g_idxs], device=self.device),
            "rewards": torch.tensor(self.rewards[idxs], device=self.device),
            "terminated": torch.tensor(self.terminated[idxs], device=self.device),
            "truncated": torch.tensor(self.truncated[idxs], device=self.device),
            "episode_id": torch.tensor(self.episode_id[idxs], device=self.device),
            "timestep": torch.tensor(self.timestep[idxs], device=self.device),
            "future_timestep": torch.tensor(self.timestep[g_idxs], device=self.device),
            "indices": torch.tensor(idxs, device=self.device),
            "goal_indices": torch.tensor(g_idxs, device=self.device),
        }
        return batch

    def sample_negative_future_goals(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size) # pick random indices from the buffer and indexing the observations
        return torch.tensor(self.obs[idxs], device=self.device)
    
    def sample_random_goals(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size) # pick random indices from the buffer and indexing the observations
        return torch.tensor(self.obs[idxs], device=self.device)
    
    def sample_positive_future_goal_episode(self, episode_index, timestep, k, gamma = 0.99):

        if episode_index not in self.episode_to_indices:
            raise ValueError(f"Episode index {episode_index} not found in buffer")

        ep_idxs = self.episode_to_indices[episode_index]
        ep_len = len(ep_idxs)

        if timestep >= ep_len - 1:
            raise ValueError(f"Timestep {timestep} is out of bounds for episode of length {ep_len}")

        max_valid_k = ep_len - 1 - timestep
        if k > max_valid_k:
            raise ValueError(f"k={k} is too large for episode of length {ep_len} at timestep {timestep}")

        geometric = torch.distributions.Geometric(probs=torch.tensor(1-gamma)) # geometric distribution to sample k with probability proportional to gamma^k
        steps_ahead = int(geometric.sample().item()) + 1 # sample k, cast to int, and add 1 to ensure it's at least t+1
        steps_ahead = min(steps_ahead, max_valid_k) # clamp so we never index past the end of the episode
        print(f"Sampled k={steps_ahead} from geometric distribution with gamma={gamma}")
        future_t = timestep + steps_ahead
        future_idx = ep_idxs[future_t]
        print(self.obs[future_idx])
        return torch.tensor(self.obs[future_idx], device=self.device) # return the future state as the positive goal

    def stats(self):
        return {
            "size": self.size,
            "capacity": self.capacity,
            "num_episodes": len(self.episode_to_indices),
            "current_episode_id": self.current_episode_id,
        }
    

class TrajectoryReplayBufferDiscrete(TrajectoryReplayBuffer):
    def __init__(self, capacity, obs_dim, action_dim, device="cpu"):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device

        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.terminated = np.zeros((capacity, 1), dtype=np.float32)
        self.truncated = np.zeros((capacity, 1), dtype=np.float32)

        self.episode_id = np.full((capacity,), -1, dtype=np.int64)
        self.timestep = np.full((capacity,), -1, dtype=np.int64)

        self.pos = 0
        self.size = 0
        self.full = False

        self.current_episode_id = 0
        self.episode_to_indices = {}

    def __len__(self):
        return self.size

    def add_episode(self, episode):
        ep_id = self.current_episode_id
        self.current_episode_id += 1

        ep_indices = []

        T = len(episode["obs"])
        for t in range(T):
            idx = self.pos

            if self.full:
                old_ep = self.episode_id[idx]
                if old_ep in self.episode_to_indices:
                    try:
                        self.episode_to_indices[old_ep].remove(idx)
                        if len(self.episode_to_indices[old_ep]) == 0:
                            del self.episode_to_indices[old_ep]
                    except ValueError:
                        pass

            self.obs[idx] = np.asarray(episode["obs"][t], dtype=np.float32)
            self.actions[idx] = np.int64(episode["actions"][t])
            self.rewards[idx] = np.asarray([episode["rewards"][t]], dtype=np.float32)
            self.next_obs[idx] = np.asarray(episode["next_obs"][t], dtype=np.float32)
            self.terminated[idx] = np.asarray([episode["terminated"][t]], dtype=np.float32)
            self.truncated[idx] = np.asarray([episode["truncated"][t]], dtype=np.float32)

            self.episode_id[idx] = ep_id
            self.timestep[idx] = t

            ep_indices.append(idx)

            self.pos = (self.pos + 1) % self.capacity
            if self.size < self.capacity:
                self.size += 1
            else:
                self.full = True

        self.episode_to_indices[ep_id] = ep_indices

    def sample(self, batch_size):
        assert self.size > 0, "Buffer is empty"
        idxs = np.random.randint(0, self.size, size=batch_size)

        return ReplayBatch(
            obs=torch.tensor(self.obs[idxs], device=self.device),
            actions=torch.tensor(self.actions[idxs], device=self.device),
            rewards=torch.tensor(self.rewards[idxs], device=self.device),
            next_obs=torch.tensor(self.next_obs[idxs], device=self.device),
            terminated=torch.tensor(self.terminated[idxs], device=self.device),
            truncated=torch.tensor(self.truncated[idxs], device=self.device),
            episode_id=torch.tensor(self.episode_id[idxs], device=self.device),
            timestep=torch.tensor(self.timestep[idxs], device=self.device),
            indices=torch.tensor(idxs, device=self.device),
        )

    def sample_positive_future_goal_batch(self, batch_size, min_k=1, max_k=None, gamma=0.99, exclude_self_loops=True, atol=1e-6):
        
        assert self.size > 0, "Buffer is empty"
        assert 0.0 <= gamma < 1.0, "gamma must be in [0, 1)"

        valid_indices = []
        future_goal_indices = []

        tries = 0
        max_tries = batch_size * 300  # higher because we may reject self-loops

        while len(valid_indices) < batch_size and tries < max_tries:
            idx = np.random.randint(0, self.size)
            ep_id = self.episode_id[idx]
            t = self.timestep[idx]

            if ep_id == -1 or ep_id not in self.episode_to_indices:
                tries += 1
                continue

            ep_idxs = self.episode_to_indices[ep_id]
            ep_len = len(ep_idxs)

            if t >= ep_len - 1:
                tries += 1
                continue

            # Reject self-loop anchor transitions: obs -> next_obs does not move
            if exclude_self_loops:
                if np.allclose(self.obs[idx], self.next_obs[idx], atol=atol, rtol=0.0):
                    tries += 1
                    continue

            max_valid_k = ep_len - 1 - t
            if max_k is not None:
                max_valid_k = min(max_valid_k, max_k)

            if max_valid_k < min_k:
                tries += 1
                continue

            ks = np.arange(min_k, max_valid_k + 1, dtype=np.int64)

            if gamma == 0.0:
                probs = np.zeros_like(ks, dtype=np.float64)
                probs[0] = 1.0
            else:
                log_weights = (ks - 1) * np.log(gamma)
                log_weights -= np.max(log_weights)
                weights = np.exp(log_weights)
                probs = weights / weights.sum()

            k = np.random.choice(ks, p=probs)
            future_t = t + k
            future_idx = ep_idxs[future_t]

            valid_indices.append(idx)
            future_goal_indices.append(future_idx)
            tries += 1

        assert len(valid_indices) == batch_size, (
            f"Could only sample {len(valid_indices)} valid transitions out of "
            f"requested {batch_size}. Increase max_tries or collect more non-self-loop data."
        )

        idxs = np.array(valid_indices, dtype=np.int64)
        g_idxs = np.array(future_goal_indices, dtype=np.int64)

        batch = {
            "obs": torch.tensor(self.obs[idxs], device=self.device),
            "actions": torch.tensor(self.actions[idxs], device=self.device),
            "next_obs": torch.tensor(self.next_obs[idxs], device=self.device),
            "future_state": torch.tensor(self.obs[g_idxs], device=self.device),
            "rewards": torch.tensor(self.rewards[idxs], device=self.device),
            "terminated": torch.tensor(self.terminated[idxs], device=self.device),
            "truncated": torch.tensor(self.truncated[idxs], device=self.device),
            "episode_id": torch.tensor(self.episode_id[idxs], device=self.device),
            "timestep": torch.tensor(self.timestep[idxs], device=self.device),
            "future_timestep": torch.tensor(self.timestep[g_idxs], device=self.device),
            "indices": torch.tensor(idxs, device=self.device),
            "goal_indices": torch.tensor(g_idxs, device=self.device),
        }
        return batch

    def sample_negative_future_goals(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size) # pick random indices from the buffer and indexing the observations
        return torch.tensor(self.obs[idxs], device=self.device)
    
    def sample_random_goals(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size) # pick random indices from the buffer and indexing the observations
        return torch.tensor(self.obs[idxs], device=self.device)
    
    def sample_positive_future_goal_episode(self, episode_index, timestep, k, gamma = 0.99):

        if episode_index not in self.episode_to_indices:
            raise ValueError(f"Episode index {episode_index} not found in buffer")

        ep_idxs = self.episode_to_indices[episode_index]
        ep_len = len(ep_idxs)

        if timestep >= ep_len - 1:
            raise ValueError(f"Timestep {timestep} is out of bounds for episode of length {ep_len}")

        max_valid_k = ep_len - 1 - timestep
        if k > max_valid_k:
            raise ValueError(f"k={k} is too large for episode of length {ep_len} at timestep {timestep}")

        geometric = torch.distributions.Geometric(probs=torch.tensor(1-gamma)) # geometric distribution to sample k with probability proportional to gamma^k
        steps_ahead = int(geometric.sample().item()) + 1 # sample k, cast to int, and add 1 to ensure it's at least t+1
        steps_ahead = min(steps_ahead, max_valid_k) # clamp so we never index past the end of the episode
        print(f"Sampled k={steps_ahead} from geometric distribution with gamma={gamma}")
        future_t = timestep + steps_ahead
        future_idx = ep_idxs[future_t]
        print(self.obs[future_idx])
        return torch.tensor(self.obs[future_idx], device=self.device) # return the future state as the positive goal

    def stats(self):
        return {
            "size": self.size,
            "capacity": self.capacity,
            "num_episodes": len(self.episode_to_indices),
            "current_episode_id": self.current_episode_id,
        }

# ---------------------------------------------------------------------------
# Shared visualisation helpers
# ---------------------------------------------------------------------------


def plot_policy_rollouts(
    env,
    policy_fn,
    goal_pos=(9, 9),
    eval_episodes=8,
    n_cols=4,
    is_discrete=False,
    step_point_size=10,
    start_size=55,
    end_size=45,
    goal_size=120,
    arrow_width=0.008,
):
    """Run evaluation rollouts and plot the trajectories on the Four-Rooms grid.

    Parameters
    ----------
    env : gymnasium.Env
        A (wrapped) FourRooms environment whose ``unwrapped`` core exposes
        ``grid_size`` and ``_blocked_cells``.
    policy_fn : callable
        ``policy_fn(obs_np: np.ndarray) -> action`` – returns an ``int`` for
        discrete environments or a 1-D ``np.ndarray`` for continuous ones.
    goal_pos : tuple[int, int]
        Integer cell coordinates ``(x, y)`` of the evaluation goal.
    eval_episodes : int
        Number of episodes to roll out.
    n_cols : int
        Number of subplot columns in the grid figure.
    is_discrete : bool
        When *True* observations are integer cell coordinates and goal-reaching
        is determined by exact cell equality.  When *False* observations are
        continuous float positions and goal-reaching is determined by the
        ``terminated`` flag returned by the environment.
    step_point_size, start_size, end_size, goal_size : int
        Scatter marker sizes for individual steps, start, end, and goal.
    arrow_width : float
        Width of the quiver arrows.
    """
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
    """Visualise Q-values or state values over the Four-Rooms grid.

    Parameters
    ----------
    env : gymnasium.Env
        A (wrapped) FourRooms environment whose ``unwrapped`` core exposes
        ``grid_size`` and ``_blocked_cells``.
    value_fn : callable
        - **Discrete** (``is_discrete=True``):
          ``value_fn(obs_batch: np.ndarray[N, obs_dim]) -> np.ndarray[N, num_actions]``
          – returns per-action Q-values for every state in the batch.
        - **Continuous** (``is_discrete=False``):
          ``value_fn(obs_batch: np.ndarray[N, obs_dim]) -> np.ndarray[N]``
          – returns a scalar value (V(s) or Q(s, pi(s))) for every state.
    actor_fn : callable or None
        Used only when ``is_discrete=False``.
        ``actor_fn(obs_batch: np.ndarray[N, obs_dim]) -> np.ndarray[N, act_dim]``
        – returns the deterministic action for every state.  When provided, the
        second panel shows the greedy action direction as a quiver plot.
    is_discrete : bool
        When *True* treats observations as integer cell coordinates and queries
        ``value_fn`` with those integer coords; shows a max-Q heatmap and a
        greedy-action colour map.  When *False* treats cell centres as
        continuous observations; shows a value heatmap and an optional action-
        direction quiver (requires ``actor_fn``).
    num_actions : int
        Number of discrete actions – only used when ``is_discrete=True``.
    action_names : list[str] or None
        Display names for each discrete action.  Defaults to
        ``['a0', 'a1', ...]``.
    goal_pos : tuple[int, int] or None
        When provided, marks the goal cell with a star on the value/Q panel.
    eval_returns : list of (int, float) or None
        Training-curve data as ``[(env_step, mean_return), ...]``.  Shown in
        the third panel; left blank when *None*.
    figsize : tuple
        Figure size passed to ``plt.subplots``.
    """
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
