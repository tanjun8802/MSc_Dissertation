import numpy as np
import gymnasium as gym
from gymnasium import spaces


class MazeGridWorld(gym.Env):
    """
    Generic discrete 2D maze:

    - observation: (x, y) integer coordinates (Box, shape=(2,))
    - action: Discrete(4) — 0:up, 1:down, 2:left, 3:right
    - walls from `maze` matrix: 0 = free, 1 = wall
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, maze, start=None, max_episode_steps=500):
        super().__init__()

        self.maze = np.array(maze, dtype=np.int32)
        assert self.maze.ndim == 2
        self.height, self.width = self.maze.shape
        self.max_episode_steps = int(max_episode_steps)

        self._free_cells = []
        self._blocked_cells = []
        for y in range(self.height):
            for x in range(self.width):
                if self.maze[y, x] == 0:
                    self._free_cells.append((x, y))
                else:
                    self._blocked_cells.append((x, y))
        self._free_cells = list(self._free_cells)
        self._blocked_cells = list(self._blocked_cells)
        self._outer_wall_cells = []

        if start is None:
            self.start = tuple(self._free_cells[0])
        else:
            self.start = tuple(start)

        self.agent_pos = None
        self._goal_pos = None
        self.t = 0

        self.grid_size = max(self.width, self.height)

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=np.array([0, 0], dtype=np.float32),
            high=np.array([self.width - 1, self.height - 1], dtype=np.float32),
            shape=(2,),
            dtype=np.float32,
        )

        self._action_to_direction = {
            0: np.array([0, 1], dtype=np.int32),   # up
            1: np.array([0, -1], dtype=np.int32),  # down
            2: np.array([-1, 0], dtype=np.int32),  # left
            3: np.array([1, 0], dtype=np.int32),   # right
        }
        self.action_names = ["Up", "Down", "Left", "Right"]

    def set_goal_position(self, goal_position):
        goal = np.asarray(goal_position, dtype=np.int32)
        if goal.shape != (2,):
            raise ValueError(f"Invalid goal position shape: {goal_position}")

        x, y = int(goal[0]), int(goal[1])
        if not (0 <= x < self.width and 0 <= y < self.height):
            raise ValueError(f"Invalid goal position: {goal_position}")
        if self.maze[y, x] == 1:
            raise ValueError(f"Goal lies in wall cell: {goal_position}")

        self._goal_pos = goal.copy()

    def valid_action_mask(self, obs):
        cell = np.asarray(obs, dtype=np.int32)
        x, y = int(cell[0]), int(cell[1])

        candidate_next = {
            0: np.array([x, y + 1], dtype=np.int32),  # up
            1: np.array([x, y - 1], dtype=np.int32),  # down
            2: np.array([x - 1, y], dtype=np.int32),  # left
            3: np.array([x + 1, y], dtype=np.int32),  # right
        }

        mask = np.zeros(4, dtype=bool)
        for a, nxt in candidate_next.items():
            nx, ny = int(nxt[0]), int(nxt[1])
            mask[a] = 0 <= nx < self.width and 0 <= ny < self.height and self.maze[ny, nx] == 0

        if not mask.any():
            mask[:] = True

        return mask

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        options = options or {}

        if "start_position" in options:
            self.agent_pos = tuple(np.asarray(options["start_position"], dtype=np.int32).tolist())
        elif "start_pos" in options:
            self.agent_pos = tuple(np.asarray(options["start_pos"], dtype=np.int32).tolist())
        else:
            idx = int(self.np_random.integers(len(self._free_cells)))
            self.agent_pos = tuple(self._free_cells[idx])

        obs = np.array(self.agent_pos, dtype=np.float32)
        return obs, {}

    def step(self, action):
        self.t += 1
        move = self._action_to_direction[int(action)]
        x, y = self.agent_pos
        nx = x + int(move[0])
        ny = y + int(move[1])

        if 0 <= nx < self.width and 0 <= ny < self.height and self.maze[ny, nx] == 0:
            self.agent_pos = (nx, ny)

        obs = np.array(self.agent_pos, dtype=np.float32)
        reward = 0.0
        terminated = False
        truncated = self.t >= self.max_episode_steps
        return obs, reward, terminated, truncated, {}

    def render(self):
        cell = 40
        img = np.ones((self.height * cell, self.width * cell, 3), dtype=np.uint8) * 255

        for y in range(self.height):
            for x in range(self.width):
                y0, y1 = y * cell, (y + 1) * cell
                x0, x1 = x * cell, (x + 1) * cell

                if self.maze[y, x] == 1:
                    img[y0:y1, x0:x1] = np.array([30, 30, 30], dtype=np.uint8)
                else:
                    img[y0:y1, x0:x1] = np.array([240, 240, 240], dtype=np.uint8)

                img[y0:y0+1, x0:x1] = 180
                img[y1-1:y1, x0:x1] = 180
                img[y0:y1, x0:x0+1] = 180
                img[y0:y1, x1-1:x1] = 180

        if self._goal_pos is not None:
            gx, gy = int(self._goal_pos[0]), int(self._goal_pos[1])
            y0, y1 = gy * cell, (gy + 1) * cell
            x0, x1 = gx * cell, (gx + 1) * cell
            img[y0+8:y1-8, x0+8:x1-8] = np.array([50, 180, 50], dtype=np.uint8)

        if self.agent_pos is not None:
            ax, ay = self.agent_pos
            y0, y1 = ay * cell, (ay + 1) * cell
            x0, x1 = ax * cell, (ax + 1) * cell
            img[y0+8:y1-8, x0+8:x1-8] = np.array([60, 120, 255], dtype=np.uint8)

        return img


class MazeGoalWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        goal_position=(1, 1),
        goal_reward=1.0,
        step_reward=0.0,
        slip_prob=0.0,
        reward_mode="goal",
        wall_penalty=-0.1,
        gamma=0.99,
        track_coverage=True,
    ):
        super().__init__(env)
        self.goal_position = np.asarray(goal_position, dtype=np.int32)
        self.goal_reward = float(goal_reward)
        self.step_reward = float(step_reward)
        self.slip_prob = float(np.clip(slip_prob, 0.0, 1.0))
        self.reward_mode = str(reward_mode).lower()
        self.wall_penalty = float(wall_penalty)
        self.gamma = float(gamma)

        self.track_coverage = bool(track_coverage)
        self.current_task = None
        self.global_visited = set()
        self.episode_visited = set()
        self.task_visited = {}
        self.task_episode_counts = {}
        self.task_new_state_curve = {}
        self.task_reuse_curve = {}

        self._sync_goal_to_env()

    def _sync_goal_to_env(self):
        if hasattr(self.env, "set_goal_position"):
            self.env.set_goal_position(self.goal_position)
        else:
            self.env._goal_pos = np.asarray(self.goal_position, dtype=np.int32).copy()

    def _state_id_from_obs(self, obs):
        s = np.asarray(obs, dtype=np.int32).reshape(-1)
        if s.shape[0] != 2:
            raise ValueError(f"Expected obs shape (2,), got {s.shape}")
        return (int(s[0]), int(s[1]))

    def set_task(self, task_id):
        self.current_task = task_id
        if task_id not in self.task_visited:
            self.task_visited[task_id] = set()
            self.task_episode_counts[task_id] = 0
            self.task_new_state_curve[task_id] = []
            self.task_reuse_curve[task_id] = []

    def _earlier_task_states(self):
        earlier = set()
        if self.current_task is None:
            return earlier
        for tid, states in self.task_visited.items():
            if tid != self.current_task:
                earlier |= states
        return earlier

    def _update_coverage_on_reset(self, obs):
        if not self.track_coverage:
            return

        state = self._state_id_from_obs(obs)
        self.episode_visited = {state}
        self.global_visited.add(state)

        if self.current_task is not None:
            self.task_visited[self.current_task].add(state)

    def _update_coverage_on_step(self, obs):
        if not self.track_coverage:
            return

        state = self._state_id_from_obs(obs)
        self.episode_visited.add(state)
        self.global_visited.add(state)

        if self.current_task is not None:
            self.task_visited[self.current_task].add(state)

    def _finalize_episode_coverage(self):
        if not self.track_coverage or self.current_task is None:
            return

        self.task_episode_counts[self.current_task] += 1

        earlier = self._earlier_task_states()
        current = self.task_visited[self.current_task]

        new_states = current - earlier
        reused_states = current & earlier
        reuse_ratio = 0.0 if len(current) == 0 else len(reused_states) / len(current)

        self.task_new_state_curve[self.current_task].append(len(new_states))
        self.task_reuse_curve[self.current_task].append(reuse_ratio)

    def state_reuse_ratio(self):
        if not self.track_coverage or self.current_task is None:
            return 0.0
        current = self.task_visited[self.current_task]
        if len(current) == 0:
            return 0.0
        earlier = self._earlier_task_states()
        reused = current & earlier
        return len(reused) / len(current)

    def new_states_in_current_task(self):
        if not self.track_coverage or self.current_task is None:
            return 0
        current = self.task_visited[self.current_task]
        earlier = self._earlier_task_states()
        return len(current - earlier)

    def coverage_stats(self):
        stats = {
            "global_unique_states": len(self.global_visited),
            "current_task": self.current_task,
        }
        if self.current_task is not None:
            stats["task_unique_states"] = len(self.task_visited[self.current_task])
            stats["task_new_states"] = self.new_states_in_current_task()
            stats["task_reuse_ratio"] = self.state_reuse_ratio()
            stats["episodes_in_task"] = self.task_episode_counts[self.current_task]
        return stats

    def set_goal_position(self, goal_position):
        self.goal_position = np.asarray(goal_position, dtype=np.int32)
        self._sync_goal_to_env()

    def compute_simple_reward(self, state, action, next_state, goal):
        sx, sy = state
        nx, ny = next_state
        gx, gy = goal

        if (nx, ny) == (gx, gy):
            return self.goal_reward
        if (nx, ny) == (sx, sy):
            return self.wall_penalty
        return self.step_reward

    def compute_shaped_reward(self, state, action, next_state, goal):
        sx, sy = state
        nx, ny = next_state
        gx, gy = goal

        if (nx, ny) == (gx, gy):
            base_r = self.goal_reward
        elif (nx, ny) == (sx, sy):
            base_r = self.wall_penalty
        else:
            base_r = self.step_reward

        width = int(self.env.unwrapped.width)
        height = int(self.env.unwrapped.height)
        max_d = (width - 1) + (height - 1)

        if max_d <= 0:
            return base_r

        def phi(xy):
            x, y = xy
            d = abs(gx - x) + abs(gy - y)
            return 1.0 - d / max_d

        phi_s = phi(state)
        phi_sp = phi(next_state)

        shaping = self.gamma * phi_sp - phi_s
        return base_r + shaping

    def reset(self, *, seed=None, options=None):
        options = options or {}
        if "goal_position" in options:
            self.set_goal_position(options["goal_position"])

        obs, info = self.env.reset(seed=seed, options=options)
        self._sync_goal_to_env()
        self._update_coverage_on_reset(obs)

        info = dict(info)
        info["goal_position"] = self.goal_position.astype(np.int32).tolist()

        if self.track_coverage:
            info["state"] = list(self._state_id_from_obs(obs))
            info["episode_unique_states"] = len(self.episode_visited)
            info["global_unique_states"] = len(self.global_visited)
            if self.current_task is not None:
                info["task_unique_states"] = len(self.task_visited[self.current_task])
                info["task_new_states"] = self.new_states_in_current_task()
                info["task_reuse_ratio"] = self.state_reuse_ratio()

        return obs, info

    def step(self, action):
        if self.slip_prob > 0.0 and self.env.np_random.random() < self.slip_prob:
            action = int(self.env.np_random.integers(0, self.env.action_space.n))

        state = tuple(np.asarray(self.env.unwrapped.agent_pos, dtype=np.int32).tolist())
        obs, _, terminated, truncated, info = self.env.step(action)

        reached = np.array_equal(obs.astype(np.int32), self.goal_position)
        next_state = tuple(obs.astype(np.int32).tolist())
        goal = tuple(self.goal_position.astype(np.int32).tolist())

        if self.reward_mode == "simple":
            reward = self.compute_simple_reward(
                state=state,
                action=action,
                next_state=next_state,
                goal=goal,
            )
        elif self.reward_mode == "shaped":
            reward = self.compute_shaped_reward(
                state=state,
                action=action,
                next_state=next_state,
                goal=goal,
            )
        else:
            reward = self.goal_reward if reached else self.step_reward

        if reached:
            terminated = True

        self._update_coverage_on_step(obs)

        if terminated or truncated:
            self._finalize_episode_coverage()

        info = dict(info)
        info["goal_position"] = self.goal_position.astype(np.int32).tolist()
        info["success"] = reached

        if self.track_coverage:
            info["state"] = list(self._state_id_from_obs(obs))
            info["episode_unique_states"] = len(self.episode_visited)
            info["global_unique_states"] = len(self.global_visited)
            if self.current_task is not None:
                info["task_unique_states"] = len(self.task_visited[self.current_task])
                info["task_new_states"] = self.new_states_in_current_task()
                info["task_reuse_ratio"] = self.state_reuse_ratio()

        return obs, reward, terminated, truncated, info
    