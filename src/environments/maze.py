import numpy as np
import gymnasium as gym
from gymnasium import spaces


class MazeGridWorld(gym.Env):
    """
    Generic continuous-position 2D maze:

    - observation: continuous (x, y) position inside free cells (Box, shape=(2,))
    - action: Box(2,) in [-1, 1], interpreted as continuous x/y movement
    - walls from `maze` matrix: 0 = free, 1 = wall
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        maze,
        start=None,
        max_episode_steps=500,
        step_scale=0.35,
        substeps=5,
        start_noise=0.15,
        goal_noise=0.0,
    ):
        super().__init__()

        self.maze = np.array(maze, dtype=np.int32)
        assert self.maze.ndim == 2
        self.height, self.width = self.maze.shape
        self.max_episode_steps = int(max_episode_steps)

        self.step_scale = float(step_scale)
        self.substeps = int(substeps)
        self.start_noise = float(start_noise)
        self.goal_noise = float(goal_noise)

        self._free_cells = []
        self._blocked_cells = []
        for y in range(self.height):
            for x in range(self.width):
                if self.maze[y, x] == 0:
                    self._free_cells.append((x, y))
                else:
                    self._blocked_cells.append((x, y))
        self._free_cells = np.asarray(self._free_cells, dtype=np.int32)
        self._blocked_cells = list(self._blocked_cells)
        self._outer_wall_cells = []

        if start is None:
            self.start = self._cell_center(self._free_cells[0]).astype(np.float32)
        else:
            start = np.asarray(start, dtype=np.float32)
            if np.allclose(start, np.round(start)):
                start_cell = np.asarray(np.round(start), dtype=np.int32)
                self.start = self._cell_center(start_cell).astype(np.float32)
            else:
                self.start = start.astype(np.float32)

        self.agent_pos = None
        self._goal_pos = None
        self.t = 0

        self.grid_size = max(self.width, self.height)

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0], dtype=np.float32),
            high=np.array([self.width - 1e-6, self.height - 1e-6], dtype=np.float32),
            shape=(2,),
            dtype=np.float32,
        )

        self.action_names = ["dx", "dy"]

    def _pos_to_cell(self, pos):
        x, y = float(pos[0]), float(pos[1])
        return int(np.floor(x)), int(np.floor(y))

    def _cell_center(self, cell):
        cell = np.asarray(cell, dtype=np.float32)
        return cell + 0.5

    def _is_valid_cell(self, cell):
        x, y = int(cell[0]), int(cell[1])
        if x < 0 or y < 0 or x >= self.width or y >= self.height:
            return False
        return self.maze[y, x] == 0

    def _is_valid_pos(self, pos):
        x, y = float(pos[0]), float(pos[1])

        if x < 0.0 or y < 0.0 or x >= self.width or y >= self.height:
            return False

        cell = self._pos_to_cell(pos)
        return self._is_valid_cell(cell)

    def _obs(self):
        return np.asarray(self.agent_pos, dtype=np.float32)

    def _action_to_delta(self, action):
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        if a.shape[0] != 2:
            raise ValueError(f"Expected action shape (2,), got {a.shape}")
        a = np.clip(a, -1.0, 1.0)
        return self.step_scale * a

    def sample_initial_state(self):
        idx = int(self.np_random.integers(0, len(self._free_cells)))
        cell = self._free_cells[idx].astype(np.float32)
        center = self._cell_center(cell)

        if self.start_noise > 0:
            noise = self.np_random.uniform(
                low=-self.start_noise, high=self.start_noise, size=(2,)
            ).astype(np.float32)
            candidate = center + noise
            if self._is_valid_pos(candidate):
                return candidate

        return center.astype(np.float32)

    def set_goal_position(self, goal_position):
        goal = np.asarray(goal_position, dtype=np.float32)
        if goal.shape != (2,):
            raise ValueError(f"Invalid goal position shape: {goal_position}")

        if np.allclose(goal, np.round(goal)):
            goal_cell = np.asarray(np.round(goal), dtype=np.int32)
            if not self._is_valid_cell(goal_cell):
                raise ValueError(f"Goal lies in wall cell: {goal_position}")
            goal = self._cell_center(goal_cell)

        if not self._is_valid_pos(goal):
            raise ValueError(f"Invalid goal position: {goal_position}")

        self._goal_pos = goal.astype(np.float32).copy()

    def valid_action_mask(self, obs):
        cell = np.asarray(np.floor(np.asarray(obs, dtype=np.float32)), dtype=np.int32)
        x, y = int(cell[0]), int(cell[1])

        candidate_next = {
            0: np.array([x, y + 1], dtype=np.int32),  # up
            1: np.array([x, y - 1], dtype=np.int32),  # down
            2: np.array([x - 1, y], dtype=np.int32),  # left
            3: np.array([x + 1, y], dtype=np.int32),  # right
        }

        mask = np.zeros(4, dtype=bool)
        for a, nxt in candidate_next.items():
            mask[a] = self._is_valid_cell(nxt)

        if not mask.any():
            mask[:] = True

        return mask

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        options = options or {}

        if "start_position" in options:
            start = np.asarray(options["start_position"], dtype=np.float32)
        elif "start_pos" in options:
            start = np.asarray(options["start_pos"], dtype=np.float32)
        else:
            start = self.sample_initial_state()

        if np.allclose(start, np.round(start)):
            start_cell = np.asarray(np.round(start), dtype=np.int32)
            if not self._is_valid_cell(start_cell):
                raise ValueError(f"Invalid start position: {start}")
            start = self._cell_center(start_cell)

        if not self._is_valid_pos(start):
            raise ValueError(f"Invalid start position: {start}")

        self.agent_pos = start.astype(np.float32).copy()

        obs = self._obs()
        return obs, {}

    def step(self, action):
        self.t += 1
        delta = self._action_to_delta(action)
        old_pos = self.agent_pos.copy()

        move = delta / float(self.substeps)
        candidate = old_pos.copy()

        for _ in range(self.substeps):
            trial = candidate + move

            trial_x = np.array([trial[0], candidate[1]], dtype=np.float32)
            if self._is_valid_pos(trial_x):
                candidate[0] = trial_x[0]

            trial_y = np.array([candidate[0], trial[1]], dtype=np.float32)
            if self._is_valid_pos(trial_y):
                candidate[1] = trial_y[1]

        self.agent_pos = candidate

        obs = self._obs()
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
            gx, gy = self._pos_to_cell(self._goal_pos)
            y0, y1 = gy * cell, (gy + 1) * cell
            x0, x1 = gx * cell, (gx + 1) * cell
            img[y0+8:y1-8, x0+8:x1-8] = np.array([50, 180, 50], dtype=np.uint8)

        if self.agent_pos is not None:
            ax, ay = self._pos_to_cell(self.agent_pos)
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
        goal_radius=0.5,
        movement_eps=1e-8,
        movement_bonus_coef=0.005,
        movement_bonus_power=1.0,
    ):
        super().__init__(env)
        self.goal_position = np.asarray(goal_position, dtype=np.float32)
        self.goal_reward = float(goal_reward)
        self.step_reward = float(step_reward)
        self.slip_prob = float(np.clip(slip_prob, 0.0, 1.0))
        self.reward_mode = str(reward_mode).lower()
        self.wall_penalty = float(wall_penalty)
        self.gamma = float(gamma)
        self.goal_radius = float(goal_radius)
        self.movement_eps = float(movement_eps)

        self.movement_bonus_coef = float(movement_bonus_coef)
        self.movement_bonus_power = float(movement_bonus_power)

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
            self.env._goal_pos = np.asarray(self.goal_position, dtype=np.float32).copy()

    def _movement_bonus(self, state, next_state):
        move_dist = float(np.linalg.norm(np.asarray(next_state) - np.asarray(state)))
        if move_dist <= self.movement_eps:
            return 0.0
        return self.movement_bonus_coef * (move_dist ** self.movement_bonus_power)

    def _state_id_from_obs(self, obs):
        s = np.asarray(obs, dtype=np.float32).reshape(-1)
        if s.shape[0] != 2:
            raise ValueError(f"Expected obs shape (2,), got {s.shape}")
        cell = np.floor(s).astype(np.int32)
        return (int(cell[0]), int(cell[1]))

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
        self.goal_position = np.asarray(goal_position, dtype=np.float32)
        self._sync_goal_to_env()

    def compute_simple_reward(self, state, action, next_state, goal):
        dist_to_goal = float(np.linalg.norm(np.asarray(next_state) - np.asarray(goal)))
        moved = float(np.linalg.norm(np.asarray(next_state) - np.asarray(state))) > self.movement_eps

        if dist_to_goal <= self.goal_radius:
            base_r = self.goal_reward
        elif not moved:
            base_r = self.wall_penalty
        else:
            base_r = self.step_reward

        return base_r + self._movement_bonus(state, next_state)

    def compute_shaped_reward(self, state, action, next_state, goal):
        gx, gy = goal
        dist_to_goal = float(np.linalg.norm(np.asarray(next_state) - np.asarray(goal)))
        moved = float(np.linalg.norm(np.asarray(next_state) - np.asarray(state))) > self.movement_eps

        if dist_to_goal <= self.goal_radius:
            base_r = self.goal_reward
        elif not moved:
            base_r = self.wall_penalty
        else:
            base_r = self.step_reward

        width = int(self.env.unwrapped.width)
        height = int(self.env.unwrapped.height)
        max_d = (width - 1) + (height - 1)

        if max_d <= 0:
            shaping = 0.0
        else:
            def phi(xy):
                x, y = float(xy[0]), float(xy[1])
                d = abs(gx - x) + abs(gy - y)
                return 1.0 - d / max_d

            phi_s = phi(state)
            phi_sp = phi(next_state)
            shaping = self.gamma * phi_sp - phi_s

        return base_r + shaping + self._movement_bonus(state, next_state)

    def reset(self, *, seed=None, options=None):
        options = options or {}
        if "goal_position" in options:
            self.set_goal_position(options["goal_position"])

        obs, info = self.env.reset(seed=seed, options=options)
        self._sync_goal_to_env()
        self._update_coverage_on_reset(obs)

        info = dict(info)
        info["goal_position"] = self.goal_position.astype(np.float32).tolist()

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
            action = self.env.np_random.uniform(low=-1.0, high=1.0, size=(2,)).astype(np.float32)

        state = np.asarray(self.env.unwrapped.agent_pos, dtype=np.float32).copy()
        obs, _, terminated, truncated, info = self.env.step(action)

        next_state = np.asarray(obs, dtype=np.float32).copy()
        goal = self.goal_position.astype(np.float32).copy()
        dist_to_goal = float(np.linalg.norm(next_state - goal))
        reached = dist_to_goal <= self.goal_radius

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
        info["goal_position"] = self.goal_position.astype(np.float32).tolist()
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