import numpy as np
import gymnasium as gym
from gymnasium import spaces


class FourRoomsGridWorld(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 8}

    def __init__(
        self,
        room_size=11,
        render_mode=None,
        max_episode_steps=500,
        step_scale=0.35,         # max displacement per step in each axis
        substeps=5,              # for safer collision handling
        start_noise=0.15,        # continuous noise around cell center
        goal_noise=0.0,          # optional noise if you later randomise goal around center
    ):
        super().__init__()
        self.room_size = int(room_size)
        self.grid_size = (self.room_size * 2) + 1
        self.render_mode = render_mode
        self.max_episode_steps = int(max_episode_steps)

        self.step_scale = float(step_scale)
        self.substeps = int(substeps)
        self.start_noise = float(start_noise)
        self.goal_noise = float(goal_noise)

        # Continuous action and observation spaces
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0], dtype=np.float32),
            high=np.array([self.grid_size - 1e-6, self.grid_size - 1e-6], dtype=np.float32),
            shape=(2,),
            dtype=np.float32,
        )

        mid_point = (self.room_size - 1) // 2
        self._door_coords = {
            (self.room_size, mid_point),
            (self.room_size, mid_point - 1),
            (self.room_size, mid_point + 1),
            (self.room_size, self.room_size + mid_point + 1),
            (self.room_size, self.room_size + mid_point),
            (self.room_size, self.room_size + mid_point + 2),
            (mid_point, self.room_size),
            (mid_point - 1, self.room_size),
            (mid_point + 1, self.room_size),
            (self.room_size + mid_point + 1, self.room_size),
            (self.room_size + mid_point + 2, self.room_size),
            (self.room_size + mid_point, self.room_size),
        }

        self._blocked_cells = self._build_walls()
        self._outer_wall_cells = self._build_outer_walls()
        self._free_cells = np.array(
            [(x, y) for y in range(self.grid_size) for x in range(self.grid_size)
             if (x, y) not in self._blocked_cells],
            dtype=np.int32,
        )

        # Continuous agent / goal positions
        self._agent_pos = np.array([0.5, 0.5], dtype=np.float32)
        self._goal_pos = None
        self._step_count = 0
    
    def _blocked_cells(self):
        return self._blocked_cells

    def _build_walls(self):
        wall = self.room_size
        blocked = set()
        for y in range(self.grid_size):
            if (wall, y) not in self._door_coords:
                blocked.add((wall, y))
        for x in range(self.grid_size):
            if (x, wall) not in self._door_coords:
                blocked.add((x, wall))
        return blocked

    def _build_outer_walls(self):
        outer = set()
        boundary_min = -1
        boundary_max = self.grid_size
        for y in range(boundary_min, boundary_max + 1):
            outer.add((boundary_min, y))
            outer.add((boundary_max, y))
        for x in range(boundary_min, boundary_max + 1):
            outer.add((x, boundary_min))
            outer.add((x, boundary_max))
        return outer

    def _pos_to_cell(self, pos):
        x, y = float(pos[0]), float(pos[1])
        return int(np.floor(x)), int(np.floor(y))

    def _cell_center(self, cell):
        cell = np.asarray(cell, dtype=np.float32)
        return cell + 0.5

    def _is_valid_cell(self, cell):
        x, y = int(cell[0]), int(cell[1])
        if x < 0 or y < 0 or x >= self.grid_size or y >= self.grid_size:
            return False
        return (x, y) not in self._blocked_cells

    def _is_valid_pos(self, pos):
        x, y = float(pos[0]), float(pos[1])

        # Stay inside outer bounds
        if x < 0.0 or y < 0.0 or x >= self.grid_size or y >= self.grid_size:
            return False

        cell = self._pos_to_cell(pos)
        return self._is_valid_cell(cell)

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

        # Allow integer cell coordinates or continuous coordinates
        # If integer-like, map to cell center
        if np.allclose(goal, np.round(goal)):
            goal_cell = np.asarray(np.round(goal), dtype=np.int32)
            if not self._is_valid_cell(goal_cell):
                raise ValueError(f"Invalid goal cell: {goal_position}")
            goal = self._cell_center(goal_cell)

        if not self._is_valid_pos(goal):
            raise ValueError(f"Invalid goal position: {goal_position}")

        self._goal_pos = goal.astype(np.float32)

    def _obs(self):
        return self._agent_pos.astype(np.float32)

    def _action_to_delta(self, action):
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        if a.shape[0] != 2:
            raise ValueError(f"Expected action shape (2,), got {a.shape}")
        a = np.clip(a, -1.0, 1.0)
        return self.step_scale * a

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}

        start = options.get("start_position")
        if start is None:
            start = self.sample_initial_state()
        else:
            start = np.asarray(start, dtype=np.float32)

            # If user passes integer cell coords, move to cell center
            if np.allclose(start, np.round(start)):
                start_cell = np.asarray(np.round(start), dtype=np.int32)
                if not self._is_valid_cell(start_cell):
                    raise ValueError(f"Invalid start position: {tuple(start.tolist())}")
                start = self._cell_center(start_cell)

        if not self._is_valid_pos(start):
            raise ValueError(f"Invalid start position: {start}")

        self._agent_pos = start.astype(np.float32).copy()
        self._step_count = 0
        return self._obs(), {}

    def step(self, action):
        delta = self._action_to_delta(action)
        old_pos = self._agent_pos.copy()

        # Substep integration for less wall tunneling
        move = delta / float(self.substeps)
        candidate = old_pos.copy()

        for _ in range(self.substeps):
            trial = candidate + move

            # Axis-wise collision handling: try x then y
            trial_x = np.array([trial[0], candidate[1]], dtype=np.float32)
            if self._is_valid_pos(trial_x):
                candidate[0] = trial_x[0]

            trial_y = np.array([candidate[0], trial[1]], dtype=np.float32)
            if self._is_valid_pos(trial_y):
                candidate[1] = trial_y[1]

        self._agent_pos = candidate
        self._step_count += 1

        truncated = self._step_count >= self.max_episode_steps
        return self._obs(), 0.0, False, truncated, {}

    def _render_rgb(self):
        scale = 20
        canvas = np.full((self.grid_size + 2, self.grid_size + 2, 3), 255, dtype=np.uint8)

        for (x, y) in self._outer_wall_cells:
            canvas[y + 1, x + 1] = np.array([30, 30, 30], dtype=np.uint8)

        for (x, y) in self._blocked_cells:
            canvas[y + 1, x + 1] = np.array([30, 30, 30], dtype=np.uint8)

        if self._goal_pos is not None:
            gx, gy = self._pos_to_cell(self._goal_pos)
            canvas[gy + 1, gx + 1] = np.array([50, 180, 50], dtype=np.uint8)

        ax, ay = self._pos_to_cell(self._agent_pos)
        canvas[ay + 1, ax + 1] = np.array([220, 40, 40], dtype=np.uint8)

        return np.kron(canvas, np.ones((scale, scale, 1), dtype=np.uint8))

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        return self._render_rgb()


class FourRoomsGoalWrapper(gym.Wrapper):
    """Sparse-reward goal wrapper with configurable goal and slight stochasticity."""

    def __init__(
        self,
        env,
        goal_position=(20, 20),
        goal_reward=1.0,
        step_reward=0.0,
        slip_prob=0.10,
        goal_radius=0.5,     # continuous success radius
    ):
        super().__init__(env)
        self.goal_reward = float(goal_reward)
        self.step_reward = float(step_reward)
        self.slip_prob = float(np.clip(slip_prob, 0.0, 1.0))
        self.goal_radius = float(goal_radius)
        self.goal_position = self._validate_goal(goal_position)
        self._sync_goal_to_env()

    def _validate_goal(self, goal_position):
        goal = np.asarray(goal_position, dtype=np.float32)
        if goal.shape != (2,):
            raise ValueError(f"Invalid goal position: {goal_position}")

        # Integer cell -> center of that cell
        if np.allclose(goal, np.round(goal)):
            goal_cell = np.asarray(np.round(goal), dtype=np.int32)
            if not self.env._is_valid_cell(goal_cell):
                raise ValueError(f"Invalid goal position: {goal_position}")
            goal = self.env._cell_center(goal_cell)

        if not self.env._is_valid_pos(goal):
            raise ValueError(f"Invalid goal position: {goal_position}")

        return goal.astype(np.float32)

    def _sync_goal_to_env(self):
        if hasattr(self.env, "set_goal_position"):
            self.env.set_goal_position(self.goal_position)
        else:
            self.env._goal_pos = self.goal_position.copy()

    def set_goal_position(self, goal_position):
        self.goal_position = self._validate_goal(goal_position)
        self._sync_goal_to_env()

    def _sample_slip_action(self):
        # Continuous random slip action
        a = self.env.np_random.uniform(low=-1.0, high=1.0, size=(2,)).astype(np.float32)
        norm = np.linalg.norm(a)
        if norm > 1e-8:
            a = a / max(1.0, norm)
        return a

    def reset(self, *, seed=None, options=None):
        options = options or {}
        if "goal_position" in options:
            self.set_goal_position(options["goal_position"])
        obs, info = self.env.reset(seed=seed, options=options)
        self._sync_goal_to_env()
        info = dict(info)
        info["goal_position"] = self.goal_position.astype(np.float32).tolist()
        return obs, info

    def step(self, action):
        slipped = bool(self.env.np_random.random() < self.slip_prob)
        effective_action = self._sample_slip_action() if slipped else action
        obs, _, _, truncated, info = self.env.step(effective_action)

        dist_to_goal = float(np.linalg.norm(obs - self.goal_position))
        at_goal = dist_to_goal <= self.goal_radius
        reward = self.goal_reward if at_goal else self.step_reward

        info = dict(info)
        info["slipped"] = slipped
        info["goal_position"] = self.goal_position.astype(np.float32).tolist()
        info["distance_to_goal"] = dist_to_goal
        return obs, reward, at_goal, truncated, info