import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces


class FourRoomsGridWorld(gym.Env):

    metadata = {"render_modes": ["rgb_array"], "render_fps": 8}

    def __init__(self, room_size=11, render_mode=None, max_episode_steps=500):
        super().__init__()
        self.room_size = int(room_size)
        self.grid_size = (self.room_size * 2) + 1
        self.render_mode = render_mode
        self.max_episode_steps = int(max_episode_steps)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0], dtype=np.float32),
            high=np.array([self.grid_size - 1.0, self.grid_size - 1.0], dtype=np.float32),
            shape=(2,),
            dtype=np.float32,
        )

        mid_point = (self.room_size - 1) // 2
        self._door_coords = {
            (self.room_size, mid_point),
            (self.room_size, self.room_size + mid_point + 1),
            (mid_point, self.room_size),
            (self.room_size + mid_point + 1, self.room_size),
        }

        self._blocked_cells = self._build_walls()
        self._outer_wall_cells = self._build_outer_walls()
        self._free_cells = np.array(
            [(x, y) for y in range(self.grid_size) for x in range(self.grid_size) if (x, y) not in self._blocked_cells],
            dtype=np.int32,
        )
        self._agent_pos = np.array([0, 0], dtype=np.int32)
        self._goal_pos = None
        self._step_count = 0

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

    def _is_valid_cell(self, cell):
        x, y = int(cell[0]), int(cell[1])
        if x < 0 or y < 0 or x >= self.grid_size or y >= self.grid_size:
            return False
        return (x, y) not in self._blocked_cells

    def sample_initial_state(self):
        idx = int(self.np_random.integers(0, len(self._free_cells)))
        return self._free_cells[idx].copy()

    def set_goal_position(self, goal_position):
        goal = np.asarray(goal_position, dtype=np.int32)
        if goal.shape != (2,) or not self._is_valid_cell(goal):
            raise ValueError(f"Invalid goal position: {goal_position}")
        self._goal_pos = goal

    def _obs(self):
        return self._agent_pos.astype(np.float32)

    def _action_to_delta(self, action):
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        if a.shape[0] != 2:
            raise ValueError(f"Expected action shape (2,), got {a.shape}")
        if abs(a[0]) >= abs(a[1]):
            dx = int(np.sign(a[0]))
            dy = 0
        else:
            dx = 0
            dy = int(np.sign(a[1]))
        return np.array([dx, dy], dtype=np.int32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        start = options.get("start_position")
        if start is None:
            start = self.sample_initial_state()
        else:
            start = np.asarray(start, dtype=np.int32)
        if not self._is_valid_cell(start):
            raise ValueError(f"Invalid start position: {tuple(start.tolist())}")
        self._agent_pos = start.copy()
        self._step_count = 0
        return self._obs(), {}

    def step(self, action):
        delta = self._action_to_delta(action)
        candidate = self._agent_pos + delta
        if self._is_valid_cell(candidate):
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
            gx, gy = self._goal_pos
            canvas[gy + 1, gx + 1] = np.array([50, 180, 50], dtype=np.uint8)

        ax, ay = self._agent_pos
        canvas[ay + 1, ax + 1] = np.array([220, 40, 40], dtype=np.uint8)
        return np.kron(canvas, np.ones((scale, scale, 1), dtype=np.uint8))

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        return self._render_rgb()


class FourRoomsGoalWrapper(gym.Wrapper):
    """Sparse-reward goal wrapper with configurable goal and slight stochasticity."""

    def __init__(self, env, goal_position=(20, 20), goal_reward=1.0, step_reward=0.0, slip_prob=0.10):
        super().__init__(env)
        self.goal_reward = float(goal_reward)
        self.step_reward = float(step_reward)
        self.slip_prob = float(np.clip(slip_prob, 0.0, 1.0))
        self.goal_position = self._validate_goal(goal_position)
        self._sync_goal_to_env()

    def _validate_goal(self, goal_position):
        goal = np.asarray(goal_position, dtype=np.int32)
        if goal.shape != (2,) or not self.env._is_valid_cell(goal):
            raise ValueError(f"Invalid goal position: {goal_position}")
        return goal

    def _sync_goal_to_env(self):
        if hasattr(self.env, "set_goal_position"):
            self.env.set_goal_position(self.goal_position)
        else:
            self.env._goal_pos = self.goal_position.copy()

    def set_goal_position(self, goal_position):
        self.goal_position = self._validate_goal(goal_position)
        self._sync_goal_to_env()

    def _sample_slip_action(self):
        directions = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=np.float32)
        idx = int(self.env.np_random.integers(0, len(directions)))
        return directions[idx]

    def reset(self, *, seed=None, options=None):
        options = options or {}
        if "goal_position" in options:
            self.set_goal_position(options["goal_position"])
        obs, info = self.env.reset(seed=seed, options=options)
        self._sync_goal_to_env()
        info = dict(info)
        info["goal_position"] = self.goal_position.astype(np.int32).tolist()
        return obs, info

    def step(self, action):
        slipped = bool(self.env.np_random.random() < self.slip_prob)
        effective_action = self._sample_slip_action() if slipped else action
        obs, _, _, truncated, info = self.env.step(effective_action)

        at_goal = bool(np.array_equal(obs.astype(np.int32), self.goal_position))
        reward = self.goal_reward if at_goal else self.step_reward

        info = dict(info)
        info["slipped"] = slipped
        info["goal_position"] = self.goal_position.astype(np.int32).tolist()
        return obs, reward, at_goal, truncated, info


