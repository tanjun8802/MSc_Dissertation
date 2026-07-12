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
    def __init__(self, env, goal_position=(1, 1), goal_reward=1.0, step_reward=0.0, slip_prob=0.0):
        super().__init__(env)
        self.goal_position = np.asarray(goal_position, dtype=np.int32)
        self.goal_reward = float(goal_reward)
        self.step_reward = float(step_reward)
        self.slip_prob = float(np.clip(slip_prob, 0.0, 1.0))
        self._sync_goal_to_env()

    def _sync_goal_to_env(self):
        if hasattr(self.env, "set_goal_position"):
            self.env.set_goal_position(self.goal_position)
        else:
            self.env._goal_pos = np.asarray(self.goal_position, dtype=np.int32).copy()

    def set_goal_position(self, goal_position):
        self.goal_position = np.asarray(goal_position, dtype=np.int32)
        self._sync_goal_to_env()

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
        if self.slip_prob > 0.0 and self.env.np_random.random() < self.slip_prob:
            action = int(self.env.np_random.integers(0, self.env.action_space.n))

        obs, _, terminated, truncated, info = self.env.step(action)

        reward = self.step_reward
        reached = np.array_equal(obs.astype(np.int32), self.goal_position)
        if reached:
            reward = self.goal_reward
            terminated = True

        info = dict(info)
        info["goal_position"] = self.goal_position.astype(np.int32).tolist()
        info["success"] = reached
        return obs, reward, terminated, truncated, info