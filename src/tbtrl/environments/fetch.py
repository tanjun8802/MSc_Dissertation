"""Fetch adapters: fixed arm resets, external goals, flat SAC observations."""

import gymnasium as gym
import numpy as np
import torch


class FixedStartFetch(gym.Wrapper):
    """Keep native Fetch rewards and continuing-task/time-limit semantics.

    Native Fetch resets fix the robot pose. PickAndPlace can randomize the object
    using Gymnasium's seeded RNG; unseeded resets advance that RNG. The desired
    goal is passed separately to TBTRL, outside the flat state vector.
    """

    diagnostic_render_mode = "rgb_array"

    def __init__(
        self,
        env_id,
        goal,
        *,
        max_horizon=50,
        initial_state_seed=42,
        randomize_object=False,
        reward_type="dense",
        render_mode=None,
        width=640,
        height=480,
    ):
        try:
            import gymnasium_robotics
        except ImportError as exc:
            raise ImportError(
                'Install Fetch dependencies with pip install -e ".[robotics,notebooks]"'
            ) from exc
        gym.register_envs(gymnasium_robotics)
        if env_id not in ("FetchReach-v4", "FetchPickAndPlace-v4"):
            raise ValueError("Use a supported v4 Fetch environment.")
        if randomize_object and env_id != "FetchPickAndPlace-v4":
            raise ValueError("Object randomization is only supported for PickAndPlace.")
        goal = np.asarray(goal, dtype=np.float32)
        if goal.shape != (3,) or not np.isfinite(goal).all():
            raise ValueError("Fetch goals must be finite (x, y, z) positions in metres.")
        if not isinstance(initial_state_seed, int) or initial_state_seed < 0:
            raise ValueError("initial_state_seed must be a nonnegative integer.")
        if reward_type not in ("dense", "sparse"):
            raise ValueError("Fetch reward_type must be dense or sparse.")
        if not isinstance(max_horizon, int) or max_horizon < 1:
            raise ValueError("max_horizon must be a positive integer.")
        env = gym.make(
            env_id,
            reward_type=reward_type,
            max_episode_steps=max_horizon,
            render_mode=render_mode,
            width=width,
            height=height,
        )
        super().__init__(env)
        self.fixed_goal = goal.copy()
        self.initial_state_seed = initial_state_seed
        self.randomize_object = randomize_object
        self._has_reset = False
        self.reward_type = reward_type
        self.achieved_slice = slice(0, 3) if env_id == "FetchReach-v4" else slice(3, 6)
        space = env.observation_space["observation"]
        self.observation_space = gym.spaces.Box(
            space.low.astype(np.float32), space.high.astype(np.float32), dtype=np.float32
        )

    def reset(self, *, seed=None, options=None):
        # Native reset restores the arm independently of the object RNG. Seed
        # once, then let unseeded episode resets advance the block-position RNG.
        reset_seed = self.initial_state_seed
        if self.randomize_object:
            reset_seed = seed if seed is not None or self._has_reset else self.initial_state_seed
        observation, info = self.env.reset(seed=reset_seed, options=options)
        self._has_reset = True
        self.env.unwrapped.goal = self.fixed_goal.astype(np.float64)
        info = dict(info, initial_state_seed=reset_seed, randomize_object=self.randomize_object)
        return np.asarray(observation["observation"], dtype=np.float32), info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        # The internal goal was fixed before stepping, so native reward/is_success
        # and the rendered target marker all refer to the same requested goal.
        return (
            np.asarray(observation["observation"], dtype=np.float32),
            float(reward),
            terminated,
            truncated,
            dict(info, success=bool(info["is_success"])),
        )

    def goal_distance(self, observation, goal):
        return float(np.linalg.norm(np.asarray(observation)[self.achieved_slice] - goal))

    def relabel_transitions(self, batch, goal):
        achieved = batch.next_obs[:, self.achieved_slice]
        target = torch.as_tensor(goal, dtype=achieved.dtype, device=achieved.device)
        distance = (achieved - target).norm(dim=-1, keepdim=True)
        reward = (
            -distance
            if self.reward_type == "dense"
            else -(distance > self.env.unwrapped.distance_threshold).to(achieved.dtype)
        )
        # Fetch is continuing even at success. Time-limit flags remain in the
        # replay batch; SAC's existing truncation-bootstrap setting handles them.
        return reward, torch.zeros_like(reward)


def reach(goal, **options):
    return FixedStartFetch("FetchReach-v4", goal, **options)


def pick_and_place(goal, **options):
    options.setdefault("randomize_object", True)
    return FixedStartFetch("FetchPickAndPlace-v4", goal, **options)
