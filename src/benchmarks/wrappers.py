from __future__ import annotations

import gymnasium as gym


class GoalInfoWrapper(gym.Wrapper):
    """
    Optional thin wrapper that ensures a `goal` entry exists in reset/step info.

    Use this only if your training pipeline expects a goal in the info dict.
    """

    def __init__(self, env, goal_fn=None):
        super().__init__(env)
        self.goal_fn = goal_fn

    def _inject_goal(self, obs, info):
        info = dict(info)
        if "goal" not in info and self.goal_fn is not None:
            info["goal"] = self.goal_fn(obs, info)
        return info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info = self._inject_goal(obs, info)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = self._inject_goal(obs, info)
        return obs, reward, terminated, truncated, info