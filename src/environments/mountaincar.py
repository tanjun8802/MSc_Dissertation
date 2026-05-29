import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces


class CustomMountainCar(gym.Wrapper): # custom wrapper to keep the sparse reward but add stochastic wind drag
    def __init__(self, env, goal_reward=1.0, step_reward=0.0, goal_velocity=0.5, wind_std=2e-3, drag_coeff=0.01):
        super().__init__(env)
        self.goal_reward = goal_reward
        self.step_reward = step_reward
        self.goal_velocity = goal_velocity
        self.wind_std = wind_std
        self.drag_coeff = drag_coeff

    def step(self, action):
        obs, _, _, truncated, info = self.env.step(action)
        position, velocity = map(float, obs)

        # Apply a small zero-mean stochastic wind disturbance after the base physics step,
        # then damp the resulting velocity to make the task more robust and less deterministic.
        rng = self.unwrapped.np_random
        wind = float(rng.normal(0.0, self.wind_std)) if self.wind_std > 0 else 0.0

        velocity = (1.0 - self.drag_coeff) * velocity + wind
        velocity = float(np.clip(velocity, -self.unwrapped.max_speed, self.unwrapped.max_speed))

        position = float(np.clip(position + velocity,
                                 self.unwrapped.min_position,
                                 self.unwrapped.max_position))
        if position <= self.unwrapped.min_position and velocity < 0:
            velocity = 0.0

        obs = np.array([position, velocity], dtype=np.float32)
        self.unwrapped.state = obs.copy()

        goal_position = self.unwrapped.goal_position
        goal_reached = position >= goal_position and abs(velocity) <= self.goal_velocity
        reward = self.goal_reward if goal_reached else self.step_reward

        info = dict(info)
        info["wind_force"] = wind
        info["drag_coeff"] = self.drag_coeff
        return obs, reward, bool(goal_reached), truncated, info