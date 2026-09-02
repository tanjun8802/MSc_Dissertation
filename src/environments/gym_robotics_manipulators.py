from __future__ import annotations
from collections.abc import Callable
from typing import Any
import gymnasium as gym
import numpy as np
import math
import os
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    BaseCallback,
)

GoalFunction = Callable[
    [Any, Any, dict],
    Any,
]

SetGoalFunction = Callable[
    [gym.Env, np.ndarray],
    None,
]

class FixedGoalManipulatorWrapper(gym.Wrapper):
    """
    Fixed-goal wrapper for Gymnasium-Robotics-style
    goal-conditioned manipulation environments.

    Compatible with environments exposing:

        observation
        achieved_goal
        desired_goal
        compute_reward(...)

    The wrapper:

        - preserves Dict observations;
        - converts floating-point observations to float32;
        - replaces desired_goal with a fixed goal;
        - recomputes rewards consistently;
        - remains compatible with HER;
        - preserves the underlying terminated/truncated flags.

    Important:
        compute_reward() respects the desired_goal argument
        supplied by HER. It must not always replace that
        argument with self.fixed_goal.
    """

    def __init__(
        self,
        env: gym.Env,
        goal: Any,
        reward_fn=None,
        success_fn=None,
    ):
        super().__init__(env)

        self._validate_goal_observation_space()

        self._original_observation_space = (
            self.observation_space
        )

        self.observation_space = (
            self._make_float32_observation_space(
                self._original_observation_space
            )
        )

        desired_goal_space = (
            self.observation_space[
                "desired_goal"
            ]
        )

        self.fixed_goal = np.asarray(
            goal,
            dtype=np.float32,
        ).copy()

        if self.fixed_goal.shape != (
            desired_goal_space.shape
        ):
            raise ValueError(
                "Goal shape mismatch. "
                f"Expected "
                f"{desired_goal_space.shape}, "
                f"got {self.fixed_goal.shape}."
            )

        if not desired_goal_space.contains(
            self.fixed_goal
        ):
            raise ValueError(
                "The selected goal is outside "
                "the desired_goal observation space.\n"
                f"Goal: {self.fixed_goal}\n"
                f"Space: {desired_goal_space}"
            )

        self._reward_fn = reward_fn
        self._success_fn = success_fn

        self._native_compute_reward = (
            self._find_method(
                "compute_reward"
            )
        )

        if (
            self._reward_fn is None
            and self._native_compute_reward is None
        ):
            raise AttributeError(
                "Could not find compute_reward() "
                "in the environment or its wrapper "
                "chain."
            )

    # =========================================================
    # Validation
    # =========================================================

    def _validate_goal_observation_space(
        self,
    ):
        if not isinstance(
            self.observation_space,
            gym.spaces.Dict,
        ):
            raise TypeError(
                "FixedGoalManipulatorWrapper requires "
                "a Dict observation space."
            )

        required_keys = {
            "observation",
            "achieved_goal",
            "desired_goal",
        }

        available_keys = set(
            self.observation_space.spaces.keys()
        )

        missing_keys = (
            required_keys - available_keys
        )

        if missing_keys:
            raise ValueError(
                "The environment is missing required "
                f"goal keys: {missing_keys}"
            )

    def _make_float32_observation_space(
        self,
        observation_space: gym.spaces.Dict,
    ) -> gym.spaces.Dict:
        spaces = {}

        for key, space in (
            observation_space.spaces.items()
        ):
            if isinstance(
                space,
                gym.spaces.Box,
            ):
                spaces[key] = gym.spaces.Box(
                    low=np.asarray(
                        space.low,
                        dtype=np.float32,
                    ),
                    high=np.asarray(
                        space.high,
                        dtype=np.float32,
                    ),
                    shape=space.shape,
                    dtype=np.float32,
                )
            else:
                spaces[key] = space

        return gym.spaces.Dict(spaces)

    # =========================================================
    # Wrapper lookup
    # =========================================================

    def _find_method(
        self,
        method_name: str,
    ):
        """
        Finds a method through nested Gymnasium wrappers.
        """

        if hasattr(
            self.env,
            "get_wrapper_attr",
        ):
            try:
                method = (
                    self.env.get_wrapper_attr(
                        method_name
                    )
                )

                if callable(method):
                    return method

            except AttributeError:
                pass

        current = self.env

        while current is not None:
            method = getattr(
                current,
                method_name,
                None,
            )

            if callable(method):
                return method

            if not hasattr(
                current,
                "env",
            ):
                break

            current = current.env

        return None

    # =========================================================
    # Observation handling
    # =========================================================

    def _rewrite_observation(
        self,
        obs: dict,
    ) -> dict:
        if not isinstance(obs, dict):
            raise TypeError(
                "Expected a dictionary observation, "
                f"got {type(obs)}."
            )

        rewritten = {}

        for key, value in obs.items():
            if isinstance(
                value,
                np.ndarray,
            ):
                rewritten[key] = np.asarray(
                    value,
                    dtype=np.float32,
                )
            else:
                rewritten[key] = value

        rewritten["desired_goal"] = (
            self.fixed_goal.astype(
                np.float32,
                copy=True,
            )
        )

        return rewritten

    # =========================================================
    # HER-compatible reward API
    # =========================================================

    def compute_reward(
        self,
        achieved_goal,
        desired_goal,
        info,
    ):
        """
        Computes the goal-conditioned reward.

        HER may provide relabelled desired_goal values,
        so desired_goal must be passed through unchanged.
        """

        if self._reward_fn is not None:
            reward = self._reward_fn(
                achieved_goal,
                desired_goal,
                info,
            )
        else:
            reward = self._native_compute_reward(
                achieved_goal,
                desired_goal,
                info,
            )

        reward_array = np.asarray(
            reward
        )

        if reward_array.ndim == 0:
            return float(
                reward_array.item()
            )

        return reward_array.astype(
            np.float32
        )

    # =========================================================
    # Success handling
    # =========================================================

    def compute_success(
        self,
        achieved_goal,
        desired_goal,
        info,
    ):
        if self._success_fn is not None:
            return self._success_fn(
                achieved_goal,
                desired_goal,
                info,
            )

        if "is_success" in info:
            return info["is_success"]

        return 0.0

    def _as_scalar(
        self,
        value,
        name: str,
    ) -> float:
        array = np.asarray(
            value
        )

        if array.size != 1:
            raise ValueError(
                f"{name} must be scalar during "
                f"normal interaction, got "
                f"shape {array.shape}."
            )

        return float(
            array.reshape(-1)[0]
        )

    def _prepare_info(
        self,
        obs: dict,
        info: dict,
    ) -> dict:
        info = dict(info)

        info["achieved_goal"] = (
            np.asarray(
                obs["achieved_goal"],
                dtype=np.float32,
            ).copy()
        )

        info["desired_goal"] = (
            np.asarray(
                obs["desired_goal"],
                dtype=np.float32,
            ).copy()
        )

        # Preserve the base environment's is_success
        # if it supplies one. If it does not, this remains
        # 0 unless a success_fn was provided.
        success = self.compute_success(
            obs["achieved_goal"],
            obs["desired_goal"],
            info,
        )

        info["is_success"] = self._as_scalar(
            success,
            "is_success",
        )

        return info

    # =========================================================
    # Gymnasium API
    # =========================================================

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ):
        obs, info = self.env.reset(
            seed=seed,
            options=options,
        )

        obs = self._rewrite_observation(
            obs
        )

        info = self._prepare_info(
            obs,
            info,
        )

        return obs, info

    def step(self, action):
        """
        Recomputes reward using the fixed-goal observation.

        Crucially, the terminated and truncated flags from
        the underlying TimeLimit/environment are preserved
        exactly.
        """

        (
            obs,
            _native_reward,
            base_terminated,
            base_truncated,
            info,
        ) = self.env.step(action)

        obs = self._rewrite_observation(
            obs
        )

        info = self._prepare_info(
            obs,
            info,
        )

        reward = self.compute_reward(
            obs["achieved_goal"],
            obs["desired_goal"],
            info,
        )

        reward_array = np.asarray(
            reward
        )

        if reward_array.ndim == 0:
            reward = float(
                reward_array.item()
            )

        return (
            obs,
            reward,
            bool(base_terminated),
            bool(base_truncated),
            info,
        )

    def close(self):
        return self.env.close()


class SuccessRateEvalCallback(
    BaseCallback
):
    """
    Periodically evaluates a goal-conditioned policy.

    This callback is designed for a normal Gymnasium
    evaluation environment, optionally wrapped with Monitor.

    It records:

        - final success rate
        - ever-success rate
        - mean return
        - minimum goal distance
        - final goal distance
        - qualified success rate

    The evaluation loop has a hard maximum number of steps
    so that a missing terminated/truncated signal cannot
    hang training indefinitely.
    """

    def __init__(
        self,
        eval_env,
        eval_freq: int,
        n_eval_episodes: int = 10,
        success_threshold: float = 0.95,
        return_threshold: float | None = None,
        required_checkpoints: int = 3,
        log_path: str | None = None,
        deterministic: bool = True,
        max_episode_steps: int | None = None,
        use_ever_success: bool = True,
        verbose: int = 1,
    ):
        super().__init__(
            verbose=verbose
        )

        self.eval_env = eval_env

        self.eval_freq = int(
            eval_freq
        )

        self.n_eval_episodes = int(
            n_eval_episodes
        )

        self.success_threshold = float(
            success_threshold
        )

        self.return_threshold = (
            None
            if return_threshold is None
            else float(return_threshold)
        )

        self.required_checkpoints = int(
            required_checkpoints
        )

        self.log_path = log_path

        self.deterministic = bool(
            deterministic
        )

        self.use_ever_success = bool(
            use_ever_success
        )

        if max_episode_steps is None:
            self.max_episode_steps = (
                self._infer_max_episode_steps(
                    eval_env
                )
            )
        else:
            self.max_episode_steps = int(
                max_episode_steps
            )

        if self.eval_freq <= 0:
            raise ValueError(
                "eval_freq must be positive."
            )

        if self.n_eval_episodes <= 0:
            raise ValueError(
                "n_eval_episodes must be positive."
            )

        if not 0.0 <= (
            self.success_threshold
        ) <= 1.0:
            raise ValueError(
                "success_threshold must be in [0, 1]."
            )

        if self.required_checkpoints <= 0:
            raise ValueError(
                "required_checkpoints must be positive."
            )

        if self.max_episode_steps <= 0:
            raise ValueError(
                "max_episode_steps must be positive."
            )

        self.eval_steps = []
        self.success_rates = []
        self.ever_success_rates = []
        self.eval_returns = []
        self.mean_min_distances = []
        self.mean_final_distances = []
        self.qualified_success_rates = []

        self.consecutive_successful_evals = 0
        self.steps_to_target = None

        if self.log_path is not None:
            os.makedirs(
                self.log_path,
                exist_ok=True,
            )

    # ========================================================
    # Environment inspection
    # ========================================================

    @staticmethod
    def _infer_max_episode_steps(
        env,
    ) -> int:
        """
        Finds the episode horizon through nested wrappers.
        """

        current = env

        while current is not None:
            if hasattr(
                current,
                "_max_episode_steps",
            ):
                return int(
                    current._max_episode_steps
                )

            if hasattr(
                current,
                "max_episode_steps",
            ):
                return int(
                    current.max_episode_steps
                )

            if hasattr(
                current,
                "spec",
            ):
                spec = current.spec

                if spec is not None:
                    max_steps = getattr(
                        spec,
                        "max_episode_steps",
                        None,
                    )

                    if max_steps is not None:
                        return int(
                            max_steps
                        )

            if not hasattr(
                current,
                "env",
            ):
                break

            current = current.env

        # Safe fallback. It is better to raise later with
        # a useful error than to permit an infinite loop.
        return 1000

    # ========================================================
    # Utilities
    # ========================================================

    @staticmethod
    def _as_scalar(
        value,
        default: float = 0.0,
    ) -> float:
        if value is None:
            return float(default)

        array = np.asarray(
            value
        )

        if array.size == 0:
            return float(default)

        return float(
            array.reshape(-1)[0]
        )

    @staticmethod
    def _as_distance(
        info: dict,
        default: float = np.inf,
    ) -> float:
        value = info.get(
            "goal_dist",
            default,
        )

        try:
            return float(value)
        except (
            TypeError,
            ValueError,
        ):
            return float(default)

    # ========================================================
    # Evaluation
    # ========================================================

    def _evaluate_success_rate(
        self,
    ):
        final_successes = []
        ever_successes = []
        episode_returns = []
        minimum_distances = []
        final_distances = []
        qualified_successes = []

        print(
            f"[Eval] starting at "
            f"timestep={self.num_timesteps:,} | "
            f"episodes={self.n_eval_episodes} | "
            f"max_steps={self.max_episode_steps}",
            flush=True,
        )

        for episode_idx in range(
            self.n_eval_episodes
        ):
            print(
                f"[Eval] episode "
                f"{episode_idx + 1}/"
                f"{self.n_eval_episodes}",
                flush=True,
            )

            obs, info = self.eval_env.reset(
                seed=(
                    self.n_calls
                    + episode_idx
                )
            )

            terminated = False
            truncated = False

            episode_return = 0.0
            episode_steps = 0

            initial_success = (
                self._as_scalar(
                    info.get(
                        "is_success",
                        0.0,
                    )
                )
                >= 1.0
            )

            ever_success = (
                initial_success
            )

            minimum_distance = (
                self._as_distance(info)
            )

            final_distance = (
                minimum_distance
            )

            while not (
                terminated or truncated
            ):
                if (
                    episode_steps
                    >= self.max_episode_steps
                ):
                    raise RuntimeError(
                        "Evaluation episode exceeded "
                        f"{self.max_episode_steps} "
                        "steps without terminated=True "
                        "or truncated=True.\n"
                        "Check the wrapper's propagation "
                        "of termination and truncation."
                    )

                action, _ = (
                    self.model.predict(
                        obs,
                        deterministic=(
                            self.deterministic
                        ),
                    )
                )

                (
                    obs,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = self.eval_env.step(
                    action
                )

                episode_return += float(
                    reward
                )

                episode_steps += 1

                current_success = (
                    self._as_scalar(
                        info.get(
                            "is_success",
                            0.0,
                        )
                    )
                    >= 1.0
                )

                ever_success = (
                    ever_success
                    or current_success
                )

                current_distance = (
                    self._as_distance(info)
                )

                minimum_distance = min(
                    minimum_distance,
                    current_distance,
                )

                final_distance = (
                    current_distance
                )

            final_success = (
                self._as_scalar(
                    info.get(
                        "is_success",
                        0.0,
                    )
                )
                >= 1.0
            )

            success_for_evaluation = (
                ever_success
                if self.use_ever_success
                else final_success
            )

            if self.return_threshold is None:
                qualified_success = (
                    success_for_evaluation
                )
            else:
                qualified_success = (
                    success_for_evaluation
                    and episode_return
                    >= self.return_threshold
                )

            final_successes.append(
                float(final_success)
            )

            ever_successes.append(
                float(ever_success)
            )

            episode_returns.append(
                episode_return
            )

            minimum_distances.append(
                minimum_distance
            )

            final_distances.append(
                final_distance
            )

            qualified_successes.append(
                float(qualified_success)
            )

            print(
                f"[Eval] episode "
                f"{episode_idx + 1}/"
                f"{self.n_eval_episodes} complete | "
                f"steps={episode_steps} | "
                f"return={episode_return:.3f} | "
                f"final_success="
                f"{int(final_success)} | "
                f"ever_success="
                f"{int(ever_success)}",
                flush=True,
            )

        metrics = {
            "success_rate": float(
                np.mean(final_successes)
            ),
            "ever_success_rate": float(
                np.mean(ever_successes)
            ),
            "mean_return": float(
                np.mean(episode_returns)
            ),
            "mean_min_distance": float(
                np.mean(minimum_distances)
            ),
            "mean_final_distance": float(
                np.mean(final_distances)
            ),
            "qualified_success_rate": float(
                np.mean(qualified_successes)
            ),
        }

        print(
            "[Eval] completed | "
            f"success={metrics['success_rate']:.3f} | "
            f"ever_success="
            f"{metrics['ever_success_rate']:.3f} | "
            f"return="
            f"{metrics['mean_return']:.3f}",
            flush=True,
        )

        return metrics

    # ========================================================
    # Saving
    # ========================================================

    def _save_results(self):
        if self.log_path is None:
            return

        save_path = os.path.join(
            self.log_path,
            "success_evaluations.npz",
        )

        np.savez(
            save_path,
            timesteps=np.asarray(
                self.eval_steps,
                dtype=np.int64,
            ),
            success_rates=np.asarray(
                self.success_rates,
                dtype=np.float32,
            ),
            ever_success_rates=np.asarray(
                self.ever_success_rates,
                dtype=np.float32,
            ),
            returns=np.asarray(
                self.eval_returns,
                dtype=np.float32,
            ),
            mean_min_distances=np.asarray(
                self.mean_min_distances,
                dtype=np.float32,
            ),
            mean_final_distances=np.asarray(
                self.mean_final_distances,
                dtype=np.float32,
            ),
            qualified_success_rates=np.asarray(
                self.qualified_success_rates,
                dtype=np.float32,
            ),
        )

    # ========================================================
    # SB3 callback interface
    # ========================================================

    def _on_step(self) -> bool:
        if (
            self.n_calls
            % self.eval_freq
            != 0
        ):
            return True

        metrics = (
            self._evaluate_success_rate()
        )

        self.eval_steps.append(
            self.num_timesteps
        )

        self.success_rates.append(
            metrics["success_rate"]
        )

        self.ever_success_rates.append(
            metrics["ever_success_rate"]
        )

        self.eval_returns.append(
            metrics["mean_return"]
        )

        self.mean_min_distances.append(
            metrics["mean_min_distance"]
        )

        self.mean_final_distances.append(
            metrics["mean_final_distance"]
        )

        self.qualified_success_rates.append(
            metrics[
                "qualified_success_rate"
            ]
        )

        required_successes = math.ceil(
            self.success_threshold
            * self.n_eval_episodes
        )

        success_condition = (
            metrics["ever_success_rate"]
            >= self.success_threshold
        )

        qualified_condition = (
            metrics[
                "qualified_success_rate"
            ]
            >= self.success_threshold
        )

        if self.return_threshold is None:
            return_condition = True
        else:
            return_condition = (
                metrics["mean_return"]
                >= self.return_threshold
            )

        passed = (
            success_condition
            and qualified_condition
            and return_condition
        )

        if passed:
            self.consecutive_successful_evals += 1
        else:
            self.consecutive_successful_evals = 0

        return_threshold_text = (
            "disabled"
            if self.return_threshold is None
            else f"{self.return_threshold:.3f}"
        )

        print(
            f"Step {self.num_timesteps:,} | "
            f"Final success: "
            f"{metrics['success_rate']:.3f} | "
            f"Ever success: "
            f"{metrics['ever_success_rate']:.3f} | "
            f"Qualified: "
            f"{metrics['qualified_success_rate']:.3f} | "
            f"Return: "
            f"{metrics['mean_return']:.3f} | "
            f"Min distance: "
            f"{metrics['mean_min_distance']:.4f} | "
            f"Final distance: "
            f"{metrics['mean_final_distance']:.4f} | "
            f"Return threshold: "
            f"{return_threshold_text} | "
            f"Required successes: "
            f"{required_successes}/"
            f"{self.n_eval_episodes} | "
            f"Consecutive: "
            f"{self.consecutive_successful_evals}/"
            f"{self.required_checkpoints}",
            flush=True,
        )

        self._save_results()

        if (
            self.consecutive_successful_evals
            >= self.required_checkpoints
        ):
            self.steps_to_target = (
                self.num_timesteps
            )

            print(
                "Target performance reached. "
                "Stopping training.",
                flush=True,
            )

            return False

        return True


# =====================
# Env factory
# =====================
class Float32Wrapper(gym.ObservationWrapper):
    def observation(self, obs):
        # obs is a dict for Fetch environments
        if isinstance(obs, dict):
            return {
                k: (v.astype(np.float32) if isinstance(v, np.ndarray) else v)
                for k, v in obs.items()
            }
        if isinstance(obs, np.ndarray):
            return obs.astype(np.float32)
        return obs


def make_env(
    env_id: str,
    render_mode: str | None = None,
    goal: np.ndarray | None = None,
    max_episode_steps: int = 50,
    seed: int | None = None,
):

    base_env = gym.make(
        env_id,
        reward_type="dense",
        max_episode_steps=max_episode_steps,
        render_mode=render_mode,
    )

    if goal is not None:
        fixed_goal = np.asarray(goal, dtype=np.float64).copy()

        class FixedGoalWrapper(gym.Wrapper):
            def __init__(self, env, fixed_goal):
                super().__init__(env)
                self.fixed_goal = fixed_goal

            def reset(self, *, seed=None, options=None):
                # The original reset randomizes the block/start state.
                obs, info = self.env.reset(seed=seed, options=options)

                # Change the environment's actual internal goal—not merely
                # the returned observation. Future step rewards and
                # info["is_success"] now use this fixed goal.
                self.env.unwrapped.goal = self.fixed_goal.copy()

                # reset() already generated obs using the random original goal,
                # so correct the returned initial observation as well.
                obs = dict(obs)
                obs["desired_goal"] = self.fixed_goal.astype(
                    np.float32
                ).copy()

                return obs, info

        base_env = FixedGoalWrapper(
            base_env,
            fixed_goal=fixed_goal,
        )

    monitored_env = Monitor(base_env)
    monitored_env = Float32Wrapper(monitored_env)

    # Optional: seed the action/observation spaces once, without making
    # every episode identical.
    if seed is not None:
        monitored_env.action_space.seed(seed)
        monitored_env.observation_space.seed(seed)

    return monitored_env
