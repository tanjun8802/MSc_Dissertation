from gymnasium import spaces
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.td3.policies import MultiInputPolicy as TD3MultiInputPolicy
from typing import Optional
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ContinuousCritic
import torch
import torch.nn as nn

# ============================================================
# Dot-product factorised Q-network
# ============================================================

class DotProductQNetwork(nn.Module):
    """
    Q(s, a, g) = phi(s, a)^T psi(g)
    """

    def __init__(
        self,
        robot_dim,
        action_dim,
        goal_dim,
        embedding_dim=256,
        hidden_dim=256,
        activation_fn=nn.ReLU,
        normalize_embeddings=True,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.normalize_embeddings = (
            normalize_embeddings
        )

        # phi(s, a)
        self.state_action_encoder = nn.Sequential(
            nn.Linear(
                robot_dim + action_dim,
                hidden_dim,
            ),
            activation_fn(),
            nn.Linear(
                hidden_dim,
                embedding_dim,
            ),
        )

        # psi(g)
        self.goal_encoder = nn.Sequential(
            nn.Linear(
                goal_dim,
                hidden_dim,
            ),
            activation_fn(),
            nn.Linear(
                hidden_dim,
                embedding_dim,
            ),
        )

    def forward(
        self,
        robot_obs,
        action,
        goal,
    ):
        # Match the network parameter dtype.
        dtype = (
            self.state_action_encoder[0]
            .weight
            .dtype
        )

        robot_obs = robot_obs.to(
            dtype=dtype
        )

        action = action.to(
            dtype=dtype
        )

        goal = goal.to(
            dtype=dtype
        )

        state_action = torch.cat(
            [robot_obs, action],
            dim=-1,
        )

        phi = self.state_action_encoder(
            state_action
        )

        psi = self.goal_encoder(goal)

        if self.normalize_embeddings:
            phi = nn.functional.normalize(
                phi,
                dim=-1,
            )

            psi = nn.functional.normalize(
                psi,
                dim=-1,
            )

        q_value = (
            phi * psi
        ).sum(
            dim=-1,
            keepdim=True,
        )

        return q_value


# ============================================================
# SB3-compatible factorised critic
# ============================================================

class DotProductContinuousCritic(
    ContinuousCritic
):
    def __init__(
        self,
        *args,
        embedding_dim=256,
        normalize_embeddings=False,
        **kwargs,
    ):
        super().__init__(
            *args,
            **kwargs,
        )

        observation_space = (
            self.observation_space
        )

        action_space = self.action_space

        if not hasattr(
            observation_space,
            "spaces",
        ):
            raise TypeError(
                "DotProductContinuousCritic "
                "requires a Dict observation space."
            )

        if "observation" not in (
            observation_space.spaces
        ):
            raise KeyError(
                "Missing 'observation' key."
            )

        if "desired_goal" not in (
            observation_space.spaces
        ):
            raise KeyError(
                "Missing 'desired_goal' key."
            )

        robot_dim = observation_space[
            "observation"
        ].shape[0]

        goal_dim = observation_space[
            "desired_goal"
        ].shape[0]

        action_dim = get_action_dim(
            action_space
        )

        hidden_dim = 256

        # Remove the standard SB3 critic networks
        # inherited from ContinuousCritic.
        for name in ["qf0", "qf1"]:
            if hasattr(self, name):
                delattr(self, name)

        # Some SB3 versions store the standard
        # critics under q_networks.
        if hasattr(self, "q_networks"):
            del self.q_networks

        # Your two actual factorised critics.
        self.q_networks = nn.ModuleList(
            [
                DotProductQNetwork(
                    robot_dim=robot_dim,
                    action_dim=action_dim,
                    goal_dim=goal_dim,
                    embedding_dim=embedding_dim,
                    hidden_dim=hidden_dim,
                    activation_fn=nn.ReLU,
                    normalize_embeddings=(
                        normalize_embeddings
                    ),
                )
                for _ in range(self.n_critics)
            ]
        )


    def forward(
        self,
        obs,
        actions,
    ):
        robot_obs = obs["observation"]
        goal = obs["desired_goal"]

        return tuple(
            q_network(
                robot_obs,
                actions,
                goal,
            )
            for q_network in self.q_networks
        )

    def q1_forward(
        self,
        obs,
        actions,
    ):
        robot_obs = obs["observation"]
        goal = obs["desired_goal"]

        return self.q_networks[0](
            robot_obs,
            actions,
            goal,
        )


# ============================================================
# Custom TD3 policy
# ============================================================

class DotProductTD3Policy(TD3MultiInputPolicy):
    """
    TD3 policy that replaces SB3's standard
    concatenated critic with the dot-product
    factorised critic.
    """

    def make_critic(
        self,
        features_extractor: (
            BaseFeaturesExtractor | None
        ) = None,
    ) -> ContinuousCritic:
        critic_kwargs = (
            self._update_features_extractor(
                self.critic_kwargs,
                features_extractor,
            )
        )

        return DotProductContinuousCritic(
            **critic_kwargs
        ).to(self.device)


import numpy as np

from stable_baselines3.common.noise import (
    NormalActionNoise,
)
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
)


class MutableNormalActionNoise(
    NormalActionNoise
):
    def set_sigma(self, sigma):
        self._sigma = (
            np.ones_like(self._sigma)
            * float(sigma)
        )

    def get_sigma(self):
        return float(
            np.asarray(self._sigma).mean()
        )


class AdaptiveActionNoiseCallback(
    BaseCallback
):
    def __init__(
        self,
        action_noise,
        success_callback,
        success_threshold=0.95,
        initial_sigma=0.02,
        min_sigma=0.01,
        max_sigma=0.08,
        increase_factor=1.5,
        decrease_factor=0.5,
        poor_eval_patience=2,
        verbose=1,
    ):
        super().__init__(verbose)

        self.action_noise = action_noise
        self.success_callback = (
            success_callback
        )

        self.success_threshold = (
            success_threshold
        )

        self.initial_sigma = initial_sigma
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma

        self.increase_factor = (
            increase_factor
        )
        self.decrease_factor = (
            decrease_factor
        )
        self.poor_eval_patience = (
            poor_eval_patience
        )

        self.previous_eval_count = 0
        self.poor_eval_count = 0

    def _on_training_start(self):
        self.action_noise.set_sigma(
            self.initial_sigma
        )

        self.previous_eval_count = 0
        self.poor_eval_count = 0

        if self.verbose:
            print(
                "[Adaptive noise] "
                "Initial sigma: "
                f"{self.action_noise.get_sigma():.4f}"
            )

    def _on_step(self):
        success_rates = (
            self.success_callback.success_rates
        )

        current_eval_count = len(
            success_rates
        )

        # Wait until a new evaluation result
        # has been produced.
        if (
            current_eval_count
            <= self.previous_eval_count
        ):
            return True

        latest_success_rate = float(
            success_rates[-1]
        )

        current_sigma = (
            self.action_noise.get_sigma()
        )

        if (
            latest_success_rate
            >= self.success_threshold
        ):
            new_sigma = max(
                self.min_sigma,
                current_sigma
                * self.decrease_factor,
            )

            self.poor_eval_count = 0
            decision = "decreased"

        else:
            self.poor_eval_count += 1

            if (
                self.poor_eval_count
                >= self.poor_eval_patience
            ):
                new_sigma = min(
                    self.max_sigma,
                    current_sigma
                    * self.increase_factor,
                )

                self.poor_eval_count = 0
                decision = "increased"

            else:
                new_sigma = current_sigma
                decision = "unchanged"

        self.action_noise.set_sigma(
            new_sigma
        )

        if self.verbose:
            print(
                "[Adaptive noise] "
                f"evaluation success="
                f"{latest_success_rate:.3f}, "
                f"sigma="
                f"{current_sigma:.4f} -> "
                f"{new_sigma:.4f}, "
                f"{decision}"
            )

        self.previous_eval_count = (
            current_eval_count
        )

        return True