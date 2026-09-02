from gymnasium import spaces
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.td3.policies import MultiInputPolicy as TD3MultiInputPolicy, Actor
from typing import Optional
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ContinuousCritic, BasePolicy
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any

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


def project_to_l2_ball(
    x: torch.Tensor,
    max_norm: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Project each final-dimension vector into an L2 ball.

    Vectors with norm <= max_norm are unchanged.
    Vectors with norm > max_norm are rescaled to max_norm.
    """
    if max_norm <= 0.0:
        raise ValueError("max_norm must be positive.")

    norms = torch.linalg.vector_norm(x, ord=2, dim=-1, keepdim=True)

    scale = torch.clamp(
        max_norm / norms.clamp_min(eps),
        max=1.0,  # Critical: clamp at 1.0, not min!
    )

    return x * scale

class FactorisedTwinCriticFetch(nn.Module):
    """
    Goal-conditioned factorised twin critic:

        Q_i(s, a, g) = phi_i(s, a)^T psi_i(g)

    Inputs:
        state:  [B, 28]
        action: [B, 4]
        goal:   [B, 3]

    Outputs:
        q1: [B, 1]
        q2: [B, 1]

    Each Q-head has independent phi and psi encoders.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        goal_dim: int,
        latent_dim: int = 128,
        hidden_dim: int = 256,
        activation_fn: type[nn.Module] = nn.ReLU,
    ):
        super().__init__()

        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.goal_dim = int(goal_dim)
        self.latent_dim = int(latent_dim)

        state_action_dim = (
            self.state_dim
            + self.action_dim
        )

        def mlp(
            input_dim: int,
            output_dim: int,
        ) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(
                    input_dim,
                    hidden_dim,
                ),
                activation_fn(),
                nn.Linear(
                    hidden_dim,
                    hidden_dim,
                ),
                activation_fn(),
                nn.Linear(
                    hidden_dim,
                    output_dim,
                ),
            )

        self.phi1 = mlp(
            state_action_dim,
            self.latent_dim,
        )

        self.psi1 = mlp(
            self.goal_dim,
            self.latent_dim,
        )

        self.phi2 = mlp(
            state_action_dim,
            self.latent_dim,
        )

        self.psi2 = mlp(
            self.goal_dim,
            self.latent_dim,
        )

    def _check_inputs(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        goal: torch.Tensor,
    ):
        if state.ndim != 2:
            raise ValueError(
                f"state must be [B, {self.state_dim}], "
                f"got {tuple(state.shape)}."
            )

        if action.ndim != 2:
            raise ValueError(
                f"action must be [B, {self.action_dim}], "
                f"got {tuple(action.shape)}."
            )

        if goal.ndim != 2:
            raise ValueError(
                f"goal must be [B, {self.goal_dim}], "
                f"got {tuple(goal.shape)}."
            )

        if state.shape[0] != action.shape[0]:
            raise ValueError(
                "State and action batch sizes differ."
            )

        if state.shape[0] != goal.shape[0]:
            raise ValueError(
                "State and goal batch sizes differ."
            )

        if state.shape[-1] != self.state_dim:
            raise ValueError(
                f"Expected state_dim={self.state_dim}, "
                f"got {state.shape[-1]}."
            )

        if action.shape[-1] != self.action_dim:
            raise ValueError(
                f"Expected action_dim={self.action_dim}, "
                f"got {action.shape[-1]}."
            )

        if goal.shape[-1] != self.goal_dim:
            raise ValueError(
                f"Expected goal_dim={self.goal_dim}, "
                f"got {goal.shape[-1]}."
            )

    def phi1_forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        return self.phi1(
            torch.cat(
                [state, action],
                dim=-1,
            )
        )

    def psi1_forward(
        self,
        goal: torch.Tensor,
    ) -> torch.Tensor:
        return self.psi1(goal)

    def phi2_forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        return self.phi2(
            torch.cat(
                [state, action],
                dim=-1,
            )
        )

    def psi2_forward(
        self,
        goal: torch.Tensor,
    ) -> torch.Tensor:
        return self.psi2(goal)

    def q1_forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        goal: torch.Tensor,
    ) -> torch.Tensor:
        self._check_inputs(
            state,
            action,
            goal,
        )

        phi = self.phi1_forward(
            state,
            action,
        )

        psi = self.psi1_forward(
            goal,
        )

        return (
            phi * psi
        ).sum(
            dim=-1,
            keepdim=True,
        )

    def q2_forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        goal: torch.Tensor,
    ) -> torch.Tensor:
        self._check_inputs(
            state,
            action,
            goal,
        )

        phi = self.phi2_forward(
            state,
            action,
        )

        psi = self.psi2_forward(
            goal,
        )

        return (
            phi * psi
        ).sum(
            dim=-1,
            keepdim=True,
        )

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        goal: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.q1_forward(
                state,
                action,
                goal,
            ),
            self.q2_forward(
                state,
                action,
                goal,
            ),
        )

    @torch.no_grad()
    def embedding_diagnostics(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        goal: torch.Tensor,
    ) -> dict[str, float]:
        """
        Optional monitoring only. It does not modify representations.
        """
        phi1 = self.phi1_forward(
            state,
            action,
        )

        psi1 = self.psi1_forward(
            goal,
        )

        phi2 = self.phi2_forward(
            state,
            action,
        )

        psi2 = self.psi2_forward(
            goal,
        )

        return {
            "phi1_norm": float(
                phi1.norm(
                    dim=-1
                ).mean().cpu()
            ),
            "psi1_norm": float(
                psi1.norm(
                    dim=-1
                ).mean().cpu()
            ),
            "phi2_norm": float(
                phi2.norm(
                    dim=-1
                ).mean().cpu()
            ),
            "psi2_norm": float(
                psi2.norm(
                    dim=-1
                ).mean().cpu()
            ),
            "phi1_abs": float(
                phi1.abs().mean().cpu()
            ),
            "psi1_abs": float(
                psi1.abs().mean().cpu()
            ),
            "phi2_abs": float(
                phi2.abs().mean().cpu()
            ),
            "psi2_abs": float(
                psi2.abs().mean().cpu()
            ),
        }

class MLPPolicyActor(nn.Module):
    """
    Deterministic goal-conditioned TD3 actor.

    Input:
        state: [B, state_dim]
        goal:  [B, goal_dim]

    Output:
        bounded continuous action: [B, action_dim]
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        goal_dim: int,
        net_arch: list[int] = [256, 256],
        activation_fn: type[nn.Module] = nn.ReLU,
        squash_output: bool = True,
    ):
        super().__init__()

        self.state_dim = int(state_dim)
        self.goal_dim = int(goal_dim)
        self.action_dim = int(action_dim)
        self.squash_output = squash_output

        input_dim = self.state_dim + self.goal_dim

        modules = []
        last_dim = input_dim

        for hidden_dim in net_arch:
            modules.append(
                nn.Linear(last_dim, hidden_dim)
            )
            modules.append(activation_fn())
            last_dim = hidden_dim

        modules.append(
            nn.Linear(last_dim, self.action_dim)
        )

        if squash_output:
            modules.append(nn.Tanh())

        self.mu_net = nn.Sequential(*modules)

    def forward(
        self,
        state: torch.Tensor,
        goal: torch.Tensor,
    ) -> torch.Tensor:
        if state.ndim == 1:
            state = state.unsqueeze(0)

        if goal.ndim == 1:
            goal = goal.unsqueeze(0)

        if state.ndim != 2:
            raise ValueError(
                "state must have shape [B, state_dim]. "
                f"Got {tuple(state.shape)}."
            )

        if goal.ndim != 2:
            raise ValueError(
                "goal must have shape [B, goal_dim]. "
                f"Got {tuple(goal.shape)}."
            )

        if state.shape[0] != goal.shape[0]:
            raise ValueError(
                "State and goal batch sizes must match: "
                f"{state.shape[0]} vs {goal.shape[0]}."
            )

        if state.shape[-1] != self.state_dim:
            raise ValueError(
                "Wrong state dimension: "
                f"expected {self.state_dim}, "
                f"got {state.shape[-1]}."
            )

        if goal.shape[-1] != self.goal_dim:
            raise ValueError(
                "Wrong goal dimension: "
                f"expected {self.goal_dim}, "
                f"got {goal.shape[-1]}."
            )

        x = torch.cat(
            [state, goal],
            dim=-1,
        )

        return self.mu_net(x)


class StandardTwinCriticFetch(nn.Module):
    """
    Standard twin critic for TD3 (non-factorised).
    
    Q_k(s, a, g) = MLP([s, a, g]),  k = 1, 2
    
    Designed for Fetch environments where:
        - s: state vector (e.g. obs["observation"], 25-dim)
        - a: action (4-dim)
        - g: goal (e.g. obs["desired_goal"], 3-dim)
    
    Architecture per critic:
        [state + action + goal] -> hidden_dim -> hidden_dim -> 1
    
    Returns two Q-values for clipped double Q-learning.
    """
    
    def __init__(
        self,
        state_dim: int = 25,
        action_dim: int = 4,
        goal_dim: int = 3,
        hidden_dim: int = 256,
    ):
        super().__init__()
        
        input_dim = state_dim + action_dim + goal_dim
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim
        self.hidden_dim = hidden_dim
        
        # Q1 network
        self.q1_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        
        # Q2 network (twin)
        self.q2_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def _to_float32(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.float64:
            return x.to(torch.float32)
        return x
    
    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        goal: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        state: [B, state_dim]
        action: [B, action_dim]
        goal: [B, goal_dim]
        
        Returns:
            (q1, q2), each [B, 1]
        """
        state = self._to_float32(state)
        action = self._to_float32(action)
        goal = self._to_float32(goal)
        
        x = torch.cat([state, action, goal], dim=-1)
        
        q1 = self.q1_net(x)
        q2 = self.q2_net(x)
        return q1, q2
    
    def q1_forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        goal: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Q1 only.
        """
        state = self._to_float32(state)
        action = self._to_float32(action)
        goal = self._to_float32(goal)
        
        x = torch.cat([state, action, goal], dim=-1)
        return self.q1_net(x)
    
    def q2_forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        goal: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Q2 only.
        """
        state = self._to_float32(state)
        action = self._to_float32(action)
        goal = self._to_float32(goal)
        
        x = torch.cat([state, action, goal], dim=-1)
        return self.q2_net(x)
    
    # ------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------
    
    def get_q_stats(
        self,
        q1: torch.Tensor,
        q2: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Compute summary statistics for Q-values.
        """
        def stats_1d(x: torch.Tensor) -> Dict[str, float]:
            return {
                "mean": x.mean().item(),
                "std": x.std().item(),
                "min": x.min().item(),
                "max": x.max().item(),
            }
        
        q1_flat = q1.detach().view(-1)
        q2_flat = q2.detach().view(-1)
        
        s1 = stats_1d(q1_flat)
        s2 = stats_1d(q2_flat)
        
        return {
            "q1_mean": s1["mean"],
            "q1_std": s1["std"],
            "q1_min": s1["min"],
            "q1_max": s1["max"],
            "q2_mean": s2["mean"],
            "q2_std": s2["std"],
            "q2_min": s2["min"],
            "q2_max": s2["max"],
        }
    
    def log_diagnostics(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        goal: torch.Tensor,
        prefix: str = "train",
    ) -> Dict[str, float]:
        """
        Run a forward pass and return diagnostics.
        """
        with torch.no_grad():
            q1, q2 = self.forward(state, action, goal)
        
        q_stats = self.get_q_stats(q1, q2)
        
        stats = {}
        for k, v in q_stats.items():
            stats[f"{prefix}/{k}"] = v
        
        return stats



class GaussianPolicyActorSAC(nn.Module):
    """
    Goal-conditioned squashed-Gaussian actor for SAC.

    Input:
        state: [B, state_dim]
        goal:  [B, goal_dim]

    Output:
        sampled action in [-1, 1]^action_dim;
        log-probability with the tanh change-of-variables correction;
        deterministic action tanh(mu).

    For FetchPush, action space is normally [-1, 1]^4.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        goal_dim: int,
        net_arch: list[int] = [256, 256],
        activation_fn: type[nn.Module] = nn.ReLU,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
    ):
        super().__init__()

        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.goal_dim = int(goal_dim)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)

        input_dim = self.state_dim + self.goal_dim

        modules = []
        last_dim = input_dim

        for hidden_dim in net_arch:
            modules.append(nn.Linear(last_dim, hidden_dim))
            modules.append(activation_fn())
            last_dim = hidden_dim

        self.backbone = nn.Sequential(*modules)

        self.mu_layer = nn.Linear(
            last_dim,
            self.action_dim,
        )

        self.log_std_layer = nn.Linear(
            last_dim,
            self.action_dim,
        )

    def _validate_inputs(
        self,
        state: torch.Tensor,
        goal: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if state.ndim == 1:
            state = state.unsqueeze(0)

        if goal.ndim == 1:
            goal = goal.unsqueeze(0)

        if state.ndim != 2:
            raise ValueError(
                "state must have shape [B, state_dim]. "
                f"Got {tuple(state.shape)}."
            )

        if goal.ndim != 2:
            raise ValueError(
                "goal must have shape [B, goal_dim]. "
                f"Got {tuple(goal.shape)}."
            )

        if state.shape[0] != goal.shape[0]:
            raise ValueError(
                "State and goal batch dimensions must match: "
                f"{state.shape[0]} vs {goal.shape[0]}."
            )

        if state.shape[-1] != self.state_dim:
            raise ValueError(
                f"Expected state dimension {self.state_dim}, "
                f"got {state.shape[-1]}."
            )

        if goal.shape[-1] != self.goal_dim:
            raise ValueError(
                f"Expected goal dimension {self.goal_dim}, "
                f"got {goal.shape[-1]}."
            )

        return state, goal

    def forward(
        self,
        state: torch.Tensor,
        goal: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            mean:    [B, action_dim]
            log_std: [B, action_dim]
        """
        state, goal = self._validate_inputs(state, goal)

        x = torch.cat(
            [state, goal],
            dim=-1,
        )

        x = self.backbone(x)

        mean = self.mu_layer(x)

        log_std = self.log_std_layer(x).clamp(
            min=self.log_std_min,
            max=self.log_std_max,
        )

        return mean, log_std

    def sample(
        self,
        state: torch.Tensor,
        goal: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Reparameterized SAC sample.

        Returns:
            action:
                Tanh-squashed sampled action [B, action_dim].
            log_prob:
                Log probability of action after tanh correction [B, 1].
            deterministic_action:
                tanh(mean), for deterministic evaluation [B, action_dim].
        """
        mean, log_std = self.forward(state, goal)

        std = log_std.exp()

        normal = torch.distributions.Normal(
            mean,
            std,
        )

        pre_tanh_action = normal.rsample()

        action = torch.tanh(pre_tanh_action)

        log_prob = normal.log_prob(pre_tanh_action)

        tanh_correction = torch.log(
            1.0 - action.pow(2) + 1e-6
        )

        log_prob = (
            log_prob - tanh_correction
        ).sum(
            dim=-1,
            keepdim=True,
        )

        deterministic_action = torch.tanh(mean)

        return (
            action,
            log_prob,
            deterministic_action,
        )

    def deterministic(
        self,
        state: torch.Tensor,
        goal: torch.Tensor,
    ) -> torch.Tensor:
        """
        Deterministic tanh(mean) action for evaluation.
        """
        mean, _ = self.forward(state, goal)

        return torch.tanh(mean)


class RunningMeanStd:
    """
    Running per-feature mean and variance using numerically stable
    parallel/Welford-style updates.
    """

    def __init__(
        self,
        shape,
        device,
        epsilon: float = 1e-4,
        dtype: torch.dtype = torch.float32,
    ):
        self.mean = torch.zeros(
            shape,
            dtype=dtype,
            device=device,
        )

        self.var = torch.ones(
            shape,
            dtype=dtype,
            device=device,
        )

        self.count = torch.tensor(
            epsilon,
            dtype=dtype,
            device=device,
        )

    @torch.no_grad()
    def update(
        self,
        x: torch.Tensor,
    ) -> None:
        """
        x has shape [feature_dim] or [batch_size, feature_dim].
        """
        if x.ndim == 1:
            x = x.unsqueeze(0)

        if x.ndim != 2:
            raise ValueError(
                "RunningMeanStd.update expects [D] or [B, D], "
                f"got {tuple(x.shape)}."
            )

        x = x.detach().to(
            device=self.mean.device,
            dtype=self.mean.dtype,
        )

        batch_count = torch.as_tensor(
            float(x.shape[0]),
            dtype=self.mean.dtype,
            device=self.mean.device,
        )

        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + (
            delta * batch_count / total_count
        )

        mean_a = self.var * self.count
        mean_b = batch_var * batch_count

        correction = (
            delta.pow(2)
            * self.count
            * batch_count
            / total_count
        )

        new_var = (
            mean_a + mean_b + correction
        ) / total_count

        self.mean.copy_(new_mean)
        self.var.copy_(new_var)
        self.count.copy_(total_count)

    def normalize(
        self,
        x: torch.Tensor,
        clip: float = 10.0,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        Normalize with the current running statistics.

        This function does not update statistics. Gradients still flow
        through x when x requires gradients.
        """
        x = x.to(
            device=self.mean.device,
            dtype=self.mean.dtype,
        )

        z = (
            x - self.mean
        ) / torch.sqrt(self.var + eps)

        return z.clamp(-clip, clip)