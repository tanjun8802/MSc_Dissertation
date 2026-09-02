from __future__ import annotations

from dataclasses import dataclass
import torch
import numpy as np
import random
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import time
from IPython.display import (
    clear_output,
    display,
)
from collections import defaultdict
from sklearn.cluster import AgglomerativeClustering

@dataclass # typed container for batches of replay data
class ReplayBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_obs: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    episode_id: torch.Tensor
    timestep: torch.Tensor
    indices: torch.Tensor

    def __len__(self):
        return self.obs.shape[0]


class TrajectoryReplayBuffer:
    def __init__(self, capacity, obs_dim, action_dim, device="cpu"):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device

        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.terminated = np.zeros((capacity, 1), dtype=np.float32)
        self.truncated = np.zeros((capacity, 1), dtype=np.float32)

        self.episode_id = np.full((capacity,), -1, dtype=np.int64)
        self.timestep = np.full((capacity,), -1, dtype=np.int64)

        self.pos = 0
        self.size = 0
        self.full = False

        self.current_episode_id = 0
        self.episode_to_indices = {}

    def __len__(self):
        return self.size
    
    def _remove_index_from_episode_mapping(self, idx):
        old_ep = self.episode_id[idx]
        if old_ep in self.episode_to_indices:
            try:
                self.episode_to_indices[old_ep].remove(idx)
                if len(self.episode_to_indices[old_ep]) == 0:
                    del self.episode_to_indices[old_ep]
            except ValueError:
                pass


    def add_transition(
        self,
        obs,
        action,
        reward,
        next_obs,
        terminated,
        truncated=False,
        episode_id=None,
        timestep=None,
    ):

        idx = self.pos

        if self.full:
            self._remove_index_from_episode_mapping(idx)

        # If caller does not provide episode metadata, create a fresh one-step episode.
        if episode_id is None:
            ep_id = self.current_episode_id
            self.current_episode_id += 1
        else:
            ep_id = int(episode_id)
            if ep_id >= self.current_episode_id:
                self.current_episode_id = ep_id + 1

        if timestep is None:
            t = 0
        else:
            t = int(timestep)

        self.obs[idx] = np.asarray(obs, dtype=np.float32)
        self.actions[idx] = np.int64(action)
        self.rewards[idx] = np.asarray([reward], dtype=np.float32)
        self.next_obs[idx] = np.asarray(next_obs, dtype=np.float32)
        self.terminated[idx] = np.asarray([terminated], dtype=np.float32)
        self.truncated[idx] = np.asarray([truncated], dtype=np.float32)

        self.episode_id[idx] = ep_id
        self.timestep[idx] = t

        if ep_id not in self.episode_to_indices:
            self.episode_to_indices[ep_id] = []
        self.episode_to_indices[ep_id].append(idx)

        self.pos = (self.pos + 1) % self.capacity

        if self.size < self.capacity:
            self.size += 1
        else:
            self.full = True

    def add_episode(self, episode):
        ep_id = self.current_episode_id
        self.current_episode_id += 1

        T = len(episode["obs"])
        ep_indices = []

        for t in range(T):
            idx_before = self.pos

            self.add_transition(
                obs=episode["obs"][t],
                action=episode["actions"][t],
                reward=episode["rewards"][t],
                next_obs=episode["next_obs"][t],
                terminated=episode["terminated"][t],
                truncated=episode["truncated"][t],
                episode_id=ep_id,
                timestep=t,
            )

            ep_indices.append(idx_before)

        self.episode_to_indices[ep_id] = ep_indices

    def sample(self, batch_size):
        assert self.size > 0, "Buffer is empty"
        idxs = np.random.randint(0, self.size, size=batch_size)

        return ReplayBatch(
            obs=torch.tensor(self.obs[idxs], device=self.device),
            actions=torch.tensor(self.actions[idxs], device=self.device),
            rewards=torch.tensor(self.rewards[idxs], device=self.device),
            next_obs=torch.tensor(self.next_obs[idxs], device=self.device),
            terminated=torch.tensor(self.terminated[idxs], device=self.device),
            truncated=torch.tensor(self.truncated[idxs], device=self.device),
            episode_id=torch.tensor(self.episode_id[idxs], device=self.device),
            timestep=torch.tensor(self.timestep[idxs], device=self.device),
            indices=torch.tensor(idxs, device=self.device),
        )

    def sample_positive_future_goal_batch(self, batch_size, min_k=1, max_k=None, gamma=0.99, exclude_self_loops=True, atol=1e-6):
        
        assert self.size > 0, "Buffer is empty"
        assert 0.0 <= gamma < 1.0, "gamma must be in [0, 1)"

        valid_indices = []
        future_goal_indices = []

        tries = 0
        max_tries = batch_size * 200  # higher because we may reject self-loops

        while len(valid_indices) < batch_size and tries < max_tries:
            idx = np.random.randint(0, self.size)
            ep_id = self.episode_id[idx]
            t = self.timestep[idx]

            if ep_id == -1 or ep_id not in self.episode_to_indices:
                tries += 1
                continue

            ep_idxs = self.episode_to_indices[ep_id]
            ep_len = len(ep_idxs)

            if t >= ep_len - 1:
                tries += 1
                continue

            # Reject self-loop anchor transitions: obs -> next_obs does not move
            if exclude_self_loops:
                if np.allclose(self.obs[idx], self.next_obs[idx], atol=atol, rtol=0.0):
                    tries += 1
                    continue

            max_valid_k = ep_len - 1 - t
            if max_k is not None:
                max_valid_k = min(max_valid_k, max_k)

            if max_valid_k < min_k:
                tries += 1
                continue

            ks = np.arange(min_k, max_valid_k + 1, dtype=np.int64)

            if gamma == 0.0:
                probs = np.zeros_like(ks, dtype=np.float64)
                probs[0] = 1.0
            else:
                log_weights = (ks - 1) * np.log(gamma)
                log_weights -= np.max(log_weights)
                weights = np.exp(log_weights)
                probs = weights / weights.sum()

            k = np.random.choice(ks, p=probs)
            future_t = t + k
            future_idx = ep_idxs[future_t]

            valid_indices.append(idx)
            future_goal_indices.append(future_idx)
            tries += 1

        assert len(valid_indices) == batch_size, (
            f"Could only sample {len(valid_indices)} valid transitions out of "
            f"requested {batch_size}. Increase max_tries or collect more non-self-loop data."
        )

        idxs = np.array(valid_indices, dtype=np.int64)
        g_idxs = np.array(future_goal_indices, dtype=np.int64)

        batch = {
            "obs": torch.tensor(self.obs[idxs], device=self.device),
            "actions": torch.tensor(self.actions[idxs], device=self.device),
            "next_obs": torch.tensor(self.next_obs[idxs], device=self.device),
            "goals": torch.tensor(self.obs[g_idxs], device=self.device),
            "rewards": torch.tensor(self.rewards[idxs], device=self.device),
            "terminated": torch.tensor(self.terminated[idxs], device=self.device),
            "truncated": torch.tensor(self.truncated[idxs], device=self.device),
            "episode_id": torch.tensor(self.episode_id[idxs], device=self.device),
            "timestep": torch.tensor(self.timestep[idxs], device=self.device),
            "future_timestep": torch.tensor(self.timestep[g_idxs], device=self.device),
            "indices": torch.tensor(idxs, device=self.device),
            "goal_indices": torch.tensor(g_idxs, device=self.device),
        }
        return batch

    def sample_negative_future_goals(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size) # pick random indices from the buffer and indexing the observations
        return torch.tensor(self.obs[idxs], device=self.device)
    
    def sample_random_goals(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size) # pick random indices from the buffer and indexing the observations
        return torch.tensor(self.obs[idxs], device=self.device)
    
    def sample_positive_future_goal_episode(self, episode_index, timestep, k, gamma = 0.99):

        if episode_index not in self.episode_to_indices:
            raise ValueError(f"Episode index {episode_index} not found in buffer")

        ep_idxs = self.episode_to_indices[episode_index]
        ep_len = len(ep_idxs)

        if timestep >= ep_len - 1:
            raise ValueError(f"Timestep {timestep} is out of bounds for episode of length {ep_len}")

        max_valid_k = ep_len - 1 - timestep
        if k > max_valid_k:
            raise ValueError(f"k={k} is too large for episode of length {ep_len} at timestep {timestep}")

        geometric = torch.distributions.Geometric(probs=torch.tensor(1-gamma)) # geometric distribution to sample k with probability proportional to gamma^k
        steps_ahead = int(geometric.sample().item()) + 1 # sample k, cast to int, and add 1 to ensure it's at least t+1
        steps_ahead = min(steps_ahead, max_valid_k) # clamp so we never index past the end of the episode
        print(f"Sampled k={steps_ahead} from geometric distribution with gamma={gamma}")
        future_t = timestep + steps_ahead
        future_idx = ep_idxs[future_t]
        print(self.obs[future_idx])
        return torch.tensor(self.obs[future_idx], device=self.device) # return the future state as the positive goal

    def stats(self):
        return {
            "size": self.size,
            "capacity": self.capacity,
            "num_episodes": len(self.episode_to_indices),
            "current_episode_id": self.current_episode_id,
        }
    

class TrajectoryReplayBufferDiscrete(TrajectoryReplayBuffer):
    def __init__(self, capacity, obs_dim, action_dim, device="cpu"):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device

        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.terminated = np.zeros((capacity, 1), dtype=np.float32)
        self.truncated = np.zeros((capacity, 1), dtype=np.float32)

        self.episode_id = np.full((capacity,), -1, dtype=np.int64)
        self.timestep = np.full((capacity,), -1, dtype=np.int64)

        self.pos = 0
        self.size = 0
        self.full = False

        self.current_episode_id = 0
        self.episode_to_indices = {}

    def __len__(self):
        return self.size

class TrajectoryReplayBufferContinuous:
    def __init__(
        self,
        capacity: int,
        state_dim: int,
        action_dim: int,
        device="cpu",
    ):
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

        # Core transition storage
        self.obs = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((capacity, state_dim), dtype=np.float32)
        self.terminated = np.zeros((capacity, 1), dtype=np.float32)
        self.truncated = np.zeros((capacity, 1), dtype=np.float32)

        # Episode tracking (same as your original buffer)
        self.episode_id = np.full((capacity,), -1, dtype=np.int64)
        self.timestep = np.full((capacity,), -1, dtype=np.int64)

        self.pos = 0
        self.size = 0
        self.full = False

        self.current_episode_id = 0
        self.episode_to_indices: Dict[int, list] = {}

    def __len__(self) -> int:
        return self.size

    def _remove_index_from_episode_mapping(self, idx: int):
        old_ep = self.episode_id[idx]
        if old_ep in self.episode_to_indices:
            try:
                self.episode_to_indices[old_ep].remove(idx)
                if len(self.episode_to_indices[old_ep]) == 0:
                    del self.episode_to_indices[old_ep]
            except ValueError:
                pass

    def add_transition(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        terminated: bool,
        truncated: bool = False,
        episode_id: Optional[int] = None,
        timestep: Optional[int] = None,
    ):
        """
        Add a single transition.

        For Fetch:
          - obs, next_obs: can be either
              * dict with key "observation", or
              * already-extracted state vector (25-dim).
          - action: continuous action vector (4-dim).
        """
        idx = self.pos

        if self.full:
            self._remove_index_from_episode_mapping(idx)

        # Handle dict obs from Fetch
        if isinstance(obs, dict):
            obs_vec = np.asarray(obs["observation"], dtype=np.float32)
        else:
            obs_vec = np.asarray(obs, dtype=np.float32)

        if isinstance(next_obs, dict):
            next_obs_vec = np.asarray(next_obs["observation"], dtype=np.float32)
        else:
            next_obs_vec = np.asarray(next_obs, dtype=np.float32)

        action_vec = np.asarray(action, dtype=np.float32)

        # Episode metadata
        if episode_id is None:
            ep_id = self.current_episode_id
            self.current_episode_id += 1
        else:
            ep_id = int(episode_id)
            if ep_id >= self.current_episode_id:
                self.current_episode_id = ep_id + 1

        if timestep is None:
            t = 0
        else:
            t = int(timestep)

        # Store
        self.obs[idx] = obs_vec
        self.actions[idx] = action_vec
        self.rewards[idx] = np.asarray([reward], dtype=np.float32)
        self.next_obs[idx] = next_obs_vec
        self.terminated[idx] = np.asarray([terminated], dtype=np.float32)
        self.truncated[idx] = np.asarray([truncated], dtype=np.float32)

        self.episode_id[idx] = ep_id
        self.timestep[idx] = t

        if ep_id not in self.episode_to_indices:
            self.episode_to_indices[ep_id] = []
        self.episode_to_indices[ep_id].append(idx)

        self.pos = (self.pos + 1) % self.capacity

        if self.size < self.capacity:
            self.size += 1
        else:
            self.full = True

    def add_episode(
        self,
        episode: Dict[str, Any],
    ):
        """
        Add a full episode.

        episode should have keys:
          - "obs": list of obs (dict or state vectors)
          - "actions": list/array of actions
          - "rewards": list/array of rewards
          - "next_obs": list of next obs
          - "terminated": list/array of terminated flags
          - "truncated": list/array of truncated flags
        """
        ep_id = self.current_episode_id
        self.current_episode_id += 1

        T = len(episode["obs"])
        ep_indices = []

        for t in range(T):
            idx_before = self.pos

            self.add_transition(
                obs=episode["obs"][t],
                action=episode["actions"][t],
                reward=episode["rewards"][t],
                next_obs=episode["next_obs"][t],
                terminated=episode["terminated"][t],
                truncated=episode["truncated"][t],
                episode_id=ep_id,
                timestep=t,
            )

            ep_indices.append(idx_before)

        self.episode_to_indices[ep_id] = ep_indices

    def sample(self, batch_size: int) -> ReplayBatch:
        """
        Sample a random batch of transitions.
        """
        assert self.size > 0, "Buffer is empty"
        idxs = np.random.randint(0, self.size, size=batch_size)

        return ReplayBatch(
            obs=torch.tensor(self.obs[idxs], device=self.device),
            actions=torch.tensor(self.actions[idxs], device=self.device),
            rewards=torch.tensor(self.rewards[idxs], device=self.device),
            next_obs=torch.tensor(self.next_obs[idxs], device=self.device),
            terminated=torch.tensor(self.terminated[idxs], device=self.device),
            truncated=torch.tensor(self.truncated[idxs], device=self.device),
            episode_id=torch.tensor(self.episode_id[idxs], device=self.device),
            timestep=torch.tensor(self.timestep[idxs], device=self.device),
            indices=torch.tensor(idxs, device=self.device),
        )

    def stats(self) -> Dict[str, Any]:
        return {
            "size": self.size,
            "capacity": self.capacity,
            "num_episodes": len(self.episode_to_indices),
            "current_episode_id": self.current_episode_id,
        }



@dataclass
class SharedReplayBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor        # computed on the fly, not stored
    next_obs: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    episode_id: torch.Tensor
    timestep: torch.Tensor
    task_id: torch.Tensor
    goal_stored: torch.Tensor    # goal that was active when collected
    indices: torch.Tensor

    def __len__(self):
        return self.obs.shape[0]


class SharedGoalBuffer:
    def __init__(self, capacity, obs_dim, action_dim, goal_dim, device="cpu"):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim
        self.device = device

        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        # No stored rewards; we'll compute them at sampling time.
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.terminated = np.zeros((capacity, 1), dtype=np.float32)
        self.truncated = np.zeros((capacity, 1), dtype=np.float32)

        self.episode_id = np.full((capacity,), -1, dtype=np.int64)
        self.timestep = np.full((capacity,), -1, dtype=np.int64)

        self.task_id = np.full((capacity,), -1, dtype=np.int64)
        self.goal_stored = np.zeros((capacity, goal_dim), dtype=np.float32)

        self.pos = 0
        self.size = 0
        self.full = False

        self.current_episode_id = 0
        self.episode_to_indices = {}

    def __len__(self):
        return self.size

    def _remove_index_from_episode_mapping(self, idx):
        old_ep = self.episode_id[idx]
        if old_ep in self.episode_to_indices:
            try:
                self.episode_to_indices[old_ep].remove(idx)
                if len(self.episode_to_indices[old_ep]) == 0:
                    del self.episode_to_indices[old_ep]
            except ValueError:
                pass

    def add_transition(
        self,
        obs,
        action,
        next_obs,
        terminated,
        truncated,
        task_id,
        goal,
        episode_id=None,
        timestep=None,
    ):
        idx = self.pos

        if self.full:
            self._remove_index_from_episode_mapping(idx)

        if episode_id is None:
            ep_id = self.current_episode_id
            self.current_episode_id += 1
        else:
            ep_id = int(episode_id)
            if ep_id >= self.current_episode_id:
                self.current_episode_id = ep_id + 1

        if timestep is None:
            t = 0
        else:
            t = int(timestep)

        self.obs[idx] = np.asarray(obs, dtype=np.float32)
        self.actions[idx] = int(action)
        self.next_obs[idx] = np.asarray(next_obs, dtype=np.float32)
        self.terminated[idx] = np.asarray([terminated], dtype=np.float32)
        self.truncated[idx] = np.asarray([truncated], dtype=np.float32)

        self.task_id[idx] = int(task_id)
        self.goal_stored[idx] = np.asarray(goal, dtype=np.float32)

        self.episode_id[idx] = ep_id
        self.timestep[idx] = t

        if ep_id not in self.episode_to_indices:
            self.episode_to_indices[ep_id] = []
        self.episode_to_indices[ep_id].append(idx)

        self.pos = (self.pos + 1) % self.capacity

        if self.size < self.capacity:
            self.size += 1
        else:
            self.full = True

    def sample(self, batch_size, goal_fn=None, device=None):
        """
        goal_fn: callable (next_obs_np, goal_np) -> reward scalar or array.
                 If None, returns zero rewards (caller must override).
        """
        assert self.size > 0, "Buffer is empty"
        idxs = np.random.randint(0, self.size, size=batch_size)

        obs_t = torch.tensor(self.obs[idxs], device=device or self.device)
        actions_t = torch.tensor(self.actions[idxs], device=device or self.device)
        next_obs_t = torch.tensor(self.next_obs[idxs], device=device or self.device)
        terminated_t = torch.tensor(self.terminated[idxs], device=device or self.device)
        truncated_t = torch.tensor(self.truncated[idxs], device=device or self.device)
        episode_id_t = torch.tensor(self.episode_id[idxs], device=device or self.device)
        timestep_t = torch.tensor(self.timestep[idxs], device=device or self.device)
        task_id_t = torch.tensor(self.task_id[idxs], device=device or self.device)
        goal_stored_t = torch.tensor(self.goal_stored[idxs], device=device or self.device)

        # Compute rewards on the fly if goal_fn is provided
        if goal_fn is not None:
            # Here we assume goal_fn can take batched numpy arrays
            next_obs_np = self.next_obs[idxs]
            goal_np = self.goal_stored[idxs]
            rewards_np = goal_fn(next_obs_np, goal_np)  # [B, 1] or [B]
            if rewards_np.ndim == 1:
                rewards_np = rewards_np[:, None]
            rewards_t = torch.tensor(rewards_np, dtype=torch.float32, device=device or self.device)
        else:
            rewards_t = torch.zeros((batch_size, 1), dtype=torch.float32, device=device or self.device)

        return SharedReplayBatch(
            obs=obs_t,
            actions=actions_t,
            rewards=rewards_t,
            next_obs=next_obs_t,
            terminated=terminated_t,
            truncated=truncated_t,
            episode_id=episode_id_t,
            timestep=timestep_t,
            task_id=task_id_t,
            goal_stored=goal_stored_t,
            indices=torch.tensor(idxs, device=device or self.device),
        )

    def sample_for_task(self, batch_size, task_id, goal, goal_fn, device=None):
        """
        Sample a batch and compute rewards for a specific task/goal.
        goal: [goal_dim] numpy or torch.
        goal_fn: (next_obs_np, goal_np) -> reward.
        """
        # Simple version: sample globally, then filter by task_id (may be inefficient if many tasks).
        # For now, we just sample and recompute rewards with the given goal.
        batch = self.sample(batch_size, goal_fn=None, device=device)

        # Override rewards using the provided goal
        next_obs_np = self.next_obs[batch.indices.cpu().numpy()]
        goal_np = np.asarray(goal, dtype=np.float32)
        # Broadcast goal to batch size
        goal_batch_np = np.broadcast_to(goal_np[None, :], (batch_size, goal_np.shape[0]))
        rewards_np = goal_fn(next_obs_np, goal_batch_np)
        if rewards_np.ndim == 1:
            rewards_np = rewards_np[:, None]
        batch.rewards = torch.tensor(rewards_np, dtype=torch.float32, device=device or self.device)

        return batch

    def stats(self):
        return {
            "size": self.size,
            "capacity": self.capacity,
            "num_episodes": len(self.episode_to_indices),
            "num_tasks": int(np.unique(self.task_id[:self.size]).size),
        }

class PhiNoveltyFilter:
    def __init__(self, phi_dim, tau_novel, max_phi_samples=20000, device="cpu"):
        self.phi_dim = phi_dim
        self.tau_novel = tau_novel
        self.max_phi_samples = max_phi_samples
        self.device = device
        self.phis = torch.empty(0, phi_dim, device=device)

    def is_novel(self, phi_new):
        # phi_new: [phi_dim] tensor on self.device
        if self.phis.numel() == 0:
            return True
        dists = torch.norm(self.phis - phi_new, dim=1)
        d_min = dists.min().item()
        return d_min > self.tau_novel

    def add(self, phi_new):
        # phi_new: [phi_dim]
        if self.phis.shape[0] >= self.max_phi_samples:
            # Simple FIFO; you can replace with clustering/compression later
            self.phis = self.phis[1:]
        self.phis = torch.cat([self.phis, phi_new.unsqueeze(0)], dim=0)


def evaluate_policy(
    env,
    policy_fn,
    episodes=8,
    return_per_episode=False,
):
    episode_returns = []
    episode_lengths = []

    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0
        ep_len = 0

        while not done:
            action = policy_fn(obs)
            obs, rew, term, trunc, _ = env.step(action)
            done = term or trunc
            ep_ret += float(rew)
            ep_len += 1

        episode_returns.append(ep_ret)
        episode_lengths.append(ep_len)

    mean_ret = float(np.mean(episode_returns)) if len(episode_returns) > 0 else 0.0
    mean_len = float(np.mean(episode_lengths)) if len(episode_lengths) > 0 else 0.0

    if return_per_episode:
        return mean_ret, mean_len, episode_returns
    return mean_ret, mean_len


def collect_episode(env, policy_fn):
    obs, _ = env.reset()
    ep = {k: [] for k in ['obs', 'actions', 'rewards', 'next_obs', 'terminated', 'truncated']}
    done = False
    while not done:
        act = policy_fn(obs)
        next_obs, rew, term, trunc, _ = env.step(act)
        ep['obs'].append(obs.astype(np.float32))
        ep['actions'].append(act.astype(np.float32))
        ep['rewards'].append(np.float32(rew))
        ep['next_obs'].append(next_obs.astype(np.float32))
        ep['terminated'].append(np.float32(term))
        ep['truncated'].append(np.float32(trunc))
        obs = next_obs
        done = term or trunc
    return ep, len(ep['obs'])

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def build_goal_batch(goal, batch_size, device):
    goal_arr = np.array(goal, dtype=np.float32)
    return torch.tensor(goal_arr, dtype=torch.float32, device=device).unsqueeze(0).expand(batch_size, -1)


def get_base_env(env):
    return env.unwrapped

def collect_valid_states_fourrooms(env):

    base_env = get_base_env(env)

    if not hasattr(base_env, "_free_cells"):
        raise RuntimeError("Base env does not expose _free_cells.")

    coords = np.asarray(base_env._free_cells, dtype=np.int32)      # [N, 2]
    states = coords.astype(np.float32)                             # obs == (x, y)

    return states, coords

def estimate_fisher_diag(
    model,
    target_model,
    replay_buffer,
    goal,
    num_actions,
    device=None,
    gamma=0.99,
    batch_size=256,
    n_batches=64,
    prefix_filter="sa_encoder",
    use_success_only=False,
):

    model.eval()
    target_model.eval()

    fisher_diag = {}

    # Initialise Fisher entries for selected parameters
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        if prefix_filter is None:
            fisher_diag[name] = torch.zeros_like(p, device=device)
        elif isinstance(prefix_filter, str):
            if name.startswith(prefix_filter):
                fisher_diag[name] = torch.zeros_like(p, device=device)
        else:
            if any(name.startswith(pref) for pref in prefix_filter):
                fisher_diag[name] = torch.zeros_like(p, device=device)

    if len(fisher_diag) == 0:
        raise ValueError("No parameters matched prefix_filter for Fisher estimation.")

    # Accumulate squared gradients
    num_batches_used = 0

    for _ in range(n_batches):
        if use_success_only and hasattr(replay_buffer, "sample_success"):
            batch = replay_buffer.sample_success(batch_size)
        else:
            batch = replay_buffer.sample(batch_size)

        obs_t = batch.obs          # [B, obs_dim]
        act_t = batch.actions.long()   # [B, 1]
        rew_t = batch.rewards      # [B, 1]
        next_obs_t = batch.next_obs
        term_t = batch.terminated
        trunc_t = batch.truncated

        B = obs_t.shape[0]
        goal_batch = build_goal_batch(goal, B, device)

        with torch.no_grad():
            next_q_vals = target_model.q_val_for_argmax_action(next_obs_t, goal_batch)
            next_q = next_q_vals.max(dim=-1, keepdim=True).values
            target = rew_t + gamma * (1.0 - term_t) * next_q

        act_onehot = torch.nn.functional.one_hot(
            act_t.squeeze(-1),
            num_classes=num_actions
        ).float()

        current_q = model(obs_t, act_onehot, goal_batch)
        td_loss = torch.nn.functional.mse_loss(current_q, target)

        model.zero_grad(set_to_none=True)
        td_loss.backward()

        for name, p in model.named_parameters():
            if name in fisher_diag and p.grad is not None:
                fisher_diag[name] += p.grad.detach().pow(2)

        num_batches_used += 1

    if num_batches_used == 0:
        raise RuntimeError("No batches used in Fisher estimation.")

    # Average over batches
    for name in fisher_diag:
        fisher_diag[name] /= float(num_batches_used)

    # Per-parameter normalisation to stabilise scale
    eps = 1e-8
    for name, F in fisher_diag.items():
        mean_val = F.mean()
        if mean_val > 0:
            fisher_diag[name] = F / (mean_val + eps)

    scale = 10.0  # try 10, 50, etc.
    for name in fisher_diag:
        fisher_diag[name] = fisher_diag[name] * scale

    model.train()
    target_model.train()

    return fisher_diag


def extract_mean_sa_embedding(
    q_network,
    buffer,
    num_actions,
    batch_size=256,
    device=None,
    as_numpy=True,
):

    if device is None:
        device = next(q_network.parameters()).device

    n_available = len(buffer)
    if n_available == 0:
        raise ValueError("Replay buffer is empty; cannot extract SA embeddings.")

    probe_bs = min(batch_size, n_available)

    q_network.eval()
    with torch.no_grad():
        probe_batch = buffer.sample(probe_bs)
        obs_t = probe_batch.obs.to(device)                              # [B, obs_dim]
        act_idx = probe_batch.actions.long().squeeze(-1).to(device)    # [B]
        act_onehot = F.one_hot(act_idx, num_classes=num_actions).float()

        phi_sa = q_network.encode_state_action(obs_t, act_onehot)      # [B, D]
        mean_sa_embedding = phi_sa.mean(dim=0)                         # [D]

    if as_numpy:
        return mean_sa_embedding.detach().cpu().numpy()
    return mean_sa_embedding.detach().cpu()


def extract_fixed_probe_sa_embedding(
    q_network,
    obs_probe,
    act_probe_idx,
    num_actions,
    device=None,
    as_numpy=True,
):

    if device is None:
        device = next(q_network.parameters()).device

    if not torch.is_tensor(obs_probe):
        obs_probe = torch.tensor(obs_probe, dtype=torch.float32, device=device)
    else:
        obs_probe = obs_probe.to(device).float()

    obs_probe = obs_probe.unsqueeze(0)  # [1, obs_dim]

    act_probe = F.one_hot(
        torch.tensor([act_probe_idx], device=device),
        num_classes=num_actions,
    ).float()  # [1, action_dim]

    q_network.eval()
    with torch.no_grad():
        phi_sa = q_network.encode_state_action(obs_probe, act_probe)   # [1, D]
        phi_sa = phi_sa.squeeze(0)

    if as_numpy:
        return phi_sa.detach().cpu().numpy()
    return phi_sa.detach().cpu()


def extract_sa_batch_for_isotropy(
    q_network,
    buffer,
    num_actions,
    batch_size=1024,
    device=None,
    as_numpy=True,
):
 
    if device is None:
        device = next(q_network.parameters()).device

    n_available = len(buffer)
    if n_available == 0:
        raise ValueError("Replay buffer is empty; cannot extract SA batch.")

    probe_bs = min(batch_size, n_available)

    q_network.eval()
    with torch.no_grad():
        probe_batch = buffer.sample(probe_bs)
        obs_t = probe_batch.obs.to(device)
        act_idx = probe_batch.actions.long().squeeze(-1).to(device)
        act_onehot = F.one_hot(act_idx, num_classes=num_actions).float()

        phi_sa = q_network.encode_state_action(obs_t, act_onehot)      # [B, D]

    if as_numpy:
        return phi_sa.detach().cpu().numpy()
    return phi_sa.detach().cpu()

def compute_embedding_drift(overall_results, q_net, device):
    drift = {}
    for goal, data in overall_results.items():
        if len(data["task_embeddings"]) == 0:
            continue

        # e.g. embedding right after this goal was learned
        emb_old = np.asarray(data["task_embeddings"][0], dtype=np.float32)

        # embedding now, with final goal encoder
        goal_arr = np.array(goal, dtype=np.float32)
        goal_t = torch.tensor(goal_arr, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            emb_new = q_net.encode_goal(goal_t).squeeze(0).cpu().numpy()

        # cosine similarity
        def _norm(x):
            return x / (np.linalg.norm(x) + 1e-8)

        cos = float(np.dot(_norm(emb_old), _norm(emb_new)))
        drift[goal] = cos
    return drift

def collect_weight_snapshot(
    qnet,
    goal_label,
    stage_label,
    sa_keywords_local,
    goal_keywords_local,
    max_samples_per_group=40000,
):
    sa_weights = []
    goal_weights = []

    for name, param in qnet.named_parameters():
        vals = param.detach().cpu().numpy().ravel()
        lname = name.lower()

        if any(k in lname for k in sa_keywords_local):
            sa_weights.append(vals)
        elif any(k in lname for k in goal_keywords_local):
            goal_weights.append(vals)

    sa_all = np.concatenate(sa_weights) if len(sa_weights) else np.array([], dtype=np.float32)
    goal_all = np.concatenate(goal_weights) if len(goal_weights) else np.array([], dtype=np.float32)

    rng = np.random.default_rng(0)

    def sample_vec(x, max_n):
        if x.size <= max_n:
            return x.astype(np.float32)
        idx = rng.choice(x.size, size=max_n, replace=False)
        return x[idx].astype(np.float32)

    snapshot = {
        "goal": str(goal_label),
        "stage": str(stage_label),
        "sa_sample": sample_vec(sa_all, max_samples_per_group),
        "goal_sample": sample_vec(goal_all, max_samples_per_group),
        "sa_stats": {
            "count": int(sa_all.size),
            "mean": float(sa_all.mean()) if sa_all.size else np.nan,
            "std": float(sa_all.std()) if sa_all.size else np.nan,
            "min": float(sa_all.min()) if sa_all.size else np.nan,
            "max": float(sa_all.max()) if sa_all.size else np.nan,
            "p01": float(np.quantile(sa_all, 0.01)) if sa_all.size else np.nan,
            "p50": float(np.quantile(sa_all, 0.50)) if sa_all.size else np.nan,
            "p99": float(np.quantile(sa_all, 0.99)) if sa_all.size else np.nan,
        },
        "goal_stats": {
            "count": int(goal_all.size),
            "mean": float(goal_all.mean()) if goal_all.size else np.nan,
            "std": float(goal_all.std()) if goal_all.size else np.nan,
            "min": float(goal_all.min()) if goal_all.size else np.nan,
            "max": float(goal_all.max()) if goal_all.size else np.nan,
            "p01": float(np.quantile(goal_all, 0.01)) if goal_all.size else np.nan,
            "p50": float(np.quantile(goal_all, 0.50)) if goal_all.size else np.nan,
            "p99": float(np.quantile(goal_all, 0.99)) if goal_all.size else np.nan,
        },
    }
    return snapshot

def compute_weight_change_from_snapshots(before_snap, after_snap, prefix="sa"):
    """
    Compare two snapshots produced by collect_weight_snapshot(...).

    prefix:
        "sa"   -> uses "sa_sample" and "sa_stats"
        "goal" -> uses "goal_sample" and "goal_stats"

    Returns:
        dict with vector drift from sampled weights + stats drift.
    """
    sample_key = f"{prefix}_sample"
    stats_key = f"{prefix}_stats"

    if sample_key not in before_snap or sample_key not in after_snap:
        raise KeyError(f"Missing sample key '{sample_key}' in snapshots")
    if stats_key not in before_snap or stats_key not in after_snap:
        raise KeyError(f"Missing stats key '{stats_key}' in snapshots")

    before_sample = np.asarray(before_snap[sample_key], dtype=np.float32).ravel()
    after_sample = np.asarray(after_snap[sample_key], dtype=np.float32).ravel()

    # They should usually match because you use fixed RNG and same max_samples_per_group,
    # but align defensively just in case.
    L = min(before_sample.size, after_sample.size)
    before_sample = before_sample[:L]
    after_sample = after_sample[:L]

    diff = after_sample - before_sample

    sample_l2 = float(np.linalg.norm(diff, ord=2))
    sample_l1 = float(np.linalg.norm(diff, ord=1))
    sample_linf = float(np.linalg.norm(diff, ord=np.inf)) if L > 0 else 0.0
    sample_mean_abs = float(np.mean(np.abs(diff))) if L > 0 else 0.0
    sample_rms = float(np.sqrt(np.mean(diff ** 2))) if L > 0 else 0.0

    before_stats = before_snap[stats_key]
    after_stats = after_snap[stats_key]

    stats_delta = {}
    for k in ["mean", "std", "min", "max", "p01", "p50", "p99"]:
        b = before_stats.get(k, np.nan)
        a = after_stats.get(k, np.nan)
        stats_delta[f"delta_{k}"] = float(a - b)

    return {
        "prefix": prefix,
        "num_compared": int(L),
        "sample_l2": sample_l2,
        "sample_l1": sample_l1,
        "sample_linf": sample_linf,
        "sample_mean_abs": sample_mean_abs,
        "sample_rms": sample_rms,
        "stats_delta": stats_delta,
    }

def plot_sa_encoder_changes_across_tasks(
    overall_results,
    seed=None,
):
    """
    Plot state-action encoder drift across tasks using the
    task-indexed overall_results structure.

    Expected structure:

        overall_results[task_idx]["goals_by_seed"][seed]
        overall_results[task_idx]["sa_weight_change"]

    Each entry in "sa_weight_change" should contain:

        - sample_l2
        - sample_l1
        - sample_linf
        - sample_mean_abs
        - sample_rms
        - stats_delta:
            - delta_mean
            - delta_std
            - delta_min
            - delta_max
            - delta_p01
            - delta_p50
            - delta_p99

    Args:
        overall_results:
            Dictionary indexed by task_idx.

        seed:
            Optional seed to analyse.

            If None, all available seeds are averaged together.
            If an integer is provided, only that seed is used.

    Returns:
        results:
            Dictionary containing the task indices, goals, and
            plotted statistics.
    """

    import numpy as np
    import matplotlib.pyplot as plt

    if not overall_results:
        raise ValueError(
            "overall_results is empty."
        )

    # ------------------------------------------------------------
    # Identify task indices
    # ------------------------------------------------------------
    task_indices = sorted(
        overall_results.keys()
    )

    # ------------------------------------------------------------
    # Identify available seeds
    # ------------------------------------------------------------
    available_seeds = sorted(
        {
            available_seed
            for task_idx in task_indices
            for available_seed in overall_results[
                task_idx
            ].get("goals_by_seed", {}).keys()
        }
    )

    if seed is not None:
        if seed not in available_seeds:
            raise ValueError(
                f"Seed {seed} was not found. "
                f"Available seeds: {available_seeds}"
            )

        seeds_to_use = [seed]

    else:
        seeds_to_use = available_seeds

    if len(seeds_to_use) == 0:
        raise ValueError(
            "No seeds found in "
            "overall_results[*]['goals_by_seed']."
        )

    # ------------------------------------------------------------
    # Extract task-wise data
    # ------------------------------------------------------------
    used_task_indices = []
    used_goals = []

    sa_l2 = []
    sa_rms = []
    sa_mean_abs = []
    delta_std = []
    delta_p50 = []

    for task_idx in task_indices:
        task_results = overall_results[
            task_idx
        ]

        entries = task_results.get(
            "sa_weight_change",
            [],
        )

        if entries is None:
            continue

        if len(entries) == 0:
            continue

        # --------------------------------------------------------
        # Resolve entries to the selected seeds
        # --------------------------------------------------------
        #
        # The expected new structure is a list aligned with SEEDS:
        #
        #     overall_results[task_idx][
        #         "sa_weight_change"
        #     ]
        #
        # If seed is specified and there is only one entry per
        # task, that entry is used directly. If multiple entries
        # exist, the function assumes their order follows SEEDS.
        # --------------------------------------------------------

        selected_entries = []

        if seed is None:
            selected_entries = [
                entry
                for entry in entries
                if entry is not None
            ]

        elif len(entries) == 1:
            # One recorded entry, usually the single-seed case.
            selected_entries = [
                entries[0]
            ]

        elif seed in available_seeds:
            seed_position = available_seeds.index(
                seed
            )

            if seed_position < len(entries):
                selected_entries = [
                    entries[seed_position]
                ]

        selected_entries = [
            entry
            for entry in selected_entries
            if entry is not None
        ]

        if len(selected_entries) == 0:
            continue

        # --------------------------------------------------------
        # Extract valid metrics
        # --------------------------------------------------------
        l2_values = []
        rms_values = []
        mean_abs_values = []
        delta_std_values = []
        delta_p50_values = []

        for entry in selected_entries:
            if "sample_l2" in entry:
                l2_values.append(
                    entry["sample_l2"]
                )

            if "sample_rms" in entry:
                rms_values.append(
                    entry["sample_rms"]
                )

            if "sample_mean_abs" in entry:
                mean_abs_values.append(
                    entry["sample_mean_abs"]
                )

            stats_delta = entry.get(
                "stats_delta",
                {},
            )

            if "delta_std" in stats_delta:
                delta_std_values.append(
                    stats_delta["delta_std"]
                )

            if "delta_p50" in stats_delta:
                delta_p50_values.append(
                    stats_delta["delta_p50"]
                )

        if len(l2_values) == 0:
            continue

        # --------------------------------------------------------
        # Resolve the actual goal coordinates
        # --------------------------------------------------------
        goals_by_seed = task_results.get(
            "goals_by_seed",
            {},
        )

        task_goal = None

        if seed is not None:
            task_goal = goals_by_seed.get(
                seed,
                None,
            )

        else:
            available_task_goals = [
                goals_by_seed.get(
                    available_seed,
                    None,
                )
                for available_seed in seeds_to_use
            ]

            available_task_goals = [
                goal
                for goal in available_task_goals
                if goal is not None
            ]

            if len(available_task_goals) > 0:
                task_goal = available_task_goals[0]

        if task_goal is None:
            task_goal = f"task_{task_idx}"

        used_task_indices.append(task_idx)
        used_goals.append(task_goal)

        sa_l2.append(
            np.mean(l2_values)
        )

        sa_rms.append(
            np.mean(rms_values)
        )

        sa_mean_abs.append(
            np.mean(mean_abs_values)
        )

        if len(delta_std_values) > 0:
            delta_std.append(
                np.mean(delta_std_values)
            )
        else:
            delta_std.append(
                np.nan
            )

        if len(delta_p50_values) > 0:
            delta_p50.append(
                np.mean(delta_p50_values)
            )
        else:
            delta_p50.append(
                np.nan
            )

    if len(used_task_indices) == 0:
        raise ValueError(
            "No valid sa_weight_change entries found "
            "in overall_results."
        )

    # ------------------------------------------------------------
    # Convert values to NumPy arrays
    # ------------------------------------------------------------
    used_task_indices = np.asarray(
        used_task_indices
    )

    sa_l2 = np.asarray(
        sa_l2,
        dtype=np.float32,
    )

    sa_rms = np.asarray(
        sa_rms,
        dtype=np.float32,
    )

    sa_mean_abs = np.asarray(
        sa_mean_abs,
        dtype=np.float32,
    )

    delta_std = np.asarray(
        delta_std,
        dtype=np.float32,
    )

    delta_p50 = np.asarray(
        delta_p50,
        dtype=np.float32,
    )

    x = np.arange(
        len(used_task_indices)
    )

    # ------------------------------------------------------------
    # Goal labels for x-axis
    # ------------------------------------------------------------
    goal_labels = []

    for task_idx, task_goal in zip(
        used_task_indices,
        used_goals,
    ):
        if isinstance(
            task_goal,
            (list, tuple, np.ndarray),
        ):
            goal_labels.append(
                f"task {task_idx}\n"
                f"{np.asarray(task_goal).tolist()}"
            )
        else:
            goal_labels.append(
                str(task_goal)
            )

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(15, 5),
    )

    # ------------------------------------------------------------
    # Left: sampled drift magnitude
    # ------------------------------------------------------------
    axes[0].plot(
        x,
        sa_l2,
        marker="o",
        linewidth=2.5,
        color="tab:blue",
        label="Sample L2",
    )

    axes[0].plot(
        x,
        sa_rms,
        marker="o",
        linewidth=2.5,
        color="tab:orange",
        label="Sample RMS",
    )

    axes[0].plot(
        x,
        sa_mean_abs,
        marker="o",
        linewidth=2.5,
        color="tab:green",
        label="Mean |Δ|",
    )

    axes[0].set_title(
        "SA Encoder Drift Magnitude"
    )

    axes[0].set_xlabel(
        "Task / Goal"
    )

    axes[0].set_ylabel(
        "Drift"
    )

    axes[0].set_xticks(
        x
    )

    axes[0].set_xticklabels(
        goal_labels,
        rotation=45,
        ha="right",
    )

    axes[0].grid(
        True,
        axis="y",
        linestyle="--",
        alpha=0.3,
    )

    axes[0].legend(
        loc="best"
    )

    # ------------------------------------------------------------
    # Right: distribution shift
    # ------------------------------------------------------------
    axes[1].plot(
        x,
        delta_std,
        marker="o",
        linewidth=2.5,
        color="tab:red",
        label="Δ std",
    )

    axes[1].plot(
        x,
        delta_p50,
        marker="o",
        linewidth=2.5,
        color="tab:purple",
        label="Δ p50",
    )

    axes[1].axhline(
        0.0,
        color="black",
        linestyle="--",
        linewidth=1,
        alpha=0.6,
    )

    axes[1].set_title(
        "SA Encoder Distribution Shift"
    )

    axes[1].set_xlabel(
        "Task / Goal"
    )

    axes[1].set_ylabel(
        "Change in Weight Statistics"
    )

    axes[1].set_xticks(
        x
    )

    axes[1].set_xticklabels(
        goal_labels,
        rotation=45,
        ha="right",
    )

    axes[1].grid(
        True,
        axis="y",
        linestyle="--",
        alpha=0.3,
    )

    axes[1].legend(
        loc="best"
    )

    # ------------------------------------------------------------
    # Figure title
    # ------------------------------------------------------------
    if seed is None:
        seed_label = (
            "all seeds averaged"
        )
    else:
        seed_label = f"seed={seed}"

    fig.suptitle(
        (
            "SA Encoder Changes Across Tasks "
            f"({seed_label})"
        ),
        fontsize=14,
    )

    fig.tight_layout()
    plt.show()

    return {
        "task_indices": used_task_indices,
        "goals": used_goals,
        "sa_l2": sa_l2,
        "sa_rms": sa_rms,
        "sa_mean_abs": sa_mean_abs,
        "delta_std": delta_std,
        "delta_p50": delta_p50,
    }

def extract_mean_sa_embedding_td3(
    critic,
    buffer,
    batch_size,
    device,
    as_numpy=True,
):
    if len(buffer) == 0:
        raise ValueError("Buffer is empty; cannot extract mean SA embedding.")

    sample_size = min(batch_size, len(buffer))
    batch = buffer.sample(sample_size)

    obs_t = batch.obs.float().to(device)
    act_t = batch.actions.float().to(device)

    with torch.no_grad():
        phi_sa = critic.encode_state_action(obs_t, act_t)   # [B, D]
        phi_mean = phi_sa.mean(dim=0)                       # [D]

    if as_numpy:
        return phi_mean.cpu().numpy()
    return phi_mean


def extract_fixed_probe_sa_embedding_td3(
    critic,
    obs_probe,
    act_probe,
    device,
    as_numpy=True,
):
    obs_t = torch.tensor(obs_probe, dtype=torch.float32, device=device).unsqueeze(0)
    act_t = torch.tensor(act_probe, dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        phi_sa = critic.encode_state_action(obs_t, act_t).squeeze(0)   # [D]

    if as_numpy:
        return phi_sa.cpu().numpy()
    return phi_sa


def extract_sa_batch_for_isotropy_td3(
    critic,
    buffer,
    batch_size,
    device,
    as_numpy=True,
):
    if len(buffer) == 0:
        raise ValueError("Buffer is empty; cannot extract SA batch.")

    sample_size = min(batch_size, len(buffer))
    batch = buffer.sample(sample_size)

    obs_t = batch.obs.float().to(device)
    act_t = batch.actions.float().to(device)

    with torch.no_grad():
        phi_sa = critic.encode_state_action(obs_t, act_t)   # [B, D]

    if as_numpy:
        return phi_sa.cpu().numpy()
    return phi_sa

def moving_average(values, window):
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return np.array([], dtype=np.float32)

    window = max(1, int(window))
    if values.size < window:
        return np.array([values.mean()], dtype=np.float32)

    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(values, kernel, mode="valid")


class EarlyStopperRL:

    def __init__(
        self,
        target_reward=0.90,
        ma_window=5,
        plateau_patience=8,
        min_improvement=0.01,
        max_return_var=None,
        min_steps_before_stop=0,
        max_epsilon_for_stop=None,
    ):
        self.target_reward = float(target_reward)
        self.ma_window = int(ma_window)
        self.plateau_patience = int(plateau_patience)
        self.min_improvement = float(min_improvement)
        self.max_return_var = max_return_var
        self.min_steps_before_stop = int(min_steps_before_stop)
        self.max_epsilon_for_stop = max_epsilon_for_stop

        self.eval_steps = []
        self.eval_means = []
        self.eval_vars = []

        self.best_ma = -np.inf
        self.best_step = None
        self.plateau_streak = 0

    def update(self, step, mean_return, episode_returns=None, epsilon=None):
        step = int(step)
        mean_return = float(mean_return)

        if episode_returns is None:
            ret_var = np.nan
        else:
            ep = np.asarray(episode_returns, dtype=np.float32)
            ret_var = float(np.var(ep)) if ep.size > 1 else 0.0

        self.eval_steps.append(step)
        self.eval_means.append(mean_return)
        self.eval_vars.append(ret_var)

        ma_series = moving_average(self.eval_means, self.ma_window)
        current_ma = float(ma_series[-1])

        improved = current_ma > (self.best_ma + self.min_improvement)
        if improved:
            self.best_ma = current_ma
            self.best_step = step
            self.plateau_streak = 0
        else:
            self.plateau_streak += 1

        reached_target = current_ma >= self.target_reward
        enough_steps = step >= self.min_steps_before_stop

        epsilon_ok = True
        if self.max_epsilon_for_stop is not None and epsilon is not None:
            epsilon_ok = float(epsilon) <= float(self.max_epsilon_for_stop)

        variance_ok = True
        if self.max_return_var is not None and not np.isnan(ret_var):
            variance_ok = ret_var <= float(self.max_return_var)

        plateau_ok = self.plateau_streak >= self.plateau_patience

        should_stop = (
            reached_target
            and plateau_ok
            and variance_ok
            and enough_steps
            #and epsilon_ok
        )

        info = {
            "step": step,
            "mean_return": mean_return,
            "return_var": ret_var,
            "current_ma": current_ma,
            "best_ma": float(self.best_ma),
            "best_step": self.best_step,
            "plateau_streak": int(self.plateau_streak),
            "reached_target": bool(reached_target),
            "plateau_ok": bool(plateau_ok),
            "variance_ok": bool(variance_ok),
            "enough_steps": bool(enough_steps),
            "epsilon_ok": bool(epsilon_ok),
            "should_stop": bool(should_stop),
        }
        return should_stop, info


def get_free_cells(maze_layout):
    free_cells = []
    for y, row in enumerate(maze_layout):
        for x, val in enumerate(row):
            if val == 0:
                free_cells.append((x, y))  # x, y order
    return free_cells

def sample_goals(num_goals, rng, cells, exclude=None):
    exclude = set() if exclude is None else set(exclude)
    candidates = [g for g in cells if g not in exclude]
    assert num_goals <= len(candidates), "num_goals exceeds available free cells"
    idx = rng.choice(len(candidates), size=num_goals, replace=False)
    return [candidates[i] for i in idx]

class AtariReplayBuffer:
    """
    Replay buffer for preprocessed Atari observations.

    Observations are stored compactly as uint8 in [0, 255] and converted to
    float32 when sampled. Expected observation shape is typically
    (4, 84, 84).

    The sampled batch has fields:
        obs
        actions
        rewards
        next_obs
        terminated
        truncated
        episode_id
        timestep
        indices
    """

    def __init__(
        self,
        capacity,
        obs_shape,
        device="cpu",
    ):
        self.capacity = int(capacity)
        self.obs_shape = tuple(obs_shape)
        self.device = torch.device(device)

        if self.capacity <= 0:
            raise ValueError(
                "capacity must be positive."
            )

        if len(self.obs_shape) != 3:
            raise ValueError(
                "obs_shape must be a three-dimensional "
                "image shape, for example (4, 84, 84)."
            )

        if any(
            int(dim) <= 0
            for dim in self.obs_shape
        ):
            raise ValueError(
                "All dimensions in obs_shape must be positive."
            )

        # Store image observations compactly.
        self.obs = np.zeros(
            (
                self.capacity,
                *self.obs_shape,
            ),
            dtype=np.uint8,
        )

        self.next_obs = np.zeros(
            (
                self.capacity,
                *self.obs_shape,
            ),
            dtype=np.uint8,
        )

        self.actions = np.zeros(
            (self.capacity,),
            dtype=np.int64,
        )

        self.rewards = np.zeros(
            (self.capacity, 1),
            dtype=np.float32,
        )

        self.terminated = np.zeros(
            (self.capacity, 1),
            dtype=np.float32,
        )

        self.truncated = np.zeros(
            (self.capacity, 1),
            dtype=np.float32,
        )

        self.episode_id = np.full(
            (self.capacity,),
            -1,
            dtype=np.int64,
        )

        self.timestep = np.full(
            (self.capacity,),
            -1,
            dtype=np.int64,
        )

        self.pos = 0
        self.size = 0
        self.full = False

        self.current_episode_id = 0
        self.current_timestep = 0

        self.episode_to_indices = {}

    def __len__(self):
        return self.size

    def _remove_index_from_episode_mapping(
        self,
        idx,
    ):
        old_ep = int(
            self.episode_id[idx]
        )

        if old_ep not in self.episode_to_indices:
            return

        try:
            self.episode_to_indices[
                old_ep
            ].remove(idx)

        except ValueError:
            pass

        if len(
            self.episode_to_indices[old_ep]
        ) == 0:
            del self.episode_to_indices[old_ep]

    def _prepare_observation(
        self,
        obs,
        name,
    ):
        obs = np.asarray(obs)

        if obs.shape != self.obs_shape:
            raise ValueError(
                f"Expected {name} shape "
                f"{self.obs_shape}, "
                f"got {obs.shape}."
            )

        if np.issubdtype(
            obs.dtype,
            np.floating,
        ):
            max_value = float(
                np.nanmax(obs)
            )

            # Support floating-point inputs in either
            # [0, 1] or [0, 255].
            if max_value <= 1.0:
                obs = obs * 255.0

        obs = np.nan_to_num(
            obs,
            nan=0.0,
            posinf=255.0,
            neginf=0.0,
        )

        obs = np.clip(
            obs,
            0.0,
            255.0,
        ).astype(
            np.uint8,
            copy=False,
        )

        return obs

    def add_transition(
        self,
        obs,
        action,
        reward,
        next_obs,
        terminated,
        truncated=False,
        episode_id=None,
        timestep=None,
    ):
        obs = self._prepare_observation(
            obs,
            name="obs",
        )

        next_obs = self._prepare_observation(
            next_obs,
            name="next_obs",
        )

        idx = self.pos

        if self.full:
            self._remove_index_from_episode_mapping(
                idx
            )

        if episode_id is None:
            ep_id = self.current_episode_id
        else:
            ep_id = int(episode_id)

        if timestep is None:
            t = self.current_timestep
        else:
            t = int(timestep)

        self.obs[idx] = obs
        self.next_obs[idx] = next_obs

        self.actions[idx] = int(action)

        self.rewards[idx, 0] = float(
            reward
        )

        self.terminated[idx, 0] = float(
            terminated
        )

        self.truncated[idx, 0] = float(
            truncated
        )

        self.episode_id[idx] = ep_id
        self.timestep[idx] = t

        if ep_id not in self.episode_to_indices:
            self.episode_to_indices[ep_id] = []

        self.episode_to_indices[
            ep_id
        ].append(idx)

        self.pos = (
            self.pos + 1
        ) % self.capacity

        if self.size < self.capacity:
            self.size += 1
        else:
            self.full = True

        self.current_episode_id = max(
            self.current_episode_id,
            ep_id,
        )

        self.current_timestep = t + 1

    def end_episode(self):
        self.current_episode_id += 1
        self.current_timestep = 0

    def sample(
        self,
        batch_size,
    ):
        if self.size <= 0:
            raise RuntimeError(
                "Cannot sample from an empty buffer."
            )

        batch_size = int(batch_size)

        if batch_size <= 0:
            raise ValueError(
                "batch_size must be positive."
            )

        indices = np.random.randint(
            low=0,
            high=self.size,
            size=batch_size,
        )

        return ReplayBatch(
            obs=torch.as_tensor(
                self.obs[indices],
                device=self.device,
            ).float(),

            actions=torch.as_tensor(
                self.actions[indices],
                device=self.device,
            ),

            rewards=torch.as_tensor(
                self.rewards[indices],
                device=self.device,
            ),

            next_obs=torch.as_tensor(
                self.next_obs[indices],
                device=self.device,
            ).float(),

            terminated=torch.as_tensor(
                self.terminated[indices],
                device=self.device,
            ),

            truncated=torch.as_tensor(
                self.truncated[indices],
                device=self.device,
            ),

            episode_id=torch.as_tensor(
                self.episode_id[indices],
                device=self.device,
            ),

            timestep=torch.as_tensor(
                self.timestep[indices],
                device=self.device,
            ),

            indices=torch.as_tensor(
                indices,
                device=self.device,
            ),
        )

    def stats(self):
        return {
            "size": self.size,
            "capacity": self.capacity,
            "num_episodes": len(
                self.episode_to_indices
            ),
            "current_episode_id": (
                self.current_episode_id
            ),
            "current_timestep": (
                self.current_timestep
            ),
            "obs_shape": self.obs_shape,
            "position": self.pos,
            "full": self.full,
            "storage_dtype": str(
                self.obs.dtype
            ),
        }

@dataclass
class HERBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_obs: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    episode_id: torch.Tensor
    timestep: torch.Tensor
    indices: torch.Tensor
    goals: torch.Tensor
    achieved_goals: torch.Tensor
    next_achieved_goals: torch.Tensor
    goal_indices: torch.Tensor

    def __len__(self):
        return self.obs.shape[0]


@dataclass
class HERBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_obs: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    episode_id: torch.Tensor
    timestep: torch.Tensor
    indices: torch.Tensor
    goals: torch.Tensor
    achieved_goals: torch.Tensor
    next_achieved_goals: torch.Tensor
    goal_indices: torch.Tensor

    def __len__(self):
        return self.obs.shape[0]


class HERTrajectoryReplayBuffer(TrajectoryReplayBuffer):
    def __init__(
        self,
        capacity,
        obs_dim,
        action_dim,
        goal_dim=3,
        device="cpu",
        goal_radius=0.05,
    ):
        super().__init__(capacity, obs_dim, action_dim, device=device)
        self.goal_dim = int(goal_dim)
        self.goal_radius = float(goal_radius)

        self.achieved_goals = np.zeros((capacity, goal_dim), dtype=np.float32)
        self.desired_goals = np.zeros((capacity, goal_dim), dtype=np.float32)
        self.next_achieved_goals = np.zeros((capacity, goal_dim), dtype=np.float32)

    def _compute_reward(self, achieved, desired):
        dist = np.linalg.norm(achieved - desired)
        return 1.0 if dist <= self.goal_radius else 0.0

    def add_transition(
        self,
        obs,
        action,
        reward,
        next_obs,
        terminated,
        truncated=False,
        episode_id=None,
        timestep=None,
        desired_goal=None,
        achieved_goal=None,
        next_achieved_goal=None,
    ):
        idx = self.pos

        if self.full:
            self._remove_index_from_episode_mapping(idx)

        if episode_id is None:
            ep_id = self.current_episode_id
            self.current_episode_id += 1
        else:
            ep_id = int(episode_id)
            if ep_id >= self.current_episode_id:
                self.current_episode_id = ep_id + 1

        t = 0 if timestep is None else int(timestep)

        obs = np.asarray(obs, dtype=np.float32).ravel()
        next_obs = np.asarray(next_obs, dtype=np.float32).ravel()
        action = np.asarray(action, dtype=np.float32).ravel()

        if achieved_goal is None:
            raise ValueError("achieved_goal is required for HER buffer")
        if next_achieved_goal is None:
            raise ValueError("next_achieved_goal is required for HER buffer")
        if desired_goal is None:
            raise ValueError("desired_goal is required for HER buffer")

        achieved_goal = np.asarray(achieved_goal, dtype=np.float32).ravel()
        next_achieved_goal = np.asarray(next_achieved_goal, dtype=np.float32).ravel()
        desired_goal = np.asarray(desired_goal, dtype=np.float32).ravel()

        self.obs[idx] = obs
        self.actions[idx] = action
        self.rewards[idx] = np.asarray([reward], dtype=np.float32)
        self.next_obs[idx] = next_obs
        self.terminated[idx] = np.asarray([terminated], dtype=np.float32)
        self.truncated[idx] = np.asarray([truncated], dtype=np.float32)

        self.episode_id[idx] = ep_id
        self.timestep[idx] = t

        self.achieved_goals[idx] = achieved_goal
        self.desired_goals[idx] = desired_goal
        self.next_achieved_goals[idx] = next_achieved_goal

        if ep_id not in self.episode_to_indices:
            self.episode_to_indices[ep_id] = []
        self.episode_to_indices[ep_id].append(idx)

        self.pos = (self.pos + 1) % self.capacity

        if self.size < self.capacity:
            self.size += 1
        else:
            self.full = True

    def sample(self, batch_size):
        return super().sample(batch_size)

    def sample_future_goal_pairs(self, batch_size, min_k=1, max_k=None, gamma=0.99):
        assert self.size > 0, "Buffer is empty"
        assert 0.0 <= gamma < 1.0, "gamma must be in [0, 1)"

        valid_indices = []
        future_goal_indices = []
        tries = 0
        max_tries = batch_size * 200

        while len(valid_indices) < batch_size and tries < max_tries:
            idx = np.random.randint(0, self.size)
            ep_id = self.episode_id[idx]
            t = self.timestep[idx]

            if ep_id == -1 or ep_id not in self.episode_to_indices:
                tries += 1
                continue

            ep_idxs = self.episode_to_indices[ep_id]
            ep_len = len(ep_idxs)

            if t >= ep_len - 1:
                tries += 1
                continue

            max_valid_k = ep_len - 1 - t
            if max_k is not None:
                max_valid_k = min(max_valid_k, max_k)

            if max_valid_k < min_k:
                tries += 1
                continue

            ks = np.arange(min_k, max_valid_k + 1, dtype=np.int64)

            if gamma == 0.0:
                probs = np.zeros_like(ks, dtype=np.float64)
                probs[0] = 1.0
            else:
                log_weights = (ks - 1) * np.log(gamma)
                log_weights -= np.max(log_weights)
                weights = np.exp(log_weights)
                probs = weights / weights.sum()

            k = np.random.choice(ks, p=probs)
            future_t = t + k
            future_idx = ep_idxs[future_t]

            valid_indices.append(idx)
            future_goal_indices.append(future_idx)
            tries += 1

        assert len(valid_indices) == batch_size, (
            f"Could only sample {len(valid_indices)} valid transitions out of "
            f"requested {batch_size}."
        )

        idxs = np.array(valid_indices, dtype=np.int64)
        g_idxs = np.array(future_goal_indices, dtype=np.int64)

        return {
            "obs": torch.tensor(self.obs[idxs], device=self.device),
            "actions": torch.tensor(self.actions[idxs], device=self.device),
            "next_obs": torch.tensor(self.next_obs[idxs], device=self.device),
            "goals": torch.tensor(self.next_achieved_goals[g_idxs], device=self.device),
            "rewards": torch.tensor(self.rewards[idxs], device=self.device),
            "terminated": torch.tensor(self.terminated[idxs], device=self.device),
            "truncated": torch.tensor(self.truncated[idxs], device=self.device),
            "episode_id": torch.tensor(self.episode_id[idxs], device=self.device),
            "timestep": torch.tensor(self.timestep[idxs], device=self.device),
            "future_timestep": torch.tensor(self.timestep[g_idxs], device=self.device),
            "indices": torch.tensor(idxs, device=self.device),
            "goal_indices": torch.tensor(g_idxs, device=self.device),
        }

    def sample_her_batch(self, batch_size, her_ratio=0.8):
        assert self.size > 0, "Buffer is empty"

        idxs = np.random.randint(0, self.size, size=batch_size)
        her_mask = np.random.rand(batch_size) < her_ratio

        goals = self.desired_goals[idxs].copy()
        rewards = self.rewards[idxs].copy()

        for i in range(batch_size):
            if not her_mask[i]:
                continue

            ep_id = self.episode_id[idxs[i]]
            t = self.timestep[idxs[i]]

            if ep_id == -1 or ep_id not in self.episode_to_indices:
                continue

            ep_idxs = self.episode_to_indices[ep_id]
            future_candidates = [j for j in ep_idxs if self.timestep[j] > t]

            if len(future_candidates) == 0:
                continue

            g_idx = np.random.choice(future_candidates)
            goals[i] = self.next_achieved_goals[g_idx]

            achieved = self.next_achieved_goals[idxs[i]]
            rewards[i] = np.array([self._compute_reward(achieved, goals[i])], dtype=np.float32)

        return {
            "obs": torch.tensor(self.obs[idxs], device=self.device),
            "actions": torch.tensor(self.actions[idxs], device=self.device),
            "next_obs": torch.tensor(self.next_obs[idxs], device=self.device),
            "goals": torch.tensor(goals, device=self.device),
            "rewards": torch.tensor(rewards, device=self.device),
            "terminated": torch.tensor(self.terminated[idxs], device=self.device),
            "truncated": torch.tensor(self.truncated[idxs], device=self.device),
            "episode_id": torch.tensor(self.episode_id[idxs], device=self.device),
            "timestep": torch.tensor(self.timestep[idxs], device=self.device),
            "indices": torch.tensor(idxs, device=self.device),
        }

    def stats(self):
        base = super().stats()
        base.update(
            {
                "goal_dim": self.goal_dim,
                "goal_radius": self.goal_radius,
            }
        )
        return base


def _as_float_vector(x):
    """
    Convert a task descriptor or encoder output into a 1-D
    NumPy float vector.
    """

    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()

    x = np.asarray(
        x,
        dtype=np.float32,
    )

    return x.reshape(-1)


def _cosine_similarity_np(x, y):
    x = _as_float_vector(x)
    y = _as_float_vector(y)

    x_norm = np.linalg.norm(x)
    y_norm = np.linalg.norm(y)

    if x_norm < 1e-8 or y_norm < 1e-8:
        return 0.0

    return float(
        np.dot(x, y)
        / (x_norm * y_norm)
    )


def _negative_l2_similarity(x, y):
    """
    Larger values mean more similar.
    """

    x = _as_float_vector(x)
    y = _as_float_vector(y)

    return -float(
        np.linalg.norm(x - y)
    )


def encode_task_for_similarity(
    raw_task,
    task_type,
    task_similarity_encoder=None,
    device=None,
):
    """
    Convert a raw task into a representation used only
    for similarity retrieval.

    Supported task types:

        "maze":
            raw_task is a coordinate such as (x, y)

        "vector":
            raw_task is a numeric descriptor

        "text":
            task_similarity_encoder must encode text

        "custom":
            task_similarity_encoder must handle raw_task
    """

    task_type = task_type.lower()

    if task_type == "maze":
        return _as_float_vector(raw_task)

    if task_type == "vector":
        return _as_float_vector(raw_task)

    if task_type in {"text", "prompt", "nlp", "custom"}:
        if task_similarity_encoder is None:
            raise ValueError(
                f"A task_similarity_encoder is required "
                f"for task_type='{task_type}'."
            )

        encoded = task_similarity_encoder(
            raw_task
        )

        return _as_float_vector(encoded)

    raise ValueError(
        f"Unknown task_type: {task_type}"
    )


def compute_task_similarity(
    new_raw_task,
    old_raw_task,
    task_type,
    task_similarity_encoder=None,
    maze_distance_fn=None,
):
    """
    Return a similarity score where larger means more similar.

    This function compares only the new task against one
    previously learned task.
    """

    task_type = task_type.lower()

    if task_type == "maze":

        if maze_distance_fn is not None:
            distance = maze_distance_fn(
                new_raw_task,
                old_raw_task,
            )

            return -float(distance)

        # Baseline: Euclidean distance in maze coordinates.
        return _negative_l2_similarity(
            new_raw_task,
            old_raw_task,
        )

    if task_type == "vector":
        new_vector = encode_task_for_similarity(
            new_raw_task,
            task_type="vector",
        )

        old_vector = encode_task_for_similarity(
            old_raw_task,
            task_type="vector",
        )

        return _cosine_similarity_np(
            new_vector,
            old_vector,
        )

    if task_type in {
        "text",
        "prompt",
        "nlp",
        "custom",
    }:
        new_vector = encode_task_for_similarity(
            new_raw_task,
            task_type=task_type,
            task_similarity_encoder=task_similarity_encoder,
        )

        old_vector = encode_task_for_similarity(
            old_raw_task,
            task_type=task_type,
            task_similarity_encoder=task_similarity_encoder,
        )

        return _cosine_similarity_np(
            new_vector,
            old_vector,
        )

    raise ValueError(
        f"Unknown task_type: {task_type}"
    )

def retrieve_similar_task_embeddings(
    new_raw_task,
    task_type,
    task_memory,
    top_k=3,
    similarity_temperature=1.0,
    minimum_similarity=None,
    task_similarity_encoder=None,
    maze_distance_fn=None,
):
    """
    Retrieve the most similar tasks from task_memory and
    construct a weighted prototype from their learned
    task embeddings.

    Only previously stored tasks are considered.

    Returns:
        None if task_memory is empty or no task passes
        minimum_similarity.

        Otherwise:
        {
            "prototype": np.ndarray,
            "indices": np.ndarray,
            "similarities": np.ndarray,
            "weights": np.ndarray,
            "retrieved_items": list,
        }
    """

    if task_memory is None:
        return None

    if len(task_memory) == 0:
        return None

    similarities = []

    for item in task_memory:

        old_raw_task = item["raw_task"]

        old_task_type = item.get(
            "task_type",
            task_type,
        )

        if old_task_type != task_type:
            raise ValueError(
                "New task type and stored task type "
                "must match for this retrieval call."
            )

        similarity = compute_task_similarity(
            new_raw_task=new_raw_task,
            old_raw_task=old_raw_task,
            task_type=task_type,
            task_similarity_encoder=(
                task_similarity_encoder
            ),
            maze_distance_fn=maze_distance_fn,
        )

        similarities.append(similarity)

    similarities = np.asarray(
        similarities,
        dtype=np.float32,
    )

    if minimum_similarity is not None:
        valid_indices = np.where(
            similarities >= minimum_similarity
        )[0]

        if len(valid_indices) == 0:
            return None

    else:
        valid_indices = np.arange(
            len(task_memory)
        )

    # Sort only previous tasks that passed the threshold.
    ranked_indices = valid_indices[
        np.argsort(
            similarities[valid_indices]
        )[::-1]
    ]

    selected_indices = ranked_indices[
        :top_k
    ]

    selected_similarities = similarities[
        selected_indices
    ]

    temperature = max(
        float(similarity_temperature),
        1e-8,
    )

    # Larger similarity means higher retrieval weight.
    logits = selected_similarities / temperature
    logits = logits - logits.max()

    weights = np.exp(logits)
    weights = weights / (
        weights.sum() + 1e-8
    )

    selected_embeddings = np.stack(
        [
            _as_float_vector(
                task_memory[i]["task_embedding"]
            )
            for i in selected_indices
        ],
        axis=0,
    )

    prototype = (
        weights[:, None]
        * selected_embeddings
    ).sum(axis=0)

    prototype_norm = np.linalg.norm(
        prototype
    )

    if prototype_norm > 1e-8:
        prototype = (
            prototype
            / prototype_norm
        )

    return {
        "prototype": prototype.astype(
            np.float32
        ),
        "indices": selected_indices,
        "similarities": selected_similarities,
        "weights": weights.astype(
            np.float32
        ),
        "retrieved_items": [
            task_memory[i]
            for i in selected_indices
        ],
    }


def evaluate_model(
    model,
    env,
    num_episodes=100,
    goal=None,
):
    successes = []
    returns = []
    lengths = []
    minimum_distances = []

    for episode in range(num_episodes):
        obs, info = env.reset(
            seed=1_000 + episode,
        )

        terminated = False
        truncated = False

        episode_return = 0.0
        episode_length = 0
        minimum_distance = float("inf")
        episode_success = False

        while not (terminated or truncated):
            action, _ = model.predict(
                obs,
                deterministic=True,
            )

            obs, reward, terminated, truncated, info = (
                env.step(action)
            )

            episode_return += float(reward)
            episode_length += 1

            if "goal_dist" in info:
                distance = float(
                    info["goal_dist"]
                )
            else:
                distance = float(
                    np.linalg.norm(
                        np.asarray(
                            obs["achieved_goal"],
                            dtype=np.float32,
                        )
                        - goal
                    )
                )

            minimum_distance = min(
                minimum_distance,
                distance,
            )

            if "is_success" in info:
                episode_success = bool(
                    info["is_success"]
                )
            elif distance <= 0.05:
                episode_success = True

        successes.append(
            float(episode_success)
        )
        returns.append(episode_return)
        lengths.append(episode_length)
        minimum_distances.append(
            minimum_distance
        )

    return {
        "goal": goal,
        "success_rate": float(
            np.mean(successes)
        ),
        "mean_return": float(
            np.mean(returns)
        ),
        "return_std": float(
            np.std(returns)
        ),
        "mean_episode_length": float(
            np.mean(lengths)
        ),
        "mean_minimum_distance": float(
            np.mean(minimum_distances)
        ),
        "best_minimum_distance": float(
            np.min(minimum_distances)
        ),
    }

def configure_factorised_learning_rates(
    model,
    actor_lr=1e-3,
    phi_lr=1e-4,
    goal_lr=1e-3,
):
    critic = model.critic

    phi_params = []
    goal_params = []

    for name, parameter in (
        critic.named_parameters()
    ):
        if "state_action_encoder" in name:
            phi_params.append(parameter)

        elif "goal_encoder" in name:
            goal_params.append(parameter)

        else:
            raise RuntimeError(
                "Unexpected critic parameter: "
                f"{name}"
            )

    if len(phi_params) == 0:
        raise RuntimeError(
            "No parameters matched "
            "'state_action_encoder'. "
            "Check critic parameter names."
        )

    if len(goal_params) == 0:
        raise RuntimeError(
            "No parameters matched "
            "'goal_encoder'. "
            "Check critic parameter names."
        )

    # Replace the optimiser rather than mutating its
    # existing groups. This prevents duplicated groups
    # if the function is accidentally called twice.
    critic.optimizer = torch.optim.Adam(
        [
            {
                "params": phi_params,
                "lr": phi_lr,
            },
            {
                "params": goal_params,
                "lr": goal_lr,
            },
        ],
        eps=1e-5,
    )

    # The actor still uses one optimiser group.
    for group in model.actor.optimizer.param_groups:
        group["lr"] = actor_lr

    print(
        "Configured factorised learning rates:"
    )

    for idx, group in enumerate(
        critic.optimizer.param_groups
    ):
        print(
            f"critic group {idx}: "
            f"lr={group['lr']}, "
            f"n_params={len(group['params'])}"
        )

    for idx, group in enumerate(
        model.actor.optimizer.param_groups
    ):
        print(
            f"actor group {idx}: "
            f"lr={group['lr']}"
        )

def analyse_seen_goal_embeddings(
    q_network,
    seen_goals,
    device=None,
    goal_labels=None,
    plot=True,
    title="Goal encoder analysis",
):
    """
    Analyse goal-encoder embeddings for all goals seen so far.

    Args:
        q_network:
            Network exposing q_network.encode_goal(goals).

        seen_goals:
            Goal array or tensor with shape:
                [num_seen_goals, goal_dim]
            A single goal with shape [goal_dim] is also accepted.

        device:
            Device used for goal encoding. If None, inferred from
            q_network parameters.

        goal_labels:
            Optional labels used only for printing. Should have length
            num_seen_goals.

        plot:
            If True, display the diagnostic plots.

        title:
            Figure title prefix.

    Returns:
        results:
            Dictionary containing embeddings and diagnostics.
    """

    import numpy as np
    import torch
    import matplotlib.pyplot as plt

    if device is None:
        try:
            device = next(q_network.parameters()).device
        except StopIteration:
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

    # Convert goals to a tensor.
    if isinstance(seen_goals, torch.Tensor):
        goals_t = seen_goals.detach().clone().to(
            device=device,
            dtype=torch.float32,
        )
    else:
        goals_t = torch.tensor(
            np.asarray(seen_goals),
            dtype=torch.float32,
            device=device,
        )

    if goals_t.ndim == 1:
        goals_t = goals_t.unsqueeze(0)

    if goals_t.ndim != 2:
        raise ValueError(
            "seen_goals must have shape "
            "[num_seen_goals, goal_dim]. "
            f"Received shape {tuple(goals_t.shape)}."
        )

    num_goals = goals_t.shape[0]

    if num_goals < 1:
        raise ValueError(
            "seen_goals must contain at least one goal."
        )

    if goal_labels is not None:
        if len(goal_labels) != num_goals:
            raise ValueError(
                "goal_labels must have the same length as seen_goals."
            )

    was_training = q_network.training
    q_network.eval()

    try:
        with torch.no_grad():
            psi_t = q_network.encode_goal(goals_t)

    finally:
        if was_training:
            q_network.train()

    if psi_t.ndim != 2:
        psi_t = psi_t.reshape(
            psi_t.shape[0],
            -1,
        )

    psi = psi_t.detach().cpu().numpy()

    num_goals, goal_dim = psi.shape

    print("=" * 70)
    print(title)
    print("=" * 70)
    print(f"Number of seen goals: {num_goals}")
    print(f"Goal embedding shape: {psi.shape}")

    if goal_labels is not None:
        print(f"Goal labels: {list(goal_labels)}")

    # ---------------------------------------------------------------
    # Basic embedding statistics
    # ---------------------------------------------------------------
    embedding_norms = np.linalg.norm(
        psi,
        axis=1,
    )

    dimension_means = psi.mean(axis=0)
    dimension_stds = psi.std(
        axis=0,
        ddof=0,
    )

    print("\nEmbedding statistics")
    print(
        "Embedding norm: "
        f"mean={embedding_norms.mean():.6f}, "
        f"std={embedding_norms.std():.6f}, "
        f"min={embedding_norms.min():.6f}, "
        f"max={embedding_norms.max():.6f}"
    )

    print(
        "Per-dimension standard deviation: "
        f"mean={dimension_stds.mean():.6f}, "
        f"std={dimension_stds.std():.6f}, "
        f"min={dimension_stds.min():.6f}, "
        f"max={dimension_stds.max():.6f}"
    )

    # ---------------------------------------------------------------
    # Pairwise distances and cosine similarities
    # ---------------------------------------------------------------
    if num_goals >= 2:
        pairwise_distances = []

        for i in range(num_goals):
            for j in range(i + 1, num_goals):
                distance = np.linalg.norm(
                    psi[i] - psi[j]
                )
                pairwise_distances.append(distance)

        pairwise_distances = np.asarray(
            pairwise_distances,
            dtype=np.float32,
        )

        print("\nPairwise goal distances")
        print(
            "Distance: "
            f"mean={pairwise_distances.mean():.6f}, "
            f"std={pairwise_distances.std():.6f}, "
            f"min={pairwise_distances.min():.6f}, "
            f"max={pairwise_distances.max():.6f}"
        )

        psi_norm = np.linalg.norm(
            psi,
            axis=1,
            keepdims=True,
        )

        psi_normalised = psi / np.clip(
            psi_norm,
            a_min=1e-8,
            a_max=None,
        )

        cosine_matrix = psi_normalised @ psi_normalised.T

        upper_triangle = np.triu_indices(
            num_goals,
            k=1,
        )

        pairwise_cosines = cosine_matrix[
            upper_triangle
        ]

        print(
            "Pairwise cosine similarity: "
            f"mean={pairwise_cosines.mean():.6f}, "
            f"std={pairwise_cosines.std():.6f}, "
            f"min={pairwise_cosines.min():.6f}, "
            f"max={pairwise_cosines.max():.6f}"
        )

    else:
        pairwise_distances = np.empty(
            0,
            dtype=np.float32,
        )

        pairwise_cosines = np.empty(
            0,
            dtype=np.float32,
        )

        cosine_matrix = np.ones(
            (1, 1),
            dtype=np.float32,
        )

        print(
            "\nPairwise distances and cosine similarities "
            "require at least two goals."
        )

    # ---------------------------------------------------------------
    # Centre embeddings across seen goals
    # ---------------------------------------------------------------
    psi_centered = psi - psi.mean(
        axis=0,
        keepdims=True,
    )

    # SVD rank is bounded by min(num_goals - 1, goal_dim).
    if num_goals >= 2:
        _, singular_values, _ = np.linalg.svd(
            psi_centered,
            full_matrices=False,
        )
    else:
        singular_values = np.zeros(
            min(num_goals, goal_dim),
            dtype=np.float32,
        )

    singular_values_squared = singular_values ** 2
    total_variance = singular_values_squared.sum()

    if total_variance > 1e-12:
        explained_variance = (
            singular_values_squared
            / total_variance
        )

        cumulative_variance = explained_variance.cumsum()
    else:
        explained_variance = np.zeros_like(
            singular_values_squared
        )

        cumulative_variance = np.zeros_like(
            explained_variance
        )

    print("\nGoal embedding singular-value analysis")

    if singular_values.size > 0:
        print(
            "Top singular values:",
            singular_values[:5],
        )

        print(
            "Cumulative explained variance "
            "(first 5 components):",
            cumulative_variance[:5],
        )
    else:
        print("No singular values available.")

    # ---------------------------------------------------------------
    # Numerical rank
    # ---------------------------------------------------------------
    if singular_values.size == 0:
        numerical_rank = 0

    elif singular_values[0] <= 1e-12:
        numerical_rank = 0

    else:
        numerical_rank = int(
            np.sum(
                singular_values
                > singular_values[0] * 1e-5
            )
        )

    # ---------------------------------------------------------------
    # Effective rank
    # ---------------------------------------------------------------
    nonzero_explained = explained_variance[
        explained_variance > 1e-12
    ]

    if nonzero_explained.size > 0:
        effective_rank = float(
            np.exp(
                -np.sum(
                    nonzero_explained
                    * np.log(nonzero_explained)
                )
            )
        )
    else:
        effective_rank = 0.0

    maximum_possible_rank = min(
        max(num_goals - 1, 0),
        goal_dim,
    )

    print(f"Numerical rank: {numerical_rank}")
    print(f"Effective rank: {effective_rank:.6f}")
    print(
        "Maximum possible centred rank: "
        f"{maximum_possible_rank}"
    )

    # ---------------------------------------------------------------
    # Covariance diagnostics
    # ---------------------------------------------------------------
    if num_goals >= 2:
        covariance = np.cov(
            psi,
            rowvar=False,
            ddof=1,
        )

        covariance = np.atleast_2d(covariance)

        covariance_diagonal = np.diag(covariance)

        covariance_off_diagonal = (
            covariance
            - np.diag(covariance_diagonal)
        )

        covariance_frobenius_norm = np.linalg.norm(
            covariance,
            ord="fro",
        )

        off_diagonal_frobenius_norm = np.linalg.norm(
            covariance_off_diagonal,
            ord="fro",
        )

        if covariance_frobenius_norm > 1e-12:
            covariance_off_diagonal_ratio = (
                off_diagonal_frobenius_norm
                / covariance_frobenius_norm
            )
        else:
            covariance_off_diagonal_ratio = 0.0

        print("\nCovariance diagnostics")
        print(
            "Covariance diagonal: "
            f"mean={covariance_diagonal.mean():.6f}, "
            f"std={covariance_diagonal.std():.6f}, "
            f"min={covariance_diagonal.min():.6f}, "
            f"max={covariance_diagonal.max():.6f}"
        )

        print(
            "Off-diagonal covariance Frobenius ratio: "
            f"{covariance_off_diagonal_ratio:.6f}"
        )

    else:
        covariance = None
        covariance_diagonal = None
        covariance_off_diagonal_ratio = None

    # ---------------------------------------------------------------
    # Collapse warnings
    # ---------------------------------------------------------------
    print("\nCollapse checks")

    if num_goals < 2:
        print(
            "WARNING: At least two distinct goals are required "
            "to assess goal collapse."
        )

    else:
        distance_scale = np.mean(
            np.linalg.norm(psi, axis=1)
        )

        distance_threshold = max(
            1e-6,
            1e-3 * distance_scale,
        )

        if pairwise_distances.max() < distance_threshold:
            print(
                "WARNING: Goal embeddings are almost identical. "
                "Possible complete collapse."
            )
        else:
            print(
                "Pairwise-distance check: "
                "embeddings are not completely collapsed."
            )

        if numerical_rank <= 1 and maximum_possible_rank > 1:
            print(
                "WARNING: Goal embeddings have effective rank "
                "close to one."
            )
        else:
            print(
                "Rank check: no clear rank-one collapse detected."
            )

        low_variance_dimensions = int(
            np.sum(
                dimension_stds
                < max(
                    1e-6,
                    1e-3 * dimension_stds.mean(),
                )
            )
        )

        print(
            "Near-zero-variance dimensions: "
            f"{low_variance_dimensions}/{goal_dim}"
        )

    # ---------------------------------------------------------------
    # Plot diagnostics
    # ---------------------------------------------------------------
    if plot:
        num_plots = 4 if num_goals >= 2 else 3

        fig, axes = plt.subplots(
            1,
            num_plots,
            figsize=(6 * num_plots, 4),
        )

        if num_plots == 1:
            axes = [axes]

        # Plot 1: cumulative explained variance.
        axes[0].plot(
            np.arange(
                1,
                len(cumulative_variance) + 1,
            ),
            cumulative_variance,
            marker="o",
            markersize=4,
        )

        axes[0].set_xlabel("Singular-value component")
        axes[0].set_ylabel("Cumulative variance explained")
        axes[0].set_title("Goal embedding spectrum")
        axes[0].set_ylim(0.0, 1.05)
        axes[0].grid(alpha=0.3)

        # Plot 2: per-dimension standard deviation.
        axes[1].bar(
            np.arange(goal_dim),
            dimension_stds,
            color="tab:orange",
        )

        axes[1].set_xlabel("Goal-embedding dimension")
        axes[1].set_ylabel("Standard deviation across goals")
        axes[1].set_title("Dimension-wise variation")
        axes[1].grid(
            axis="y",
            alpha=0.3,
        )

        # Plot 3: pairwise distance histogram.
        if num_goals >= 2:
            axes[2].hist(
                pairwise_distances,
                bins=min(30, max(5, len(pairwise_distances))),
                alpha=0.75,
                color="tab:green",
            )

            axes[2].set_xlabel("Pairwise Euclidean distance")
            axes[2].set_ylabel("Count")
            axes[2].set_title("Goal separation")
            axes[2].grid(alpha=0.3)

            # Plot 4: cosine similarity matrix.
            image = axes[3].imshow(
                cosine_matrix,
                vmin=-1.0,
                vmax=1.0,
                cmap="coolwarm",
                aspect="auto",
            )

            axes[3].set_xlabel("Goal index")
            axes[3].set_ylabel("Goal index")
            axes[3].set_title("Goal cosine similarity")

            fig.colorbar(
                image,
                ax=axes[3],
                fraction=0.046,
                pad=0.04,
            )

        else:
            axes[2].axis("off")
            axes[2].text(
                0.5,
                0.5,
                "At least two goals\n"
                "are required for\n"
                "pairwise diagnostics.",
                ha="center",
                va="center",
                transform=axes[2].transAxes,
            )

        fig.suptitle(
            (
                f"{title} | "
                f"numerical rank={numerical_rank}, "
                f"effective rank={effective_rank:.2f}"
            ),
            y=1.03,
        )

        plt.tight_layout()
        plt.show()

    return {
        "goals": goals_t.detach().cpu(),
        "psi": psi,
        "embedding_norms": embedding_norms,
        "dimension_means": dimension_means,
        "dimension_stds": dimension_stds,
        "pairwise_distances": pairwise_distances,
        "pairwise_cosines": pairwise_cosines,
        "cosine_matrix": cosine_matrix,
        "singular_values": singular_values,
        "explained_variance": explained_variance,
        "cumulative_variance": cumulative_variance,
        "numerical_rank": numerical_rank,
        "effective_rank": effective_rank,
        "maximum_possible_rank": maximum_possible_rank,
        "covariance": covariance,
        "covariance_diagonal": covariance_diagonal,
        "covariance_off_diagonal_ratio": (
            covariance_off_diagonal_ratio
        ),
    }

def inspect_raw_phi_norms(
    q_network,
    batch,
):
    actions = (
        batch.actions
        .long()
        .view(-1)
    )

    action_onehot = F.one_hot(
        actions,
        num_classes=q_network.num_actions,
    ).to(
        dtype=batch.obs.dtype,
        device=batch.obs.device,
    )

    sa = torch.cat(
        [
            batch.obs,
            action_onehot,
        ],
        dim=-1,
    )

    with torch.no_grad():
        raw_phi = q_network.sa_encoder(
            sa
        )

        raw_phi_norms = torch.linalg.vector_norm(
            raw_phi,
            ord=2,
            dim=-1,
        )

    return {
        "mean": raw_phi_norms.mean().item(),
        "std": raw_phi_norms.std(
            unbiased=False
        ).item(),
        "min": raw_phi_norms.min().item(),
        "max": raw_phi_norms.max().item(),
    }

def keep_last_fraction_of_buffer(
    buffer,
    keep_fraction: float = 0.2,
):
    """
    Return a new buffer containing only the last `keep_fraction` of transitions
    from the given task buffer. No feature-based selection; purely based on
    insertion order.

    Parameters
    ----------
    buffer : TrajectoryReplayBufferDiscrete
        The buffer for a single task.
    keep_fraction : float
        Fraction of samples to keep (e.g., 0.2 to keep 20%).

    Returns
    -------
    new_buffer : TrajectoryReplayBufferDiscrete
        A new buffer containing only the selected subset of transitions.
    """
    N = len(buffer)
    if N == 0:
        # Return an empty buffer with same spec
        return TrajectoryReplayBufferDiscrete(
            capacity=buffer.capacity,
            obs_dim=buffer.obs_dim,
            action_dim=buffer.action_dim,
            device=buffer.device,
        )

    keep_count = max(1, int(N * keep_fraction))
    start_idx = N - keep_count  # logical index of first transition to keep

    # ---------------------------------------------------------
    # Map logical indices [start_idx, N) to physical indices in circular buffer
    # ---------------------------------------------------------

    # Logical order: 0,1,2,...,N-1 corresponds to physical indices:
    # if not full: physical = logical
    # if full:     physical = (pos - N + logical) % capacity

    if not buffer.full:
        phys_indices = np.arange(start_idx, N)
    else:
        phys_indices = (buffer.pos - N + np.arange(start_idx, N)) % buffer.capacity

    # ---------------------------------------------------------
    # Create new buffer and copy selected transitions
    # ---------------------------------------------------------

    new_buffer = TrajectoryReplayBufferDiscrete(
        capacity=buffer.capacity,
        obs_dim=buffer.obs_dim,
        action_dim=buffer.action_dim,
        device=buffer.device,
    )

    for p_idx in phys_indices:
        new_buffer.add_transition(
            obs=buffer.obs[p_idx],
            action=buffer.actions[p_idx],
            reward=buffer.rewards[p_idx, 0],
            next_obs=buffer.next_obs[p_idx],
            terminated=buffer.terminated[p_idx, 0],
            truncated=buffer.truncated[p_idx, 0],
            episode_id=buffer.episode_id[p_idx],
            timestep=buffer.timestep[p_idx],
        )

    return new_buffer

def keep_goal_reaching_episodes(
    buffer,
    goal,
    max_episodes=None,
    min_episodes=None,
    atol=1e-6,
):
    """
    Return a new buffer containing:
      - All goal-reaching episodes (up to max_episodes, taking the most recent).
      - If needed, additional non-goal-reaching episodes to reach min_episodes.
    """
    # 1. Identify goal-reaching episodes
    goal_reaching_ep_ids = []
    non_goal_reaching_ep_ids = []

    for ep_id, indices in buffer.episode_to_indices.items():
        reached = False
        for idx in indices:
            if np.allclose(buffer.next_obs[idx], goal, atol=atol):
                reached = True
                break
        if reached:
            goal_reaching_ep_ids.append(ep_id)
        else:
            non_goal_reaching_ep_ids.append(ep_id)

    # Keep most recent goal-reaching episodes
    if max_episodes is not None:
        goal_reaching_ep_ids = goal_reaching_ep_ids[-max_episodes:]

    # 2. If we have too few, add non-goal-reaching episodes
    if min_episodes is not None:
        total_needed = max(0, min_episodes - len(goal_reaching_ep_ids))
        if total_needed > 0:
            # Take the most recent non-goal-reaching episodes
            non_goal_reaching_ep_ids = non_goal_reaching_ep_ids[-total_needed:]
            selected_ep_ids = goal_reaching_ep_ids + non_goal_reaching_ep_ids
        else:
            selected_ep_ids = goal_reaching_ep_ids
    else:
        selected_ep_ids = goal_reaching_ep_ids

    # 3. Create new buffer
    new_buffer = TrajectoryReplayBufferDiscrete(
        capacity=buffer.capacity,
        obs_dim=buffer.obs_dim,
        action_dim=buffer.action_dim,
        device=buffer.device,
    )

    # 4. Re-add selected episodes
    for ep_id in selected_ep_ids:
        indices = buffer.episode_to_indices[ep_id]
        ep_data = {
            "obs": buffer.obs[indices],
            "actions": buffer.actions[indices],
            "rewards": buffer.rewards[indices],
            "next_obs": buffer.next_obs[indices],
            "terminated": buffer.terminated[indices],
            "truncated": buffer.truncated[indices],
        }
        new_buffer.add_episode(ep_data)

    return new_buffer



def evaluate_factorised_atari_policy(
    q_network,
    make_env_fn,
    goal,
    device,
    num_actions: int,
    seed: int = 42,
    max_steps: int = 27_000,
    video_path: str | None = None,
    render_live: bool = False,
    render_every: int = 4,
    show_plot: bool = True,
    display_width: int = 480,
    live_delay: float = 0.03,
):
    """
    Evaluate FactorisedDQN_QNetwork_Atari.

    Expected network interface:

        q_network.q_val_for_argmax_action(
            obs,
            goal,
        )

    Expected observation:

        [C, H, W]
        or
        [H, W, C]

    The network's own _prepare_obs() handles:
        - float conversion;
        - pixel normalization;
        - channels-last permutation when configured.

    Parameters
    ----------
    q_network:
        Trained FactorisedDQN_QNetwork_Atari.

    make_env_fn:
        Factory accepting:

            seed
            goal
            max_horizon
            render_mode

    goal:
        One-dimensional goal such as:

            np.array([5.0], dtype=np.float32)

    device:
        Torch device.

    num_actions:
        Number of discrete actions.

    video_path:
        Optional path for video output.

        If not None, this function attempts to use
        imageio/FFmpeg. For live rendering only, use None.

    render_live:
        Replaces the notebook output with each displayed frame.

    render_every:
        Displays one frame every N environment steps.

    show_plot:
        Displays reward and score curves after evaluation.
    """

    if max_steps <= 0:
        raise ValueError(
            "max_steps must be positive."
        )

    if render_every <= 0:
        raise ValueError(
            "render_every must be positive."
        )

    goal = np.asarray(
        goal,
        dtype=np.float32,
    ).reshape(-1)

    q_network = q_network.to(
        device
    )

    q_network.eval()

    env = make_env_fn(
        seed=seed,
        goal=goal,
        max_horizon=max_steps,
        render_mode="rgb_array",
    )

    video_writer = None

    if video_path is not None:
        try:
            import imageio.v2 as imageio

            video_dir = os.path.dirname(
                video_path
            )

            if video_dir:
                os.makedirs(
                    video_dir,
                    exist_ok=True,
                )

            video_writer = imageio.get_writer(
                video_path,
                fps=30,
                codec="libx264",
            )

        except Exception as error:
            env.close()

            raise RuntimeError(
                "Video output requires an imageio video "
                "backend. Use video_path=None for live "
                "notebook rendering, or install "
                "imageio[ffmpeg]."
            ) from error

    obs, info = env.reset(
        seed=seed
    )

    episode_return = 0.0
    terminated = False
    truncated = False

    actions = []
    rewards = []
    scores = []
    frames = []

    start_time = time.perf_counter()

    try:
        for step_idx in range(
            max_steps
        ):
            obs_array = np.asarray(
                obs
            )

            if obs_array.ndim != 3:
                raise ValueError(
                    "Expected one Atari image with "
                    "three dimensions. "
                    f"Got shape {obs_array.shape}."
                )

            obs_tensor = torch.as_tensor(
                obs_array,
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)

            goal_tensor = torch.as_tensor(
                goal,
                dtype=torch.float32,
                device=device,
            )

            # The network's _prepare_obs() performs
            # pixel normalization and any channels-last
            # permutation internally.
            with torch.inference_mode():
                q_values = (
                    q_network
                    .q_val_for_argmax_action(
                        obs=obs_tensor,
                        goal=goal_tensor,
                    )
                )

            if q_values.ndim != 2:
                raise ValueError(
                    "Expected q_values with shape "
                    "[B, num_actions]. "
                    f"Got {tuple(q_values.shape)}."
                )

            if q_values.shape[0] != 1:
                raise ValueError(
                    "Expected evaluation batch size 1."
                )

            if q_values.shape[1] != num_actions:
                raise ValueError(
                    "The Q-value action dimension does "
                    "not match num_actions. "
                    f"Expected {num_actions}, "
                    f"got {q_values.shape[1]}."
                )

            action = int(
                torch.argmax(
                    q_values[0]
                ).item()
            )

            (
                next_obs,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(action)

            reward = float(
                reward
            )

            episode_return += reward

            actions.append(action)
            rewards.append(reward)

            if isinstance(
                info,
                dict,
            ):
                score = float(
                    info.get(
                        "current_score",
                        np.nan,
                    )
                )
            else:
                score = np.nan

            scores.append(score)

            if (
                step_idx % render_every
                == 0
            ):
                frame = env.render()

                if frame is not None:
                    frame = np.asarray(
                        frame
                    )

                    frames.append(frame)

                    if video_writer is not None:
                        video_writer.append_data(
                            frame
                        )

                    if render_live:
                        display_atari_frame(
                            frame=frame,
                            step_idx=step_idx + 1,
                            action=action,
                            score=score,
                            width=display_width,
                            delay=live_delay,
                        )

            obs = next_obs

            if terminated or truncated:
                break

        elapsed = (
            time.perf_counter()
            - start_time
        )

        final_info = dict(
            info
            if isinstance(info, dict)
            else {}
        )

        finite_scores = [
            value
            for value in scores
            if np.isfinite(value)
        ]

        final_score = (
            finite_scores[-1]
            if finite_scores
            else None
        )

        result = {
            "goal": goal.copy(),
            "seed": int(seed),
            "steps": len(actions),
            "episode_return": float(
                episode_return
            ),
            "final_score": (
                None
                if final_score is None
                else float(final_score)
            ),
            "success": bool(
                final_info.get(
                    "reached",
                    False,
                )
            ),
            "terminated": bool(
                terminated
            ),
            "truncated": bool(
                truncated
            ),
            "actions": np.asarray(
                actions,
                dtype=np.int64,
            ),
            "rewards": np.asarray(
                rewards,
                dtype=np.float32,
            ),
            "scores": np.asarray(
                scores,
                dtype=np.float32,
            ),
            "frames": frames,
            "elapsed_seconds": float(
                elapsed
            ),
        }

        print(
            "\nFactorised Atari policy evaluation",
            flush=True,
        )

        print(
            f"Goal: {goal}",
            flush=True,
        )

        print(
            f"Steps: {result['steps']}",
            flush=True,
        )

        print(
            f"Return: "
            f"{result['episode_return']:.3f}",
            flush=True,
        )

        print(
            f"Final score: "
            f"{result['final_score']}",
            flush=True,
        )

        print(
            f"Success: "
            f"{result['success']}",
            flush=True,
        )

        print(
            f"Terminated: "
            f"{result['terminated']}",
            flush=True,
        )

        print(
            f"Truncated: "
            f"{result['truncated']}",
            flush=True,
        )

        print(
            f"Elapsed: "
            f"{result['elapsed_seconds']:.2f}s",
            flush=True,
        )

        if video_path is not None:
            print(
                f"Video saved to: "
                f"{video_path}",
                flush=True,
            )

        if show_plot:
            plot_atari_evaluation(
                result
            )

        return result

    finally:
        if video_writer is not None:
            video_writer.close()

        env.close()


def display_atari_frame(
    frame,
    step_idx: int,
    action: int,
    score: float | None = None,
    width: int = 480,
    delay: float = 0.03,
):
    """
    Displays a single RGB frame by replacing the
    previous notebook output.

    This works with inline notebook output better
    than plt.pause() alone.
    """

    frame = np.asarray(
        frame
    )

    clear_output(
        wait=True
    )

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    ax.imshow(frame)
    ax.axis("off")

    title = (
        f"step={step_idx} | "
        f"action={action}"
    )

    if score is not None and np.isfinite(
        score
    ):
        title += (
            f" | score={score:.1f}"
        )

    ax.set_title(title)

    fig.set_size_inches(
        width / 100.0,
        width / 100.0,
    )

    plt.show()

    plt.close(fig)

    if delay > 0.0:
        time.sleep(delay)


def plot_atari_evaluation(
    result: dict,
):
    """
    Plot rewards and score over the rollout.
    """

    rewards = result[
        "rewards"
    ]

    scores = result[
        "scores"
    ]

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10, 6),
        sharex=True,
    )

    axes[0].plot(
        rewards,
        label="reward",
    )

    axes[0].set_ylabel(
        "Reward"
    )

    axes[0].grid(
        alpha=0.3
    )

    axes[0].legend()

    if np.isfinite(
        scores
    ).any():
        axes[1].plot(
            scores,
            label="score",
        )

    axes[1].set_xlabel(
        "Environment step"
    )

    axes[1].set_ylabel(
        "Score"
    )

    axes[1].grid(
        alpha=0.3
    )

    axes[1].legend()

    fig.suptitle(
        "Factorised Atari policy evaluation"
    )

    plt.tight_layout()
    plt.show()


def evaluate_policy_with_success(
    env,
    policy_fn,
    episodes: int = 8,
    seed: int | None = None,
):
    """
    Evaluate a Fetch-style policy.

    Returns:
        mean_return,
        mean_length,
        success_rate,
        mean_final_distance
    """
    returns = []
    lengths = []
    successes = []
    final_distances = []

    for episode_idx in range(episodes):
        if seed is None:
            observation_dict, _ = env.reset()
        else:
            observation_dict, _ = env.reset(
                seed=seed + episode_idx
            )

        episode_return = 0.0
        episode_length = 0
        info = {}

        while True:
            action = policy_fn(observation_dict)

            (
                observation_dict,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(action)

            episode_return += float(reward)
            episode_length += 1

            if terminated or truncated:
                break

        achieved_goal = np.asarray(
            observation_dict["achieved_goal"],
            dtype=np.float32,
        )

        desired_goal = np.asarray(
            observation_dict["desired_goal"],
            dtype=np.float32,
        )

        final_distance = float(
            np.linalg.norm(
                achieved_goal - desired_goal
            )
        )

        success = float(
            info.get("is_success", 0.0)
        )

        returns.append(episode_return)
        lengths.append(episode_length)
        successes.append(success)
        final_distances.append(final_distance)

    return (
        float(np.mean(returns)),
        float(np.mean(lengths)),
        float(np.mean(successes)),
        float(np.mean(final_distances)),
    )

def keep_last_fraction_of_buffer_continuous(
    buffer: TrajectoryReplayBufferContinuous,
    keep_fraction: float = 0.2,
) -> TrajectoryReplayBufferContinuous:
    """
    Return a new continuous-action replay buffer containing the most
    recently inserted `keep_fraction` of valid transitions.

    The source buffer may be partially filled or a full circular buffer.
    Transitions are copied in chronological insertion order, from oldest
    retained transition to newest retained transition.

    Parameters
    ----------
    buffer:
        A TrajectoryReplayBufferContinuous instance.

    keep_fraction:
        Fraction of the currently valid transitions to retain.
        For example, 0.2 retains the newest 20% of transitions.

    Returns
    -------
    TrajectoryReplayBufferContinuous
        A new buffer with the selected transitions, their continuous
        actions, termination flags, and episode metadata preserved.
    """
    if not 0.0 < keep_fraction <= 1.0:
        raise ValueError(
            "keep_fraction must be in the interval (0, 1]. "
            f"Got {keep_fraction}."
        )

    n_transitions = len(buffer)

    # ---------------------------------------------------------
    # Empty source buffer
    # ---------------------------------------------------------

    if n_transitions == 0:
        return TrajectoryReplayBufferContinuous(
            capacity=buffer.capacity,
            state_dim=buffer.state_dim,
            action_dim=buffer.action_dim,
            device=buffer.device,
        )

    # Number of most recent valid transitions to preserve.
    keep_count = max(
        1,
        int(n_transitions * keep_fraction),
    )

    logical_start = n_transitions - keep_count

    # ---------------------------------------------------------
    # Map chronological/logical indices to physical array indices
    # ---------------------------------------------------------
    #
    # Partially-filled buffer:
    #
    #   logical index i -> physical index i
    #
    # Full circular buffer:
    #
    #   buffer.pos points to the next position that would be
    #   overwritten. That is also the physical location of the
    #   oldest retained transition.
    #
    #   logical index i -> (buffer.pos - buffer.size + i) % capacity
    #
    # Since a full buffer has size == capacity:
    #
    #   (buffer.pos - capacity + i) % capacity
    #   == (buffer.pos + i) % capacity
    # ---------------------------------------------------------

    logical_indices = np.arange(
        logical_start,
        n_transitions,
        dtype=np.int64,
    )

    if not buffer.full:
        physical_indices = logical_indices

    else:
        physical_indices = (
            buffer.pos
            - buffer.size
            + logical_indices
        ) % buffer.capacity

    # ---------------------------------------------------------
    # Construct destination buffer
    # ---------------------------------------------------------

    new_buffer = TrajectoryReplayBufferContinuous(
        capacity=buffer.capacity,
        state_dim=buffer.state_dim,
        action_dim=buffer.action_dim,
        device=buffer.device,
    )

    # ---------------------------------------------------------
    # Copy the selected transitions in chronological order
    # ---------------------------------------------------------

    for physical_idx in physical_indices:
        new_buffer.add_transition(
            obs=buffer.obs[physical_idx].copy(),

            # Continuous action vector, normally shape (4,) for Fetch.
            action=buffer.actions[physical_idx].copy(),

            reward=float(
                buffer.rewards[
                    physical_idx,
                    0,
                ]
            ),

            next_obs=buffer.next_obs[
                physical_idx
            ].copy(),

            terminated=bool(
                buffer.terminated[
                    physical_idx,
                    0,
                ]
            ),

            truncated=bool(
                buffer.truncated[
                    physical_idx,
                    0,
                ]
            ),

            episode_id=int(
                buffer.episode_id[
                    physical_idx
                ]
            ),

            timestep=int(
                buffer.timestep[
                    physical_idx
                ]
            ),
        )

    return new_buffer


def integrate_trimmed_task_buffer(
    task_id,
    trimmed_buffer_new,
    global_buffer,        # list of transitions
    buffer_map,           # dict: task_id -> buffer_id
    buffer_contents,      # dict: buffer_id -> list of transitions
    similarity_matrix,
    X=0.7
):
    """
    Inputs:
      - task_id: int
      - trimmed_buffer_new: list (already trimmed by keep_buffer_fraction)
      - global_buffer: list of all transitions currently stored
      - buffer_map: task_id -> buffer_id
      - buffer_contents: buffer_id -> list of transitions
      - similarity_matrix: 2D array or similar, S[i, j]
      - X: similarity threshold

    Returns:
      - global_buffer: updated flat list of all transitions
      - buffer_map: updated task_id -> buffer_id
      - buffer_contents: updated buffer_id -> list of transitions
    """
    existing_tasks = [t for t in buffer_map.keys() if t != task_id]

    # First task: just create its own buffer
    if not existing_tasks:
        new_buf_id = f"buf_{task_id}"
        buffer_map[task_id] = new_buf_id
        buffer_contents[new_buf_id] = list(trimmed_buffer_new)
        global_buffer = list(trimmed_buffer_new)
        return global_buffer, buffer_map, buffer_contents

    # Most similar existing task
    sims = {t: similarity_matrix[task_id, t] for t in existing_tasks}
    i_star = max(sims, key=sims.get)
    s = sims[i_star]

    if abs(s - 1.0) < 1e-6:
        # Regime 1: identical -> discard this task's buffer
        buffer_map[task_id] = buffer_map[i_star]
        # global_buffer and buffer_contents unchanged

    elif s <= X:
        # Regime 2: dissimilar -> new independent buffer
        new_buf_id = f"buf_{task_id}"
        buffer_map[task_id] = new_buf_id
        buffer_contents[new_buf_id] = list(trimmed_buffer_new)

        # Append to global buffer
        global_buffer = global_buffer + list(trimmed_buffer_new)

    else:
        # Regime 3: partially similar -> merge fraction into i_star's buffer
        r = (1.0 - s) / (1.0 - X)  # merge ratio
        buf_id_star = buffer_map[i_star]

        n_add = max(1, int(len(trimmed_buffer_new) * r))
        to_add = trimmed_buffer_new[-n_add:]

        # Update logical buffer for i_star
        buffer_contents[buf_id_star] = buffer_contents[buf_id_star] + to_add

        # Map this task to the same buffer
        buffer_map[task_id] = buf_id_star

        # Rebuild global buffer from all logical buffers
        global_buffer = []
        for buf in buffer_contents.values():
            global_buffer.extend(buf)

    return global_buffer, buffer_map, buffer_contents