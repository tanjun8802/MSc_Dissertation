from dataclasses import dataclass
import torch
import numpy as np


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

    def add_episode(self, episode):
        ep_id = self.current_episode_id
        self.current_episode_id += 1

        ep_indices = []

        T = len(episode["obs"])
        for t in range(T):
            idx = self.pos

            if self.full:
                old_ep = self.episode_id[idx]
                if old_ep in self.episode_to_indices:
                    try:
                        self.episode_to_indices[old_ep].remove(idx)
                        if len(self.episode_to_indices[old_ep]) == 0:
                            del self.episode_to_indices[old_ep]
                    except ValueError:
                        pass

            self.obs[idx] = np.asarray(episode["obs"][t], dtype=np.float32)
            self.actions[idx] = np.asarray(episode["actions"][t], dtype=np.float32).reshape(-1)
            self.rewards[idx] = np.asarray([episode["rewards"][t]], dtype=np.float32)
            self.next_obs[idx] = np.asarray(episode["next_obs"][t], dtype=np.float32)
            self.terminated[idx] = np.asarray([episode["terminated"][t]], dtype=np.float32)
            self.truncated[idx] = np.asarray([episode["truncated"][t]], dtype=np.float32)

            self.episode_id[idx] = ep_id
            self.timestep[idx] = t

            ep_indices.append(idx)

            self.pos = (self.pos + 1) % self.capacity
            if self.size < self.capacity:
                self.size += 1
            else:
                self.full = True

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
            "future_state": torch.tensor(self.obs[g_idxs], device=self.device),
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