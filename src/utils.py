from dataclasses import dataclass
import torch
import numpy as np
import random
import torch.nn.functional as F
import matplotlib.pyplot as plt


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
            self.actions[idx] = np.int64(episode["actions"][t])
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
        max_tries = batch_size * 300  # higher because we may reject self-loops

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


def evaluate_policy(env, policy_fn, episodes=8, options=None):
    returns, lengths = [], []
    for _ in range(episodes):

        if options is not None and "start_position" in options:
            obs, _ = env.reset(options={"start_position": options["start_position"]})
        else:
            obs, _ = env.reset()
        done = False
        ep_ret, ep_len = 0.0, 0
        while not done:
            act = policy_fn(obs)
            obs, rew, term, trunc, _ = env.step(act)
            ep_ret += float(rew)
            ep_len += 1
            done = term or trunc
        returns.append(ep_ret)
        lengths.append(ep_len)
    return float(np.mean(returns)), float(np.mean(lengths))


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
        done_t = torch.clamp(term_t + trunc_t, 0.0, 1.0)

        B = obs_t.shape[0]
        goal_batch = build_goal_batch(goal, B, device)

        with torch.no_grad():
            next_q_vals = target_model.q_val_for_argmax_action(next_obs_t, goal_batch)
            next_q = next_q_vals.max(dim=-1, keepdim=True).values
            target = rew_t + gamma * (1.0 - done_t) * next_q

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