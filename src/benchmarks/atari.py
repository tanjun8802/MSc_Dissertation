"""Atari / ALE benchmark adapter.

Provides Gymnasium-style access to the ~57 Atari 2600 games via ``ale-py``.
Standard preprocessing wrappers (frame-stacking, grayscale, resize, …) are
applied by default so that the returned environment is ready for DQN-style
training or Atari-100k experiments.
"""

from __future__ import annotations

from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces


# ---------------------------------------------------------------------------
# Canonical Atari-100k game set (26 games used by Kaiser et al., 2020)
# ---------------------------------------------------------------------------
ATARI_100K_GAMES: tuple[str, ...] = (
    "Alien",
    "Amidar",
    "Assault",
    "Asterix",
    "BankHeist",
    "BattleZone",
    "Boxing",
    "Breakout",
    "ChopperCommand",
    "CrazyClimber",
    "DemonAttack",
    "Freeway",
    "Frostbite",
    "Gopher",
    "Hero",
    "Jamesbond",
    "Kangaroo",
    "Krull",
    "KungFuMaster",
    "MsPacman",
    "Pong",
    "PrivateEye",
    "Qbert",
    "RoadRunner",
    "Seaquest",
    "UpNDown",
)

# Full list of the 57 standard Atari games (Bellemare et al., 2013)
ATARI_57_GAMES: tuple[str, ...] = (
    "Alien",
    "Amidar",
    "Assault",
    "Asterix",
    "Asteroids",
    "Atlantis",
    "BankHeist",
    "BattleZone",
    "BeamRider",
    "Berzerk",
    "Bowling",
    "Boxing",
    "Breakout",
    "Centipede",
    "ChopperCommand",
    "CrazyClimber",
    "Defender",
    "DemonAttack",
    "DoubleDunk",
    "Enduro",
    "FishingDerby",
    "Freeway",
    "Frostbite",
    "Gopher",
    "Gravitar",
    "Hero",
    "IceHockey",
    "Jamesbond",
    "Kangaroo",
    "Krull",
    "KungFuMaster",
    "MontezumaRevenge",
    "MsPacman",
    "NameThisGame",
    "Phoenix",
    "Pitfall",
    "Pong",
    "PrivateEye",
    "Qbert",
    "Riverraid",
    "RoadRunner",
    "Robotank",
    "Seaquest",
    "Skiing",
    "Solaris",
    "SpaceInvaders",
    "StarGunner",
    "Surround",
    "Tennis",
    "TimePilot",
    "Tutankham",
    "UpNDown",
    "Venture",
    "VideoPinball",
    "WizardOfWor",
    "YarsRevenge",
    "Zaxxon",
)


def require_ale() -> None:
    """Raise a helpful ``ImportError`` if *ale-py* is not installed."""
    try:
        import ale_py  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "ale-py is not installed.  Install it with "
            "`pip install ale-py` or `uv pip install ale-py`."
        ) from exc


# ---------------------------------------------------------------------------
# Pre-processing wrappers (follow the standard DeepMind Atari stack)
# ---------------------------------------------------------------------------


class NoopResetWrapper(gym.Wrapper):
    """Execute a random number of no-ops on reset for stochastic starts."""

    def __init__(self, env: gym.Env, noop_max: int = 30):
        super().__init__(env)
        self.noop_max = noop_max
        self.noop_action = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        noops = self.np_random.integers(1, self.noop_max + 1)
        for _ in range(noops):
            obs, _, terminated, truncated, info = self.env.step(self.noop_action)
            if terminated or truncated:
                obs, info = self.env.reset()
        return obs, info


class MaxAndSkipWrapper(gym.Wrapper):
    """Return the max of the last *skip* frames and repeat action *skip* times."""

    def __init__(self, env: gym.Env, skip: int = 4):
        super().__init__(env)
        self._skip = skip
        self._obs_buffer = deque(maxlen=2)

    def step(self, action):
        total_reward = 0.0
        terminated = truncated = False
        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += float(reward)
            if terminated or truncated:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, terminated, truncated, info


class FireResetWrapper(gym.Wrapper):
    """Press FIRE on reset for environments that require it to start."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == "FIRE"

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self.env.reset(seed=seed, options=options)
        obs, _, terminated, truncated, info = self.env.step(1)
        if terminated or truncated:
            obs, info = self.env.reset()
        return obs, info


class GrayscaleResizeWrapper(gym.ObservationWrapper):
    """Convert RGB frames to 84×84 grayscale (channel-first by default)."""

    def __init__(self, env: gym.Env, width: int = 84, height: int = 84):
        super().__init__(env)
        self._width = width
        self._height = height
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(1, height, width), dtype=np.uint8
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        # Weighted grayscale conversion
        gray = np.dot(obs[..., :3].astype(np.float32), [0.2989, 0.5870, 0.1140])
        gray = gray.astype(np.uint8)
        # Resize using nearest-neighbour (avoids a cv2 / PIL dependency)
        resized = self._resize_nearest(gray, self._height, self._width)
        return resized[np.newaxis, :, :]  # (1, H, W)

    @staticmethod
    def _resize_nearest(img: np.ndarray, h: int, w: int) -> np.ndarray:
        src_h, src_w = img.shape[:2]
        row_idx = (np.arange(h) * src_h / h).astype(int)
        col_idx = (np.arange(w) * src_w / w).astype(int)
        return img[np.ix_(row_idx, col_idx)]


class FrameStackWrapper(gym.Wrapper):
    """Stack the last *n_frames* observations along the channel axis."""

    def __init__(self, env: gym.Env, n_frames: int = 4):
        super().__init__(env)
        self._n_frames = n_frames
        self._frames: deque[np.ndarray] = deque(maxlen=n_frames)
        c, h, w = env.observation_space.shape  # type: ignore[misc]
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(c * n_frames, h, w), dtype=np.uint8
        )

    def _get_obs(self) -> np.ndarray:
        return np.concatenate(list(self._frames), axis=0)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        for _ in range(self._n_frames):
            self._frames.append(obs)
        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info


class EpisodicLifeWrapper(gym.Wrapper):
    """End the episode (but don't reset) when a life is lost."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.was_real_done = terminated or truncated
        lives = self.env.unwrapped.ale.lives()
        if 0 < lives < self.lives:
            terminated = True
        self.lives = lives
        return obs, reward, terminated, truncated, info

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if self.was_real_done:
            obs, info = self.env.reset(seed=seed, options=options)
        else:
            obs, _, _, _, info = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs, info


class ClipRewardWrapper(gym.RewardWrapper):
    """Clip rewards to {-1, 0, +1}."""

    def reward(self, reward: float) -> float:
        return float(np.sign(reward))


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def make_atari_env(
    game: str,
    *,
    seed: int = 0,
    render_mode: str | None = None,
    frame_stack: int = 4,
    noop_max: int = 30,
    frame_skip: int = 4,
    episodic_life: bool = True,
    clip_rewards: bool = True,
    grayscale: bool = True,
    image_size: int = 84,
    full_action_space: bool = False,
    max_episode_steps: int | None = 108_000,
) -> gym.Env:
    """Create a preprocessed Atari environment (Gymnasium-compatible).

    Parameters
    ----------
    game:
        Atari game name, e.g. ``"Breakout"`` or ``"Pong"``.
    seed:
        Random seed passed to the environment on creation.
    render_mode:
        ``None`` or ``"rgb_array"``.
    frame_stack:
        Number of consecutive frames to stack (0 to disable).
    noop_max:
        Maximum number of no-op actions on reset.
    frame_skip:
        Number of frames to repeat each action.
    episodic_life:
        If ``True``, losing a life ends the episode.
    clip_rewards:
        If ``True``, clip rewards to {-1, 0, +1}.
    grayscale:
        If ``True``, convert frames to 84×84 grayscale.
    image_size:
        Height/width of the output frame when *grayscale* is ``True``.
    full_action_space:
        If ``True``, expose all 18 Atari actions instead of the
        game-specific minimal set.
    max_episode_steps:
        Hard limit on the number of frames per episode (``None`` for no limit).
    """
    require_ale()

    # Build the canonical ALE environment id
    env_id = f"ALE/{game}-v5"

    env = gym.make(
        env_id,
        frameskip=1,  # we handle frame-skip with our own wrapper
        repeat_action_probability=0.0,
        full_action_space=full_action_space,
        max_episode_steps=max_episode_steps,
        render_mode=render_mode,
    )
    env.reset(seed=seed)

    # Standard Atari preprocessing stack
    if noop_max > 0:
        env = NoopResetWrapper(env, noop_max=noop_max)
    if frame_skip > 1:
        env = MaxAndSkipWrapper(env, skip=frame_skip)
    if episodic_life:
        env = EpisodicLifeWrapper(env)

    # Fire on reset if the game requires it
    action_meanings = env.unwrapped.get_action_meanings()
    if "FIRE" in action_meanings and len(action_meanings) > 1:
        if action_meanings[1] == "FIRE":
            env = FireResetWrapper(env)

    if clip_rewards:
        env = ClipRewardWrapper(env)
    if grayscale:
        env = GrayscaleResizeWrapper(env, width=image_size, height=image_size)
    if frame_stack > 0:
        env = FrameStackWrapper(env, n_frames=frame_stack)

    return env
