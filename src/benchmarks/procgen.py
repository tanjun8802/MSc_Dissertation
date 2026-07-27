"""Procgen benchmark adapter.

Provides Gymnasium-style access to the 16 procedurally generated games from
the Procgen suite (Cobbe et al., 2020).  The wrapper normalises observations
to ``float32 [0, 1]`` and exposes standard ``gymnasium.Env`` attributes.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces


# ---------------------------------------------------------------------------
# Procgen game catalogue
# ---------------------------------------------------------------------------

PROCGEN_GAMES: tuple[str, ...] = (
    "bigfish",
    "bossfight",
    "caveflyer",
    "chaser",
    "climber",
    "coinrun",
    "dodgeball",
    "fruitbot",
    "heist",
    "jumper",
    "leaper",
    "maze",
    "miner",
    "ninja",
    "plunder",
    "starpilot",
)

PROCGEN_DISTRIBUTION_MODES: tuple[str, ...] = ("easy", "hard", "exploration", "memory")


def require_procgen() -> None:
    """Raise a helpful ``ImportError`` if *procgen* is not installed."""
    try:
        import procgen  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "procgen is not installed.  Install it with "
            "`pip install procgen` or `uv pip install procgen`."
        ) from exc


# ---------------------------------------------------------------------------
# Gymnasium adapter
# ---------------------------------------------------------------------------


class ProcgenGymnasiumAdapter(gym.Env):
    """Thin Gymnasium wrapper around a single Procgen environment.

    Procgen ships its own vectorised API.  This adapter creates a
    single-instance ``ProcgenEnv`` and re-exposes it with the standard
    ``gymnasium.Env`` interface (``reset`` / ``step`` / ``render``).

    Observations are normalised to ``float32 ∈ [0, 1]`` by default so
    that downstream code does not have to handle ``uint8`` frames.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 15}

    def __init__(
        self,
        game: str,
        *,
        num_levels: int = 0,
        start_level: int = 0,
        distribution_mode: str = "easy",
        seed: int = 0,
        render_mode: str | None = None,
        normalize_obs: bool = True,
    ):
        super().__init__()

        require_procgen()
        from procgen import ProcgenEnv  # type: ignore[import-untyped]

        if game not in PROCGEN_GAMES:
            available = ", ".join(PROCGEN_GAMES)
            raise ValueError(
                f"Unsupported Procgen game '{game}'. Expected one of: {available}."
            )
        if distribution_mode not in PROCGEN_DISTRIBUTION_MODES:
            raise ValueError(
                f"Unsupported distribution_mode '{distribution_mode}'. "
                f"Expected one of: {', '.join(PROCGEN_DISTRIBUTION_MODES)}."
            )

        self._venv = ProcgenEnv(
            num_envs=1,
            env_name=game,
            num_levels=num_levels,
            start_level=start_level,
            distribution_mode=distribution_mode,
            rand_seed=seed,
        )

        self._game = game
        self._normalize_obs = normalize_obs
        self.render_mode = render_mode

        # Derive single-env spaces from the vectorised environment
        obs_space = self._venv.observation_space["rgb"]
        h, w, c = obs_space.shape

        if normalize_obs:
            self.observation_space = spaces.Box(
                low=0.0, high=1.0, shape=(c, h, w), dtype=np.float32,
            )
        else:
            self.observation_space = spaces.Box(
                low=0, high=255, shape=(c, h, w), dtype=np.uint8,
            )

        self.action_space = self._venv.action_space

        self._last_rgb: np.ndarray | None = None

    # ----- helpers ---------------------------------------------------------

    def _process_obs(self, obs_dict: dict) -> np.ndarray:
        """Extract the RGB frame, transpose to CHW, and optionally normalise."""
        frame = obs_dict["rgb"][0]  # (H, W, C) uint8
        frame = np.transpose(frame, (2, 0, 1))  # (C, H, W)
        self._last_rgb = obs_dict["rgb"][0]  # keep HWC for render
        if self._normalize_obs:
            return frame.astype(np.float32) / 255.0
        return frame

    # ----- gymnasium.Env interface -----------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        del seed, options  # Procgen seed is set at construction time
        obs_dict = self._venv.reset()
        obs = self._process_obs(obs_dict)
        return obs, {"game": self._game}

    def step(self, action):
        action_arr = np.asarray([int(action)], dtype=np.int32)
        obs_dict, rewards, dones, infos = self._venv.step(action_arr)
        obs = self._process_obs(obs_dict)
        reward = float(rewards[0])
        terminated = bool(dones[0])
        truncated = False
        info: dict = {k: v[0] if hasattr(v, "__getitem__") else v for k, v in infos.items()}
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode is None:
            return None
        if self.render_mode != "rgb_array":
            raise NotImplementedError(
                f"Unsupported render_mode '{self.render_mode}'. Only 'rgb_array' is supported."
            )
        return self._last_rgb

    def close(self):
        if hasattr(self, "_venv"):
            self._venv.close()


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def make_procgen_env(
    game: str,
    *,
    num_levels: int = 0,
    start_level: int = 0,
    distribution_mode: str = "easy",
    seed: int = 0,
    render_mode: str | None = None,
    normalize_obs: bool = True,
) -> gym.Env:
    """Create a single Procgen environment with Gymnasium interface.

    Parameters
    ----------
    game:
        One of the 16 Procgen game names, e.g. ``"coinrun"`` or ``"starpilot"``.
    num_levels:
        Number of unique levels. ``0`` means unlimited.
    start_level:
        Lowest seed for level generation.
    distribution_mode:
        ``"easy"`` (default), ``"hard"``, ``"exploration"``, or ``"memory"``.
    seed:
        Random seed.
    render_mode:
        ``None`` or ``"rgb_array"``.
    normalize_obs:
        If ``True`` (default), observations are ``float32 ∈ [0, 1]`` in CHW
        layout.  Otherwise ``uint8 ∈ [0, 255]`` in CHW layout.
    """
    return ProcgenGymnasiumAdapter(
        game=game,
        num_levels=num_levels,
        start_level=start_level,
        distribution_mode=distribution_mode,
        seed=seed,
        render_mode=render_mode,
        normalize_obs=normalize_obs,
    )
