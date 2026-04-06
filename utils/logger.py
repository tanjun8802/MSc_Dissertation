"""
logger.py
=========
Lightweight experiment logger that writes metrics to the console and a CSV file.

No external dependencies beyond the Python standard library.
"""

from __future__ import annotations

import csv
import os
import time
from typing import Any

from utils.metrics import EpisodeMetrics


class Logger:
    """Logs episode metrics to stdout and a CSV file.

    Parameters
    ----------
    log_dir :
        Directory where the CSV file will be written.
        Created automatically if it does not exist.
    filename :
        Name of the CSV file inside *log_dir*.
    verbose :
        Whether to print each episode summary to stdout.
    """

    def __init__(
        self,
        log_dir: str = "logs",
        filename: str = "metrics.csv",
        verbose: bool = True,
    ) -> None:
        self.log_dir = log_dir
        self.verbose = verbose
        self._start_time = time.time()
        self._csv_path = os.path.join(log_dir, filename)
        self._csv_file = None
        self._csv_writer = None
        # Trajectory log (written on demand via log_trajectory)
        self._traj_path = os.path.join(log_dir, "trajectory.csv")
        self._traj_file = None
        self._traj_writer = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_episode(self, episode: int, metrics: EpisodeMetrics) -> None:
        """Log a training episode."""
        self._write_row(episode, metrics, mode="train")
        if self.verbose:
            elapsed = time.time() - self._start_time
            print(f"[{elapsed:>7.1f}s] {metrics}")

    def log_eval(self, episode: int, metrics: EpisodeMetrics) -> None:
        """Log an evaluation episode."""
        self._write_row(episode, metrics, mode="eval")
        if self.verbose:
            elapsed = time.time() - self._start_time
            print(f"[{elapsed:>7.1f}s] {metrics}  ← EVAL")

    def log_trajectory(
        self,
        episode: int,
        trajectory: list[tuple[int, int, int, float]],
    ) -> None:
        """Write a single episode's trajectory to ``trajectory.csv``.

        Parameters
        ----------
        episode :
            Episode index to associate with this trajectory.
        trajectory :
            List of ``(step, state, action, reward)`` tuples recorded during
            the episode.
        """
        if not trajectory:
            return
        fieldnames = ["episode", "step", "state", "action", "reward"]
        if self._traj_writer is None:
            os.makedirs(self.log_dir, exist_ok=True)
            self._traj_file = open(self._traj_path, "w", newline="")  # noqa: SIM115
            self._traj_writer = csv.DictWriter(self._traj_file, fieldnames=fieldnames)
            self._traj_writer.writeheader()
        for step, state, action, reward in trajectory:
            self._traj_writer.writerow(
                {
                    "episode": episode,
                    "step": step,
                    "state": state,
                    "action": action,
                    "reward": reward,
                }
            )
        if self._traj_file is not None:
            self._traj_file.flush()

    def close(self) -> None:
        """Flush and close all open CSV files."""
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
        if self._traj_file is not None:
            self._traj_file.close()
            self._traj_file = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_csv_open(self, fieldnames: list[str]) -> None:
        """Open the CSV file lazily on the first write."""
        if self._csv_writer is not None:
            return
        os.makedirs(self.log_dir, exist_ok=True)
        self._csv_file = open(self._csv_path, "w", newline="")  # noqa: SIM115
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=fieldnames)
        self._csv_writer.writeheader()

    def _write_row(self, episode: int, metrics: EpisodeMetrics, mode: str) -> None:
        row: dict[str, Any] = {
            "episode": episode,
            "mode": mode,
            "total_reward": metrics.total_reward,
            "length": metrics.length,
            "epsilon": metrics.epsilon if metrics.epsilon is not None else "",
            "elapsed_s": round(time.time() - self._start_time, 3),
        }
        self._ensure_csv_open(list(row.keys()))
        self._csv_writer.writerow(row)
        if self._csv_file is not None:
            self._csv_file.flush()

    def __del__(self) -> None:
        self.close()
