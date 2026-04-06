"""
config.py
=========
Lightweight YAML config loader for experiment scripts.

Usage
-----
    from utils.config import load_config

    cfg = load_config("configs/gcrl.yaml")
    # access nested values:
    lr = cfg["agent"]["alpha"]
"""

from __future__ import annotations

import os
from typing import Any

import yaml


def load_config(path: str) -> dict[str, Any]:
    """Load a YAML file and return its contents as a nested dict.

    Parameters
    ----------
    path :
        Absolute or relative path to a ``.yaml`` / ``.yml`` file.

    Returns
    -------
    dict
        Parsed YAML contents.  Returns an empty dict if the file does not
        exist so that callers can safely fall back to hardcoded defaults.
    """
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        return yaml.safe_load(fh) or {}
