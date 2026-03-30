"""
experiments/
============
Experiment runners that orchestrate environment–agent interaction loops.

Modules
-------
base_experiment : Abstract experiment class with a standard train/eval loop.
run_experiment  : Entry-point script for launching experiments from the CLI.
"""

from experiments.base_experiment import BaseExperiment

__all__ = ["BaseExperiment"]
