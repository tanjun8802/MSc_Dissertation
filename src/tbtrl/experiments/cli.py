"""Command-line entry point for the supported experiments."""

import argparse
import logging

from .config import load_config
from .runner import run_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--output", required=True, help="New output directory; existing runs are never overwritten."
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--smoke", action="store_true", help="Small real training run for pipeline checks."
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["scratch", "sequential", "recovery"],
        choices=["scratch", "sequential", "recovery"],
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config = load_config(args.config)
    if args.smoke:
        config = config.smoke()
    run_experiment(config, args.output, device=args.device, modes=tuple(args.modes))


if __name__ == "__main__":
    main()
