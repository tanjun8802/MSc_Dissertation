from __future__ import annotations

import argparse
import json

from .exorl import (
    DEFAULT_EXORL_DATA_ROOT,
    describe_exorl_dataset,
    download_exorl_dataset,
    iter_exorl_episode_files,
    load_exorl_episode,
    make_exorl_env,
)
from .ogbench import (
    DEFAULT_OGBENCH_DATA_ROOT,
    download_ogbench_datasets,
    load_ogbench_datasets,
    make_ogbench_env,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark helpers for ExORL and OGBench.")
    subparsers = parser.add_subparsers(dest="benchmark", required=True)

    exorl_parser = subparsers.add_parser("exorl", help="Work with ExORL datasets and suite-backed envs.")
    exorl_subparsers = exorl_parser.add_subparsers(dest="action", required=True)

    exorl_download = exorl_subparsers.add_parser("download", help="Download an ExORL dataset archive.")
    exorl_download.add_argument("domain")
    exorl_download.add_argument("algorithm")
    exorl_download.add_argument("--data-root", default=str(DEFAULT_EXORL_DATA_ROOT))
    exorl_download.add_argument("--force-download", action="store_true")

    exorl_inspect = exorl_subparsers.add_parser("inspect", help="Inspect one downloaded ExORL replay directory.")
    exorl_inspect.add_argument("domain")
    exorl_inspect.add_argument("algorithm")
    exorl_inspect.add_argument("--data-root", default=str(DEFAULT_EXORL_DATA_ROOT))

    exorl_env = exorl_subparsers.add_parser("env", help="Smoke-test a suite-backed ExORL task.")
    exorl_env.add_argument("task")
    exorl_env.add_argument("--seed", type=int, default=0)

    ogbench_parser = subparsers.add_parser("ogbench", help="Work with OGBench datasets and envs.")
    ogbench_subparsers = ogbench_parser.add_subparsers(dest="action", required=True)

    ogbench_download = ogbench_subparsers.add_parser("download", help="Download one or more OGBench datasets.")
    ogbench_download.add_argument("datasets", nargs="+")
    ogbench_download.add_argument("--data-root", default=str(DEFAULT_OGBENCH_DATA_ROOT))

    ogbench_env = ogbench_subparsers.add_parser("env", help="Smoke-test an OGBench env without downloading data.")
    ogbench_env.add_argument("dataset")
    ogbench_env.add_argument("--task-id", type=int, default=1)

    ogbench_dataset = ogbench_subparsers.add_parser("dataset", help="Load an OGBench dataset and print split shapes.")
    ogbench_dataset.add_argument("dataset")
    ogbench_dataset.add_argument("--data-root", default=str(DEFAULT_OGBENCH_DATA_ROOT))

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.benchmark == "exorl" and args.action == "download":
        replay_dir = download_exorl_dataset(
            domain=args.domain,
            algorithm=args.algorithm,
            data_root=args.data_root,
            force_download=args.force_download,
        )
        print(replay_dir)
        return

    if args.benchmark == "exorl" and args.action == "inspect":
        spec = describe_exorl_dataset(args.domain, args.algorithm, data_root=args.data_root)
        episode_files = list(iter_exorl_episode_files(spec.replay_dir))
        first_episode = load_exorl_episode(episode_files[0]) if episode_files else {}
        summary = {
            "domain": spec.domain,
            "algorithm": spec.algorithm,
            "replay_dir": str(spec.replay_dir),
            "num_episode_files": len(episode_files),
            "first_episode_shapes": {key: list(value.shape) for key, value in first_episode.items()},
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    if args.benchmark == "exorl" and args.action == "env":
        env = make_exorl_env(args.task, seed=args.seed)
        obs, info = env.reset(seed=args.seed)
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, step_info = env.step(action)
        summary = {
            "obs_shape": list(obs.shape),
            "next_obs_shape": list(next_obs.shape),
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "info": info,
            "step_info": step_info,
        }
        env.close()
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    if args.benchmark == "ogbench" and args.action == "download":
        output_dir = download_ogbench_datasets(args.datasets, dataset_dir=args.data_root)
        print(output_dir)
        return

    if args.benchmark == "ogbench" and args.action == "env":
        env = make_ogbench_env(args.dataset)
        obs, info = env.reset(options={"task_id": args.task_id})
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, step_info = env.step(action)
        summary = {
            "obs_shape": list(obs.shape),
            "next_obs_shape": list(next_obs.shape),
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "goal_in_reset_info": "goal" in info,
            "success_in_step_info": "success" in step_info,
        }
        env.close()
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    if args.benchmark == "ogbench" and args.action == "dataset":
        _, train_dataset, val_dataset = load_ogbench_datasets(
            args.dataset,
            dataset_dir=args.data_root,
            compact_dataset=False,
        )
        summary = {
            "train": {key: list(value.shape) for key, value in train_dataset.items()},
            "val": {key: list(value.shape) for key, value in val_dataset.items()},
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    parser.error("Unsupported command.")


if __name__ == "__main__":
    main()