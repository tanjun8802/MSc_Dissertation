"""Audit effective settings directly against code in the original working notebooks.

This reads the pinned Git revision; it never executes notebook cells. A small
expression evaluator resolves numeric constants, task-dependent overrides and
sampled goals. Run in CI with the original commit present in Git history.
"""

import ast
import inspect
import json
import operator
import subprocess
from pathlib import Path

import numpy as np

from tbtrl.environments.registry import load_layout
from tbtrl.experiments.config import load_config, training_seed, training_settings
from tbtrl.training.dqn import train_dqn
from tbtrl.training.sac import train_sac

REVISION = "0606a8f664974ecd8af2785fc575d20785f17b24"
ROOT = Path(__file__).resolve().parents[1]


def source(path):
    return subprocess.check_output(["git", "show", f"{REVISION}:{path}"], text=True)


def resolve(node, values):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return values[node.id]
    if isinstance(node, (ast.List, ast.Tuple)):
        return [resolve(n, values) for n in node.elts]
    if isinstance(node, ast.Dict):
        return {resolve(k, values): resolve(v, values) for k, v in zip(node.keys, node.values)}
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -resolve(node.operand, values)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return resolve(node.left, values) + resolve(node.right, values)
    if isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.ops[0], ast.Eq):
        return operator.eq(resolve(node.left, values), resolve(node.comparators[0], values))
    if isinstance(node, ast.IfExp):
        return resolve(node.body if resolve(node.test, values) else node.orelse, values)
    if isinstance(node, ast.Call):
        name = ast.unparse(node.func)
        args = [resolve(a, values) for a in node.args]
        if name == "float":
            return float(*args)
        if name == "np.array":
            return args[0]
    raise ValueError(ast.unparse(node))


def audit():
    trainers = ast.parse(source("src/trainer.py"))
    defaults = {}
    for node in trainers.body:
        if getattr(node, "name", "") in (
            "dqn_train_phi_psi_adjustment",
            "sac_train_tbtrl_maze",
            "sac_train_tbtrl",
        ):
            defaults[node.name] = {
                arg.arg: ast.literal_eval(value)
                for arg, value in zip(
                    node.args.args[-len(node.args.defaults) :], node.args.defaults
                )
            }
    ignored = {
        "seed",
        "q_network",
        "q_target_network",
        "actor",
        "critic",
        "critic_target",
        "env",
        "make_env",
        "obs_dim",
        "action_dim",
        "goal",
        "task_id",
        "device",
        "replay_task_buffers",
        "task_goals",
        "env_id",
        "lr_task_code",
        "td_steps",
    }
    reports = []
    for path in sorted((ROOT / "configs").glob("*.json")):
        config = load_config(path)
        if config.actor_type != "mlp":
            continue  # The new actor experiment has no original-notebook equivalent.
        notebook = json.loads(source(f"src/experiments/working/{config.name}.ipynb"))
        tree = ast.parse(
            "\n\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
        )
        values = {}
        for node in tree.body:
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
            ):
                try:
                    values[node.targets[0].id] = resolve(node.value, values)
                except (ValueError, KeyError, TypeError):
                    pass
        assert config.seeds == values["SEEDS"], (config.name, "seeds")
        np.testing.assert_equal(
            load_layout("fourrooms" if "FourRooms" in config.name else "maze"),
            values["MAZE_LAYOUT"],
        )
        if config.algorithm == "dqn":
            cells = [
                (x, y)
                for y, row in enumerate(values["MAZE_LAYOUT"])
                for x, wall in enumerate(row)
                if wall == 0
            ]
            rng = np.random.default_rng(0)
            expected_goals = [
                cells[i] for i in rng.choice(len(cells), size=values["NUM_GOALS"], replace=False)
            ]
        else:
            expected_goals = [t["goal"] for t in values["CROSS_ENV_TASKS"]]
        np.testing.assert_equal(config.goals, expected_goals)
        # Network widths/embedding radii and environment parameters are part of
        # the experiment too, rather than incidental trainer arguments.
        model_names = {"FactorisedDQN_QNetwork_BallNorm", "FactorisedTwinCriticFetch"}
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in model_names
            ):
                for kw in node.keywords:
                    if kw.arg in config.model:
                        assert config.model[kw.arg] == resolve(kw.value, values), (
                            config.name,
                            "model",
                            kw.arg,
                        )
        factory_name = "make_env" if config.algorithm == "dqn" else "make_maze_env"
        factory = next(
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == factory_name
        )
        env_values = dict(values)
        for arg, default in zip(
            factory.args.args[-len(factory.args.defaults) :], factory.args.defaults
        ):
            env_values[arg.arg] = resolve(default, values)
        for arg, default in zip(factory.args.kwonlyargs, factory.args.kw_defaults):
            if default is not None:
                env_values[arg.arg] = resolve(default, values)
        for node in ast.walk(factory):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "MazeGoalWrapper"
            ):
                for kw in node.keywords:
                    if kw.arg in config.environment_options:
                        env_values[kw.arg] = resolve(kw.value, env_values)
        for key, value in config.environment_options.items():
            assert value == env_values[key], (
                config.name,
                "environment",
                key,
                value,
                env_values[key],
            )
        assert config.eval_episodes == 8
        calls = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in defaults
        ]
        for call in calls:
            kw = {k.arg: k.value for k in call.keywords}
            buffer_expr = ast.unparse(kw["replay_task_buffers"])
            mode = (
                "scratch"
                if buffer_expr == "{}"
                else "recovery"
                if ("recovery" in buffer_expr or "_ft" in buffer_expr)
                else "sequential"
            )
            for task_id in range(len(config.goals)):
                context = dict(
                    values,
                    task_idx=task_id,
                    task_id=task_id,
                    seed=config.seeds[0],
                    action_dim=2,
                    initial_alpha=0.2,
                )
                expected = {k: v for k, v in defaults[call.func.id].items() if k not in ignored}
                expected.update({k: resolve(v, context) for k, v in kw.items() if k not in ignored})
                trainer = train_dqn if config.algorithm == "dqn" else train_sac
                actual = {k: p.default for k, p in inspect.signature(trainer).parameters.items()}
                actual.update(training_settings(config, mode, task_id))
                # None chooses SAC's standard -action_dim entropy target.
                if actual.get("target_entropy") is None:
                    actual["target_entropy"] = -2.0
                for key, value in expected.items():
                    assert actual[key] == value, (
                        config.name,
                        mode,
                        task_id,
                        key,
                        actual[key],
                        value,
                    )
                assert training_seed(config, config.seeds[0], mode, task_id) == resolve(
                    kw["seed"], context
                )
        reports.append(
            dict(
                experiment=config.name,
                seeds=config.seeds,
                goals=len(config.goals),
                phases=len(calls),
                status="all trainer arguments match",
            )
        )
    return reports


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))
