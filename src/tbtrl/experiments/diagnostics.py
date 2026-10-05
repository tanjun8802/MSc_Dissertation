"""Per-goal notebook diagnostics, kept outside the optimizer/update loops.

All probes are deterministic and use separate environments; diagnostics never
sample the training buffer or consume the training random generators.
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from tbtrl.evaluation import make_policy


def _array(tensor):
    return tensor.detach().cpu().numpy()


def _stats(values):
    norms = np.linalg.norm(values, axis=-1)
    return dict(
        mean=float(norms.mean()),
        std=float(norms.std()),
        min=float(norms.min()),
        max=float(norms.max()),
    )


def _cosine(values):
    unit = values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-8)
    return unit @ unit.T


def _pca(values):
    centered = values - values.mean(0)
    _, _, basis = np.linalg.svd(centered, full_matrices=False)
    projected = centered @ basis[:2].T
    return np.pad(projected, ((0, 0), (0, max(0, 2 - projected.shape[1]))))


def _serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _serializable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_serializable(item) for item in value]
    return value


class NotebookDiagnostics:
    """Display and save Q/policy maps, rollouts, embeddings and weight changes.

    Pass an instance as run_experiment(..., on_task_end=NotebookDiagnostics()).
    Each task saves diagnostics.json and PNGs next to its checkpoint. Display can
    be disabled for automated checks. Unknown environments still receive numeric
    embedding, loss, weight and learning-curve diagnostics; spatial plots require
    a maze layout and two-dimensional observations.
    """

    def __init__(self, *, display=True, rollout_episodes=8):
        self.display = display
        self.rollout_episodes = rollout_episodes
        self.history = {}

    @torch.no_grad()
    def __call__(self, *, result, config, factory, record, task_goals, before, directory):
        import matplotlib.pyplot as plt

        algorithm = config.algorithm
        model = result.network if algorithm == "dqn" else result.critic
        device = next(model.parameters()).device
        task_ids = sorted(task_goals)
        goal = config.goals[record["task_id"]]
        env = factory(goal=goal)
        states, actions = [], []
        try:
            rng = np.random.default_rng(record["seed"] + 10_000)
            for index in range(10):
                obs, _ = env.reset(seed=record["seed"] + 10_000 + index)
                states.append(obs)
                actions.append(
                    index % env.action_space.n
                    if algorithm == "dqn"
                    else rng.uniform(-1, 1, env.action_space.shape)
                )
            layout = getattr(env.unwrapped, "maze", None)
        finally:
            env.close()
        state = torch.as_tensor(np.asarray(states), dtype=torch.float32, device=device)
        goals = torch.as_tensor(
            np.asarray([task_goals[i] for i in task_ids]), dtype=torch.float32, device=device
        )
        # NumPy's uniform probes default to float64, which MPS cannot transfer.
        # Choose the dtype at construction, before moving data to the device.
        action_dtype = torch.int64 if algorithm == "dqn" else torch.float32
        action = torch.as_tensor(np.asarray(actions), dtype=action_dtype, device=device)
        # Normalized SAC runs must probe the same inputs used by their policy/critic.
        if algorithm == "sac":
            if config.training.get("normalize_state_inputs", False):
                state = result.state_normalizer.normalize(
                    state, clip=config.training.get("obs_norm_clip", 10.0)
                )
            if config.training.get("normalize_goal_inputs", False):
                goals = result.goal_normalizer.normalize(
                    goals, clip=config.training.get("obs_norm_clip", 10.0)
                )
        indices = result.buffer.chronological_indices()
        indices = indices[np.linspace(0, len(indices) - 1, min(1024, len(indices)), dtype=int)]
        replay_state = torch.as_tensor(
            result.buffer.obs[indices], dtype=torch.float32, device=device
        )
        replay_action = torch.as_tensor(
            result.buffer.actions[indices], dtype=action_dtype, device=device
        )
        if algorithm == "dqn":
            action = F.one_hot(action.long(), model.num_actions).float()
            replay_action = F.one_hot(replay_action.long().view(-1), model.num_actions).float()
            branches = {
                "q": dict(
                    psi=_array(model.goal_encoder(goals)),
                    phi_fixed=_array(model.sa_encoder(torch.cat([state, action], -1))),
                    phi_replay=_array(
                        model.sa_encoder(torch.cat([replay_state, replay_action], -1))
                    ),
                    bounded_psi=_array(model.encode_goal(goals)),
                    bounded_phi=_array(model.encode_state_action(state, action)),
                )
            }
            models = {"q": result.network, "target": result.target}
        else:
            if config.training.get("normalize_state_inputs", False):
                replay_state = result.state_normalizer.normalize(
                    replay_state, clip=config.training.get("obs_norm_clip", 10.0)
                )
            branches = {
                f"critic{head}": dict(
                    psi=_array(getattr(model, f"psi{head}_forward")(goals)),
                    phi_fixed=_array(getattr(model, f"phi{head}_forward")(state, action.float())),
                    phi_replay=_array(
                        getattr(model, f"phi{head}_forward")(replay_state, replay_action)
                    ),
                )
                for head in (1, 2)
            }
            models = {"actor": result.actor, "critic": result.critic, "target": result.target}
        changes = {}
        for group, network in models.items():
            for name, value in network.named_parameters():
                key = f"{group}.{name}"
                prefix = key.rsplit(".", 2)[0]
                old = before[key]
                delta = value.detach().cpu() - old
                entry = changes.setdefault(prefix, dict(squared_change=0.0, squared_initial=0.0))
                entry["squared_change"] += float(delta.square().sum())
                entry["squared_initial"] += float(old.square().sum())
        for entry in changes.values():
            entry["l2_change"] = entry["squared_change"] ** 0.5
            entry["relative_change"] = entry["l2_change"] / max(
                entry["squared_initial"] ** 0.5, 1e-8
            )
        previous = self.history.get((record["seed"], record["mode"]))
        for name, data in branches.items():
            data["psi_cosine"] = _cosine(data["psi"])
            data["norms"] = {key: _stats(data[key]) for key in ("psi", "phi_fixed", "phi_replay")}
            covariance = (
                np.cov(data["phi_replay"], rowvar=False)
                if len(indices) > 1
                else np.zeros((data["psi"].shape[1],) * 2)
            )
            data["covariance_eigenvalues"] = np.linalg.eigvalsh(np.atleast_2d(covariance))
            if previous is not None:
                old = previous[name]
                common = min(len(old["psi"]), len(data["psi"]))
                data["psi_drift"] = np.linalg.norm(
                    data["psi"][:common] - old["psi"][:common], axis=-1
                )
                data["phi_fixed_drift"] = np.linalg.norm(
                    data["phi_fixed"] - old["phi_fixed"], axis=-1
                )
        self.history[record["seed"], record["mode"]] = branches
        diagnostic = dict(
            task_ids=task_ids,
            goals=[task_goals[i] for i in task_ids],
            fixed_states=np.asarray(states),
            fixed_actions=np.asarray(actions),
            branches=branches,
            weight_changes=changes,
        )
        directory = Path(directory)
        (directory / "diagnostics.json").write_text(
            json.dumps(_serializable(diagnostic), indent=2, allow_nan=False) + "\n"
        )
        title = f"{record['mode']} | seed {record['seed']} | goal {record['task_id']}: {goal}"
        print(
            f"\n{title}\nSteps: {result.steps}; threshold: {result.steps_to_threshold}; time: {result.elapsed_seconds:.2f}s",
            flush=True,
        )
        for name, data in branches.items():
            print(
                f"{name} raw embedding norms: {data['norms']}\nGoal cosine matrix:\n{np.round(data['psi_cosine'], 3)}",
                flush=True,
            )
        print(
            "Encoder/actor weight changes:",
            {k: round(v["l2_change"], 6) for k, v in changes.items()},
            flush=True,
        )

        def emit(fig, name):
            fig.suptitle(title)
            fig.tight_layout()
            fig.savefig(directory / f"{name}.png", dpi=110, bbox_inches="tight")
            if self.display:
                from IPython.display import Image, display

                display(Image(filename=str(directory / f"{name}.png")))
            plt.close(fig)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        if result.evaluations:
            axes[0].plot(*zip(*result.evaluations), marker="o")
        axes[0].set(title="Evaluation return", xlabel="Environment steps")
        if algorithm == "sac" and result.success_rates:
            axes[0].plot(*zip(*result.success_rates), label="Success rate")
            axes[0].legend()
        for key in (
            ("td", "replay", "total")
            if algorithm == "dqn"
            else ("td1", "td2", "replay1", "replay2", "actor")
        ):
            axes[1].plot(
                [r["step"] for r in result.losses], [r[key] for r in result.losses], label=key
            )
        axes[1].set(title="Training losses", xlabel="Environment steps")
        axes[1].legend()
        axes[2].bar(
            range(len(record["evaluation"])), [r["success_rate"] for r in record["evaluation"]]
        )
        axes[2].set(title="Success on every goal", xlabel="Goal index", ylim=(0, 1))
        emit(fig, "learning")
        for name, data in branches.items():
            fig, axes = plt.subplots(2, 3, figsize=(15, 8))
            im = axes[0, 0].imshow(data["psi_cosine"], vmin=-1, vmax=1, cmap="coolwarm")
            fig.colorbar(im, ax=axes[0, 0])
            axes[0, 0].set(
                title=f"{name}: goal cosine similarity",
                xticks=range(len(task_ids)),
                xticklabels=task_ids,
            )
            count = len(data["psi"])
            points = _pca(np.concatenate([data["psi"], data["phi_fixed"]]))
            axes[0, 1].scatter(*points[:count].T, marker="*", label="Raw psi")
            axes[0, 1].scatter(*points[count:].T, label="Fixed raw phi")
            for i in range(count):
                axes[0, 1].annotate(str(task_ids[i]), points[i])
            axes[0, 1].set(title="Joint embedding PCA (per-goal basis)")
            axes[0, 1].legend()
            for key in ("psi", "phi_fixed", "phi_replay"):
                axes[0, 2].hist(np.linalg.norm(data[key], axis=-1), bins=15, alpha=0.5, label=key)
            axes[0, 2].set(title="Raw embedding norms")
            axes[0, 2].legend()
            axes[1, 0].plot(data["covariance_eigenvalues"], marker="o")
            axes[1, 0].set(title="Replay phi covariance spectrum")
            keys = list(changes)
            axes[1, 1].barh(keys, [changes[k]["l2_change"] for k in keys])
            axes[1, 1].set(title="Parameter L2 change")
            for key in ("psi_drift", "phi_fixed_drift"):
                if key in data:
                    axes[1, 2].plot(data[key], marker="o", label=key)
            axes[1, 2].set(title="Embedding drift since previous goal")
            if previous is not None:
                axes[1, 2].legend()
            emit(fig, f"embeddings-{name}")
        if layout is not None and state.shape[1] == 2:
            self._spatial(result, config, record, factory, np.asarray(layout), emit)

    @torch.no_grad()
    def _spatial(self, result, config, record, factory, layout, emit):
        import matplotlib.pyplot as plt

        goal = config.goals[record["task_id"]]
        model = result.network if config.algorithm == "dqn" else result.actor
        device = next(model.parameters()).device
        y, x = np.where(layout == 0)
        states = np.column_stack([x, y]).astype(np.float32)
        if config.algorithm == "sac":
            states += 0.5
        obs = torch.as_tensor(states, dtype=torch.float32, device=device)
        goals = (
            torch.as_tensor(goal, dtype=torch.float32, device=device)
            .view(1, -1)
            .expand(len(states), -1)
        )
        if config.algorithm == "dqn":
            values = _array(model.q_val_for_argmax_action(obs, goals))
            best = values.argmax(-1)
            arrows = np.array([[0, 1], [0, -1], [-1, 0], [1, 0]])[best]
            panel_values = [values[:, a] for a in range(4)] + [values.max(-1)]
            labels = ["Q(up)", "Q(down)", "Q(left)", "Q(right)", "Max Q / greedy policy"]
        else:
            if config.training.get("normalize_state_inputs", False):
                obs = result.state_normalizer.normalize(obs)
            if config.training.get("normalize_goal_inputs", False):
                goals = result.goal_normalizer.normalize(goals)
            actions = result.actor.deterministic(obs, goals)
            q1 = _array(result.critic.q1_forward(obs, actions, goals)).ravel()
            q2 = _array(result.critic.q2_forward(obs, actions, goals)).ravel()
            arrows = _array(actions)
            panel_values = [q1, q2, np.minimum(q1, q2), np.abs(q1 - q2)]
            labels = ["Q1(s, pi, g)", "Q2(s, pi, g)", "Min Q / policy", "Critic disagreement"]
        fig, axes = plt.subplots(1, len(panel_values), figsize=(4 * len(panel_values), 4))
        for ax, values, label in zip(axes, panel_values, labels):
            grid = np.full(layout.shape, np.nan)
            grid[y, x] = values
            ax.imshow(layout, cmap="Greys", origin="upper")
            im = ax.imshow(grid, cmap="viridis", origin="upper")
            fig.colorbar(im, ax=ax)
            ax.quiver(x, y, arrows[:, 0], -arrows[:, 1], color="white", scale=20)
            ax.scatter(
                goal[0] - (0.5 if config.algorithm == "sac" else 0),
                goal[1] - (0.5 if config.algorithm == "sac" else 0),
                marker="*",
                color="red",
            )
            ax.set(title=label)
        emit(fig, "q-policy")
        policy = make_policy(
            result,
            goal,
            config.algorithm,
            normalize_state=config.training.get("normalize_state_inputs", False),
            normalize_goal=config.training.get("normalize_goal_inputs", False),
            clip=config.training.get("obs_norm_clip", 10.0),
        )
        n = self.rollout_episodes
        fig, axes = plt.subplots(
            (n + 3) // 4, min(4, n), figsize=(4 * min(4, n), 4 * ((n + 3) // 4)), squeeze=False
        )
        for episode, ax in enumerate(axes.flat):
            if episode >= n:
                ax.axis("off")
                continue
            env = factory(goal=goal)
            try:
                obs, _ = env.reset(seed=record["seed"] + 500_000 + episode)
                path, total = [obs], 0.0
                while True:
                    obs, reward, terminated, truncated, info = env.step(policy(obs))
                    path.append(obs)
                    total += reward
                    if terminated or truncated:
                        break
            finally:
                env.close()
            path = np.asarray(path) - (0.5 if config.algorithm == "sac" else 0)
            ax.imshow(layout, cmap="Greys")
            ax.plot(*path.T, ".-", markersize=2)
            ax.scatter(*path[0], color="green", label="Start")
            ax.scatter(*path[-1], color="red", label="End")
            ax.scatter(
                *(np.asarray(goal) - (0.5 if config.algorithm == "sac" else 0)),
                marker="*",
                color="gold",
                s=120,
            )
            ax.set(title=f"Episode {episode}: return={total:.3f}, length={len(path) - 1}")
        emit(fig, "rollouts")
