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
from tbtrl.random import preserve_rng_state

from .diagnostic_plots import (
    plot_actor_losses,
    plot_critic_objective,
    plot_drift,
    plot_embeddings,
    plot_gradients,
    plot_learning,
    plot_parameters,
    plot_transfer,
)
from .figure_style import (
    BRANCHES,
    COLORS,
    DEFAULT_STYLE,
    MODES,
    finish,
    save_figure,
    styled,
    subplots,
)


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
    Each task saves diagnostics.json and publication PNG/PDF/SVG figures next to its checkpoint. Display can
    be disabled for automated checks. Unknown environments still receive numeric
    embedding, loss, weight and learning-curve diagnostics; spatial plots require
    a maze layout and two-dimensional observations. RGB environments use rollout GIFs.
    """

    def __init__(self, *, display=True, rollout_episodes=8, gif_episodes=None, style=DEFAULT_STYLE):
        if not isinstance(rollout_episodes, int) or rollout_episodes < 1:
            raise ValueError("rollout_episodes must be a positive integer.")
        self.style = style
        if gif_episodes is not None and (not isinstance(gif_episodes, int) or gif_episodes < 1):
            raise ValueError("gif_episodes must be a positive integer.")
        self.gif_episodes = gif_episodes
        self.display = display
        self.rollout_episodes = rollout_episodes
        self.history = {}

    @torch.no_grad()
    @preserve_rng_state()
    @styled
    def __call__(self, *, result, config, factory, record, task_goals, before, directory):
        algorithm = config.algorithm
        model = result.network if algorithm == "dqn" else result.critic
        device = next(model.parameters()).device
        task_ids = sorted(task_goals)
        goal = config.goals[record["task_id"]]
        env = factory(goal=goal)
        states, actions = [], []
        try:
            render_mode = getattr(env, "diagnostic_render_mode", None)
            rng = np.random.default_rng(record["seed"] + 10_000)
            for index in range(10):
                obs, _ = env.reset(seed=record["seed"] + 10_000 + index)
                if render_mode == "rgb_array":
                    # Fixed-start environments otherwise yield ten identical
                    # probes. Seeded short trajectories stay fixed across goals.
                    for _ in range(index):
                        obs, _, terminated, truncated, _ = env.step(
                            rng.uniform(-1, 1, env.action_space.shape).astype(np.float32)
                        )
                        if terminated or truncated:
                            break
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
            if hasattr(result.actor, "state_factors"):
                actor_psi = _array(result.actor.goal_factors(goals))
                fixed_factors = result.actor.state_factors(state)
                replay_factors = result.actor.state_factors(replay_state)
                for name, fixed, replay in zip(
                    ("actor_mu", "actor_log_std"), fixed_factors, replay_factors
                ):
                    branches[name] = dict(
                        psi=actor_psi,
                        phi_fixed=_array(fixed).reshape(-1, result.actor.latent_dim),
                        phi_replay=_array(replay).reshape(-1, result.actor.latent_dim),
                        phi_fixed_by_action=_array(fixed),
                        phi_replay_by_action=_array(replay),
                    )
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
            if "phi_replay_by_action" in data and len(indices) > 1:
                per_action = data["phi_replay_by_action"]
                covariance = np.mean(
                    [np.cov(per_action[:, j, :], rowvar=False) for j in range(per_action.shape[1])],
                    axis=0,
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
        if render_mode is not None:
            diagnostic["rollout_render_mode"] = render_mode
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

        self.render(
            result=result,
            config=config,
            record=record,
            diagnostic=diagnostic,
            directory=directory,
            factory=factory,
            layout=np.asarray(layout) if layout is not None and state.shape[1] == 2 else None,
        )

    @torch.no_grad()
    @styled
    def render(self, *, result, config, record, diagnostic, directory, factory=None, layout=None):
        """Render existing numeric diagnostics without collecting or updating anything."""
        import matplotlib.pyplot as plt

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        branches, task_ids = diagnostic["branches"], diagnostic["task_ids"]
        changes = diagnostic["weight_changes"]

        def emit(fig, name):
            from io import BytesIO

            branch = next((BRANCHES[k] for k in branches if name.endswith(f"-{k}")), None)
            context = (
                f"{MODES[record['mode']][0]} · goal {record['task_id']} · seed {record['seed']}"
            )
            finish(fig, title=f"{branch}\n{context}" if branch else context)
            try:
                save_figure(fig, directory / name, style=self.style)
                if self.display:
                    from IPython.display import Image, display

                    image = BytesIO()
                    fig.savefig(
                        image,
                        format="png",
                        dpi=self.style.display_dpi,
                        bbox_inches="tight",
                        pad_inches=0.06,
                    )
                    display(Image(data=image.getvalue()))
            finally:
                plt.close(fig)

        emit(plot_learning(result, config, record, style=self.style), "learning")
        emit(plot_critic_objective(result, config, style=self.style), "critic-losses")
        if config.algorithm == "sac":
            emit(plot_actor_losses(result, config, style=self.style), "actor-losses")
        if getattr(result, "transfer_probes", None):
            emit(plot_transfer(result, task_ids, style=self.style), "transfer-probes")
        gradient_rows = [r for r in result.losses if "actor_current_grad_norm" in r]
        if config.algorithm == "sac" and gradient_rows:
            emit(plot_gradients(gradient_rows, style=self.style), "actor-gradients")
        if changes:
            emit(plot_parameters(changes, style=self.style), "parameter-changes")
        for name, data in branches.items():
            emit(plot_embeddings(name, data, task_ids, style=self.style), f"embeddings-{name}")
            if record["mode"] == "sequential" and "psi_drift" in data:
                emit(plot_drift(data, task_ids, style=self.style), f"drift-{name}")
        if layout is not None:
            self._spatial(result, config, record, factory, np.asarray(layout), emit)
        elif factory is not None and diagnostic.get("rollout_render_mode") == "rgb_array":
            from .rollouts import render_rollout_gifs

            render_rollout_gifs(
                result,
                config,
                record,
                factory,
                directory,
                display=self.display,
                episodes=self.gif_episodes or config.eval_episodes,
            )

    @torch.no_grad()
    def _spatial(self, result, config, record, factory, layout, emit):
        from matplotlib.colors import ListedColormap, Normalize
        from matplotlib.lines import Line2D
        from matplotlib.patches import Circle
        from matplotlib.ticker import MaxNLocator

        goal = np.asarray(config.goals[record["task_id"]])
        discrete = config.algorithm == "dqn"
        model = result.network if discrete else result.actor
        device = next(model.parameters()).device
        y, x = np.where(layout == 0)
        states = np.column_stack([x, y]).astype(np.float32) + (0 if discrete else 0.5)
        obs = torch.as_tensor(states, dtype=torch.float32, device=device)
        goals = (
            torch.as_tensor(goal, dtype=torch.float32, device=device)
            .view(1, -1)
            .expand(len(states), -1)
        )
        if discrete:
            values = _array(model.q_val_for_argmax_action(obs, goals))
            arrows = np.array([[0, 1], [0, -1], [-1, 0], [1, 0]])[values.argmax(-1)]
            panels = [values[:, a] for a in range(4)] + [values.max(-1)]
            labels = [
                "Q(a₀): Δy = +1",
                "Q(a₁): Δy = −1",
                "Q(a₂): Δx = −1",
                "Q(a₃): Δx = +1",
                "Maximum Q and greedy action",
            ]
            policy_index = 4
        else:
            clip = config.training.get("obs_norm_clip", 10.0)
            if config.training.get("normalize_state_inputs", False):
                obs = result.state_normalizer.normalize(obs, clip=clip)
            if config.training.get("normalize_goal_inputs", False):
                goals = result.goal_normalizer.normalize(goals, clip=clip)
            actions = result.actor.deterministic(obs, goals)
            q1 = _array(result.critic.q1_forward(obs, actions, goals)).ravel()
            q2 = _array(result.critic.q2_forward(obs, actions, goals)).ravel()
            arrows = _array(actions)
            panels = [q1, q2, np.minimum(q1, q2), np.abs(q1 - q2)]
            labels = [
                "Critic 1 at policy action",
                "Critic 2 at policy action",
                "Minimum Q and policy action",
                "Absolute critic disagreement",
            ]
            policy_index = 2
        # Share limits across comparable Q panels; disagreement has its own nonnegative scale.
        comparable = np.concatenate(panels if discrete else panels[:3])
        low, high = float(comparable.min()), float(comparable.max())
        if low == high:
            low, high = low - 1e-6, high + 1e-6
        norm = Normalize(low, high)
        height, width = layout.shape
        extent = (-0.5, width - 0.5, height - 0.5, -0.5) if discrete else (0, width, height, 0)
        wall_map = ListedColormap(["#F4F5F7", "#454A52"])

        def maze_axes(ax):
            ax.imshow(
                layout,
                cmap=wall_map,
                vmin=0,
                vmax=1,
                origin="upper",
                extent=extent,
                interpolation="nearest",
            )
            ax.set(xlabel="Maze x (cells)", ylabel="Maze y (cells)", aspect="equal")
            ax.xaxis.set_major_locator(MaxNLocator(5, integer=True))
            ax.yaxis.set_major_locator(MaxNLocator(5, integer=True))
            ax.scatter(
                *goal,
                marker="*",
                s=85,
                color="#F0E442",
                edgecolor="#222222",
                linewidth=0.7,
                zorder=5,
            )
            if not discrete:
                ax.add_patch(
                    Circle(
                        goal,
                        config.environment_options.get("goal_radius", 0.5),
                        fill=False,
                        linestyle="--",
                        edgecolor="#222222",
                        linewidth=0.7,
                        zorder=4,
                    )
                )

        fig, axes = subplots(
            (len(panels) + 1) // 2,
            2,
            style=self.style,
            height=3.1 * ((len(panels) + 1) // 2),
            squeeze=False,
        )
        for i, (ax, values, label) in enumerate(zip(axes.flat, panels, labels)):
            maze_axes(ax)
            grid = np.full(layout.shape, np.nan)
            grid[y, x] = values
            disagreement = not discrete and i == 3
            im = ax.imshow(
                grid,
                cmap="magma" if disagreement else "viridis",
                norm=Normalize(0, max(float(values.max()), 1e-6)) if disagreement else norm,
                origin="upper",
                extent=extent,
                interpolation="nearest",
            )
            fig.colorbar(im, ax=ax, label="|Q₁ − Q₂|" if disagreement else "Q value", shrink=0.85)
            if i == policy_index:
                # angles=xy honours the downward-positive maze y axis; no manual sign flip.
                ax.quiver(
                    states[:, 0],
                    states[:, 1],
                    arrows[:, 0],
                    arrows[:, 1],
                    angles="xy",
                    scale_units="xy",
                    scale=3,
                    color="white",
                    edgecolor="#222222",
                    linewidth=0.25,
                    width=0.008,
                )
            ax.set_title(f"({chr(97 + i)}) {label}")
        for ax in list(axes.flat)[len(panels) :]:
            ax.axis("off")
        finish(
            fig,
            note=f"Goal {record['task_id']} at {tuple(float(v) for v in goal)}: yellow star; dashed circle: continuous success radius. Maze y increases downward. Q panels share a colour scale; disagreement uses a separate scale. Policy arrows: one action unit spans ⅓ cell on this diagram.",
        )
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
        fig, axes = subplots(
            (n + 1) // 2,
            min(2, n),
            style=self.style,
            height=2.7 * ((n + 1) // 2) + 0.6,
            squeeze=False,
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
            path = np.asarray(path)
            maze_axes(ax)
            ax.plot(
                *path.T,
                color=COLORS[0],
                linewidth=1.2,
                marker=".",
                markersize=2,
                markevery=max(1, len(path) // 15),
            )
            ax.scatter(
                *path[0],
                color=COLORS[0],
                marker="o",
                s=25,
                edgecolor="white",
                linewidth=0.5,
                zorder=6,
            )
            ax.scatter(
                *path[-1],
                color=COLORS[1],
                marker="s",
                s=25,
                edgecolor="white",
                linewidth=0.5,
                zorder=6,
            )
            outcome = "success" if info.get("success", False) else "not reached"
            ax.set_title(
                f"Rollout {episode + 1}: {outcome}\nReturn {total:.2f} · {len(path) - 1} environment steps"
            )
        handles = [
            Line2D([], [], color=COLORS[0], marker="o", linestyle="none", label="Start"),
            Line2D([], [], color=COLORS[1], marker="s", linestyle="none", label="End"),
            Line2D(
                [],
                [],
                color="#F0E442",
                marker="*",
                markeredgecolor="#222222",
                linestyle="none",
                label="Goal",
            ),
        ]
        axes[-1, 0].legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.23), ncol=3)
        finish(
            fig,
            note="Deterministic policy actions; separate seeded diagnostic rollouts. Environment randomness may still apply. Coordinates match the maze; paths are not smoothed.",
        )
        emit(fig, "rollouts")
