"""Independent, named ablations for the continuous Maze transfer experiment."""

from copy import deepcopy

EXPERIMENTS = (
    "historical_reset",
    "matched_reset",
    "temperature_carry",
    "scaled_regularization",
    "policy_retention",
)
ACTORS = ("baseline", "factorised_only", "mlp_replay", "factorised_tbtrl")


def configure_transfer(config, experiment="temperature_carry", actor_variant="factorised_tbtrl"):
    """Return an independent config; each preset documents one incremental change.

    Input is the original factorised-actor config. Historical reset keeps its
    numerical settings. All other presets match initial alpha across phases.
    Scaled regularisation and policy retention each extend temperature_carry,
    rather than silently combining both experimental changes.
    """
    if experiment not in EXPERIMENTS or actor_variant not in ACTORS:
        raise ValueError(f"Choose experiment from {EXPERIMENTS} and actor_variant from {ACTORS}.")
    if config.algorithm != "sac" or config.actor_type != "factorised":
        raise ValueError("Start from the factorised SAC experiment configuration.")
    result = deepcopy(config)
    result.entropy_transfer = (
        "reset" if experiment in ("historical_reset", "matched_reset") else "carry"
    )
    if experiment != "historical_reset":
        for settings in (result.training, result.scratch, result.first_task, result.recovery):
            settings["initial_alpha"] = 0.1
    if actor_variant in ("baseline", "mlp_replay"):
        result.actor_type, result.actor_model = "mlp", {}
    if actor_variant in ("baseline", "factorised_only"):
        result.training["actor_tbtrl"] = None
    elif actor_variant == "mlp_replay":
        result.training["actor_tbtrl"] = {"replay_loss_coef": 1.0}
    if experiment == "scaled_regularization":
        if actor_variant != "factorised_tbtrl":
            raise ValueError("scaled_regularization requires factorised_tbtrl.")
        options = result.training["actor_tbtrl"]
        rank = result.actor_model.get("latent_dim", 16)
        options["covariance_target_variance"] = options["phi_norm_target"] ** 2 / (2 * rank)
    if experiment == "policy_retention":
        if actor_variant not in ("factorised_tbtrl", "mlp_replay"):
            raise ValueError("policy_retention requires an actor replay variant.")
        result.training["actor_tbtrl"]["retention_kl_coef"] = 0.01
    warmup = result.training["warmup_steps"]
    result.training["transfer_probe_steps"] = sorted(
        {0, max(0, warmup - 1), warmup, warmup + 100, warmup + 1000, warmup + 5000}
    )
    result.training["actor_gradient_diagnostics"] = True
    result.name = f"TBTRL_GridworldMaze_SAC_Transfer-{experiment}-{actor_variant}"
    return result.validate()
