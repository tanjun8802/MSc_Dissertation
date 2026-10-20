"""TBTRL candidate selection around MADE's native discovery workflow.

No replacement energy model, validity checker, novelty rule or phase diagram.
Only public observed state is encoded; evaluation-only reference metrics never
enter policy features, rewards or replay. The history summary is an approximate
state representation, not a claim of a fully observed Markov encoding.
"""

import json
import random
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np
import torch

MADE_COMMIT = "bffda0c9a1fa1904b35c0ef67b455881c975df86"
DEFAULT_SYSTEMS = (("Li", "O"), ("Na", "O"), ("K", "O"), ("Mg", "O"), ("Ca", "O"))


def make_tasks(systems=DEFAULT_SYSTEMS):
    from pymatgen.core import Element

    tasks = []
    for system in systems:
        if not 2 <= len(set(system)) <= 3:
            raise ValueError("Use binary or ternary systems (native phase-diagram visualisation)")
        goal = np.zeros(118, dtype=np.float32)
        for symbol in system:
            goal[Element(symbol).Z - 1] = 1
        tasks.append(dict(name="-".join(system), elements=list(system), goal=goal.tolist()))
    return tasks


class CountingOracle:
    """Count proposal AND reference-setup evaluations; no shared reward cache."""

    def __init__(self, oracle, log_path=None):
        self.oracle, self.evaluations = oracle, 0
        self.log_path = Path(log_path) if log_path is not None else None

    @contextmanager
    def _output(self):
        # The pinned MADE version ignores logfile in relax_kwargs and constructs
        # FIRE with stdout logging. Scope redirection to the synchronous oracle call;
        # never patch ASE or capture the trainer's progress/figures. Exceptions propagate.
        if self.log_path is None:
            yield
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with (
            self.log_path.open("a") as log,
            redirect_stdout(log),
            redirect_stderr(log),
        ):
            yield

    def __getattr__(self, name):
        return getattr(self.oracle, name)

    def evaluate(self, structure):
        self.evaluations += 1
        with self._output():
            return self.oracle.evaluate(structure)

    def compute_elemental_energies_from_structures(self, entries):
        # The delegate evaluates each reference internally (not through this wrapper).
        self.evaluations += len(entries)
        with self._output():
            return self.oracle.compute_elemental_energies_from_structures(entries)


def structure_features(structure):
    """Composition + cell/geometry descriptors, without predicted/oracle energies."""
    composition = np.zeros(118, dtype=np.float32)
    for element, amount in structure.composition.items():
        composition[element.Z - 1] = amount / len(structure)
    distances = structure.distance_matrix
    pair_distances = distances[np.triu_indices(len(structure), 1)]
    histogram = np.histogram(pair_distances, bins=np.linspace(0, 10, 17))[0].astype(float)
    histogram /= max(1, len(pair_distances))
    lengths = np.asarray(structure.lattice.abc) / max(structure.volume, 1e-8) ** (1 / 3)
    properties = [
        len(structure) / 20,
        np.log1p(structure.volume / len(structure)) / 5,
        float(structure.density) / 10,
        *lengths,
        *np.asarray(structure.lattice.angles) / 180,
    ]
    return np.concatenate((composition, properties, histogram)).astype(np.float32)


class MaterialsEnvironment:
    def __init__(
        self,
        dataset,
        oracle,
        *,
        seed=42,
        budget=50,
        campaigns=5,
        pool_size=1024,
        generator="random",
        generator_device="cpu",
        max_stoichiometry=20,
        start_with_all_stable=True,
        stability_tolerance=0.1,
        filter_by_smact=True,
        oracle_log_path=None,
    ):
        from made.envs import ConvexHullEnvironment

        if budget < 1 or campaigns < 1 or pool_size <= budget:
            raise ValueError("Require positive budgets and a candidate pool larger than a campaign")
        if generator not in ("random", "chemeleon"):
            raise ValueError("generator must be random or chemeleon")
        self.settings = dict(
            seed=seed,
            budget=budget,
            campaigns=campaigns,
            pool_size=pool_size,
            generator=generator,
            generator_device=generator_device,
            max_stoichiometry=max_stoichiometry,
            start_with_all_stable=start_with_all_stable,
            stability_tolerance=stability_tolerance,
            filter_by_smact=filter_by_smact,
            oracle_log_path=str(oracle_log_path) if oracle_log_path is not None else None,
        )
        self.oracle = CountingOracle(oracle, oracle_log_path)
        self.env = ConvexHullEnvironment(
            dataset=dataset,
            oracle=self.oracle,
            budget=budget,
            start_with_all_stable=start_with_all_stable,
            compute_elemental_from_oracle=True,
            stability_tolerance=stability_tolerance,
            include_near_stable_from_ground_truth=True,
            filter_by_smact=filter_by_smact,
            max_stoichiometry=max_stoichiometry,
        )
        self.campaign, self.total_queries = 0, 0
        self.campaign_metrics, self.campaign_histories, self.discoveries = [], [], []
        self.campaign_states, self.campaign_structures = [], []
        self.action_dim = 146  # 118 composition + 9 cell + 16 distance bins + 3 context

    @property
    def finished(self):
        return self.total_queries >= self.settings["budget"] * self.settings["campaigns"]

    @property
    def progress(self):
        return self.total_queries / (self.settings["budget"] * self.settings["campaigns"])

    def _discovery_count(self):
        from made.utils.convex_hull_utils import safe_e_above_hull

        return int(
            sum(
                safe_e_above_hull(self.env.observed_pd, entry) <= self.env.stability_tolerance
                for entry in self.env.newly_discovered_entries
            )
        )

    def reset(self):
        from made.agents.filters import NoOpFilter
        from made.agents.generators import CachedGenerator, ChemeleonGenerator, RandomGenerator
        from made.agents.planners.grid_search import GridSearchPlanner

        self.campaign += 1
        seed = self.settings["seed"] + 100_003 * self.campaign
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        # The constructor already initialised campaign 1; avoid a duplicate reference evaluation.
        if self.campaign > 1:
            self.env.reset()
        self.planner = GridSearchPlanner(
            max_stoichiometry=self.settings["max_stoichiometry"],
            seed=seed,
            return_all_compositions=True,
            filter_by_smact_validity=self.settings["filter_by_smact"],
        )
        base = (
            RandomGenerator(seed=seed)
            if self.settings["generator"] == "random"
            else ChemeleonGenerator(
                task="csp", device=self.settings["generator_device"], batch_size=32
            )
        )
        self.generator = CachedGenerator(
            base, num_initial_candidates=self.settings["pool_size"], cache_by_composition=False
        )
        self.filter = NoOpFilter()  # matches upstream random/chemeleon generative baselines
        self._feature_cache = {}
        self.previous_discoveries = 0
        return self._observation()

    def _observation(self, done=False):
        from made.utils.convex_hull_utils import extract_structure_from_entry
        from made.utils.structure_hash import structure_hash

        native = self.env.get_state()  # deliberately NOT get_latest_metrics()
        entries = self.env.observed_pd.all_entries
        observed = [extract_structure_from_entry(entry) for entry in entries]
        observed_features = np.stack([structure_features(s) for s in observed if s is not None])
        energies = [self.env.observed_pd.get_form_energy_per_atom(e) for e in entries]
        state = np.concatenate(
            (
                observed_features.mean(axis=0),
                [
                    self.env.query_count / self.settings["budget"],
                    self._discovery_count() / self.settings["budget"],
                    np.tanh(min(energies)),
                    np.tanh(np.mean(energies)),
                    len(entries) / (len(self.env.initial_structures) + self.settings["budget"]),
                ],
            )
        )
        if done:
            self.candidates = []
            return state.astype(np.float32), np.empty((0, self.action_dim), dtype=np.float32)
        for component in (self.planner, self.generator, self.filter):
            component.update_state(native)
        plan = self.planner.propose(native)
        self.candidates = self.filter.filter(self.generator.generate(plan, native), native)
        if not self.candidates:
            raise RuntimeError(
                "Native MADE generator/filter exhausted its candidates; increase pool_size"
            )
        features = []
        for structure in self.candidates:
            key = structure_hash(structure)
            if key not in self._feature_cache:
                self._feature_cache[key] = structure_features(structure)
            base = self._feature_cache[key]
            # Observed-hull energy and descriptor distance are available BEFORE paying the oracle.
            hull_energy = self.env.observed_pd.get_hull_energy_per_atom(structure.composition)
            hull_energy -= self.env.observed_pd.get_reference_energy_per_atom(structure.composition)
            distance = np.linalg.norm(observed_features - base, axis=1).min()
            attempts = sum(
                e.composition.reduced_formula == structure.composition.reduced_formula
                for e in self.env.proposed_entries
            )
            features.append(
                np.concatenate(
                    (
                        base,
                        [
                            np.tanh(hull_energy),
                            np.tanh(distance),
                            attempts / self.settings["budget"],
                        ],
                    )
                )
            )
        return state.astype(np.float32), np.asarray(features, dtype=np.float32)

    def step(self, index):
        observation, done = self.env.step(self.candidates[index])
        self.total_queries += 1
        count = self._discovery_count()
        # With gamma=1, rewards telescope to final native stable/unique/novel discoveries.
        reward = count - self.previous_discoveries
        self.previous_discoveries = count
        self.discoveries.append(
            dict(
                campaign=self.campaign,
                queries=self.env.query_count,
                total_queries=self.total_queries,
                discoveries=count,
                formula=observation["reduced_formula"],
                formation_energy=observation["formation_energy_per_atom"],
                e_above_hull=observation["e_above_hull"],
            )
        )
        if done:
            self.campaign_metrics.append(dict(self.env.get_latest_metrics()))
            self.campaign_histories.append(self.env.get_metrics_history())
            self.campaign_states.append(self.env.get_state())
            self.campaign_structures.append(self.env.get_proposal_structures())
        state, features = self._observation(done)
        return state, features, reward, done

    def metrics(self):
        return dict(
            queries=self.total_queries,
            query_budget=self.settings["budget"] * self.settings["campaigns"],
            campaign=self.campaign,
            campaign_queries=self.env.query_count,
            discoveries=self._discovery_count(),
            oracle_evaluations=self.oracle.evaluations,
            reference_evaluations=self.oracle.evaluations - self.total_queries,
        )

    def save(self, path):
        from monty.json import MontyEncoder

        (path / "discovery.json").write_text(json.dumps(self.discoveries, indent=2))
        (path / "native-metrics.json").write_text(
            json.dumps(self.campaign_histories, cls=MontyEncoder, indent=2)
        )
        (path / "observed-phase-diagram.json").write_text(
            json.dumps(self.env.get_state(), cls=MontyEncoder, indent=2)
        )
        (path / "campaign-states.json").write_text(
            json.dumps(self.campaign_states, cls=MontyEncoder, indent=2)
        )
        (path / "benchmark.json").write_text(
            json.dumps(
                dict(
                    benchmark="MADE",
                    upstream_commit=MADE_COMMIT,
                    settings=self.settings,
                    campaign_metrics=self.campaign_metrics,
                    **self.metrics(),
                ),
                cls=MontyEncoder,
                indent=2,
            )
        )
        for campaign, structures in enumerate(self.campaign_structures, 1):
            for i, structure in enumerate(structures):
                structure.to(filename=str(path / f"campaign_{campaign}_proposal_{i:03d}.cif"))

    def plot_domain(self):
        plot = self.env.get_pd_plot(ground_truth=False, backend="matplotlib")
        return plot.figure if hasattr(plot, "figure") else plot


def smoke_dataset(elements=("Li", "O")):
    """Explicit offline plumbing fixture, never used by full benchmark settings."""
    from made.data.chemical_system import PhaseDiagramDataset
    from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
    from pymatgen.core import Lattice, Structure

    class OfflineDataset(PhaseDiagramDataset):
        def _load_dataset(self):
            entries = []
            for element in self.elements:
                structure = Structure(Lattice.cubic(3), [element], [[0, 0, 0]])
                entries.append(
                    PDEntry(structure.composition, 0, attribute={"structure": structure.as_dict()})
                )
            return PhaseDiagram(entries)

    return OfflineDataset(list(elements))
