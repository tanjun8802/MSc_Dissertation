"""PMO's native MolDQN molecular-edit MDP and TDC oracles, with TBTRL selection."""

import heapq
import json
import sys
from functools import lru_cache

import numpy as np

PMO_COMMIT = "2da631be85af8d10a2bb43f2de76a03171166190"
DEFAULT_ORACLES = (
    "QED",
    "Albuterol_Similarity",
    "Mestranol_Similarity",
    "Perindopril_MPO",
    "Ranolazine_MPO",
)


def load_oracle(name):
    """Load PMO's pinned TDC scorers with a scoped Python-3 compatibility alias.

    TDC 0.3.6 imports ``rdkit.six.iteritems``; modern RDKit removed its bundled
    six module. The standalone six implements the same helper. No scoring code
    is changed, and the temporary alias is removed after import.
    """
    import six
    from tdc import Oracle

    missing = "rdkit.six" not in sys.modules
    if missing:
        sys.modules["rdkit.six"] = six
    try:
        return Oracle(name=name)
    finally:
        if missing:
            sys.modules.pop("rdkit.six", None)


def make_tasks(names=DEFAULT_ORACLES, *, goal_capacity=None):
    if len(set(names)) != len(names) or not names:
        raise ValueError("Provide distinct oracle names")
    capacity = len(names) if goal_capacity is None else goal_capacity
    if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity < len(names):
        raise ValueError("goal_capacity must be a fixed integer >= the number of tasks")
    return [
        dict(name=name, goal=np.eye(capacity, dtype=np.float32)[i].tolist())
        for i, name in enumerate(names)
    ]


def top_auc(scores, budget, frequency=100, finish=False, top_n=10):
    """PMO main/optimizer.py top_auc, using scores in unique-query order.

    Same log-frequency trapezoids, zero origin and optional remaining-budget
    plateau as upstream (MIT; see _vendor/MOL_OPT_LICENSE.txt).
    """
    if not scores:
        return 0.0
    total, previous, called = 0.0, 0.0, 0
    for index in range(frequency, min(len(scores), budget), frequency):
        value = float(np.mean(sorted(scores[:index], reverse=True)[:top_n]))
        total += frequency * (value + previous) / 2
        previous, called = value, index
    value = float(np.mean(sorted(scores, reverse=True)[:top_n]))
    total += (len(scores) - called) * (value + previous) / 2
    if finish and len(scores) < budget:
        total += (budget - len(scores)) * value
    return total / budget


class MolecularEnvironment:
    def __init__(
        self,
        oracle_name,
        *,
        budget=10_000,
        episode_length=40,
        max_steps=None,
        fingerprint_length=1024,
        evaluator=None,
    ):
        from rdkit.Chem import rdFingerprintGenerator

        from ._vendor.moldqn import Molecule

        if budget < 1 or episode_length < 1:
            raise ValueError("Budgets must be positive")
        if evaluator is None:
            evaluator = load_oracle(oracle_name)
        self.evaluator = evaluator
        self.name, self.budget, self.episode_length = oracle_name, budget, episode_length
        self.max_steps = max_steps or budget * 20
        self.fingerprint_length = fingerprint_length
        self.generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=2, fpSize=fingerprint_length
        )
        # Bound feature caching independently of the global replay budget.
        self.fingerprint = lru_cache(maxsize=2048)(self._fingerprint)
        self.scores, self.query_history = {}, []
        self._top = []
        self._auc_area, self._auc_previous, self._auc_called = 0.0, 0.0, 0
        self.steps, self.episode = 0, 0
        self.env = Molecule(
            atom_types={"C", "O", "N"},
            discount_factor=0.9,
            oracle=self._score,
            init_mol="C",
            allow_removal=True,
            allow_no_modification=True,
            allow_bonds_between_rings=False,
            allowed_ring_sizes={5, 6},
            max_steps=episode_length,
        )

    def _score(self, smiles):
        from ._vendor.moldqn import canonicalize

        smiles = canonicalize(smiles)
        if smiles is None:
            raise ValueError("Native candidate generator returned invalid SMILES")
        if smiles not in self.scores:
            if len(self.scores) >= self.budget:
                raise RuntimeError("Oracle budget exhausted")
            score = float(self.evaluator(smiles))
            if not np.isfinite(score):
                raise ValueError("Oracle returned a non-finite score")
            self.scores[smiles] = score
            heapq.heappush(self._top, score)
            if len(self._top) > 10:
                heapq.heappop(self._top)
            mean = float(np.mean(self._top))
            if len(self.scores) % 100 == 0:
                self._auc_area += 100 * (mean + self._auc_previous) / 2
                self._auc_previous, self._auc_called = mean, len(self.scores)
            self.query_history.append(
                dict(
                    queries=len(self.scores),
                    smiles=smiles,
                    score=score,
                    best=max(self._top),
                    top10=mean,
                )
            )
        return self.scores[smiles]

    def _fingerprint(self, smiles):
        from rdkit import Chem

        return self.generator.GetFingerprintAsNumPy(Chem.MolFromSmiles(smiles)).astype(np.float32)

    @property
    def progress(self):
        return len(self.scores) / self.budget

    @property
    def finished(self):
        return len(self.scores) >= self.budget or self.steps >= self.max_steps

    def _observation(self, done=False):
        state = np.concatenate(
            (
                self.fingerprint(self.env.state),
                [
                    (self.episode_length - self.env.num_steps_taken) / self.episode_length,
                    self.progress,
                ],
            )
        ).astype(np.float32)
        self.candidates = [] if done else sorted(self.env.get_valid_actions())
        features = (
            np.stack([self.fingerprint(s) for s in self.candidates])
            if self.candidates
            else np.empty((0, self.fingerprint_length), dtype=np.float32)
        )
        return state, features

    def reset(self):
        self.episode += 1
        self.env.reset()
        return self._observation()

    def step(self, index):
        _, _, reward, done = self.env.step(self.candidates[index])
        self.steps += 1
        done = done or self.finished
        state, actions = self._observation(done)
        return state, actions, reward, done

    def metrics(self):
        # Incremental equivalent of upstream top_auc: avoid re-sorting every
        # historical prefix on every edit (including free cached revisits).
        count = len(self.scores)
        mean = float(np.mean(self._top)) if self._top else 0.0
        area = self._auc_area + (count - self._auc_called) * (mean + self._auc_previous) / 2
        return dict(
            queries=count,
            query_budget=self.budget,
            episode=self.episode,
            best=max(self._top, default=0),
            top10=mean,
            top10_auc=area / self.budget,
            budget_exhausted=count >= self.budget,
            step_limit_reached=self.steps >= self.max_steps,
        )

    def save(self, path):
        (path / "molecules.json").write_text(json.dumps(self.query_history, indent=2))
        (path / "benchmark.json").write_text(
            json.dumps(
                dict(
                    benchmark="PMO",
                    upstream_commit=PMO_COMMIT,
                    oracle=self.name,
                    atom_types=["C", "N", "O"],
                    allowed_ring_sizes=[5, 6],
                    initial_molecule="C",
                    episode_length=self.episode_length,
                    fingerprint_radius=2,
                    fingerprint_length=self.fingerprint_length,
                    score_cache_entries=len(self.scores),
                    fingerprint_cache_entries=self.fingerprint.cache_info().currsize,
                    max_steps=self.max_steps,
                    **self.metrics(),
                ),
                indent=2,
            )
        )

    def plot_domain(self):
        from rdkit import Chem
        from rdkit.Chem import Draw

        top = sorted(self.scores.items(), key=lambda item: item[1], reverse=True)[:8]
        return Draw.MolsToGridImage(
            [Chem.MolFromSmiles(s) for s, _ in top],
            molsPerRow=4,
            subImgSize=(350, 270),
            legends=[f"{v:.4f}" for _, v in top],
            useSVG=True,
        )
