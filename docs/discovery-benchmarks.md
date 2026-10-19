# Molecular and materials discovery

The two independent notebooks are:

- [PMO molecular optimisation](../experiments/working/TBTRL_PMO_Molecular_Optimisation.ipynb).
- [MADE materials discovery](../experiments/working/TBTRL_MADE_Materials_Discovery.ipynb).

They compare **scratch** with **sequential TBTRL**, using the same factorised
candidate network, target-task budgets, candidate-generation settings, seeds,
exploration schedule and optimiser settings. They are new benchmark adapters;
the maze/Fetch trainers, configurations and existing notebook results are unchanged.
Full research budgets are the default. Set `SMOKE = True` explicitly for plumbing
checks. No positive learning or transfer result is asserted by these checks.

## Installation

The notebooks are tracked with Git LFS. After cloning or pulling this branch,
materialise the two discovery notebooks before opening them:

```bash
git lfs install
git lfs pull --include="experiments/working/TBTRL_PMO_*.ipynb,experiments/working/TBTRL_MADE_*.ipynb" --exclude=""
```

Python 3.12 is required. PMO's optional dependencies are separate from robotics:

```bash
uv sync --locked --group dev --extra notebooks --extra pmo
.venv/bin/python -m ipykernel install --user --name tbtrl-pmo --display-name "TBTRL PMO"
.venv/bin/jupyter lab experiments/working/TBTRL_PMO_Molecular_Optimisation.ipynb
```

For MADE (or both notebooks in one environment):

```bash
uv sync --locked --group dev --extra notebooks --extra pmo --extra materials
uv pip install --python .venv/bin/python --no-deps \
  'made @ git+https://github.com/diffractivelabs/MADE.git@bffda0c9a1fa1904b35c0ef67b455881c975df86'
.venv/bin/python -m ipykernel install --user --name tbtrl-discovery --display-name "TBTRL discovery"
.venv/bin/jupyter lab experiments/working/TBTRL_MADE_Materials_Discovery.ipynb
```

Run the notebook with that installed kernel. Use `.venv/bin/python` and
`.venv/bin/jupyter` after installing MADE: a subsequent default `uv sync`/`uv run`
can remove packages outside the lockfile. `--no-deps` on MADE is intentional:
the `materials` extra supplies its tested local environment/oracle/workflow
dependencies without installing unrelated cloud services and optional generators.
Upstream eagerly imports MACE/DSPy even for random-generator experiments, so those
are included. ORB is pinned at 0.5.4 because later releases changed the API used
by this MADE revision. Ordinary `pip install -e '.[notebooks,pmo,materials]'`
followed by the same pinned MADE install is also supported; `uv.lock` records
the tested dependency resolution.

For a **full** MADE run, set `MATERIALS_PROJECT_API_KEY` in your kernel's environment
before starting it. The key is never saved in experiment outputs. ORB downloads
its pretrained weights on first use. With `GENERATOR="random"` (default), no LLM
or Chemeleon API/model is used. To experiment with the upstream Chemeleon
generator, install its `chemeleon-dng` dependency using the upstream instructions,
then select `GENERATOR="chemeleon"`; that optional generator is not covered by
the offline smoke test. Both modes must use the same generator/weights.

## Native mechanisms and deliberate adaptations

| | PMO | MADE |
| --- | --- | --- |
| Upstream | [`wenhao-gao/mol_opt`, `2da631be`](https://github.com/wenhao-gao/mol_opt/tree/2da631be85af8d10a2bb43f2de76a03171166190) | [`diffractivelabs/MADE`, `bffda0c9`](https://github.com/diffractivelabs/MADE/tree/bffda0c9a1fa1904b35c0ef67b455881c975df86) |
| Preserved | MolDQN valid atom/bond edits, reward per edit, canonical score cache, TDC objectives, 40-step finite episodes, C/N/O, ring sizes 5/6, no bonds between rings, initial `C`, radius-2/1024-bit fingerprints | Native planner, cached candidate generator, no-op filter, crystal oracle/relaxation, SMACT checks, novelty matcher, phase-diagram updates and evaluation metrics |
| Replaced | Original noisy/prioritised MolDQN optimiser | Candidate scorer/selector in the native generator workflow |
| Budget | 10,000 **unique canonical molecules** per objective/run | 50 **proposal evaluations** per campaign, five campaigns per task/run |
| Default tasks | QED, Albuterol Similarity, Mestranol Similarity, Perindopril MPO, Ranolazine MPO | Li–O, Na–O, K–O, Mg–O, Ca–O |
| Primary plots | Top-10 score vs calls, losses, top molecule SVGs | Native mSUN discovery curves for every campaign, losses, observed phase diagram |

PMO's environment is vendored with its MIT notice. Changes to that file are
Python 3 imports, an inline RDKit valence helper and formatting; its edit and
reward algorithms are unchanged. TDC 0.3.6 is the version used by upstream.
It imports a removed `rdkit.six` helper; `load_oracle` supplies the equivalent
standalone `six` module only during import and removes the alias afterwards.
The five default native scorers are checked with the installed RDKit version.
Extra learned TDC oracles may require downloads/legacy model compatibility;
they are not part of the default experiment.

MADE uses 0.1 eV/atom stability tolerance, initial known (near-)stable phases,
SMACT filtering, a 20-atom limit, a 1,024-candidate native cached pool and an ORB
oracle with FIRE relaxation (`fmax=.02`, 500 steps, cell relaxation). Reference
energies are recomputed through the selected oracle, not silently taken from
an absent reference file. The random generator is MADE's native baseline, not
an approximation of its Chemeleon agent. These runs do not reproduce the whole
paper's set of agents or chemical systems.

## TBTRL and candidate actions

The shared candidate trainer reuses `FactorisedQNetwork` and the existing
covariance, goal-separation and norm penalties:

\[
Q_\theta(s,a,g)=\phi_\theta(s,a)^\top\psi_\theta(g),\qquad
\|\phi\|_2\leq8,\;\|\psi\|_2\leq8.
\]

The input for an action is the molecule/crystal's semantic descriptor. An action
index is only a temporary reference into the current valid candidate list.
Replay stores its descriptor and the **actual next candidate descriptors**, so
changing the candidate order or count does not change the meaning of an action.
Double-DQN targets are:

\[
a^*=\arg\max_{a'\in\mathcal A(s')} Q_\theta(s',a',g),\quad
y=r+\gamma(1-d)Q_{\bar\theta}(s',a^*,g),
\]

where finite episode/campaign and oracle-budget endpoints have `d=1`. The loss is

\[
\mathcal L=\mathcal L_{\mathrm{TD,current}}
+\lambda_R\mathcal L_{\mathrm{TD,archive}}
+\lambda_C\mathcal L_{\mathrm{covariance}}
+\lambda_S\mathcal L_{\mathrm{separation}}
+\lambda_N\mathcal L_{\mathrm{raw\ norm}}.
\]

The notebook parameter cell exposes all coefficients and exploration settings.
Sequential runs carry both encoders, target network and Adam state. Scratch
reinitialises these at each task and uses only current-task experience. Both use
the same factorised architecture and current-task penalties. PMO task descriptors
are one-hot IDs in the configured objective catalogue; MADE descriptors identify
the elements present (118 fixed dimensions). Future task rewards never enter
training or separation losses before that task is encountered.

MADE does not provide an RL reward. The adapter adds

\[
r_t=D(H_{t+1})-D(H_t),\qquad \gamma=1,
\]

where `D` counts native newly discovered structures that remain stable relative
to the **observed** hull. A new hull can invalidate previous discoveries, so
negative rewards are possible. Return telescopes to final discoveries. Neither
ground-truth precision/recall nor unqueried energies enter reward or policy
inputs. Features summarise observed composition/geometry, remaining budget,
observed energies and candidate novelty proxies. This is an **approximate history
representation**; it does not establish full Markov sufficiency.

## Memory and oracle accounting

The candidate pool is ragged, unlike maze actions. The new candidate replay
therefore enforces **both** a fixed transition capacity (500) and a numpy payload
byte budget (64 MiB), split between recent and archived experience. At task end,
the most recent 40% of the retained recent buffer is eligible for admission.
Learned task similarity reduces redundant admission and groups related tasks;
at the group limit the most similar pair is consolidated. Larger groups are
thinned when archive slots or bytes are exhausted. At least one incoming
original-task example is admitted before budget rebalancing. The completed
individual buffer is then cleared.

Unlike goal relabelling in maze/Fetch, molecular objectives can require paid
oracle labels and materials rewards depend on the historical hull. Consequently
every archived sample keeps its **original task, reward, terminal flag and next
candidate set**, even when groups merge. No oracle is called by a gradient update.
The previous maze/Fetch global-replay implementation is unchanged.

Memory tables show retained samples, payload bytes, original task counts, group
sizes and admission/merge decisions. This is a bound on replay, **not total
process memory**: model weights, Python bookkeeping, task descriptors, oracle
caches, native candidate pools and result histories have separate costs. PMO's
score cache is bounded by its per-task query budget and feature cache by 2,048
entries. MADE resets its pool per campaign and saves all campaigns' proposals.

PMO counts only newly scored canonical molecules. A 20×budget transition limit
prevents infinite cached revisits; hitting it is reported as an under-budget
run, not successful completion. Its top-10 AUC matches upstream's 100-query
trapezoidal convention but does not extrapolate unspent budget. MADE counts
every proposal and separately reports elemental/initial-phase evaluation costs,
including resets. Timings include task environment setup; shared dataset/model
loading in the notebook setup cell is outside per-task timings. Candidate
generation is unlabelled, not free computation.

## Reading the results

Logs and plots appear directly in the running notebook cell after every task.
Each task saves its config/provenance, JSONL progress, loss components, inference
checkpoint, native results and PNG/PDF/SVG figures. Checkpoints contain model
weights and dimensions, not a resumable oracle/environment state. The output
directory must be new, so rerunning cannot silently mix runs.

PMO saves ranked molecule/query data and vector molecular drawings. MADE saves
native metrics for every campaign, observed phase-diagram states and every
campaign's proposal CIFs; the displayed phase diagram is the last campaign.
Native AF/EF fields computed without a supplied baseline should not be interpreted
as paired transfer statistics. Use the plotted curves and AUDC/mSUN tables for
the implemented comparison.

Aggregate curves show mean ±1 sample SD across seeds; no band is shown for one
seed. MADE averages campaigns **within each seed first**, avoiding false
replication. These are online optimisation curves, not an uncharged validation
set. The first task should match between modes. Later speedups must be assessed
at equal target-oracle counts, with all source-task and reference costs reported.
The five-task order is a starting experiment; repeat alternative orders and
seeds before drawing scientific conclusions.

## Validation

```bash
.venv/bin/python -m pytest tests/test_discovery.py
.venv/bin/python scripts/check_discovery_notebooks.py
```

The notebook check executes both complete notebooks in fresh kernels with two
tasks and both modes, real updates/replay, inline figures and result exports.
PMO uses native TDC scores and molecular edits. MADE uses its native environment,
planner/generator and Lennard-Jones oracle with an explicit offline dataset.
It tests interfaces/budgets, not realistic crystal energies or ORB convergence.
Full Materials Project/ORB and optional Chemeleon runs require their external
data/models and are deliberately not performed by CI. Core/robotics checks do
not install these optional research dependencies.
