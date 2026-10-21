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

To keep an existing robotics/PMO environment intact, use a dedicated MADE
environment from the repository root instead:

```bash
UV_PROJECT_ENVIRONMENT=.venv-made uv sync --locked --no-dev --extra notebooks --extra materials
uv pip install --python .venv-made/bin/python --no-deps \
  'made @ git+https://github.com/diffractivelabs/MADE.git@bffda0c9a1fa1904b35c0ef67b455881c975df86'
.venv-made/bin/python -m ipykernel install --user --name tbtrl-made --display-name "TBTRL MADE"
```

Select **TBTRL MADE** in the notebook. TBTRL's core dependencies and editable
source are installed in this environment too. After an exact resync, reinstall
the separately pinned MADE package. No API key is needed in smoke mode; the
full notebook prompts for it using `getpass` if the kernel environment lacks it.

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

## Revised MADE pilot (after the 2026-10-19 results)

The original executed notebook is preserved byte-for-byte in
[`experiments/recorded/TBTRL_MADE_Pilot_20261019.ipynb`](../experiments/recorded/TBTRL_MADE_Pilot_20261019.ipynb).
The working notebook is cleared and identifies its new protocol as
`made-transfer-v2-capacity-regularization`. These are development settings, not
established improvements. Full ORB training must be rerun before judging them.
The generator, oracle, rewards, five chemical systems, 50-proposal campaigns,
five campaigns and the user's single-seed pilot remain the same.

| Setting | Original MADE pilot | Revised MADE pilot |
| --- | --- | --- |
| Replay payload limit | 64 MiB | 1 GiB (allocated as data arrive) |
| Recent/archive payload allowance | 32/32 MiB | 512/512 MiB |
| Recent/archive transition limits | 250/250 | 500/2,000 |
| Admission | Tail reduced immediately by similarity | Keep 40% tail; discard only under archive capacity pressure |
| Historical sampling | Uniform over stored transitions | Balanced over original task IDs |
| TD batch / warmup | 16 / 16 | 32 / 32 |
| Covariance batch | Same 16 TD samples | Separate 128 samples, half current/half historical when available |
| Covariance target variance | 1 | 1/64 |
| Historical TD coefficient | 1 | 0.5 |
| Covariance coefficient | 0.01 | 0.1 |
| Task-separation coefficient | 0.01 | 1 |
| Norm coefficient / threshold | 0.001 / 7.9 | Unchanged safety penalty |
| Notebook progress interval | Every proposal | Every 10 proposals and campaign end |

The covariance loss is now applied to `phi / sqrt(v)` in this notebook, i.e.
it measures `||Cov(phi)/v - I||_F^2 / d`. With `v=1/d`, its target centred
squared norm is one, leaving room inside the existing norm bound for a mean
component carrying value magnitude. A 16-sample covariance has rank at most
15, so it cannot match a 64-dimensional identity. The larger regularisation
batch reduces this constraint once enough distinct transitions exist; repeated
samples during early warmup still cannot manufacture extra rank. Sampling is
without replacement whenever the relevant pool is large enough.

The task-separation threshold stays at cosine 0.85. Its coefficient is increased
because the original embeddings became almost parallel. Historical TD is
downweighted rather than removed, and balanced task sampling avoids letting the
largest archived task dominate rehearsal. These choices require ablation; scalar
loss size alone is not a gradient measurement. Logs now include weighted losses,
weighted gradient norms before clipping, current/historical gradient cosine,
representation spread/norms and maximum task cosine.

The norm penalty is deliberately not forced to activate. It guards excessive
embedding magnitude; it does not by itself create diverse representations.

Similar tasks can share an archive group without losing samples. When slots or
bytes are exhausted, the largest group is thinned, starting with its most
represented original task and oldest samples. The recent buffer is cleared at
task end. Task identity, reward, next candidates and terminal flags survive
grouping. This MADE option does not change the legacy PMO defaults or maze/Fetch
trainers. All replay bytes exclude Python/model/oracle/dataset overhead.

ORB still uses CPU on Apple Silicon (CUDA when available); its device is separate
from the TBTRL learner's device. Pinned MADE ignores `logfile` in `relax_kwargs`,
so the adapter redirects stdout/stderr **only during synchronous oracle and
reference-evaluation calls** to `oracle-details.log`. FIRE lines and oracle
warnings no longer fill the notebook; errors still propagate. Inspect this file
when diagnosing relaxation. Run one experiment at a time in a notebook kernel:
Python stream redirection is process-wide for the duration of each call.

The plots now show weighted regulariser losses/gradients, numeric task similarities,
and endpoint discovery counts by campaign. The summary explicitly distinguishes
mean mSUN/AUDC over campaigns from last-campaign mSUN; campaigns are not treated
as independent seeds.

### RL audit and limits

- The learner is **candidate-action Double DQN**, with no separate SAC actor.
  It selects the next action using the online network and evaluates that same
  action using the target network. Targets use only saved valid next candidates.
- `gamma=1` and the incremental discovery reward make a campaign's return equal
  its final discovery count. The 50-proposal campaign is a finite task horizon:
  the final transition is terminal and never bootstraps into the next campaign.
  The Q bound is 64, sufficient for at most 50 additional discoveries here.
- Sequential mode carries both encoders, target and Adam state; scratch resets
  all three. Epsilon resets per task and decays to 0.05 after the first third of
  its budget; it does not reset per campaign. One optimiser/soft-target update
  is made per transition after warmup (`tau=0.01`, learning rate `1e-4`).
- Historical learning uses the original task's reward/context. It cannot safely
  recompute a MADE reward for another chemistry or current hull. Replay calls
  neither the oracle nor evaluation-only ground-truth metrics.
- The hand-built observation summarises history and is not guaranteed Markov.
  Candidate descriptors omit many aspects of chemistry. Better representations
  remain a separate experiment, not an effect established by retuning.
- The pinned native oracle returns relaxed **energies**, but the native environment
  attaches the original proposal geometry to phase-diagram entries and novelty
  checks. This upstream behaviour is preserved; mSUN must be interpreted under
  that convention, not as experimental validation of relaxed distinct materials.

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

The candidate pool is ragged, unlike maze actions. The default (legacy PMO) candidate replay
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

## Controlled MADE study and resuming

Use [TBTRL_MADE_Transfer_Study.ipynb](../experiments/working/TBTRL_MADE_Transfer_Study.ipynb)
for the next experiments. The executed pilot notebook is preserved. The study
keeps the revised pilot configuration, but separates exploration, TD replay and
regularisation random-number streams. Therefore rerun seed 42 under this protocol;
do not include the old pilot as one of its replicates.

The notebook defaults to three seeds, five systems and scratch/TBTRL. Each method
uses five 50-proposal online campaigns per task. The optional method sets are:

| Method | Transfer | Historical sampling | Covariance / separation |
| --- | --- | --- | --- |
| Scratch | New network, target and optimiser each task | None | Pilot coefficients within current task; separation is zero with one task |
| TBTRL | Network, target, optimiser and archive | Task-balanced | Pilot coefficients |
| Fine-tuning | Network, target and optimiser | None; no archive | Both off |
| Uniform replay | Network, target, optimiser and same tail archive | Uniform over stored transitions | Both off |
| No covariance | As TBTRL | As TBTRL | Only covariance off |
| No separation | As TBTRL | As TBTRL | Only separation off |
| Smaller archive | As TBTRL, at most 100 archived transitions | As TBTRL | Pilot coefficients |

Uniform replay here controls the **sampling distribution**, not a different
reservoir admission scheme. It uses the same 40% tail and capacity-driven archive
allocation, retaining original task labels. The smaller archive keeps the recent
buffer and byte limit unchanged; only the archive's transition limit changes
(100 in the full configuration; scaled down in smoke mode). All methods retain
norm safety bounds/penalties and the same factorised architecture. Separate
ablations are needed to isolate factorisation itself or similarity-based eviction.

### What is resumable?

Keep `RUN_ID`, settings and software versions unchanged and rerun the notebook
with `RESUME=True`. After **each completed campaign**, an atomic `resume.pt`
contains:

- online and target networks, Adam state and regulariser state;
- recent and archived transitions, original task IDs and grouping state;
- exploration/replay/regularisation and global RNG states;
- native observed phase diagram, discovery history, reference energies and
  evaluation counters; oracle weights and credentials are excluded;
- completed campaign logs, losses and elapsed committed-work time.

Restoration bypasses MADE environment construction, avoiding repeated reference
and proposal evaluations for completed campaigns. The next reset creates the next
campaign's candidate pool. If interrupted during campaign 4, campaigns 1–3 remain
committed and campaign 4 starts again. Up to one unfinished campaign's proposals
and reference setup may be repeated. The interrupted campaign's previous calls
are real extra compute; result budgets/time describe committed work, not all
aborted attempts. This is **not** interruption recovery inside an ORB relaxation.

Completed tasks have `complete.pt` and `results.json`; completed jobs are skipped.
Training, retention and recovery all use this mechanism. A process lock prevents
two kernels writing the same run. Manifests reject changes to task order, seeds,
methods, configs, device, recorded dependency versions, implementation hashes and
notebook-provided oracle/environment/data fingerprints. Choose a new `RUN_ID` for
changes. Old pilot checkpoints cannot be resumed because they lack optimiser,
replay and native-environment state. A manifest is a consistency check, not a
promise of bitwise reproducibility across hardware or nondeterministic kernels.

Only load your own local checkpoint files: restoration uses pickle for native
phase diagrams and replay arrays. Runtime files remain ignored by Git. Full
learner checkpoints are retained at task boundaries for later evaluation/recovery;
these can occupy several GiB across seeds/methods. That disk storage is separate
from the bounded in-memory replay payload. Recovery additionally retains its
campaign-boundary checkpoints. Frozen evaluation checkpoints omit replay and Adam
state, avoiding large archive copies for every probe. Existing completed task plots are redisplayed inline on resume. Rerunning
the summary cells regenerates aggregate inline plots without oracle calls.

### Retention and recovery

`RUN_RETENTION=False` and `RUN_RECOVERY=False` by default. The notebook reports
proposal budgets before either is enabled; reference evaluations are additional.
Evaluation does not update parameters, optimiser or replay. Greedy frozen policies
use held-out task/campaign seeds shared across methods and stages. Scratch gets
one evaluation of each independent model on its own task. Sequential methods
get all learned tasks evaluated after every acquisition. Heatmap diagonals are
post-acquisition reference performance; later rows show retention.

Each recovery branch loads a fresh copy of the final sequential learner and
retrains one earlier task for a fixed budget (default two campaigns). Main training
and other recovery branches are untouched. All tasks are evaluated before recovery
and after every recovery campaign, with the same held-out seeds used for retention.
The table reports the first measured boundary regaining 90% of that task's own
post-acquisition mSUN. Missing/zero references and failure within budget are
explicitly reported; passing at step zero means no recovery was needed. Probe
calls are reported separately and do not count as training proposals.

Summary curves average campaigns **within each seed**, then show mean ±1 sample SD
across seeds. Campaigns are not independent replicates. Candidate diagnostics show
composition-level proposal counts/rewards, predicted Q versus realised remaining
campaign return, feature covariance eigenvalues and current/replay gradient cosine.
Returns follow an exploratory changing policy, so the Q scatter is diagnostic,
not a direct unbiased test of optimal Q calibration. Neither covariance nor task
cosine alone demonstrates behavioural retention.

Run the dedicated fresh-kernel offline check with:

```bash
PYTHONPATH=src .venv/bin/python scripts/check_made_study.py
```

It exercises the new notebook in smoke mode, including retention/recovery and a
second pass that must reuse all completed work. It never uses Materials Project
credentials or downloads ORB weights.
