# Research figures

All working notebooks use the same rendering code in
`tbtrl.experiments.figure_style`, `diagnostic_plots`, `diagnostics` and `plots`.
Training progress still prints directly into the notebook cell. After each goal,
`NotebookDiagnostics` displays figures inline and exports them beside the
checkpoint. No plotting settings change the trainer, replay buffer, loss
coefficients, stopping criteria or experiment configuration.

## Exports and reuse

The default exports are **300 dpi PNG, PDF and SVG**, with embedded TrueType
fonts in PDFs and editable text in SVGs. Lines, labels and markers remain vector
objects; heatmaps are raster images within vector files. Inline previews use
120 dpi to keep notebook output sizes manageable. Use the exported files for a
paper rather than a screenshot of the notebook.

Per-goal files include `learning`, `critic-losses`, `actor-losses`, `actor-gradients`,
`transfer-probes`, `embeddings-*`, `drift-*`, `parameter-changes`, `q-policy` and
`rollouts`, where the relevant data exist. Summary calls export
`figures/run-overview.*` and `figures/cost-comparison.*` under the run directory.
Parameter changes are shown once per goal, instead of repeated for every encoder.
Generic environments receive numerical diagnostics. Maze spatial plots require
a layout and two-dimensional observations. Fetch uses the same numerical plots
and inline RGB rollout GIFs in place of Q maps, arrows and maze trajectories.
`NotebookDiagnostics(gif_episodes=1)` controls the number of simulator episodes
per trained goal. GIFs include goal, seed, timestep, return, distance and success;
`rollouts.json` stores their reset state and evaluation metadata. Each GIF also
has a final PNG frame. GIFs are raster simulator views; the numerical figures
retain the shared vector/300 dpi export formats.

Use one immutable style object throughout a notebook to adapt the figures to a
venue. Dimensions are in inches and font sizes in points. The default width of
7.2 inches suits a full-width multipanel figure; inspect the final layout when
using a narrower width. Matplotlib settings are scoped to rendering calls.

```python
from tbtrl.experiments.diagnostics import NotebookDiagnostics
from tbtrl.experiments.figure_style import FigureStyle
from tbtrl.experiments.plots import plot_run, plot_comparison

style = FigureStyle(width=7.2, font_size=9, dpi=600, formats=("pdf", "png"))
diagnostics = NotebookDiagnostics(style=style)
# Pass diagnostics as on_task_end when running the existing experiment.
# Progress and per-goal figures remain in that execution cell.
overview = plot_run(output, style=style)
comparison = plot_comparison(output, style=style)
```

Existing committed notebook outputs are preserved. To refresh figures from a
saved run, including the latest replay experiment, use:

```python
from tbtrl.experiments.render import render_saved_run

figures = render_saved_run(output, display=True)
```

This writes to `output/publication/` by default and returns that path. It reads
`summary.json`, task checkpoints, loss logs and optional `diagnostics.json`;
it does not retrain or change those inputs. Numeric figures reuse saved values.
Spatial figures evaluate the saved models on CPU in separate seeded diagnostic
environments, so regenerated rollouts may differ from GPU rollouts. Set
`spatial=False` to render saved numerical data only, without constructing models
or running environments. A `figure-manifest.json` records the source and style.
Python, NumPy and Torch random states are preserved. Checkpoints use the current
version-1 format; missing diagnostics are not fabricated.

## Reading the figures

- **Consistent visual identities:** scratch is blue, sequential transfer orange,
  independent recovery green, with distinct markers and line styles. Goal IDs
  retain their colours across per-goal panels. Loss components have matching
  colours in the actor-objective and gradient panels.
- **Learning:** return and success occupy separate axes. Success is a percentage;
  environment steps and optimiser updates are explicitly distinguished. All-goal
  bars outline the current goal and hatch future goals in sequential runs.
  Checkpoint success is marked unavailable when the trainer did not record it.
- **Costs:** unsolved tasks show consumed budget, explicitly marked as unsolved.
  Skipped recovery uses a hollow diamond at zero cost. Comparison curves include
  unsuccessful budgets and report seed counts. Bands are one *sample standard
  deviation* across seeds, not confidence intervals; one seed has no band.
  Each mode requires complete seed/goal coverage. Wall time is in minutes and
  depends on hardware. Accumulated recovery sums independent trials that each
  restart from the final sequential model.
- **Retention:** heatmap rows identify the goal just trained; columns identify
  the goal evaluated, including future goals. Colours use the fixed 0–100%
  scale and small matrices include numerical annotations. Per-goal success is
  an empirical evaluation mean, not a confidence bound. Recovery/transfer probes
  use their own evaluation seeds, as stated on the figures.
- **Actor objectives and gradients:** weighted terms include their configured
  coefficients. Critic TD plots show the unweighted logged losses; total weighted
  critic objectives have their own figure. Symlog axes
  retain zero values and state their linear region. Gradient alignment is
  undefined when either gradient is zero; these values are omitted instead of
  misleadingly appearing orthogonal. Negative cosine indicates local conflict,
  not a direct measurement of forgetting.
- **Representations:** goal similarity uses a fixed −1 to +1 colour scale.
  PCA axes show explained variance and are fitted independently for each panel;
  positions cannot be compared directly across stages. Raw factor norms use
  empirical cumulative distributions with sample counts, so differently sized
  probe sets can be compared. Covariance eigenvalues are ordered by magnitude.
  Drift compares corresponding probes across sequential stages; a changing
  embedding does not itself establish a loss of task performance.
- **Spatial diagnostics:** comparable Q panels share colour limits and an
  explicit colourbar. Critic disagreement has a separate nonnegative scale.
  Policy arrows follow the environment's actual coordinate system, where the
  displayed maze's y coordinate increases downwards. Rollouts identify start,
  endpoint, goal, success, return and length; paths are not smoothed.

These changes improve presentation and interpretation. They do not turn a
single-seed result into a statistical claim or establish convergence from a
smoke run. Choose the experimental seed count and uncertainty analysis for the
claim being made, and check exported figures at their final publication size.
