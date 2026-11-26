# Phase-resolved density-bound analysis

Run from the package root after the simulator has produced the raw files:

```bash
python analysis/analyze_all.py --workers 2
python analysis/aggregate_full_results.py
python analysis/validate_exact_single_particle.py
python analysis/validate_selected_quadrature.py
python analysis/validate_coarsening.py
python analysis/audit_outputs.py
```

`--wait` keeps the launcher running until the remaining simulation groups are
complete. Complete groups are skipped on resumption. A single group can be
recomputed with, for example,

```bash
python analysis/analyze_full_group.py 5 1.0
```

Requirements are Python with NumPy, SciPy and pandas, and a C++17 compiler.
The endpoint-histogram shared library is compiled automatically. The figure
script additionally requires Matplotlib. All default paths are relative to
the package, not to the original workspace.

## Inputs

For each `engine/raw/Nnnn_Ee.e_rr` prefix, the analysis reads the engine's
`_meta.json`, `_forward.bin`, `_reverse.bin` and `_origins.csv`. Their exact
layouts are documented by the engine. The original integer-period results
are read from `baseline/analysis_runs/*_result.json`. These saved results
provide both the unchanged true entropy-production averages and the existing
integer-period candidate, including its training-only selection score.

The full physical parameter grid is N = 1, 2, 5, 10, 20, 50, 60, 80, 100 and
E = 0.1, 0.2, 0.4, 0.5, 0.8, 1.0, with five original particle realizations.
The observation grids have K = 5, 10, 20, 40, 80, 160 boxes per side.

## Probability calculation

For a window ending at phase j/M with duration k/M, forward relative-pair
histograms use the saved forward frames at both endpoints. Reverse histograms
use the same forward endpoint as their initial state and the simulated reverse
endpoint at elapsed time k/M. All ordered particle pairs, including identical
particle indices, are included. The reverse displacement probability is
transposed by reversing its box-displacement index. No equality of forward
and reverse distributions is assumed.

The collective Gaussian drift is integrated over the actual window. Positive
cloud-in-cell histograms and positive Gaussian/box-kernel contractions retain
rare tails without an FFT floor. The auxiliary histogram sizes are S = 320 for
N <= 20 and S = 640 for larger N. All coarser observation grids are obtained
from K = 160 by the exact stochastic coarsening matrix.

## Training and independent evaluation

For each target realization, the other four realizations provide the training
histograms. Their histograms are transformed using the target's fixed
collective drift and diffusion parameters, as in the original analysis.
Inner leave-one-out validation among those four peers chooses the clipping
cap and either the full fitted log ratio or its odd part. The odd part is a
trial function, not an assertion that the reverse protocol can be omitted.

Every fitted function is evaluated through

```
sum(P_forward * f) - sum(P_reverse_transposed * expm1(f)).
```

The target realization is excluded from fitting, phase/lag selection, partition
selection and the decision to retain the original integer-period candidate.
Diagonal bins are retained. Individual finite-sample scores are not clipped
to zero or to the true entropy production.

For M = 8, recorded lags are k/M, k = 1,...,8. The new family contains both a
common lag averaged over all phases and partitions of one complete period
into nonoverlapping windows. A common lag uses the sum of its M edge scores
divided by k. A partition uses the sum of its edge scores. Dynamic programming
tests every cyclic starting phase and uses training scores only.

The primary expanded strategy uses a one-standard-error training guard. The
four inner-validation fold scores are summed along the selected schedule with
the same normalization as the final bound. The new schedule is used only when
its mean validation score minus one descriptive standard error exceeds the
saved integer-strategy validation score. The unpenalized training maximum is
also retained for audit. This is a selection rule, not a confidence guarantee.

For N = 1 the endpoint probabilities are exact wrapped Gaussian probabilities.
The trial function is the exact log ratio and all four validation values are
identical. The separate validation tables examine M = 8, 16 and 32; the main
data use M = 8 uniformly for every N.

`validate_selected_quadrature.py` reconstructs the selected functions and
schedules for N = 2 and 5 at E = 1 from the original S = 320 training data,
then reevaluates the held-out target at S = 640. It performs no reselection.
Across K = 10, 20 and 160, the largest individual relative change is below
0.015 percent, compared with mean gains of 19 to 46 percent over the original
integer-period strategy at K = 160.

## Outputs

`results/*_result.json` records each target's baseline rate, phase-resolved
rate, unpenalized expanded rate and guarded expanded rate for every K. It also
records the training means, validation-fold scores, selected lags, phase,
partition, function caps and function types.
The JSON field `selected_has_subperiod` describes the selected schedule within
the phase-resolved candidate family, even when the expanded strategy retains
the integer-period candidate. It does not describe the final expanded choice.

`results/*_edge_scores.csv` contains all phase-window trials, their training
and independently evaluated scores, and their four inner-validation scores.
`results/*_probability_checks.csv` records normalization and nonnegativity
checks.

`summary/replicates.csv` is the tidy per-realization table.
`summary/fig3_K*_summary.csv` contains one row per original (N,E) point.
The column prefix `short_` denotes the phase-resolved family, which includes
the full-period endpoint k=M as well as all smaller lags. The prefix `expanded_`
denotes the guarded union with the original integer-period family. All means
are arithmetic means over the five realizations. The corresponding `_sd`
columns give realization spread, not standard errors. Training sets overlap
between outer folds, so these five estimates must not be interpreted as five
statistically independent confidence replicates.

For clarity, the tidy table includes `phase_candidate_has_subperiod` and
`expanded_has_subperiod` separately. Their group means are
`phase_candidate_subperiod_fraction` and `expanded_subperiod_fraction`.
The older column names `selected_has_subperiod` and `subperiod_fraction` are
retained as aliases of the phase-candidate fields for plotting compatibility.
The column `short_fraction_selected` gives the fraction of realizations for
which the expanded strategy actually chose the phase-resolved candidate.

The validation directory additionally records direct coarse-grid contraction
checks and the complete output audit. Coarsening from K = 160 agrees with
direct Gaussian/bin integration at the coarser K values to floating-point
precision in the representative forward and reverse windows tested.

The raw data suffice to reproduce every probability, fit and table. Large
intermediate probability arrays are therefore not stored separately.
