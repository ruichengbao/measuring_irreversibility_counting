# Estimating Irreversibility via Cross-Correlations and Random Coarse-Graining

This repository contains the simulation and analysis code for the revised manuscript. The numerical examples ask how much entropy production (EP) can be inferred from the asymmetry of two-time cross-correlations. In the four-state model a noisy measurement returns one label. In the particle model each observation returns the number of particles in every spatial bin.

**[Many-Body Simulation Methods (PDF)](MANY_BODY_SIMULATION_METHODS.pdf)** explains how Fig. 3 is computed from two-time correlations of box counts. The simulation averages over collective translation to reduce sampling error in those correlations. The note gives the integration scheme, the evaluation on independent realizations, and the numerical checks, including Eqs. (S41)–(S43) and (S47)–(S48) cited in the Supplemental Material. [LaTeX source](numerics/Fig3_corrected/MANY_BODY_SIMULATION_METHODS.tex)

**To reproduce the current Fig. 3**, install the Python dependencies and run the command below from the repository root. The script reads the supplied plotting tables and writes a vector PDF and a PNG in `numerics/Fig3_interacting_inset/`.

~~~sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python numerics/Fig3_interacting_inset/plot_fig3_interacting_inset.py --grid 20
~~~

The recorded Python stack uses Python 3.12 with NumPy, SciPy, pandas and Matplotlib. Running the particle dynamics or compiling analysis kernels requires a C++17 compiler available as `g++`. The reference simulations used GCC 14.2 and libstdc++ on Linux. Particle sizes and charges follow the archived seed assignments on both GNU and Apple standard libraries. Thermal trajectories can differ across libraries because their normal random-number generators differ. Three supplementary plots also use `pdflatex` with `standalone`, TikZ and PGFPlots 1.18.

## Find a calculation

| Figure | Physical comparison | Source and inputs |
| --- | --- | --- |
| Fig. 2 and Fig. S1 | How readout errors affect EP recovery in a ring and in two multicycle networks | [Models, parameter tables and curves](numerics/inherited/exact_error_recovery_study_bundle/) |
| Fig. 3 | Integer-period counting bound in the interacting system, with 20 × 20 bins. The inset varies bin width at `N = 5` and `E = 0.1`. | [Periodic simulation and analysis](numerics/Fig3_corrected/), [main panels and inset](numerics/Fig3_interacting_inset/) |
| Fig. S2 | Noisy single-label readout of the four-state push–pull model under steady and periodic driving | [Generator and reference curves](numerics/inherited/revised_v1.1/) |
| Fig. S3 | Effect of changing the spatial resolution to 5, 10 or 160 bins per side on the same trajectories | [Figure source](numerics/FigS_boxes/FigS_boxes.tex), [grid-specific tables](numerics/Fig3_grid_update/plotting_tables/) |
| Fig. S4 | Steady-state density bound versus the fluctuation-theorem thermodynamic uncertainty relation for a specified counting current (5 × 5 bins), and periodic EP per particle (20 × 20 bins) | [Steady simulation and analysis](numerics/FigS3_update/), [plot](numerics/FigS3_update/figures/plot_figS3.py) |
| Fig. S5 | Counting bound in a solvable harmonic cluster versus the true EP coefficient `3N/2`, and the finite-field approach to the weak-field prediction | [Panels (a,b) and three-panel figure](numerics/FigS5_update/), [harmonic model and panel (c) data](numerics/revision_3rd/) |
| Fig. S6 | Gain from observing short windows at different phases and running the corresponding reversed field schedules | [Forward/reverse simulation and analysis](numerics/Reverse_protocol_lag_study/), [figure source](numerics/Reverse_protocol_lag_study/figures/FigS_phase_windows.tex) |

## Reproduce the figures from the supplied tables

Run each command from the repository root. The source directory for Fig. S4 retains its working name, `FigS3_update`.

~~~sh
(cd numerics/FigS_boxes && pdflatex -halt-on-error FigS_boxes.tex)
python numerics/FigS3_update/figures/plot_figS3.py
(cd numerics/FigS5_update && python make_figS5_interaction.py && pdflatex -halt-on-error FigS5_harmonic.tex)
(cd numerics/Reverse_protocol_lag_study/figures && pdflatex -halt-on-error FigS_phase_windows.tex)
~~~

The finite-state examples run independently of the particle simulations. The first command regenerates the three recovery curves, the second plots their supplied reference tables together, and the third calculates the steady and periodic push–pull curves and saves `four_state_random_cg_outputs/figures/FigS2.pdf`.

~~~sh
python numerics/inherited/exact_error_recovery_study_bundle/scripts/generate_interior_recovery_study.py
python numerics/inherited/exact_error_recovery_study_bundle/regenerate_exact_recovery_family_improved_v2.py
python numerics/inherited/revised_v1.1/scripts/generate_four_state_random_cg_example.py
~~~

## Recompute the numerical results

The repository includes simulation parameters and seeds, per-realization reference values, plotting tables and validation summaries. The commands below generate the raw trajectories locally. Each block starts at the repository root and runs in its own subshell. Reduce `--workers` to fit the available CPU and memory. The complete many-particle grid is substantially more expensive than redrawing the figures.

### Periodic driving and spatial resolution

The periodic calculation covers nine particle numbers, six field amplitudes and five realizations at each point. It uses 240 interacting simulations and 30 exact single-particle calculations. The code simulates the relative coordinates and averages analytically over the collective Gaussian displacement, reducing sampling error in the same measured count correlations. Each held-out interacting realization is scored with a function, lag and initial phase chosen using the other four realizations. [Model, estimator and validation methods](numerics/Fig3_corrected/NUMERICAL_METHODS.md)

~~~sh
(
  cd numerics/Fig3_corrected
  python simulation/run_simulations.py --workers 4
  python analysis/analyze_all_groups.py --workers 2
  python analysis/analyze_exact_n1.py
  python analysis/aggregate_results.py
  python ../Fig3_grid_update/figures/aggregate_grid_tables.py --source data/analysis_runs
  python ../Fig3_interacting_inset/aggregate_inset.py --source data/analysis_runs
  python ../Fig3_interacting_inset/plot_fig3_interacting_inset.py --grid 20
)
~~~

To check aggregation without simulating, use `numerics/Reverse_protocol_lag_study/baseline/analysis_runs` as the `--source` for the grid and inset scripts when invoking them from the repository root. It contains the 270 integer-period reference records. Each group’s quality factor is the mean estimated EP rate divided by the mean true rate. The error bars on the rates show sample standard deviations across five realizations. For `N = 100` and `E = 1`, the 20 × 20 table gives a quality factor of **0.41556**. [Fig. 3 numerical values](numerics/Fig3_grid_update/plotting_tables/fig3_K20_summary.csv)

### Steady driving

The steady calculation uses five field amplitudes, including the same five particle realizations per condition as in the periodic study. The left panel of Fig. S4 uses the steady simulation below. Its right panel reads the periodic 20 × 20 table.

~~~sh
(
  cd numerics/FigS3_update
  python engine/run_steady.py --workers 4
  python analysis/exact_steady_n1.py
  python analysis/run_all_analysis.py --workers 2
  python analysis/aggregate_steady.py --output data
  python figures/plot_figS3.py
)
~~~

### Reversed driving

Fig. S6 starts reverse branches at the endpoint of the forward process and applies the field schedule in reverse chronological order. The analysis compares integer-period observations with windows as short as one eighth of a period, on the same particle realizations. In the single-particle case at `E = 1`, the recovered fraction increases from **0.36184** to **0.93323**. The table includes conditions for which the held-out estimate decreases. [Fig. S6 numerical values](numerics/Reverse_protocol_lag_study/analysis/summary/fig3_K20_summary.csv)

~~~sh
(
  cd numerics/Reverse_protocol_lag_study
  python engine/run_full_grid.py --workers 4 --resume
  python engine/validate_all.py
  python analysis/analyze_all.py --workers 2
  python analysis/aggregate_full_results.py
  python analysis/audit_outputs.py
)
~~~

The reverse analysis uses the supplied integer-period reference records by default. To compare against a fresh periodic run, pass `--old ../Fig3_corrected/data/analysis_runs` to both `analyze_all.py` and `audit_outputs.py`. Add `--force` to `analyze_all.py` when switching baselines in an existing output directory. [Reverse trajectory format](numerics/Reverse_protocol_lag_study/engine/README.md) · [Window selection and evaluation](numerics/Reverse_protocol_lag_study/analysis/README.md)

### Analytical controls and numerical checks

~~~sh
python numerics/revision_3rd/scripts/exact_brownian_benchmark.py
python numerics/revision_3rd/scripts/harmonic_cluster.py
python numerics/revision_3rd/scripts/generate_N_scaling.py
~~~

The periodic and steady validation launchers regenerate smaller-time-step runs for comparison with production. The reverse package also tests direct spatial coarsening and the exact one-particle limit. [Periodic validation summaries](numerics/Fig3_corrected/validation/) · [Steady-state comparisons](numerics/FigS3_update/analysis/validation/) · [Reverse-protocol checks](numerics/Reverse_protocol_lag_study/analysis/validation/)

## Earlier implementation and license

The original [phase-locked C++ program](main_phase_locked.cpp) is retained for the earlier version of the paper. The revised small-system results and current figures use the simulation and analysis under `numerics/`.

The software is distributed under the [MIT License](LICENSE).
