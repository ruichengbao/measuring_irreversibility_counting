# Constant-field many-body simulations

The physical model is unchanged from the corrected periodically driven calculation, except that the external field is constant: E(t)=E. The system is a periodic square of side 5 with kBT = 1, WCA energy scale 1, spring stiffness 5, rest length 0.001 and cutoff 2. Spring force is zero outside the cutoff. The continuously shifted spring potential gives the same force inside the cutoff. Particle sizes are drawn independently from {0.01, 0.02, 0.03}; charges from {0.9, 1, 1.1}; D = mu = 0.01/sigma.

The C++ source uses the same projected, driven Metropolis-adjusted Langevin update and pair-force cache as the corrected periodic simulations. The constant field appears in both the Gaussian proposal and Hastings acceptance. The proposal drift is tamed, but the interaction energy is not clipped. Accepted single-particle updates modify their affected force-cache entries. The cache is independently rebuilt every 1000 sweeps, and discrepancies are recorded.

Disorder seeds and particle assignments match the previous periodic simulations. Thermal seeds are independent. The production step is 0.0002 and the initial transient is 10 time units. Durations are recorded per run and range from 160 to 10000 time units. The final 160 time units are saved every 0.05 time units, including both endpoints, giving 3201 configurations per run.

The collective center of friction is independent of relative dynamics and has drift QE/Gamma and diffusion 1/Gamma. It is represented analytically. The collective entropy-production rate is E^2 Q^2/Gamma. Relative external work on any interval is exactly E[Pi(end)-Pi(start)], where Pi=sum_a q_a y_a,x. The simulation stores this boundary expression and an independent accumulated sum over steps. No work estimate is clipped to its theoretical limit.

## Files and reproduction

Compile flags: g++ -O3 -march=native -ffp-contract=off -std=c++17.

Run `python run_steady.py --workers 6`. Existing completed runs are skipped unless `--force` is passed. `jobs_manifest_steady.json` records all 200 interacting runs. The analytic single-particle case is evaluated separately by the analysis code.

For each prefix in `raw/`:

- `_meta.json`: all physical parameters, exact particle assignments, random seeds, actual step, sampling interval, durations, acceptance, cache and work checks.
- `_snapshots.bin`: little-endian float64 rows [time,y0x,y0y,...,y(N-1)x,y(N-1)y]. Coordinates are unwrapped relative positions.
- `_periods.csv`: one row per unit-time block, containing internal external work, the exact boundary expression, endpoint polarizations, interaction energy and acceptance.

The legacy keys `period`, `burn_periods`, and `production_periods` refer to unit-time blocks in this constant-field calculation; there is no driving period. The metadata separately specifies `field_protocol=constant` and `statistics_block_duration=1`.

`python run_steady_validation.py` generates two fourfold-step-refinement cases at N = 100 and E = 0.1 or 1, plus a zero-field N = 20 control. `check_steady_engine.py` checks record dimensions and times, all finite values, particle assignment consistency, center constraints, work identities and period continuity. Saved relative coordinates supply the same conditional collective average for the density estimator and for both first and second moments of the selected TUR observable.
