# Forward and reversed-protocol many-body simulations

The simulator retains the corrected Fig. 3 model: a two-dimensional periodic
square of side 5, kBT=1, WCA repulsion with epsilon=1 and pair diameter equal to
the mean of the two particle diameters, and a harmonic pair force with k=5,
r0=0.001 and cutoff 2. The spring force vanishes outside its cutoff. The
continuous spring potential is shifted to zero at that cutoff. Particle
mobilities are D_a=mu_a=0.01/sigma_a, with the same original particle diameters,
charges and disorder seeds used in every corresponding archived Fig. 3 run.
The field is E(t)=E[1+cos(2*pi*t)] and its period is one.

The C++ simulator uses the same projected driven tamed Metropolis-adjusted
Langevin integrator and pair cache as the previous corrected simulation.
The physical time step is 0.0002. Internal forces and thermal noise are
simulated in the coordinates relative to the center of friction. The
independent collective Gaussian motion is integrated in the subsequent
correlation analysis. These numerical operations do not alter the measured
normalized density correlations.

## Reproduction

Run from the engine directory:

    python run_full_grid.py --workers 6 --resume

The launcher compiles `simulate_shortlag.cpp` with g++ when needed. It uses
`jobs_manifest.json`, which contains all 270 runs, including the exact model
assignments and seeds. No old trajectory file is needed to rerun this grid.
Python requires numpy and pandas. The full grid covers N=1,2,5,10,20,50,60,80,100
and E=0.1,0.2,0.4,0.5,0.8,1, with five original realizations at every point.
`status.json` records progress and `*_checks.json` files record validation.

Each run discards ten initial periods. The observation-phase grid has eight
points per period. Relative positions are saved at every phase, including
time zero. The settings are:

| N | Production periods | Reverse origins per phase | Origin cycle spacing |
|---|---:|---:|---:|
| 1 | 4 | 3 | 1 |
| 2,5 | 400 | 100 | 4 |
| 10,20 | 128 | 32 | 4 |
| 50,60,80,100 | 48 | 12 | 4 |

For N=1 all relative positions are exactly zero. Consequently, correlation
analysis gives the exact single-particle Gaussian result without sampling
error. The small N=1 arrays preserve the same file interface as the many-body
runs.

## Reversed protocol

For every saved endpoint selected as a reverse origin, the simulation copies
the actual forward relative configuration, rebuilds its force cache and
starts a new independent thermal-noise stream. If the forward endpoint is
t_end, the reverse branch has positive physical elapsed time s and is driven
by E_R(s)=E(t_end-s). Its midpoint force uses E(t_end-s-dt/2). The field is not
negated and the physical integration step is not reversed.

A branch runs for one complete period and saves its endpoints at every lag
m/8, m=1,...,8. Thus the same branch supplies all reversed endpoint
distributions ending at the same forward phase. The matching forward pair
uses the saved forward configurations at t_end-m/8 and t_end. Subsequent
analysis transposes the reverse box-index pair when forming the KL bound.

The collective Gaussian displacement has the same positive integrated field
in forward and reverse experiments:

    mean_x = (Q/Gamma) * E * [tau +
               (sin(2*pi*t_end)-sin(2*pi*(t_end-tau)))/(2*pi)]
    variance_x = variance_y = 2*tau/Gamma.

The true entropy-production rates used for comparison remain the previously
converged Fig. 3 values. Newly recorded period work is an auxiliary trajectory
diagnostic and does not replace those values.

## Binary and table formats

All binary files contain native little-endian float64 values.

- `*_forward.bin`: shape (production_periods*8+1, 1+2*N). Each row contains
  time, y_0x,y_0y,...,y_(N-1)x,y_(N-1)y.
- `*_reverse.bin`: shape (n_branches,8,N,2). The second index m-1 denotes
  positive elapsed time m/8.
- `*_origins.csv`: branch_id, exact forward_end_index, end_time, phase_index,
  origin_cycle, noise_seed and numerical diagnostics. Its order is the first
  index of the reverse array. For lag m/8 the forward start row is exactly
  forward_end_index-m.
- `*_periods.csv`: work and integration-by-parts work per forward period,
  endpoint polarization, potential and acceptance diagnostic.
- `*_meta.json`: model parameters, particles, dimensions, seeds and timing.
  It is written only after the run has completed.
- `*_checks.json`: dimensions, time grid, branch phase references, center
  projection, cache accuracy and identity with the manifest assignments.

The launcher validates every completed run. Relative coordinates are stored
unwrapped, and no box histogram or microscopic configuration is discarded
before these files are written.
