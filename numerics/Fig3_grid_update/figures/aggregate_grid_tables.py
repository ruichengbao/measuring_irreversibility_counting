#!/usr/bin/env python3
"""Rebuild grid-specific plotting tables from the periodic per-realization results.

No trajectory simulation or optimization is rerun. Every K uses its own saved
cross-validated lag, phase and fitted function. Pass the directory containing the 270 *_result.json files as --source.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def aggregate(source, output):
    files = sorted(Path(source).glob('*_result.json'))
    records = [json.loads(f.read_text()) for f in files]
    d = pd.DataFrame(records).sort_values(['E', 'N', 'rep'])
    if len(d) != 270:
        raise ValueError(f'Expected 270 saved realizations, found {len(d)}')
    if d.duplicated(['N', 'E', 'rep']).any():
        raise ValueError('Duplicate realization keys')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for K in [5, 10, 20, 160]:
        rows = []
        for (N, E), a in d.groupby(['N', 'E']):
            true = a.true_rate.mean()
            estimates = a[f'estimator_K{K}']
            est = estimates.mean()
            ratios = estimates / a.true_rate
            rows.append(dict(
                N=N, E=E, true_rate=true,
                true_rate_sd=a.true_rate.std(ddof=1),
                true_rate_thermal_sem=np.sqrt(np.sum(a.true_rate_sampling_se**2))/len(a),
                estimator_rate=est, estimator_rate_sd=estimates.std(ddof=1),
                quality_factor_Q=est/true,
                quality_factor_mean_of_ratios=ratios.mean(),
                quality_factor_sd=ratios.std(ddof=1),
                realizations=len(a), boxes_per_side=K,
                lag_min=a[f'lag_K{K}'].min(), lag_max=a[f'lag_K{K}'].max()))
        pd.DataFrame(rows).to_csv(output/f'fig3_K{K}_summary.csv', index=False)
    print(f'Aggregated {len(d)} saved realizations into four 54-row tables.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True,
                   help='periodic directory containing the per-realization JSON files')
    p.add_argument('--output', type=Path, default=ROOT/'plotting_tables')
    a = p.parse_args()
    aggregate(a.source, a.output)
