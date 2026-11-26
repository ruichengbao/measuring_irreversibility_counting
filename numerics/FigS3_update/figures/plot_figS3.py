#!/usr/bin/env python3
"""Reproduce Supplemental Fig. S4 from compact steady-state and periodic plotting tables.

The left panel compares the independently selected density and weak-FTUR rates.
The right panel rescales the existing periodic Fig. 3 data by N. No simulation,
lag selection, or refitting is performed by this plotting script.
"""
from pathlib import Path
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, ScalarFormatter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FIELDS = [.1, .2, .5, .8, 1.]
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 12.,
    'mathtext.fontset': 'stix', 'axes.labelsize': 14.,
    'axes.linewidth': 1.1, 'xtick.labelsize': 12., 'ytick.labelsize': 12.,
    'legend.fontsize': 10., 'lines.linewidth': 2., 'lines.markersize': 4.5,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.transparent': False,
})


def draw(steady_path, periodic_path, output, K=5):
    steady = pd.read_csv(steady_path)
    steady = steady[steady.K == K].copy()
    if len(steady) != 45:
        raise ValueError(f'Expected 45 steady-state (N,E) points at K={K}, got {len(steady)}')
    if not np.isfinite(steady[['Q_density', 'Q_tur']].to_numpy()).all():
        raise ValueError('Non-finite or missing steady-state values')
    if steady.duplicated(['N', 'E']).any():
        raise ValueError('Duplicated steady-state plotting keys')
    periodic = pd.read_csv(periodic_path)
    periodic = periodic[np.isclose(periodic.E, .1)].sort_values('N')
    if len(periodic) != 9:
        raise ValueError('Expected nine periodic system sizes')
    if not np.isfinite(periodic[['true_rate', 'true_rate_sd',
                                'estimator_rate', 'estimator_rate_sd']].to_numpy()).all():
        raise ValueError('Non-finite or missing periodic values')
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.2, 5.15),
                                gridspec_kw={'width_ratios': [1.28, 1.]})
    fig.subplots_adjust(left=.075, right=.985, bottom=.145, top=.90, wspace=.33)
    for ax in (a, b):
        ax.set_xlim(-3, 105)
        ax.set_xticks([0, 20, 40, 60, 80, 100])
        ax.set_xlabel(r'$N$')
        ax.tick_params(length=4, width=.9, top=True, right=True)
    a.axhline(1., color='#aaaaaa', ls='--', lw=1.1, zorder=0)
    for E, color in zip(FIELDS, COLORS):
        rows = steady[np.isclose(steady.E, E)].sort_values('N')
        if len(rows) != 9:
            raise ValueError(f'Missing system sizes for E={E}')
        a.plot(rows.N, rows.Q_density, color=color, marker='s', ls='-')
        a.plot(rows.N, rows.Q_tur, color=color, marker='^', ls='--')
    a.set_ylabel('Normalized estimator')
    upper = max(1.05, 1.06 * steady[['Q_density', 'Q_tur']].max().max())
    a.set_ylim(-.025, upper)
    a.set_yticks(np.arange(0, 1.01, .2))
    fields = [Line2D([], [], color=c, lw=2., label=rf'$E={e:.2f}$')
              for e, c in zip(FIELDS, COLORS)]
    color_legend = a.legend(handles=fields, loc='upper center', bbox_to_anchor=(.5, .92),
                            ncol=5, frameon=False, handlelength=1.2, columnspacing=.8,
                            handletextpad=.35, fontsize=9.9, borderaxespad=0.)
    methods = [Line2D([], [], color='#333333', marker='s', ls='-',
                      label=r'$\dot\sigma^{\rm est,*}/\dot\sigma$'),
               Line2D([], [], color='#333333', marker='^', ls='--',
                      label=r'$\dot\sigma^{\rm TUR,*}/\dot\sigma$')]
    a.legend(handles=methods, loc='upper center', bbox_to_anchor=(.5, .80),
             ncol=2, frameon=False, handlelength=2.3, columnspacing=2.,
             fontsize=11., borderaxespad=0.)
    a.add_artist(color_legend)

    N = periodic.N.to_numpy()
    true = periodic.true_rate.to_numpy() / N
    true_sd = periodic.true_rate_sd.to_numpy() / N
    est = periodic.estimator_rate.to_numpy() / N
    est_sd = periodic.estimator_rate_sd.to_numpy() / N
    b.errorbar(N, true, yerr=true_sd, color='#4c78a8', marker='o', ls='-',
                capsize=0, label=r'$\sigma_{[0,\mathcal{T}]}/(N\mathcal{T})$')
    b.errorbar(N, est, yerr=est_sd, color='#72b7b2', marker='s', ls='--',
                capsize=0, label=r'$\sigma^{\rm est,*}_{[0,\mathcal{T}]}/(N\mathcal{T})$')
    b.set_ylabel('Entropy production / $N$')
    lo = min(np.min(true-true_sd), np.min(est-est_sd))
    hi = max(np.max(true+true_sd), np.max(est+est_sd))
    b.set_ylim(max(0, lo-.10*(hi-lo)), hi+.16*(hi-lo))
    b.yaxis.set_major_locator(MaxNLocator(nbins=5))
    b.ticklabel_format(axis='y', style='sci', scilimits=(-3,-3), useMathText=True)
    b.legend(loc='upper right', frameon=True, borderpad=.45, handlelength=2., fontsize=10.5)
    for label, ax in zip(['(a)', '(b)'], [a, b]):
        ax.text(-.15, 1.035, label, transform=ax.transAxes, fontsize=19,
                ha='left', va='bottom')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(output.with_suffix('.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--steady', type=Path, default=ROOT/'data'/'steady_summary.csv')
    p.add_argument('--periodic', type=Path, default=ROOT.parent/'Fig3_grid_update'/'plotting_tables'/'fig3_K20_summary.csv')
    p.add_argument('--grid', type=int, default=5)
    p.add_argument('--output', type=Path, default=ROOT/'figures'/'FigS4')
    args = p.parse_args()
    draw(args.steady, args.periodic, args.output, args.grid)
