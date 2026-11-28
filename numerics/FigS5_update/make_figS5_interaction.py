#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fig. S5 of the SM: weak-field coefficient of the counting estimator in the exactly
solvable harmonic Brownian cluster (SM Sec. VIII C), as a function of the spring strength
kappa (panel a) and of N (panel b), together with the coefficient of the true EP, 3N/2.

Self-contained: needs only numpy, scipy, matplotlib (preinstalled on Colab).
Colab: upload this file and run `!python make_figS5_interaction.py` (or paste it into one cell).
Output: ./figures_out/FigS_interaction.pdf and .png (a few seconds).

Model (beta = mu = q = D = 1, L = 5, period T = 1, lag tau = T, M = 20 bins):
  pair energy U = (kappa/2) sum_{a<b} (y_a - y_b)^2 on unwrapped coordinates, centre R
  periodic, observed positions (R + y_a) mod L, field E(t) = E[1 + sin(2 pi t/T)].
  v_s = 2 D tau/N + 2D/(mu kappa N) (1 - 1/N)(1 - e^{-z}),
  v_d = 2 D tau/N + 2D/(mu kappa N) (1 - 1/N + e^{-z}/N),       z = mu kappa N tau,
  C_d(E) = (1/M^2)[1 + 2 sum_r sinc^2(r/M) F_r cos(2 pi r d/M - k_r mu q E tau)],
  F_r = e^{-k_r^2 v_s/2}/N + (1 - 1/N) e^{-k_r^2 v_d/2},
  A = (2 M T/tau) sum_d [dC_d/dE(0)]^2 / C_d(0)            (closed form, no fitting).
kappa = 0 means independent particles; the true EP per period is (3/2) N E^2 for all kappa.
"""
from pathlib import Path
import numpy as np
from scipy.special import roots_legendre
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

L = 5.0
MU = Q = D = TAU = PERIOD = 1.0
OUT = Path('figures_out')
OUT.mkdir(exist_ok=True)


def variances(n, kappa, tau=TAU):
    if kappa == 0:
        return 2 * D * tau, np.inf
    if np.isinf(kappa):
        return 2 * D * tau / n, 2 * D * tau / n
    z = MU * kappa * n * tau
    base = 2 * D * tau / n
    pref = 2 * D / (MU * kappa * n)
    return (base + pref * (1 - 1 / n) * (-np.expm1(-z)),
            base + pref * (1 - 1 / n + np.exp(-z) / n))


def bins_fourier(var, E, M, tau=TAU, modes=512):
    if np.isinf(var):
        return np.full(M, 1.0 / M ** 2), np.zeros(M)
    r = np.arange(1, modes + 1, dtype=float)
    k = 2 * np.pi * r / L
    amp = np.sinc(r / M) ** 2 * np.exp(-var * k * k / 2)
    th = 2 * np.pi * np.outer(np.arange(M), r) / M - k[None, :] * MU * Q * E * tau
    return (1 + 2 * np.cos(th) @ amp) / M ** 2, 2 * np.sin(th) @ (amp * k * MU * Q * tau) / M ** 2


def bins_quadrature(var, E, M, tau=TAU, order=64):
    """Positive wrapped-Gaussian integration (used when Fourier sums cancel)."""
    if np.isinf(var):
        return np.full(M, 1.0 / M ** 2), np.zeros(M)
    h = L / M
    x, w = roots_legendre(order)
    u = np.concatenate(((x - 1) * h / 2, (x + 1) * h / 2))
    wt = np.tile(w * h / 2, 2) * (h - np.abs(u)) / L
    drift = MU * Q * E * tau
    nim = int(np.ceil((abs(drift) + 12 * np.sqrt(var)) / L)) + 2
    img = np.arange(-nim, nim + 1)
    zz = np.arange(M)[:, None, None] * h + u[None, :, None] - drift + L * img[None, None, :]
    dens = np.exp(-zz * zz / (2 * var)) / np.sqrt(2 * np.pi * var)
    return (np.einsum('duw,u->d', dens, wt),
            np.einsum('duw,u->d', dens * zz / var, wt) * MU * Q * tau)


def bins(var, E, M, tau=TAU):
    C, dC = bins_fourier(var, E, M, tau)
    if np.min(C) < 1e-12:
        C, dC = bins_quadrature(var, E, M, tau)
    return C, dC


def corr_cluster(n, kappa, E, M=20, tau=TAU):
    vs, vd = variances(n, kappa, tau)
    cs, ds = bins(vs, E, M, tau)
    if n == 1:
        return cs, ds
    cd, dd = bins(vd, E, M, tau)
    return cs / n + (1 - 1 / n) * cd, ds / n + (1 - 1 / n) * dd


def weak_coeff(n, kappa, M=20, tau=TAU):
    C, dC = corr_cluster(n, kappa, 0.0, M, tau)
    rev = (-np.arange(M)) % M
    C = (C + C[rev]) / 2
    dC = (dC - dC[rev]) / 2
    return PERIOD / tau * 2 * M * np.sum(dC ** 2 / C)


def est_rate(n, kappa, E, M=20, tau=TAU):
    C, _ = corr_cluster(n, kappa, E, M, tau)
    rev = (-np.arange(M)) % M
    return PERIOD / tau * M / 2 * np.sum((C - C[rev]) * np.log(C / C[rev]))


plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 11., 'mathtext.fontset': 'stix',
    'axes.labelsize': 13., 'axes.linewidth': 1.05, 'xtick.labelsize': 10.5,
    'ytick.labelsize': 10.5, 'legend.fontsize': 9.6, 'lines.linewidth': 1.8,
    'lines.markersize': 4., 'xtick.direction': 'in', 'ytick.direction': 'in',
    'pdf.fonttype': 42, 'ps.fonttype': 42,
})
COLORS = ['#4c78a8', '#72b7b2', '#54a24b', '#e89c38']
MARKERS = ['o', 's', '^', 'D']


def figure():
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.25))
    fig.subplots_adjust(left=.095, right=.985, bottom=.19, top=.90, wspace=.37)
    for ax, lab in zip(axs, ['(a)', '(b)']):
        ax.grid(True, ls='--', lw=.5, color='#dddddd')
        ax.set_axisbelow(True)
        ax.tick_params(length=4, width=.8)
        ax.text(-.16, 1.045, lab, fontsize=17, transform=ax.transAxes, ha='left', va='bottom')
    a, b = axs
    kap = np.logspace(-3, 1, 81)
    for i, N in enumerate([1, 5, 20, 50]):
        a.loglog(kap, [weak_coeff(N, k) for k in kap], color=COLORS[i], lw=1.8, label=rf'$N={N}$')
    Ns = np.unique(np.r_[np.arange(1, 11), np.round(np.geomspace(10, 100, 31)).astype(int)])
    for i, k in enumerate([.01, .1, 1.]):
        b.loglog(Ns, [weak_coeff(int(n), k) for n in Ns], color=COLORS[i], lw=1.8,
                 marker=MARKERS[i], markevery=max(1, len(Ns) // 7), ms=4., label=rf'$\kappa={k:g}$')
    b.loglog(Ns, 1.5 * Ns, color='k', ls='--', lw=1.3, label=r'true EP: $3N/2$')
    a.set_xlabel(r'$\kappa$'); a.set_ylabel(r'$\mathcal{A}_{N,\kappa}$')
    b.set_xlabel(r'$N$'); b.set_ylabel(r'$\mathcal{A}_{N,\kappa}$')
    a.set_xlim(.001, 10.)
    a.legend(frameon=False, loc='lower right', ncol=2, columnspacing=1.)
    b.set_ylim(4e-4, 3e3)
    b.legend(frameon=False, loc='upper left', fontsize=9., ncol=2, columnspacing=.8, handlelength=1.8)
    fig.savefig(OUT / 'FigS_interaction.pdf', bbox_inches='tight')
    fig.savefig(OUT / 'FigS_interaction.png', dpi=220, bbox_inches='tight')
    plt.close(fig)


def checks():
    # closed-form coefficient against the exact finite-field estimator at small field
    r = est_rate(20, .1, 1e-3) / (weak_coeff(20, .1) * 1e-6)
    assert abs(r - 1) < 1e-5, r
    # independent and rigid limits
    assert abs(weak_coeff(5, 1e-9) - weak_coeff(5, 0.0)) < 1e-8
    assert abs(weak_coeff(5, 1e9) - weak_coeff(5, np.inf)) < 1e-6
    # the estimator never exceeds 2/3 of the true EP in this model
    for k in [.01, .1, 1.]:
        for e in [.01, .1, .5, 1.]:
            assert est_rate(20, k, e) <= (2 / 3) * 1.5 * 20 * e * e + 1e-12
    print('checks passed')


if __name__ == '__main__':
    checks()
    figure()
    print('wrote', OUT / 'FigS_interaction.pdf')
