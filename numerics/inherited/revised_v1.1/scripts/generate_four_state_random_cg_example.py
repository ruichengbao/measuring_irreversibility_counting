#!/usr/bin/env python3
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt

states = [(0,0),(1,0),(1,1),(0,1)]
A_steady = 1.0
A0 = 1.0
A1 = 0.8
T_period = 0.2

def mapping_four_state(x: float) -> np.ndarray:
    T = np.zeros((4,4), dtype=float)
    for i, (a,b) in enumerate(states):
        T[0,i] = x*(1-b)
        T[1,i] = (1-x)*a
        T[2,i] = x*b
        T[3,i] = (1-x)*(1-a)
    return T

def mapping_three_state(x: float) -> np.ndarray:
    T = np.zeros((3,4), dtype=float)
    for i, (a,b) in enumerate(states):
        T[0,i] = x*(1-b)
        T[1,i] = x*b + (1-x)*a
        T[2,i] = (1-x)*(1-a)
    return T

def generator_ring(kplus: float, kminus: float) -> np.ndarray:
    L = np.zeros((4,4), dtype=float)
    for j in range(4):
        L[(j+1)%4, j] += kplus
        L[(j-1)%4, j] += kminus
        L[j, j] -= (kplus + kminus)
    return L

def steady_distribution() -> np.ndarray:
    return np.ones(4, dtype=float) / 4.0

def true_epr_rate(kplus: float, kminus: float, p: np.ndarray | None = None) -> float:
    if p is None:
        p = steady_distribution()
    L = generator_ring(kplus, kminus)
    val = 0.0
    for j in range(4):
        for i in range(4):
            if i == j:
                continue
            kij = L[i, j]
            if kij > 0:
                kji = L[j, i]
                val += kij * p[j] * np.log((kij * p[j]) / (kji * p[i]))
    return float(val)

def cg_flux_matrix(T: np.ndarray, kplus: float, kminus: float, p: np.ndarray | None = None) -> np.ndarray:
    if p is None:
        p = steady_distribution()
    L = generator_ring(kplus, kminus)
    M = T.shape[0]
    F = np.zeros((M, M), dtype=float)
    for j in range(4):
        for i in range(4):
            if i == j:
                continue
            kij = L[i, j]
            if kij != 0:
                F += np.outer(T[:, i], T[:, j]) * kij * p[j]
    return F

def cg_epr_rate(T: np.ndarray, kplus: float, kminus: float, p: np.ndarray | None = None) -> float:
    F = cg_flux_matrix(T, kplus, kminus, p)
    M = T.shape[0]
    val = 0.0
    for n in range(M):
        for m in range(M):
            if m == n:
                continue
            if F[m, n] > 0 and F[n, m] > 0:
                val += F[m, n] * np.log(F[m, n] / F[n, m])
    return float(val)

def propagator_steady(kplus: float, kminus: float, tau: float) -> np.ndarray:
    return expm(generator_ring(kplus, kminus) * tau)

def joint_initial_final(K: np.ndarray, p: np.ndarray | None = None) -> np.ndarray:
    if p is None:
        p = steady_distribution()
    return (K * p).T

def correlation_matrix(T: np.ndarray, K: np.ndarray, p: np.ndarray | None = None) -> np.ndarray:
    J = joint_initial_final(K, p)
    return T @ J @ T.T

def estimator_from_C(C: np.ndarray) -> float:
    M = C.shape[0]
    val = 0.0
    for m in range(M):
        for n in range(m):
            if C[n, m] > 0 and C[m, n] > 0:
                val += (C[n, m] - C[m, n]) * np.log(C[n, m] / C[m, n])
    return float(val)

def observable_bound_rate_steady(x: float, tau: float) -> float:
    T = mapping_four_state(x)
    K = propagator_steady(np.exp(A_steady/2), np.exp(-A_steady/2), tau)
    C = correlation_matrix(T, K)
    return estimator_from_C(C) / tau

def optimize_observable_bound_rate_steady(x: float) -> tuple[float, float]:
    if x <= 1e-12 or x >= 1 - 1e-12:
        return 0.0, 0.0
    res = minimize_scalar(
        lambda t: -observable_bound_rate_steady(x, t),
        bounds=(1e-5, 10.0),
        method='bounded',
        options={'xatol': 1e-5},
    )
    return float(res.x), float(-res.fun)

def affinity_periodic(t: float) -> float:
    return A0 + A1 * np.sin(2 * np.pi * t / T_period)

def kplus_periodic(t: float) -> float:
    return np.exp(affinity_periodic(t) / 2)

def kminus_periodic(t: float) -> float:
    return np.exp(-affinity_periodic(t) / 2)

def period_average(func) -> float:
    return float(quad(func, 0.0, T_period, limit=300)[0] / T_period)

def propagator_one_period() -> np.ndarray:
    Splus = np.zeros((4,4), dtype=float)
    Sminus = np.zeros((4,4), dtype=float)
    for j in range(4):
        Splus[(j+1)%4, j] = 1.0
        Sminus[(j-1)%4, j] = 1.0
    I = np.eye(4)
    Iplus = quad(kplus_periodic, 0.0, T_period, limit=300)[0]
    Iminus = quad(kminus_periodic, 0.0, T_period, limit=300)[0]
    Lint = Iplus * Splus + Iminus * Sminus - (Iplus + Iminus) * I
    return expm(Lint)

plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 17,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 13,
    'axes.linewidth': 1.2,
    'savefig.bbox': 'tight',
    'mathtext.fontset': 'stix',
    'font.family': 'STIXGeneral',
})

colors = {
    'true': '#355C7D',
    'cg4': '#5AAE9E',
    'obs': '#D98C3F',
    'cg3': '#8C6BB1',
}
styles = {'true': '-', 'cg4': '--', 'obs': ':', 'cg3': '-.'}
linew = {'true': 2.8, 'cg4': 2.6, 'obs': 2.6, 'cg3': 2.6}

def style_axes(ax):
    for side in ['top','right','left','bottom']:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(1.2)
    ax.tick_params(direction='out', length=5, width=1.1)
    ax.set_xlim(0,1)
    ax.set_xticks(np.linspace(0,1,6))
    ax.grid(False)

def plot_curves(ax, x, true_y, cg4_y, obs_y, cg3_y, ylabel, title=None):
    ax.plot(x, true_y, styles['true'], color=colors['true'], lw=linew['true'], label='true EP')
    ax.plot(x, cg4_y, styles['cg4'], color=colors['cg4'], lw=linew['cg4'], label='4-state random CG')
    ax.plot(x, obs_y, styles['obs'], color=colors['obs'], lw=linew['obs'], label='4-state finite-time bound')
    ax.plot(x, cg3_y, styles['cg3'], color=colors['cg3'], lw=linew['cg3'], label='3-state random CG')
    ax.set_xlabel(r'random weight $x$')
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=16, pad=8)
    style_axes(ax)

def main(outdir: str = 'four_state_random_cg_outputs'):
    out = Path(outdir)
    (out / 'figures').mkdir(parents=True, exist_ok=True)
    (out / 'data').mkdir(parents=True, exist_ok=True)

    x_values = np.linspace(0.0, 1.0, 101)

    steady_true = true_epr_rate(np.exp(A_steady/2), np.exp(-A_steady/2))
    steady_rows = []
    for x in x_values:
        T4 = mapping_four_state(float(x))
        T3 = mapping_three_state(float(x))
        tau_opt, obs_rate = optimize_observable_bound_rate_steady(float(x))
        row = {
            'x': float(x),
            'sigma_true_rate': float(steady_true),
            'sigma_cg4_rate': float(cg_epr_rate(T4, np.exp(A_steady/2), np.exp(-A_steady/2))),
            'sigma_obs_rate_opt': float(obs_rate),
            'tau_opt': float(tau_opt),
            'sigma_cg3_rate': float(cg_epr_rate(T3, np.exp(A_steady/2), np.exp(-A_steady/2))),
        }
        for k in row:
            if isinstance(row[k], float) and abs(row[k]) < 1e-14:
                row[k] = 0.0
        steady_rows.append(row)
    steady_df = pd.DataFrame(steady_rows)
    steady_df.to_csv(out / 'data' / 'steady_four_state_example.csv', index=False)

    K_period = propagator_one_period()
    periodic_true = period_average(lambda t: true_epr_rate(kplus_periodic(t), kminus_periodic(t)))
    periodic_rows = []
    for x in x_values:
        T4 = mapping_four_state(float(x))
        T3 = mapping_three_state(float(x))
        row = {
            'x': float(x),
            'sigma_true_rate_period_avg': float(periodic_true),
            'sigma_cg4_rate_period_avg': period_average(lambda t, Tmap=T4: cg_epr_rate(Tmap, kplus_periodic(t), kminus_periodic(t))),
            'sigma_obs_rate_period_avg': float(estimator_from_C(correlation_matrix(T4, K_period)) / T_period),
            'sigma_cg3_rate_period_avg': period_average(lambda t, Tmap=T3: cg_epr_rate(Tmap, kplus_periodic(t), kminus_periodic(t))),
        }
        for k in row:
            if isinstance(row[k], float) and abs(row[k]) < 1e-14:
                row[k] = 0.0
        periodic_rows.append(row)
    periodic_df = pd.DataFrame(periodic_rows)
    periodic_df.to_csv(out / 'data' / 'periodic_four_state_example.csv', index=False)

    fig = plt.figure(figsize=(6.2,4.4))
    ax = fig.add_subplot(111)
    plot_curves(
        ax,
        steady_df['x'].values,
        steady_df['sigma_true_rate'].values,
        steady_df['sigma_cg4_rate'].values,
        steady_df['sigma_obs_rate_opt'].values,
        steady_df['sigma_cg3_rate'].values,
        ylabel='Entropy production rate',
        title='Nonequilibrium steady state',
    )
    ax.legend(loc='upper center', ncol=2, frameon=True, framealpha=0.95)
    fig.savefig(out / 'figures' / 'four_state_steady.png', dpi=300)
    fig.savefig(out / 'figures' / 'four_state_steady.pdf')
    plt.close(fig)

    fig = plt.figure(figsize=(6.2,4.4))
    ax = fig.add_subplot(111)
    plot_curves(
        ax,
        periodic_df['x'].values,
        periodic_df['sigma_true_rate_period_avg'].values,
        periodic_df['sigma_cg4_rate_period_avg'].values,
        periodic_df['sigma_obs_rate_period_avg'].values,
        periodic_df['sigma_cg3_rate_period_avg'].values,
        ylabel='Period-averaged entropy production rate',
        title='Periodically driven steady state',
    )
    ax.legend(loc='upper center', ncol=2, frameon=True, framealpha=0.95)
    fig.savefig(out / 'figures' / 'four_state_periodic.png', dpi=300)
    fig.savefig(out / 'figures' / 'four_state_periodic.pdf')
    plt.close(fig)

    fig = plt.figure(figsize=(10.6,4.15))
    ax1 = fig.add_subplot(121)
    plot_curves(
        ax1,
        steady_df['x'].values,
        steady_df['sigma_true_rate'].values,
        steady_df['sigma_cg4_rate'].values,
        steady_df['sigma_obs_rate_opt'].values,
        steady_df['sigma_cg3_rate'].values,
        ylabel='Entropy production rate',
    )
    ax1.text(-0.13, 1.03, '(a)', transform=ax1.transAxes, fontsize=17)
    ax1.set_title('Nonequilibrium steady state', fontsize=16, pad=8)

    ax2 = fig.add_subplot(122)
    plot_curves(
        ax2,
        periodic_df['x'].values,
        periodic_df['sigma_true_rate_period_avg'].values,
        periodic_df['sigma_cg4_rate_period_avg'].values,
        periodic_df['sigma_obs_rate_period_avg'].values,
        periodic_df['sigma_cg3_rate_period_avg'].values,
        ylabel='',
    )
    ax2.text(-0.13, 1.03, '(b)', transform=ax2.transAxes, fontsize=17)
    ax2.set_title('Periodically driven steady state', fontsize=16, pad=8)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, frameon=True, framealpha=0.95, bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout(rect=[0,0,1,0.92], w_pad=2.0)
    fig.savefig(out / 'figures' / 'FigS2.png', dpi=300)
    fig.savefig(out / 'figures' / 'FigS2.pdf')
    plt.close(fig)

    params = {
        'state_labels': {
            '1': {'a': 0, 'b': 0},
            '2': {'a': 1, 'b': 0},
            '3': {'a': 1, 'b': 1},
            '4': {'a': 0, 'b': 1},
        },
        'transitions': '1<->2<->3<->4<->1',
        'steady_case': {'k_plus': 'exp(A/2)', 'k_minus': 'exp(-A/2)', 'A': A_steady},
        'periodic_case': {
            'A(t)': 'A0 + A1*sin(2*pi*t/T)',
            'A0': A0, 'A1': A1, 'T': T_period,
            'k_plus(t)': 'exp(A(t)/2)', 'k_minus(t)': 'exp(-A(t)/2)',
        },
        'four_state_mapping': {
            'O_I': 'x*(1-b)', 'O_II': '(1-x)*a', 'O_III': 'x*b', 'O_IV': '(1-x)*(1-a)',
        },
        'three_state_mapping': {
            'A': 'x*(1-b)', 'B': 'x*b + (1-x)*a', 'C': '(1-x)*(1-a)',
        },
        'steady_observable_estimator': 'max_tau sigma_est[0,tau]/tau computed numerically with bounded scalar optimization on tau in [1e-5, 10]',
        'periodic_observable_estimator': 'sigma_est[0,T]/T over one full driving period',
    }
    (out / 'data' / 'four_state_example_parameters.json').write_text(json.dumps(params, indent=2))

if __name__ == '__main__':
    main()
