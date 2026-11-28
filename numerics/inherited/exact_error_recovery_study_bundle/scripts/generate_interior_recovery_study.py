#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "axes.linewidth": 1.1,
    "savefig.bbox": "tight",
    "mathtext.fontset": "stix",
    "font.family": "STIXGeneral",
})

def stationary_distribution(K: np.ndarray) -> np.ndarray:
    w, v = np.linalg.eig(K)
    idx = np.argmin(np.abs(w))
    p = np.real(v[:, idx])
    if np.all(p <= 0):
        p = -p
    p = p / p.sum()
    return p

def entropy_production_rate(K: np.ndarray, p: np.ndarray | None = None) -> float:
    if p is None:
        p = stationary_distribution(K)
    n = K.shape[0]
    val = 0.0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if K[i, j] > 0 and K[j, i] > 0:
                val += K[i, j] * p[j] * np.log((K[i, j] * p[j]) / (K[j, i] * p[i]))
    return float(val)

def mapping_from_permutation(eps: float, perm: list[int]) -> np.ndarray:
    n = len(perm)
    T = np.zeros((n, n), dtype=float)
    for i in range(n):
        T[i, i] += 1.0 - eps
        T[perm[i], i] += eps
    return T

def coarse_entropy_production_rate(K: np.ndarray, T: np.ndarray, p: np.ndarray | None = None) -> float:
    if p is None:
        p = stationary_distribution(K)
    n = K.shape[0]
    m = T.shape[0]
    F = np.zeros((m, m), dtype=float)
    for i in range(n):
        for j in range(n):
            if i != j and K[i, j] > 0:
                F += np.outer(T[:, i], T[:, j]) * K[i, j] * p[j]
    val = 0.0
    for a in range(m):
        for b in range(m):
            if a != b and F[a, b] > 0 and F[b, a] > 0:
                val += F[a, b] * np.log(F[a, b] / F[b, a])
    return float(val)

def add_edge(K: np.ndarray, source: int, target: int, forward: float, backward: float) -> None:
    K[target, source] += forward
    K[source, target] += backward

def finalize_generator(K: np.ndarray) -> np.ndarray:
    for j in range(K.shape[1]):
        K[j, j] = -K[:, j].sum()
    return K

def build_nonuniform_ring() -> tuple[np.ndarray, list[int], dict]:
    K = np.zeros((4, 4), dtype=float)
    cw = [2.0, 1.2, 1.5, 1.8]
    ccw = [0.5, 0.8, 0.7, 0.9]
    for j in range(4):
        i = (j + 1) % 4
        K[i, j] += cw[j]
        K[j, i] += ccw[j]
    finalize_generator(K)
    perm = [3, 0, 1, 2]  # cyclic shift to the previous neighbour
    meta = {
        "model": "nonuniform_4_state_ring",
        "clockwise_rates": cw,
        "counterclockwise_rates": ccw,
        "permutation": perm,
        "description": "state i is reported as i with probability 1-eps and as i-1 with probability eps",
    }
    return K, perm, meta

def build_exact_ladder() -> tuple[np.ndarray, list[int], dict]:
    K = np.zeros((6, 6), dtype=float)
    top = [(2.0, 0.5), (1.5, 0.7), (1.2, 0.6)]
    bottom = top[:]  # exact copy
    rungs = [(0.9, 0.9), (0.9, 0.9), (0.9, 0.9)]
    top_edges = [(0, 1), (1, 2), (2, 0)]
    bot_edges = [(3, 4), (4, 5), (5, 3)]
    for (s, t), (fwd, bwd) in zip(top_edges, top):
        add_edge(K, s, t, fwd, bwd)
    for (s, t), (fwd, bwd) in zip(bot_edges, bottom):
        add_edge(K, s, t, fwd, bwd)
    for (s, t), (fwd, bwd) in zip([(0, 3), (1, 4), (2, 5)], rungs):
        add_edge(K, s, t, fwd, bwd)
    finalize_generator(K)
    perm = [3, 4, 5, 0, 1, 2]  # pairwise swap across the rungs
    meta = {
        "model": "nonuniform_6_state_ladder_exact",
        "top_triangle_rates_(clockwise, counterclockwise)": top,
        "bottom_triangle_rates_(clockwise, counterclockwise)": bottom,
        "rung_rates_(forward, backward)": rungs,
        "permutation": perm,
        "description": "each rung pair is confused with probability eps",
    }
    return K, perm, meta

def build_perturbed_ladder() -> tuple[np.ndarray, list[int], dict]:
    K = np.zeros((6, 6), dtype=float)
    top = [(2.0, 0.5), (1.5, 0.7), (1.2, 0.6)]
    bottom = [(1.8, 0.55), (1.3, 0.75), (1.0, 0.65)]  # symmetry broken
    rungs = [(0.9, 0.9), (0.9, 0.9), (0.9, 0.9)]
    top_edges = [(0, 1), (1, 2), (2, 0)]
    bot_edges = [(3, 4), (4, 5), (5, 3)]
    for (s, t), (fwd, bwd) in zip(top_edges, top):
        add_edge(K, s, t, fwd, bwd)
    for (s, t), (fwd, bwd) in zip(bot_edges, bottom):
        add_edge(K, s, t, fwd, bwd)
    for (s, t), (fwd, bwd) in zip([(0, 3), (1, 4), (2, 5)], rungs):
        add_edge(K, s, t, fwd, bwd)
    finalize_generator(K)
    perm = [3, 4, 5, 0, 1, 2]
    meta = {
        "model": "nonuniform_6_state_ladder_perturbed",
        "top_triangle_rates_(clockwise, counterclockwise)": top,
        "bottom_triangle_rates_(clockwise, counterclockwise)": bottom,
        "rung_rates_(forward, backward)": rungs,
        "permutation": perm,
        "description": "same local confusion as the exact ladder, but copy symmetry is broken",
    }
    return K, perm, meta

def compute_curve(K: np.ndarray, perm: list[int], eps_grid: np.ndarray) -> tuple[pd.DataFrame, dict]:
    p = stationary_distribution(K)
    sigma_true = entropy_production_rate(K, p)
    rows = []
    for eps in eps_grid:
        T = mapping_from_permutation(float(eps), perm)
        sigma_cg = coarse_entropy_production_rate(K, T, p)
        rows.append({
            "epsilon": float(eps),
            "sigma_true": float(sigma_true),
            "sigma_cg": float(sigma_cg),
            "ratio": float(sigma_cg / sigma_true if sigma_true > 0 else np.nan),
        })
    df = pd.DataFrame(rows)
    info = {
        "stationary_distribution": p.tolist(),
        "sigma_true": float(sigma_true),
        "max_ratio_strict_interior": float(df[(df["epsilon"] > 0) & (df["epsilon"] < 1)]["ratio"].max()),
        "argmax_ratio_strict_interior": float(df[(df["epsilon"] > 0) & (df["epsilon"] < 1)].iloc[df[(df["epsilon"] > 0) & (df["epsilon"] < 1)]["ratio"].argmax()]["epsilon"]),
    }
    return df, info

def style_axes(ax):
    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(1.1)
    ax.tick_params(direction="out", length=4.5, width=1.0)
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks(np.linspace(0.0, 1.0, 6))
    ax.grid(False)

def make_figure(dfs: dict[str, pd.DataFrame], out_fig: Path) -> None:
    colors = {
        "cg": "#4C78A8",
        "true": "#333333",
    }
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.8), sharex=True)
    panels = [
        ("ring", "Nonuniform 4-state ring"),
        ("exact", "Copy-symmetric multicycle"),
        ("pert", "Symmetry-broken multicycle"),
    ]
    for ax, (key, title), label in zip(axes, panels, ["(a)", "(b)", "(c)"]):
        df = dfs[key]
        ax.plot(df["epsilon"], df["ratio"], color=colors["cg"], lw=2.6, label=r"$\dot{\sigma}_{\rm cg}/\dot{\sigma}$")
        ax.axhline(1.0, color=colors["true"], lw=1.8, ls="--", label="exact recovery")
        ax.set_title(title, pad=8)
        ax.set_xlabel(r"error probability $\epsilon$")
        ax.text(-0.13, 1.03, label, transform=ax.transAxes, fontsize=15)
        style_axes(ax)
    axes[0].set_ylabel(r"recovery ratio $\dot{\sigma}_{\rm cg}/\dot{\sigma}$")
    axes[0].set_ylim(0.45, 1.02)
    axes[1].set_ylim(0.95, 1.05)
    axes[2].set_ylim(0.88, 1.02)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=True, framealpha=0.95, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout(rect=[0, 0, 1, 0.94], w_pad=2.0)
    fig.savefig(out_fig.with_suffix(".png"), dpi=300)
    fig.savefig(out_fig.with_suffix(".pdf"))
    plt.close(fig)

def main(outdir: str = "interior_recovery_outputs") -> None:
    out = Path(outdir)
    (out / "data").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    eps_grid = np.linspace(0.0, 1.0, 501)

    models = {
        "ring": build_nonuniform_ring(),
        "exact": build_exact_ladder(),
        "pert": build_perturbed_ladder(),
    }

    summaries = {}
    dfs = {}
    for key, (K, perm, meta) in models.items():
        df, info = compute_curve(K, perm, eps_grid)
        df.to_csv(out / "data" / f"{key}_curve.csv", index=False)
        summaries[key] = {**meta, **info}
        dfs[key] = df

    pd.DataFrame([
        {"model": k, "sigma_true": summaries[k]["sigma_true"], "max_ratio_strict_interior": summaries[k]["max_ratio_strict_interior"], "argmax_ratio_strict_interior": summaries[k]["argmax_ratio_strict_interior"]}
        for k in ["ring", "exact", "pert"]
    ]).to_csv(out / "data" / "summary_table.csv", index=False)

    make_figure(dfs, out / "figures" / "interior_recovery_examples")

    (out / "data" / "model_metadata.json").write_text(pd.Series(summaries).to_json(indent=2))

if __name__ == "__main__":
    main()
