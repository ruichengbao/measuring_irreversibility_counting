#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch

plt.rcParams.update({
    "font.size": 18,
    "axes.labelsize": 19,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
    "axes.titlesize": 18,
    "axes.linewidth": 1.2,
    "mathtext.fontset": "stix",
    "font.family": "STIXGeneral",
    "savefig.bbox": "tight",
})


def style_axes(ax):
    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(1.2)
    ax.tick_params(direction="out", length=7, width=1.2)
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks(np.linspace(0.0, 1.0, 6))
    ax.grid(False)


def draw_triangle_cycle(ax, pts, node_fc, edge_c, node_ec="#6F6F6F", lw=1.9, mscale=11):
    radius = 0.05
    for p in pts:
        ax.add_patch(Circle(p, radius, facecolor=node_fc, edgecolor=node_ec, lw=1.0, zorder=3))
    order = [0, 1, 2, 0]
    for a, b in zip(order[:-1], order[1:]):
        p1 = np.array(pts[a])
        p2 = np.array(pts[b])
        vec = p2 - p1
        start = p1 + 0.20 * vec
        end = p2 - 0.20 * vec
        ax.add_patch(FancyArrowPatch(
            start, end, arrowstyle='-|>', mutation_scale=mscale,
            lw=lw, color=edge_c, connectionstyle="arc3,rad=0.0", zorder=2
        ))


def draw_copy_schematic(inax, symmetry_broken=False):
    inax.set_axis_off()
    inax.set_xlim(0, 1)
    inax.set_ylim(0, 1)

    # Pastel palette suitable for manuscript figures
    blue_node = "#D7E8F4"
    blue_edge = "#7EA9C6"
    peach_node = "#F7DDD0"
    peach_edge = "#C88F6A"
    bridge_c = "#B8B8B8"

    left_pts = [(0.23, 0.74), (0.11, 0.44), (0.35, 0.44)]
    right_pts = [(0.77, 0.74), (0.65, 0.44), (0.89, 0.44)]

    draw_triangle_cycle(inax, left_pts, blue_node, blue_edge)
    if symmetry_broken:
        draw_triangle_cycle(inax, right_pts, peach_node, peach_edge)
    else:
        draw_triangle_cycle(inax, right_pts, blue_node, blue_edge)

    # three dashed bridge edges between corresponding states
    for p1, p2 in [(left_pts[0], right_pts[0]), (left_pts[1], right_pts[1]), (left_pts[2], right_pts[2])]:
        inax.plot(
            [p1[0] + 0.055, p2[0] - 0.055],
            [p1[1], p2[1]],
            color=bridge_c,
            lw=1.9,
            ls=(0, (3.2, 2.8)),
            zorder=1,
        )


def main():
    base = Path(__file__).resolve().parent
    ring = pd.read_csv(base / "data" / "ring_curve.csv")
    exact = pd.read_csv(base / "data" / "exact_curve.csv")
    pert = pd.read_csv(base / "data" / "pert_curve.csv")

    fig, axes = plt.subplots(1, 3, figsize=(16.2, 4.9), sharex=True)
    panels = [
        (ring, "Nonuniform 4-state ring"),
        (exact, "Copy-symmetric multicycle"),
        (pert, "Symmetry-broken multicycle"),
    ]

    for ax, (df, title), panel_label in zip(axes, panels, ["(a)", "(b)", "(c)"]):
        ax.plot(df["epsilon"], df["ratio"], color="#4C78A8", lw=3.6,
                label=r"$\dot{\sigma}_{\rm cg}/\dot{\sigma}$")
        ax.axhline(1.0, color="#333333", lw=2.7, ls="--", label="exact recovery")
        ax.set_title(title, pad=14)
        ax.set_xlabel(r"error probability $\epsilon$")
        ax.text(-0.13, 1.07, panel_label, transform=ax.transAxes, fontsize=18, va="center")
        style_axes(ax)

    axes[0].set_ylabel(r"recovery ratio $\dot{\sigma}_{\rm cg}/\dot{\sigma}$")
    axes[0].set_ylim(0.45, 1.015)
    axes[1].set_ylim(0.94, 1.05)
    axes[2].set_ylim(0.88, 1.02)

    # Larger in-panel schematics, positioned in the blank lower-left region
    inax_b = axes[1].inset_axes([0.02, 0.01, 0.56, 0.46])
    draw_copy_schematic(inax_b, symmetry_broken=False)

    inax_c = axes[2].inset_axes([0.02, 0.01, 0.56, 0.46])
    draw_copy_schematic(inax_c, symmetry_broken=True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2,
               frameon=True, framealpha=0.95, bbox_to_anchor=(0.5, 1.12))

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.93], w_pad=2.3)

    fig.savefig(base / "exact_recovery_family_improved_v2.png", dpi=300)
    fig.savefig(base / "exact_recovery_family_improved_v2.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
