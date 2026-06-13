"""Draw the candidate Chicago DAG for Section 4.3 (Stage 2 output).

Instantiates the Definition 2.1 template (Figure 2.1) with the Chicago
variables: it is isomorphic to the seven-node schematic (same roles, same
edges), only relabelled. Mirrors Figure 2.1's style -- navy-edged boxes, a
hexagon for the compliance gate, a shaded plate around the latent world layer
(M, Y), and curved arrows.

Run (from drafts/complete/): uv run python scripts/data/candidate_dag.py
Output: results/data/fig_08_candidate_dag.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, RegularPolygon, FancyBboxPatch as _Box

FIGURES_DIR = Path(__file__).resolve().parents[2] / "results" / "data"

NAVY = "#2a2a4a"
PLATE = "#eef0f6"
ARROW = "#1b1b2e"


def add_box(ax, x, y, w, h, sym, desc, hexagon=False):
    """Add a node centred at (x, y). Returns the patch (for arrow clipping)."""
    if hexagon:
        patch = RegularPolygon(
            (x, y), numVertices=6, radius=max(w, h) * 0.62,
            orientation=0, facecolor="white", edgecolor=NAVY, linewidth=1.6, zorder=3,
        )
    else:
        patch = FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            facecolor="white", edgecolor=NAVY, linewidth=1.6, zorder=3,
        )
    ax.add_patch(patch)
    ax.text(x, y + h * 0.20, sym, ha="center", va="center",
            fontsize=15, zorder=4)
    if desc:
        ax.text(x, y - h * 0.22, desc, ha="center", va="center",
                fontsize=7.4, color="#333333", zorder=4, linespacing=1.15)
    patch._node_centre = (x, y)
    return patch


def arrow(ax, a, b, rad=0.0):
    ax.add_patch(FancyArrowPatch(
        posA=a._node_centre, posB=b._node_centre, patchA=a, patchB=b,
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle="-|>", mutation_scale=15, linewidth=1.5,
        color=ARROW, shrinkA=2, shrinkB=2, zorder=2,
    ))


def main() -> None:
    fig, ax = plt.subplots(figsize=(8.4, 9.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Latent world-layer plate behind M and Y.
    ax.add_patch(_Box(
        (0.45, 1.0), 4.15, 4.05,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        facecolor=PLATE, edgecolor="#c9cde0", linewidth=1.0, zorder=1,
    ))
    ax.text(0.7, 4.78, "world layer (latent)", ha="left", va="center",
            fontsize=7.5, style="italic", color="#7a7f9a", zorder=2)

    # Nodes (centre coordinates).
    xs = add_box(ax, 3.4, 9.2, 2.5, 1.0, r"$\mathbf{X}_s$",
                 "Gross floor area\nPrimary property type")
    xo = add_box(ax, 2.3, 6.4, 2.2, 1.05, r"$X_o$",
                 "Year built\nNo. of buildings\nData year")
    c = add_box(ax, 6.6, 6.4, 1.9, 1.4, r"$C$", "Compliance\ngate", hexagon=True)
    m = add_box(ax, 2.7, 3.7, 3.0, 1.25, r"$M$",
                "Electricity, natural gas,\nother fuel use\n(district steam / chilled:\nstructural zero)")
    y = add_box(ax, 1.95, 1.85, 2.1, 1.0, r"$Y$", "Total GHG\nemissions")
    mobs = add_box(ax, 6.6, 2.3, 2.3, 1.0, r"$M^{\mathrm{obs}}$",
                   "observed\nenergy use")
    yobs = add_box(ax, 4.7, 0.55, 2.3, 0.95, r"$Y^{\mathrm{obs}}$",
                   "observed GHG")

    # Edges (the Definition 2.1 template).
    arrow(ax, xs, c, rad=-0.18)      # Xs -> C
    arrow(ax, xs, m, rad=0.30)       # Xs -> M
    arrow(ax, xo, m, rad=0.0)        # Xo -> M
    arrow(ax, m, y, rad=0.10)        # M -> Y
    arrow(ax, m, mobs, rad=-0.12)    # M -> M_obs
    arrow(ax, c, mobs, rad=0.0)      # C -> M_obs
    arrow(ax, y, yobs, rad=-0.15)    # Y -> Y_obs
    arrow(ax, c, yobs, rad=-0.45)    # C -> Y_obs

    fig.tight_layout(pad=0.3)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "fig_08_candidate_dag.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
