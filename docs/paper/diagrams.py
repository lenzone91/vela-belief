"""Render original conceptual diagrams as SVG for HTML and PDF for print.

These illustrations contain no experimental data. Result plots remain in results.py.
"""

from pathlib import Path

BLUE = "#e4eff7"
GREEN = "#e6f0e8"
AMBER = "#fbefda"
INK = "#233b4b"
GRAY = "#f0f2f4"


def _canvas(plt, height):
    fig, ax = plt.subplots(figsize=(7.2, height))
    ax.set(xlim=(0, 10), ylim=(0, height))
    ax.axis("off")
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.98)
    return fig, ax


def _box(ax, x, y, width, height, label, color=BLUE, size=10):
    from matplotlib.patches import FancyBboxPatch

    ax.add_patch(
        FancyBboxPatch(
            (x - width / 2, y - height / 2),
            width,
            height,
            boxstyle="round,pad=0.035,rounding_size=0.08",
            facecolor=color,
            edgecolor=INK,
            linewidth=1,
        )
    )
    ax.text(x, y, label, ha="center", va="center", fontsize=size, color=INK, linespacing=1.4)


def _arrow(ax, start, end, *, dashed=False):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": "-|>",
            "color": INK,
            "linewidth": 1.1,
            "linestyle": "--" if dashed else "-",
            "shrinkA": 3,
            "shrinkB": 3,
        },
    )


def _note(ax, x, y, label, size=9):
    ax.text(x, y, label, ha="center", va="center", fontsize=size, color=INK)


def _save(plt, fig, destination, name):
    for extension in ("svg", "pdf"):
        fig.savefig(destination / f"{name}.{extension}")
    plt.close(fig)


def generate(source: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination = source / "_generated"
    destination.mkdir(exist_ok=True)
    with plt.rc_context({"font.family": "DejaVu Sans", "svg.fonttype": "none", "pdf.fonttype": 42}):
        fig, ax = _canvas(plt, 4.3)
        _box(ax, 2.1, 3.8, 2.6, 0.55, "New observation / block", GRAY, 9)
        _box(ax, 6.0, 3.8, 2.6, 0.55, "Encode new evidence", BLUE, 9)
        _arrow(ax, (3.4, 3.8), (4.7, 3.8))
        _arrow(ax, (6.0, 3.5), (6.0, 3.12))
        _box(ax, 5.0, 2.8, 8.8, 0.55, "ROUTE  →  UPDATE / ALLOCATE", BLUE, 11)
        _arrow(ax, (5, 2.5), (5, 2.16))
        _box(ax, 5.0, 1.85, 8.8, 0.55, "CONSOLIDATE  →  FORGET  (when scheduled)", AMBER, 10)
        ax.plot([5, 5, 2.2], [1.55, 1.27, 1.27], color=INK, linewidth=1.1)
        _arrow(ax, (2.2, 1.27), (2.2, 1.08))
        _box(ax, 2.2, 0.8, 3.0, 0.55, "Bounded state: K × dₘ", GREEN, 10)
        _box(ax, 7.25, 0.8, 3.7, 0.55, "Read → predict / infer", GREEN, 10)
        _arrow(ax, (3.7, 0.8), (5.4, 0.8))
        ax.plot([0.45, 0.45], [0.8, 2.8], color=INK, linestyle="--", linewidth=1)
        _arrow(ax, (0.7, 0.8), (0.45, 0.8), dashed=True)
        _arrow(ax, (0.45, 2.8), (0.6, 2.8), dashed=True)
        _note(
            ax, 5, 0.2, "Persistent state is carried to the next update; capacity remains fixed.", 9
        )
        _save(plt, fig, destination, "memory-cycle")

        fig, ax = _canvas(plt, 4.9)
        _box(ax, 2.4, 4.25, 3.7, 0.7, "Slot i\nred car / Alice", BLUE)
        _box(ax, 7.6, 4.25, 3.7, 0.7, "Slot j\nelectric car / Alice", BLUE)
        _arrow(ax, (2.4, 3.88), (4.2, 3.24))
        _arrow(ax, (7.6, 3.88), (5.8, 3.24))
        _box(ax, 5, 2.9, 6.5, 0.65, "Identity evidence + candidate merge", AMBER)
        _arrow(ax, (5, 2.55), (5, 2.15))
        _box(ax, 5, 1.8, 8.2, 0.65, "Compare future-query predictions before / after", GRAY, 10)
        _arrow(ax, (3.4, 1.45), (2.5, 1.05))
        _arrow(ax, (6.6, 1.45), (7.5, 1.05))
        _box(ax, 2.5, 0.65, 4, 0.7, "Accept: integrate attributes\nrelease one row", GREEN, 9)
        _box(ax, 7.5, 0.65, 4, 0.7, "Reject: keep both rows\nretain distinct evidence", GRAY, 9)
        _save(plt, fig, destination, "consolidation")

        fig, ax = _canvas(plt, 5.2)
        _note(
            ax, 5, 4.9, "ACTIVE CONTEXT: memory positions + recent text + generation reserve ≤ C", 9
        )
        _box(ax, 5, 4.2, 8.8, 0.65, "Before overflow: ordinary text context", GRAY)
        _arrow(ax, (2.2, 3.85), (2.2, 3.35))
        _arrow(ax, (7.2, 3.85), (7.2, 3.35))
        _box(ax, 2.2, 3, 3.3, 0.65, "Evicted block 1", AMBER)
        _box(ax, 7.2, 3, 4.3, 0.65, "Recent verbatim suffix", BLUE)
        _arrow(ax, (2.2, 2.65), (2.2, 2.1))
        _arrow(ax, (7.2, 2.65), (7.2, 2.1))
        _box(ax, 2.2, 1.75, 3.3, 0.65, "Update M₀ → M₁", GREEN)
        _box(ax, 7.2, 1.75, 4.3, 0.65, "Read [P(M₁), recent text]", BLUE)
        _arrow(ax, (3.88, 1.75), (5.02, 1.75))
        _arrow(ax, (2.2, 1.4), (2.2, 0.9))
        _box(
            ax,
            5,
            0.55,
            8.8,
            0.65,
            "Next overflow: update M₁ with block 2 → M₂; retain new suffix",
            GREEN,
            9,
        )
        _save(plt, fig, destination, "overflow")

        fig, ax = _canvas(plt, 7.5)
        labels = [
            "1  Single-vector belief baseline — implemented",
            "2  Structured slots — capacity and readout",
            "3  Sparse routing — selective writes",
            "4  Allocation and forgetting — changing relevance",
            "5  Consolidation — merge and regain capacity",
            "6  Representation pressure — ablations and analysis",
        ]
        for index, label in enumerate(labels):
            y = 7.05 - index * 0.8
            _box(ax, 5, y, 8.5, 0.53, label, GREEN if index == 0 else BLUE, 10)
            if index < 5:
                _arrow(ax, (5, y - 0.29), (5, y - 0.51))
        _arrow(ax, (3.1, 2.76), (2.5, 2.13))
        _arrow(ax, (6.9, 2.76), (7.5, 2.13))
        _box(ax, 2.5, 1.75, 4.2, 0.7, "7  World-model state\nfilter → rollout → plan", AMBER, 10)
        _box(
            ax,
            7.5,
            1.75,
            4.2,
            0.7,
            "8  LLM overflow / VELA-S\nfrozen reader + exact suffix",
            AMBER,
            10,
        )
        _arrow(ax, (7.5, 1.36), (7.5, 0.94))
        _box(ax, 7.5, 0.55, 4.2, 0.7, "9  VELA-KV\nlayer-aware memory reader", AMBER, 10)
        _note(
            ax,
            2.5,
            0.55,
            "Every stage: controlled comparisons,\narchived metrics, limits and failures",
            9,
        )
        _save(plt, fig, destination, "roadmap")
