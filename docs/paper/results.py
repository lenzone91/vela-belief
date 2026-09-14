"""Build article tables and a vector figure from the checked-in result snapshot.

No training, checkpoint access, PyTorch import, or network call is needed.
"""

import hashlib
import json
from pathlib import Path


def _table(title: str, headers: list[str], rows: list[list[str]]) -> str:
    lines = [f".. list-table:: {title}", "   :header-rows: 1", ""]
    for row in [headers, *rows]:
        lines.append(f"   * - {row[0]}")
        lines.extend(f"     - {cell}" for cell in row[1:])
    return "\n".join(lines) + "\n"


def generate(source: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    snapshot = source.parent / "stage1-results.json"
    raw = snapshot.read_bytes()
    records = json.loads(raw)
    if [r["seed"] for r in records] != [42, 43, 44]:
        raise ValueError("Update the article protocol when changing the stage 1 run set")
    generated = source / "_generated"
    generated.mkdir(exist_ok=True)

    rows = []
    for record in records:
        learned = record["test"]["learned"]
        rows.append(
            [
                str(record["seed"]),
                str(record["best_step"]),
                f"{learned['belief_kl']:.8f}",
                f"{100 * learned['state_accuracy']:.2f}%",
                f"{learned['next_observation_nll']:.6f}",
                f"{record['long_test_tail']['learned']['belief_kl']:.8f}",
            ]
        )
    text = _table(
        "Held-out GRU results. Tail KL scores positions 64--255 of length-256 sequences.",
        ["Seed", "Update", "Belief KL", "Accuracy", "Next NLL", "Tail KL"],
        rows,
    )
    labels = {
        "learned": "Learned GRU",
        "exact_filter": "Exact filter",
        "observation_only": "Observation only",
    }
    rows = []
    for key, label in labels.items():
        scores = records[0]["test"][key]
        rows.append(
            [
                label,
                f"{scores['belief_kl']:.8f}",
                f"{scores['state_accuracy'] * 100:.2f}%",
                f"{scores['next_observation_nll']:.6f}",
            ]
        )
    text += "\n" + _table(
        "Same-sequence comparison for seed 42.",
        ["Model", "Belief KL", "Accuracy", "Next NLL"],
        rows,
    )
    (generated / "tables.rst").write_text(text, encoding="utf-8")
    (generated / "provenance.rst").write_text(
        "The tables and figure are generated from ``docs/stage1-results.json``. "
        "The exact input snapshot has SHA-256:\n\n"
        f".. code-block:: text\n\n   {hashlib.sha256(raw).hexdigest()}\n",
        encoding="utf-8",
    )

    with plt.rc_context({"font.size": 9, "svg.fonttype": "none", "pdf.fonttype": 42}):
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), layout="constrained")
        seeds = [r["seed"] for r in records]
        for split, label, style in [
            ("test", "GRU, length 64", "o-"),
            ("long_test_tail", "GRU, long-sequence tail", "s--"),
        ]:
            axes[0].plot(
                seeds, [r[split]["learned"]["belief_kl"] for r in records], style, label=label
            )
        axes[0].plot(
            seeds,
            [r["test"]["observation_only"]["belief_kl"] for r in records],
            "^:",
            color="0.45",
            label="Observation only, length 64",
        )
        axes[0].set(yscale="log", ylabel="Belief KL (nats; log scale)", xlabel="Experiment seed")
        for (key, label), marker in zip(labels.items(), ("o", "x", "^"), strict=True):
            axes[1].plot(
                seeds,
                [100 * r["test"][key]["state_accuracy"] for r in records],
                marker=marker,
                label=label,
            )
        axes[1].set(ylabel="Hidden-state accuracy (%)", xlabel="Experiment seed", ylim=(65, 86))
        for axis in axes:
            axis.set_xticks(seeds)
            axis.grid(alpha=0.2)
            axis.legend(fontsize=7, loc="best")
        for extension in ("svg", "pdf"):
            fig.savefig(generated / f"stage1.{extension}")
        plt.close(fig)
