"""Figures for the scalability and ablation studies."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

AGENT_COLOR = {
    "random": "#888888",
    "greedy": "#1f77b4",
    "layer-aware": "#2ca02c",
    "drl": "#d62728",
    "ilp": "#9467bd",
}
AGENT_LABEL = {
    "random": "Random",
    "greedy": "Greedy",
    "layer-aware": "Layer-Aware",
    "drl": "DRL (ours)",
    "ilp": "ILP (LP baseline)",
}


def figure_scalability() -> None:
    data = np.load(RESULTS_DIR / "scalability.npz")
    sizes = [4, 8, 12, 16, 20]
    agents = ["random", "layer-aware", "drl", "ilp"]
    fig, axs = plt.subplots(1, 2, figsize=(7.5, 3.0))

    for agent in agents:
        med_dec = []
        cost = []
        cost_err = []
        for n in sizes:
            decs = []
            costs = []
            for s in range(3):
                key_d = f"n{n}__{agent}__seed{s}__decision_ms"
                key_c = f"n{n}__{agent}__seed{s}__cost_total"
                decs.extend(data[key_d].tolist())
                costs.append(float(data[key_c].mean()))
            med_dec.append(np.median(decs))
            cost.append(np.mean(costs))
            cost_err.append(np.std(costs))
        axs[0].plot(sizes, med_dec, marker="o", color=AGENT_COLOR[agent], label=AGENT_LABEL[agent])
        axs[1].errorbar(
            sizes, cost, yerr=cost_err,
            marker="o", color=AGENT_COLOR[agent], label=AGENT_LABEL[agent],
            capsize=2.0, linewidth=1.2,
        )
    axs[0].set_xlabel("number of UAVs")
    axs[0].set_ylabel("median decision time (ms)")
    axs[0].set_yscale("log")
    axs[0].set_title("(a) decision-time scaling")
    axs[0].grid(True, which="both", linestyle=":", alpha=0.4)
    axs[0].legend(fontsize=8)

    axs[1].set_xlabel("number of UAVs")
    axs[1].set_ylabel("mean per-step cost")
    axs[1].set_yscale("log")
    axs[1].set_title("(b) cost scaling")
    axs[1].grid(True, which="both", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "scalability.pdf")
    plt.close(fig)


def figure_ablation() -> None:
    data = np.load(RESULTS_DIR / "ablation.npz")

    # Build a small table of mean (cost, served) per (ablation, agent)
    rows = []
    for ablation in ["with", "without"]:
        for agent in ["greedy", "layer-aware", "drl", "ilp_with", "ilp_without"]:
            cs, sv = [], []
            for s in range(3):
                k_c = f"{ablation}__{agent}__seed{s}__cost_total"
                k_s = f"{ablation}__{agent}__seed{s}__pct_demand_served"
                if k_c in data.files:
                    cs.append(data[k_c].mean())
                    sv.append(data[k_s].mean())
            if cs:
                rows.append((ablation, agent, np.mean(cs), np.mean(sv)))

    # Two-panel: cost and served, grouped by agent with side-by-side
    # with/without bars per agent. This avoids the long rotated labels of the
    # previous layout.
    fig, axs = plt.subplots(1, 2, figsize=(7.5, 3.2))

    # Each agent gets a position; ILP shares the slot with the
    # ilp_with / ilp_without pair.
    grouped_agents = ["greedy", "layer-aware", "ilp", "drl"]
    label_for_group = {
        "greedy": "Greedy",
        "layer-aware": "Layer-\nAware",
        "ilp": "ILP",
        "drl": "DRL (ours)",
    }
    color_for_group = {
        "greedy": AGENT_COLOR["greedy"],
        "layer-aware": AGENT_COLOR["layer-aware"],
        "ilp": AGENT_COLOR["ilp"],
        "drl": AGENT_COLOR["drl"],
    }

    bar_w = 0.36
    centres = np.arange(len(grouped_agents))

    def lookup(ablation: str, agent: str):
        if agent == "ilp":
            agent = "ilp_with" if ablation == "with" else "ilp_without"
        return next(
            ((c, s) for (a, ag, c, s) in rows if a == ablation and ag == agent),
            None,
        )

    for k, ablation in enumerate(["with", "without"]):
        offsets = (k - 0.5) * bar_w
        cs = []
        sv = []
        cs_pos = []
        sv_pos = []
        for j, ag in enumerate(grouped_agents):
            row = lookup(ablation, ag)
            if row is None:
                continue
            cs.append(row[0])
            sv.append(row[1] * 100.0)
            cs_pos.append(centres[j] + offsets)
            sv_pos.append(centres[j] + offsets)
        hatch = "" if ablation == "with" else "//"
        edgecol = "black"
        face = [color_for_group[grouped_agents[int(round(p - offsets))]] for p in cs_pos]
        axs[0].bar(
            cs_pos, cs, width=bar_w,
            color=face, edgecolor=edgecol, linewidth=0.5, hatch=hatch,
            label=(r"$\omega = 25$ (refined)" if ablation == "with" else r"$\omega = 0$ (original)"),
        )
        axs[1].bar(
            sv_pos, sv, width=bar_w,
            color=face, edgecolor=edgecol, linewidth=0.5, hatch=hatch,
            label=(r"$\omega = 25$ (refined)" if ablation == "with" else r"$\omega = 0$ (original)"),
        )

    for ax in axs:
        ax.set_xticks(centres)
        ax.set_xticklabels([label_for_group[g] for g in grouped_agents], fontsize=8)
        ax.grid(True, axis="y", linestyle=":", alpha=0.4)

    axs[0].set_yscale("log")
    axs[0].set_ylabel("mean cost / step (log)")
    axs[0].set_title("(a) cost")
    axs[0].legend(fontsize=7, loc="upper right")

    axs[1].set_ylabel("served (%)")
    axs[1].set_ylim(0, 110)
    axs[1].set_title("(b) coverage")
    axs[1].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ablation.pdf")
    plt.close(fig)


def main() -> None:
    figure_scalability()
    figure_ablation()
    print(f"Wrote scalability/ablation figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
