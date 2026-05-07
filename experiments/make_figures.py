"""Generate the figures used in the experimental paper.

Reads:
    results/training_history.npz
    results/evaluation.npz
    results/evaluation_summary.json

Writes:
    figures/training_curve.pdf
    figures/cost_by_scenario.pdf
    figures/served_by_scenario.pdf
    figures/latency_vs_decision_time.pdf
    figures/per_step_cost.pdf
    figures/layer_usage.pdf
    figures/migration_count.pdf
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import dataclasses

from agents.baselines import GreedyLatencyPlacement, LayerAwareHeuristic, RandomPlacement
from agents.drl import DRLPlacementAgent
from agents.ilp import ILPPlacement
from experiments.scenarios import SCENARIOS
from simulator import NTNEnvironment

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)


AGENT_ORDER = ["random", "greedy", "layer-aware", "drl", "ilp"]
AGENT_LABELS = {
    "random": "Random",
    "greedy": "Greedy",
    "layer-aware": "Layer-Aware",
    "drl": "DRL (ours)",
    "ilp": "ILP (LP baseline)",
}
AGENT_COLOR = {
    "random": "#888888",
    "greedy": "#1f77b4",
    "layer-aware": "#2ca02c",
    "drl": "#d62728",
    "ilp": "#9467bd",
}


def figure_training_curve() -> None:
    data = np.load(RESULTS_DIR / "training_history.npz")
    eps = data["episodes"]
    rewards = data["rewards"]
    pl = data["policy_loss"]
    vl = data["value_loss"]
    fig, axs = plt.subplots(1, 2, figsize=(7.5, 2.7))
    axs[0].plot(eps, rewards, color="#d62728", linewidth=1.0)
    # smoothed
    if len(rewards) > 5:
        kernel = np.ones(5) / 5
        smoothed = np.convolve(rewards, kernel, mode="valid")
        axs[0].plot(eps[: len(smoothed)] + 2, smoothed, color="black", linewidth=1.4, label="moving avg (5)")
        axs[0].legend(loc="lower right", fontsize=8)
    axs[0].set_xlabel("episodes")
    axs[0].set_ylabel("episode reward")
    axs[0].set_title("(a) PPO reward curve")
    axs[0].grid(True, linestyle=":", alpha=0.5)

    axs[1].plot(eps, vl, color="#1f77b4", linewidth=1.0, label="value loss")
    axs[1].set_xlabel("episodes")
    axs[1].set_ylabel("value loss")
    axs[1].set_yscale("log")
    axs[1].set_title("(b) value-loss decay")
    axs[1].grid(True, which="both", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "training_curve.pdf")
    plt.close(fig)


def figure_cost_and_served() -> None:
    with open(RESULTS_DIR / "evaluation_summary.json") as f:
        summary = json.load(f)
    scenarios = list(SCENARIOS.keys())
    n_agents = len(AGENT_ORDER)
    width = 0.16

    fig, ax = plt.subplots(figsize=(7.5, 3.0))
    x = np.arange(len(scenarios))
    for k, agent in enumerate(AGENT_ORDER):
        means = [summary[s][agent]["mean_cost_total"] for s in scenarios]
        stds = [summary[s][agent]["std_cost_total"] for s in scenarios]
        ax.bar(
            x + (k - n_agents / 2) * width + width / 2,
            means,
            width=width,
            label=AGENT_LABELS[agent],
            color=AGENT_COLOR[agent],
            yerr=stds,
            capsize=2.0,
            error_kw={"linewidth": 0.7, "alpha": 0.6},
        )
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.set_ylabel("mean total cost (per step, log)")
    ax.set_title("Total objective cost across scenarios")
    ax.legend(loc="upper left", fontsize=8, ncol=3)
    ax.grid(True, axis="y", which="both", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "cost_by_scenario.pdf")
    plt.close(fig)

    # Served fraction
    fig, ax = plt.subplots(figsize=(7.5, 3.0))
    for k, agent in enumerate(AGENT_ORDER):
        means = [summary[s][agent]["mean_pct_demand_served"] * 100 for s in scenarios]
        ax.bar(
            x + (k - n_agents / 2) * width + width / 2,
            means,
            width=width,
            label=AGENT_LABELS[agent],
            color=AGENT_COLOR[agent],
        )
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.set_ylabel("served demand (%)")
    ax.set_ylim(0, 110)
    ax.set_title("QoS coverage of (user, service) demand")
    ax.legend(loc="lower right", fontsize=8, ncol=3)
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "served_by_scenario.pdf")
    plt.close(fig)


def figure_latency_vs_decision_time() -> None:
    with open(RESULTS_DIR / "evaluation_summary.json") as f:
        summary = json.load(f)
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    for agent in AGENT_ORDER:
        xs = [summary[s][agent]["median_decision_ms"] for s in SCENARIOS]
        ys = [summary[s][agent]["mean_avg_latency_ms"] for s in SCENARIOS]
        ax.scatter(
            xs, ys,
            s=60,
            color=AGENT_COLOR[agent],
            label=AGENT_LABELS[agent],
            edgecolor="black", linewidth=0.5,
        )
    ax.set_xlabel("median decision time per step (ms)")
    ax.set_ylabel("mean access latency (ms)")
    ax.set_xscale("log")
    ax.set_title("Speed–quality trade-off")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "latency_vs_decision_time.pdf")
    plt.close(fig)


def figure_per_step_cost() -> None:
    data = np.load(RESULTS_DIR / "evaluation.npz")
    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    scen = "small"
    for agent in AGENT_ORDER:
        key = f"{scen}__{agent}__cost_total"
        arr = data[key]  # (S, T)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        t = np.arange(len(mean))
        ax.plot(t, mean, color=AGENT_COLOR[agent], label=AGENT_LABELS[agent], linewidth=1.4)
        ax.fill_between(t, mean - std, mean + std, color=AGENT_COLOR[agent], alpha=0.15)
    ax.set_yscale("log")
    ax.set_xlabel("step (each step = 60 s)")
    ax.set_ylabel("step cost (log)")
    ax.set_title(f"Per-step cost trace ({scen} scenario, mean ± std over 5 seeds)")
    ax.legend(loc="upper right", fontsize=8, ncol=3)
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "per_step_cost.pdf")
    plt.close(fig)


def figure_layer_usage() -> None:
    """Run a single deterministic episode per agent and capture the per-layer
    placement counts. This is purely an instrumentation helper; not part of
    the main quantitative comparison."""
    drl = DRLPlacementAgent(
        node_feature_dim=NTNEnvironment.NODE_FEATURE_DIM,
        svc_feature_dim=NTNEnvironment.SERVICE_FEATURE_DIM,
        embed_dim=64, n_heads=4, seed=0,
    )
    drl.load(str(RESULTS_DIR / "drl_policy.pt"))
    agents = {
        "random": RandomPlacement(seed=7),
        "greedy": GreedyLatencyPlacement(),
        "layer-aware": LayerAwareHeuristic(),
        "ilp": ILPPlacement(time_limit_s=6.0),
        "drl": drl,
    }
    cfg = SCENARIOS["small"]
    layer_counts = {a: {"LEO": 0, "HAPS": 0, "UAV": 0} for a in agents}
    for agent_name, agent in agents.items():
        env = NTNEnvironment(dataclasses.replace(cfg))
        obs = env.reset()
        for _ in range(cfg.horizon):
            X = agent.place(env, obs)
            for j in range(X.shape[1]):
                for i in range(X.shape[0]):
                    if X[i, j] == 1:
                        layer_counts[agent_name][env.nodes[i].layer.value] += 1
            env.apply_placement(X)
            env.step()
            obs = env.observe()

    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    layer_names = ["LEO", "HAPS", "UAV"]
    layer_colors = {"LEO": "#9467bd", "HAPS": "#2ca02c", "UAV": "#ff7f0e"}
    bottom = np.zeros(len(AGENT_ORDER))
    for layer in layer_names:
        vals = np.array([layer_counts[a][layer] for a in AGENT_ORDER])
        total = sum(layer_counts[AGENT_ORDER[0]].values())  # rough denominator
        # Normalise each agent's bar to its own total (fraction of placements)
        totals = np.array(
            [sum(layer_counts[a].values()) for a in AGENT_ORDER]
        ).astype(float)
        totals = np.where(totals == 0, 1.0, totals)
        fracs = vals / totals
        ax.bar(
            range(len(AGENT_ORDER)),
            fracs,
            bottom=bottom,
            color=layer_colors[layer],
            label=layer,
            edgecolor="white",
        )
        bottom = bottom + fracs
    ax.set_xticks(range(len(AGENT_ORDER)))
    ax.set_xticklabels([AGENT_LABELS[a] for a in AGENT_ORDER], rotation=15)
    ax.set_ylabel("placement share")
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-layer share of microservice placements (small scenario)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "layer_usage.pdf")
    plt.close(fig)


def figure_migration_count() -> None:
    data = np.load(RESULTS_DIR / "evaluation.npz")
    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    scenarios = list(SCENARIOS.keys())
    width = 0.16
    x = np.arange(len(scenarios))
    n_agents = len(AGENT_ORDER)
    for k, agent in enumerate(AGENT_ORDER):
        means = []
        for scen in scenarios:
            key = f"{scen}__{agent}__cost_migration"
            arr = data[key]
            means.append(arr.mean())
        ax.bar(
            x + (k - n_agents / 2) * width + width / 2,
            means,
            width=width,
            label=AGENT_LABELS[agent],
            color=AGENT_COLOR[agent],
        )
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.set_ylabel("mean migration cost / step")
    ax.set_title("Migration overhead")
    ax.legend(loc="upper right", fontsize=8, ncol=3)
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "migration_count.pdf")
    plt.close(fig)


def main() -> None:
    figure_training_curve()
    figure_cost_and_served()
    figure_latency_vs_decision_time()
    figure_per_step_cost()
    figure_layer_usage()
    figure_migration_count()
    print(f"Wrote figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
