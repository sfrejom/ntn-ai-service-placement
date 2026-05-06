"""Scalability and ablation studies.

Runs two extra experiments:

1. Scalability: vary the UAV population in [4, 8, 12, 16, 20] (with the
   service count scaled proportionally) and record DRL and ILP decision
   times and cost. Random and layer-aware are included as cheap reference
   curves.

2. Ablation: disable the unserved-demand term (w_unserved = 0) for the
   greedy/layer-aware/ILP/DRL policies and re-evaluate the small scenario.
   This isolates how much of the DRL agent's cost reduction is due to the
   refined objective.

Outputs:
    results/scalability.npz
    results/ablation.npz
"""

from __future__ import annotations

import dataclasses
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

from agents.baselines import GreedyLatencyPlacement, LayerAwareHeuristic, RandomPlacement
from agents.drl import DRLPlacementAgent
from agents.ilp import ILPPlacement
from experiments.scenarios import SCENARIOS
from simulator import EpisodeConfig, NTNEnvironment

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"


def _eval_episode(env_cfg: EpisodeConfig, agent) -> dict:
    env = NTNEnvironment(env_cfg)
    obs = env.reset()
    decision_times = []
    costs = []
    served = []
    lat = []
    for _ in range(env_cfg.horizon):
        t0 = time.perf_counter()
        X = agent.place(env, obs)
        decision_times.append((time.perf_counter() - t0) * 1000.0)
        info = env.apply_placement(X)
        costs.append(info.cost_total)
        served.append(info.pct_demand_served)
        lat.append(info.avg_access_latency_ms if np.isfinite(info.avg_access_latency_ms) else np.nan)
        env.step()
        obs = env.observe()
    return {
        "decision_ms": np.array(decision_times),
        "cost_total": np.array(costs),
        "pct_demand_served": np.array(served),
        "avg_latency_ms": np.array(lat),
    }


def scalability_experiment(seeds: int = 3) -> None:
    drl = DRLPlacementAgent(
        node_feature_dim=NTNEnvironment.NODE_FEATURE_DIM,
        svc_feature_dim=NTNEnvironment.SERVICE_FEATURE_DIM,
        embed_dim=64, n_heads=4, seed=0,
    )
    drl.load(str(RESULTS_DIR / "drl_policy.pt"))

    sizes = [4, 8, 12, 16, 20]  # UAV count
    agents = {
        "random": RandomPlacement(seed=11),
        "layer-aware": LayerAwareHeuristic(),
        "ilp": ILPPlacement(time_limit_s=10.0),
        "drl": drl,
    }
    out = {}
    for n_uav in sizes:
        n_services = max(8, n_uav)
        n_users = max(20, 2 * n_uav)
        base = EpisodeConfig(
            n_leo=4, n_haps=2, n_uav=n_uav,
            n_services=n_services, n_users=n_users,
            horizon=15, step_seconds=60.0,
            demand_intensity=1.0, seed=10000 + n_uav,
        )
        for s in range(seeds):
            cfg = dataclasses.replace(base, seed=base.seed + s)
            for agent_name, agent in agents.items():
                key = (n_uav, agent_name, s)
                m = _eval_episode(cfg, agent)
                out[f"n{n_uav}__{agent_name}__seed{s}__decision_ms"] = m["decision_ms"]
                out[f"n{n_uav}__{agent_name}__seed{s}__cost_total"] = m["cost_total"]
                out[f"n{n_uav}__{agent_name}__seed{s}__pct_demand_served"] = m["pct_demand_served"]
                print(
                    f"n_uav={n_uav:2d} seed={s} {agent_name:11s} "
                    f"cost={m['cost_total'].mean():8.2f} "
                    f"dec={np.median(m['decision_ms']):6.2f}ms "
                    f"served={m['pct_demand_served'].mean()*100:.0f}%"
                )
    np.savez(RESULTS_DIR / "scalability.npz", **out)


def ablation_experiment(seeds: int = 3) -> None:
    drl = DRLPlacementAgent(
        node_feature_dim=NTNEnvironment.NODE_FEATURE_DIM,
        svc_feature_dim=NTNEnvironment.SERVICE_FEATURE_DIM,
        embed_dim=64, n_heads=4, seed=0,
    )
    drl.load(str(RESULTS_DIR / "drl_policy.pt"))
    cfg_with = SCENARIOS["small"]
    cfg_without = dataclasses.replace(cfg_with, w_unserved=0.0)

    agents = {
        "greedy": GreedyLatencyPlacement(),
        "layer-aware": LayerAwareHeuristic(),
        "ilp_with": ILPPlacement(time_limit_s=8.0, w_unserved=25.0),
        "ilp_without": ILPPlacement(time_limit_s=8.0, w_unserved=0.0),
        "drl": drl,
    }
    out = {}
    for ablation_name, base in [("with", cfg_with), ("without", cfg_without)]:
        for s in range(seeds):
            cfg = dataclasses.replace(base, seed=base.seed + s)
            for agent_name, agent in agents.items():
                # ilp_with only matters for ablation_name="with"; ilp_without for "without"
                if agent_name == "ilp_with" and ablation_name == "without":
                    continue
                if agent_name == "ilp_without" and ablation_name == "with":
                    continue
                m = _eval_episode(cfg, agent)
                out[f"{ablation_name}__{agent_name}__seed{s}__cost_total"] = m["cost_total"]
                out[f"{ablation_name}__{agent_name}__seed{s}__pct_demand_served"] = m["pct_demand_served"]
                print(
                    f"unserved={ablation_name:7s} seed={s} {agent_name:13s} "
                    f"cost={m['cost_total'].mean():8.2f} "
                    f"served={m['pct_demand_served'].mean()*100:.0f}%"
                )
    np.savez(RESULTS_DIR / "ablation.npz", **out)


def main() -> None:
    print("=== Scalability ===")
    scalability_experiment(seeds=3)
    print("=== Ablation ===")
    ablation_experiment(seeds=3)


if __name__ == "__main__":
    main()
