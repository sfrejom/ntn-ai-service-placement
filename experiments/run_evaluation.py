"""Evaluate every placement policy across the four held-out scenarios.

For each (scenario, policy) pair we run `n_seeds` independent episodes and
record per-step metrics. The output is a single .npz file with a structured
table that the plotting scripts consume.
"""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from agents.baselines import (
    GreedyLatencyPlacement,
    LayerAwareHeuristic,
    PlacementAgent,
    RandomPlacement,
)
from agents.drl import DRLPlacementAgent
from agents.ilp import ILPPlacement
from experiments.scenarios import SCENARIOS
from simulator import EpisodeConfig, NTNEnvironment

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def _run_episode(env_cfg: EpisodeConfig, agent: PlacementAgent) -> Dict[str, np.ndarray]:
    env = NTNEnvironment(env_cfg)
    obs = env.reset()
    metrics = {
        "cost_total": [],
        "cost_latency": [],
        "cost_energy": [],
        "cost_migration": [],
        "cost_violation": [],
        "avg_latency_ms": [],
        "pct_demand_served": [],
        "decision_ms": [],
        "n_active_uav": [],
        "n_capacity_violations": [],
        "n_latency_violations": [],
    }
    for t in range(env_cfg.horizon):
        t0 = time.perf_counter()
        X = agent.place(env, obs)
        decision_ms = (time.perf_counter() - t0) * 1000.0
        info = env.apply_placement(X)
        n_active_uav = sum(1 for n in env.nodes if n.layer.value == "UAV" and n.active)
        metrics["cost_total"].append(info.cost_total)
        metrics["cost_latency"].append(info.cost_latency)
        metrics["cost_energy"].append(info.cost_energy)
        metrics["cost_migration"].append(info.cost_migration)
        metrics["cost_violation"].append(info.cost_violation)
        metrics["avg_latency_ms"].append(info.avg_access_latency_ms if np.isfinite(info.avg_access_latency_ms) else np.nan)
        metrics["pct_demand_served"].append(info.pct_demand_served)
        metrics["decision_ms"].append(decision_ms)
        metrics["n_active_uav"].append(n_active_uav)
        metrics["n_capacity_violations"].append(info.capacity_violations)
        metrics["n_latency_violations"].append(info.latency_violations)
        env.step()
        obs = env.observe()
    return {k: np.array(v) for k, v in metrics.items()}


def main(n_seeds: int = 5, ilp_time_limit: float = 6.0) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    # Construct policies. The DRL agent must be loaded only if a trained
    # checkpoint exists.
    drl = DRLPlacementAgent(
        node_feature_dim=NTNEnvironment.NODE_FEATURE_DIM,
        svc_feature_dim=NTNEnvironment.SERVICE_FEATURE_DIM,
        embed_dim=64,
        n_heads=4,
        seed=0,
    )
    drl_ckpt = RESULTS_DIR / "drl_policy.pt"
    if drl_ckpt.exists():
        drl.load(str(drl_ckpt))
    else:
        raise FileNotFoundError(
            f"DRL checkpoint not found at {drl_ckpt}. "
            "Run run_training.py first."
        )

    agents: Dict[str, PlacementAgent] = {
        "random": RandomPlacement(seed=7),
        "greedy": GreedyLatencyPlacement(),
        "layer-aware": LayerAwareHeuristic(),
        "ilp": ILPPlacement(time_limit_s=ilp_time_limit),
        "drl": drl,
    }

    results: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
    for scen_name, base_cfg in SCENARIOS.items():
        print(f"== Scenario: {scen_name} ==")
        results[scen_name] = {agent_name: {} for agent_name in agents}
        for seed_idx in range(n_seeds):
            cfg = dataclasses.replace(base_cfg, seed=base_cfg.seed + seed_idx)
            for agent_name, agent in agents.items():
                t0 = time.perf_counter()
                ep_metrics = _run_episode(cfg, agent)
                wall = time.perf_counter() - t0
                slot = results[scen_name][agent_name].setdefault(
                    "per_seed", []
                )
                slot.append(ep_metrics)
                print(
                    f"  seed={seed_idx} agent={agent_name:11s} "
                    f"avg_cost={ep_metrics['cost_total'].mean():.2f} "
                    f"avg_lat={np.nanmean(ep_metrics['avg_latency_ms']):.2f}ms "
                    f"served={ep_metrics['pct_demand_served'].mean()*100:.1f}% "
                    f"wall={wall:.2f}s"
                )

    # Convert to a single .npz friendly structure
    out: Dict[str, np.ndarray] = {}
    for scen_name, scen_data in results.items():
        for agent_name, agent_data in scen_data.items():
            per_seed = agent_data["per_seed"]
            T = per_seed[0]["cost_total"].shape[0]
            S = len(per_seed)
            for metric in per_seed[0].keys():
                stacked = np.stack([p[metric] for p in per_seed])  # (S, T)
                out[f"{scen_name}__{agent_name}__{metric}"] = stacked

    np.savez(RESULTS_DIR / "evaluation.npz", **out)

    # Also store a flat JSON summary
    summary = {}
    for scen_name in SCENARIOS:
        summary[scen_name] = {}
        for agent_name in agents:
            arr = out[f"{scen_name}__{agent_name}__cost_total"]
            lat = out[f"{scen_name}__{agent_name}__avg_latency_ms"]
            served = out[f"{scen_name}__{agent_name}__pct_demand_served"]
            decision = out[f"{scen_name}__{agent_name}__decision_ms"]
            mig = out[f"{scen_name}__{agent_name}__cost_migration"]
            cap_viol = out[f"{scen_name}__{agent_name}__n_capacity_violations"]
            summary[scen_name][agent_name] = {
                "mean_cost_total": float(arr.mean()),
                "std_cost_total": float(arr.mean(axis=1).std()),
                "mean_avg_latency_ms": float(np.nanmean(lat)),
                "mean_pct_demand_served": float(served.mean()),
                "median_decision_ms": float(np.median(decision)),
                "mean_migration_cost": float(mig.mean()),
                "mean_capacity_violations": float(cap_viol.mean()),
            }
    with open(RESULTS_DIR / "evaluation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--ilp-time-limit", type=float, default=6.0)
    args = p.parse_args()
    main(args.seeds, args.ilp_time_limit)
