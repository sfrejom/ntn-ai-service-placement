"""Train the DRL placement agent.

Outputs:
    results/training_history.npz   -- per-update training log
    results/drl_policy.pt          -- trained model parameters
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from agents.drl import DRLPlacementAgent, TrainConfig
from experiments.scenarios import make_training_factory
from simulator import NTNEnvironment

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def main(total_episodes: int = 600) -> None:
    factory = make_training_factory(rng_seed=42)
    cfg = TrainConfig(
        total_episodes=total_episodes,
        rollout_episodes=4,
        epochs_per_update=4,
        minibatch_size=16,
        learning_rate=3e-4,
        clip_eps=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        gamma=0.95,
        gae_lambda=0.9,
        embed_dim=64,
        n_heads=4,
        seed=0,
    )

    agent = DRLPlacementAgent(
        node_feature_dim=NTNEnvironment.NODE_FEATURE_DIM,
        svc_feature_dim=NTNEnvironment.SERVICE_FEATURE_DIM,
        embed_dim=cfg.embed_dim,
        n_heads=cfg.n_heads,
        seed=cfg.seed,
    )

    history = []
    def cb(log):
        history.append(log)
        if log["episodes_done"] % 20 == 0:
            print(
                f"ep={log['episodes_done']:4d} "
                f"reward={log['mean_episode_reward']:8.2f} "
                f"pl={log['policy_loss']:.3f} "
                f"vl={log['value_loss']:.3f} "
                f"ent={log['entropy']:.3f}"
            )

    t0 = time.perf_counter()
    agent.train(factory, cfg, log_callback=cb)
    wall = time.perf_counter() - t0
    print(f"Training finished in {wall:.1f}s ({total_episodes} episodes)")

    # Save
    agent.save(str(RESULTS_DIR / "drl_policy.pt"))
    np.savez(
        RESULTS_DIR / "training_history.npz",
        episodes=np.array([h["episodes_done"] for h in history]),
        rewards=np.array([h["mean_episode_reward"] for h in history]),
        policy_loss=np.array([h["policy_loss"] for h in history]),
        value_loss=np.array([h["value_loss"] for h in history]),
        entropy=np.array([h["entropy"] for h in history]),
        wall_seconds=np.array([wall]),
    )
    with open(RESULTS_DIR / "training_meta.json", "w") as f:
        json.dump(
            {
                "total_episodes": total_episodes,
                "wall_seconds": wall,
                "config": cfg.__dict__,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=600)
    args = p.parse_args()
    main(args.episodes)
