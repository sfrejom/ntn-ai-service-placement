"""Scenario factories used for both training and evaluation.

Each factory takes a seed and returns an `EpisodeConfig`. The training factory
applies domain randomisation over node counts, demand intensity, and number of
services. The evaluation scenarios are deterministic so that all policies are
compared on the exact same dynamics.

Scenarios:

* `small`:   16 nodes, 8 services, baseline demand, horizon 30.
* `large`:   24 nodes, 10 services, baseline demand, horizon 30.
* `surge`:   16 nodes, 8 services, demand spike (intensity 2.0), horizon 30.
* `volatile`: same as `small` but with shorter step (30 s) and higher UAV
              count, exposing more energy-driven replacements.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from simulator import EpisodeConfig


def make_training_factory(rng_seed: int = 0) -> Callable[[int], EpisodeConfig]:
    rng_master = np.random.default_rng(rng_seed)

    def factory(seed: int) -> EpisodeConfig:
        rng = np.random.default_rng((rng_master.integers(0, 2**31 - 1), seed))
        return EpisodeConfig(
            n_leo=int(rng.integers(3, 6)),
            n_haps=int(rng.integers(1, 3)),
            n_uav=int(rng.integers(6, 11)),
            n_services=int(rng.integers(6, 11)),
            n_users=int(rng.integers(20, 41)),
            region_radius_km=80.0,
            horizon=int(rng.integers(15, 25)),
            step_seconds=60.0,
            demand_intensity=float(rng.uniform(0.6, 1.6)),
            seed=seed,
        )

    return factory


SCENARIOS = {
    "small": EpisodeConfig(
        n_leo=4, n_haps=2, n_uav=8, n_services=8, n_users=30,
        region_radius_km=80.0, horizon=30, step_seconds=60.0,
        demand_intensity=1.0, seed=2025,
    ),
    "large": EpisodeConfig(
        n_leo=6, n_haps=2, n_uav=10, n_services=10, n_users=40,
        region_radius_km=100.0, horizon=30, step_seconds=60.0,
        demand_intensity=1.0, seed=2026,
    ),
    "surge": EpisodeConfig(
        n_leo=4, n_haps=2, n_uav=8, n_services=8, n_users=30,
        region_radius_km=80.0, horizon=30, step_seconds=60.0,
        demand_intensity=2.0, seed=2027,
    ),
    "volatile": EpisodeConfig(
        n_leo=4, n_haps=2, n_uav=10, n_services=8, n_users=30,
        region_radius_km=80.0, horizon=40, step_seconds=30.0,
        demand_intensity=1.0, seed=2028,
    ),
}
