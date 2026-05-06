"""Baseline placement policies.

All baselines share the `PlacementAgent` interface: they take an `Observation`
and the current `NTNEnvironment` (read-only) and return an (N, M) binary
placement matrix.

The three baselines are:

* `RandomPlacement`: assigns each service to a random feasible node, ignoring
  capacity and previous assignments. Serves as a lower bound on quality.
* `GreedyLatencyPlacement`: per-service, pick the feasible node minimising the
  expected user-side access latency, subject to the running CPU/memory tally
  to keep capacity feasibility.
* `LayerAwareHeuristic`: a rule-based policy encoding the layer-suitability
  matrix from the position paper (UAV for tight latency / stateful, HAPS for
  mid-latency, LEO for latency-tolerant / wide-area), with first-fit within
  the chosen layer and continuity bias for stateful services already placed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import numpy as np

from simulator.environment import NTNEnvironment, Observation
from simulator.nodes import NodeLayer


class PlacementAgent(ABC):
    name: str = "agent"

    @abstractmethod
    def place(self, env: NTNEnvironment, obs: Observation) -> np.ndarray:
        """Return an (N, M) int array."""

    def reset(self) -> None:
        """Hook for stateful agents."""


class RandomPlacement(PlacementAgent):
    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def place(self, env: NTNEnvironment, obs: Observation) -> np.ndarray:
        N = len(env.nodes)
        M = len(env.services)
        feas = env.feasibility_mask()
        X = np.zeros((N, M), dtype=np.int32)
        for j in range(M):
            cands = np.where(feas[:, j])[0]
            if len(cands) == 0:
                continue
            i = int(self.rng.choice(cands))
            X[i, j] = 1
        return X


class GreedyLatencyPlacement(PlacementAgent):
    """Pick the lowest-latency feasible node for each service, with a running
    capacity tally so that we don't blow capacity blindly."""

    name = "greedy"

    def place(self, env: NTNEnvironment, obs: Observation) -> np.ndarray:
        N = len(env.nodes)
        M = len(env.services)
        feas = env.feasibility_mask()
        X = np.zeros((N, M), dtype=np.int32)
        cpu_tally = np.zeros(N)
        mem_tally = np.zeros(N)
        # Sort services by tightness of latency budget (tightest first)
        order = sorted(range(M), key=lambda j: env.services[j].max_latency_ms)
        for j in order:
            svc = env.services[j]
            # Mean expected latency per candidate node (base + average user prop)
            scores = []
            for i, node in enumerate(env.nodes):
                if not feas[i, j]:
                    continue
                if cpu_tally[i] + svc.cpu_demand > node.cpu_capacity:
                    continue
                if mem_tally[i] + svc.mem_demand > node.mem_capacity:
                    continue
                # Average user latency to this node
                ground = np.linalg.norm(env.users[:, :2] - node.position[None, :2], axis=1)
                in_cov = ground <= node.coverage_radius_km
                if not in_cov.any():
                    continue
                avg_dist_km = ground[in_cov].mean()
                lat = node.base_delay_ms + avg_dist_km / 299.792458
                scores.append((lat, i))
            if not scores:
                continue
            scores.sort()
            i_best = scores[0][1]
            X[i_best, j] = 1
            cpu_tally[i_best] += svc.cpu_demand
            mem_tally[i_best] += svc.mem_demand
        return X


class LayerAwareHeuristic(PlacementAgent):
    """Rule-based policy encoding the layer-suitability matrix from the
    position paper.

    Rules (ordered by service category):
    * If service is stateful AND latency-critical (<= 10 ms), prefer UAV. If
      no UAV fits, fall back to HAPS, then LEO.
    * If service is latency-critical (<= 10 ms) and stateless, prefer UAV,
      then HAPS.
    * If service is mid-latency (<= 50 ms), prefer HAPS, then LEO, then UAV.
    * Otherwise (latency-tolerant), prefer LEO, then HAPS.

    Within the chosen layer we apply first-fit by capacity, with a continuity
    bias for stateful services: if the previous placement node still fits the
    capacity check, keep it.
    """

    name = "layer-aware"

    def place(self, env: NTNEnvironment, obs: Observation) -> np.ndarray:
        N = len(env.nodes)
        M = len(env.services)
        feas = env.feasibility_mask()
        X = np.zeros((N, M), dtype=np.int32)
        cpu_tally = np.zeros(N)
        mem_tally = np.zeros(N)
        prev = obs.previous_assignment

        layer_buckets = {layer: [] for layer in NodeLayer}
        for i, node in enumerate(env.nodes):
            if node.active:
                layer_buckets[node.layer].append(i)

        # Sort services so stateful + latency-critical are placed first
        def priority(j: int) -> tuple:
            s = env.services[j]
            return (s.max_latency_ms, -int(s.is_stateful), s.cpu_demand)

        order = sorted(range(M), key=priority)

        for j in order:
            svc = env.services[j]
            # Decide preferred layer order
            if svc.max_latency_ms <= 10.0:
                layer_pref = [NodeLayer.UAV, NodeLayer.HAPS, NodeLayer.LEO]
            elif svc.max_latency_ms <= 50.0:
                layer_pref = [NodeLayer.HAPS, NodeLayer.UAV, NodeLayer.LEO]
            else:
                layer_pref = [NodeLayer.LEO, NodeLayer.HAPS, NodeLayer.UAV]

            placed = False
            # Continuity for stateful: try previous host first if still feasible
            if svc.is_stateful and prev.shape == X.shape:
                prev_hosts = np.where(prev[:, j] == 1)[0]
                for i in prev_hosts:
                    node = env.nodes[i]
                    if (
                        feas[i, j]
                        and node.active
                        and cpu_tally[i] + svc.cpu_demand <= node.cpu_capacity
                        and mem_tally[i] + svc.mem_demand <= node.mem_capacity
                    ):
                        X[i, j] = 1
                        cpu_tally[i] += svc.cpu_demand
                        mem_tally[i] += svc.mem_demand
                        placed = True
                        break

            if placed:
                continue

            for layer in layer_pref:
                # Sort candidates within the layer by remaining capacity slack
                cands = [i for i in layer_buckets[layer] if feas[i, j]]
                cands.sort(
                    key=lambda i: (
                        env.nodes[i].cpu_capacity - cpu_tally[i] - svc.cpu_demand,
                    ),
                    reverse=True,
                )
                for i in cands:
                    node = env.nodes[i]
                    if (
                        cpu_tally[i] + svc.cpu_demand <= node.cpu_capacity
                        and mem_tally[i] + svc.mem_demand <= node.mem_capacity
                    ):
                        # Coverage sanity check
                        ground = np.linalg.norm(
                            env.users[:, :2] - node.position[None, :2], axis=1
                        )
                        if (ground <= node.coverage_radius_km).any():
                            X[i, j] = 1
                            cpu_tally[i] += svc.cpu_demand
                            mem_tally[i] += svc.mem_demand
                            placed = True
                            break
                if placed:
                    break
        return X
