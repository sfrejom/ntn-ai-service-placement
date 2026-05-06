"""Episode-driven NTN environment.

Each episode is a sequence of placement rounds. Between rounds, the simulator
advances time by `step_seconds`, which moves LEO satellites along their
ground tracks, drains UAV batteries, replaces UAVs whose battery has run out,
and resamples user demand. A placement round consists of:

1. Observe: caller reads `obs()` for the current network state.
2. Decide: caller produces a placement matrix X (N x M, binary).
3. Apply: `apply_placement(X)` accounts for capacity, latency and replication
   constraints. The cost terms (latency, energy, migration, violation) are
   reported in a `Step` named-tuple.
4. Advance: time progresses one tick, the world is updated.

The environment is policy-agnostic: baselines and the DRL agent both consume
the same `Observation`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .nodes import Node, NodeLayer, build_node_population
from .services import Microservice, build_service_catalog


@dataclass
class EpisodeConfig:
    n_leo: int = 4
    n_haps: int = 2
    n_uav: int = 8
    n_services: int = 8
    n_users: int = 30
    region_radius_km: float = 80.0
    horizon: int = 30
    step_seconds: float = 60.0
    demand_intensity: float = 1.0
    seed: Optional[int] = None
    # Cost weights, mirroring the position-paper objective
    w_latency: float = 1.0
    w_energy: float = 1.0
    w_migration: float = 1.0
    w_violation: float = 50.0
    # Per-unit penalty for uncovered (user, service) demand. This is the
    # extension the experimental paper introduces to the original cost
    # formulation: it forbids policies from "winning" by placing every
    # service on a tiny-coverage UAV and ignoring the demand they cannot
    # serve. Documented in CHANGES.md.
    w_unserved: float = 25.0
    # Migration penalty is per-second of estimated downtime
    migration_unit_cost: float = 1.0


@dataclass
class Observation:
    node_features: np.ndarray   # (N, F_n)
    service_features: np.ndarray  # (M, F_s)
    demand_per_service: np.ndarray  # (M,)
    previous_assignment: np.ndarray  # (N, M)
    user_positions: np.ndarray  # (U, 3) for visualisation
    layer_mask: np.ndarray      # (N, 3) one-hot of layer
    available_mask: np.ndarray  # (N,) 1 if node is active and within coverage


@dataclass
class StepInfo:
    cost_latency: float
    cost_energy: float
    cost_migration: float
    cost_violation: float
    placed_services: int
    capacity_violations: int
    latency_violations: int
    avg_access_latency_ms: float
    pct_demand_served: float
    n_active_nodes: int
    cost_total: float


class NTNEnvironment:
    """Stateful NTN environment for cross-layer placement experiments."""

    NODE_FEATURE_DIM = 12
    SERVICE_FEATURE_DIM = 7

    def __init__(self, config: EpisodeConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.t: int = 0
        self.nodes: List[Node] = []
        self.services: List[Microservice] = []
        self.users: np.ndarray = np.zeros((0, 3))
        self.user_demand: np.ndarray = np.zeros(0)
        self.previous_X: np.ndarray = np.zeros((0, 0), dtype=np.int32)
        self._initialised: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def reset(self) -> Observation:
        cfg = self.config
        self.rng = np.random.default_rng(cfg.seed)
        self.nodes = build_node_population(
            self.rng,
            n_leo=cfg.n_leo,
            n_haps=cfg.n_haps,
            n_uav=cfg.n_uav,
            region_radius_km=cfg.region_radius_km,
        )
        self.services = build_service_catalog(self.rng, size=cfg.n_services)
        # Users uniformly scattered in the region
        self.users = np.column_stack(
            [
                self.rng.uniform(-cfg.region_radius_km, cfg.region_radius_km, size=cfg.n_users),
                self.rng.uniform(-cfg.region_radius_km, cfg.region_radius_km, size=cfg.n_users),
                np.zeros(cfg.n_users),
            ]
        )
        self.user_demand = self._sample_demand()
        self.previous_X = np.zeros((len(self.nodes), len(self.services)), dtype=np.int32)
        self.t = 0
        self._initialised = True
        return self.observe()

    def _sample_demand(self) -> np.ndarray:
        # Demand per service, scaled by configured intensity
        cfg = self.config
        d = self.rng.gamma(shape=2.0, scale=1.0, size=cfg.n_services)
        d = d * cfg.demand_intensity
        return d

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def observe(self) -> Observation:
        layer_idx = {NodeLayer.LEO: 0, NodeLayer.HAPS: 1, NodeLayer.UAV: 2}
        N = len(self.nodes)
        M = len(self.services)
        nfeat = np.zeros((N, self.NODE_FEATURE_DIM), dtype=np.float32)
        layer_mask = np.zeros((N, 3), dtype=np.float32)
        available = np.zeros(N, dtype=np.float32)
        for i, n in enumerate(self.nodes):
            nfeat[i, 0] = layer_idx[n.layer]  # categorical, also one-hot below
            nfeat[i, 1] = n.position[0] / 100.0
            nfeat[i, 2] = n.position[1] / 100.0
            nfeat[i, 3] = n.altitude_km / 1500.0
            nfeat[i, 4] = n.cpu_load
            nfeat[i, 5] = n.mem_load
            nfeat[i, 6] = n.energy_fraction
            nfeat[i, 7] = n.base_delay_ms
            nfeat[i, 8] = float(n.cpu_capacity)
            nfeat[i, 9] = float(n.mem_capacity)
            nfeat[i, 10] = 1.0 if n.active else 0.0
            nfeat[i, 11] = n.coverage_radius_km / 600.0
            layer_mask[i, layer_idx[n.layer]] = 1.0
            available[i] = 1.0 if n.active else 0.0

        sfeat = np.zeros((M, self.SERVICE_FEATURE_DIM), dtype=np.float32)
        for j, s in enumerate(self.services):
            sfeat[j, 0] = s.cpu_demand
            sfeat[j, 1] = s.mem_demand
            sfeat[j, 2] = s.max_latency_ms
            sfeat[j, 3] = s.bandwidth_mbps
            sfeat[j, 4] = float(s.is_stateful)
            sfeat[j, 5] = s.state_size_mb / 500.0
            sfeat[j, 6] = float(s.max_replicas)

        return Observation(
            node_features=nfeat,
            service_features=sfeat,
            demand_per_service=self.user_demand.astype(np.float32),
            previous_assignment=self.previous_X.copy(),
            user_positions=self.users.copy(),
            layer_mask=layer_mask,
            available_mask=available,
        )

    # ------------------------------------------------------------------
    # Placement evaluation and dynamics
    # ------------------------------------------------------------------
    def _node_user_propagation_ms(self, node: Node) -> np.ndarray:
        """Return per-user one-way propagation delay in ms."""
        if len(self.users) == 0:
            return np.zeros(0)
        diffs = self.users - node.position[None, :]
        dist_km = np.linalg.norm(diffs, axis=1)
        return dist_km / (299.792458 * 1.0)  # 1ms per ~300 km of LoS path

    def evaluate_placement(self, X: np.ndarray) -> StepInfo:
        """Evaluate the placement matrix and return cost components.

        Important: this routine does not mutate node state. `apply_placement`
        does. Callers that just want to score a candidate placement (e.g. a
        baseline or an ILP solver) can use this directly.
        """
        N, M = X.shape
        cfg = self.config

        cost_latency = 0.0
        cost_energy = 0.0
        cost_violation = 0.0
        capacity_violations = 0
        latency_violations = 0
        served_per_user_service = np.zeros((len(self.users), M))
        latencies_per_user_service = np.full((len(self.users), M), np.inf)

        # Per-node load tally for capacity-violation checking
        cpu_used = np.zeros(N)
        mem_used = np.zeros(N)
        for j, svc in enumerate(self.services):
            replica_count = int(X[:, j].sum())
            if replica_count == 0:
                # All users for this service are unserved
                cost_violation += cfg.w_violation * float(self.user_demand[j])
                continue
            if replica_count > svc.max_replicas:
                cost_violation += cfg.w_violation * (replica_count - svc.max_replicas)

            for i, node in enumerate(self.nodes):
                if X[i, j] == 0:
                    continue
                if not node.active:
                    cost_violation += cfg.w_violation
                    continue
                # Latency constraint check
                if node.base_delay_ms > svc.max_latency_ms:
                    latency_violations += 1
                    cost_violation += cfg.w_violation
                # Aggregate node usage
                cpu_used[i] += svc.cpu_demand
                mem_used[i] += svc.mem_demand
                # Per-user serving feasibility: must be within coverage
                prop = self._node_user_propagation_ms(node)
                full_lat = node.base_delay_ms + prop
                # Coverage: ground-projected horizontal distance must fit
                # within the node's footprint radius.
                ground_dist = np.linalg.norm(
                    self.users[:, :2] - node.position[None, :2], axis=1
                )
                in_coverage = ground_dist <= node.coverage_radius_km
                feasible = in_coverage & (full_lat <= svc.max_latency_ms)
                # For each feasible user, record the minimum latency replica
                latencies_per_user_service[feasible, j] = np.minimum(
                    latencies_per_user_service[feasible, j], full_lat[feasible]
                )
                served_per_user_service[feasible, j] = 1.0
                # Energy cost is non-zero only for energy-bound nodes
                if np.isfinite(node.energy_capacity_wh) and node.energy_remaining_wh > 0:
                    cost_energy += cfg.w_energy * svc.cpu_demand / (node.energy_remaining_wh + 1e-3)

        # Capacity overflow detection
        for i, node in enumerate(self.nodes):
            if cpu_used[i] > node.cpu_capacity + 1e-6:
                capacity_violations += 1
                cost_violation += cfg.w_violation * (cpu_used[i] - node.cpu_capacity)
            if mem_used[i] > node.mem_capacity + 1e-6:
                capacity_violations += 1
                cost_violation += cfg.w_violation * (mem_used[i] - node.mem_capacity)

        # Latency cost: only for users that have at least one serving replica.
        # Per-user / per-service min latency replaced by 0 if unserved (mask
        # zeros it out below).
        masked = np.where(np.isfinite(latencies_per_user_service), latencies_per_user_service, 0.0)
        weight = self.user_demand[None, :] * served_per_user_service
        cost_latency = cfg.w_latency * float((masked * weight).sum())

        # Unserved-demand cost (extension to the original objective): for
        # every (user, service) pair with non-zero demand that is not covered
        # by any selected replica, we add a flat penalty per unit of demand.
        # Without this term, a policy can drive the cost arbitrarily low by
        # placing every service on a single tiny-coverage UAV, since the
        # latency cost only sums over users that *are* served.
        unserved_mask = 1.0 - served_per_user_service
        cost_unserved = cfg.w_unserved * float(
            (unserved_mask * self.user_demand[None, :]).sum()
        )
        cost_violation += cost_unserved

        # Migration cost: change between previous and current assignment for stateful services
        cost_migration = 0.0
        if self.previous_X.shape == X.shape:
            diff = np.abs(X - self.previous_X)
            for j, svc in enumerate(self.services):
                if svc.is_stateful:
                    cost_migration += cfg.w_migration * cfg.migration_unit_cost * svc.migration_cost * float(diff[:, j].sum())

        # Aggregate metrics for reporting
        finite_lat = latencies_per_user_service[np.isfinite(latencies_per_user_service)]
        avg_lat = float(finite_lat.mean()) if finite_lat.size > 0 else float("inf")
        # Demand-weighted fraction of served (user, service) pairs. This is
        # the metric the experimental paper reports as "QoS coverage".
        total_demand = float((self.user_demand[None, :] * np.ones((len(self.users), len(self.services)))).sum())
        served_demand = float((served_per_user_service * self.user_demand[None, :]).sum())
        pct_served = served_demand / max(total_demand, 1e-9)

        cost_total = cost_latency + cost_energy + cost_migration + cost_violation
        return StepInfo(
            cost_latency=cost_latency,
            cost_energy=cost_energy,
            cost_migration=cost_migration,
            cost_violation=cost_violation,
            placed_services=int((X.sum(axis=0) > 0).sum()),
            capacity_violations=capacity_violations,
            latency_violations=latency_violations,
            avg_access_latency_ms=avg_lat if np.isfinite(avg_lat) else float("inf"),
            pct_demand_served=pct_served,
            n_active_nodes=int(sum(n.active for n in self.nodes)),
            cost_total=cost_total,
        )

    def apply_placement(self, X: np.ndarray) -> StepInfo:
        info = self.evaluate_placement(X)
        # Push the load on the actual node objects so that the energy step can
        # consume battery proportionally to the CPU load assigned this round.
        for n in self.nodes:
            n.reset_allocation()
        for j, svc in enumerate(self.services):
            for i, n in enumerate(self.nodes):
                if X[i, j] == 1 and n.active:
                    n.cpu_used += svc.cpu_demand
                    n.mem_used += svc.mem_demand
                    n.deployed_services.append(j)
        self.previous_X = X.copy()
        return info

    def step(self) -> None:
        """Advance time one tick, applying mobility, energy, replacements and
        new demand."""
        cfg = self.config
        for n in self.nodes:
            n.step_position(cfg.step_seconds)
            n.consume_energy(cfg.step_seconds)
        self._replace_dead_uavs()
        self._handle_leo_handover()
        # Resample demand: small autoregressive change
        self.user_demand = 0.7 * self.user_demand + 0.3 * self._sample_demand()
        self.t += 1

    def _replace_dead_uavs(self) -> None:
        cfg = self.config
        for i, n in enumerate(self.nodes):
            if n.layer == NodeLayer.UAV and not n.active:
                # Replace with a fresh UAV at a random location in the region
                position = np.array(
                    [
                        self.rng.uniform(-cfg.region_radius_km, cfg.region_radius_km),
                        self.rng.uniform(-cfg.region_radius_km, cfg.region_radius_km),
                        self.rng.uniform(0.1, 1.5),
                    ]
                )
                battery = float(self.rng.uniform(80.0, 140.0))
                self.nodes[i] = Node(
                    node_id=n.node_id,
                    layer=NodeLayer.UAV,
                    position=position,
                    cpu_capacity=4.0,
                    mem_capacity=8.0,
                    energy_capacity_wh=battery,
                    energy_remaining_wh=battery,
                    base_idle_power_w=120.0,
                    base_compute_power_w=15.0,
                    coverage_radius_km=10.0,
                )
                # The previous assignment for this index is no longer valid
                self.previous_X[i, :] = 0

    def _handle_leo_handover(self) -> None:
        """If a LEO satellite has crossed the region and is far away, re-spawn
        it on the opposite edge to maintain a steady population."""
        cfg = self.config
        for i, n in enumerate(self.nodes):
            if n.layer != NodeLayer.LEO:
                continue
            distance = float(np.linalg.norm(n.position[:2]))
            if distance > 300.0:
                # Reverse direction and reset to the boundary
                angle = self.rng.uniform(0.0, 2.0 * np.pi)
                position = np.array(
                    [200.0 * np.cos(angle), 200.0 * np.sin(angle), n.position[2]]
                )
                speed = 7.0
                direction = -position[:2] / np.linalg.norm(position[:2])
                velocity = np.array([direction[0] * speed, direction[1] * speed, 0.0])
                self.nodes[i] = Node(
                    node_id=n.node_id,
                    layer=NodeLayer.LEO,
                    position=position,
                    cpu_capacity=n.cpu_capacity,
                    mem_capacity=n.mem_capacity,
                    energy_capacity_wh=n.energy_capacity_wh,
                    energy_remaining_wh=n.energy_remaining_wh,
                    base_idle_power_w=0.0,
                    base_compute_power_w=0.0,
                    velocity=velocity,
                    coverage_radius_km=n.coverage_radius_km,
                )
                self.previous_X[i, :] = 0

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def feasibility_mask(self) -> np.ndarray:
        """Return an (N, M) boolean mask where mask[i, j] is True iff node i
        currently meets the latency constraint of service j and is active.
        Capacity is *not* checked here -- it is left to the policy."""
        N, M = len(self.nodes), len(self.services)
        mask = np.zeros((N, M), dtype=bool)
        for i, n in enumerate(self.nodes):
            if not n.active:
                continue
            for j, s in enumerate(self.services):
                if n.base_delay_ms <= s.max_latency_ms:
                    mask[i, j] = True
        return mask
