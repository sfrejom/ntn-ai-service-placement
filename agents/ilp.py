"""ILP placement baseline.

The placement decision in a single round is encoded as the ILP

    min  sum_{i,j} d_j * lat_ij * x_ij + lambda_e * sum_{i,j} c_j * x_ij / E_i
       + lambda_m * sum_{j: stateful_j, i} mu_j * |x_ij - x^prev_ij|
    s.t. sum_j c_j * x_ij <= C_i           for all i
         sum_j m_j * x_ij <= M_i           for all i
         sum_i x_ij <= K_j                  for all j
         sum_i x_ij >= 1                    for all j (every service must be placed somewhere)
         x_ij = 0  if base_delay_i > lambda_j or coverage_i misses every user

Latency cost per (i, j) is the demand-weighted sum of access latency over
users currently within node i's coverage. Migration cost is linearised by
introducing auxiliary y_ij = |x_ij - x^prev_ij| with the standard pair of
inequalities.

This is solved with PuLP/CBC. The instance sizes used in the experiments
(~16 nodes x ~10 services) are well within solver reach. The ILP serves as a
strong (effectively oracle) baseline against which the DRL agent is compared.
"""

from __future__ import annotations

import time

import numpy as np
import pulp

from simulator.environment import NTNEnvironment, Observation
from .baselines import PlacementAgent


class ILPPlacement(PlacementAgent):
    name = "ilp"

    def __init__(
        self,
        time_limit_s: float = 5.0,
        w_latency: float = 1.0,
        w_energy: float = 1.0,
        w_migration: float = 1.0,
        w_unserved: float = 25.0,
    ) -> None:
        self.time_limit_s = time_limit_s
        self.w_latency = w_latency
        self.w_energy = w_energy
        self.w_migration = w_migration
        self.w_unserved = w_unserved
        self.last_solve_time_s: float = 0.0
        self.last_status: str = ""

    def place(self, env: NTNEnvironment, obs: Observation) -> np.ndarray:
        N = len(env.nodes)
        M = len(env.services)
        feas = env.feasibility_mask()

        # Pre-compute per (i, j) latency cost contributions
        # cost_ij = sum_u d_j * (base_delay_i + dist(u, i)/c)  for u within coverage
        U = len(env.users)
        lat_cost = np.zeros((N, M))
        coverage_ok = np.zeros((N, M), dtype=bool)
        # serves[i, j, u] = 1 iff selecting node i for service j would let it
        # serve user u (within coverage and within latency budget). Used to
        # build the unserved-demand z variables below.
        serves = np.zeros((N, M, U), dtype=bool)
        for i, node in enumerate(env.nodes):
            ground = np.linalg.norm(env.users[:, :2] - node.position[None, :2], axis=1)
            in_cov = ground <= node.coverage_radius_km
            if not in_cov.any():
                continue
            prop = ground / 299.792458
            for j, svc in enumerate(env.services):
                if not feas[i, j]:
                    continue
                full_lat = node.base_delay_ms + prop
                serve_mask = in_cov & (full_lat <= svc.max_latency_ms)
                if not serve_mask.any():
                    continue
                # Pessimistic: average latency, weighted by demand
                lat_cost[i, j] = float(env.user_demand[j]) * float(full_lat[serve_mask].mean())
                coverage_ok[i, j] = True
                serves[i, j, :] = serve_mask

        # Energy cost penalises placement on energy-bound nodes
        eng_cost = np.zeros((N, M))
        for i, node in enumerate(env.nodes):
            if not np.isfinite(node.energy_capacity_wh):
                continue
            for j, svc in enumerate(env.services):
                eng_cost[i, j] = svc.cpu_demand / max(node.energy_remaining_wh, 1e-3)

        # Build the ILP
        prob = pulp.LpProblem("ntn_placement", pulp.LpMinimize)

        x = {}
        for i in range(N):
            for j in range(M):
                if coverage_ok[i, j] and env.nodes[i].active:
                    x[i, j] = pulp.LpVariable(f"x_{i}_{j}", lowBound=0, upBound=1, cat=pulp.LpBinary)

        # Migration aux variables for stateful services
        y = {}
        prev = obs.previous_assignment
        for j, svc in enumerate(env.services):
            if not svc.is_stateful:
                continue
            for i in range(N):
                if (i, j) not in x:
                    continue
                y[i, j] = pulp.LpVariable(f"y_{i}_{j}", lowBound=0, upBound=1, cat=pulp.LpContinuous)

        # Coverage variables z_{u, j}: 1 iff at least one selected replica
        # of service j covers user u. Adding these lets the ILP minimise the
        # extended objective that penalises uncovered demand. Without them,
        # the ILP could "win" by placing services on tiny-coverage UAVs and
        # ignoring the users they cannot reach.
        z = {}
        for j in range(M):
            for u in range(U):
                # Only useful if at least one feasible node would serve (u, j)
                serving_nodes = [i for i in range(N) if (i, j) in x and serves[i, j, u]]
                if not serving_nodes:
                    continue
                z[u, j] = pulp.LpVariable(f"z_{u}_{j}", lowBound=0, upBound=1, cat=pulp.LpContinuous)
                # z_{u,j} <= sum_i x_{i,j} over serving nodes
                prob += z[u, j] <= pulp.lpSum(x[i, j] for i in serving_nodes), f"cov_{u}_{j}"

        # Objective
        objective = pulp.lpSum(
            self.w_latency * lat_cost[i, j] * x[i, j]
            + self.w_energy * eng_cost[i, j] * x[i, j]
            for (i, j) in x
        )
        if y:
            objective += pulp.lpSum(
                self.w_migration * env.services[j].migration_cost * y[i, j] for (i, j) in y
            )
        # Unserved demand penalty: w_unserved * sum_{u, j} D_j * (1 - z_{u, j})
        # Constants drop out of the optimisation, so we add a -w_unserved *
        # D_j * z_{u, j} term. Pairs with no feasible serving node always
        # contribute the full penalty (no z variable, treated as z=0).
        if z:
            objective += pulp.lpSum(
                -self.w_unserved * float(env.user_demand[j]) * z[u, j]
                for (u, j) in z
            )
        prob += objective

        # Capacity constraints
        for i in range(N):
            node = env.nodes[i]
            cpu_terms = [env.services[j].cpu_demand * x[i, j] for j in range(M) if (i, j) in x]
            mem_terms = [env.services[j].mem_demand * x[i, j] for j in range(M) if (i, j) in x]
            if cpu_terms:
                prob += pulp.lpSum(cpu_terms) <= node.cpu_capacity, f"cpu_{i}"
            if mem_terms:
                prob += pulp.lpSum(mem_terms) <= node.mem_capacity, f"mem_{i}"

        # Replication constraints + at-least-one-copy where feasible
        for j, svc in enumerate(env.services):
            terms = [x[i, j] for i in range(N) if (i, j) in x]
            if not terms:
                continue
            prob += pulp.lpSum(terms) <= svc.max_replicas, f"replicas_{j}"
            prob += pulp.lpSum(terms) >= 1, f"placed_{j}"

        # Migration aux constraints
        for (i, j), yij in y.items():
            prev_val = float(prev[i, j]) if prev.shape == (N, M) else 0.0
            prob += yij >= x[i, j] - prev_val, f"y_pos_{i}_{j}"
            prob += yij >= prev_val - x[i, j], f"y_neg_{i}_{j}"

        # Solve
        t0 = time.perf_counter()
        solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=self.time_limit_s)
        status = prob.solve(solver)
        self.last_solve_time_s = time.perf_counter() - t0
        self.last_status = pulp.LpStatus.get(status, str(status))

        X = np.zeros((N, M), dtype=np.int32)
        for (i, j), var in x.items():
            v = var.value()
            if v is not None and v > 0.5:
                X[i, j] = 1
        return X
