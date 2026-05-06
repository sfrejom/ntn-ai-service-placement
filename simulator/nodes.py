"""NTN node models: LEO satellites, HAPS, and UAVs.

Each node tracks its position, capacity, current load, energy state, and a base
propagation delay derived from its layer/altitude. The choice of layer-specific
parameters follows the ranges quoted in the position paper (Table I) and in the
3GPP TR 38.811 / TR 38.821 channel-model documents:

* GEO: ignored in the simulator. Its >600 ms RTT places it outside the latency
  envelope of any candidate microservice and would never be picked by any
  reasonable placement policy. Including it would only inflate the action
  space.
* LEO: altitudes 550-1200 km (representative of Starlink/OneWeb shells), with
  a one-way base propagation delay of 2-7 ms. Ground-track velocity ~7 km/s
  drives a transit time of a few minutes across the regional footprint we
  simulate.
* HAPS: 18-25 km altitude, quasi-stationary, with a base one-way delay of
  60-170 us purely from propagation; we add a small access-network constant
  to land in the 1-5 ms RTT range of the position paper.
* UAVs: <2 km altitude, mobile, severely energy- and compute-constrained.
  Battery endurance is in the order of tens of minutes for consumer-grade
  rotary-wing platforms.

All distances are in kilometres, all delays in milliseconds, all energies in
watt-hours unless stated otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

import numpy as np


class NodeLayer(Enum):
    LEO = "LEO"
    HAPS = "HAPS"
    UAV = "UAV"


# Speed of light in km/ms
C_KM_PER_MS = 299.792458


def _propagation_delay_ms(altitude_km: float, slant_factor: float = 1.1) -> float:
    """One-way propagation delay assuming a slant path slightly longer than
    the straight-down altitude. Slant factor 1.1 is a small approximation for
    non-zenith elevation angles within a regional footprint."""
    return slant_factor * altitude_km / C_KM_PER_MS


@dataclass
class Node:
    """A single NTN node.

    Position is stored in a local Cartesian frame (km) anchored on the
    centre of the region of interest. The third coordinate is altitude.
    """

    node_id: int
    layer: NodeLayer
    position: np.ndarray  # shape (3,), km
    cpu_capacity: float  # GHz-equivalent
    mem_capacity: float  # GiB
    energy_capacity_wh: float  # Wh; np.inf for satellite/HAPS to abstract their long endurance
    energy_remaining_wh: float
    base_idle_power_w: float  # Watts when idle (relevant for UAV)
    base_compute_power_w: float  # Watts at full CPU load
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))  # km/s
    cpu_used: float = 0.0
    mem_used: float = 0.0
    active: bool = True
    coverage_radius_km: float = 50.0

    # Bookkeeping for migration / failure handling
    deployed_services: List[int] = field(default_factory=list)

    @property
    def altitude_km(self) -> float:
        return float(self.position[2])

    @property
    def base_delay_ms(self) -> float:
        return _propagation_delay_ms(max(self.altitude_km, 1e-3))

    @property
    def cpu_load(self) -> float:
        return self.cpu_used / self.cpu_capacity if self.cpu_capacity > 0 else 0.0

    @property
    def mem_load(self) -> float:
        return self.mem_used / self.mem_capacity if self.mem_capacity > 0 else 0.0

    @property
    def energy_fraction(self) -> float:
        if not np.isfinite(self.energy_capacity_wh):
            return 1.0
        return max(self.energy_remaining_wh / self.energy_capacity_wh, 0.0)

    def step_position(self, dt_s: float) -> None:
        self.position = self.position + self.velocity * dt_s

    def consume_energy(self, dt_s: float) -> None:
        if not np.isfinite(self.energy_capacity_wh):
            return  # LEO/HAPS abstracted as effectively unlimited at our timescale
        load_factor = min(1.0, self.cpu_load)
        power_w = self.base_idle_power_w + load_factor * self.base_compute_power_w
        consumed_wh = power_w * dt_s / 3600.0
        self.energy_remaining_wh = max(0.0, self.energy_remaining_wh - consumed_wh)
        if self.energy_remaining_wh <= 0.0:
            self.active = False

    def reset_allocation(self) -> None:
        self.cpu_used = 0.0
        self.mem_used = 0.0
        self.deployed_services = []


def build_node_population(
    rng: np.random.Generator,
    n_leo: int = 4,
    n_haps: int = 2,
    n_uav: int = 8,
    region_radius_km: float = 80.0,
) -> List[Node]:
    """Build a population of nodes spanning the three layers.

    Returned positions are in a 3-D Cartesian frame centred on the region.
    LEO satellites are spawned with horizontal velocity vectors so that the
    environment can step them through the field of view. UAVs are scattered
    inside the region with low/zero velocity.
    """

    nodes: List[Node] = []
    next_id = 0

    # ---- LEO satellites --------------------------------------------------
    for _ in range(n_leo):
        altitude = float(rng.uniform(550.0, 1200.0))
        # Random direction across the footprint
        angle = rng.uniform(0.0, 2.0 * np.pi)
        # Spawn beyond the region edge so the satellite "passes over" us
        offset = rng.uniform(150.0, 250.0)
        position = np.array(
            [offset * np.cos(angle), offset * np.sin(angle), altitude]
        )
        # Ground-track speed ~7 km/s, project onto -direction for a flyover
        speed = 7.0
        direction = -position[:2] / np.linalg.norm(position[:2])
        velocity = np.array([direction[0] * speed, direction[1] * speed, 0.0])
        nodes.append(
            Node(
                node_id=next_id,
                layer=NodeLayer.LEO,
                position=position,
                cpu_capacity=24.0,  # cumulative for an on-board edge stack
                mem_capacity=64.0,
                energy_capacity_wh=np.inf,
                energy_remaining_wh=np.inf,
                base_idle_power_w=0.0,
                base_compute_power_w=0.0,
                velocity=velocity,
                coverage_radius_km=600.0,
            )
        )
        next_id += 1

    # ---- HAPS ------------------------------------------------------------
    for _ in range(n_haps):
        altitude = float(rng.uniform(18.0, 25.0))
        offset = rng.uniform(0.0, 25.0)
        angle = rng.uniform(0.0, 2.0 * np.pi)
        position = np.array(
            [offset * np.cos(angle), offset * np.sin(angle), altitude]
        )
        velocity = np.array([rng.normal(0, 0.005), rng.normal(0, 0.005), 0.0])
        nodes.append(
            Node(
                node_id=next_id,
                layer=NodeLayer.HAPS,
                position=position,
                cpu_capacity=16.0,
                mem_capacity=32.0,
                energy_capacity_wh=np.inf,  # solar-powered at our timescale
                energy_remaining_wh=np.inf,
                base_idle_power_w=0.0,
                base_compute_power_w=0.0,
                velocity=velocity,
                coverage_radius_km=200.0,
            )
        )
        next_id += 1

    # ---- UAVs ------------------------------------------------------------
    for _ in range(n_uav):
        altitude = float(rng.uniform(0.1, 1.5))
        position = np.array(
            [
                rng.uniform(-region_radius_km, region_radius_km),
                rng.uniform(-region_radius_km, region_radius_km),
                altitude,
            ]
        )
        velocity = np.array(
            [rng.normal(0, 0.005), rng.normal(0, 0.005), 0.0]
        )  # near-stationary, slow drift
        # Energy budget calibrated so UAVs deplete in tens of minutes under load
        battery_wh = float(rng.uniform(80.0, 140.0))
        nodes.append(
            Node(
                node_id=next_id,
                layer=NodeLayer.UAV,
                position=position,
                cpu_capacity=4.0,
                mem_capacity=8.0,
                energy_capacity_wh=battery_wh,
                energy_remaining_wh=battery_wh,
                base_idle_power_w=120.0,  # hover power for a small rotary platform
                base_compute_power_w=15.0,
                velocity=velocity,
                coverage_radius_km=10.0,
            )
        )
        next_id += 1

    return nodes
