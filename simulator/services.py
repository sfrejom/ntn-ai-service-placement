"""Microservice catalogue used in the experiments.

The catalogue mixes service profiles with very different latency, compute,
state, and bandwidth requirements so that no single layer can host all of
them. This forces the placement policy to actually exploit the heterogeneity
of the NTN.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class Microservice:
    service_id: int
    name: str
    cpu_demand: float       # fraction of CPU capacity (in same GHz units as Node.cpu_capacity)
    mem_demand: float       # GiB
    max_latency_ms: float   # one-way deadline; placement must satisfy base_delay <= max_latency
    bandwidth_mbps: float
    is_stateful: int        # 0 / 1 flag
    state_size_mb: float    # used to estimate migration cost
    max_replicas: int = 2

    @property
    def migration_cost(self) -> float:
        """Approximate time-cost (s) of live-migrating the service when stateful.

        Modelled as state_size / 50 Mbps, the typical inter-node throughput
        observed in the QoS-evaluation paper for FANET links."""
        if not self.is_stateful:
            return 0.0
        return self.state_size_mb / (50.0 / 8.0)  # 50 Mbps -> MB/s


# Pre-defined service catalogue. The mix is intentionally heterogeneous.
# - Latency-critical, stateful: needs low-altitude (UAV), live migration matters
# - Latency-critical, stateless: edge (UAV/HAPS)
# - Bulk / IoT aggregation: latency-tolerant, fits LEO
# - Mid-tier compute: HAPS-friendly
_DEFAULT_CATALOGUE = [
    dict(name="ar-overlay",     cpu_demand=1.2, mem_demand=1.5, max_latency_ms=3.0,  bandwidth_mbps=20.0,  is_stateful=1, state_size_mb=120.0, max_replicas=1),
    dict(name="health-monitor", cpu_demand=0.6, mem_demand=0.8, max_latency_ms=5.0,  bandwidth_mbps=4.0,   is_stateful=1, state_size_mb=60.0,  max_replicas=2),
    dict(name="video-relay",    cpu_demand=2.0, mem_demand=2.0, max_latency_ms=8.0,  bandwidth_mbps=30.0,  is_stateful=0, state_size_mb=0.0,   max_replicas=2),
    dict(name="env-sensing",    cpu_demand=0.4, mem_demand=0.5, max_latency_ms=15.0, bandwidth_mbps=2.0,   is_stateful=0, state_size_mb=0.0,   max_replicas=3),
    dict(name="iot-aggregator", cpu_demand=0.8, mem_demand=1.5, max_latency_ms=80.0, bandwidth_mbps=8.0,   is_stateful=0, state_size_mb=0.0,   max_replicas=2),
    dict(name="map-update",     cpu_demand=1.5, mem_demand=2.5, max_latency_ms=60.0, bandwidth_mbps=15.0,  is_stateful=1, state_size_mb=200.0, max_replicas=1),
    dict(name="bulk-transfer",  cpu_demand=1.0, mem_demand=4.0, max_latency_ms=200.0, bandwidth_mbps=50.0, is_stateful=0, state_size_mb=0.0,   max_replicas=2),
    dict(name="cdn-cache",      cpu_demand=1.5, mem_demand=6.0, max_latency_ms=40.0, bandwidth_mbps=40.0,  is_stateful=1, state_size_mb=400.0, max_replicas=2),
    dict(name="mqtt-broker",    cpu_demand=0.3, mem_demand=0.5, max_latency_ms=20.0, bandwidth_mbps=3.0,   is_stateful=1, state_size_mb=20.0,  max_replicas=2),
    dict(name="auth-service",   cpu_demand=0.5, mem_demand=1.0, max_latency_ms=25.0, bandwidth_mbps=2.0,   is_stateful=1, state_size_mb=15.0,  max_replicas=2),
]


def build_service_catalog(rng: np.random.Generator, size: int = 10) -> List[Microservice]:
    """Sample `size` microservices from the default catalogue (with replacement
    if size > len(catalogue))."""

    base = _DEFAULT_CATALOGUE.copy()
    services: List[Microservice] = []
    for i in range(size):
        spec = base[i % len(base)].copy()
        # Add small jitter so duplicates are not identical
        if i >= len(base):
            spec["cpu_demand"] = max(0.1, spec["cpu_demand"] * float(rng.uniform(0.85, 1.15)))
            spec["mem_demand"] = max(0.1, spec["mem_demand"] * float(rng.uniform(0.85, 1.15)))
        services.append(Microservice(service_id=i, **spec))
    return services
