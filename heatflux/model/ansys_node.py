from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class AnsysNode:
    node_id: int
    x: float
    y: float
    z: float

