from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


def normalize_masked_policy(policy, legal_mask) -> np.ndarray:
    probs = np.asarray(policy, dtype=np.float32).reshape(-1)
    mask = np.where(np.asarray(legal_mask, dtype=np.float32).reshape(-1) > 0.5, 1.0, 0.0).astype(np.float32)
    probs = probs * mask
    total = float(probs.sum())
    if total <= 1e-9:
        return uniform_legal_policy(mask)
    return (probs / total).astype(np.float32)


def uniform_legal_policy(legal_mask) -> np.ndarray:
    mask = np.where(np.asarray(legal_mask, dtype=np.float32).reshape(-1) > 0.5, 1.0, 0.0).astype(np.float32)
    total = float(mask.sum())
    if total <= 1e-9:
        mask = np.ones_like(mask, dtype=np.float32)
        total = float(mask.sum())
    return (mask / total).astype(np.float32)


def regret_matching(regret_sum, legal_mask) -> np.ndarray:
    regrets = np.asarray(regret_sum, dtype=np.float32).reshape(-1)
    positive = np.maximum(regrets, 0.0)
    return normalize_masked_policy(positive, legal_mask)


def average_policy(strategy_sum, legal_mask, regret_sum=None) -> np.ndarray:
    averaged = normalize_masked_policy(strategy_sum, legal_mask)
    if float(averaged.sum()) > 1e-9 and np.any(np.asarray(strategy_sum, dtype=np.float32) > 1e-9):
        return averaged
    if regret_sum is not None:
        return regret_matching(regret_sum, legal_mask)
    return uniform_legal_policy(legal_mask)


@dataclass
class TabularNode:
    legal_mask: np.ndarray
    regret_sum: np.ndarray
    strategy_sum: np.ndarray
    visits: int = 0

    @classmethod
    def new(cls, legal_mask) -> "TabularNode":
        mask = np.asarray(legal_mask, dtype=np.float32).reshape(-1).copy()
        size = int(mask.shape[0])
        return cls(
            legal_mask=mask,
            regret_sum=np.zeros(size, dtype=np.float32),
            strategy_sum=np.zeros(size, dtype=np.float32),
            visits=0,
        )

    def merge_legal_mask(self, legal_mask) -> None:
        merged = np.maximum(self.legal_mask, np.asarray(legal_mask, dtype=np.float32).reshape(-1))
        self.legal_mask = merged.astype(np.float32)

    def to_payload(self) -> Dict[str, object]:
        return {
            "legal_mask": np.asarray(self.legal_mask, dtype=np.float32).copy(),
            "regret_sum": np.asarray(self.regret_sum, dtype=np.float32).copy(),
            "strategy_sum": np.asarray(self.strategy_sum, dtype=np.float32).copy(),
            "visits": int(self.visits),
        }

    @classmethod
    def from_payload(cls, payload: Dict[str, object]) -> "TabularNode":
        return cls(
            legal_mask=np.asarray(payload["legal_mask"], dtype=np.float32).copy(),
            regret_sum=np.asarray(payload["regret_sum"], dtype=np.float32).copy(),
            strategy_sum=np.asarray(payload["strategy_sum"], dtype=np.float32).copy(),
            visits=int(payload.get("visits", 0)),
        )


@dataclass
class TabularPolicyEntry:
    legal_mask: np.ndarray
    average_policy: np.ndarray
    current_policy: np.ndarray
    visits: int = 0

    def to_payload(self) -> Dict[str, object]:
        return {
            "legal_mask": np.asarray(self.legal_mask, dtype=np.float32).copy(),
            "average_policy": np.asarray(self.average_policy, dtype=np.float32).copy(),
            "current_policy": np.asarray(self.current_policy, dtype=np.float32).copy(),
            "visits": int(self.visits),
        }

    @classmethod
    def from_payload(cls, payload: Dict[str, object]) -> "TabularPolicyEntry":
        return cls(
            legal_mask=np.asarray(payload["legal_mask"], dtype=np.float32).copy(),
            average_policy=np.asarray(payload["average_policy"], dtype=np.float32).copy(),
            current_policy=np.asarray(payload["current_policy"], dtype=np.float32).copy(),
            visits=int(payload.get("visits", 0)),
        )


@dataclass
class TabularPolicySnapshot:
    policy_table: Dict[str, TabularPolicyEntry]
    metadata: Dict[str, object]

    def to_payload(self) -> Dict[str, object]:
        return {
            "policy_table": {key: entry.to_payload() for key, entry in self.policy_table.items()},
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_payload(cls, payload: Dict[str, object]) -> "TabularPolicySnapshot":
        table = {
            str(key): TabularPolicyEntry.from_payload(entry)
            for key, entry in dict(payload.get("policy_table", {})).items()
        }
        return cls(policy_table=table, metadata=dict(payload.get("metadata", {})))


def freeze_policy_snapshot(node_store: Dict[str, TabularNode], metadata: Dict[str, object] | None = None) -> TabularPolicySnapshot:
    frozen: Dict[str, TabularPolicyEntry] = {}
    for key, node in node_store.items():
        legal_mask = np.asarray(node.legal_mask, dtype=np.float32)
        frozen[key] = TabularPolicyEntry(
            legal_mask=legal_mask.copy(),
            average_policy=average_policy(node.strategy_sum, legal_mask, regret_sum=node.regret_sum),
            current_policy=regret_matching(node.regret_sum, legal_mask),
            visits=int(node.visits),
        )
    return TabularPolicySnapshot(policy_table=frozen, metadata=dict(metadata or {}))


def serialize_node_store(node_store: Dict[str, TabularNode]) -> Dict[str, Dict[str, object]]:
    return {key: node.to_payload() for key, node in node_store.items()}


def deserialize_node_store(payload: Dict[str, Dict[str, object]]) -> Dict[str, TabularNode]:
    return {str(key): TabularNode.from_payload(value) for key, value in dict(payload or {}).items()}
