from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type

from .deps import QuantumCircuit


@dataclass(frozen=True)
class QualityResult:
    quality: float
    details: Dict[str, Any]


class BaseTask:
    """Interface expected by sat_checker.py and __main__.py."""
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = dict(params or {})

    def build_circuit(self) -> QuantumCircuit:
        raise NotImplementedError

    def quality_from_counts(self, counts: Dict[str, int], shots: int) -> QualityResult:
        raise NotImplementedError

    def quality_range(self) -> Tuple[float, float]:
        return (0.0, 1.0)

    def reward(self, bitstring: str) -> float:
        """Per-shot reward in [0,1] used for moment-based risk bounds.

        For Bernoulli quality metrics, reward ∈ {0,1}.
        For non-Bernoulli metrics, reward can be any value in [0,1] consistent with quality().
        """
        raise NotImplementedError("Task must implement reward(bitstring).")


class GHZSuccessTask(BaseTask):
    """Quality = P(00..0 or 11..1) for an n-qubit GHZ state."""
    def build_circuit(self) -> QuantumCircuit:
        n = int(self.params.get("n_qubits", 3))
        if n < 2:
            raise ValueError("GHZ requires n_qubits >= 2")
        qc = QuantumCircuit(n)
        qc.h(0)
        for i in range(n - 1):
            qc.cx(i, i + 1)
        qc.measure_all()
        return qc

    def quality_from_counts(self, counts: Dict[str, int], shots: int) -> QualityResult:
        n = int(self.params.get("n_qubits", 3))
        good0 = "0" * n
        good1 = "1" * n
        good = int(counts.get(good0, 0)) + int(counts.get(good1, 0))
        q = float(good) / float(shots) if shots > 0 else 0.0
        return QualityResult(quality=q, details={"good_strings": [good0, good1], "good_counts": good})

    def reward(self, bitstring: str) -> float:
        n = int(self.params.get("n_qubits", 3))
        good0 = "0" * n
        good1 = "1" * n
        return 1.0 if (bitstring == good0 or bitstring == good1) else 0.0


def _cut_value(bitstring: str, edges: List[Tuple[int, int]]) -> int:
    # bitstring is MSB..LSB; qubit i corresponds to bitstring[-1-i].
    val = 0
    for u, v in edges:
        bu = bitstring[-1 - u]
        bv = bitstring[-1 - v]
        if bu != bv:
            val += 1
    return val


class QAOAMaxCutTask(BaseTask):
    """Toy QAOA MaxCut: quality = expected cut value / |E|, in [0,1]."""
    def build_circuit(self) -> QuantumCircuit:
        n = int(self.params.get("n_nodes", 4))
        p = int(self.params.get("p", 1))
        gamma = float(self.params.get("gamma", 0.8))
        beta = float(self.params.get("beta", 0.6))

        edges = self._edges(n)

        qc = QuantumCircuit(n)
        qc.h(list(range(n)))

        for _layer in range(p):
            for (u, v) in edges:
                qc.cx(u, v)
                qc.rz(2.0 * gamma, v)
                qc.cx(u, v)
            for i in range(n):
                qc.rx(2.0 * beta, i)

        qc.measure_all()
        return qc

    def _edges(self, n: int) -> List[Tuple[int, int]]:
        edges_in = self.params.get("edges", None)
        if edges_in is None:
            return [(i, (i + 1) % n) for i in range(n)]
        edges: List[Tuple[int, int]] = []
        for e in edges_in:
            u, v = int(e[0]), int(e[1])
            if not (0 <= u < n and 0 <= v < n and u != v):
                raise ValueError("Invalid edge in edges")
            edges.append((u, v))
        return edges

    def quality_from_counts(self, counts: Dict[str, int], shots: int) -> QualityResult:
        n = int(self.params.get("n_nodes", 4))
        edges = self._edges(n)
        m = len(edges)
        if shots <= 0 or m <= 0:
            return QualityResult(quality=0.0, details={"edges": m})

        exp_cut = 0.0
        for bit, c in counts.items():
            exp_cut += float(c) * float(_cut_value(bit, edges))

        exp_cut /= float(shots)
        q = exp_cut / float(m)
        return QualityResult(quality=float(q), details={"expected_cut": exp_cut, "edges": m})

    def reward(self, bitstring: str) -> float:
        n = int(self.params.get("n_nodes", 4))
        edges = self._edges(n)
        m = len(edges)
        if m <= 0:
            return 0.0
        return float(_cut_value(bitstring, edges)) / float(m)


TASK_REGISTRY: Dict[str, Type[BaseTask]] = {
    "ghz_success": GHZSuccessTask,
    "qaoa_maxcut": QAOAMaxCutTask,
}