from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .deps import AerSimulator, QuantumCircuit, transpile
from .utils import run_backend_result


def _basis_strings(n_qubits: int) -> List[str]:
    # Qiskit counts are strings like '0101' with leftmost = highest classical bit index.
    return [format(i, f"0{n_qubits}b") for i in range(2 ** n_qubits)]


@dataclass
class ReadoutMitigationModel:
    """Simple correlated readout mitigation using a full assignment matrix.

    For n_qubits <= 4 this is feasible: 2^n calibration circuits.

    The pseudo-inverse of the assignment matrix is cached to avoid recomputing it
    on every mitigation call.
    """

    n_qubits: int
    assignment: np.ndarray  # shape (2^n, 2^n), A[i,j]=P(meas=i | prep=j)
    basis: List[str]
    assignment_pinv: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # Precompute a stable pseudo-inverse once.
        self.assignment_pinv = np.linalg.pinv(self.assignment)

    def mitigate_counts(self, counts: Dict[str, int]) -> Dict[str, float]:
        shots = sum(int(v) for v in counts.values())
        if shots <= 0:
            return {b: 0.0 for b in self.basis}

        y = np.zeros((2 ** self.n_qubits,), dtype=float)
        for i, b in enumerate(self.basis):
            y[i] = float(counts.get(b, 0)) / float(shots)

        # Solve x ≈ A^{-1} y (use pseudo-inverse for stability)
        x = self.assignment_pinv @ y

        # Clamp negatives and renormalize
        x = np.maximum(x, 0.0)
        s = float(x.sum())
        if s > 0:
            x = x / s

        return {b: float(x[i]) for i, b in enumerate(self.basis)}


def calibrate_readout_mitigator(
    backend: AerSimulator,
    n_qubits: int,
    shots_per_state: int = 256,
    noise_model=None,
    seed: Optional[int] = 1234,
    optimization_level: int = 1,
    max_qubits: int = 4,
) -> ReadoutMitigationModel:
    """Build assignment matrix A by preparing each computational basis state and measuring.

    By default this is limited to `max_qubits=4` because it requires 2^n calibration circuits.
    """
    if n_qubits > int(max_qubits):
        raise ValueError(f"Readout mitigation is too expensive for n_qubits > {int(max_qubits)}")
    sps = int(shots_per_state)
    if sps <= 0:
        raise ValueError("shots_per_state must be a positive integer")

    basis = _basis_strings(n_qubits)
    dim = 2 ** n_qubits
    A = np.zeros((dim, dim), dtype=float)

    for j, prep in enumerate(basis):
        qc = QuantumCircuit(n_qubits)

        # prep string is MSB..LSB; qubit i corresponds to the i-th bit from the right.
        for q in range(n_qubits):
            if prep[-1 - q] == "1":
                qc.x(q)

        qc.measure_all()

        tqc = transpile(qc, backend=backend, optimization_level=optimization_level)
        res = run_backend_result(
            backend,
            tqc,
            shots=sps,
            noise_model=noise_model,
            seed=(None if seed is None else int(seed) + int(j)),
        )
        counts = res.get_counts(0)

        for i, meas in enumerate(basis):
            A[i, j] = float(counts.get(meas, 0)) / float(sps)

    return ReadoutMitigationModel(n_qubits=n_qubits, assignment=A, basis=basis)
