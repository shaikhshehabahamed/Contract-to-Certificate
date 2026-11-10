from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .deps import AerSimulator, QuantumCircuit, transpile
from .readout import ReadoutMitigationModel, calibrate_readout_mitigator
from .utils import run_backend_result


@dataclass
class RunOutcome:
    counts_raw: Dict[str, int]
    counts_mitigated: Optional[Dict[str, float]]
    depth: int
    size: int
    metadata: Dict[str, Any]


def build_simulator(method: str = "automatic") -> AerSimulator:
    return AerSimulator(method=method)


def run_with_optional_mitigation(
    circuit: QuantumCircuit,
    shots: int,
    noise_model=None,
    seed: Optional[int] = 1234,
    optimization_level: int = 1,
    simulator_method: str = "automatic",
    readout_mitigation: bool = False,
    mitigation_shots_per_state: int = 256,
    max_mitigated_qubits: int = 4,
    *,
    is_transpiled: bool = False,
) -> RunOutcome:
    """Run a circuit on an AerSimulator, optionally applying readout mitigation.

    Parameters
    ----------
    is_transpiled:
        If True, `circuit` is assumed to already be transpiled for the chosen backend.
        This avoids double-transpilation in planning loops.
    """
    backend = build_simulator(method=simulator_method)

    if is_transpiled:
        tqc = circuit
    else:
        tqc = transpile(circuit, backend=backend, optimization_level=optimization_level)

    n_qubits = tqc.num_qubits
    mitigator: Optional[ReadoutMitigationModel] = None

    counts_mitigated: Optional[Dict[str, float]] = None
    mitigation_meta: Dict[str, Any] = {"enabled": False}

    if readout_mitigation and n_qubits <= max_mitigated_qubits:
        mitigator = calibrate_readout_mitigator(
            backend=backend,
            n_qubits=n_qubits,
            shots_per_state=mitigation_shots_per_state,
            noise_model=noise_model,
            seed=seed,
            optimization_level=optimization_level,
        )
        mitigation_meta = {
            "enabled": True,
            "shots_per_state": int(mitigation_shots_per_state),
            "n_qubits": int(n_qubits),
            "calibration_circuits": int(2**n_qubits),
        }

    res = run_backend_result(backend, tqc, shots=int(shots), noise_model=noise_model, seed=seed)
    counts_raw = res.get_counts(0)

    if mitigator is not None:
        counts_mitigated = mitigator.mitigate_counts(dict(counts_raw))

    meta = {
        "optimization_level": int(optimization_level),
        "simulator_method": str(simulator_method),
        "readout_mitigation": mitigation_meta,
        "is_transpiled_input": bool(is_transpiled),
    }

    return RunOutcome(
        counts_raw=dict(counts_raw),
        counts_mitigated=counts_mitigated,
        depth=int(tqc.depth()),
        size=int(tqc.size()),
        metadata=meta,
    )
