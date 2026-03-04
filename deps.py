from __future__ import annotations

"""Centralized optional dependencies (Qiskit / Aer).

This package is intentionally lightweight, but several modules depend on Qiskit
and qiskit-aer. To avoid repeating import boilerplate (and to keep error
messages consistent), we centralize the imports here.
"""

# NOTE: Do not add heavyweight optional deps here unless needed.

try:
    from qiskit import QuantumCircuit, transpile
except Exception as _e:  # pragma: no cover
    raise ImportError(
        "qiskit is required to run this package. Install it with: pip install qiskit"
    ) from _e

try:
    from qiskit_aer import AerSimulator
except Exception as _e:  # pragma: no cover
    try:
        # Older monolithic installations
        from qiskit.providers.aer import AerSimulator  # type: ignore
    except Exception as _e2:  # pragma: no cover
        raise ImportError(
            "qiskit-aer is required to run this package. Install it with: pip install qiskit-aer"
        ) from _e2

# Noise primitives (used by qqos.noise)
try:
    from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError
except Exception as _e:  # pragma: no cover
    try:
        # Older monolithic installations
        from qiskit.providers.aer.noise import NoiseModel, depolarizing_error, ReadoutError  # type: ignore
    except Exception as _e2:  # pragma: no cover
        raise ImportError(
            "qiskit-aer is required for noise modeling. Install it with: pip install qiskit-aer"
        ) from _e2
