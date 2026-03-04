from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

try:
    from scipy.stats import beta as _beta  # type: ignore
except Exception:  # pragma: no cover
    _beta = None

from statistics import NormalDist

from .deps import QuantumCircuit, transpile, AerSimulator, NoiseModel, depolarizing_error, ReadoutError

from .utils import run_backend_result

from .contract import NoiseSpec


_DEFAULT_1Q_GATE_NAMES = [
    "id", "x", "y", "z", "h", "s", "sdg", "sx", "sxdg",
    "rx", "ry", "rz",
    "u", "u1", "u2", "u3", "p",
]

_DEFAULT_2Q_GATE_NAMES = [
    "cx", "cz", "swap", "ecr",
]


def _clip_p_ro(p: float) -> float:
    return float(min(0.5, max(0.0, p)))


def _clip_p(p: float) -> float:
    return float(min(1.0, max(0.0, p)))


@dataclass(frozen=True)
class NoiseBox:
    """A rectangular ambiguity set over (p1, p2, p_ro)."""
    p1_low: float
    p1_high: float
    p2_low: float
    p2_high: float
    p_ro_low: float
    p_ro_high: float
    source: str = "box"
    meta: Dict[str, Any] = field(default_factory=dict)

    def corners(self, model: str = "depolarizing_readout") -> List[NoiseSpec]:
        xs = [_clip_p(self.p1_low), _clip_p(self.p1_high)]
        ys = [_clip_p(self.p2_low), _clip_p(self.p2_high)]
        zs = [_clip_p_ro(self.p_ro_low), _clip_p_ro(self.p_ro_high)]
        out: List[NoiseSpec] = []
        for p1 in xs:
            for p2 in ys:
                for p_ro in zs:
                    out.append(
                        NoiseSpec(
                            model=model,
                            p1=float(p1),
                            p2=float(p2),
                            p_ro=float(p_ro),
                            uncertainty_scale=0.0,
                            uncertainty_model="scale",
                            drift_scale=0.0,
                            calibration_shots=0,
                            calibration_delta=0.05,
                        )
                    )
        return out

    def worst_corner(self, model: str = "depolarizing_readout") -> NoiseSpec:
        return NoiseSpec(
            model=model,
            p1=_clip_p(self.p1_high),
            p2=_clip_p(self.p2_high),
            p_ro=_clip_p_ro(self.p_ro_high),
            uncertainty_scale=0.0,
            uncertainty_model="scale",
            drift_scale=0.0,
            calibration_shots=0,
            calibration_delta=0.05,
        )

    def summary(self) -> Dict[str, Any]:
        return {
            "p1": [float(self.p1_low), float(self.p1_high)],
            "p2": [float(self.p2_low), float(self.p2_high)],
            "p_ro": [float(self.p_ro_low), float(self.p_ro_high)],
            "source": self.source,
            "meta": self.meta,
        }


def worst_case_params(noise: NoiseSpec) -> NoiseSpec:
    """Legacy one-sided worst-case inflation within the uncertainty set."""
    s = max(0.0, float(noise.uncertainty_scale))
    return NoiseSpec(
        model=noise.model,
        p1=_clip_p(noise.p1 * (1.0 + s)),
        p2=_clip_p(noise.p2 * (1.0 + s)),
        p_ro=_clip_p_ro(noise.p_ro * (1.0 + s)),
        uncertainty_scale=noise.uncertainty_scale,
        uncertainty_model=noise.uncertainty_model,
        drift_scale=noise.drift_scale,
        calibration_shots=noise.calibration_shots,
        calibration_delta=noise.calibration_delta,
    )


def noise_box_from_spec(noise: NoiseSpec) -> NoiseBox:
    """Construct a parameter box based on (uncertainty_model, uncertainty_scale, drift_scale)."""
    s = max(0.0, float(noise.uncertainty_scale))
    d = max(0.0, float(noise.drift_scale))

    if str(noise.uncertainty_model) == "box":
        p1_low = _clip_p(noise.p1 * (1.0 - s))
        p1_high = _clip_p(noise.p1 * (1.0 + s))
        p2_low = _clip_p(noise.p2 * (1.0 - s))
        p2_high = _clip_p(noise.p2 * (1.0 + s))
        p_ro_low = _clip_p_ro(noise.p_ro * (1.0 - s))
        p_ro_high = _clip_p_ro(noise.p_ro * (1.0 + s))
        box = NoiseBox(p1_low, p1_high, p2_low, p2_high, p_ro_low, p_ro_high, source="declared_box", meta={"uncertainty_scale": s})
    else:
        # "scale" and "calibrated_box" without calibration fall back to one-sided inflation
        wc = worst_case_params(noise)
        box = NoiseBox(
            p1_low=_clip_p(noise.p1),
            p1_high=_clip_p(wc.p1),
            p2_low=_clip_p(noise.p2),
            p2_high=_clip_p(wc.p2),
            p_ro_low=_clip_p_ro(noise.p_ro),
            p_ro_high=_clip_p_ro(wc.p_ro),
            source="one_sided_scale",
            meta={"uncertainty_scale": s},
        )

    if d > 0:
        # Expand for drift (conservative symmetric expansion).
        box = NoiseBox(
            p1_low=_clip_p(box.p1_low * (1.0 - d)),
            p1_high=_clip_p(box.p1_high * (1.0 + d)),
            p2_low=_clip_p(box.p2_low * (1.0 - d)),
            p2_high=_clip_p(box.p2_high * (1.0 + d)),
            p_ro_low=_clip_p_ro(box.p_ro_low * (1.0 - d)),
            p_ro_high=_clip_p_ro(box.p_ro_high * (1.0 + d)),
            source=box.source,
            meta={**box.meta, "drift_scale": d},
        )

    return box


def build_noise_model(noise: NoiseSpec, gate_names_1q=None, gate_names_2q=None) -> NoiseModel:
    """Build an Aer NoiseModel for the prototype."""
    if noise.model != "depolarizing_readout":
        raise ValueError(f"Unsupported noise model: {noise.model}")

    gate_names_1q = list(gate_names_1q or _DEFAULT_1Q_GATE_NAMES)
    gate_names_2q = list(gate_names_2q or _DEFAULT_2Q_GATE_NAMES)

    nm = NoiseModel()

    p1 = _clip_p(float(noise.p1))
    p2 = _clip_p(float(noise.p2))
    p_ro = _clip_p_ro(float(noise.p_ro))

    if p1 > 0:
        err1 = depolarizing_error(p1, 1)
        nm.add_all_qubit_quantum_error(err1, gate_names_1q)

    if p2 > 0:
        err2 = depolarizing_error(p2, 2)
        nm.add_all_qubit_quantum_error(err2, gate_names_2q)

    if p_ro > 0:
        p = p_ro
        ro = ReadoutError([[1 - p, p], [p, 1 - p]])
        nm.add_all_qubit_readout_error(ro)

    return nm


def noise_summary(noise: NoiseSpec) -> Dict[str, Any]:
    return {
        "p1": float(noise.p1),
        "p2": float(noise.p2),
        "p_ro": float(noise.p_ro),
        "uncertainty_scale": float(noise.uncertainty_scale),
        "uncertainty_model": str(noise.uncertainty_model),
        "drift_scale": float(noise.drift_scale),
        "calibration_shots": int(noise.calibration_shots),
        "calibration_delta": float(noise.calibration_delta),
    }


# ---------------------------
# Calibration → box (prototype)
# ---------------------------

def _clopper_pearson_interval(k: int, n: int, alpha: float) -> Tuple[float, float]:
    """Binomial CI for a proportion.

    Prefers the exact Clopper–Pearson interval when SciPy is available.
    Falls back to Wilson score otherwise.
    """
    if n <= 0:
        return (0.0, 1.0)
    k = int(max(0, min(n, k)))
    a = max(1e-12, min(0.999999999, float(alpha)))

    if _beta is not None:
        lo = 0.0 if k == 0 else float(_beta.ppf(a / 2.0, k, n - k + 1))
        hi = 1.0 if k == n else float(_beta.ppf(1.0 - a / 2.0, k + 1, n - k))
        return (float(lo), float(hi))

    # Wilson score interval (fallback)
    phat = float(k) / float(n)
    z = NormalDist().inv_cdf(1.0 - a / 2.0)
    denom = 1.0 + (z * z) / float(n)
    center = (phat + (z * z) / (2.0 * float(n))) / denom
    rad = (z / denom) * math.sqrt((phat * (1.0 - phat) / float(n)) + (z * z) / (4.0 * float(n) * float(n)))
    return (float(max(0.0, center - rad)), float(min(1.0, center + rad)))


def _p1_from_xx_error_rate(e: float) -> float:
    # For 2 noisy 1q gates with depol p1: p_total = 1-(1-p1)^2; e = p_total/2.
    e = float(min(0.5, max(0.0, e)))
    p_total = min(1.0, 2.0 * e)
    return float(1.0 - math.sqrt(max(0.0, 1.0 - p_total)))


def _p2_from_cxcx_error_rate(e: float) -> float:
    # For 2 noisy 2q gates with depol p2: p_total = 1-(1-p2)^2; e = (3/4) p_total.
    e = float(min(0.75, max(0.0, e)))
    p_total = min(1.0, (4.0 / 3.0) * e)
    return float(1.0 - math.sqrt(max(0.0, 1.0 - p_total)))


def calibrate_noise_box(
    backend: Optional[AerSimulator],
    noise_true: NoiseSpec,
    shots_per_experiment: int,
    delta: float,
    seed: int = 1234,
    drift_scale: float = 0.0,
    extra_scale: float = 0.0,
    optimization_level: int = 1,
) -> NoiseBox:
    """Calibrate (p1, p2, p_ro) and return a robust parameter box.

    This is a prototype calibration routine consistent with the simplified noise model:
      - p_ro estimated from |0>,|1> readout flips (with gate noise disabled)
      - p1 estimated from 'X; X' returning to |0> (with readout disabled)
      - p2 estimated from 'CX; CX' returning to |00> (with readout disabled)

    Returns a NoiseBox whose bounds are (1-δ) confidence intervals, optionally expanded by:
      - extra_scale (relative expansion)
      - drift_scale  (relative expansion)
    """
    backend = backend or AerSimulator(method="density_matrix")
    shots = int(shots_per_experiment)
    if shots <= 0:
        raise ValueError("shots_per_experiment must be > 0")
    if not (0.0 < float(delta) < 1.0):
        raise ValueError("delta must be in (0,1)")

    # Split calibration risk across 3 parameters (conservative union bound)
    alpha = float(delta) / 3.0

    # --- Readout p_ro calibration (gate noise disabled) ---
    nm_ro = build_noise_model(
        NoiseSpec(model=noise_true.model, p1=0.0, p2=0.0, p_ro=noise_true.p_ro)
    )

    # Prepare |0> and measure
    qc0 = QuantumCircuit(1)
    qc0.measure_all()
    tqc0 = transpile(qc0, backend=backend, optimization_level=optimization_level)
    res0 = run_backend_result(backend, tqc0, shots=shots, noise_model=nm_ro, seed=seed)
    c0 = dict(res0.get_counts(0))
    err0 = int(c0.get("1", 0))

    # Prepare |1> and measure
    qc1 = QuantumCircuit(1)
    qc1.x(0)
    qc1.measure_all()
    tqc1 = transpile(qc1, backend=backend, optimization_level=optimization_level)
    res1 = run_backend_result(backend, tqc1, shots=shots, noise_model=nm_ro, seed=seed + 1)
    c1 = dict(res1.get_counts(0))
    err1 = int(c1.get("0", 0))

    k_ro = int(err0 + err1)
    n_ro = int(2 * shots)
    p_ro_lo, p_ro_hi = _clopper_pearson_interval(k_ro, n_ro, alpha)

    # --- p1 calibration (readout disabled) ---
    nm_p1 = build_noise_model(
        NoiseSpec(model=noise_true.model, p1=noise_true.p1, p2=0.0, p_ro=0.0)
    )
    qc_xx = QuantumCircuit(1)
    qc_xx.x(0)
    qc_xx.x(0)
    qc_xx.measure_all()
    tqc_xx = transpile(qc_xx, backend=backend, optimization_level=optimization_level)
    res_xx = run_backend_result(backend, tqc_xx, shots=shots, noise_model=nm_p1, seed=seed + 2)
    c_xx = dict(res_xx.get_counts(0))
    k_xx = int(c_xx.get("1", 0))
    e_xx_hat = float(k_xx) / float(shots)
    e_xx_lo, e_xx_hi = _clopper_pearson_interval(k_xx, shots, alpha)
    p1_lo = _p1_from_xx_error_rate(e_xx_lo)
    p1_hi = _p1_from_xx_error_rate(e_xx_hi)

    # --- p2 calibration (readout disabled) ---
    nm_p2 = build_noise_model(
        NoiseSpec(model=noise_true.model, p1=0.0, p2=noise_true.p2, p_ro=0.0)
    )
    qc_cxcx = QuantumCircuit(2)
    qc_cxcx.cx(0, 1)
    qc_cxcx.cx(0, 1)
    qc_cxcx.measure_all()
    tqc_cxcx = transpile(qc_cxcx, backend=backend, optimization_level=optimization_level)
    res_cxcx = run_backend_result(backend, tqc_cxcx, shots=shots, noise_model=nm_p2, seed=seed + 3)
    c_cxcx = dict(res_cxcx.get_counts(0))
    k_bad = int(shots - int(c_cxcx.get("00", 0)))
    e2_hat = float(k_bad) / float(shots)
    e2_lo, e2_hi = _clopper_pearson_interval(k_bad, shots, alpha)
    p2_lo = _p2_from_cxcx_error_rate(e2_lo)
    p2_hi = _p2_from_cxcx_error_rate(e2_hi)

    # Optional extra scale (e.g., prior cushion) and drift expansion.
    s = max(0.0, float(extra_scale))
    d = max(0.0, float(drift_scale))

    def _expand(lo: float, hi: float, cap: float = 1.0) -> Tuple[float, float]:
        lo2 = max(0.0, lo * (1.0 - s))
        hi2 = min(cap, hi * (1.0 + s))
        lo3 = max(0.0, lo2 * (1.0 - d))
        hi3 = min(cap, hi2 * (1.0 + d))
        return float(lo3), float(hi3)

    p1_lo2, p1_hi2 = _expand(float(min(p1_lo, p1_hi)), float(max(p1_lo, p1_hi)), cap=1.0)
    p2_lo2, p2_hi2 = _expand(float(min(p2_lo, p2_hi)), float(max(p2_lo, p2_hi)), cap=1.0)
    p_ro_lo2, p_ro_hi2 = _expand(float(min(p_ro_lo, p_ro_hi)), float(max(p_ro_lo, p_ro_hi)), cap=0.5)

    meta = {
        "shots_per_experiment": shots,
        "delta": float(delta),
        "alpha_per_param": float(alpha),
        "readout": {"k_errors": k_ro, "n": n_ro, "p_hat": float(k_ro) / float(n_ro)},
        "p1": {"k_errors": k_xx, "n": shots, "e_hat": e_xx_hat},
        "p2": {"k_errors": k_bad, "n": shots, "e_hat": e2_hat},
        "extra_scale": s,
        "drift_scale": d,
    }

    return NoiseBox(
        p1_low=_clip_p(p1_lo2),
        p1_high=_clip_p(p1_hi2),
        p2_low=_clip_p(p2_lo2),
        p2_high=_clip_p(p2_hi2),
        p_ro_low=_clip_p_ro(p_ro_lo2),
        p_ro_high=_clip_p_ro(p_ro_hi2),
        source="calibrated_box",
        meta=meta,
    )