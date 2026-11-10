from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Union

from .contract import (
    QQoSContract,
    TaskSpec,
    Budget,
    RiskPolicy,
    NoiseSpec,
    CONTRACT_TYPE_CHOICES,
    RISK_METHOD_CHOICES,
    PILOT_CORRECTION_CHOICES,
    CHECKPOINT_SCHEDULE_CHOICES,
    DRIFT_DETECTOR_CHOICES,
    NOISE_MODEL_CHOICES,
    NOISE_UNCERTAINTY_MODEL_CHOICES,
)


def _require(d: Dict[str, Any], key: str) -> Any:
    if key not in d:
        raise ValueError(f"Missing required field: {key}")
    return d[key]


def load_contract(path: Union[str, Path]) -> QQoSContract:
    """Load a QQoS contract from a JSON file."""
    path = Path(path)
    obj = json.loads(path.read_text(encoding="utf-8"))

    task_obj = _require(obj, "task")
    task = TaskSpec(
        name=_require(task_obj, "name"),
        params=dict(task_obj.get("params", {})),
    )

    budget_obj = _require(obj, "budget")
    rmax = budget_obj.get("runtime_max_s", None)
    runtime_max_s = None if rmax is None else float(rmax)
    budget = Budget(
        shots_max=int(_require(budget_obj, "shots_max")),
        depth_max=int(_require(budget_obj, "depth_max")),
        runtime_max_s=runtime_max_s,
    )

    risk_obj = obj.get("risk_policy", {})
    risk = RiskPolicy(
        method=risk_obj.get("method", "hoeffding"),
        pilot_shots=int(risk_obj.get("pilot_shots", 512)),
        pilot_delta=float(risk_obj.get("pilot_delta", 0.05)),
        safety_floor_gap=float(risk_obj.get("safety_floor_gap", 1e-6)),
        pilot_correction=str(risk_obj.get("pilot_correction", "split")),
        pilot_selection_fraction=float(risk_obj.get("pilot_selection_fraction", 0.5)),
        drift_window=int(risk_obj.get("drift_window", 0)),
        drift_delta_fraction=float(risk_obj.get("drift_delta_fraction", 0.5)),
        checkpoint_schedule=str(risk_obj.get("checkpoint_schedule", "fixed")),
        checkpoint_batch=int(risk_obj.get("checkpoint_batch", 0)),
        checkpoint_ratio=float(risk_obj.get("checkpoint_ratio", 1.5)),
        # New: change-point detection + restarts
        drift_detect=bool(risk_obj.get("drift_detect", False)),
        drift_detector=str(risk_obj.get("drift_detector", "page_hinkley")),
        ph_delta=float(risk_obj.get("ph_delta", 0.005)),
        ph_lambda=float(risk_obj.get("ph_lambda", 0.05)),
        ph_min_instances=int(risk_obj.get("ph_min_instances", 5)),
        min_segment_shots=int(risk_obj.get("min_segment_shots", 0)),
        max_restarts=int(risk_obj.get("max_restarts", 5)),
        restart_gamma=float(risk_obj.get("restart_gamma", 0.2)),
    )

    noise_obj = obj.get("noise", {})
    noise = NoiseSpec(
        model=noise_obj.get("model", "depolarizing_readout"),
        p1=float(noise_obj.get("p1", 0.001)),
        p2=float(noise_obj.get("p2", 0.01)),
        p_ro=float(noise_obj.get("p_ro", 0.02)),
        uncertainty_scale=float(noise_obj.get("uncertainty_scale", 0.0)),
        uncertainty_model=noise_obj.get("uncertainty_model", "scale"),
        drift_scale=float(noise_obj.get("drift_scale", 0.0)),
        calibration_shots=int(noise_obj.get("calibration_shots", 0)),
        calibration_delta=float(noise_obj.get("calibration_delta", 0.05)),
    )

    contract = QQoSContract(
        task=task,
        qmin=obj.get("qmin", None),
        eps_max=obj.get("eps_max", None),
        delta=float(obj.get("delta", 0.05)),
        budget=budget,
        contract_type=obj.get("contract_type", "hard"),
        risk_policy=risk,
        noise=noise,
        fallback=bool(obj.get("fallback", True)),
    )

    return validate_contract(contract)


def validate_contract(contract: QQoSContract) -> QQoSContract:
    """Validate a contract and return it unchanged.

    The simulator clips some probabilities internally as a numerical safeguard,
    but public contract inputs should fail fast when they are outside the
    declared model domain. That keeps reported experiments reproducible.
    """
    if contract.contract_type not in CONTRACT_TYPE_CHOICES:
        raise ValueError(f"contract_type must be one of {sorted(CONTRACT_TYPE_CHOICES)}")

    if str(contract.risk_policy.method) not in RISK_METHOD_CHOICES:
        raise ValueError(f"risk_policy.method must be one of {sorted(RISK_METHOD_CHOICES)}")

    if contract.contract_type in ("hard", "soft"):
        if str(contract.risk_policy.method) not in {"hoeffding", "empirical_bernstein"}:
            raise ValueError(
                f"risk_policy.method='{contract.risk_policy.method}' is only valid for "
                f"contract_type='anytime'; hard/soft contracts only support 'hoeffding' or 'empirical_bernstein'"
            )

    if str(contract.risk_policy.pilot_correction) not in PILOT_CORRECTION_CHOICES:
        raise ValueError(f"risk_policy.pilot_correction must be one of {sorted(PILOT_CORRECTION_CHOICES)}")

    if str(contract.risk_policy.checkpoint_schedule) not in CHECKPOINT_SCHEDULE_CHOICES:
        raise ValueError(f"risk_policy.checkpoint_schedule must be one of {sorted(CHECKPOINT_SCHEDULE_CHOICES)}")

    if str(contract.risk_policy.drift_detector) not in DRIFT_DETECTOR_CHOICES:
        raise ValueError(f"risk_policy.drift_detector must be one of {sorted(DRIFT_DETECTOR_CHOICES)}")

    if str(contract.noise.model) not in NOISE_MODEL_CHOICES:
        raise ValueError(f"noise.model must be one of {sorted(NOISE_MODEL_CHOICES)}")

    if str(contract.noise.uncertainty_model) not in NOISE_UNCERTAINTY_MODEL_CHOICES:
        raise ValueError(f"noise.uncertainty_model must be one of {sorted(NOISE_UNCERTAINTY_MODEL_CHOICES)}")

    if contract.qmin is None and contract.eps_max is None:
        raise ValueError("Contract must define either qmin or eps_max.")

    if contract.qmin is not None and not (0.0 <= float(contract.qmin) <= 1.0):
        raise ValueError("qmin must be in [0,1]")

    if contract.eps_max is not None and not (0.0 <= float(contract.eps_max) <= 1.0):
        raise ValueError("eps_max must be in [0,1]")

    if contract.qmin is not None and contract.eps_max is not None:
        implied_qmin = 1.0 - float(contract.eps_max)
        if not math.isclose(float(contract.qmin), implied_qmin, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("qmin and eps_max disagree; expected qmin == 1 - eps_max")

    q = contract.resolved_qmin()
    if not (0.0 <= q <= 1.0):
        raise ValueError("resolved qmin must be in [0,1]")

    if not (0.0 < contract.delta < 1.0):
        raise ValueError("delta must be in (0,1)")

    if contract.budget.shots_max <= 0 or contract.budget.depth_max <= 0:
        raise ValueError("shots_max and depth_max must be positive")

    if contract.risk_policy.pilot_shots < 0:
        raise ValueError("risk_policy.pilot_shots must be >= 0")

    if not (0.0 <= float(contract.risk_policy.pilot_delta) < 1.0):
        raise ValueError("risk_policy.pilot_delta must be in [0,1)")

    if float(contract.risk_policy.safety_floor_gap) < 0.0:
        raise ValueError("risk_policy.safety_floor_gap must be >= 0")

    if contract.risk_policy.drift_window < 0:
        raise ValueError("risk_policy.drift_window must be >= 0")

    if contract.risk_policy.drift_window > 0 and not (0.0 < contract.risk_policy.drift_delta_fraction < 1.0):
        raise ValueError("risk_policy.drift_delta_fraction must be in (0,1) when drift_window > 0")

    if str(contract.risk_policy.pilot_correction) == "split":
        if not (0.0 < float(contract.risk_policy.pilot_selection_fraction) < 1.0):
            raise ValueError("risk_policy.pilot_selection_fraction must be in (0,1) when pilot_correction='split'")

    if contract.risk_policy.checkpoint_batch < 0:
        raise ValueError("risk_policy.checkpoint_batch must be >= 0")

    if float(contract.risk_policy.checkpoint_ratio) <= 1.0:
        raise ValueError("risk_policy.checkpoint_ratio must be > 1")

    if not (0.0 <= float(contract.noise.p1) <= 1.0):
        raise ValueError("noise.p1 must be in [0,1]")

    if not (0.0 <= float(contract.noise.p2) <= 1.0):
        raise ValueError("noise.p2 must be in [0,1]")

    if not (0.0 <= float(contract.noise.p_ro) <= 0.5):
        raise ValueError("noise.p_ro must be in [0,0.5]")

    if float(contract.noise.uncertainty_scale) < 0.0:
        raise ValueError("noise.uncertainty_scale must be >= 0")

    if float(contract.noise.drift_scale) < 0.0:
        raise ValueError("noise.drift_scale must be >= 0")

    if contract.noise.calibration_shots < 0:
        raise ValueError("noise.calibration_shots must be >= 0")

    if str(contract.noise.uncertainty_model) == "calibrated_box" and contract.noise.calibration_shots <= 0:
        raise ValueError("noise.calibration_shots must be > 0 when uncertainty_model='calibrated_box'")

    if contract.noise.calibration_shots > 0 and not (0.0 < float(contract.noise.calibration_delta) < 1.0):
        raise ValueError("noise.calibration_delta must be in (0,1) when calibration_shots > 0")

    if contract.risk_policy.max_restarts < 0:
        raise ValueError("risk_policy.max_restarts must be >= 0")

    if not (0.0 < float(contract.risk_policy.restart_gamma) < 1.0):
        raise ValueError("risk_policy.restart_gamma must be in (0,1)")

    if contract.risk_policy.min_segment_shots < 0:
        raise ValueError("risk_policy.min_segment_shots must be >= 0")

    if float(contract.risk_policy.ph_delta) < 0.0:
        raise ValueError("risk_policy.ph_delta must be >= 0")

    if float(contract.risk_policy.ph_lambda) <= 0.0:
        raise ValueError("risk_policy.ph_lambda must be > 0")

    if contract.risk_policy.ph_min_instances <= 0:
        raise ValueError("risk_policy.ph_min_instances must be > 0")

    return contract
