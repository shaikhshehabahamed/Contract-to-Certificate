from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from .risk import RiskBound


@dataclass
class ExecutionPlan:
    """A risk-bounded execution plan (circuit + settings + shot allocation)."""

    task_name: str
    task_params: Dict[str, Any]

    optimization_level: int
    simulator_method: str
    readout_mitigation: bool
    mitigation_shots_per_state: int

    # For hard/soft contracts: fixed production shots. For anytime: batch size.
    shots: int

    # Readout calibration shots (mitigation) -- one-time overhead at execution.
    calibration_shots: int

    # Planning-time probe budget (pilot). Interpreted as *total* pilot shots used by the planner.
    pilot_shots: int
    pilot_quality: float
    mu_lower_bound: float

    # Contract metadata
    qmin: float
    delta: float
    delta_pilot: float
    delta_production: float

    depth: int
    size: int

    risk_bound: RiskBound

    # ---- Anytime support ----
    contract_type: str = "hard"  # "hard" | "soft" | "anytime"
    shots_max: int = 0  # only meaningful for anytime (production shot cap)
    stopping_rule: Dict[str, Any] = field(default_factory=dict)  # serialized stopping rule

    # Noise calibration shots (planning-time overhead for calibrated_box, or reserved budget).
    noise_calibration_shots: int = 0

    def __post_init__(self) -> None:
        # Backwards compatibility if older serialized plans set this to null.
        if self.noise_calibration_shots is None:
            self.noise_calibration_shots = 0
        if self.stopping_rule is None:
            self.stopping_rule = {}

    def to_json_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["risk_bound"] = asdict(self.risk_bound)
        return d

    @staticmethod
    def from_json_dict(d: Dict[str, Any]) -> "ExecutionPlan":
        """Load an ExecutionPlan previously produced by to_json_dict()."""
        obj = dict(d or {})

        rb = obj.get("risk_bound") or {}
        if isinstance(rb, RiskBound):
            rb_obj = rb
        else:
            rb_obj = RiskBound(
                method=str(rb.get("method", "")),
                delta=float(rb.get("delta", 0.0)),
                guarantee=str(rb.get("guarantee", "")),
                details=dict(rb.get("details") or {}),
            )
        obj["risk_bound"] = rb_obj

        # Defaults for newer fields
        obj.setdefault("contract_type", "hard")
        obj.setdefault("shots_max", 0)
        obj.setdefault("stopping_rule", {})
        obj.setdefault("noise_calibration_shots", 0)

        return ExecutionPlan(**obj)


@dataclass
class UNSATCertificate:
    reasons: List[str]
    minimal_relaxations: Dict[str, Any]
    best_effort_plan: Optional[Dict[str, Any]] = None
