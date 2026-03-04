from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Literal


ContractType = Literal["hard", "soft", "anytime"]

NoiseUncertaintyModel = Literal["scale", "box", "calibrated_box"]


@dataclass(frozen=True)
class NoiseSpec:
    """Noise model parameters plus an uncertainty/drift policy.

    This prototype focuses on Aer custom noise models:
    - depolarizing error on 1q gates
    - depolarizing error on 2q gates
    - symmetric readout bit-flip error

    Uncertainty modes:
      - uncertainty_model="scale": one-sided worst-case inflation (legacy)
            p' = p * (1 + uncertainty_scale)
      - uncertainty_model="box": two-sided box around nominal
            p ∈ [p*(1-uncertainty_scale), p*(1+uncertainty_scale)]
      - uncertainty_model="calibrated_box": learn a box from calibration experiments, then (optionally)
        expand it with uncertainty_scale and drift_scale.

    Drift:
      - drift_scale expands the uncertainty box to account for future drift between calibration and execution.

    Calibration (only for uncertainty_model="calibrated_box"):
      - calibration_shots: shots per calibration experiment (0 disables calibration)
      - calibration_delta: confidence level for the calibration-derived intervals (per-parameter allocation is internal)
    """
    model: Literal["depolarizing_readout"] = "depolarizing_readout"
    p1: float = 0.001          # 1-qubit depolarizing
    p2: float = 0.01           # 2-qubit depolarizing
    p_ro: float = 0.02         # readout bitflip

    # Uncertainty specification
    uncertainty_scale: float = 0.0
    uncertainty_model: NoiseUncertaintyModel = "scale"

    # Drift (extra inflation of the learned/declared box)
    drift_scale: float = 0.0

    # Optional calibration (calibrated_box)
    calibration_shots: int = 0
    calibration_delta: float = 0.05


RiskMethod = Literal[
    "hoeffding",
    "empirical_bernstein",
    "anytime_hoeffding",
    "anytime_empirical_bernstein",
    "betting_cs",
    "cs_normal_mixture",
    "mixture_cs",
    "bernoulli_mixture_cs",
]

PilotCorrection = Literal["bonferroni", "split"]

CheckpointSchedule = Literal["fixed", "geometric"]

# Online drift / change-point detection
DriftDetector = Literal["none", "page_hinkley"]

# Runtime validation helpers (kept in sync with the Literal type aliases above).
CONTRACT_TYPE_CHOICES = {"hard", "soft", "anytime"}

NOISE_MODEL_CHOICES = {"depolarizing_readout"}
NOISE_UNCERTAINTY_MODEL_CHOICES = {"scale", "box", "calibrated_box"}

RISK_METHOD_CHOICES = {
    "hoeffding",
    "empirical_bernstein",
    "anytime_hoeffding",
    "anytime_empirical_bernstein",
    "betting_cs",
    "cs_normal_mixture",
    "mixture_cs",
    "bernoulli_mixture_cs",
}
PILOT_CORRECTION_CHOICES = {"bonferroni", "split"}
CHECKPOINT_SCHEDULE_CHOICES = {"fixed", "geometric"}
DRIFT_DETECTOR_CHOICES = {"none", "page_hinkley"}



@dataclass(frozen=True)
class RiskPolicy:
    """Risk policy (ρ).

    Supported methods:
      - "hoeffding": fixed-time Hoeffding bounds (baseline)
      - "empirical_bernstein": fixed-time empirical Bernstein lower bound (tighter, variance-adaptive)
      - "anytime_hoeffding": time-uniform Hoeffding via a union-bound schedule (valid for all n)
      - "anytime_empirical_bernstein": time-uniform empirical Bernstein via a union-bound schedule (bounded rewards)
      - "betting_cs": time-uniform confidence sequence (Bernoulli via CP intervals + union bound schedule;
        otherwise falls back to anytime_hoeffding)
      - "cs_normal_mixture"/"mixture_cs": time-uniform confidence sequence for bounded rewards
      - "bernoulli_mixture_cs": time-uniform confidence sequence specialized for Bernoulli rewards

    Pilot selection / adaptivity handling (hard contracts):

      Two options are supported for turning pilot data into a valid lower bound used for sizing production shots:

      - pilot_correction="bonferroni" (legacy):
            Evaluate every candidate on the pilot budget and compute per-candidate (and per-noise-corner) bounds
            using a Bonferroni split of pilot_delta.

      - pilot_correction="split" (recommended):
            Use data splitting to avoid selection bias:
              1) use pilot_selection_fraction of pilot shots to *select* a promising candidate (no inference)
              2) use the remaining pilot shots to compute a bound on the selected candidate only
                 (pilot_delta is only split across noise corners, not across candidates)

      pilot_selection_fraction controls the split when pilot_correction="split".

    Drift-aware + restartable stopping (anytime execution):

      There are two complementary layers:

      (A) Drift guard (window certificate):
        - drift_window: if >0, require a trailing tail window of at least W shots to also satisfy a
          time-uniform lower bound. Because shots arrive in batches, the tail window is computed from
          complete trailing batches (it may contain >= W shots). This guards against late-stage degradation.
        - drift_delta_fraction: fraction of the per-segment delta allocated to the window bound
          (rest to the segment bound).

      (B) Change-point detection + restarts:
        - drift_detect: if True, monitor batch performance online and restart the certificate
          when a statistically meaningful drop is detected.
        - drift_detector: "page_hinkley" (default) or "none".
        - ph_delta, ph_lambda, ph_min_instances: Page–Hinkley detector parameters.
        - min_segment_shots: minimum shots in a segment before restarts are allowed.
        - max_restarts: hard cap on restarts to avoid infinite loops.
        - restart_gamma: delta spending parameter for restarts. Segment r uses:
              delta_r = (1 - gamma) * gamma^r * delta_total
          so sum_r delta_r = delta_total. Smaller gamma allocates more risk to early segments.
    """
    method: RiskMethod = "hoeffding"
    pilot_shots: int = 512
    pilot_delta: float = 0.05  # portion of δ reserved for pilot estimation (hard contracts)
    safety_floor_gap: float = 1e-6  # to avoid division by zero

    pilot_correction: PilotCorrection = "split"
    pilot_selection_fraction: float = 0.5  # fraction of pilot_shots used for selection (split mode)

    # Optional drift-guard (anytime contracts)
    drift_window: int = 0
    drift_delta_fraction: float = 0.5

    # Optional checkpoint schedule for anytime execution
    # - fixed: constant batch_shots each check
    # - geometric: increase batch size multiplicatively to reduce overhead
    checkpoint_schedule: CheckpointSchedule = "fixed"
    checkpoint_batch: int = 0  # 0 => auto (derived from pilot_shots)
    checkpoint_ratio: float = 1.5

    # ---- New: change-point detection + restartable confidence sequences (anytime) ----
    drift_detect: bool = False
    drift_detector: DriftDetector = "page_hinkley"

    # Page–Hinkley detector params (applied to per-batch reward means)
    ph_delta: float = 0.005
    ph_lambda: float = 0.05
    ph_min_instances: int = 5

    # Restart controls
    min_segment_shots: int = 0
    max_restarts: int = 5
    restart_gamma: float = 0.2


@dataclass(frozen=True)
class Budget:
    shots_max: int
    depth_max: int
    runtime_max_s: Optional[float] = None  # kept for completeness; not enforced in this prototype


@dataclass(frozen=True)
class TaskSpec:
    """Task definition. 'name' selects a built-in workload."""
    name: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class QQoSContract:
    """Quantum QoS contract (𝒞).

    Matches the paper idea: 𝒞=(Task, ε or Qmin, δ, S, T, D, ρ).
    """
    task: TaskSpec

    # Quality requirement: either a minimum quality OR a maximum error.
    qmin: Optional[float] = None
    eps_max: Optional[float] = None  # if provided, qmin is interpreted as 1-eps_max by default

    # Confidence requirement
    delta: float = 0.05

    # Budget requirements
    budget: Budget = field(default_factory=lambda: Budget(shots_max=8192, depth_max=200))

    # Contract type (hard / soft / anytime)
    contract_type: ContractType = "hard"

    # Risk policy ρ (optional)
    risk_policy: RiskPolicy = field(default_factory=RiskPolicy)

    # Noise model
    noise: NoiseSpec = field(default_factory=NoiseSpec)

    # Fallback requirement: if not feasible, return best classical/approx solution + explanation
    fallback: bool = True

    def resolved_qmin(self) -> float:
        if self.qmin is not None:
            return float(self.qmin)
        if self.eps_max is not None:
            return float(1.0 - self.eps_max)
        raise ValueError("Contract must define either qmin or eps_max.")