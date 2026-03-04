from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .deps import AerSimulator, transpile
from .compiler import ExecutionPlan
from .readout import ReadoutMitigationModel, calibrate_readout_mitigator
from .risk import anytime_lower_bound, restart_delta, split_delta
from .tasks import BaseTask
from .utils import merge_counts_int, reward_stats_from_counts, run_backend_result, window_counts


# ---------------------------------------------------------------------
# Drift / change-point detection
# ---------------------------------------------------------------------

@dataclass
class PageHinkleyState:
    """Internal state for a Page–Hinkley change detector."""

    n: int = 0
    mean: float = 0.0
    m_t: float = 0.0
    m_max: float = 0.0


class PageHinkleyDetector:
    """Page–Hinkley detector tuned for *downward* drift in a bounded stream.

    We use the standard cumulative deviation statistic

        m_t = sum_{i<=t} (x_i - mean_i - delta)

    and track its running maximum m_max. A *drop* is detected when:

        m_max - m_t > lambda

    This is appropriate when we want to trigger on decreases in a performance metric
    (e.g., per-batch mean reward).
    """

    def __init__(self, delta: float = 0.005, lambda_: float = 0.05, min_instances: int = 5):
        self.delta = float(delta)
        self.lambda_ = float(lambda_)
        self.min_instances = int(min_instances)
        self.state = PageHinkleyState()

    def reset(self) -> None:
        self.state = PageHinkleyState()

    def update(self, x: float) -> bool:
        s = self.state
        s.n += 1

        # Running mean
        x = float(x)
        s.mean += (x - s.mean) / float(s.n)

        # Cumulative deviation and running maximum
        s.m_t += x - s.mean - self.delta
        s.m_max = max(s.m_max, s.m_t)

        if s.n < self.min_instances:
            return False

        # Detect significant *decrease* from the best-seen cumulative level.
        return (s.m_max - s.m_t) > self.lambda_

    def snapshot(self) -> Dict[str, Any]:
        s = self.state
        return {
            "type": "page_hinkley",
            "n": int(s.n),
            "mean": float(s.mean),
            "m_t": float(s.m_t),
            "m_max": float(s.m_max),
            "delta": float(self.delta),
            "lambda": float(self.lambda_),
            "min_instances": int(self.min_instances),
        }


# ---------------------------------------------------------------------
# Anytime execution (confidence sequences + optional drift guard/restarts)
# ---------------------------------------------------------------------

def _expected_reward_from_probs(task: BaseTask, probs: Dict[str, float]) -> float:
    """Compute E[reward(bitstring)] under a probability distribution."""
    s = 0.0
    for bit, p in probs.items():
        if p:
            s += float(p) * float(task.reward(bit))
    return float(s)


def _window_sums(batches: List[Tuple[float, int]], window: int) -> Tuple[float, int]:
    """Like utils.window_counts but for (sum_reward, shots) batches.

    We merge the most recent *whole* batches until reaching at least `window` shots.
    """
    w = int(window)
    if w <= 0:
        return 0.0, 0
    total_shots = 0
    total_sum = 0.0
    for s, n in reversed(batches):
        total_sum += float(s)
        total_shots += int(n)
        if total_shots >= w:
            break
    return float(total_sum), int(total_shots)


def execute_anytime_with_restarts(
    task: BaseTask,
    circuit,
    plan: ExecutionPlan,
    noise_model,
    seed: int = 2026,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Execute an anytime plan with optional drift-guard and drift-aware restarts.

    - Certificates (lower bounds) are computed from **RAW counts** only.
    - If readout mitigation is enabled, we also compute **estimate-only** means from
      mitigated probabilities for convenience, but they are not used for certificates.
    """
    backend = AerSimulator(method=plan.simulator_method)
    tqc = transpile(circuit, backend=backend, optimization_level=plan.optimization_level)

    mitigator: Optional[ReadoutMitigationModel] = None
    readout_calib_shots = 0
    if plan.readout_mitigation and tqc.num_qubits <= 4:
        sps = int(plan.mitigation_shots_per_state or 256)
        mitigator = calibrate_readout_mitigator(
            backend=backend,
            n_qubits=tqc.num_qubits,
            shots_per_state=sps,
            noise_model=noise_model,
            seed=seed,
            optimization_level=plan.optimization_level,
        )
        readout_calib_shots = int((2 ** int(tqc.num_qubits)) * int(sps))

    base_batch = int(plan.shots)
    # plan.shots_max is the *production* shot cap (pilot/calibration reserved at planning time).
    max_production_shots = int(plan.shots_max)
    max_shots = max_production_shots

    rule = dict(plan.stopping_rule or {})
    schedule = str(rule.get("checkpoint_schedule", "fixed") or "fixed")
    ratio = float(rule.get("checkpoint_ratio", 1.5) or 1.5)
    ratio = min(4.0, max(1.05, ratio))
    if schedule not in ("fixed", "geometric"):
        schedule = "fixed"

    batch = int(max(1, base_batch))

    # Drift-guard parameters (optional)
    window = int(rule.get("window_shots", 0) or 0)
    require_window = bool(rule.get("require_window_lb", False))

    # Restartable CS / drift detection
    drift_detect = bool(rule.get("drift_detect", False))
    restart_gamma = float(rule.get("restart_gamma", 0.2) or 0.2)
    max_restarts = int(rule.get("max_restarts", 5) or 5)
    min_segment_shots = int(rule.get("min_segment_shots", 0) or 0)

    drift_detector = str(rule.get("drift_detector", "page_hinkley") or "page_hinkley")
    ph_delta = float(rule.get("ph_delta", 0.005) or 0.005)
    ph_lambda = float(rule.get("ph_lambda", 0.05) or 0.05)
    ph_min_instances = int(rule.get("ph_min_instances", 5) or 5)

    detector: Optional[PageHinkleyDetector] = None
    if drift_detect and drift_detector == "page_hinkley":
        detector = PageHinkleyDetector(delta=ph_delta, lambda_=ph_lambda, min_instances=ph_min_instances)

    qmin = float(plan.qmin)
    delta_total = float(plan.delta_production)

    # Overall history across all segments
    history: List[Dict[str, Any]] = []
    restarts: List[Dict[str, Any]] = []

    # Segment state (since last restart)
    seg_counts_raw: Dict[str, int] = {}
    seg_batches_raw: List[Tuple[Dict[str, int], int]] = []
    seg_batches_est: List[Tuple[float, int]] = []  # (expected_sum_reward, shots)

    seg_shots = 0
    seg_checks = 0
    restart_idx = 0

    # Total state (since execution start)
    total_counts_raw: Dict[str, int] = {}
    total_shots = 0
    total_checks = 0

    satisfied = False
    stop_n_total = 0
    stop_n_segment = 0

    last = {
        "mean_seg_raw": 0.0,
        "mean_seg_est": 0.0,
        "L_seg": float("-inf"),
        "mean_win_raw": None,
        "mean_win_est": None,
        "L_win": None,
    }

    def _segment_deltas(r: int) -> Tuple[float, float, float]:
        """Return (delta_r_total, delta_seg, delta_win)."""
        d_r = restart_delta(delta_total=delta_total, restart_index=r, gamma=restart_gamma) if drift_detect else delta_total
        if require_window and window > 0:
            f = float(rule.get("delta_window_fraction", 0.5))
            # Avoid degenerate 0/1 splits.
            if not (0.0 < f < 1.0):
                f = 0.5
            d_seg, d_win = split_delta(d_r, f)
            return float(d_r), float(d_seg), float(d_win)
        return float(d_r), float(d_r), 0.0

    def _compute_bounds(seg_counts_local: Dict[str, int], seg_checks_local: int, delta_seg: float) -> Tuple[float, float]:
        sum_r, sum_r2, nn, succ = reward_stats_from_counts(task, seg_counts_local)
        if nn <= 0:
            return 0.0, float("-inf")
        mean = float(sum_r) / float(nn)
        ex2 = float(sum_r2) / float(nn)
        var = max(0.0, ex2 - mean * mean)
        L = float(
            anytime_lower_bound(
                method=str(plan.risk_bound.method),
                sample_mean=mean,
                n=nn,
                delta_total=float(delta_seg),
                t=int(seg_checks_local),
                value_range=task.quality_range(),
                sample_var=var,
                successes=succ,
            )
        )
        return mean, L

    while total_shots < max_shots:
        this = int(min(batch, max_shots - total_shots))
        res = run_backend_result(backend, tqc, shots=this, noise_model=noise_model, seed=seed + total_shots)
        counts_raw = dict(res.get_counts(0))

        # Update totals (certificates always use RAW counts)
        total_counts_raw = merge_counts_int(total_counts_raw, counts_raw)
        total_shots += int(this)
        total_checks += 1

        # Update segment (RAW)
        seg_counts_raw = merge_counts_int(seg_counts_raw, counts_raw)
        seg_batches_raw.append((counts_raw, int(this)))
        seg_shots += int(this)
        seg_checks += 1

        # Optional estimate-only tracking from mitigated probabilities (no rounding-to-counts)
        batch_exp_sum_est = 0.0
        if mitigator is not None:
            probs = mitigator.mitigate_counts(counts_raw)
            batch_mean_est = _expected_reward_from_probs(task, probs)
            batch_exp_sum_est = float(batch_mean_est) * float(this)
            seg_batches_est.append((float(batch_exp_sum_est), int(this)))

        # Update checkpoint batch size (geometric schedule)
        if schedule == "geometric":
            batch = int(max(1, math.ceil(float(batch) * ratio)))

        # Per-segment deltas
        delta_r, delta_seg, delta_win = _segment_deltas(restart_idx)

        # Compute segment certificate (RAW)
        mean_seg_raw, L_seg = _compute_bounds(seg_counts_raw, seg_checks, delta_seg)

        # Estimate-only segment mean (mitigated)
        if mitigator is not None and seg_shots > 0:
            seg_sum_est = sum(float(s) for s, _n in seg_batches_est)
            mean_seg_est = float(seg_sum_est) / float(seg_shots)
        else:
            mean_seg_est = float(mean_seg_raw)

        mean_win_raw = None
        mean_win_est = None
        L_win = None
        if require_window and window > 0:
            # Window certificate uses RAW counts. Note: window_counts merges whole batches until >= window shots.
            w_counts_raw, w_shots_raw = window_counts(seg_batches_raw, window=window)
            mean_win_raw, L_win = _compute_bounds(w_counts_raw, seg_checks, float(delta_win))

            # Estimate-only window mean from mitigated probabilities
            if mitigator is not None:
                w_sum_est, w_shots_est = _window_sums(seg_batches_est, window=window)
                mean_win_est = float(w_sum_est) / float(w_shots_est) if w_shots_est > 0 else 0.0

        last = {
            "mean_seg_raw": float(mean_seg_raw),
            "mean_seg_est": float(mean_seg_est),
            "L_seg": float(L_seg),
            "mean_win_raw": float(mean_win_raw) if mean_win_raw is not None else None,
            "mean_win_est": float(mean_win_est) if mean_win_est is not None else None,
            "L_win": float(L_win) if L_win is not None else None,
        }

        # Feed detector using per-batch RAW mean reward (bounded in [0,1])
        if detector is not None:
            sum_b, _, n_b, _ = reward_stats_from_counts(task, counts_raw)
            batch_mean_raw = float(sum_b) / float(max(1, n_b))
            triggered = detector.update(batch_mean_raw)
        else:
            triggered = False

        ok_seg = L_seg >= qmin
        ok_win = (not require_window) or ((L_win is not None) and (L_win >= qmin))

        rec = {
            "check": int(total_checks),
            "restart": int(restart_idx),
            "segment_check": int(seg_checks),
            "n_total": int(total_shots),
            "n_segment": int(seg_shots),
            "delta_total": float(delta_total),
            "delta_segment_total": float(delta_r),
            "delta_segment": float(delta_seg),
            "delta_window": float(delta_win),
            # Back-compat: mean_segment/mean_window refer to RAW means.
            "mean_segment": float(mean_seg_raw),
            "mean_segment_raw": float(mean_seg_raw),
            "mean_segment_est": float(mean_seg_est),
            "L_segment": float(L_seg),
            "mean_window": float(mean_win_raw) if mean_win_raw is not None else None,
            "mean_window_raw": float(mean_win_raw) if mean_win_raw is not None else None,
            "mean_window_est": float(mean_win_est) if mean_win_est is not None else None,
            "L_window": float(L_win) if L_win is not None else None,
            "ok_segment": bool(ok_seg),
            "ok_window": bool(ok_win),
            "detector_triggered": bool(triggered),
            "detector": detector.snapshot() if detector is not None else None,
        }
        history.append(rec)

        if verbose:
            msg = (
                f"[ANYTIME] r={restart_idx:2d} t={seg_checks:4d} "
                f"nseg={seg_shots:6d} ntot={total_shots:6d}  mean_raw≈{mean_seg_raw:.6f}  L≈{L_seg:.6f}"
            )
            if mitigator is not None:
                msg += f" | mean_est≈{mean_seg_est:.6f}"
            if require_window:
                msg += (
                    f" | meanW_raw≈{(mean_win_raw or 0.0):.6f}  "
                    f"LW≈{(L_win if L_win is not None else float('-inf')):.6f}"
                )
                if mitigator is not None and mean_win_est is not None:
                    msg += f" | meanW_est≈{mean_win_est:.6f}"
            if triggered:
                msg += "  [DRIFT?]"
            print(msg)

        # Success
        if ok_seg and ok_win:
            satisfied = True
            stop_n_total = int(total_shots)
            stop_n_segment = int(seg_shots)
            break

        # Drift detected -> restart certificate (if allowed)
        if drift_detect and triggered and seg_shots >= min_segment_shots and restart_idx < max_restarts:
            restarts.append(
                {
                    "restart_index": int(restart_idx),
                    "at_total_shots": int(total_shots),
                    "segment_shots": int(seg_shots),
                    "reason": "detector_triggered",
                    "detector": detector.snapshot() if detector is not None else None,
                    "last_mean_segment": float(mean_seg_raw),
                    "last_mean_segment_raw": float(mean_seg_raw),
                    "last_mean_segment_est": float(mean_seg_est),
                    "last_L_segment": float(L_seg),
                    "last_mean_window": float(mean_win_raw) if mean_win_raw is not None else None,
                    "last_mean_window_raw": float(mean_win_raw) if mean_win_raw is not None else None,
                    "last_mean_window_est": float(mean_win_est) if mean_win_est is not None else None,
                    "last_L_window": float(L_win) if L_win is not None else None,
                }
            )
            # Reset segment state and increment restart counter.
            restart_idx += 1
            seg_counts_raw = {}
            seg_batches_raw = []
            seg_batches_est = []
            seg_shots = 0
            seg_checks = 0
            if detector is not None:
                detector.reset()

    # Accounting: production shots (within plan.shots_max) plus readout calibration executed here.
    shots_max_total_executed = int(max_production_shots + readout_calib_shots)
    shots_max_total_accounted = int(shots_max_total_executed + int(plan.noise_calibration_shots or 0))

    return {
        "satisfied": bool(satisfied),
        "stop_n_total": int(stop_n_total),
        "stop_n_total_including_readout_calibration": int(stop_n_total + readout_calib_shots),
        "stop_n_segment": int(stop_n_segment),
        "calibration_shots": {
            "readout_executed": int(readout_calib_shots),
            # Noise calibration is performed during planning for calibrated_box (sat_checker._resolve_noise_box).
            "noise_reserved": int(plan.noise_calibration_shots or 0),
        },
        "shots_max_total_executed": int(shots_max_total_executed),
        "shots_max_total_accounted": int(shots_max_total_accounted),
        "production_shots_max": int(max_shots),
        "final_n_total": int(total_shots),
        "final_n_total_including_readout_calibration": int(total_shots + readout_calib_shots),
        "final_restart_index": int(restart_idx),
        # Back-compat: final_mean_* refer to RAW means.
        "final_mean_segment": float(last["mean_seg_raw"]),
        "final_mean_segment_raw": float(last["mean_seg_raw"]),
        "final_mean_segment_est": float(last["mean_seg_est"]),
        "final_L_segment": float(last["L_seg"]),
        "final_mean_window": last["mean_win_raw"],
        "final_mean_window_raw": last["mean_win_raw"],
        "final_mean_window_est": last["mean_win_est"],
        "final_L_window": last["L_win"],
        "history": history,
        "restarts": restarts,
        "policy": {
            "drift_guard": {"enabled": bool(require_window), "window_shots": int(window)},
            "drift_detect": bool(drift_detect),
            "restart_gamma": float(restart_gamma),
            "max_restarts": int(max_restarts),
            "min_segment_shots": int(min_segment_shots),
            "detector": str(drift_detector),
            "ph_delta": float(ph_delta),
            "ph_lambda": float(ph_lambda),
            "ph_min_instances": int(ph_min_instances),
        },
    }
