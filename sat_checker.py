from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

from .deps import AerSimulator, transpile
from .compiler import ExecutionPlan, UNSATCertificate
from .contract import QQoSContract, NoiseSpec
from .noise import (
    NoiseBox,
    build_noise_model,
    calibrate_noise_box,
    noise_box_from_spec,
    noise_summary,
    worst_case_params,
)
from .risk import (
    RiskBound,
    anytime_lower_bound,
    empirical_bernstein_lower_mean_bound,
    hoeffding_lower_mean_bound,
    hoeffding_required_shots,
    hoeffding_tail_risk,
    restart_delta,
    split_delta,
)
from .simulator import run_with_optional_mitigation
from .tasks import TASK_REGISTRY, BaseTask
from .utils import MIN_POS, reward_stats_from_counts


@dataclass
class SatisfiabilityResult:
    sat: bool
    plan: Optional[ExecutionPlan]
    certificate: Optional[UNSATCertificate]


def _normalized_risk_method(contract_type: str, method: str) -> str:
    """Ensure a risk method is appropriate for the contract type.

    - For anytime contracts, fixed-time methods are automatically upgraded to their
      anytime-valid counterparts.
    - For non-anytime contracts, this prototype supports only fixed-time methods.
    """
    ct = str(contract_type)
    m = str(method)

    if ct == "anytime":
        if m == "hoeffding":
            return "anytime_hoeffding"
        if m == "empirical_bernstein":
            return "anytime_empirical_bernstein"
        return m

    # hard / soft
    if m not in {"hoeffding", "empirical_bernstein"}:
        raise ValueError(
            f"RiskPolicy.method='{m}' requires contract_type='anytime' in this prototype."
        )
    return m


def _candidate_settings(contract: QQoSContract) -> List[Dict[str, Any]]:
    """Candidate compile/runtime settings to try."""
    opts = [0, 1, 2, 3]
    # This prototype always tries both with/without mitigation (feasibility checked later).
    mits = [False, True]
    return [
        {
            "optimization_level": int(opt),
            "readout_mitigation": bool(mit),
            "simulator_method": "density_matrix",
        }
        for opt in opts
        for mit in mits
    ]


def _pilot_stats(
    task: BaseTask, counts: Dict[str, int], shots: int
) -> Tuple[float, Optional[float], Optional[int]]:
    """Return (sample_mean_quality, sample_var, successes_or_None_if_not_bernoulli)."""
    q = float(task.quality_from_counts(counts, shots).quality)
    sum_r, sum_r2, n, succ = reward_stats_from_counts(task, counts)
    if n <= 1:
        return q, None, succ
    mean_r = float(sum_r) / float(n)
    ex2 = float(sum_r2) / float(n)
    var = max(0.0, ex2 - mean_r * mean_r)
    return q, float(var), succ


def _noise_calibration_cost(noise: NoiseSpec) -> int:
    # Our calibration uses 4 circuits (measure |0>,|1> and 1q/2q RB-ish proxies).
    if noise.uncertainty_model == "calibrated_box" and int(noise.calibration_shots) > 0:
        return 4 * int(noise.calibration_shots)
    return 0


def _resolve_noise_box(
    contract: QQoSContract, backend: Optional[AerSimulator] = None
) -> Tuple[Optional[NoiseBox], List[NoiseSpec], NoiseSpec, int]:
    """Return (noise_box, corners, worst_case_corner, calibration_cost).

    If uncertainty_model == "calibrated_box", we perform a small calibration routine
    (simulated on the provided backend) and return an inferred ambiguity set.
    """
    um = str(contract.noise.uncertainty_model)
    calib_cost = _noise_calibration_cost(contract.noise)

    if um == "calibrated_box":
        backend = backend or AerSimulator(method="density_matrix")
        box = calibrate_noise_box(
            backend=backend,
            noise_true=contract.noise,
            shots_per_experiment=int(contract.noise.calibration_shots),
            delta=float(contract.noise.calibration_delta),
            seed=2468,
            drift_scale=float(contract.noise.drift_scale),
            extra_scale=float(contract.noise.uncertainty_scale),
            optimization_level=1,
        )
        corners = box.corners(model=contract.noise.model)
        wc = box.worst_corner(model=contract.noise.model)
        return box, corners, wc, calib_cost

    if um == "box":
        box = noise_box_from_spec(contract.noise)
        corners = box.corners(model=contract.noise.model)
        wc = box.worst_corner(model=contract.noise.model)
        return box, corners, wc, calib_cost

    # "scale" / legacy modes: single worst-case spec.
    wc = worst_case_params(contract.noise)
    return None, [wc], wc, calib_cost


def _split_budget(total: int, k: int) -> List[int]:
    """Split an integer budget into k nonnegative integers summing to total."""
    total = int(max(0, total))
    k = int(max(1, k))
    base = total // k
    rem = total % k
    return [base + (1 if i < rem else 0) for i in range(k)]


def check_and_plan(contract: QQoSContract) -> SatisfiabilityResult:
    """Check satisfiability and (if SAT) return an execution plan.

    Notes on accounting
    -------------------
    This function *simulates* pilot and (optional) noise calibration shots to form a plan.
    The returned plan's shot accounting is consistent with what this function actually runs:
      total_shots ≈ pilot_shots_total + calibration_shots + production_shots.

    Readout mitigation is treated as an *execution-time* convenience only:
    certificates always use RAW counts.
    """
    # Resolve task
    if contract.task.name not in TASK_REGISTRY:
        raise ValueError(f"Unknown task: {contract.task.name}")
    task = TASK_REGISTRY[contract.task.name](contract.task.params)

    qmin = float(contract.resolved_qmin())
    delta = float(contract.delta)

    # Normalize method (critical for anytime correctness).
    method = _normalized_risk_method(contract.contract_type, contract.risk_policy.method)

    # Compile the base circuit once; transpile per candidate option.
    base_circ = task.build_circuit()
    backend = AerSimulator(method="density_matrix")

    # Noise handling (robust sets)
    noise_box, corners, wc_noise, noise_calib_shots = _resolve_noise_box(contract, backend=backend)
    noise_model = build_noise_model(wc_noise)

    # Candidate preprocessing (depth/size + mitigation feasibility)
    raw_candidates = _candidate_settings(contract)

    candidate_infos: List[Dict[str, Any]] = []
    min_depth_seen: Optional[int] = None
    for cand in raw_candidates:
        opt = int(cand["optimization_level"])
        sim_method = str(cand["simulator_method"])
        tqc = transpile(base_circ, backend=backend, optimization_level=opt)
        depth = int(tqc.depth())
        size = int(tqc.size())
        n_qubits = int(tqc.num_qubits)

        min_depth_seen = depth if min_depth_seen is None else min(min_depth_seen, depth)
        if depth > int(contract.budget.depth_max):
            continue

        mit = bool(cand["readout_mitigation"])
        mit_sps = 0
        ro_calib_shots = 0
        if mit and n_qubits <= 4:
            mit_sps = 256
            ro_calib_shots = int((2 ** n_qubits) * mit_sps)
        else:
            mit = False
            mit_sps = 0
            ro_calib_shots = 0

        candidate_infos.append(
            {
                "opt": opt,
                "sim_method": sim_method,
                "mit": mit,
                "mit_sps": mit_sps,
                "ro_calib_shots": ro_calib_shots,
                "tqc": tqc,
                "depth": depth,
                "size": size,
                "n_qubits": n_qubits,
            }
        )

    if not candidate_infos:
        cert = UNSATCertificate(
            reasons=["No candidate meets depth_max"],
            minimal_relaxations={"depth_max_minimum_needed": int(min_depth_seen or 0)},
            best_effort_plan=None,
        )
        return SatisfiabilityResult(sat=False, plan=None, certificate=cert)

    # Pilot budget (total across candidate selection + inference)
    pilot_budget = int(contract.risk_policy.pilot_shots)
    pilot_budget = max(0, pilot_budget)

    # Readout calibration is an execution-time overhead (one-time); noise calibration is planning-time.
    # We still account for both in the contract budget.
    best_plan: Optional[ExecutionPlan] = None
    best_effort_plan: Optional[ExecutionPlan] = None
    best_effort_info: Dict[str, Any] = {}

    best_mu_lb_seen = float("-inf")
    min_shots_needed_seen: Optional[int] = None

    # ------------------------------------------------------------
    # ANYTIME contracts: choose a candidate, then execute a stopping rule.
    # ------------------------------------------------------------
    if str(contract.contract_type) == "anytime":
        drift_window = int(getattr(contract.risk_policy, "drift_window", 0) or 0)
        drift_frac = float(getattr(contract.risk_policy, "drift_delta_fraction", 0.5) or 0.5)
        drift_frac = min(0.95, max(0.05, drift_frac))
        drift_enabled = drift_window > 0

        drift_detect = bool(getattr(contract.risk_policy, "drift_detect", False))
        restart_gamma = float(getattr(contract.risk_policy, "restart_gamma", 0.2))
        max_restarts = int(getattr(contract.risk_policy, "max_restarts", 5))
        min_segment_shots = int(getattr(contract.risk_policy, "min_segment_shots", 0) or 0)

        drift_detector = str(getattr(contract.risk_policy, "drift_detector", "page_hinkley") or "page_hinkley")
        ph_delta = float(getattr(contract.risk_policy, "ph_delta", 0.005) or 0.005)
        ph_lambda = float(getattr(contract.risk_policy, "ph_lambda", 0.05) or 0.05)
        ph_min_instances = int(getattr(contract.risk_policy, "ph_min_instances", 5) or 5)

        # Checkpoint schedule
        schedule = str(getattr(contract.risk_policy, "checkpoint_schedule", "fixed") or "fixed")
        chk_batch = int(getattr(contract.risk_policy, "checkpoint_batch", 0) or 0)
        chk_ratio = float(getattr(contract.risk_policy, "checkpoint_ratio", 1.5) or 1.5)
        chk_ratio = min(4.0, max(1.05, chk_ratio))
        if schedule not in ("fixed", "geometric"):
            schedule = "fixed"

        if chk_batch > 0:
            batch = int(chk_batch)
        else:
            batch = max(64, int(min(2048, max(64, pilot_budget // 2 if pilot_budget > 0 else 256))))

        # Allocate pilot across candidates just for ranking.
        K = len(candidate_infos)
        pilot_allocs = _split_budget(pilot_budget, K)

        best_idx = 0
        best_mu_lb = float("-inf")
        best_pilot_mean = 0.0
        best_pilot_var: Optional[float] = None
        best_pilot_succ: Optional[int] = None

        # Delta used for a *time-uniform* pilot lower bound (ranking only).
        delta0 = float(delta)
        if drift_detect:
            delta0 = float(restart_delta(delta_total=float(delta), restart_index=0, gamma=float(restart_gamma)))

        for i, info in enumerate(candidate_infos):
            shots_i = int(pilot_allocs[i])
            if shots_i <= 0:
                continue
            seed = 7777 + 17 * int(info["opt"]) + i
            out = run_with_optional_mitigation(
                circuit=info["tqc"],
                shots=shots_i,
                noise_model=noise_model,
                seed=seed,
                optimization_level=int(info["opt"]),
                simulator_method=str(info["sim_method"]),
                readout_mitigation=False,
                mitigation_shots_per_state=0,
                is_transpiled=True,
            )
            counts_raw = out.counts_raw
            pilot_mean, pilot_var, pilot_succ = _pilot_stats(task, counts_raw, shots_i)

            sum_r, sum_r2, n, succ = reward_stats_from_counts(task, counts_raw)
            if n <= 0:
                continue
            mean_r = float(sum_r) / float(n)
            ex2 = float(sum_r2) / float(n)
            var_r = max(0.0, ex2 - mean_r * mean_r)

            mu_lb = float(
                anytime_lower_bound(
                    method=method,
                    sample_mean=mean_r,
                    n=n,
                    delta_total=float(delta0),
                    value_range=task.quality_range(),
                    sample_var=float(var_r),
                    successes=succ,
                )
            )

            if mu_lb > best_mu_lb:
                best_mu_lb = float(mu_lb)
                best_idx = int(i)
                best_pilot_mean = float(pilot_mean)
                best_pilot_var = pilot_var
                best_pilot_succ = pilot_succ

        chosen = candidate_infos[int(best_idx)]
        opt = int(chosen["opt"])
        mit = bool(chosen["mit"])
        sim_method = str(chosen["sim_method"])
        depth = int(chosen["depth"])
        size = int(chosen["size"])
        ro_calib_shots = int(chosen["ro_calib_shots"])
        mit_sps = int(chosen["mit_sps"])

        calib_total = int(ro_calib_shots) + int(noise_calib_shots)
        shots_max = int(contract.budget.shots_max) - int(pilot_budget) - int(calib_total)

        # Ensure we can take at least one batch.
        if shots_max < batch:
            # Best effort: still return a plan that uses whatever production shots remain.
            best_effort_info = {
                "reason": "insufficient_shots_for_one_batch",
                "production_shots_max": int(max(0, shots_max)),
                "batch_shots": int(batch),
            }
            cert = UNSATCertificate(
                reasons=["Shot budget insufficient for anytime batch + calibration/pilot"],
                minimal_relaxations={
                    "shots_max_minimum_needed": int(pilot_budget) + int(calib_total) + int(batch)
                },
                best_effort_plan=None if not contract.fallback else best_effort_info,
            )
            return SatisfiabilityResult(sat=False, plan=None, certificate=cert)

        # Guarantee message (fix: window is over whole trailing batches)
        guarantee_msg = (
            f"Anytime certificate (method={method}): stop at the first n where L_n >= {qmin:.6f}. "
            f"Then P(true_mean >= {qmin:.6f}) >= {1.0 - delta:.6f} (time-uniform)."
        )
        if drift_enabled:
            guarantee_msg = (
                f"Anytime+drift-guard certificate (method={method}): stop at the first n where "
                f"L_n >= {qmin:.6f} AND L_tail >= {qmin:.6f} for a tail window of at least W={drift_window} shots "
                f"(computed from complete trailing batches). "
                f"Then P(global_mean >= {qmin:.6f} AND tail_mean >= {qmin:.6f}) >= {1.0 - delta:.6f} "
                f"(time-uniform; union-split)."
            )
        if drift_detect:
            guarantee_msg = (
                f"Restartable anytime certificate (drift-aware; method={method}): monitor for change-points and restart "
                f"the confidence sequence. In segment r, use delta_r=(1-gamma)*gamma^r*delta with gamma={restart_gamma:.3f}. "
                f"Stop when the segment bound L_seg >= {qmin:.6f}"
                + (
                    f" and the tail-window bound L_tail >= {qmin:.6f} (W={drift_window}, complete batches)"
                    if drift_enabled
                    else ""
                )
                + f". Overall failure probability across all segments is <= {delta:.6f} (union over restarts)."
            )

        delta_segment0 = float(restart_delta(delta_total=float(delta), restart_index=0, gamma=restart_gamma)) if drift_detect else float(delta)

        rb = RiskBound(
            method=str(method),
            delta=float(delta),
            guarantee=str(guarantee_msg),
            details={
                "contract_type": "anytime",
                "stopping_rule": "stop when time-uniform lower bound L_n >= qmin",
                "lower_bound_method": str(method),
                "delta_total": float(delta),
                "batch_shots": int(batch),
                "shots_max": int(shots_max),
                "noise_worst_case": noise_summary(wc_noise),
                "noise_worst_case_params": asdict(wc_noise),
                "noise_ambiguity_set": noise_box.summary() if noise_box is not None else None,
                "noise_corner_evaluation": "worst_corner_only",
                "noise_corners_total": int(len(corners)),
                "pilot": {
                    "pilot_shots_total": int(pilot_budget),
                    "pilot_shots_per_candidate": list(pilot_allocs),
                    "pilot_mean_selected": float(best_pilot_mean),
                    "pilot_time_uniform_L_selected": float(best_mu_lb),
                },
                "calibration_shots": {"readout": int(ro_calib_shots), "noise": int(noise_calib_shots)},
                "drift_guard": {"enabled": bool(drift_enabled), "window_shots": int(drift_window), "delta_window_fraction": float(drift_frac)},
                "restarts": {
                    "enabled": bool(drift_detect),
                    "delta_segment0": float(delta_segment0),
                    "restart_gamma": float(restart_gamma),
                    "max_restarts": int(max_restarts),
                    "min_segment_shots": int(min_segment_shots),
                    "drift_detector": str(drift_detector),
                    "ph_delta": float(ph_delta),
                    "ph_lambda": float(ph_lambda),
                    "ph_min_instances": int(ph_min_instances),
                },
                "budget_accounting": {
                    "total_shots_max": int(contract.budget.shots_max),
                    "pilot_shots": int(pilot_budget),
                    "calibration_shots": {"readout": int(ro_calib_shots), "noise": int(noise_calib_shots)},
                    "production_shots_max": int(shots_max),
                },
            },
        )

        plan = ExecutionPlan(
            task_name=contract.task.name,
            task_params=contract.task.params,
            optimization_level=opt,
            simulator_method=sim_method,
            readout_mitigation=mit,
            mitigation_shots_per_state=int(mit_sps),
            shots=int(batch),
            calibration_shots=int(ro_calib_shots),
            noise_calibration_shots=int(noise_calib_shots),
            pilot_shots=int(pilot_budget),
            pilot_quality=float(best_pilot_mean),
            mu_lower_bound=float(best_mu_lb),
            qmin=float(qmin),
            delta=float(delta),
            delta_pilot=0.0,
            delta_production=float(delta),
            depth=int(depth),
            size=int(size),
            risk_bound=rb,
            contract_type="anytime",
            shots_max=int(shots_max),
            stopping_rule={
                "type": "time_uniform_lower_bound",
                "method": str(method),
                "delta": float(delta),
                "batch_shots": int(batch),
                "checkpoint_schedule": str(schedule),
                "checkpoint_ratio": float(chk_ratio),
                "max_shots": int(shots_max),
                "qmin": float(qmin),
                "require_window_lb": bool(drift_enabled),
                "window_shots": int(drift_window),
                "delta_window_fraction": float(drift_frac),
                # Restarts
                "drift_detect": bool(drift_detect),
                "delta_segment0": float(delta_segment0),
                "restart_gamma": float(restart_gamma),
                "max_restarts": int(max_restarts),
                "min_segment_shots": int(min_segment_shots),
                "drift_detector": str(drift_detector),
                "ph_delta": float(ph_delta),
                "ph_lambda": float(ph_lambda),
                "ph_min_instances": int(ph_min_instances),
            },
        )

        return SatisfiabilityResult(sat=True, plan=plan, certificate=None)

    # ------------------------------------------------------------
    # HARD/SOFT contracts: fixed-N plan sized using a pilot lower bound.
    # ------------------------------------------------------------
    pilot_delta = float(getattr(contract.risk_policy, "pilot_delta", 0.0) or 0.0)

    # Clamp to [0, min(delta/2, delta - MIN_POS)] but do NOT floor to MIN_POS.
    pilot_delta_total = min(max(0.0, pilot_delta), float(delta) * 0.5)
    pilot_delta_total = min(pilot_delta_total, float(delta) - float(MIN_POS))
    delta_prod = float(delta - pilot_delta_total)

    pilot_correction = str(getattr(contract.risk_policy, "pilot_correction", "split") or "split")
    if pilot_correction not in {"split", "bonferroni"}:
        pilot_correction = "split"

    K = len(candidate_infos)

    if pilot_correction == "split":
        sel_frac = float(getattr(contract.risk_policy, "pilot_selection_fraction", 0.5) or 0.5)
        sel_frac = min(0.95, max(0.05, sel_frac))

        sel_total = int(round(float(pilot_budget) * sel_frac))
        sel_total = min(sel_total, pilot_budget)
        inf_total = int(pilot_budget - sel_total)

        sel_allocs = _split_budget(sel_total, K)

        # Selection: choose candidate with best pilot mean (RAW, under worst-case noise).
        best_sel_idx = 0
        best_sel_mean = float("-inf")
        for i, info in enumerate(candidate_infos):
            shots_i = int(sel_allocs[i])
            if shots_i <= 0:
                continue
            seed = 9000 + 17 * int(info["opt"]) + i
            out = run_with_optional_mitigation(
                circuit=info["tqc"],
                shots=shots_i,
                noise_model=noise_model,
                seed=seed,
                optimization_level=int(info["opt"]),
                simulator_method=str(info["sim_method"]),
                readout_mitigation=False,
                mitigation_shots_per_state=0,
                is_transpiled=True,
            )
            q_sel = float(task.quality_from_counts(out.counts_raw, shots_i).quality)
            if q_sel > best_sel_mean:
                best_sel_mean = float(q_sel)
                best_sel_idx = int(i)

        chosen = candidate_infos[int(best_sel_idx)]
        opt = int(chosen["opt"])
        mit = bool(chosen["mit"])
        sim_method = str(chosen["sim_method"])
        depth = int(chosen["depth"])
        size = int(chosen["size"])
        ro_calib_shots = int(chosen["ro_calib_shots"])
        mit_sps = int(chosen["mit_sps"])

        # Inference: use remaining pilot shots on selected candidate only.
        if inf_total <= 0:
            # No inference budget => no certified plan; best effort only.
            best_mu_lb_seen = max(best_mu_lb_seen, float(best_sel_mean))
            best_effort_info = {
                "reason": "no_inference_pilot_budget",
                "pilot_shots_total": int(pilot_budget),
                "pilot_selection_shots_total": int(sel_total),
                "pilot_inference_shots": int(inf_total),
                "selected_candidate": {"opt": int(opt), "readout_mitigation": bool(mit)},
                "selection_mean": float(best_sel_mean),
            }
        else:
            seed = 10000 + 17 * int(opt)
            out = run_with_optional_mitigation(
                circuit=chosen["tqc"],
                shots=inf_total,
                noise_model=noise_model,
                seed=seed,
                optimization_level=int(opt),
                simulator_method=str(sim_method),
                readout_mitigation=False,
                mitigation_shots_per_state=0,
                is_transpiled=True,
            )
            counts_inf = out.counts_raw
            pilot_quality, pilot_var, pilot_succ = _pilot_stats(task, counts_inf, inf_total)

            # Lower bound on mean quality from inference pilot only (selection bias avoided by split).
            if method == "empirical_bernstein" and pilot_var is not None and pilot_delta_total > 0.0:
                mu_lb = float(
                    empirical_bernstein_lower_mean_bound(
                        sample_mean=float(pilot_quality),
                        sample_var=float(pilot_var),
                        n=int(inf_total),
                        delta=float(pilot_delta_total),
                        value_range=task.quality_range(),
                    )
                )
            else:
                mu_lb = float(
                    hoeffding_lower_mean_bound(
                        sample_mean=float(pilot_quality),
                        n=int(inf_total),
                        delta=float(pilot_delta_total) if pilot_delta_total > 0.0 else 0.0,
                        value_range=task.quality_range(),
                    )
                )

            best_mu_lb_seen = max(best_mu_lb_seen, float(mu_lb))

            req = None
            if pilot_delta_total < delta and delta_prod > 0.0:
                # Choose production N so that P(sample_mean < qmin) <= delta_prod assuming mu >= mu_lb.
                req = hoeffding_required_shots(
                    mu_lb=float(mu_lb),
                    qmin=float(qmin),
                    delta=float(delta_prod),
                    value_range=task.quality_range(),
                    safety_floor_gap=float(contract.risk_policy.safety_floor_gap),
                )

            if req is not None:
                min_shots_needed_seen = req if min_shots_needed_seen is None else min(min_shots_needed_seen, req)

            calib_total = int(ro_calib_shots) + int(noise_calib_shots)
            total_cost = int(pilot_budget) + int(calib_total) + (int(req) if req is not None else 0)

            if req is not None and total_cost <= int(contract.budget.shots_max):
                tail = float(
                    hoeffding_tail_risk(
                        mu_lb=float(mu_lb),
                        qmin=float(qmin),
                        n=int(req),
                        value_range=task.quality_range(),
                    )
                )
                rb = RiskBound(
                    method=str(method),
                    delta=float(delta),
                    guarantee=(
                        f"P(achieved_quality >= {qmin:.6f}) >= {1.0 - delta:.6f} "
                        f"(δ_total={delta:.6f}; pilot δ={pilot_delta_total:.6f}; production δ={delta_prod:.6f}; "
                        f"pilot_correction=split; noise=worst_corner_only). "
                        f"Certificates use RAW counts; readout mitigation affects point estimates only."
                    ),
                    details={
                        "pilot_correction": "split",
                        "pilot_shots_total": int(pilot_budget),
                        "pilot_selection_fraction": float(sel_frac),
                        "pilot_selection_shots_total": int(sel_total),
                        "pilot_inference_shots": int(inf_total),
                        "pilot_delta_total": float(pilot_delta_total),
                        "mu_lower_bound": float(mu_lb),
                        "pilot_quality": float(pilot_quality),
                        "pilot_var": pilot_var,
                        "required_shots": int(req),
                        "tail_risk_bound": float(tail),
                        "total_shots_cost_including_pilot": int(total_cost),
                        "calibration_shots": {"readout": int(ro_calib_shots), "noise": int(noise_calib_shots)},
                        "noise_worst_case": noise_summary(wc_noise),
                "noise_worst_case_params": asdict(wc_noise),
                        "noise_ambiguity_set": noise_box.summary() if noise_box is not None else None,
                        "noise_corner_evaluation": "worst_corner_only",
                        "noise_corners_total": int(len(corners)),
                    },
                )
                plan = ExecutionPlan(
                    task_name=contract.task.name,
                    task_params=contract.task.params,
                    optimization_level=int(opt),
                    simulator_method=str(sim_method),
                    readout_mitigation=bool(mit),
                    mitigation_shots_per_state=int(mit_sps),
                    shots=int(req),
                    calibration_shots=int(ro_calib_shots),
                    noise_calibration_shots=int(noise_calib_shots),
                    pilot_shots=int(pilot_budget),
                    pilot_quality=float(pilot_quality),
                    mu_lower_bound=float(mu_lb),
                    qmin=float(qmin),
                    delta=float(delta),
                    delta_pilot=float(pilot_delta_total),
                    delta_production=float(delta_prod),
                    depth=int(depth),
                    size=int(size),
                    risk_bound=rb,
                    contract_type=str(contract.contract_type),
                    shots_max=0,
                    stopping_rule={},
                )
                return SatisfiabilityResult(sat=True, plan=plan, certificate=None)

            # Best effort (not certified)
            best_effort_info = {
                "pilot_correction": "split",
                "pilot_shots_total": int(pilot_budget),
                "pilot_selection_shots_total": int(sel_total),
                "pilot_inference_shots": int(inf_total),
                "pilot_quality": float(pilot_quality),
                "mu_lower_bound": float(mu_lb),
                "optimization_level": int(opt),
                "readout_mitigation": bool(mit),
                "depth": int(depth),
                "size": int(size),
                "noise_worst_case": noise_summary(wc_noise),
                "noise_worst_case_params": asdict(wc_noise),
                "noise_ambiguity_set": noise_box.summary() if noise_box is not None else None,
                "risk_method": str(method),
            }

    # Fallback / alternative: Bonferroni pilot correction (pilot spread across candidates)
    if pilot_correction == "bonferroni":
        pilot_allocs = _split_budget(pilot_budget, K)
        pilot_delta_per = float(pilot_delta_total) / float(max(1, K)) if pilot_delta_total > 0.0 else 0.0

        for i, info in enumerate(candidate_infos):
            shots_i = int(pilot_allocs[i])
            if shots_i <= 0:
                continue

            opt = int(info["opt"])
            mit = bool(info["mit"])
            sim_method = str(info["sim_method"])
            depth = int(info["depth"])
            size = int(info["size"])
            ro_calib_shots = int(info["ro_calib_shots"])
            mit_sps = int(info["mit_sps"])

            seed = 11000 + 17 * opt + i
            out = run_with_optional_mitigation(
                circuit=info["tqc"],
                shots=shots_i,
                noise_model=noise_model,
                seed=seed,
                optimization_level=int(opt),
                simulator_method=str(sim_method),
                readout_mitigation=False,
                mitigation_shots_per_state=0,
                is_transpiled=True,
            )

            counts_raw = out.counts_raw
            pilot_quality, pilot_var, pilot_succ = _pilot_stats(task, counts_raw, shots_i)

            if method == "empirical_bernstein" and pilot_var is not None and pilot_delta_per > 0.0:
                mu_lb = float(
                    empirical_bernstein_lower_mean_bound(
                        sample_mean=float(pilot_quality),
                        sample_var=float(pilot_var),
                        n=int(shots_i),
                        delta=float(pilot_delta_per),
                        value_range=task.quality_range(),
                    )
                )
            else:
                mu_lb = float(
                    hoeffding_lower_mean_bound(
                        sample_mean=float(pilot_quality),
                        n=int(shots_i),
                        delta=float(pilot_delta_per) if pilot_delta_per > 0.0 else 0.0,
                        value_range=task.quality_range(),
                    )
                )

            best_mu_lb_seen = max(best_mu_lb_seen, float(mu_lb))

            req = None
            if pilot_delta_total < delta and delta_prod > 0.0:
                req = hoeffding_required_shots(
                    mu_lb=float(mu_lb),
                    qmin=float(qmin),
                    delta=float(delta_prod),
                    value_range=task.quality_range(),
                    safety_floor_gap=float(contract.risk_policy.safety_floor_gap),
                )

            if req is not None:
                min_shots_needed_seen = req if min_shots_needed_seen is None else min(min_shots_needed_seen, req)

            calib_total = int(ro_calib_shots) + int(noise_calib_shots)
            total_cost = int(pilot_budget) + int(calib_total) + (int(req) if req is not None else 0)

            if req is None:
                continue
            if total_cost > int(contract.budget.shots_max):
                continue

            tail = float(
                hoeffding_tail_risk(
                    mu_lb=float(mu_lb),
                    qmin=float(qmin),
                    n=int(req),
                    value_range=task.quality_range(),
                )
            )
            rb = RiskBound(
                method=str(method),
                delta=float(delta),
                guarantee=(
                    f"P(achieved_quality >= {qmin:.6f}) >= {1.0 - delta:.6f} "
                    f"(δ_total={delta:.6f}; pilot δ={pilot_delta_total:.6f}; production δ={delta_prod:.6f}; "
                    f"pilot_correction=bonferroni; noise=worst_corner_only). "
                    f"Certificates use RAW counts; readout mitigation affects point estimates only."
                ),
                details={
                    "pilot_correction": "bonferroni",
                    "pilot_shots_total": int(pilot_budget),
                    "pilot_shots_per_candidate": list(pilot_allocs),
                    "pilot_shots_selected": int(shots_i),
                    "pilot_delta_total": float(pilot_delta_total),
                    "pilot_delta_per_candidate": float(pilot_delta_per),
                    "mu_lower_bound": float(mu_lb),
                    "pilot_quality": float(pilot_quality),
                    "pilot_var": pilot_var,
                    "required_shots": int(req),
                    "tail_risk_bound": float(tail),
                    "total_shots_cost_including_pilot": int(total_cost),
                    "calibration_shots": {"readout": int(ro_calib_shots), "noise": int(noise_calib_shots)},
                    "noise_worst_case": noise_summary(wc_noise),
                "noise_worst_case_params": asdict(wc_noise),
                    "noise_ambiguity_set": noise_box.summary() if noise_box is not None else None,
                    "noise_corner_evaluation": "worst_corner_only",
                    "noise_corners_total": int(len(corners)),
                },
            )
            plan = ExecutionPlan(
                task_name=contract.task.name,
                task_params=contract.task.params,
                optimization_level=int(opt),
                simulator_method=str(sim_method),
                readout_mitigation=bool(mit),
                mitigation_shots_per_state=int(mit_sps),
                shots=int(req),
                calibration_shots=int(ro_calib_shots),
                noise_calibration_shots=int(noise_calib_shots),
                pilot_shots=int(pilot_budget),
                pilot_quality=float(pilot_quality),
                mu_lower_bound=float(mu_lb),
                qmin=float(qmin),
                delta=float(delta),
                delta_pilot=float(pilot_delta_total),
                delta_production=float(delta_prod),
                depth=int(depth),
                size=int(size),
                risk_bound=rb,
                contract_type=str(contract.contract_type),
                shots_max=0,
                stopping_rule={},
            )

            if best_plan is None:
                best_plan = plan
            else:
                plan_cost = int(plan.shots) + int(plan.calibration_shots) + int(plan.noise_calibration_shots) + int(plan.pilot_shots)
                best_cost = int(best_plan.shots) + int(best_plan.calibration_shots) + int(best_plan.noise_calibration_shots) + int(best_plan.pilot_shots)
                if plan_cost < best_cost or (plan_cost == best_cost and plan.mu_lower_bound > best_plan.mu_lower_bound):
                    best_plan = plan

        if best_plan is not None:
            return SatisfiabilityResult(sat=True, plan=best_plan, certificate=None)

    # ------------------------------------------------------------
    # UNSAT: build a certificate + (optional) best-effort suggestion.
    # ------------------------------------------------------------
    reasons: List[str] = []
    relax: Dict[str, Any] = {}

    if min_depth_seen is not None and min_depth_seen > int(contract.budget.depth_max):
        reasons.append("Depth bound too tight")
        relax["depth_max_minimum_needed"] = int(min_depth_seen)

    if min_shots_needed_seen is not None:
        needed = int(pilot_budget) + int(min_shots_needed_seen) + int(_noise_calibration_cost(contract.noise))
        if needed > int(contract.budget.shots_max):
            reasons.append("Shot budget insufficient for the requested δ and qmin")
            relax["shots_max_minimum_needed"] = int(needed)

    if not reasons:
        um = str(contract.noise.uncertainty_model)
        if um in {"box", "calibrated_box"}:
            reasons.append("Robust noise ambiguity set makes contract impossible under current budget")
        elif float(contract.noise.uncertainty_scale) > 0:
            reasons.append("Noise uncertainty makes contract impossible (under worst-case errors)")
        else:
            reasons.append("Quality requirement too strict under the assumed noise")
        relax["suggested_qmin"] = float(best_mu_lb_seen) if math.isfinite(best_mu_lb_seen) else 0.0
        relax["suggested_eps_max"] = float(1.0 - relax["suggested_qmin"])

    best_effort: Optional[Dict[str, Any]] = None
    include_best_effort = bool(contract.fallback) or str(contract.contract_type) == "soft"
    if include_best_effort:
        # Choose a conservative candidate for a best-effort run:
        #   1) prefer no readout mitigation (avoids expensive calibration),
        #   2) then prefer smaller depth,
        #   3) then prefer lower optimization level.
        best_cand = min(
            candidate_infos,
            key=lambda x: (int(x.get("ro_calib_shots", 0)), int(x.get("depth", 0)), int(x.get("opt", 0))),
        )

        opt = int(best_cand["opt"])
        sim_method = str(best_cand["sim_method"])
        mit = bool(best_cand["mit"])
        mit_sps = int(best_cand["mit_sps"])
        ro_calib_shots = int(best_cand["ro_calib_shots"])
        depth = int(best_cand["depth"])
        size = int(best_cand["size"])

        # Spend remaining budget on production (pilot/noise-calibration/readout-calibration are reserved).
        remaining_prod = int(contract.budget.shots_max) - int(pilot_budget) - int(noise_calib_shots) - int(ro_calib_shots)
        remaining_prod = int(max(0, remaining_prod))

        pilot_q = float(best_effort_info.get("pilot_quality", 0.0) if isinstance(best_effort_info, dict) else 0.0)
        mu_lb = float(best_effort_info.get("mu_lower_bound", best_mu_lb_seen) if isinstance(best_effort_info, dict) else best_mu_lb_seen)

        rb_be = RiskBound(
            method=str(method),
            delta=float(delta),
            guarantee=(
                "BEST EFFORT ONLY (NOT CERTIFIED): no feasible plan can guarantee the requested qmin under the "
                "current constraints. This plan uses the remaining budget to produce an empirical estimate."
            ),
            details={
                "requested_qmin": float(qmin),
                "requested_delta": float(delta),
                "reasons": list(reasons),
                "minimal_relaxations": dict(relax),
                "planner_summary": dict(best_effort_info or {}),
                "noise_worst_case": noise_summary(wc_noise),
                "noise_worst_case_params": asdict(wc_noise),
                "noise_ambiguity_set": noise_box.summary() if noise_box is not None else None,
                "budget_accounting": {
                    "total_shots_max": int(contract.budget.shots_max),
                    "pilot_shots": int(pilot_budget),
                    "noise_calibration_shots": int(noise_calib_shots),
                    "readout_calibration_shots": int(ro_calib_shots),
                    "production_shots": int(remaining_prod),
                },
            },
        )

        be_plan = ExecutionPlan(
            task_name=contract.task.name,
            task_params=contract.task.params,
            optimization_level=int(opt),
            simulator_method=str(sim_method),
            readout_mitigation=bool(mit),
            mitigation_shots_per_state=int(mit_sps),
            shots=int(remaining_prod),
            calibration_shots=int(ro_calib_shots),
            noise_calibration_shots=int(noise_calib_shots),
            pilot_shots=int(pilot_budget),
            pilot_quality=float(pilot_q),
            mu_lower_bound=float(mu_lb),
            qmin=float(qmin),
            delta=float(delta),
            delta_pilot=float(pilot_delta_total),
            delta_production=float(delta_prod),
            depth=int(depth),
            size=int(size),
            risk_bound=rb_be,
            contract_type=str(contract.contract_type),
            shots_max=0,
            stopping_rule={},
        )

        best_effort = {
            "note": "Uncertified best-effort plan (auto-included for soft contracts, or when --fallback is set).",
            "plan": be_plan.to_json_dict(),
        }

    cert = UNSATCertificate(reasons=reasons, minimal_relaxations=relax, best_effort_plan=best_effort)
    return SatisfiabilityResult(sat=False, plan=None, certificate=cert)
