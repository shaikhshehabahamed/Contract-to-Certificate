from __future__ import annotations

"""CLI entrypoint for the QQoS prototype.

Run:
  python -m qqos --help
"""

import argparse
import json
import os
import sys
from dataclasses import asdict
from typing import Any, Dict, Optional

# Allow running as a script: python qqos/__main__.py
if __package__ in (None, ""):
    _pkg_dir = os.path.dirname(__file__)
    _parent = os.path.dirname(_pkg_dir)
    if _parent not in sys.path:
        sys.path.insert(0, _parent)
    __package__ = "qqos"

from .compiler import ExecutionPlan
from .contract import Budget, NoiseSpec, QQoSContract, RiskPolicy, TaskSpec
from .noise import build_noise_model, worst_case_params
from .online import execute_anytime_with_restarts
from .parser import load_contract
from .sat_checker import check_and_plan
from .simulator import run_with_optional_mitigation
from .tasks import TASK_REGISTRY


def _coerce_scalar(v: str) -> Any:
    """Coerce a CLI scalar to bool/int/float when safe.

    IMPORTANT: preserve strings with leading zeros (e.g., IDs like "012") by keeping them as strings.
    """
    v = str(v)

    if v.lower() in {"true", "false"}:
        return v.lower() == "true"

    # Preserve leading-zero numeric-looking strings as strings: "012", "007", "012.3", "012e2".
    if len(v) > 1 and v[0] == "0" and v[1].isdigit():
        return v

    try:
        return int(v)
    except Exception:
        pass

    try:
        return float(v)
    except Exception:
        pass

    return v


def _parse_kv_list(kvs: Optional[list[str]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not kvs:
        return out
    for item in kvs:
        if "=" not in item:
            raise ValueError(f"Invalid key=value item: {item}")
        k, v = item.split("=", 1)
        out[str(k)] = _coerce_scalar(v)
    return out


def _expected_reward_from_probs(task, probs: Dict[str, float]) -> float:
    s = 0.0
    for bit, p in probs.items():
        if p:
            s += float(p) * float(task.reward(bit))
    return float(s)


def _write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")


def _build_contract_from_args(args: argparse.Namespace) -> QQoSContract:
    task_params = _parse_kv_list(args.task_param)

    task = TaskSpec(name=str(args.task), params=task_params)

    budget = Budget(
        shots_max=int(args.shots_max),
        depth_max=int(args.depth_max),
    )

    noise = NoiseSpec(
        model=str(args.noise_model),
        p1=float(args.p1),
        p2=float(args.p2),
        p_ro=float(args.p_ro),
        uncertainty_model=str(args.uncertainty_model),
        uncertainty_scale=float(args.uncertainty_scale),
        drift_scale=float(args.drift_scale),
        calibration_shots=int(args.noise_calibration_shots),
        calibration_delta=float(args.noise_calibration_delta),
    )

    risk = RiskPolicy(
        method=str(args.risk_method),
        pilot_shots=int(args.pilot_shots),
        pilot_delta=float(args.pilot_delta),
        pilot_correction=str(args.pilot_correction),
        pilot_selection_fraction=float(args.pilot_selection_fraction),
        safety_floor_gap=float(args.safety_floor_gap),
        # Anytime extras
        checkpoint_schedule=str(args.checkpoint_schedule),
        checkpoint_batch=int(args.checkpoint_batch),
        checkpoint_ratio=float(args.checkpoint_ratio),
        drift_window=int(args.drift_window),
        drift_delta_fraction=float(args.drift_delta_fraction),
        drift_detect=bool(args.drift_detect),
        restart_gamma=float(args.restart_gamma),
        max_restarts=int(args.max_restarts),
        min_segment_shots=int(args.min_segment_shots),
        drift_detector=str(args.drift_detector),
        ph_delta=float(args.ph_delta),
        ph_lambda=float(args.ph_lambda),
        ph_min_instances=int(args.ph_min_instances),
    )

    return QQoSContract(
        task=task,
        qmin=float(args.qmin) if args.qmin is not None else None,
        eps_max=float(args.eps_max) if args.eps_max is not None else None,
        delta=float(args.delta),
        budget=budget,
        noise=noise,
        risk_policy=risk,
        contract_type=str(args.contract_type),
        fallback=bool(args.fallback),
    )


def _load_noise_worst_case_from_plan(plan: ExecutionPlan, contract: QQoSContract) -> NoiseSpec:
    """Prefer the planner's explicit worst-case parameters if available."""
    d = (plan.risk_bound.details or {})
    params = d.get("noise_worst_case_params")
    if isinstance(params, dict) and params:
        try:
            return NoiseSpec(**params)
        except Exception:
            pass
    # Fallback: recompute from the contract.
    return worst_case_params(contract.noise)


def _execute_fixed_plan(task, plan: ExecutionPlan, contract: QQoSContract, outfile: str, seed: int = 2026) -> None:
    wc = _load_noise_worst_case_from_plan(plan, contract)
    noise_model = build_noise_model(wc)

    out = run_with_optional_mitigation(
        circuit=task.build_circuit(),
        shots=int(plan.shots),
        noise_model=noise_model,
        seed=seed,
        optimization_level=int(plan.optimization_level),
        simulator_method=str(plan.simulator_method),
        readout_mitigation=bool(plan.readout_mitigation),
        mitigation_shots_per_state=int(plan.mitigation_shots_per_state),
    )

    achieved_raw = float(task.quality_from_counts(out.counts_raw, int(plan.shots)).quality)

    achieved_est = None
    if out.counts_mitigated is not None:
        achieved_est = float(_expected_reward_from_probs(task, out.counts_mitigated))

    print("\n=== EXECUTION RESULT (fixed shots) ===")
    print(f"depth={out.depth} size={out.size} shots={plan.shots}")
    print(f"achieved_raw  ={achieved_raw:.6f}")
    if achieved_est is not None:
        print(f"achieved_est  ={achieved_est:.6f}  (readout mitigated; estimate-only)")
    print("======================================\n")

    _write_json(
        outfile,
        {
            "plan": plan.to_json_dict(),
            "execution": {
                "achieved_raw": achieved_raw,
                "achieved_est": achieved_est,
                "depth": out.depth,
                "size": out.size,
                "metadata": out.metadata,
            },
        },
    )


def _execute_anytime_plan(task, plan: ExecutionPlan, contract: QQoSContract, outfile: str, seed: int = 2026) -> None:
    wc = _load_noise_worst_case_from_plan(plan, contract)
    noise_model = build_noise_model(wc)

    out = execute_anytime_with_restarts(
        task=task,
        circuit=task.build_circuit(),
        plan=plan,
        noise_model=noise_model,
        seed=seed,
        verbose=True,
    )

    # Derive a couple of back-compat keys for convenience.
    out["stop_n"] = out.get("stop_n_total", 0)
    out["stop_n_including_readout_calibration"] = out.get(
        "stop_n_total_including_readout_calibration", out.get("stop_n_total", 0)
    )
    out["final_mean_global"] = out.get("final_mean_segment", 0.0)
    out["final_L_global"] = out.get("final_L_segment", float("-inf"))

    print("\n=== EXECUTION RESULT (anytime) ===")
    print(f"satisfied={out.get('satisfied')} stop_n_total={out.get('stop_n_total')}")
    print(f"final_mean_segment_raw={out.get('final_mean_segment_raw')}")
    print(f"final_L_segment={out.get('final_L_segment')}")
    if out.get("final_L_window") is not None:
        print(f"final_L_window={out.get('final_L_window')}")
    print("==================================\n")

    _write_json(outfile, {"plan": plan.to_json_dict(), "anytime_execution": out})


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m qqos")

    parser.add_argument("--contract", type=str, default=None, help="Path to a JSON contract file")
    parser.add_argument("--out_plan", type=str, default="plan.json", help="Write plan JSON here")
    parser.add_argument("--out", type=str, default="result.json", help="Write result JSON here")
    parser.add_argument("--execute", action="store_true", help="Execute the plan after planning")
    parser.add_argument(
        "--execute_best_effort",
        action="store_true",
        help="If UNSAT and fallback info exists, execute the best-effort plan (uncertified).",
    )

    # Task + contract core
    parser.add_argument("--task", type=str, default="ghz_success", choices=sorted(TASK_REGISTRY.keys()))
    parser.add_argument("--task_param", action="append", default=[], help="Task param key=value (repeatable)")
    parser.add_argument("--qmin", type=float, default=None)
    parser.add_argument("--eps_max", type=float, default=None)
    parser.add_argument("--delta", type=float, default=1e-3)
    parser.add_argument("--contract_type", type=str, default="hard", choices=["hard", "soft", "anytime"])
    parser.add_argument("--fallback", action="store_true", help="Include best-effort suggestions on UNSAT")

    # Budget
    parser.add_argument("--shots_max", type=int, default=12000)
    parser.add_argument("--depth_max", type=int, default=200)

    # Noise
    parser.add_argument("--noise_model", type=str, default="depolarizing_readout")
    parser.add_argument("--p1", type=float, default=0.001)
    parser.add_argument("--p2", type=float, default=0.01)
    parser.add_argument("--p_ro", type=float, default=0.02)
    parser.add_argument("--uncertainty_model", type=str, default="scale", choices=["scale", "box", "calibrated_box"])
    parser.add_argument("--uncertainty_scale", type=float, default=0.2)
    parser.add_argument("--drift_scale", type=float, default=0.0)
    parser.add_argument("--noise_calibration_shots", type=int, default=0)
    parser.add_argument("--noise_calibration_delta", type=float, default=0.05)
    parser.add_argument("--noise_seed", type=int, default=None)

    # Risk / planner settings
    parser.add_argument(
        "--risk_method",
        type=str,
        default="anytime_hoeffding",
        choices=[
            "hoeffding",
            "empirical_bernstein",
            "anytime_hoeffding",
            "anytime_empirical_bernstein",
            "betting_cs",
            "cs_normal_mixture",
            "mixture_cs",
            "bernoulli_mixture_cs",
        ],
    )
    parser.add_argument("--pilot_shots", type=int, default=512, help="Total pilot shots used by the planner")
    parser.add_argument("--pilot_delta", type=float, default=5e-4)
    parser.add_argument("--pilot_correction", type=str, default="split", choices=["split", "bonferroni"])
    parser.add_argument("--pilot_selection_fraction", type=float, default=0.5)
    parser.add_argument("--safety_floor_gap", type=float, default=1e-4)

    # Anytime-only knobs (safe to pass for non-anytime; ignored)
    parser.add_argument("--checkpoint_schedule", type=str, default="fixed", choices=["fixed", "geometric"])
    parser.add_argument("--checkpoint_batch", type=int, default=0, help="Batch size; 0 => auto")
    parser.add_argument("--checkpoint_ratio", type=float, default=1.5)

    parser.add_argument(
        "--drift_window",
        type=int,
        default=0,
        help=(
            "If >0, require a trailing tail window of at least W shots to also certify qmin. "
            "Because shots arrive in batches, the tail window is computed from complete trailing batches "
            "(so it may contain >= W shots)."
        ),
    )
    parser.add_argument("--drift_delta_fraction", type=float, default=0.5)

    parser.add_argument("--drift_detect", action="store_true")
    parser.add_argument("--restart_gamma", type=float, default=0.2)
    parser.add_argument("--max_restarts", type=int, default=5)
    parser.add_argument("--min_segment_shots", type=int, default=0)
    parser.add_argument("--drift_detector", type=str, default="page_hinkley", choices=["page_hinkley"])
    parser.add_argument("--ph_delta", type=float, default=0.005)
    parser.add_argument("--ph_lambda", type=float, default=0.05)
    parser.add_argument("--ph_min_instances", type=int, default=5)

    args = parser.parse_args(argv)

    contract = load_contract(args.contract) if args.contract else _build_contract_from_args(args)

    result = check_and_plan(contract)

    if result.sat and result.plan is not None:
        plan = result.plan
        print("SAT. Plan:")
        print(json.dumps(plan.to_json_dict(), indent=2, sort_keys=True))

        _write_json(args.out_plan, plan.to_json_dict())

        if args.execute:
            if contract.task.name not in TASK_REGISTRY:
                raise ValueError(f"Unknown task: {contract.task.name}")
            task = TASK_REGISTRY[contract.task.name](contract.task.params)

            if plan.contract_type == "anytime":
                _execute_anytime_plan(task, plan, contract, args.out, seed=(int(args.noise_seed) if args.noise_seed is not None else 2026))
            else:
                _execute_fixed_plan(task, plan, contract, args.out, seed=(int(args.noise_seed) if args.noise_seed is not None else 2026))

        return 0

    # UNSAT
    cert = result.certificate
    out_doc = {
        "sat": False,
        "certificate": asdict(cert) if cert is not None else None,
    }
    _write_json(args.out, out_doc)
    print("UNSAT. Certificate written to:", args.out)
    if cert is not None:
        print(json.dumps(asdict(cert), indent=2, sort_keys=True))

    if args.execute_best_effort and cert is not None and isinstance(cert.best_effort_plan, dict):
        be = cert.best_effort_plan
        plan_dict = be.get("plan") if isinstance(be.get("plan"), dict) else None
        if plan_dict is None and all(k in be for k in ("task_name", "risk_bound")):
            # Back-compat: sometimes best_effort_plan *is* a plan dict.
            plan_dict = be

        if plan_dict is not None:
            try:
                be_plan = ExecutionPlan.from_json_dict(plan_dict)
                if contract.task.name not in TASK_REGISTRY:
                    raise ValueError(f"Unknown task: {contract.task.name}")
                task = TASK_REGISTRY[contract.task.name](contract.task.params)

                print("\nExecuting BEST-EFFORT plan (NOT CERTIFIED)...\n")
                if be_plan.contract_type == "anytime":
                    _execute_anytime_plan(task, be_plan, contract, args.out, seed=(int(args.noise_seed) if args.noise_seed is not None else 2026))
                else:
                    _execute_fixed_plan(task, be_plan, contract, args.out, seed=(int(args.noise_seed) if args.noise_seed is not None else 2026))
            except Exception as e:
                print("Could not execute best-effort plan:", str(e))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
