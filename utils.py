from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from .tasks import BaseTask

MIN_POS = math.nextafter(0.0, 1.0)


def run_backend_result(backend: Any, circuit: Any, shots: int, noise_model: Any = None, seed: Optional[int] = None):
    """Run an Aer backend robustly across qiskit-aer versions.

    Some versions accept noise_model in backend.run(...); others require backend.set_options(noise_model=...).
    """
    kwargs: Dict[str, Any] = {"shots": int(shots)}
    if seed is not None:
        kwargs["seed_simulator"] = int(seed)

    if noise_model is None:
        return backend.run(circuit, **kwargs).result()

    # Newer qiskit-aer accepts noise_model directly in backend.run(...)
    try:
        return backend.run(circuit, noise_model=noise_model, **kwargs).result()
    except TypeError as e:
        msg = str(e)
        # Only fall back when noise_model isn't a supported run() kwarg.
        if "noise_model" not in msg or "unexpected keyword argument" not in msg:
            raise

    # Fallback: older qiskit-aer expects noise_model set via backend options.
    prev_noise_model = None
    had_prev = False
    try:
        if hasattr(backend, "options") and hasattr(backend.options, "noise_model"):
            prev_noise_model = backend.options.noise_model
            had_prev = True
    except Exception:
        prev_noise_model = None
        had_prev = False

    try:
        if hasattr(backend, "set_options"):
            backend.set_options(noise_model=noise_model)
        return backend.run(circuit, **kwargs).result()
    finally:
        # Restore prior state so noise_model doesn't "leak" into future runs.
        if hasattr(backend, "set_options"):
            try:
                backend.set_options(noise_model=(prev_noise_model if had_prev else None))
            except Exception:
                pass



def merge_counts_int(a: Dict[str, int], b: Dict[str, int]) -> Dict[str, int]:
    """Merge two count dicts (int-valued) by summing counts."""
    out: Dict[str, int] = dict(a or {})
    for k, v in (b or {}).items():
        out[k] = int(out.get(k, 0)) + int(v)
    return out


def counts_from_probs_safe(probs: Dict[str, float], shots: int) -> Dict[str, int]:
    """
    Convert probabilities (or nonnegative weights) into integer counts with exact total `shots`.

    - Negative weights are clipped to 0.
    - If weights do not sum to 1, they are normalized.
    - Uses a robust "largest remainder" scheme that handles rare floating-point edge cases.

    Returns a dict with:
        sum(counts.values()) == shots    (for shots > 0 and non-empty probs)
    """
    shots = int(shots)
    if shots <= 0:
        return {k: 0 for k in (probs or {})}

    items = [(k, max(0.0, float(v))) for k, v in (probs or {}).items()]
    if not items:
        return {}

    s = float(sum(v for _, v in items))
    if s <= 0.0:
        return {k: 0 for k, _ in items}

    scaled = [(k, (v / s) * float(shots)) for k, v in items]
    floors: Dict[str, int] = {k: int(math.floor(x)) for k, x in scaled}
    total = int(sum(floors.values()))
    rem = int(shots - total)

    # Fractional parts for deterministic remainder distribution.
    fracs = [(float(x) - math.floor(float(x)), k) for k, x in scaled]

    if rem > 0:
        fracs_sorted = sorted(fracs, reverse=True)
        for i in range(rem):
            floors[fracs_sorted[i % len(fracs_sorted)][1]] += 1
    elif rem < 0:
        # Rare edge case (float rounding): remove 1 from smallest fractional parts (that are >0).
        need = -rem
        fracs_sorted = sorted(fracs)  # smallest first
        i = 0
        while need > 0 and i < len(fracs_sorted):
            k = fracs_sorted[i][1]
            if floors.get(k, 0) > 0:
                floors[k] -= 1
                need -= 1
            else:
                i += 1
        if need > 0:
            for k in list(floors.keys()):
                if need <= 0:
                    break
                if floors[k] > 0:
                    take = min(floors[k], need)
                    floors[k] -= take
                    need -= take

    # Final sanity: enforce nonnegativity and exact total.
    for k in list(floors.keys()):
        if floors[k] < 0:
            floors[k] = 0

    total2 = int(sum(floors.values()))
    if total2 != shots and floors:
        diff = int(shots - total2)
        if diff > 0:
            kmax = max(floors, key=lambda kk: floors[kk])
            floors[kmax] += diff
        elif diff < 0:
            diff = -diff
            for k, _ in sorted(floors.items(), key=lambda kv: kv[1], reverse=True):
                if diff <= 0:
                    break
                take = min(floors[k], diff)
                floors[k] -= take
                diff -= take

    return floors

def scale_counts_to_n(counts: Dict[str, int], target_n: int) -> Dict[str, int]:
    """
    Proportionally scale an integer-count dictionary to have exact total `target_n`.
    """
    target_n = int(target_n)
    if target_n <= 0:
        return {}
    n0 = int(sum(int(v) for v in (counts or {}).values()))
    if n0 <= 0:
        return {}
    probs = {k: float(v) / float(n0) for k, v in counts.items()}
    return counts_from_probs_safe(probs, target_n)


def window_counts(batches: List[Tuple[Dict[str, int], int]], window: int) -> Tuple[Dict[str, int], int]:
    """
    Merge the most recent *whole* batches until reaching at least `window` shots.

    Without per-shot ordering, extracting an *exact* last-W-shot window is not well-defined.
    This function avoids fabricating fractional batches (which would break concentration guarantees)
    and instead returns a conservative recent window comprised of complete trailing batches.

    batches: list of (counts_dict, shots_in_batch) in chronological order.
    Returns: (merged_counts, total_shots_used) where total_shots_used >= window unless not enough data exists.
    """
    window = int(window)
    if window <= 0:
        return {}, 0

    merged: Dict[str, int] = {}
    total = 0

    for counts, n in reversed(batches or []):
        n = int(n)
        if n <= 0:
            continue
        merged = merge_counts_int(merged, counts)
        total += n
        if total >= window:
            break

    return merged, total

def reward_stats_from_counts(task: BaseTask, counts: Dict[str, int]) -> Tuple[float, float, int, Optional[int]]:
    """Return (sum_r, sum_r2, n, successes_or_none_if_not_bernoulli)."""
    n = int(sum(int(v) for v in (counts or {}).values()))
    if n <= 0:
        return 0.0, 0.0, 0, None

    sum_r = 0.0
    sum_r2 = 0.0
    bern_ok = True
    succ = 0

    for bit, c in counts.items():
        r = float(task.reward(bit))
        c = int(c)
        sum_r += float(c) * r
        sum_r2 += float(c) * (r * r)
        if r == 1.0:
            succ += c
        elif r == 0.0:
            pass
        else:
            bern_ok = False

    return sum_r, sum_r2, n, succ if bern_ok else None