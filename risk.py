from __future__ import annotations

import math

MIN_POS = math.nextafter(0.0, 1.0)

from dataclasses import dataclass
from statistics import NormalDist
from typing import Optional, Tuple

try:
    # Optional exact Beta quantiles for Clopper–Pearson bounds
    from scipy.stats import beta as _beta  # type: ignore
except Exception:  # pragma: no cover
    _beta = None


@dataclass(frozen=True)
class RiskBound:
    method: str
    delta: float
    guarantee: str
    details: dict


# ---------------------------------------------------------------------
# Fixed-time (non-anytime) bounds
# ---------------------------------------------------------------------

def hoeffding_lower_mean_bound(
    sample_mean: float,
    n: int,
    delta: float,
    value_range: Tuple[float, float] = (0.0, 1.0),
) -> float:
    """Lower bound on the true mean using fixed-time Hoeffding.

    If delta underflows to 0.0, we return -inf (maximally conservative) rather than raising.
    """
    if n <= 0:
        return float("-inf")
    d = float(delta)
    if d <= 0.0:
        return float("-inf")
    if not (d < 1.0):
        raise ValueError("delta must be in (0,1)")
    a, b = value_range
    width = float(b - a)
    if width <= 0:
        raise ValueError("Invalid value_range")

    log_inv = -math.log(max(d, MIN_POS))  # stable log(1/d)
    eps = math.sqrt((width * width) * log_inv / (2.0 * float(n)))
    return float(sample_mean) - float(eps)


def empirical_bernstein_lower_mean_bound(
    sample_mean: float,
    sample_var: float,
    n: int,
    delta: float,
    value_range: Tuple[float, float] = (0.0, 1.0),
) -> float:
    """Lower confidence bound via an empirical Bernstein inequality (fixed-time).

    Uses a common (variance-adaptive) form:
      L = mean - sqrt( 2 * V * log(3/delta) / n ) - 3 * (b-a) * log(3/delta) / (n-1)

    Notes:
      - Valid for bounded rewards in [a,b] and i.i.d. sampling (fixed-time).
      - If n <= 1, falls back to -inf (no useful bound).
    """
    if n <= 1:
        return float("-inf")
    d = float(delta)
    if d <= 0.0:
        return float("-inf")
    if not (d < 1.0):
        raise ValueError("delta must be in (0,1)")
    a, b = value_range
    width = float(b - a)
    if width <= 0:
        raise ValueError("Invalid value_range")

    v = max(0.0, float(sample_var))
    logterm = math.log(3.0) - math.log(max(d, MIN_POS))  # stable log(3/d)
    rad = math.sqrt(2.0 * v * logterm / float(n))
    bias = 3.0 * width * logterm / float(n - 1)
    return float(sample_mean) - float(rad) - float(bias)


def hoeffding_required_shots(
    mu_lb: float,
    qmin: float,
    delta: float,
    value_range: Tuple[float, float] = (0.0, 1.0),
    safety_floor_gap: float = 1e-6,
) -> Optional[int]:
    """Return N s.t. P(sample_mean < qmin) <= delta assuming true mean >= mu_lb."""
    if not (0.0 < float(delta) < 1.0):
        raise ValueError("delta must be in (0,1)")
    a, b = value_range
    width = float(b - a)
    gap = float(mu_lb - qmin)
    if gap <= float(safety_floor_gap):
        return None
    n = math.log(1.0 / float(delta)) * (width * width) / (2.0 * gap * gap)
    return int(math.ceil(n))


def hoeffding_tail_risk(
    mu_lb: float,
    qmin: float,
    n: int,
    value_range: Tuple[float, float] = (0.0, 1.0),
) -> float:
    """Upper bound on P(sample_mean < qmin) assuming true mean >= mu_lb."""
    if n <= 0:
        return 1.0
    a, b = value_range
    width = float(b - a)
    gap = float(mu_lb - qmin)
    if gap <= 0:
        return 1.0
    return float(math.exp(-2.0 * float(n) * gap * gap / (width * width)))


# ---------------------------------------------------------------------
# Anytime-valid (time-uniform) bounds / confidence sequences
# ---------------------------------------------------------------------

def _alpha_n(delta_total: float, n: int) -> float:
    """Per-check alpha such that sum_n alpha_n <= delta_total.

    Uses an inverse-square schedule:
        alpha_n = delta_total * 6/pi^2 * 1/n^2

    We intentionally do **not** floor alpha upward: flooring would overspend risk
    across an unbounded number of checks. If alpha underflows to 0.0, callers
    should handle it conservatively (e.g., by returning -inf bounds).
    """
    d = float(delta_total)
    if d <= 0.0:
        return 0.0
    if n <= 0:
        return float(d)
    c = 6.0 / (math.pi * math.pi)
    a = float(d) * c / (float(n) * float(n))
    return float(min(0.999999999, max(0.0, a)))


def _clopper_pearson_lower(k: int, n: int, alpha: float) -> float:
    """One-sided (1-alpha) Clopper–Pearson lower bound for Bernoulli mean."""
    if n <= 0:
        return 0.0
    k = int(max(0, min(n, k)))
    alpha = float(min(0.999999999, max(MIN_POS, alpha)))

    if _beta is not None:
        if k == 0:
            return 0.0
        return float(_beta.ppf(alpha, k, n - k + 1))

    # Normal fallback (approximate)
    phat = float(k) / float(n)
    z = NormalDist().inv_cdf(1.0 - alpha)
    rad = z * math.sqrt(max(0.0, phat * (1.0 - phat) / float(n)))
    return float(max(0.0, phat - rad))


def _normal_mixture_boundary(v: float, alpha: float, rho: float = 1.0) -> float:
    """One-sided normal-mixture uniform boundary for sub-Gaussian processes.

    For a process with MGF bound:
        E exp(λ S_t - λ^2 v/2) <= 1   for all λ in R
    the normal-mixture supermartingale yields the boundary b(v) defined by:
        (1 / sqrt(1 + rho v)) * exp( rho s^2 / (2(1 + rho v)) ) = 1/alpha

    Solving gives:
        b(v) = sqrt( (1 + rho v)/rho * (2 log(1/alpha) + log(1 + rho v)) ).

    Notes:
      - rho > 0 is a tuning parameter (fixed in advance). rho=1 is a safe default.
      - v is the intrinsic time (variance proxy), not the number of samples.
    """
    v = float(max(0.0, v))
    alpha = float(min(0.999999999, max(MIN_POS, alpha)))
    rho = float(max(1e-12, rho))
    term = 2.0 * math.log(1.0 / alpha) + math.log1p(rho * v)
    return float(math.sqrt((1.0 + rho * v) / rho * term))


def cs_normal_mixture_lower_mean_bound(
    sample_mean: float,
    n: int,
    delta: float,
    value_range: Tuple[float, float] = (0.0, 1.0),
    rho: float = 1.0,
) -> float:
    """Time-uniform lower bound for bounded rewards via a normal-mixture CS.

    Assumes bounded X_i in [a,b]. The bound is *anytime-valid* under standard
    conditional sub-Gaussian / martingale assumptions (e.g., bounded increments
    with a conditional Hoeffding-lemma MGF bound), which is the right setting
    for online, adaptive execution.

    Returns L_n such that P(∀n: μ >= L_n) >= 1 - delta.
    """
    if n <= 0:
        return float("-inf")
    if not (0.0 < float(delta) < 1.0):
        raise ValueError("delta must be in (0,1)")
    a, b = value_range
    width = float(b - a)
    if width <= 0:
        raise ValueError("Invalid value_range")

    # Scale into [0,1] for a clean variance proxy.
    m = float(min(b, max(a, sample_mean)))
    m01 = (m - float(a)) / width

    # For [0,1]-bounded increments, sigma^2 = 1/4.
    sigma2 = 0.25
    v = float(n) * sigma2
    bnd = _normal_mixture_boundary(v=v, alpha=float(delta), rho=float(rho))
    L01 = m01 - bnd / float(n)
    L01 = min(1.0, max(0.0, L01))
    return float(a) + width * float(L01)


def bernoulli_mixture_cs_lower_mean_bound(
    successes: int,
    n: int,
    delta: float,
    a: float = 0.5,
    b: float = 0.5,
    tol: float = 1e-9,
    max_iter: int = 80,
) -> float:
    """Anytime-valid (Ville) lower bound for Bernoulli mean via Beta-mixture inversion.

    Define the Bayes factor / mixture likelihood ratio against a point null p0:
        BF_t(p0) = m(data) / L(data | p0)
    where m(data) is the Beta(a,b) marginal likelihood.

    Under p0, BF_t(p0) is a nonnegative martingale with mean 1, so by Ville:
        P_{p0}(∃t: BF_t(p0) >= 1/delta) <= delta.

    Inverting the test yields a confidence sequence:
        CS_t = { p0 : BF_t(p0) < 1/delta }
    and the returned bound is inf CS_t (restricted to [0, p_hat]).

    This is typically tighter than union-bound CP, especially when checks are frequent.
    """
    if n <= 0:
        return 0.0
    if not (0.0 < float(delta) < 1.0):
        raise ValueError("delta must be in (0,1)")

    k = int(max(0, min(int(successes), int(n))))
    if k <= 0:
        return 0.0
    if k >= n:
        # All successes -> lower bound is still < 1; return a conservative bound near 1.
        return float(max(0.0, 1.0 - (delta ** (1.0 / max(1.0, n)))))

    # log Beta(x,y) = lgamma(x)+lgamma(y)-lgamma(x+y)
    def log_beta(x: float, y: float) -> float:
        return math.lgamma(x) + math.lgamma(y) - math.lgamma(x + y)

    log_marg = log_beta(k + a, (n - k) + b) - log_beta(a, b)
    log_thr = math.log(1.0 / float(delta))

    phat = float(k) / float(n)
    hi = max(1e-12, min(phat, 1.0 - 1e-12))
    lo = 1e-12

    def log_bf(p0: float) -> float:
        p0 = float(min(1.0 - 1e-15, max(1e-15, p0)))
        return log_marg - (k * math.log(p0) + (n - k) * math.log(1.0 - p0))

    # If even at p_hat BF >= threshold, the set may be empty; fall back.
    if log_bf(hi) >= log_thr:
        return 0.0

    # If at lo BF < threshold, bound is ~0.
    if log_bf(lo) < log_thr:
        return 0.0

    for _ in range(int(max_iter)):
        mid = 0.5 * (lo + hi)
        if log_bf(mid) >= log_thr:
            lo = mid
        else:
            hi = mid
        if abs(hi - lo) <= tol * max(1.0, hi):
            break
    return float(max(0.0, min(1.0, hi)))


def anytime_lower_bound(
    method: str,
    sample_mean: float,
    n: int,
    delta_total: float,
    t: Optional[int] = None,
    value_range: Tuple[float, float] = (0.0, 1.0),
    sample_var: Optional[float] = None,
    successes: Optional[int] = None,
) -> float:
    """Return a lower confidence sequence L such that P(∀checks: μ >= L) >= 1 - delta_total."""
    method = str(method)
    if n <= 0:
        return float("-inf")
    dt = float(delta_total)
    if dt <= 0.0:
        return float("-inf")
    if not (dt < 1.0):
        raise ValueError("delta_total must be in (0,1)")

    # Clamp mean to the declared range (helps with numerical issues)
    a, b = value_range
    m = float(min(b, max(a, float(sample_mean))))

    time_index = int(t) if t is not None else int(n)

    # --- Modern time-uniform confidence sequences (no union schedule needed) ---
    if method in ("cs_normal_mixture", "mixture_cs"):
        return float(cs_normal_mixture_lower_mean_bound(m, int(n), float(delta_total), value_range=value_range))

    if method == "bernoulli_mixture_cs":
        if successes is not None and value_range == (0.0, 1.0):
            return float(bernoulli_mixture_cs_lower_mean_bound(int(successes), int(n), float(delta_total)))
        # If reward isn't Bernoulli, fall back to bounded CS.
        return float(cs_normal_mixture_lower_mean_bound(m, int(n), float(delta_total), value_range=value_range))

    # --- Union-schedule variants (simple, conservative, anytime-valid) ---
    if method == "anytime_hoeffding":
        alpha = _alpha_n(float(dt), int(time_index))
        return float(hoeffding_lower_mean_bound(m, int(n), alpha, value_range=value_range))

    if method == "anytime_empirical_bernstein":
        alpha = _alpha_n(float(dt), int(time_index))
        if sample_var is not None and n > 1:
            return float(
                empirical_bernstein_lower_mean_bound(
                    sample_mean=m,
                    sample_var=float(sample_var),
                    n=int(n),
                    delta=float(alpha),
                    value_range=value_range,
                )
            )
        return float(hoeffding_lower_mean_bound(m, int(n), alpha, value_range=value_range))

    if method == "betting_cs":
        # Legacy Bernoulli CS: CP + union schedule
        if successes is not None and value_range == (0.0, 1.0):
            alpha = _alpha_n(float(dt), int(time_index))
            return float(_clopper_pearson_lower(int(successes), int(n), alpha))
        alpha = _alpha_n(float(dt), int(time_index))
        return float(hoeffding_lower_mean_bound(m, int(n), alpha, value_range=value_range))

    # --- Fixed-time methods (not time-uniform) ---
    if method == "empirical_bernstein":
        if sample_var is not None:
            return float(
                empirical_bernstein_lower_mean_bound(
                    m, float(sample_var), int(n), float(delta_total), value_range=value_range
                )
            )
        return float(hoeffding_lower_mean_bound(m, int(n), float(delta_total), value_range=value_range))

    # Default: fixed-time Hoeffding
    return float(hoeffding_lower_mean_bound(m, int(n), float(delta_total), value_range=value_range))
# ---------------------------------------------------------------------
# Restartable delta spending (for drift-aware certificates)
# ---------------------------------------------------------------------

def restart_delta(
    delta_total: float,
    restart_index: int,
    gamma: float = 0.2,
) -> float:
    """Allocate a slice of delta for a given restart/segment.

    Geometric spending:
        delta_r = (1 - gamma) * gamma^r * delta_total

    so that sum_{r>=0} delta_r = delta_total.

    We do **not** floor the returned value upward: flooring would overspend risk
    across an unbounded number of restarts. If the computed value underflows to
    0.0, downstream computations should clamp only inside log() calls.
    """
    d_total = float(delta_total)
    if d_total <= 0.0:
        return 0.0
    if not (d_total < 1.0):
        raise ValueError("delta_total must be in (0,1)")
    r = int(max(0, restart_index))
    g = float(gamma)
    if not (0.0 < g < 1.0):
        raise ValueError("gamma must be in (0,1)")
    d = float(d_total) * (1.0 - g) * (g ** float(r))
    return float(min(0.999999999, max(0.0, d)))

def split_delta(delta_total: float, frac: float) -> tuple[float, float]:
    """Split delta_total into (delta_a, delta_b) with delta_b = frac * delta_total.

    This split is exact (delta_a + delta_b == delta_total) and does not introduce artificial floors
    that could overspend risk.
    """
    d_total = float(delta_total)
    if d_total < 0.0:
        raise ValueError("delta_total must be >= 0")
    if not (d_total < 1.0):
        raise ValueError("delta_total must be in [0,1)")
    if d_total == 0.0:
        return 0.0, 0.0

    f = float(frac)
    if not (0.0 <= f <= 1.0):
        raise ValueError("frac must be in [0,1]")
    db = float(d_total) * f
    da = float(d_total) - db
    return float(da), float(db)