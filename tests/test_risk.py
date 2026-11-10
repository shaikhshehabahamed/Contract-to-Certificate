import math
import unittest

from qqos.risk import (
    bernoulli_mixture_cs_lower_mean_bound,
    hoeffding_lower_mean_bound,
    hoeffding_required_shots,
    split_delta,
)


class RiskBoundTests(unittest.TestCase):
    def test_bernoulli_mixture_all_successes_is_conservative(self) -> None:
        # With a Jeffreys Beta(1/2, 1/2) mixture and one success, the marginal
        # likelihood is 1/2, so the lower endpoint is delta / 2.
        got = bernoulli_mixture_cs_lower_mean_bound(successes=1, n=1, delta=0.05)
        self.assertTrue(math.isclose(got, 0.025, rel_tol=1e-12, abs_tol=1e-12))

    def test_hoeffding_required_shots_matches_tail_bound(self) -> None:
        n = hoeffding_required_shots(mu_lb=0.8, qmin=0.7, delta=1e-3)
        self.assertIsNotNone(n)
        self.assertGreaterEqual(int(n or 0), 346)

    def test_split_delta_is_exact(self) -> None:
        a, b = split_delta(0.001, 0.25)
        self.assertTrue(math.isclose(a + b, 0.001, rel_tol=0.0, abs_tol=1e-18))
        self.assertTrue(math.isclose(b, 0.00025, rel_tol=0.0, abs_tol=1e-18))

    def test_zero_delta_lower_bound_is_maximally_conservative(self) -> None:
        self.assertEqual(hoeffding_lower_mean_bound(sample_mean=1.0, n=10, delta=0.0), float("-inf"))


if __name__ == "__main__":
    unittest.main()
