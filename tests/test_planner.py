import unittest
from unittest.mock import patch

from qqos import sat_checker
from qqos.contract import Budget, NoiseSpec, QQoSContract, RiskPolicy, TaskSpec


class PlannerTests(unittest.TestCase):
    def test_box_uncertainty_evaluates_all_corners(self) -> None:
        one_candidate = [
            {
                "optimization_level": 0,
                "readout_mitigation": False,
                "simulator_method": "density_matrix",
            }
        ]

        contract = QQoSContract(
            task=TaskSpec(name="ghz_success", params={"n_qubits": 3}),
            qmin=0.2,
            delta=0.001,
            budget=Budget(shots_max=10_000, depth_max=100),
            risk_policy=RiskPolicy(
                method="hoeffding",
                pilot_shots=512,
                pilot_delta=0.0005,
                pilot_correction="split",
                pilot_selection_fraction=0.5,
            ),
            noise=NoiseSpec(
                p1=0.001,
                p2=0.01,
                p_ro=0.02,
                uncertainty_model="box",
                uncertainty_scale=0.2,
            ),
        )

        with patch.object(sat_checker, "_candidate_settings", return_value=one_candidate):
            result = sat_checker.check_and_plan(contract)

        self.assertTrue(result.sat)
        self.assertIsNotNone(result.plan)
        details = result.plan.risk_bound.details if result.plan is not None else {}
        self.assertEqual(details["noise_corner_evaluation"], "all_corners")
        self.assertEqual(details["noise_corners_total"], 8)
        self.assertEqual(len(details["inference_corner_results"]), 8)
        self.assertIn("noise_certificate_corner", details)

    def test_calibrated_box_reports_end_to_end_risk_accounting(self) -> None:
        one_candidate = [
            {
                "optimization_level": 0,
                "readout_mitigation": False,
                "simulator_method": "density_matrix",
            }
        ]
        corner = NoiseSpec(p1=0.001, p2=0.01, p_ro=0.02)

        contract = QQoSContract(
            task=TaskSpec(name="ghz_success", params={"n_qubits": 3}),
            qmin=0.1,
            delta=0.01,
            budget=Budget(shots_max=10_000, depth_max=100),
            risk_policy=RiskPolicy(
                method="hoeffding",
                pilot_shots=64,
                pilot_delta=0.005,
                pilot_correction="split",
                pilot_selection_fraction=0.5,
            ),
            noise=NoiseSpec(
                p1=0.001,
                p2=0.01,
                p_ro=0.02,
                uncertainty_model="calibrated_box",
                calibration_shots=1,
                calibration_delta=0.02,
            ),
        )

        with patch.object(sat_checker, "_candidate_settings", return_value=one_candidate):
            with patch.object(sat_checker, "_resolve_noise_box", return_value=(None, [corner], corner, 4)):
                result = sat_checker.check_and_plan(contract)

        self.assertTrue(result.sat)
        self.assertIsNotNone(result.plan)
        details = result.plan.risk_bound.details if result.plan is not None else {}
        accounting = details["risk_accounting"]
        self.assertAlmostEqual(accounting["statistical_delta"], 0.01)
        self.assertAlmostEqual(accounting["noise_calibration_delta"], 0.02)
        self.assertAlmostEqual(accounting["end_to_end_delta_bound"], 0.03)
        self.assertEqual(details["calibration_shots"]["noise"], 4)
        self.assertIn("calibrated finite noise box", result.plan.risk_bound.guarantee)

    def test_qaoa_reward_uses_empirical_bernstein_fixed_time_plan(self) -> None:
        one_candidate = [
            {
                "optimization_level": 0,
                "readout_mitigation": False,
                "simulator_method": "density_matrix",
            }
        ]

        contract = QQoSContract(
            task=TaskSpec(
                name="qaoa_maxcut",
                params={"n_nodes": 4, "p": 1, "gamma": 0.8, "beta": 0.6},
            ),
            qmin=0.3,
            delta=0.01,
            budget=Budget(shots_max=10_000, depth_max=200),
            risk_policy=RiskPolicy(
                method="empirical_bernstein",
                pilot_shots=512,
                pilot_delta=0.005,
                pilot_correction="split",
                pilot_selection_fraction=0.5,
            ),
            noise=NoiseSpec(
                p1=0.001,
                p2=0.01,
                p_ro=0.02,
                uncertainty_model="scale",
                uncertainty_scale=0.2,
            ),
        )

        with patch.object(sat_checker, "_candidate_settings", return_value=one_candidate):
            result = sat_checker.check_and_plan(contract)

        self.assertTrue(result.sat)
        self.assertIsNotNone(result.plan)
        plan = result.plan
        details = plan.risk_bound.details
        self.assertEqual(plan.task_name, "qaoa_maxcut")
        self.assertEqual(plan.risk_bound.method, "empirical_bernstein")
        self.assertGreaterEqual(plan.mu_lower_bound, contract.resolved_qmin())
        self.assertGreater(details["required_shots"], 0)
        self.assertIsNone(details["inference_corner_results"][0]["successes"])

    def test_planner_drops_estimate_only_readout_mitigation_overhead(self) -> None:
        equivalent_candidates = [
            {
                "optimization_level": 0,
                "readout_mitigation": True,
                "simulator_method": "density_matrix",
            },
            {
                "optimization_level": 0,
                "readout_mitigation": False,
                "simulator_method": "density_matrix",
            },
        ]

        contract = QQoSContract(
            task=TaskSpec(name="ghz_success", params={"n_qubits": 3}),
            qmin=0.1,
            delta=0.01,
            budget=Budget(shots_max=10_000, depth_max=100),
            risk_policy=RiskPolicy(
                method="hoeffding",
                pilot_shots=512,
                pilot_delta=0.005,
                pilot_correction="bonferroni",
            ),
            noise=NoiseSpec(p1=0.001, p2=0.01, p_ro=0.02),
        )

        with patch.object(sat_checker, "_candidate_settings", return_value=equivalent_candidates):
            with patch.object(sat_checker, "_run_corner_pilot", return_value=(1.0, 0.0, None)):
                result = sat_checker.check_and_plan(contract)

        self.assertTrue(result.sat)
        self.assertIsNotNone(result.plan)
        plan = result.plan
        self.assertFalse(plan.readout_mitigation)
        self.assertEqual(plan.calibration_shots, 0)
        policy = plan.risk_bound.details["readout_mitigation_policy"]
        self.assertTrue(policy["certificate_uses_raw_counts"])
        self.assertTrue(policy["dropped_estimate_only_equivalent"])


if __name__ == "__main__":
    unittest.main()
