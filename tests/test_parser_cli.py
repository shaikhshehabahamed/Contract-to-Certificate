import json
import subprocess
import sys
import tempfile
import unittest

from qqos.contract import Budget, NoiseSpec, QQoSContract, RiskPolicy, TaskSpec
from qqos.parser import load_contract, validate_contract


class ParserCliTests(unittest.TestCase):
    def test_cli_version_runs(self) -> None:
        proc = subprocess.run(
            [sys.executable, "-m", "qqos", "--version"],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("python -m qqos", proc.stdout)

    def test_cli_sat_writes_result_without_execute(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            contract_path = f"{tmp}/sat_contract.json"
            plan_path = f"{tmp}/plan.json"
            result_path = f"{tmp}/result.json"
            with open(contract_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "task": {"name": "ghz_success", "params": {"n_qubits": 3}},
                        "qmin": 0.1,
                        "delta": 0.1,
                        "budget": {"shots_max": 1000, "depth_max": 100},
                        "risk_policy": {
                            "method": "hoeffding",
                            "pilot_shots": 64,
                            "pilot_delta": 0.05,
                            "pilot_correction": "split",
                        },
                        "noise": {"p1": 0.001, "p2": 0.01, "p_ro": 0.02},
                    },
                    f,
                )

            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "qqos",
                    "--contract",
                    contract_path,
                    "--out_plan",
                    plan_path,
                    "--out",
                    result_path,
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            with open(result_path, "r", encoding="utf-8") as f:
                result = json.load(f)
            self.assertTrue(result["sat"])
            self.assertIn("plan", result)

    def test_load_contract_rejects_invalid_noise_probability(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/bad_contract.json"
            with open(path, "w", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "task": {"name": "ghz_success", "params": {"n_qubits": 3}},
                            "qmin": 0.8,
                            "delta": 0.001,
                            "budget": {"shots_max": 1000, "depth_max": 100},
                            "noise": {"p1": -0.01},
                        }
                    )
                )
            with self.assertRaisesRegex(ValueError, "noise.p1"):
                load_contract(path)

    def test_validate_contract_rejects_missing_quality_target(self) -> None:
        contract = QQoSContract(
            task=TaskSpec(name="ghz_success", params={"n_qubits": 3}),
            qmin=None,
            eps_max=None,
            delta=0.001,
            budget=Budget(shots_max=1000, depth_max=100),
            risk_policy=RiskPolicy(),
            noise=NoiseSpec(),
        )
        with self.assertRaisesRegex(ValueError, "qmin|eps_max"):
            validate_contract(contract)

    def test_validate_contract_rejects_inconsistent_quality_targets(self) -> None:
        contract = QQoSContract(
            task=TaskSpec(name="ghz_success", params={"n_qubits": 3}),
            qmin=0.8,
            eps_max=0.1,
            delta=0.001,
            budget=Budget(shots_max=1000, depth_max=100),
            risk_policy=RiskPolicy(),
            noise=NoiseSpec(),
        )
        with self.assertRaisesRegex(ValueError, "qmin and eps_max disagree"):
            validate_contract(contract)

    def test_validate_contract_rejects_calibrated_box_without_calibration_shots(self) -> None:
        contract = QQoSContract(
            task=TaskSpec(name="ghz_success", params={"n_qubits": 3}),
            qmin=0.8,
            delta=0.001,
            budget=Budget(shots_max=1000, depth_max=100),
            risk_policy=RiskPolicy(),
            noise=NoiseSpec(uncertainty_model="calibrated_box", calibration_shots=0),
        )
        with self.assertRaisesRegex(ValueError, "calibration_shots"):
            validate_contract(contract)


if __name__ == "__main__":
    unittest.main()
