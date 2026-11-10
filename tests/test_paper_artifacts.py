import hashlib
import tempfile
import unittest
from pathlib import Path

from scripts import build_paper_artifacts as bpa


class PaperArtifactTests(unittest.TestCase):
    def test_manifest_records_hashes_and_source_contracts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "examples" / "contracts" / "demo.json"
            generated = root / "paper_artifacts" / "tables" / "demo.csv"
            source.parent.mkdir(parents=True)
            generated.parent.mkdir(parents=True)
            source.write_text('{"demo": true}\n', encoding="utf-8")
            generated.write_text("a,b\n1,2\n", encoding="utf-8")

            manifest = bpa._manifest(
                [generated],
                [{"case_id": "demo", "label": "Demo", "contract": "examples/contracts/demo.json"}],
                root=root,
            )

            rel = "paper_artifacts/tables/demo.csv"
            expected_generated_hash = hashlib.sha256(generated.read_bytes()).hexdigest()
            expected_source_hash = hashlib.sha256(source.read_bytes()).hexdigest()

            self.assertEqual(manifest["artifact_schema_version"], 2)
            self.assertIn(rel, manifest["generated_files"])
            self.assertEqual(manifest["generated_file_hashes"][rel], expected_generated_hash)
            self.assertEqual(manifest["source_contracts"][0]["sha256"], expected_source_hash)
            self.assertIn("dependency_versions", manifest)

    def test_task_shape_describes_ghz_and_qaoa_rows(self) -> None:
        ghz = bpa._task_shape("ghz_success", {"n_qubits": 5})
        qaoa = bpa._task_shape("qaoa_maxcut", {"n_nodes": 4, "p": 1})

        self.assertEqual(ghz["workload_size"], "5 qubits")
        self.assertEqual(qaoa["workload_size"], "4 nodes, p=1")
        self.assertEqual(qaoa["quality_metric"], "normalized MaxCut reward")


if __name__ == "__main__":
    unittest.main()
