"""Reference labels are verified against real SQLite behavior, without a model."""

import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from sae_dashboard.manifests import load_manifests
from tools.sql_injection_dataset import DATASET, MANIFESTS, audit_dataset


class SqlDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = audit_dataset()

    def test_all_reference_labels_have_successful_database_witnesses(self):
        self.assertEqual(self.report["status"], "passed", self.report["pairs"])
        self.assertEqual(self.report["pairs_by_split"], {"train": 20, "validation": 20, "test": 20})
        self.assertEqual(self.report["checks"]["successful_B_witnesses"], 60)
        self.assertEqual(self.report["checks"]["ordinary_equivalence"], 120)
        self.assertEqual(self.report["checks"]["A_quote_inputs"], 60)
        self.assertEqual(self.report["checks"]["capture_regions"], 480)

    def test_standard_loader_reads_all_assets_with_evaluator_metadata_separate(self):
        manifests = load_manifests([str(DATASET / name) for name in MANIFESTS.values()])
        self.assertEqual([len(m.pairs) for m in manifests], [20, 20, 20])
        for manifest in manifests:
            for pair in manifest.pairs:
                for condition in (pair.a, pair.b):
                    self.assertIsNone(condition.image)
                    self.assertNotIn("reference_mechanism", condition.text)
                    self.assertNotIn("template_group", condition.text)
                    self.assertNotIn("witness_inputs", condition.text)

    def test_swapping_reference_conditions_is_detected_by_behavior(self):
        with tempfile.TemporaryDirectory() as temp:
            copied = Path(temp) / "sql_injection"
            shutil.copytree(DATASET, copied)
            metadata_path = copied / "dataset_metadata.json"
            metadata = json.loads(metadata_path.read_text())
            case = metadata["cases"]["train_lookup_catalog"]
            a, b = (copied / case["files"][side] for side in ("A", "B"))
            a_content, b_content = a.read_bytes(), b.read_bytes()
            a.write_bytes(b_content)
            b.write_bytes(a_content)
            for side in ("A", "B"):
                case["sha256"][side] = hashlib.sha256((copied / case["files"][side]).read_bytes()).hexdigest()
            metadata_path.write_text(json.dumps(metadata))
            report = audit_dataset(copied)
            result = next(p for p in report["pairs"] if p["pair_id"] == "train_lookup_catalog")
            self.assertEqual(result["status"], "failed")
            self.assertEqual(report["status"], "failed")

    def test_unreviewed_asset_change_invalidates_provenance(self):
        with tempfile.TemporaryDirectory() as temp:
            copied = Path(temp) / "sql_injection"
            shutil.copytree(DATASET, copied)
            sample = copied / "train/lookup_catalog_A.txt"
            sample.write_text(sample.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "Stale metadata hash"):
                audit_dataset(copied)


if __name__ == "__main__":
    unittest.main()
