from __future__ import annotations

import unittest

from backend.services.model_registry import list_models, model_paths


class ModelRegistryAuditTests(unittest.TestCase):
    def test_models_report_checkpoint_lineage_without_local_paths(self) -> None:
        models = list_models()
        self.assertEqual(len(models), 3)
        for item in models:
            self.assertEqual(item["status"], "available")
            self.assertEqual(item["dataset_version"], "3")
            self.assertEqual(item["input_dimensions"], [32, 32])
            self.assertIn("REAL", item["class_mapping"])
            self.assertEqual(item["validation_status"], "cifake_v3_held_out_test_evaluated")
            self.assertEqual(set(item["checkpoint"]), {"id", "epoch", "size_bytes"})
            self.assertNotIn("path", item["checkpoint"])

    def test_registry_checkpoint_is_selected_evaluation_checkpoint(self) -> None:
        for item in list_models():
            _, _, checkpoint_path, _ = model_paths(item["id"])
            self.assertTrue(checkpoint_path.is_file())
            self.assertEqual(checkpoint_path.parent.name, item["run_id"])
            self.assertEqual(checkpoint_path.stat().st_size, item["checkpoint"]["size_bytes"])


if __name__ == "__main__":
    unittest.main()
