"""
Unit and regression tests for data integrity, corruption control, and statistical evaluation.
"""

import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.dataset import load_ground_truth, build_stratified_folds, extract_numeric_id
from src.parsing import parse_model_output
from src.corruption import build_b_records, build_bprime_v3
from src.metrics import bootstrap_ci, signflip_permutation_p, mcnemar_exact, clopper_pearson_ci

class TestVLMTemporalStateTracking(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
        cls.gt_path = os.path.join(cls.data_dir, "ground_truth_manifest_50.json")
        cls.split_path = os.path.join(cls.data_dir, "split_summary_seed20260917.json")
        cls.meta = load_ground_truth(cls.gt_path)

    def test_dataset_size_and_structure(self):
        self.assertEqual(len(self.meta), 50, "Expected exactly 50 video records")
        for r in self.meta:
            self.assertEqual(len(r["round_positions"]), 6, "Each video must have 6 rounds")
            self.assertIn(r["initial_position"], ["A", "B", "C", "D"])
            self.assertIn(r["final_position"], ["A", "B", "C", "D"])

    def test_starting_position_shortcut_confound(self):
        """Regression test verifying the known prompt shortcut where target_card == initial_position."""
        all_same = all(r["target_card"] == r["initial_position"] for r in self.meta)
        self.assertTrue(all_same, "100% of dataset records should exhibit the initial position shortcut")

    def test_cross_validation_folds_partition(self):
        folds = build_stratified_folds(self.meta, self.split_path)
        self.assertEqual(len(folds), 5, "Expected 5 folds")
        for f in folds:
            self.assertEqual(len(f), 10, "Each fold must contain 10 held-out videos")
        flattened = [v for f in folds for v in f]
        self.assertEqual(len(set(flattened)), 50, "Folds must form a disjoint partition of all 50 videos")

    def test_hungarian_control_invariants(self):
        """Verifies Hungarian control B' v3 invariants: format matched, never equals true sequence."""
        folds = build_stratified_folds(self.meta, self.split_path)
        train_rows = [r for r in self.meta if r["id"] not in set(folds[0])]
        b_records = build_b_records(train_rows)
        bp_records, audit, summary = build_bprime_v3(train_rows, fold_idx=0)
        
        self.assertEqual(len(bp_records), 40)
        self.assertEqual(summary["letter_freq_l1"], 0, "Letter frequencies must match exactly")
        
        for b, bp in zip(b_records, bp_records):
            # Prompt must match exactly
            self.assertEqual(b["conversations"][0]["value"], bp["conversations"][0]["value"])
            # Target must NOT be identical to true trace
            self.assertNotEqual(b["conversations"][1]["value"], bp["conversations"][1]["value"])

    def test_parser_robustness(self):
        traj_sample = "Initial: A\nRound 1: A\nRound 2: B\nRound 3: B\nRound 4: C\nRound 5: D\nFinal: D"
        p = parse_model_output(traj_sample)
        self.assertEqual(p["initial"], "A")
        self.assertEqual(p["final"], "D")
        self.assertEqual(p["rounds"][1], "A")
        self.assertEqual(p["rounds"][5], "D")

        # Test messy or verbose output
        messy_sample = "The target card started at position B, then moved from B to C. Final position: C."
        p2 = parse_model_output(messy_sample)
        self.assertEqual(p2["final"], "C")

    def test_statistical_test_functions(self):
        # McNemar test
        a = np.array([1, 1, 0, 0])
        b = np.array([1, 0, 1, 0])
        n10, n01, p_val = mcnemar_exact(a, b)
        self.assertEqual((n10, n01), (1, 1))
        self.assertEqual(p_val, 1.0)

        # Clopper Pearson
        lo, hi = clopper_pearson_ci(0, 50)
        self.assertEqual(lo, 0.0)
        self.assertAlmostEqual(hi, 0.071, places=2)

if __name__ == "__main__":
    unittest.main()
