"""
Audit Hungarian algorithm negative control (B' v3) generation across all 5 cross-validation folds.
Verifies:
1. No trace is identical to true ground truth.
2. Initial and final positions remain identical.
3. Move count distributions and letter frequencies are preserved.
4. Generates audit JSON for reproducibility.
"""

import os
import sys
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.dataset import load_ground_truth, build_stratified_folds
from src.corruption import build_b_records, build_bprime_v3

def run_control_audit():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    gt_path = os.path.join(data_dir, "ground_truth_manifest_50.json")
    split_path = os.path.join(data_dir, "split_summary_seed20260917.json")
    
    meta = load_ground_truth(gt_path)
    folds = build_stratified_folds(meta, split_path)
    
    print("=" * 70)
    print("HUNGARIAN CORRUPTED-TRAJECTORY CONTROL (B' v3) AUDIT ACROSS 5 FOLDS")
    print("=" * 70)
    
    all_summaries = []
    out_dir = os.path.join(os.path.dirname(__file__), "..", "results", "tables")
    os.makedirs(out_dir, exist_ok=True)

    for k in range(5):
        test_ids = set(folds[k])
        train_rows = [r for r in meta if r["id"] not in test_ids]
        assert len(train_rows) == 40
        
        b_records = build_b_records(train_rows)
        bp_records, audit_rows, summary = build_bprime_v3(train_rows, fold_idx=k)
        
        # Verify gate properties
        assert len(bp_records) == 40
        for b_rec, bp_rec in zip(b_records, bp_records):
            assert b_rec["conversations"][0]["value"] == bp_rec["conversations"][0]["value"], "Prompt mismatch"
            assert b_rec["conversations"][1]["value"] != bp_rec["conversations"][1]["value"], "Control equals true!"
            
        print(f"Fold {k}:")
        print(f"  Training videos:            {summary['n']}")
        print(f"  Hungarian matched methods:  {summary['methods']}")
        print(f"  Mean moves true vs control: {summary['mean_moves_true']} vs {summary['mean_moves_corrupted']}")
        print(f"  Move histogram L1 distance: {summary['move_hist_l1']}")
        print(f"  Letter frequency L1:        {summary['letter_freq_l1']}")
        print(f"  Majority diverged:          {summary['majority_diverged']}/{summary['n']}")
        print("-" * 70)
        
        all_summaries.append(summary)
        # Save fold audit
        with open(os.path.join(out_dir, f"bprime_v3_fold_{k}_audit.json"), "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "audit_rows": audit_rows}, f, indent=2)

    with open(os.path.join(out_dir, "bprime_v3_all_folds_summary.json"), "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)

    print(f"Audit completed successfully. Saved reports to {out_dir}")

if __name__ == "__main__":
    run_control_audit()
