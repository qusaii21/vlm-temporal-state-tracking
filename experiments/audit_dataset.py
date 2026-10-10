"""
Audit dataset to verify:
1. Target card identity equals initial position (the prompt shortcut).
2. Starting position distribution and move distribution across rounds.
3. Stay-put baseline theoretical expectation.
"""

import os
import sys
import json
from collections import Counter

# Add repo root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.dataset import load_ground_truth

def run_dataset_audit():
    gt_path = os.path.join(os.path.dirname(__file__), "..", "data", "ground_truth_manifest_50.json")
    if not os.path.isfile(gt_path):
        print(f"Error: {gt_path} not found.")
        return

    meta = load_ground_truth(gt_path)
    print(f"Auditing {len(meta)} video records from: {gt_path}")
    print("-" * 65)

    same_init_count = sum(1 for r in meta if r["target_card"] == r["initial_position"])
    print(f"Shortcut Check: target_card == initial_position in {same_init_count}/{len(meta)} videos ({same_init_count/len(meta)*100:.1f}%)")
    print("  -> Crucial finding: Prompt reveals initial position directly!")

    total_rounds = 0
    target_at_start = 0
    target_moved_away = 0
    final_matches_start = 0

    for r in meta:
        init = r["initial_position"]
        if r["final_position"] == init:
            final_matches_start += 1
        for p in r["round_positions"]:
            total_rounds += 1
            if p == init:
                target_at_start += 1
            else:
                target_moved_away += 1

    print("\nState Distribution across 6 rounds:")
    print(f"  Total (video, round) instances: {total_rounds}")
    print(f"  Target at initial position:    {target_at_start} ({target_at_start/total_rounds:.3f})")
    print(f"  Target moved away:             {target_moved_away} ({target_moved_away/total_rounds:.3f})")
    print(f"  Final position == initial:     {final_matches_start}/{len(meta)} ({final_matches_start/len(meta):.3f})")

    print("\nTheoretical Baselines:")
    print("  Uniform Random Chance:      25.0%")
    print(f"  Stay-put Policy Per-Round:  {target_at_start/total_rounds*100:.1f}%")
    print(f"  Stay-put Policy Final:      {final_matches_start/len(meta)*100:.1f}%")
    print("-" * 65)

if __name__ == "__main__":
    run_dataset_audit()
