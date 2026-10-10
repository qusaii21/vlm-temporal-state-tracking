"""
Data loading and fold generation utilities for card relocation experiment.
"""

import os
import re
import json
import random
from typing import Dict, List, Any

POSITIONS = ["A", "B", "C", "D"]

def extract_numeric_id(s: str) -> int:
    """Extract integer ID from video ID string like card_relocation_s_L1_0001_v3."""
    m = re.search(r"_(\d{4})_", s)
    return int(m.group(1)) if m else 0

def load_ground_truth(gt_path: str) -> List[Dict[str, Any]]:
    """
    Load ground truth manifest records.
    Returns list of parsed video metadata dictionaries sorted by video ID.
    """
    with open(gt_path, "r", encoding="utf-8") as f:
        gt = json.load(f)
    
    rows = []
    for vid, item in gt.items():
        ans = item["answer"]
        ans = json.loads(ans) if isinstance(ans, str) else ans
        trace = ans["position_trace"]
        keys = sorted(trace.keys(), key=int)
        pos = [trace[k] for k in keys]
        init = ans["initial_position"]
        moves, prev = [], init
        for p in pos:
            moves.append(0 if p == prev else 1)
            prev = p
        rows.append({
            "id": vid,
            "video_file": os.path.basename(item["video_path"]),
            "target_card": ans["target_card"],
            "initial_position": init,
            "round_positions": pos,
            "final_position": pos[-1],
            "move_trace": moves,
        })
    rows.sort(key=lambda r: extract_numeric_id(r["id"]))
    return rows

def build_stratified_folds(
    meta: List[Dict[str, Any]], 
    split_summary_path: str = None, 
    cv_seed: int = 20261001,
    n_folds: int = 5
) -> List[List[str]]:
    """
    Fold 0 is the original fixed 10 held-out set (from split_summary).
    Folds 1-4 are the remaining 40 videos, stratified by final position.
    """
    if split_summary_path and os.path.isfile(split_summary_path):
        with open(split_summary_path, "r", encoding="utf-8") as f:
            fold0 = list(json.load(f)["test_ids"])
    else:
        # Fallback if split summary not passed
        fold0 = [r["id"] for r in meta[:10]]
        
    rest = [r for r in meta if r["id"] not in set(fold0)]
    rng = random.Random(cv_seed)
    ordered = []
    for pos in POSITIONS:
        g = sorted([r["id"] for r in rest if r["final_position"] == pos], key=extract_numeric_id)
        rng.shuffle(g)
        ordered.extend(g)
        
    folds = [sorted(fold0, key=extract_numeric_id)] + [[] for _ in range(n_folds - 1)]
    for i, v in enumerate(ordered):
        folds[1 + i % (n_folds - 1)].append(v)
    folds = [sorted(f, key=extract_numeric_id) for f in folds]
    return folds
