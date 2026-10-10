"""
Format-matched Hungarian algorithm corrupted-trajectory negative control (B' v3).

Constructs negative control training traces that preserve output syntax, move count distributions,
and letter frequencies, but borrow intermediate rounds from other videos to sever temporal correspondence.
"""

import random
from collections import Counter
from typing import List, Dict, Tuple, Any
import numpy as np
from scipy.optimize import linear_sum_assignment

from src.prompts import prompt_trajectory, render_trajectory, make_conversation_record

POSITIONS = ["A", "B", "C", "D"]
INF_COST = 10 ** 6
MAX_DERANGE_TRIES = 500

def move_count_of(initial: str, seq: List[str]) -> int:
    """Calculate the number of positional transitions."""
    prev, n = initial, 0
    for p in seq:
        n += int(p != prev)
        prev = p
    return n

def build_b_records(train_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build Condition B (True Trajectory) supervision records."""
    return [
        make_conversation_record(
            r,
            prompt_trajectory(r["target_card"]),
            render_trajectory(r["initial_position"], r["round_positions"], r["final_position"])
        )
        for r in train_rows
    ]

def build_bprime_v3(
    train_rows: List[Dict[str, Any]], 
    fold_idx: int = 0, 
    base_seed: int = 56182047
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Build Condition B' v3 (Corrupted Trajectory Control) supervision records.
    Uses Hungarian matching on move counts between training videos.
    """
    rng = random.Random(base_seed + fold_idx)

    def derange(seq):
        n = len(seq)
        if n <= 1:
            return None
        idx = list(range(n))
        for _ in range(MAX_DERANGE_TRIES):
            rng.shuffle(idx)
            if all(seq[idx[i]] != seq[i] for i in range(n)):
                return [seq[idx[i]] for i in range(n)]
        return None

    def substitute(seq):
        return [rng.choice([p for p in POSITIONS if p != v]) for v in seq]

    groups = {}
    for r in train_rows:
        n_int = len(r["round_positions"]) - 1
        if n_int <= 0:
            raise RuntimeError(f"{r['id']}: no intermediate rounds, cannot corrupt")
        groups.setdefault(n_int, []).append(r)

    records, audit, methods = [], [], Counter()
    for n_int, group in sorted(groups.items()):
        col, valid = None, None
        if len(group) >= 2:
            n = len(group)
            true_seqs = [g["round_positions"][:-1] for g in group]
            inits = [g["initial_position"] for g in group]
            finals = [g["final_position"] for g in group]
            true_mv = [move_count_of(inits[i], group[i]["round_positions"]) for i in range(n)]
            cost = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    if i == j or true_seqs[j] == true_seqs[i]:
                        cost[i, j] = INF_COST
                    else:
                        cost[i, j] = abs(true_mv[i] - move_count_of(inits[i], true_seqs[j] + [finals[i]]))
            _, col = linear_sum_assignment(cost)
            valid = [cost[i, col[i]] < INF_COST for i in range(n)]

        for slot, row in enumerate(group):
            true_seq = row["round_positions"][:-1]
            src, assigned = None, None
            if col is not None and valid[slot]:
                src = group[col[slot]]
                assigned, method = list(src["round_positions"][:-1]), "hungarian_move_matched"
                src = src["id"]
            if assigned is None:
                d = derange(true_seq)
                assigned, method = (d, "intra_derangement") if d is not None else (substitute(true_seq), "fallback_substitution")
            
            corrupted = assigned + [row["final_position"]]
            methods[method] += 1
            assert len(corrupted) == len(row["round_positions"])
            assert corrupted[-1] == row["final_position"], "Final position must match"
            assert corrupted[:-1] != true_seq, f"{row['id']}: corruption identical to GT"
            
            changed = sum(1 for i in range(n_int) if corrupted[i] != row["round_positions"][i])
            records.append(
                make_conversation_record(
                    row,
                    prompt_trajectory(row["target_card"]),
                    render_trajectory(row["initial_position"], corrupted, row["final_position"])
                )
            )
            audit.append({
                "id": row["id"],
                "source": src,
                "method": method,
                "n_intermediate": n_int,
                "n_changed": changed,
                "true": row["round_positions"],
                "corrupted": corrupted,
                "true_moves": move_count_of(row["initial_position"], row["round_positions"]),
                "corrupted_moves": move_count_of(row["initial_position"], corrupted)
            })

    order = {r["id"]: i for i, r in enumerate(train_rows)}
    records.sort(key=lambda x: order[x["id"]])
    audit.sort(key=lambda x: order[x["id"]])

    # Audit statistics
    gt_f, co_f = Counter(), Counter()
    for a in audit:
        gt_f.update(a["true"])
        co_f.update(a["corrupted"])
    mh_t = Counter(a["true_moves"] for a in audit)
    mh_c = Counter(a["corrupted_moves"] for a in audit)
    
    summary = {
        "fold": fold_idx,
        "n": len(audit),
        "methods": dict(methods),
        "mean_moves_true": round(sum(a["true_moves"] for a in audit) / len(audit), 3),
        "mean_moves_corrupted": round(sum(a["corrupted_moves"] for a in audit) / len(audit), 3),
        "move_hist_l1": sum(abs(mh_t.get(m, 0) - mh_c.get(m, 0)) for m in set(mh_t) | set(mh_c)),
        "letter_freq_l1": sum(abs(gt_f.get(p, 0) - co_f.get(p, 0)) for p in POSITIONS),
        "majority_diverged": sum(1 for a in audit if a["n_changed"] > a["n_intermediate"] / 2),
    }
    return records, audit, summary
