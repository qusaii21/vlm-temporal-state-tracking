"""
Robust parsing for VLM outputs (trajectory prompt and single-letter prompt).
"""

import re
from typing import Dict, Any

LETTER = r"[ABCD]"
ENUM_RE = re.compile(r"\bA\s*,\s*B\s*,\s*C\s*,?\s*(?:AND|OR)\s*D\b", re.I)

def clean_text(t: str) -> str:
    """Strip markdown formatting and enumeration lists."""
    t = re.sub(r"[*_`#]+", " ", t.strip())
    return ENUM_RE.sub(" ", t)

def parse_model_output(text: str) -> Dict[str, Any]:
    """
    Parses VLM generation output into structured round predictions and final answer.
    """
    out = {
        "final": None,
        "final_method": None,
        "initial": None,
        "rounds": {},
        "round_moves": {}
    }
    up = clean_text(text).upper()
    
    # Initial position
    m = re.search(rf"\bINITIAL(?:\s+POSITION)?\s*[:\-]?\s*\(?({LETTER})\b", up)
    if m:
        out["initial"] = m.group(1)
        
    # Intermediate rounds
    for rm in re.finditer(rf"\bROUND\s*(\d+)\s*[:\-]?\s*([^\n]*)", up):
        idx, body = int(rm.group(1)), rm.group(2)
        mv = re.search(rf"MOVED?\s+FROM\s+\(?({LETTER})\)?\s+TO\s+\(?({LETTER})\b", body)
        if mv:
            out["rounds"][idx] = mv.group(2)
            out["round_moves"][idx] = 1
            continue
        st = re.search(rf"STAY(?:ED|S)?\s+(?:AT|IN)\s+\(?({LETTER})\b", body)
        if st:
            out["rounds"][idx] = st.group(1)
            out["round_moves"][idx] = 0
            continue
        bare = re.search(rf"\b({LETTER})\b", body)
        if bare:
            out["rounds"][idx] = bare.group(1)
        if re.search(r"\bMOVED?\b", body):
            out["round_moves"][idx] = 1
        elif re.search(r"\bSTAY", body) or re.search(r"\bDID\s+NOT\s+MOVE\b", body):
            out["round_moves"][idx] = 0

    # Final position extraction
    s = up.strip().strip(".").strip()
    if len(s) == 1 and s in "ABCD":
        out["final"], out["final_method"] = s, "bare_letter"
    else:
        m = re.search(rf"\bFINAL(?:\s+POSITION)?\s*[:\-]?\s*(?:IS\s+)?\(?({LETTER})\b", up)
        if m:
            out["final"], out["final_method"] = m.group(1), "final_label"
            
    if out["final"] is None:
        m = re.search(rf"\b(?:POSITION|ANSWER|CARD)\s+(?:IS\s+)?\(?({LETTER})\b", up)
        if m:
            out["final"], out["final_method"] = m.group(1), "answer_phrase"
            
    if out["final"] is None and out["rounds"]:
        out["final"], out["final_method"] = out["rounds"][max(out["rounds"])], "last_round"
        
    if out["final"] is None:
        letters = re.findall(rf"\b({LETTER})\b", up)
        if letters:
            out["final"], out["final_method"] = letters[-1], "last_letter_fallback"
            
    return out
