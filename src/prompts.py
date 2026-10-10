"""
Prompt templates and supervision target formatting for VLM temporal state tracking.
"""

from typing import List, Dict, Any

def prompt_final(card: str) -> str:
    """Prompt asking for only the final position of target card."""
    return f"""Watch the full card trick video.

The target card is the card labeled {card}.

There are four fixed screen positions:
A, B, C, and D.

Track the target card through the entire video.

What is the final position of the target card?

Answer with ONLY one letter:
A
B
C
or
D"""

def prompt_trajectory(card: str) -> str:
    """Prompt asking for intermediate trajectory across all rounds."""
    return f"""Watch the full card trick video.

The target card is the card labeled {card}.

There are four fixed screen positions:
A, B, C, and D.

Track the target card through every round.

Report the target card's position after each round,
in chronological order.

Use exactly this format:

Initial: X
Round 1: X
Round 2: X
...
Final: X

Replace X with A, B, C, or D.

Do not add anything else."""

PROMPTS = {
    "final": prompt_final,
    "trajectory": prompt_trajectory
}

def render_trajectory(initial: str, rounds: List[str], final: str) -> str:
    """Renders formatted multi-round trajectory string."""
    lines = [f"Initial: {initial}"] + [f"Round {i}: {p}" for i, p in enumerate(rounds, 1)] + [f"Final: {final}"]
    return "\n".join(lines)

def make_conversation_record(row: Dict[str, Any], prompt: str, target_answer: str) -> Dict[str, Any]:
    """Formats training item into standard multimodal Qwen conversation format."""
    return {
        "id": row["id"],
        "video": row["video_file"],
        "conversations": [
            {"from": "human", "value": "<video>\n" + prompt},
            {"from": "gpt", "value": target_answer}
        ]
    }
