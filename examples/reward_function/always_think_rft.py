"""
Reward function for the "thinking" format prompt variant.
Uses <thinking>...</thinking> tags (instead of <think>) + \boxed{} answer.
"""

import re
from typing import Any

from mathruler.grader import extract_boxed_content, grade_answer


REWARD_NAME = "always_think_rft"
REWARD_TYPE = "batch"


def format_reward(response: str) -> float:
    pattern = re.compile(r"<thinking>.*</thinking>.*\\boxed\{.*\}.*", re.DOTALL)
    return 1.0 if re.fullmatch(pattern, response) else 0.0


def accuracy_reward(response: str, ground_truth: str) -> float:
    answer = extract_boxed_content(response)
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


def compute_score(reward_inputs: list[dict[str, Any]], format_weight: float = 0.1) -> list[dict[str, float]]:
    scores = []
    for reward_input in reward_inputs:
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])
        fmt = format_reward(response)
        acc = accuracy_reward(response, reward_input["ground_truth"])
        scores.append({
            "overall": (1 - format_weight) * acc + format_weight * fmt,
            "format": fmt,
            "accuracy": acc,
        })
    return scores
