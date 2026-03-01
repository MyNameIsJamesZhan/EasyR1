"""
Adaptive-Thinking reward function.

From "To Think or Not To Think" (Li et al., 2025, arXiv:2503.16188), Appendix E.3.

Format reward: 1 if response matches either valid format:
  (1) <answer> ... </answer>
  (2) <think> ... </think> <answer> ... </answer>

Accuracy reward: extract answer from <answer> tag and grade against ground truth,
same as Thinking-RFT.
"""

import re
from typing import Any

from mathruler.grader import extract_boxed_content, grade_answer


REWARD_NAME = "adaptive_thinking"
REWARD_TYPE = "batch"

# Both valid response formats
_DIRECT_ANSWER = re.compile(r"<answer>.*?</answer>", re.DOTALL)
_THINK_THEN_ANSWER = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)
_ANSWER_TAG = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def format_reward(response: str) -> float:
    """1.0 if response is in either the direct-answer or think-then-answer format."""
    r = response.strip()
    if _THINK_THEN_ANSWER.fullmatch(r) or _DIRECT_ANSWER.fullmatch(r):
        return 1.0
    return 0.0


def accuracy_reward(response: str, ground_truth: str) -> float:
    """Extract answer from <answer> tag and grade against ground truth."""
    match = _ANSWER_TAG.search(response)
    if not match:
        return 0.0
    answer = match.group(1).strip()
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


def compute_score(
    reward_inputs: list[dict[str, Any]], format_weight: float = 0.1
) -> list[dict[str, float]]:
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
