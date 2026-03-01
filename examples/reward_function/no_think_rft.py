"""
No-Thinking-RFT reward function.

From "To Think or Not To Think" (Li et al., 2025, arXiv:2503.16188).

Eliminates the format reward entirely. Accuracy is based on exact equality
matching between the model's raw response and the ground truth, after
normalizing whitespace and case. This forces the model to output the answer
directly without any thinking or formatting overhead.
"""

import re
from typing import Any


REWARD_NAME = "no_think_rft"
REWARD_TYPE = "batch"


def accuracy_reward(response: str, ground_truth: str) -> float:
    """Exact-match accuracy: 1.0 if response matches ground truth, else 0.0."""
    response_clean = response.strip().lower()
    ground_truth_clean = ground_truth.strip().lower()

    # Direct match
    if response_clean == ground_truth_clean:
        return 1.0

    # Also accept if the response starts with or contains the answer as a
    # standalone token (handles minor trailing punctuation from the model)
    if re.fullmatch(re.escape(ground_truth_clean) + r"[.\s]*", response_clean):
        return 1.0

    return 0.0


def compute_score(reward_inputs: list[dict[str, Any]], **kwargs) -> list[dict[str, float]]:
    scores = []
    for reward_input in reward_inputs:
        response = reward_input["response"]
        acc = accuracy_reward(response, reward_input["ground_truth"])
        scores.append({"overall": acc, "format": 0.0, "accuracy": acc})
    return scores
