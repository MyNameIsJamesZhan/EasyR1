# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for adaGRPO advantage estimation."""

import numpy as np
import pytest
import torch

from verl.trainer.core_algos import AdvantageEstimator, compute_advantage_return, gauss_diff


def _make_batch(scores_per_sample, seq_len=10):
    """Helper: build token_level_rewards and response_mask from per-sample scalar scores."""
    bsz = len(scores_per_sample)
    token_level_rewards = torch.zeros(bsz, seq_len)
    for i, s in enumerate(scores_per_sample):
        token_level_rewards[i, -1] = s
    response_mask = torch.ones(bsz, seq_len)
    return token_level_rewards, response_mask


# ---------------------------------------------------------------------------
# gauss_diff tests
# ---------------------------------------------------------------------------


def test_gauss_diff_group0_wins():
    nums1 = [0.9, 1.0, 0.8]
    nums2 = [0.1, 0.0, 0.2]
    assert gauss_diff(nums1, nums2) > 0.9, "group0 clearly wins → win_prob should be > 0.9"
    assert gauss_diff(nums2, nums1) < 0.1, "group0 clearly loses → win_prob should be < 0.1"


def test_gauss_diff_equal_groups():
    nums1 = [0.5, 0.5, 0.5]
    nums2 = [0.5, 0.5, 0.5]
    result = gauss_diff(nums1, nums2)
    assert abs(result - 0.5) < 0.01, f"Equal groups → win_prob ≈ 0.5, got {result}"


def test_gauss_diff_small_group_fallback():
    """Single-element groups → variance undefined, fallback to mean comparison."""
    result_win = gauss_diff([1.0], [0.0])
    result_lose = gauss_diff([0.0], [1.0])
    assert result_win in (0.0, 1.0), f"Fallback should return 0.0 or 1.0, got {result_win}"
    assert result_lose in (0.0, 1.0), f"Fallback should return 0.0 or 1.0, got {result_lose}"
    assert result_win == 1.0, "group0=[1.0] > group1=[0.0] → should return 1.0"
    assert result_lose == 0.0, "group0=[0.0] < group1=[1.0] → should return 0.0"


# ---------------------------------------------------------------------------
# compute_ada_grpo_outcome_advantage tests
# ---------------------------------------------------------------------------


def test_ada_grpo_advantage_group0_wins():
    """
    2 questions × 4 rollouts (2 per group).
    q0: group_0 wins (rewards 1.0 vs 0.0)
    q1: group_1 wins (rewards 0.0 vs 1.0)
    """
    seq_len = 10
    token_level_rewards = torch.zeros(8, seq_len)
    token_level_rewards[0, -1] = 1.0   # q0 g0 r0
    token_level_rewards[1, -1] = 1.0   # q0 g0 r1
    token_level_rewards[2, -1] = 0.0   # q0 g1 r0
    token_level_rewards[3, -1] = 0.0   # q0 g1 r1
    token_level_rewards[4, -1] = 0.0   # q1 g0 r0
    token_level_rewards[5, -1] = 0.0   # q1 g0 r1
    token_level_rewards[6, -1] = 1.0   # q1 g1 r0
    token_level_rewards[7, -1] = 1.0   # q1 g1 r1

    response_mask = torch.ones(8, seq_len)
    index = np.array(["q0", "q0", "q0", "q0", "q1", "q1", "q1", "q1"])
    group_index = np.array([0, 0, 1, 1, 0, 0, 1, 1])

    adv, ret = compute_advantage_return(
        AdvantageEstimator.ADA_GRPO,
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
        group_index=group_index,
    )

    # q0: group_0 wins → rows 0,1 should have higher advantage than rows 2,3
    assert adv[0, -1] > adv[2, -1], "q0: winning group_0 should have higher adv than losing group_1"
    assert adv[1, -1] > adv[3, -1], "q0: winning group_0 should have higher adv than losing group_1"

    # q1: group_1 wins → rows 6,7 should have higher advantage than rows 4,5
    assert adv[6, -1] > adv[4, -1], "q1: winning group_1 should have higher adv than losing group_0"
    assert adv[7, -1] > adv[5, -1], "q1: winning group_1 should have higher adv than losing group_0"

    # advantages and returns are identical (outcome-only reward, Phase 1)
    torch.testing.assert_close(adv, ret)


def test_ada_grpo_advantages_zero_on_padding():
    """Advantage must be zero wherever response_mask == 0."""
    token_level_rewards = torch.zeros(4, 10)
    token_level_rewards[0, -1] = 1.0
    token_level_rewards[1, -1] = 1.0

    # mask out the first 5 tokens for all samples
    response_mask = torch.ones(4, 10)
    response_mask[:, :5] = 0.0

    index = np.array(["q0", "q0", "q0", "q0"])
    group_index = np.array([0, 0, 1, 1])

    adv, _ = compute_advantage_return(
        AdvantageEstimator.ADA_GRPO,
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
        group_index=group_index,
    )

    # all padded positions should be exactly zero
    pad_region = adv[:, :5]
    assert pad_region.abs().sum().item() == 0.0, "Padding positions must have zero advantage"


def test_ada_grpo_requires_two_groups():
    """Only group_0 present → should raise AssertionError."""
    token_level_rewards = torch.zeros(4, 10)
    response_mask = torch.ones(4, 10)
    index = np.array(["q0", "q0", "q0", "q0"])
    group_index_bad = np.array([0, 0, 0, 0])  # missing group 1

    with pytest.raises((AssertionError, ValueError)):
        compute_advantage_return(
            AdvantageEstimator.ADA_GRPO,
            token_level_rewards=token_level_rewards,
            response_mask=response_mask,
            index=index,
            group_index=group_index_bad,
        )


def test_ada_grpo_reorder_invariant():
    """
    Advantages must be the same regardless of row order in the batch.
    Simulates _balance_batch() reordering that moves group_index together with batch.
    """
    seq_len = 10
    token_level_rewards = torch.zeros(8, seq_len)
    token_level_rewards[0, -1] = 1.0
    token_level_rewards[1, -1] = 0.9
    token_level_rewards[2, -1] = 0.1
    token_level_rewards[3, -1] = 0.0
    token_level_rewards[4, -1] = 0.0
    token_level_rewards[5, -1] = 0.1
    token_level_rewards[6, -1] = 0.9
    token_level_rewards[7, -1] = 1.0

    response_mask = torch.ones(8, seq_len)
    index = np.array(["q0", "q0", "q0", "q0", "q1", "q1", "q1", "q1"])
    group_index = np.array([0, 0, 1, 1, 0, 0, 1, 1])

    adv_orig, _ = compute_advantage_return(
        AdvantageEstimator.ADA_GRPO,
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
        group_index=group_index,
    )

    # shuffle rows
    perm = [3, 6, 0, 7, 1, 4, 2, 5]
    token_level_rewards_shuffled = token_level_rewards[perm]
    response_mask_shuffled = response_mask[perm]
    index_shuffled = index[perm]
    group_index_shuffled = group_index[perm]

    adv_shuffled, _ = compute_advantage_return(
        AdvantageEstimator.ADA_GRPO,
        token_level_rewards=token_level_rewards_shuffled,
        response_mask=response_mask_shuffled,
        index=index_shuffled,
        group_index=group_index_shuffled,
    )

    # un-shuffle and compare
    unperm = [0] * 8
    for new_idx, old_idx in enumerate(perm):
        unperm[old_idx] = new_idx
    adv_unshuffled = adv_shuffled[unperm]

    torch.testing.assert_close(adv_orig, adv_unshuffled, msg="Advantages must be reorder-invariant")
