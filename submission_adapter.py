"""Stable grading interface for the SFT and DPO implementation.

Keep the function signatures below unchanged. Each function should be a thin
wrapper around your own implementation; you do not need to place the training
code itself in this file.
"""

from __future__ import annotations
from alignment.batching import (
    build_preference_batch,
    build_response_masked_batch,
)

from alignment.scoring import (
    compute_response_log_probabilities,
    compute_response_only_sft_loss,
)

from alignment.dpo import (
    computepreferencelossfromscores,
    computeresponseonlydpoloss,
)


def build_sft_batch(tokenizer, prompts, responses, max_length):
    """Return input_ids, attention_mask, and response-only labels."""

    return build_response_masked_batch(
        tokenizer=tokenizer,
        prompts=prompts,
        responses=responses,
        maximum_sequence_length=max_length,
    )


def compute_response_logprobs(model, tokenizer, prompts, responses, max_length):
    """Return one summed response-token log-probability per input row."""

    return compute_response_log_probabilities(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        responses=responses,
        maximum_sequence_length=max_length,
    )


def compute_sft_loss(model, tokenizer, prompts, responses, max_length):
    """Return a scalar response-only SFT loss using either documented reduction."""

    return compute_response_only_sft_loss(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        responses=responses,
        maximum_sequence_length=max_length,
    )


def build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length):
    """Return separate response-masked LM batches for chosen and rejected rows."""

    return build_preference_batch(
        tokenizer=tokenizer,
        prompts=prompts,
        chosen_responses=chosen_responses,
        rejected_responses=rejected_responses,
        maximum_sequence_length=max_length,
    )


def compute_dpo_loss_from_logps(
    policy_chosen_logps,
    policy_rejected_logps,
    reference_chosen_logps,
    reference_rejected_logps,
    beta,
):
    """Return the scalar mean DPO objective from per-example log probabilities."""

    return computepreferencelossfromscores(
        policy_chosen_logps,
        policy_rejected_logps,
        reference_chosen_logps,
        reference_rejected_logps,
        beta,
    )


def compute_dpo_loss(
    policy_model,
    reference_model,
    tokenizer,
    prompts,
    chosen_responses,
    rejected_responses,
    beta,
    max_length,
):
    """Return a scalar mean DPO loss with a frozen reference computation."""

    return computeresponseonlydpoloss(
        policy_model,
        reference_model,
        tokenizer,
        prompts,
        chosen_responses,
        rejected_responses,
        beta,
        max_length,
    )
