from __future__ import annotations

import torch
import torch.nn.functional as functional

from alignment.batching import (
    IGNORE_LABEL,
    build_response_masked_batch,
)


def move_batch_to_model_device(model, language_model_batch):
    first_parameter = next(model.parameters(), None)

    if first_parameter is not None:
        model_device = first_parameter.device
    else:
        first_buffer = next(model.buffers(), None)
        model_device = (
            first_buffer.device
            if first_buffer is not None
            else torch.device("cpu")
        )

    device_batch = {}

    for tensor_name, batch_tensor in language_model_batch.items():
        device_batch[tensor_name] = batch_tensor.to(
            device=model_device,
            non_blocking=True,
        )

    return device_batch


def compute_batch_response_statistics(model, language_model_batch):
    required_tensor_names = (
        "input_ids",
        "attention_mask",
        "labels",
    )

    for tensor_name in required_tensor_names:
        if tensor_name not in language_model_batch:
            raise ValueError(f"The batch is missing {tensor_name}.")

        batch_tensor = language_model_batch[tensor_name]

        if not isinstance(batch_tensor, torch.Tensor):
            raise TypeError(f"{tensor_name} must be a torch.Tensor.")

        if batch_tensor.ndim != 2:
            raise ValueError(f"{tensor_name} must have shape [batch, sequence].")

        if batch_tensor.dtype != torch.long:
            raise TypeError(f"{tensor_name} must have dtype torch.long.")

    expected_batch_shape = language_model_batch["input_ids"].shape

    for tensor_name in required_tensor_names:
        if language_model_batch[tensor_name].shape != expected_batch_shape:
            raise ValueError(
                "input_ids, attention_mask, and labels "
                "must have identical shapes."
            )

    batch_size, sequence_length = expected_batch_shape

    if batch_size == 0 or sequence_length == 0:
        raise ValueError("The batch must contain examples and tokens.")

    device_batch = move_batch_to_model_device(
        model=model,
        language_model_batch=language_model_batch,
    )

    model_outputs = model(
        input_ids=device_batch["input_ids"],
        attention_mask=device_batch["attention_mask"],
    )

    if isinstance(model_outputs, dict):
        prediction_logits = model_outputs["logits"]
    else:
        prediction_logits = model_outputs.logits

    if prediction_logits.ndim != 3:
        raise ValueError("Model logits must have shape [batch, sequence, vocabulary].")

    if prediction_logits.shape[:2] != expected_batch_shape:
        raise ValueError("Model logits must match the input batch and sequence lengths.")

    shifted_response_labels = device_batch["labels"][:, 1:]
    shifted_attention_mask = device_batch["attention_mask"][:, 1:]

    scoreable_response_positions = (
        shifted_response_labels.ne(IGNORE_LABEL)
        & shifted_attention_mask.ne(0)
    )

    response_token_counts = scoreable_response_positions.sum(dim=1)

    response_row_indices, response_position_indices = (
        scoreable_response_positions.nonzero(as_tuple=True)
    )

    if response_row_indices.numel() == 0:
        gradient_connected_zero = (
            prediction_logits[:, :1, :1].float().sum() * 0.0
        )

        response_log_probabilities = (
            torch.zeros(
                batch_size,
                dtype=torch.float32,
                device=prediction_logits.device,
            )
            + gradient_connected_zero
        )

        return response_log_probabilities, response_token_counts

    selected_prediction_logits = prediction_logits[
        response_row_indices,
        response_position_indices,
        :,
    ]

    selected_response_token_ids = shifted_response_labels[
        response_row_indices,
        response_position_indices,
    ]

    response_token_negative_log_probabilities = functional.cross_entropy(
        selected_prediction_logits.float(),
        selected_response_token_ids,
        reduction="none",
    )

    response_log_probabilities = torch.zeros(
        batch_size,
        dtype=response_token_negative_log_probabilities.dtype,
        device=prediction_logits.device,
    ).index_add(
        dim=0,
        index=response_row_indices,
        source=-response_token_negative_log_probabilities,
    )

    return response_log_probabilities, response_token_counts


def compute_response_log_probabilities(
    model,
    tokenizer,
    prompts,
    responses,
    maximum_sequence_length,
):
    language_model_batch = build_response_masked_batch(
        tokenizer=tokenizer,
        prompts=prompts,
        responses=responses,
        maximum_sequence_length=maximum_sequence_length,
    )

    response_log_probabilities, _ = compute_batch_response_statistics(
        model=model,
        language_model_batch=language_model_batch,
    )

    return response_log_probabilities


def compute_batch_sft_loss(model, language_model_batch):
    response_log_probabilities, response_token_counts = (
        compute_batch_response_statistics(
            model=model,
            language_model_batch=language_model_batch,
        )
    )

    total_response_token_count = response_token_counts.sum().clamp_min(1)

    return (
        -response_log_probabilities.sum()
        / total_response_token_count
    )


def compute_response_only_sft_loss(
    model,
    tokenizer,
    prompts,
    responses,
    maximum_sequence_length,
):
    language_model_batch = build_response_masked_batch(
        tokenizer=tokenizer,
        prompts=prompts,
        responses=responses,
        maximum_sequence_length=maximum_sequence_length,
    )

    return compute_batch_sft_loss(
        model=model,
        language_model_batch=language_model_batch,
    )