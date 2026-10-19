from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral

import torch


IGNORE_LABEL = -100


def _validate_text_rows(text_rows, argument_name):
    if isinstance(text_rows, (str, bytes)):
        raise TypeError(f"{argument_name} must be a sequence of strings.")

    if not isinstance(text_rows, Sequence):
        raise TypeError(f"{argument_name} must be a sequence of strings.")

    for row_index, text in enumerate(text_rows):
        if not isinstance(text, str):
            raise TypeError(
                f"{argument_name}[{row_index}] must be a string."
            )


def _validate_maximum_length(maximum_sequence_length):
    if (
        isinstance(maximum_sequence_length, bool)
        or not isinstance(maximum_sequence_length, Integral)
    ):
        raise TypeError("maximum_sequence_length must be an integer.")

    if maximum_sequence_length < 1:
        raise ValueError("maximum_sequence_length must be positive.")


def _get_padding_token_id(tokenizer):
    padding_token_id = getattr(tokenizer, "pad_token_id", None)

    if padding_token_id is None:
        padding_token_id = getattr(tokenizer, "eos_token_id", None)

    if padding_token_id is None:
        raise ValueError(
            "The tokenizer must provide a pad_token_id or eos_token_id."
        )

    return int(padding_token_id)


def encode_prompt_response(
    tokenizer,
    prompt,
    response,
    maximum_sequence_length,
):
    _validate_maximum_length(maximum_sequence_length)

    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string.")

    if not isinstance(response, str):
        raise TypeError("response must be a string.")

    prompt_token_ids = list(
        tokenizer.encode(prompt, add_special_tokens=False)
    )
    response_token_ids = list(
        tokenizer.encode(response, add_special_tokens=False)
    )

    end_of_sequence_token_id = getattr(tokenizer, "eos_token_id", None)

    if end_of_sequence_token_id is not None:
        response_token_ids.append(int(end_of_sequence_token_id))

    combined_token_ids = prompt_token_ids + response_token_ids
    combined_labels = (
        [IGNORE_LABEL] * len(prompt_token_ids) + response_token_ids
    )

    return {
        "input_ids": combined_token_ids[:maximum_sequence_length],
        "labels": combined_labels[:maximum_sequence_length],
    }


def pad_encoded_examples(
    encoded_examples,
    padding_token_id,
):
    if len(encoded_examples) == 0:
        raise ValueError("encoded_examples must contain at least one example.")

    maximum_batch_sequence_length = 0

    for example_index, encoded_example in enumerate(encoded_examples):
        token_ids = encoded_example["input_ids"]
        labels = encoded_example["labels"]

        if len(token_ids) != len(labels):
            raise ValueError(
                f"Example {example_index} has different token and label lengths."
            )

        maximum_batch_sequence_length = max(
            maximum_batch_sequence_length,
            len(token_ids),
        )

    if maximum_batch_sequence_length == 0:
        raise ValueError("The batch must contain at least one token.")

    batch_size = len(encoded_examples)
    batch_shape = (batch_size, maximum_batch_sequence_length)

    batch_input_ids = torch.full(
        batch_shape,
        fill_value=padding_token_id,
        dtype=torch.long,
    )
    batch_attention_mask = torch.zeros(
        batch_shape,
        dtype=torch.long,
    )
    batch_labels = torch.full(
        batch_shape,
        fill_value=IGNORE_LABEL,
        dtype=torch.long,
    )

    for example_index, encoded_example in enumerate(encoded_examples):
        example_sequence_length = len(encoded_example["input_ids"])

        if example_sequence_length == 0:
            continue

        batch_input_ids[
            example_index, :example_sequence_length
        ] = torch.tensor(
            encoded_example["input_ids"],
            dtype=torch.long,
        )

        batch_attention_mask[
            example_index, :example_sequence_length
        ] = 1

        batch_labels[
            example_index, :example_sequence_length
        ] = torch.tensor(
            encoded_example["labels"],
            dtype=torch.long,
        )

    return {
        "input_ids": batch_input_ids,
        "attention_mask": batch_attention_mask,
        "labels": batch_labels,
    }


def build_response_masked_batch(
    tokenizer,
    prompts,
    responses,
    maximum_sequence_length,
):
    _validate_maximum_length(maximum_sequence_length)
    _validate_text_rows(prompts, "prompts")
    _validate_text_rows(responses, "responses")

    if len(prompts) == 0:
        raise ValueError("prompts must contain at least one example.")

    if len(prompts) != len(responses):
        raise ValueError(
            "prompts and responses must contain the same number of examples."
        )

    padding_token_id = _get_padding_token_id(tokenizer)
    encoded_examples = []

    for prompt, response in zip(prompts, responses):
        encoded_example = encode_prompt_response(
            tokenizer=tokenizer,
            prompt=prompt,
            response=response,
            maximum_sequence_length=maximum_sequence_length,
        )
        encoded_examples.append(encoded_example)

    return pad_encoded_examples(
        encoded_examples=encoded_examples,
        padding_token_id=padding_token_id,
    )


def build_preference_batch(
    tokenizer,
    prompts,
    chosen_responses,
    rejected_responses,
    maximum_sequence_length,
):
    _validate_maximum_length(maximum_sequence_length)
    _validate_text_rows(prompts, "prompts")
    _validate_text_rows(chosen_responses, "chosen_responses")
    _validate_text_rows(rejected_responses, "rejected_responses")

    if not (
        len(prompts)
        == len(chosen_responses)
        == len(rejected_responses)
    ):
        raise ValueError(
            "prompts, chosen_responses, and rejected_responses "
            "must contain the same number of examples."
        )

    chosen_batch = build_response_masked_batch(
        tokenizer=tokenizer,
        prompts=prompts,
        responses=chosen_responses,
        maximum_sequence_length=maximum_sequence_length,
    )

    rejected_batch = build_response_masked_batch(
        tokenizer=tokenizer,
        prompts=prompts,
        responses=rejected_responses,
        maximum_sequence_length=maximum_sequence_length,
    )

    return {
        "chosen": chosen_batch,
        "rejected": rejected_batch,
    }