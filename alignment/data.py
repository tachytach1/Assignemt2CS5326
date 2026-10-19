from __future__ import annotations

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

from alignment.batching import (
    IGNORE_LABEL,
    encode_prompt_response,
    pad_encoded_examples,
)


def formatinstruction(instruction):
    if not isinstance(instruction, str):
        raise TypeError("Instruction must be a string.")

    instruction = instruction.strip()

    if not instruction:
        raise ValueError("Instruction must not be empty.")

    return f"### Instruction:\n{instruction}\n\n### Response:\n"


def extractfirstexchange(sourcerow):
    messages = sourcerow.get("messages")

    if not isinstance(messages, list) or len(messages) < 2:
        return None

    usermessage = messages[0]
    assistantmessage = messages[1]

    if not isinstance(usermessage, dict):
        return None

    if not isinstance(assistantmessage, dict):
        return None

    if usermessage.get("role") != "user":
        return None

    if assistantmessage.get("role") != "assistant":
        return None

    instruction = usermessage.get("content")
    response = assistantmessage.get("content")

    if not isinstance(instruction, str) or not isinstance(response, str):
        return None

    instruction = instruction.strip()
    response = response.strip()

    if not instruction or not response:
        return None

    return instruction, response


def collatesftbatch(examples, paddingtokenid):
    return pad_encoded_examples(
        encoded_examples=examples,
        padding_token_id=paddingtokenid,
    )


def preparesftsplit(
    tokenizer,
    datasetname,
    datasetrevision,
    sourcesplit,
    requestedexamples,
    maximumlength,
    seed,
    shufflebuffer,
    seeninstructions,
):
    from datasets import Dataset, load_dataset
    from tqdm.auto import tqdm

    sourcedataset = load_dataset(
        datasetname,
        split=sourcesplit,
        revision=datasetrevision,
        streaming=True,
    )

    sourcedataset = sourcedataset.shuffle(
        seed=seed,
        buffer_size=shufflebuffer,
    )

    acceptedexamples = []

    statistics = {
        "source_split": sourcesplit,
        "requested_examples": requestedexamples,
        "examined_examples": 0,
        "invalid_examples": 0,
        "duplicate_instructions": 0,
        "examples_without_response_tokens": 0,
        "accepted_examples": 0,
        "examples_at_sequence_limit": 0,
        "total_sequence_tokens": 0,
        "total_response_tokens": 0,
    }

    with tqdm(
        total=requestedexamples,
        desc=f"Preparing {sourcesplit}",
        unit="example",
    ) as progress:
        for sourcerow in sourcedataset:
            statistics["examined_examples"] += 1

            exchange = extractfirstexchange(sourcerow)

            if exchange is None:
                statistics["invalid_examples"] += 1
                continue

            instruction, response = exchange

            instructionhash = hashlib.sha256(
                instruction.encode("utf-8")
            ).hexdigest()

            if instructionhash in seeninstructions:
                statistics["duplicate_instructions"] += 1
                continue

            prompt = formatinstruction(instruction)

            encodedexample = encode_prompt_response(
                tokenizer=tokenizer,
                prompt=prompt,
                response=response,
                maximum_sequence_length=maximumlength,
            )

            responsetokencount = 0

            for tokenlabel in encodedexample["labels"][1:]:
                if tokenlabel != IGNORE_LABEL:
                    responsetokencount += 1

            if responsetokencount == 0:
                statistics["examples_without_response_tokens"] += 1
                continue

            sequencelength = len(encodedexample["input_ids"])

            sourceidentifier = sourcerow.get("prompt_id")

            if sourceidentifier is None:
                sourceidentifier = instructionhash

            acceptedexample = {
                "source_id": str(sourceidentifier),
                "instruction_hash": instructionhash,
                "instruction": instruction,
                "prompt": prompt,
                "response": response,
                "input_ids": encodedexample["input_ids"],
                "labels": encodedexample["labels"],
                "sequence_length": sequencelength,
                "response_token_count": responsetokencount,
            }

            acceptedexamples.append(acceptedexample)
            seeninstructions.add(instructionhash)

            statistics["accepted_examples"] += 1
            statistics["total_sequence_tokens"] += sequencelength
            statistics["total_response_tokens"] += responsetokencount

            if sequencelength == maximumlength:
                statistics["examples_at_sequence_limit"] += 1

            progress.update(1)

            if len(acceptedexamples) == requestedexamples:
                break

    if len(acceptedexamples) != requestedexamples:
        raise RuntimeError(
            f"{sourcesplit} supplied only {len(acceptedexamples)} "
            f"usable examples; requested {requestedexamples}."
        )

    statistics["mean_sequence_length"] = (
        statistics["total_sequence_tokens"] / len(acceptedexamples)
    )
    statistics["mean_response_length"] = (
        statistics["total_response_tokens"] / len(acceptedexamples)
    )

    return Dataset.from_list(acceptedexamples), statistics


def preparesftdata(
    outputpath,
    modelname,
    modelrevision,
    datasetname,
    datasetrevision,
    trainingexamples,
    validationexamples,
    maximumlength,
    seed,
    shufflebuffer,
):
    from datasets import DatasetDict
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer

    positivearguments = {
        "trainingexamples": trainingexamples,
        "validationexamples": validationexamples,
        "maximumlength": maximumlength,
        "shufflebuffer": shufflebuffer,
    }

    for argumentname, argumentvalue in positivearguments.items():
        if argumentvalue < 1:
            raise ValueError(f"{argumentname} must be positive.")

    destination = Path(outputpath)

    if destination.exists():
        raise FileExistsError(
            f"{destination} already exists. "
            "Choose a new output path to preserve existing prepared data."
        )

    hubclient = HfApi()

    resolveddatasetrevision = hubclient.dataset_info(
        datasetname,
        revision=datasetrevision,
    ).sha

    resolvedmodelrevision = hubclient.model_info(
        modelname,
        revision=modelrevision,
    ).sha

    tokenizer = AutoTokenizer.from_pretrained(
        modelname,
        revision=resolvedmodelrevision,
        use_fast=True,
    )

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("The tokenizer needs a padding or EOS token.")

        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    seeninstructions = set()

    trainingdataset, trainingstatistics = preparesftsplit(
        tokenizer=tokenizer,
        datasetname=datasetname,
        datasetrevision=resolveddatasetrevision,
        sourcesplit="train_sft",
        requestedexamples=trainingexamples,
        maximumlength=maximumlength,
        seed=seed,
        shufflebuffer=shufflebuffer,
        seeninstructions=seeninstructions,
    )

    validationdataset, validationstatistics = preparesftsplit(
        tokenizer=tokenizer,
        datasetname=datasetname,
        datasetrevision=resolveddatasetrevision,
        sourcesplit="test_sft",
        requestedexamples=validationexamples,
        maximumlength=maximumlength,
        seed=seed,
        shufflebuffer=shufflebuffer,
        seeninstructions=seeninstructions,
    )

    prepareddataset = DatasetDict(
        {
            "train": trainingdataset,
            "validation": validationdataset,
        }
    )

    manifest = {
        "model_name": modelname,
        "model_revision": resolvedmodelrevision,
        "dataset_name": datasetname,
        "dataset_revision": resolveddatasetrevision,
        "seed": seed,
        "shuffle_buffer": shufflebuffer,
        "maximum_sequence_length": maximumlength,
        "prompt_template": (
            "### Instruction:\n{instruction}\n\n### Response:\n"
        ),
        "conversation_policy": "first user and assistant exchange only",
        "tokenization_policy": "separate prompt and response encoding",
        "truncation_policy": "right truncation after response EOS append",
        "padding_side": "right",
        "padding_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "training": trainingstatistics,
        "validation": validationstatistics,
        "package_versions": {},
    }

    for packagename in (
        "torch",
        "transformers",
        "datasets",
        "huggingface_hub",
    ):
        manifest["package_versions"][packagename] = version(packagename)

    destination.mkdir(parents=True, exist_ok=False)

    prepareddataset.save_to_disk(str(destination / "dataset"))
    tokenizer.save_pretrained(str(destination / "tokenizer"))

    manifestpath = destination / "manifest.json"
    manifestpath.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"Prepared SFT data saved to {destination.resolve()}")

    return prepareddataset, tokenizer, manifest


def main():
    parser = argparse.ArgumentParser(
        description="Prepare reproducible, tokenized SFT subsets."
    )

    parser.add_argument("--outputpath", required=True)
    parser.add_argument("--modelname", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--modelrevision", default="main")
    parser.add_argument(
        "--datasetname",
        default="HuggingFaceH4/ultrachat_200k",
    )
    parser.add_argument("--datasetrevision", default="main")
    parser.add_argument("--trainingexamples", type=int, default=30000)
    parser.add_argument("--validationexamples", type=int, default=1000)
    parser.add_argument("--maximumlength", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shufflebuffer", type=int, default=1000)

    arguments = parser.parse_args()

    preparesftdata(
        outputpath=arguments.outputpath,
        modelname=arguments.modelname,
        modelrevision=arguments.modelrevision,
        datasetname=arguments.datasetname,
        datasetrevision=arguments.datasetrevision,
        trainingexamples=arguments.trainingexamples,
        validationexamples=arguments.validationexamples,
        maximumlength=arguments.maximumlength,
        seed=arguments.seed,
        shufflebuffer=arguments.shufflebuffer,
    )


if __name__ == "__main__":
    main()