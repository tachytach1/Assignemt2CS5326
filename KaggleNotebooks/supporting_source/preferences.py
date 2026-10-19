from __future__ import annotations

import hashlib
import json
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import HfApi
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from alignment.batching import encode_prompt_response, pad_encoded_examples
from alignment.data import formatinstruction


def extracthhpair(sourcerow):
    chosenconversation = sourcerow.get("chosen")
    rejectedconversation = sourcerow.get("rejected")

    if not isinstance(chosenconversation, str):
        return None

    if not isinstance(rejectedconversation, str):
        return None

    conversations = []

    for conversation in (chosenconversation, rejectedconversation):
        conversation = conversation.strip()

        if not conversation.startswith("Human:"):
            return None

        if conversation.count("\n\nHuman:") != 0:
            return None

        if conversation.count("\n\nAssistant:") != 1:
            return None

        instruction, response = conversation.split("\n\nAssistant:", 1)
        instruction = instruction.removeprefix("Human:").strip()
        response = response.strip()

        if not instruction or not response:
            return None

        conversations.append((instruction, response))

    choseninstruction, chosenresponse = conversations[0]
    rejectedinstruction, rejectedresponse = conversations[1]

    if choseninstruction != rejectedinstruction:
        return None

    return choseninstruction, chosenresponse, rejectedresponse


def extractultrafeedbackpair(sourcerow):
    chosenmessages = sourcerow.get("chosen")
    rejectedmessages = sourcerow.get("rejected")

    exchanges = []

    for messages in (chosenmessages, rejectedmessages):
        if not isinstance(messages, list) or len(messages) != 2:
            return None

        usermessage, assistantmessage = messages

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

        exchanges.append((instruction, response))

    choseninstruction, chosenresponse = exchanges[0]
    rejectedinstruction, rejectedresponse = exchanges[1]

    if choseninstruction != rejectedinstruction:
        return None

    return choseninstruction, chosenresponse, rejectedresponse


def collatepreferencebatch(examples, paddingtokenid):
    chosenexamples = []
    rejectedexamples = []

    for example in examples:
        chosenexamples.append(example["chosen"])
        rejectedexamples.append(example["rejected"])

    return {
        "chosen": pad_encoded_examples(chosenexamples, paddingtokenid),
        "rejected": pad_encoded_examples(rejectedexamples, paddingtokenid),
    }


def preparepreferencesplit(
    tokenizer,
    datasetname,
    datasetrevision,
    sourcesplit,
    extractor,
    requestedpairs,
    maximumlength,
    maximumpromptlength,
    seed,
    shufflebuffer,
    seeninstructions,
):
    sourcedataset = load_dataset(
        datasetname,
        revision=datasetrevision,
        split=sourcesplit,
        streaming=True,
    ).shuffle(seed=seed, buffer_size=shufflebuffer)

    acceptedpairs = []

    statistics = {
        "source_split": sourcesplit,
        "examined_pairs": 0,
        "unsupported_or_invalid_pairs": 0,
        "duplicate_instructions": 0,
        "identical_response_pairs": 0,
        "long_prompt_pairs": 0,
        "pairs_without_response_tokens": 0,
        "identical_tokenized_pairs": 0,
        "accepted_pairs": 0,
        "chosen_response_tokens": 0,
        "rejected_response_tokens": 0,
    }

    with tqdm(
        total=requestedpairs,
        desc=f"Preparing {sourcesplit}",
        unit="pair",
    ) as progress:
        for sourcerow in sourcedataset:
            statistics["examined_pairs"] += 1
            extractedpair = extractor(sourcerow)

            if extractedpair is None:
                statistics["unsupported_or_invalid_pairs"] += 1
                continue

            instruction, chosenresponse, rejectedresponse = extractedpair

            instructionhash = hashlib.sha256(
                instruction.encode("utf-8")
            ).hexdigest()

            if instructionhash in seeninstructions:
                statistics["duplicate_instructions"] += 1
                continue

            if chosenresponse == rejectedresponse:
                statistics["identical_response_pairs"] += 1
                continue

            prompt = formatinstruction(instruction)
            promptlength = len(
                tokenizer.encode(prompt, add_special_tokens=False)
            )

            if promptlength > maximumpromptlength:
                statistics["long_prompt_pairs"] += 1
                continue

            chosenexample = encode_prompt_response(
                tokenizer,
                prompt,
                chosenresponse,
                maximumlength,
            )
            rejectedexample = encode_prompt_response(
                tokenizer,
                prompt,
                rejectedresponse,
                maximumlength,
            )

            chosenresponsecount = 0
            rejectedresponsecount = 0

            for tokenlabel in chosenexample["labels"][1:]:
                if tokenlabel != -100:
                    chosenresponsecount += 1

            for tokenlabel in rejectedexample["labels"][1:]:
                if tokenlabel != -100:
                    rejectedresponsecount += 1

            if chosenresponsecount == 0 or rejectedresponsecount == 0:
                statistics["pairs_without_response_tokens"] += 1
                continue

            if chosenexample["input_ids"] == rejectedexample["input_ids"]:
                statistics["identical_tokenized_pairs"] += 1
                continue

            acceptedpairs.append(
                {
                    "instruction_hash": instructionhash,
                    "instruction": instruction,
                    "prompt": prompt,
                    "chosen_response": chosenresponse,
                    "rejected_response": rejectedresponse,
                    "chosen": chosenexample,
                    "rejected": rejectedexample,
                    "chosen_response_token_count": chosenresponsecount,
                    "rejected_response_token_count": rejectedresponsecount,
                }
            )

            seeninstructions.add(instructionhash)
            statistics["accepted_pairs"] += 1
            statistics["chosen_response_tokens"] += chosenresponsecount
            statistics["rejected_response_tokens"] += rejectedresponsecount

            progress.update(1)

            if len(acceptedpairs) == requestedpairs:
                break

    if len(acceptedpairs) != requestedpairs:
        raise RuntimeError(
            f"{sourcesplit} supplied {len(acceptedpairs)} usable pairs, "
            f"but {requestedpairs} were requested. "
            f"Filtering statistics: {statistics}"
        )

    return Dataset.from_list(acceptedpairs), statistics


def preparepreferences(
    checkpointpath,
    track,
    outputpath,
    trainingpairs=15000,
    validationpairs=1000,
    maximumlength=512,
    maximumpromptlength=256,
    seed=42,
    shufflebuffer=1000,
    datasetrevision="main",
):
    for argumentname, argumentvalue in {
        "trainingpairs": trainingpairs,
        "validationpairs": validationpairs,
        "maximumlength": maximumlength,
        "maximumpromptlength": maximumpromptlength,
        "shufflebuffer": shufflebuffer,
    }.items():
        if argumentvalue < 1:
            raise ValueError(f"{argumentname} must be positive.")

    if maximumpromptlength >= maximumlength:
        raise ValueError("The prompt limit must leave room for response tokens.")

    if track == "hh":
        datasetname = "Anthropic/hh-rlhf"
        trainingsplit = "train"
        validationsplit = "test"
        extractor = extracthhpair
    elif track == "ultrafeedback":
        datasetname = "HuggingFaceH4/ultrafeedback_binarized"
        trainingsplit = "train_prefs"
        validationsplit = "test_prefs"
        extractor = extractultrafeedbackpair
    else:
        raise ValueError("track must be hh or ultrafeedback.")

    checkpointpath = Path(checkpointpath)
    destination = Path(outputpath)

    if destination.exists():
        raise FileExistsError(f"Output directory already exists: {destination}")

    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpointpath),
        local_files_only=True,
        fix_mistral_regex=False,
    )

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("The tokenizer needs a padding or EOS token.")
        tokenizer.pad_token = tokenizer.eos_token

    resolvedrevision = HfApi().dataset_info(
        datasetname,
        revision=datasetrevision,
    ).sha

    seeninstructions = set()

    trainingdataset, trainingstatistics = preparepreferencesplit(
        tokenizer,
        datasetname,
        resolvedrevision,
        trainingsplit,
        extractor,
        trainingpairs,
        maximumlength,
        maximumpromptlength,
        seed,
        shufflebuffer,
        seeninstructions,
    )

    validationdataset, validationstatistics = preparepreferencesplit(
        tokenizer,
        datasetname,
        resolvedrevision,
        validationsplit,
        extractor,
        validationpairs,
        maximumlength,
        maximumpromptlength,
        seed,
        shufflebuffer,
        seeninstructions,
    )

    manifest = {
        "track": track,
        "dataset_name": datasetname,
        "dataset_revision": resolvedrevision,
        "sft_checkpoint": str(checkpointpath),
        "checkpoint_metadata": json.loads(
            (checkpointpath / "checkpoint.json").read_text(encoding="utf-8")
        ),
        "tokenizer_file_sha256": hashlib.sha256(
            (checkpointpath / "tokenizer.json").read_bytes()
        ).hexdigest(),
        "seed": seed,
        "shuffle_buffer": shufflebuffer,
        "maximum_sequence_length": maximumlength,
        "maximum_prompt_length": maximumpromptlength,
        "long_prompt_policy": "reject; do not truncate the instruction separately",
        "conversation_policy": "single user and assistant exchange only",
        "prompt_template": "### Instruction:\n{instruction}\n\n### Response:\n",
        "padding_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "training": trainingstatistics,
        "validation": validationstatistics,
    }

    destination.mkdir(parents=True, exist_ok=False)

    DatasetDict(
        {
            "train": trainingdataset,
            "validation": validationdataset,
        }
    ).save_to_disk(str(destination / "dataset"))

    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(manifest, indent=2))
    print("Prepared preference data:", destination.resolve())

    return manifest
