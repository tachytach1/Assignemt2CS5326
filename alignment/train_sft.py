from __future__ import annotations

import argparse
import json
import math
import random
import time
from functools import partial
from importlib.metadata import version
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from alignment.data import collatesftbatch
from alignment.scoring import compute_batch_response_statistics


def writejson(destination, contents):
    destination.write_text(
        json.dumps(contents, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def appendmetrics(destination, metrics):
    with destination.open("a", encoding="utf-8") as outputfile:
        outputfile.write(json.dumps(metrics, allow_nan=False) + "\n")


def countresponsetokens(batch):
    return int(
        (
            batch["labels"][:, 1:].ne(-100)
            & batch["attention_mask"][:, 1:].ne(0)
        ).sum().item()
    )


def evaluatevalidation(model, validationloader, device):
    previousmode = model.training
    model.eval()
    totalnegativeprobability = torch.zeros((), device=device)
    totaltokens = torch.zeros((), dtype=torch.long, device=device)

    try:
        with torch.no_grad():
            for batch in validationloader:
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=device.type == "cuda",
                ):
                    probabilities, tokencounts = compute_batch_response_statistics(
                        model, batch
                    )

                totalnegativeprobability += -probabilities.sum()
                totaltokens += tokencounts.sum()
    finally:
        model.train(previousmode)

    tokencount = int(totaltokens.item())

    if tokencount == 0:
        raise ValueError("Validation contains no scoreable response tokens.")

    validationloss = float(totalnegativeprobability.item()) / tokencount

    if not math.isfinite(validationloss):
        raise FloatingPointError("Validation loss is not finite.")

    return validationloss, tokencount


def savecheckpoint(model, tokenizer, destination, metadata):
    destination.mkdir(parents=True, exist_ok=True)
    previouscache = model.config.use_cache
    model.config.use_cache = True

    try:
        model.save_pretrained(str(destination), safe_serialization=True)
        tokenizer.save_pretrained(str(destination))
        writejson(destination / "checkpoint.json", metadata)
    finally:
        model.config.use_cache = previouscache


def trainaccumulationwindow(
    model, batches, optimizer, scaler, device, gradientclipping
):
    batchtokencounts = []

    for batch in batches:
        tokencount = countresponsetokens(batch)

        if tokencount == 0:
            raise ValueError("A training batch has no scoreable response tokens.")

        batchtokencounts.append(tokencount)

    windowtokencount = sum(batchtokencounts)
    optimizer.zero_grad(set_to_none=True)
    windownegativeprobability = torch.zeros((), device=device)

    for batch in batches:
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            probabilities, tokencounts = compute_batch_response_statistics(
                model, batch
            )
            batchnegativeprobability = -probabilities.sum()
            normalizedloss = batchnegativeprobability / windowtokencount

        scaler.scale(normalizedloss).backward()
        windownegativeprobability += batchnegativeprobability.detach()

    scaler.unscale_(optimizer)
    gradientnorm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), gradientclipping
    )

    if device.type == "cpu" and not torch.isfinite(gradientnorm).item():
        raise FloatingPointError("CPU training gradients are not finite.")

    previousscale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    updatesucceeded = scaler.get_scale() >= previousscale

    windowloss = float(windownegativeprobability.item()) / windowtokencount
    gradientvalue = float(gradientnorm.item())

    if not math.isfinite(windowloss):
        raise FloatingPointError("Training loss is not finite.")

    if not math.isfinite(gradientvalue):
        gradientvalue = None

    return windowloss, windowtokencount, gradientvalue, updatesucceeded


def savetrainingplots(metricspath, destination):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plotting

    trainingsteps = []
    traininglosses = []
    validationsteps = []
    validationlosses = []

    with metricspath.open(encoding="utf-8") as inputfile:
        for line in inputfile:
            record = json.loads(line)

            if record["event"] == "training":
                trainingsteps.append(record["optimizer_step"])
                traininglosses.append(record["training_loss"])
            elif record["event"] == "validation":
                validationsteps.append(record["optimizer_step"])
                validationlosses.append(record["validation_loss"])

    figure, axis = plotting.subplots(figsize=(8, 5))
    axis.plot(trainingsteps, traininglosses, label="Training (logged window)")
    axis.plot(validationsteps, validationlosses, marker="o", label="Validation")
    axis.set_xlabel("Successful optimizer updates")
    axis.set_ylabel("Response-token mean negative log probability")
    axis.set_title("SFT loss; no smoothing")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(destination, dpi=180)
    plotting.close(figure)


def trainsft(arguments):
    from datasets import load_from_disk
    from tqdm.auto import tqdm
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        get_cosine_schedule_with_warmup,
    )

    positivearguments = {
        "batchsize": arguments.batchsize,
        "validationbatchsize": arguments.validationbatchsize,
        "accumulation": arguments.accumulation,
        "epochs": arguments.epochs,
        "evaluationinterval": arguments.evaluationinterval,
        "logginginterval": arguments.logginginterval,
    }

    for argumentname, argumentvalue in positivearguments.items():
        if argumentvalue < 1:
            raise ValueError(f"{argumentname} must be positive.")

    if arguments.workers < 0 or arguments.maxsteps < 0:
        raise ValueError("workers and maxsteps must be nonnegative.")

    for argumentname in ("learningrate", "gradientclipping"):
        argumentvalue = getattr(arguments, argumentname)
        if not math.isfinite(argumentvalue) or argumentvalue <= 0:
            raise ValueError(f"{argumentname} must be finite and positive.")

    if not math.isfinite(arguments.weightdecay) or arguments.weightdecay < 0:
        raise ValueError("weightdecay must be finite and nonnegative.")

    if not 0 <= arguments.warmupfraction <= 1:
        raise ValueError("warmupfraction must be between zero and one.")

    device = torch.device(arguments.device)

    if device.type not in ("cuda", "cpu"):
        raise ValueError("This trainer supports CPU or CUDA devices.")

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Run GPU training on Kaggle.")
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)

    destination = Path(arguments.outputpath)

    if destination.exists():
        raise FileExistsError("Choose a new outputpath; existing runs are preserved.")

    preparedpath = Path(arguments.datapath)
    manifest = json.loads(
        (preparedpath / "manifest.json").read_text(encoding="utf-8")
    )
    dataset = load_from_disk(str(preparedpath / "dataset"))
    tokenizer = AutoTokenizer.from_pretrained(str(preparedpath / "tokenizer"))

    if len(dataset["train"]) == 0 or len(dataset["validation"]) == 0:
        raise ValueError("Training and validation datasets must not be empty.")

    if tokenizer.pad_token_id != manifest["padding_token_id"]:
        raise ValueError("The tokenizer padding ID differs from the manifest.")

    if tokenizer.eos_token_id != manifest["eos_token_id"]:
        raise ValueError("The tokenizer EOS ID differs from the manifest.")

    random.seed(arguments.seed)
    torch.manual_seed(arguments.seed)

    if device.type == "cuda":
        torch.cuda.manual_seed_all(arguments.seed)

    generator = torch.Generator().manual_seed(arguments.seed)
    collator = partial(collatesftbatch, paddingtokenid=tokenizer.pad_token_id)
    loaderoptions = {
        "collate_fn": collator,
        "num_workers": arguments.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": arguments.workers > 0,
    }

    trainingloader = DataLoader(
        dataset["train"],
        batch_size=arguments.batchsize,
        shuffle=True,
        generator=generator,
        **loaderoptions,
    )
    validationloader = DataLoader(
        dataset["validation"],
        batch_size=arguments.validationbatchsize,
        shuffle=False,
        **loaderoptions,
    )

    plannedsteps = math.ceil(len(trainingloader) / arguments.accumulation)
    plannedsteps *= arguments.epochs

    if arguments.maxsteps:
        plannedsteps = min(plannedsteps, arguments.maxsteps)

    warmupsteps = math.ceil(plannedsteps * arguments.warmupfraction)

    model = AutoModelForCausalLM.from_pretrained(
        manifest["model_name"],
        revision=manifest["model_revision"],
        torch_dtype=torch.float32,
        attn_implementation="sdpa",
    ).to(device)
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id

    if arguments.checkpointing:
        model.gradient_checkpointing_enable()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=arguments.learningrate,
        weight_decay=arguments.weightdecay,
        fused=device.type == "cuda",
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmupsteps,
        num_training_steps=plannedsteps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    destination.mkdir(parents=True, exist_ok=False)
    metricspath = destination / "metrics.jsonl"
    configuration = {
        "arguments": vars(arguments),
        "data_manifest": manifest,
        "trainable_parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "nominal_effective_batch_size": arguments.batchsize * arguments.accumulation,
        "world_size": 1,
        "planned_optimizer_attempts": plannedsteps,
        "warmup_steps": warmupsteps,
        "sft_reduction": "response-token mean across each accumulation window",
        "checkpoint_selection": "lowest validation response-token mean loss",
        "checkpoint_contents": "model, tokenizer, and metadata; no optimizer resume state",
        "package_versions": {},
    }

    for packagename in ("torch", "transformers", "datasets", "matplotlib"):
        configuration["package_versions"][packagename] = version(packagename)

    writejson(destination / "configuration.json", configuration)
    starttime = time.perf_counter()
    optimizersteps = 0
    attemptedsteps = 0
    skippedsteps = 0
    consecutiveskips = 0
    beststep = 0
    lastvalidationstep = 0
    trainingtokens = 0

    validationloss, validationtokens = evaluatevalidation(model, validationloader, device)
    bestloss = validationloss
    appendmetrics(metricspath, {
        "event": "validation",
        "optimizer_step": 0,
        "validation_loss": validationloss,
        "response_tokens": validationtokens,
        "elapsed_seconds": time.perf_counter() - starttime,
    })
    savecheckpoint(model, tokenizer, destination / "best", {
        "optimizer_step": 0,
        "validation_loss": bestloss,
        "selection": "lowest validation loss",
    })

    model.train()

    with tqdm(total=plannedsteps, desc="SFT optimizer attempts") as progress:
        for epoch in range(arguments.epochs):
            iterator = iter(trainingloader)

            while attemptedsteps < plannedsteps:
                batches = []

                for microbatchnumber in range(arguments.accumulation):
                    batch = next(iterator, None)
                    if batch is None:
                        break
                    batches.append(batch)

                if not batches:
                    break

                usedlearningrate = optimizer.param_groups[0]["lr"]
                windowloss, windowtokens, gradientnorm, updatesucceeded = (
                    trainaccumulationwindow(
                        model, batches, optimizer, scaler, device,
                        arguments.gradientclipping,
                    )
                )
                attemptedsteps += 1
                trainingtokens += windowtokens

                if updatesucceeded:
                    optimizersteps += 1
                    consecutiveskips = 0
                    scheduler.step()
                else:
                    skippedsteps += 1
                    consecutiveskips += 1

                if consecutiveskips >= 10:
                    raise FloatingPointError("Ten consecutive AMP updates were skipped.")

                if (
                    attemptedsteps == 1
                    or attemptedsteps % arguments.logginginterval == 0
                    or attemptedsteps == plannedsteps
                    or not updatesucceeded
                ):
                    appendmetrics(metricspath, {
                        "event": "training",
                        "epoch": epoch + 1,
                        "optimizer_step": optimizersteps,
                        "attempted_step": attemptedsteps,
                        "training_loss": windowloss,
                        "learning_rate": usedlearningrate,
                        "gradient_norm": gradientnorm,
                        "response_tokens": windowtokens,
                        "update_skipped": not updatesucceeded,
                        "elapsed_seconds": time.perf_counter() - starttime,
                    })

                progress.update(1)
                progress.set_postfix(loss=f"{windowloss:.3f}", updates=optimizersteps)

                shouldvalidate = (
                    updatesucceeded
                    and optimizersteps % arguments.evaluationinterval == 0
                ) or attemptedsteps == plannedsteps

                if shouldvalidate:
                    validationloss, validationtokens = evaluatevalidation(
                        model, validationloader, device
                    )
                    lastvalidationstep = optimizersteps
                    appendmetrics(metricspath, {
                        "event": "validation",
                        "optimizer_step": optimizersteps,
                        "validation_loss": validationloss,
                        "response_tokens": validationtokens,
                        "elapsed_seconds": time.perf_counter() - starttime,
                    })

                    if validationloss < bestloss:
                        bestloss = validationloss
                        beststep = optimizersteps
                        savecheckpoint(model, tokenizer, destination / "best", {
                            "optimizer_step": beststep,
                            "validation_loss": bestloss,
                            "selection": "lowest validation loss",
                        })

                    savetrainingplots(metricspath, destination / "loss.png")

            if attemptedsteps >= plannedsteps:
                break

    if lastvalidationstep != optimizersteps:
        raise RuntimeError("The final model has not been validated.")

    savecheckpoint(model, tokenizer, destination / "final", {
        "optimizer_step": optimizersteps,
        "validation_loss": validationloss,
        "selection": "final training state",
    })

    summary = {
        "optimizer_steps": optimizersteps,
        "attempted_steps": attemptedsteps,
        "skipped_steps": skippedsteps,
        "best_optimizer_step": beststep,
        "best_validation_loss": bestloss,
        "final_validation_loss": validationloss,
        "processed_response_tokens": trainingtokens,
        "elapsed_seconds": time.perf_counter() - starttime,
        "peak_allocated_gpu_gib": (
            torch.cuda.max_memory_allocated(device) / 1024 ** 3
            if device.type == "cuda" else None
        ),
    }
    writejson(destination / "summary.json", summary)
    savetrainingplots(metricspath, destination / "loss.png")
    print(json.dumps(summary, indent=2, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description="Train Qwen with response-only SFT.")
    parser.add_argument("--datapath", required=True)
    parser.add_argument("--outputpath", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batchsize", type=int, default=2)
    parser.add_argument("--validationbatchsize", type=int, default=2)
    parser.add_argument("--accumulation", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--maxsteps", type=int, default=0)
    parser.add_argument("--learningrate", type=float, default=2e-5)
    parser.add_argument("--warmupfraction", type=float, default=0.03)
    parser.add_argument("--weightdecay", type=float, default=0.1)
    parser.add_argument("--gradientclipping", type=float, default=1.0)
    parser.add_argument("--evaluationinterval", type=int, default=100)
    parser.add_argument("--logginginterval", type=int, default=20)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpointing", action="store_true")
    trainsft(parser.parse_args())


if __name__ == "__main__":
    main()
