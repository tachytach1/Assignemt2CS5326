from __future__ import annotations

import math
from numbers import Real

import torch
import torch.nn.functional as functional

from alignment.batching import build_preference_batch
from alignment.scoring import compute_batch_response_statistics


def validatepreferencescores(
    policychosenscores,
    policyrejectedscores,
    referencechosenscores,
    referencerejectedscores,
    preferencestrength,
):
    namedscores = {
        "policy chosen scores": policychosenscores,
        "policy rejected scores": policyrejectedscores,
        "reference chosen scores": referencechosenscores,
        "reference rejected scores": referencerejectedscores,
    }

    expectedshape = None

    for scorename, scoretensor in namedscores.items():
        if not isinstance(scoretensor, torch.Tensor):
            raise TypeError(f"{scorename} must be a torch.Tensor.")

        if scoretensor.ndim != 1:
            raise ValueError(f"{scorename} must have shape [batch].")

        if scoretensor.numel() == 0:
            raise ValueError(f"{scorename} must not be empty.")

        if not scoretensor.is_floating_point():
            raise TypeError(f"{scorename} must use a floating-point dtype.")

        if expectedshape is None:
            expectedshape = scoretensor.shape
        elif scoretensor.shape != expectedshape:
            raise ValueError("All preference scores must have identical shapes.")

    if policychosenscores.device != policyrejectedscores.device:
        raise ValueError("Chosen and rejected policy scores must share a device.")

    if (
        isinstance(preferencestrength, bool)
        or not isinstance(preferencestrength, Real)
    ):
        raise TypeError("Preference strength must be a real number.")

    if (
        not math.isfinite(float(preferencestrength))
        or preferencestrength < 0
    ):
        raise ValueError("Preference strength must be finite and nonnegative.")


def promotepreferencescoreprecision(sequencescores):
    if sequencescores.dtype in (
        torch.float16,
        torch.bfloat16,
    ):
        return sequencescores.float()

    return sequencescores


def computepreferencerewards(
    policychosenscores,
    policyrejectedscores,
    referencechosenscores,
    referencerejectedscores,
    preferencestrength,
):
    validatepreferencescores(
        policychosenscores,
        policyrejectedscores,
        referencechosenscores,
        referencerejectedscores,
        preferencestrength,
    )

    policydevice = policychosenscores.device

    chosenpolicyscores = promotepreferencescoreprecision(
        policychosenscores
    )
    rejectedpolicyscores = promotepreferencescoreprecision(
        policyrejectedscores
    )

    chosenreferencescores = promotepreferencescoreprecision(
        referencechosenscores.detach()
    ).to(device=policydevice)

    rejectedreferencescores = promotepreferencescoreprecision(
        referencerejectedscores.detach()
    ).to(device=policydevice)

    chosenrewards = preferencestrength * (
        chosenpolicyscores - chosenreferencescores
    )
    rejectedrewards = preferencestrength * (
        rejectedpolicyscores - rejectedreferencescores
    )

    return chosenrewards, rejectedrewards


def computepreferencelossandmetrics(
    policychosenscores,
    policyrejectedscores,
    referencechosenscores,
    referencerejectedscores,
    preferencestrength,
):
    chosenrewards, rejectedrewards = computepreferencerewards(
        policychosenscores,
        policyrejectedscores,
        referencechosenscores,
        referencerejectedscores,
        preferencestrength,
    )

    rewarddifferences = chosenrewards - rejectedrewards
    preferenceloss = -functional.logsigmoid(rewarddifferences).mean()

    detachedchosenrewards = chosenrewards.detach()
    detachedrejectedrewards = rejectedrewards.detach()

    preferencemetrics = {
        "dpo_loss": preferenceloss.detach(),
        "reward_accuracy": (
            detachedchosenrewards > detachedrejectedrewards
        ).float().mean(),
        "reward_margin": rewarddifferences.detach().mean(),
        "chosen_reward": detachedchosenrewards.mean(),
        "rejected_reward": detachedrejectedrewards.mean(),
    }

    return preferenceloss, preferencemetrics


def computepreferencelossfromscores(
    policychosenscores,
    policyrejectedscores,
    referencechosenscores,
    referencerejectedscores,
    preferencestrength,
):
    chosenrewards, rejectedrewards = computepreferencerewards(
        policychosenscores,
        policyrejectedscores,
        referencechosenscores,
        referencerejectedscores,
        preferencestrength,
    )

    return -functional.logsigmoid(
        chosenrewards - rejectedrewards
    ).mean()


def computebatchpreferencelossandmetrics(
    policymodel,
    referencemodel,
    preferencebatch,
    preferencestrength,
):
    if policymodel is referencemodel:
        raise ValueError("Policy and reference must be separate model objects.")

    if "chosen" not in preferencebatch or "rejected" not in preferencebatch:
        raise ValueError("The preference batch must contain chosen and rejected.")

    chosenbatch = preferencebatch["chosen"]
    rejectedbatch = preferencebatch["rejected"]

    policychosenstatistics = compute_batch_response_statistics(
        policymodel,
        chosenbatch,
    )
    policyrejectedstatistics = compute_batch_response_statistics(
        policymodel,
        rejectedbatch,
    )

    policychosenscores = policychosenstatistics[0]
    policyrejectedscores = policyrejectedstatistics[0]

    with torch.no_grad():
        referencechosenstatistics = compute_batch_response_statistics(
            referencemodel,
            chosenbatch,
        )
        referencerejectedstatistics = compute_batch_response_statistics(
            referencemodel,
            rejectedbatch,
        )

    referencechosenscores = referencechosenstatistics[0]
    referencerejectedscores = referencerejectedstatistics[0]

    return computepreferencelossandmetrics(
        policychosenscores,
        policyrejectedscores,
        referencechosenscores,
        referencerejectedscores,
        preferencestrength,
    )


def computeresponseonlydpoloss(
    policymodel,
    referencemodel,
    tokenizer,
    prompts,
    chosenresponses,
    rejectedresponses,
    preferencestrength,
    maximumsequencelength,
):
    preferencebatch = build_preference_batch(
        tokenizer,
        prompts,
        chosenresponses,
        rejectedresponses,
        maximumsequencelength,
    )

    preferenceresult = computebatchpreferencelossandmetrics(
        policymodel,
        referencemodel,
        preferencebatch,
        preferencestrength,
    )

    return preferenceresult[0]