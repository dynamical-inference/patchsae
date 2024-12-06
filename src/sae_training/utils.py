import math
from typing import Dict, Optional

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from src.sae_training.hooked_vit import HookedVisionTransformer

SAE_DIM = 49152


def process_model_inputs(
    batch: Dict, vit: HookedVisionTransformer, device: str, process_labels: bool = False
) -> torch.Tensor:
    """Process input images through the ViT processor."""
    if process_labels:
        labels = [f"A photo of a {label}" for label in batch["label"]]
        return vit.processor(
            images=batch["image"], text=labels, return_tensors="pt", padding=True
        ).to(device)

    return vit.processor(
        images=batch["image"], text="", return_tensors="pt", padding=True
    ).to(device)


def get_model_activations(
    model: HookedVisionTransformer, inputs: dict, block_layer, module_name, class_token
) -> torch.Tensor:
    """Extract activations from a specific layer of the vision transformer model."""
    hook_location = (block_layer, module_name)

    # Run model forward pass and extract activations from cache
    _, cache = model.run_with_cache([hook_location], **inputs)
    activations = cache[hook_location]

    batch_size = inputs["pixel_values"].shape[0]
    if activations.shape[0] != batch_size:
        activations = activations.transpose(0, 1)

    # Extract class token if specified
    if class_token:
        activations = activations[0, :, :]

    return activations


def get_scheduler(scheduler_name: Optional[str], optimizer: optim.Optimizer, **kwargs):
    def get_warmup_lambda(warm_up_steps, training_steps):
        def lr_lambda(steps):
            if steps < warm_up_steps:
                return (steps + 1) / warm_up_steps
            else:
                return (training_steps - steps) / (training_steps - warm_up_steps)

        return lr_lambda

    # heavily derived from hugging face although copilot helped.
    def get_warmup_cosine_lambda(warm_up_steps, training_steps, lr_end):
        def lr_lambda(steps):
            if steps < warm_up_steps:
                return (steps + 1) / warm_up_steps
            else:
                progress = (steps - warm_up_steps) / (training_steps - warm_up_steps)
                return lr_end + 0.5 * (1 - lr_end) * (1 + math.cos(math.pi * progress))

        return lr_lambda

    if scheduler_name is None or scheduler_name.lower() == "constant":
        return lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda steps: 1.0)
    elif scheduler_name.lower() == "constantwithwarmup":
        warm_up_steps = kwargs.get("warm_up_steps", 500)
        return lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda steps: min(1.0, (steps + 1) / warm_up_steps),
        )
    elif scheduler_name.lower() == "linearwarmupdecay":
        warm_up_steps = kwargs.get("warm_up_steps", 0)
        training_steps = kwargs.get("training_steps")
        lr_lambda = get_warmup_lambda(warm_up_steps, training_steps)
        return lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif scheduler_name.lower() == "cosineannealing":
        training_steps = kwargs.get("training_steps")
        eta_min = kwargs.get("lr_end", 0)
        return lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=training_steps, eta_min=eta_min
        )
    elif scheduler_name.lower() == "cosineannealingwarmup":
        warm_up_steps = kwargs.get("warm_up_steps", 0)
        training_steps = kwargs.get("training_steps")
        eta_min = kwargs.get("lr_end", 0)
        lr_lambda = get_warmup_cosine_lambda(warm_up_steps, training_steps, eta_min)
        return lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif scheduler_name.lower() == "cosineannealingwarmrestarts":
        training_steps = kwargs.get("training_steps")
        eta_min = kwargs.get("lr_end", 0)
        num_cycles = kwargs.get("num_cycles", 1)
        T_0 = training_steps // num_cycles
        return lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=T_0, eta_min=eta_min
        )
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")
