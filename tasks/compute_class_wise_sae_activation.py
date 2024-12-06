import argparse
import os
from typing import Dict

import numpy as np
import torch
from tqdm import tqdm

from src.sae_training.config import Config
from src.sae_training.hooked_vit import HookedVisionTransformer
from src.sae_training.sparse_autoencoder import SparseAutoencoder
from src.sae_training.utils import get_model_activations
from tasks.utils import (
    SAE_DIM,
    get_sae_and_vit,
    load_and_organize_dataset,
    process_batch,
    setup_save_directory,
)


def get_sae_activations(
    model_activations: torch.Tensor, sae: SparseAutoencoder, threshold: float
) -> torch.Tensor:
    """Get binary SAE activations above threshold.

    Args:
        model_activations: Input activations from vision transformer
        sae: The sparse autoencoder model
        threshold: Activation threshold

    Returns:
        Binary tensor indicating which features were active
    """
    _, cache = sae.run_with_cache(model_activations)
    activations = cache["hook_hidden_post"] > threshold
    return activations.sum(dim=0).sum(dim=0)


def process_class_batch(
    batch_data: list,
    sae: SparseAutoencoder,
    vit: HookedVisionTransformer,
    cfg: Config,
    threshold: float,
    device: str,
) -> np.ndarray:
    """Process a single batch of class data and get SAE feature activations."""
    batch_inputs = process_batch(vit, batch_data, device)
    transformer_activations = get_model_activations(
        vit, batch_inputs, cfg.block_layer, cfg.module_name, cfg.class_token
    )
    active_features = get_sae_activations(transformer_activations, sae, threshold)
    return active_features.cpu().numpy()


def compute_class_feature_counts(
    class_data: list,
    sae: SparseAutoencoder,
    vit: HookedVisionTransformer,
    cfg: Config,
    batch_size: int,
    threshold: float,
    device: str,
) -> np.ndarray:
    """Compute SAE feature activation counts for a single class."""
    feature_counts = np.zeros(SAE_DIM)
    num_batches = (len(class_data) + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min((batch_idx + 1) * batch_size, len(class_data))
        batch_data = class_data[batch_start:batch_end]

        batch_counts = process_class_batch(batch_data, sae, vit, cfg, threshold, device)
        feature_counts += batch_counts
        torch.cuda.empty_cache()

    return feature_counts


def compute_all_class_activations(
    classnames: list,
    data_by_class: Dict,
    sae: SparseAutoencoder,
    vit: HookedVisionTransformer,
    cfg: Config,
    batch_size: int,
    threshold: float,
    device: str,
) -> np.ndarray:
    """Compute SAE activation counts across all classes."""
    class_activation_counts = np.zeros((len(classnames), SAE_DIM))

    for class_idx, classname in enumerate(tqdm(classnames)):
        class_data = data_by_class[classname]
        class_counts = compute_class_feature_counts(
            class_data, sae, vit, cfg, batch_size, threshold, device
        )
        class_activation_counts[class_idx] = class_counts

    return class_activation_counts


def main(
    sae_path: str,
    vit_type: str,
    device: str,
    dataset_name: str,
    root_dir: str,
    save_name: str,
    backbone: str = "openai/clip-vit-base-patch16",
    batch_size: int = 8,
    model_path: str = None,
    config_path: str = None,
    threshold: float = 0.2,
):
    """Main function to compute and save class-wise SAE activation counts."""

    save_directory = setup_save_directory(
        root_dir, save_name, sae_path, vit_type, dataset_name
    )

    classnames, data_by_class = load_and_organize_dataset(dataset_name)

    sae, vit, cfg = get_sae_and_vit(
        sae_path,
        vit_type,
        device,
        backbone,
        model_path=model_path,
        config_path=config_path,
        classnames=classnames,
    )

    class_activation_counts = compute_all_class_activations(
        classnames, data_by_class, sae, vit, cfg, batch_size, threshold, device
    )

    # Save results
    save_path = os.path.join(save_directory, "cls_sae_cnt.npy")
    np.save(save_path, class_activation_counts)
    print(f"Class activation counts saved at {save_directory}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save class-wise SAE activation count")
    parser.add_argument("--root_dir", type=str, default=".", help="Root directory")
    parser.add_argument("--dataset_name", type=str, default="imagenet")
    parser.add_argument(
        "--sae_path", type=str, required=True, help="SAE ckpt path (ends with xxx.pt)"
    )
    parser.add_argument(
        "--vit_type", type=str, default="base", help="choose between [base, maple]"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.2, help="threshold for SAE activation"
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument(
        "--model_path",
        type=str,
        help="CLIP model path in the case of not using the default",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        help="CLIP config path in the case of using maple",
    )
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    main(
        sae_path=args.sae_path,
        vit_type=args.vit_type,
        device=args.device,
        dataset_name=args.dataset_name,
        root_dir=args.root_dir,
        save_name="out/feature_data",
        batch_size=args.batch_size,
        model_path=args.model_path,
        config_path=args.config_path,
        threshold=args.threshold,
    )
