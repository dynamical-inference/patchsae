import json
import os
from collections import defaultdict
from typing import Dict, Tuple

import torch
from datasets import Dataset, load_dataset
from tqdm import tqdm

from src.models.utils import get_adapted_clip, get_base_clip
from src.sae_training.config import Config
from src.sae_training.hooked_vit import HookedVisionTransformer
from src.sae_training.sparse_autoencoder import SparseAutoencoder

# Dataset configurations
DATASET_INFO = {
    "imagenet": {
        "path": "evanarlian/imagenet_1k_resized_256",
        "split": "train",
    },
    "imagenet-sketch": {
        "path": "clip-benchmark/wds_imagenet_sketch",
        "split": "train",
    },
    "oxford_flowers": {
        "path": "nelorth/oxford-flowers",
        "split": "train",
    },
    "caltech101": {
        "path": "HuggingFaceM4/Caltech-101",
        "split": "train",
        "name": "with_background_category",
    },
}

SAE_DIM = 49152


def load_sae(sae_path: str, device: str) -> tuple[SparseAutoencoder, Config]:
    """Load a sparse autoencoder model from a checkpoint file."""
    checkpoint = torch.load(sae_path, map_location="cpu")

    if "cfg" in checkpoint:
        cfg = Config(checkpoint["cfg"])
    else:
        cfg = Config(checkpoint["config"])
    sae = SparseAutoencoder(cfg, device)
    sae.load_state_dict(checkpoint["state_dict"])
    sae.eval().to(device)

    return sae, cfg


def load_hooked_vit(
    cfg: Config,
    vit_type: str,
    backbone: str,
    device: str,
    model_path: str = None,
    config_path: str = None,
    classnames: list[str] = None,
) -> HookedVisionTransformer:
    """Load a vision transformer model with hooks."""
    if vit_type == "base":
        model, processor = get_base_clip(backbone)
    else:
        model, processor = get_adapted_clip(
            cfg, vit_type, model_path, config_path, backbone, classnames
        )

    return HookedVisionTransformer(model, processor, device=device)


def get_sae_and_vit(
    sae_path: str,
    vit_type: str,
    device: str,
    backbone: str,
    model_path: str = None,
    config_path: str = None,
    classnames: list[str] = None,
) -> tuple[SparseAutoencoder, HookedVisionTransformer, Config]:
    """Load both SAE and ViT models."""
    sae, cfg = load_sae(sae_path, device)
    vit = load_hooked_vit(
        cfg, vit_type, backbone, device, model_path, config_path, classnames
    )
    return sae, vit, cfg


def load_and_organize_dataset(dataset_name: str) -> Tuple[list, Dict]:
    # TODO: ERR for imagenet (gets killed after 75%)
    """
    Load dataset and organize data by class.
    Return classnames and data by class.
    Requried for classification_with_top_k_masking.py and compute_class_wise_sae_activation.py
    """
    dataset = load_dataset(**DATASET_INFO[dataset_name])
    classnames = get_classnames(dataset_name, dataset)

    data_by_class = defaultdict(list)
    for data_item in tqdm(dataset):
        classname = classnames[data_item["label"]]
        data_by_class[classname].append(data_item)

    return classnames, data_by_class


def get_classnames(
    dataset_name: str, dataset: Dataset = None, data_root: str = "./configs/classnames"
) -> list[str]:
    """Get class names for a dataset."""

    filename = f"{data_root}/{dataset_name}_classnames"
    txt_filename = filename + ".txt"
    json_filename = filename + ".json"

    if not os.path.exists(txt_filename) and not os.path.exists(json_filename):
        raise ValueError(f"Dataset {dataset_name} not supported")

    filename = json_filename if os.path.exists(json_filename) else txt_filename

    with open(filename, "r") as file:
        if dataset_name == "caltech101":
            class_names = [line.strip() for line in file.readlines()]
        elif dataset_name == "imagenet" or dataset_name == "imagenet-sketch":
            class_names = [
                " ".join(line.strip().split(" ")[1:]) for line in file.readlines()
            ]
        elif dataset_name == "oxford_flowers":
            assert dataset is not None, "Dataset must be provided for Oxford Flowers"
            new_class_dict = {}
            class_names = json.load(file)
            classnames_from_hf = dataset.features["label"].names
            for i, class_name in enumerate(classnames_from_hf):
                new_class_dict[i] = class_names[class_name]
            class_names = list(new_class_dict.values())

        else:
            raise ValueError(f"Dataset {dataset_name} not supported")

    return class_names


def setup_save_directory(
    root_dir: str, save_name: str, sae_path: str, vit_type: str, dataset_name: str
) -> str:
    """Set and create the save directory path."""
    sae_run_name = sae_path.split("/")[-2]
    save_directory = (
        f"{root_dir}/{save_name}/sae_{sae_run_name}/{vit_type}/{dataset_name}"
    )
    os.makedirs(save_directory, exist_ok=True)
    return save_directory


def get_sae_activations(
    model_activations: torch.Tensor, sae: SparseAutoencoder
) -> torch.Tensor:
    """Extract and process activations from the sparse autoencoder."""
    hook_name = "hook_hidden_post"

    # Run SAE forward pass and get activations from cache
    _, cache = sae.run_with_cache(model_activations)
    sae_activations = cache[hook_name]

    # Average across sequence length dimension if needed
    if len(sae_activations.size()) > 2:
        sae_activations = sae_activations.mean(dim=1)

    return sae_activations


def process_batch(vit, batch_data, device):
    """Process a single batch of images."""
    images = [data["image"] for data in batch_data]

    inputs = vit.processor(
        images=images, text="", return_tensors="pt", padding=True
    ).to(device)
    return inputs


def get_max_acts_and_images(
    datasets: dict, feat_data_root: str, sae_runname: str, vit_name: str
) -> tuple[dict, dict]:
    """Load and return maximum activations and mean activations for each dataset."""
    max_act_imgs = {}
    mean_acts = {}

    for dataset_name in datasets:
        # Load max activating image indices
        max_act_path = os.path.join(
            feat_data_root,
            f"{sae_runname}/{vit_name}/{dataset_name}",
            "max_activating_image_indices.pt",
        )
        max_act_imgs[dataset_name] = torch.load(max_act_path, map_location="cpu").to(
            torch.int32
        )

        # Load mean activations
        mean_acts_path = os.path.join(
            feat_data_root,
            f"{sae_runname}/{vit_name}/{dataset_name}",
            "sae_mean_acts.pt",
        )
        mean_acts[dataset_name] = torch.load(mean_acts_path, map_location="cpu").numpy()

    return max_act_imgs, mean_acts


def load_datasets(include_imagenet: bool = False, seed: int = 1):
    """Load multiple datasets from HuggingFace."""
    if include_imagenet:
        return {
            "imagenet": load_dataset(
                "evanarlian/imagenet_1k_resized_256", split="train"
            ).shuffle(seed=seed),
            "imagenet-sketch": load_dataset(
                "clip-benchmark/wds_imagenet_sketch", split="test"
            ).shuffle(seed=seed),
            "caltech101": load_dataset(
                "HuggingFaceM4/Caltech-101",
                "with_background_category",
                split="train",
            ).shuffle(seed=seed),
        }
    else:
        return {
            "imagenet-sketch": load_dataset(
                "clip-benchmark/wds_imagenet_sketch", split="test"
            ).shuffle(seed=seed),
            "caltech101": load_dataset(
                "HuggingFaceM4/Caltech-101",
                "with_background_category",
                split="train",
            ).shuffle(seed=seed),
        }


def get_all_classnames(datasets, data_root):
    """Get class names for all datasets."""
    class_names = {}
    for dataset_name, dataset in datasets.items():
        class_names[dataset_name] = get_classnames(dataset_name, dataset, data_root)

    # imagenet classnames are required to classnames for maple
    if "imagenet" not in class_names:
        filename = f"{data_root}/imagenet_classnames"
        txt_filename = filename + ".txt"
        json_filename = filename + ".json"

        if not os.path.exists(txt_filename) and not os.path.exists(json_filename):
            raise ValueError(f"Dataset {dataset_name} not supported")

        filename = json_filename if os.path.exists(json_filename) else txt_filename

        with open(filename, "r") as file:
            class_names["imagenet"] = [
                " ".join(line.strip().split(" ")[1:]) for line in file.readlines()
            ]

    return class_names
