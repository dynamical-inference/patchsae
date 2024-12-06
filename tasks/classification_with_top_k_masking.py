import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from src.models.templates.openai_imagenet_templates import openai_imagenet_template
from src.sae_training.config import Config
from src.sae_training.hooked_vit import Hook, HookedVisionTransformer
from src.sae_training.sparse_autoencoder import SparseAutoencoder
from tasks.utils import (
    SAE_DIM,
    get_sae_and_vit,
    load_and_organize_dataset,
    process_batch,
    setup_save_directory,
)

TOPK_LIST = [1, 2, 5, 10, 50, 100, 500, 1000, 2000, SAE_DIM]
SAE_BIAS = -0.105131256516992


def calculate_text_features(model, device, classnames):
    """Calculate mean text features across templates for each class."""
    mean_text_features = 0

    for template_fn in openai_imagenet_template:
        # Generate prompts and convert to token IDs
        prompts = [template_fn(c) for c in classnames]
        prompt_ids = [
            model.processor(
                text=p, return_tensors="pt", padding=False, truncation=True
            ).input_ids[0]
            for p in prompts
        ]

        # Process batch
        padded_prompts = pad_sequence(prompt_ids, batch_first=True).to(device)

        # Get text features
        with torch.no_grad():
            text_features = model.model.get_text_features(padded_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            mean_text_features += text_features

    return mean_text_features / len(openai_imagenet_template)


def create_sae_hooks(vit_type, cfg, cls_features, sae, device, hook_type="on"):
    """Create SAE hooks based on model type and hook type."""
    # Setup clamping parameters
    clamp_feat_dim = torch.ones(SAE_DIM).bool()
    clamp_value = torch.zeros(SAE_DIM) if hook_type == "on" else torch.ones(SAE_DIM)
    clamp_value = clamp_value.to(device)
    clamp_value[cls_features] = 1.0 if hook_type == "on" else 0.0

    def process_activations(activations, is_maple=False):
        """Helper function to process activations with SAE"""
        act = activations.transpose(0, 1) if is_maple else activations
        processed = (
            sae.forward_clamp(
                act[:, :, :], clamp_feat_dim=clamp_feat_dim, clamp_value=clamp_value
            )[0]
            - SAE_BIAS
        )
        return processed

    def hook_fn_default(activations):
        activations[:, :, :] = process_activations(activations)
        return (activations,)

    def hook_fn_maple(activations):
        activations = process_activations(activations, is_maple=True)
        return activations.transpose(0, 1)

    # Create appropriate hook based on model type
    hook_fn = hook_fn_maple if vit_type == "maple" else hook_fn_default
    is_custom = vit_type == "maple"

    return [
        Hook(
            cfg.block_layer,
            cfg.module_name,
            hook_fn,
            return_module_output=False,
            is_custom=is_custom,
        )
    ]


def get_predictions(vit, inputs, text_features, vit_type, hooks=None):
    """Get model predictions with optional hooks."""
    with torch.no_grad():
        if hooks:
            vit_out = vit.run_with_hooks(hooks, return_type="output", **inputs)
        else:
            vit_out = vit(return_type="output", **inputs)

        image_features = vit_out.image_embeds if vit_type == "base" else vit_out
        logit_scale = vit.model.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
        preds = logits.argmax(dim=-1)

    return preds.cpu().numpy().tolist()


def classify_with_top_k_masking(
    class_data: list,
    cls_idx: int,
    sae: SparseAutoencoder,
    vit: HookedVisionTransformer,
    cls_sae_cnt: torch.Tensor,
    text_features: torch.Tensor,
    batch_size: int,
    device: str,
    vit_type: str,
    cfg: Config,
):
    """Classify images with top-k feature masking."""
    num_batches = (len(class_data) + batch_size - 1) // batch_size

    preds_dict = defaultdict(list)

    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min((batch_idx + 1) * batch_size, len(class_data))
        batch_data = class_data[batch_start:batch_end]

        batch_inputs = process_batch(vit, batch_data, device)

        # Get predictions without SAE
        preds_dict["no_sae"].extend(
            get_predictions(vit, batch_inputs, text_features, vit_type)
        )
        torch.cuda.empty_cache()

        # Get top features for current class
        loaded_cls_sae_idx = cls_sae_cnt[cls_idx].argsort()[::-1]

        for topk in TOPK_LIST:
            cls_features = loaded_cls_sae_idx[:topk].tolist()

            # Get predictions with features ON
            hooks_on = create_sae_hooks(vit_type, cfg, cls_features, sae, device, "on")
            preds_dict[f"on_{topk}"].extend(
                get_predictions(vit, batch_inputs, text_features, vit_type, hooks_on)
            )
            torch.cuda.empty_cache()

            # Get predictions with features OFF
            hooks_off = create_sae_hooks(
                vit_type, cfg, cls_features, sae, device, "off"
            )
            preds_dict[f"off_{topk}"].extend(
                get_predictions(vit, batch_inputs, text_features, vit_type, hooks_off)
            )
            torch.cuda.empty_cache()

    return preds_dict


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
    cls_wise_sae_activation_path: str = None,
):
    class_feature_type = cls_wise_sae_activation_path.split("/")[-3]
    save_directory = setup_save_directory(
        root_dir, save_name, sae_path, f"{class_feature_type}_{vit_type}", dataset_name
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

    cls_sae_cnt = np.load(cls_wise_sae_activation_path)

    if vit_type == "base":
        text_features = calculate_text_features(vit, device, classnames)
    else:
        text_features = vit.model.get_text_features()

    metrics_dict = {}
    for class_idx, classname in enumerate(tqdm(classnames)):
        preds_dict = classify_with_top_k_masking(
            data_by_class[classname],
            class_idx,
            sae,
            vit,
            cls_sae_cnt,
            text_features,
            batch_size,
            device,
            vit_type,
            cfg,
        )

        metrics_dict[class_idx] = {}
        for k, v in preds_dict.items():
            metrics_dict[class_idx][k] = v.count(class_idx) / len(v) * 100

    metrics_df = pd.DataFrame(metrics_dict)
    metrics_df.to_csv(f"{save_directory}/metrics.csv", index=False)
    print(f"metrics.csv saved at {save_directory}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform classification with top-k masking"
    )
    parser.add_argument("--root_dir", type=str, default=".", help="Root directory")
    parser.add_argument("--dataset_name", type=str, default="imagenet")
    parser.add_argument(
        "--sae_path", type=str, required=True, help="SAE ckpt path (ends with xxx.pt)"
    )
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument(
        "--cls_wise_sae_activation_path",
        type=str,
        help="path for cls_sae_cnt.npy",
    )
    parser.add_argument(
        "--vit_type", type=str, default="base", help="choose between [base, maple]"
    )
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
        cls_wise_sae_activation_path=args.cls_wise_sae_activation_path,
    )
