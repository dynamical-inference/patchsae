import os
import pickle
from functools import partial

import torch
from transformers import CLIPModel, CLIPProcessor

from src.models.clip import clip
from src.models.config.maple import get_maple_config


def load_clip_model(cfg, model_type: str):
    """Load and configure a CLIP model."""
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    design_details = {
        "trainer": model_type,
        "vision_depth": 0,
        "language_depth": 0,
        "vision_ctx": 0,
        "language_ctx": 0,
    }

    if model_type == "maple":
        design_details["trainer"] = "MaPLe"
        design_details["maple_length"] = cfg.TRAINER.MAPLE.N_CTX

    model = clip.build_model(state_dict or model.state_dict(), design_details)
    return model


def load_checkpoint(fpath: str) -> dict:
    """Load a model checkpoint file.

    Handles loading both Python 3 and Python 2 saved checkpoints by catching UnicodeDecodeError.

    Args:
        fpath: Path to the checkpoint file

    Returns:
        The loaded checkpoint dictionary

    Raises:
        ValueError: If fpath is None
        FileNotFoundError: If checkpoint file does not exist
        Exception: If checkpoint cannot be loaded

    Examples:
        >>> checkpoint = load_checkpoint('models/checkpoint.pth')
    """
    if fpath is None:
        raise ValueError("Checkpoint path cannot be None")

    if not os.path.exists(fpath):
        raise FileNotFoundError(f"No checkpoint file found at {fpath}")

    device = None if torch.cuda.is_available() else "cpu"

    try:
        return torch.load(fpath, map_location=device)

    except UnicodeDecodeError:
        # Handle Python 2 checkpoints
        pickle.load = partial(pickle.load, encoding="latin1")
        pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
        return torch.load(fpath, pickle_module=pickle, map_location=device)

    except Exception as e:
        raise Exception(f"Failed to load checkpoint from {fpath}: {str(e)}")


def _remove_prompt_learner_tokens(state_dict: dict) -> dict:
    """Remove prompt learner token vectors from state dict."""
    token_keys = ["prompt_learner.token_prefix", "prompt_learner.token_suffix"]
    for key in token_keys:
        if key in state_dict:
            del state_dict[key]
    return state_dict


def load_state_dict_without_prompt_learner(ckpt_path: str) -> dict:
    """Load checkpoint and remove prompt learner token vectors."""
    checkpoint = load_checkpoint(ckpt_path)
    state_dict = checkpoint["state_dict"]
    return _remove_prompt_learner_tokens(state_dict)


def get_base_clip(backbone: str) -> tuple[CLIPModel, CLIPProcessor]:
    """Load base CLIP model and processor."""
    model = CLIPModel.from_pretrained(backbone)
    processor = CLIPProcessor.from_pretrained(backbone)
    return model, processor


def get_adapted_clip(
    cfg,
    model_type: str,
    model_path: str,
    config_path: str,
    backbone: str,
    classnames: list[str],
) -> tuple[CLIPModel, CLIPProcessor]:
    """Load and configure adapted CLIP model with custom prompt learning.

    Args:
        cfg: Model configuration
        model_type: Type of prompt learning ('maple')
        model_path: Path to model checkpoint
        classnames: Optional list of class names

    Returns:
        Tuple of (model, processor)
    """
    if model_type == "maple":
        cfg = get_maple_config(custom_clip_cfg=config_path)

    clip_model = load_clip_model(cfg, model_type)
    model_statedict = load_state_dict_without_prompt_learner(model_path)

    model_types = {
        "maple": "src.models.architecture.maple",
    }

    if model_type not in model_types:
        raise ValueError(f"Unsupported model type: {model_type}")

    module = __import__(model_types[model_type], fromlist=["CustomCLIP"])
    model = module.CustomCLIP(cfg, classnames, clip_model)

    model.load_state_dict(model_statedict, strict=False)

    processor = CLIPProcessor.from_pretrained(backbone)
    return model, processor
