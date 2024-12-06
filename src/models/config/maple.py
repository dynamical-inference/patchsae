from yacs.config import CfgNode

from src.models.config.default_config import get_default_config


def get_maple_config(custom_clip_cfg=None):
    """Get configuration for MaPLe model.

    Args:
        custom_clip_cfg: Optional custom CLIP config

    Returns:
        Config object with MaPLe settings
    """
    cfg = get_default_config()
    cfg.TRAINER = CfgNode()
    cfg.TRAINER.MAPLE = CfgNode()
    cfg.TRAINER.MAPLE.N_CTX = 2  # number of context vectors at the vision branch
    cfg.TRAINER.MAPLE.CTX_INIT = (
        "a photo of a"  # initialization words (only for language prompts)
    )
    cfg.TRAINER.MAPLE.PREC = "fp16"  # fp16, fp32, amp
    # If both variables below are set to 0, 0, will the config will degenerate to COOP model
    cfg.TRAINER.MAPLE.PROMPT_DEPTH = (
        9  # Max 12, minimum 0, for 0 it will act as shallow IVLP prompting (J=1)
    )
    cfg.DATASET = CfgNode()
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new

    cfg.TRAINER.NAME = "MaPLe"
    cfg.MODEL = CfgNode()
    cfg.MODEL.BACKBONE = CfgNode()
    cfg.MODEL.BACKBONE.NAME = "ViT-B/16"

    cfg.merge_from_file(custom_clip_cfg)

    return cfg
