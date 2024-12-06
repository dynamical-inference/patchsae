from src.demo.core import SAETester
from tasks.utils import (
    get_all_classnames,
    get_max_acts_and_images,
    get_sae_and_vit,
    load_datasets,
)


def load_sae_tester(sae_path, include_imagenet=False):
    datasets = load_datasets(include_imagenet=include_imagenet)
    classnames = get_all_classnames(datasets, data_root="./configs/classnames")

    root = "./out/feature_data"
    sae_runname = "sae_base"
    vit_name = "base"

    if include_imagenet is False:
        datasets["imagenet"] = None

    max_act_imgs, mean_acts = get_max_acts_and_images(
        datasets, root, sae_runname, vit_name
    )

    sae_tester = {}

    sae, vit, cfg = get_sae_and_vit(
        sae_path=sae_path,
        vit_type="base",
        device="cpu",
        backbone="openai/clip-vit-base-patch16",
        model_path=None,
        classnames=None,
    )
    sae_clip = SAETester(vit, cfg, sae, mean_acts, max_act_imgs, datasets, classnames)

    sae, vit, cfg = get_sae_and_vit(
        sae_path=sae_path,
        vit_type="maple",
        device="cpu",
        model_path="./data/clip/maple/imagenet/model.pth.tar-2",
        config_path="./configs/models/maple/vit_b16_c2_ep5_batch4_2ctx.yaml",
        backbone="openai/clip-vit-base-patch16",
        classnames=classnames["imagenet"],
    )
    sae_maple = SAETester(vit, cfg, sae, mean_acts, max_act_imgs, datasets, classnames)
    sae_tester["CLIP"] = sae_clip
    sae_tester["MaPLE-imagenet"] = sae_maple
    return sae_tester
