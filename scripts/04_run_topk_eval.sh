#! /bin/bash

PYTHONPATH=./ python tasks/classification_with_top_k_masking.py \
    --root_dir ./ \
    --dataset_name imagenet \
    --sae_path patchsae_checkpoints/YOUR-OWN-PATH/clip-vit-base-patch16_-2_resid_49152.pt \
    --cls_wise_sae_activation_path ./out/feature_data/sae_openai/base/imagenet/cls_sae_cnt.npy \
    --vit_type base \
    > logs/04_test_topk_eval.txt
