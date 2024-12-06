#! /bin/bash

PYTHONPATH=./ nohup python -u tasks/compute_sae_feature_data.py \
    --root_dir ./ \
    --dataset_name imagenet \
    --sae_path patchsae_checkpoints/YOUR-OWN-PATH/clip-vit-base-patch16_-2_resid_49152.pt \
    --vit_type base > logs/02_test_extract.txt
