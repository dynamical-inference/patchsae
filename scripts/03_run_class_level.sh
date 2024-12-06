#! /bin/bash

PYTHONPATH=./ nohup python -u tasks/compute_class_wise_sae_activation.py \
    --root_dir ./ \
    --dataset_name imagenet \
    --threshold 0.2 \
    --sae_path /patchsae_checkpoints/YOUR-OWN-PATH/clip-vit-base-patch16_-2_resid_49152.pt \
    --vit_type base > logs/03_test_class_level.txt 2> logs/03_test_class_level.err &
