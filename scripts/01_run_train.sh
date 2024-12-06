#! /bin/bash

PYTHONPATH=./ nohup python -u tasks/train_sae_vit.py \
--batch_size 128 \
--checkpoint_path patchsae_checkpoints \
--n_checkpoints 10 \
--use_ghost_grads \
--log_to_wandb --wandb_project patchsae_test --wandb_entity hyesulim-hs \
> logs/01_test_training.txt
