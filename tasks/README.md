# PatchSAE: Training & Usage Guide

## 📋 Table of Contents
- [Train SAE](#-train-sae)
- [Extract SAE Latent Data](#-extract-sae-latent-data)
- [Compute Class-Level SAE Latents](#-compute-class-level-sae-latents)
- [Steer Classification](#-steer-classification)

## 🔧 Train SAE

Train a sparse autoencoder on CLIP features.

### Prerequisites
- CLIP checkpoint
- Training dataset (images only)

### Training Command
```bash
PYTHONPATH=./ python tasks/train_sae_vit.py
```
> 📝 Configuration files will be added soon

### Outputs
- SAE checkpoint (`.pt` file)

### Monitoring
- View our [training logs on W&B](https://api.wandb.ai/links/hyesulim-hs/7dx90sq0)

---

## 📊 Extract SAE Latent Data

Extract and save SAE latent activations for downstream analysis.

### Prerequisites
- CLIP checkpoint
- SAE checkpoint (from training step)
- Dataset (can differ from training dataset)

### Run with Original CLIP
```bash
PYTHONPATH=./ python tasks/compute_sae_feature_data.py \
    --root_dir ./ \
    --dataset_name imagenet \
    --sae_path /PATH/TO/SAE_CKPT.pt \
    --vit_type base
```

### Run with Adapted CLIP (e.g., MaPLe)
1. Download MaPLe from the [official repo](https://github.com/muzairkhattak/multimodal-prompt-learning/tree/main?tab=readme-ov-file#model-zoo) or [Google Drive](https://drive.google.com/drive/folders/1EvuvgR8566bL0T7ucvAL3LFVwuUPMRas)

2. Run extraction:
```bash
PYTHONPATH=./ python tasks/compute_sae_feature_data.py \
    --root_dir ./ \
    --dataset_name imagenet \
    --sae_path /PATH/TO/SAE_CKPT.pt \
    --vit_type maple \
    --model_path /PATH/TO/MAPLE_CKPT \  # e.g., .../model.pth.tar-5
    --config_path /PATH/TO/MAPLE_CFG \  # e.g., .../configs/models/maple/vit_b16_c2_ep5_batch4_2ctx.yaml
```

### Output Files
All files will be saved to: `{root_dir}/out/feature_data/{vit_type}/{dataset_name}/`

- `max_activating_image_indices.pt`
- `max_activating_image_label_indices.pt`
- `max_activating_image_values.pt`
- `sae_mean_acts.pt`
- `sae_sparsity.pt`

### Analysis
Explore the extracted features with our [patchsae/analysis/analysis.ipynb](https://github.com/hyesulim/patchsae/blob/9f28fdc6ffb7beccb5c2b8ee629b6752b904aa23/analysis/analysis.ipynb)

---

## 🧩 Compute Class-Level SAE Latents

Compute class-level SAE activation patterns.

### Prerequisites
- CLIP checkpoint
- SAE checkpoint
- SAE feature data (from previous step)
- Dataset (must be the SAME dataset used in the extraction step)

### Run with Original CLIP
```bash
PYTHONPATH=./ python tasks/compute_class_wise_sae_activation.py \
    --root_dir ./ \
    --dataset_name imagenet \
    --threshold 0.2 \
    --sae_path /PATH/TO/SAE_CKPT.pt \
    --vit_type base
```

### Run with Adapted CLIP (e.g., MaPLe)
```bash
PYTHONPATH=./ python tasks/compute_class_wise_sae_activation.py \
    --root_dir ./ \
    --dataset_name imagenet \
    --threshold 0.2 \
    --sae_path /PATH/TO/SAE_CKPT.pt \
    --vit_type maple \
    --model_path /PATH/TO/MAPLE_CKPT \  # e.g., .../model.pth.tar-5
    --config_path /PATH/TO/MAPLE_CFG \  # e.g., .../configs/models/maple/vit_b16_c2_ep5_batch4_2ctx.yaml
```

### Output File
- `cls_sae_cnt.npy` - Matrix of shape `(num_sae_latents, num_classes)`

---

## 🎯 Steer Classification

Evaluate classification using feature steering with SAE latents.

### Prerequisites
- CLIP checkpoint
- SAE checkpoint
- Class-level activation data (`cls_sae_cnt.npy` from previous step)
- Dataset (must be the SAME dataset used for class-level activations, though can be a different split)

### Run with Original CLIP
```bash
PYTHONPATH=./ python tasks/classification_with_top_k_masking.py \
    --root_dir ./ \
    --dataset_name imagenet \
    --sae_path /PATH/TO/SAE_CKPT.pt \
    --cls_wise_sae_activation_path /PATH/TO/cls_sae_cnt.npy
```

### Run with Adapted CLIP (e.g., MaPLe)
```bash
PYTHONPATH=./ python tasks/classification_with_top_k_masking.py \
    --root_dir ./ \
    --dataset_name imagenet \
    --sae_path /PATH/TO/SAE_CKPT.pt \
    --cls_wise_sae_activation_path /PATH/TO/cls_sae_cnt.npy \
    --vit_type maple \
    --model_path /PATH/TO/MAPLE_CKPT \  # e.g., .../model.pth.tar-5
    --config_path /PATH/TO/MAPLE_CFG \  # e.g., .../configs/models/maple/vit_b16_c2_ep5_batch4_2ctx.yaml
```

### Output File
Output will be saved to `eval_outputs/`:
- `metrics.csv` - Contains class-wise True Positive Rate (TPR = TP/(TP+FP+TN+FN)) for each masking configuration
  - Results for both "on" and "off" conditions
  - For k values in [1, 2, 5, 10, 50, 100, 500, 1000, 2000, SAE_DIM]
