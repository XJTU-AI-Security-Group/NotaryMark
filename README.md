# NotaryMark (Multi-bit Watermark Training and Auditing Pipeline)

NotaryMark is an independent research codebase for training and evaluating a **multi-bit image watermarking pipeline** in latent diffusion-style models.  
This repository provides a configurable training script that supports:

- Hugging Face datasets **or** local (image + JSONL) datasets
- Optional mixed **clean / augmented** training views
- Training-time logging + epoch checkpoints
- (Optional) exporting a **snapshot of the original dataset** (images + metadata.jsonl) before training to facilitate reproducibility

---


## Features

- Watermark model training via `train_watermark_model.py`
- Detector training via `train_watermark_detector.py`
- BCH codebook design via `generate_bch_codebook.py`
- Aggregated-result decoding via `decode_aggregated_watermarks.py`
- Watermark embedding via `embed_watermarks.py`
- Detector-based repeated audit via `audit_hypothesis_30_images.py`
- Watermark prediction / extraction via `predict_watermarks.py`
- Repeated majority-vote aggregation via `aggregate_majority_vote.py`

---

## Requirements

- Python **>= 3.10**
- Recommended: CUDA-enabled GPU
- Conda environment used in development: `multi_bit`

---

## Installation

### 1) Create environment

```bash
conda create -n multi_bit python=3.10 -y
conda activate multi_bit
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Verify installation

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

---
---

## Dataset Formats

### Option A: Hugging Face dataset

Example dataset:
- `reach-vb/pokemon-blip-captions` (image + BLIP captions)

You can specify:
- dataset name: `--hf_name`
- split: `--hf_split` (e.g., `train`)
- image/text columns: `--hf_image_col`, `--hf_text_col`

### Option B: Local dataset (image folder + JSONL)

Directory layout:

```
data_dir/
  1.png
  2.jpg
  ...
metadata.jsonl
```

JSONL format (one JSON object per line):

```json
{"file_name": "1.png", "prompt": "a text caption ..."}
{"file_name": "2.jpg", "prompt": "a text caption ..."}
```

Notes:
- `train_watermark_model.py` expects the text field name `prompt`
- `train_watermark_detector.py` also expects the text field name `prompt`

---

## Main Training

Show all options:

```bash
python train_watermark_model.py --help
```

### Example 1: Train with Hugging Face dataset

```bash
python train_watermark_model.py \
  --dataset_mode hf \
  --hf_name reach-vb/pokemon-blip-captions \
  --hf_split train \
  --hf_image_col image \
  --hf_text_col text \
  --vae_path CompVis/stable-diffusion-v1-4 \
  --clip_path openai/clip-vit-large-patch14 \
  --output_dir ./outputs/pokemon \
  --epochs 75 \
  --batch_size 4 \
  --image_size 512
```

### Example 2: Train with local JSONL dataset

```bash
python train_watermark_model.py \
  --dataset_mode local \
  --data_dir ./data/images \
  --jsonl_file ./data/metadata.jsonl \
  --vae_path CompVis/stable-diffusion-v1-4 \
  --clip_path openai/clip-vit-large-patch14 \
  --output_dir ./outputs/pokemon \
  --epochs 75
```

Outputs under `--output_dir` include:

- `training_log_by_step_train.csv`
- `training_log_by_step_test.csv`
- `model_epoch_*.pth`
- optional evaluation image folders

---

## Detector Training

Show all options:

```bash
python train_watermark_detector.py --help
```

Example:

```bash
python train_watermark_detector.py \
  --data_dir ./data/images \
  --jsonl_file ./data/metadata.jsonl \
  --pretrained_checkpoint ./outputs/train_run/model_epoch_75.pth \
  --vae_path CompVis/stable-diffusion-v1-4 \
  --clip_path openai/clip-vit-large-patch14 \
  --benign_negative_dir ./data/benign_negatives \
  --output_dir ./outputs/detector_run \
  --bit_length 1024 \
  --detector_epochs 50 \
  --batch_size 4
```

This script saves:

- detector checkpoints
- detector train/test CSV logs
- the CLI args inside each detector checkpoint

---

## Watermark Embedding

Embed a fixed watermark into all images in one folder:

```bash
python embed_watermarks.py \
  --input_dir ./data/images \
  --output_dir ./outputs/watermarked_images \
  --checkpoint ./outputs/train_run/model_epoch_75.pth \
  --vae_path CompVis/stable-diffusion-v1-4 \
  --bit_length 32 \
  --watermark 10011110111011000011111100101100 \
  --a 0.5
```

---

## Watermark Prediction

Predict watermark bits from a folder of images:

```bash
python predict_watermarks.py \
  --input_dir ./outputs/watermarked_images \
  --checkpoint ./outputs/train_run/model_epoch_75.pth \
  --output_csv ./outputs/predictions/bit_predictions.csv \
  --vae_path CompVis/stable-diffusion-v1-4 \
  --clip_path openai/clip-vit-large-patch14 \
  --bit_length 32 \
  --watermark 10011110111011000011111100101100
```

The output CSV contains per-image predictions and bit-level accuracy statistics.

---

## Aggregation

Aggregate repeated samples with majority vote:

```bash
python aggregate_majority_vote.py \
  --input ./outputs/predictions/bit_predictions.csv \
  --sample_size 30 \
  --num_trials 100 \
  --output_csv ./outputs/predictions/aggregation_trials.csv
```

This produces one row per aggregation trial, including the aggregated bitstring.

---

## BCH Codebook Generation

Design a shortened BCH code and generate a codebook:

```bash
python generate_bch_codebook.py \
  --length 32 \
  --bit_acc 0.94 \
  --num_codewords 16 \
  --output_csv ./codebook/bch_codebook_L32_t4.csv \
  --metadata_json ./codebook/bch_codebook_L32_t4_metadata.json \
  --non_interactive
```

Outputs:

- BCH codebook CSV
- metadata JSON

---

## Decode Aggregated Results with Codebook

Decode aggregated watermark strings using the generated codebook:

```bash
python decode_aggregated_watermarks.py \
  --input_csv ./outputs/predictions/aggregation_trials.csv \
  --output_csv ./outputs/predictions/decoded_trials.csv \
  --codebook_csv ./codebook/bch_codebook_L32_t4.csv \
  --metadata_json ./codebook/bch_codebook_L32_t4_metadata.json \
  --decoder_mode bch_then_match
```

Supported modes:

- `auto`
- `bch_then_match`
- `nearest`

---

## Detector-Based Audit

Run repeated 30-image hypothesis tests on a suspect image folder:

```bash
python audit_hypothesis_30_images.py \
  --suspect_dir ./audit/suspect_images \
  --null_dir ./audit/null_images \
  --checkpoint ./outputs/train_run/model_epoch_75.pth \
  --detector_checkpoint ./outputs/detector_run/detector_model_epoch_50.pth \
  --vae_path CompVis/stable-diffusion-v1-4 \
  --clip_path openai/clip-vit-large-patch14 \
  --trial_size 30 \
  --num_trials 100 \
  --num_null_samples 10000 \
  --output_csv ./outputs/audit/trial_results.csv \
  --output_score_csv ./outputs/audit/image_scores.csv
```

This script:

1. scores suspect and null images with the detector
2. builds an empirical null distribution
3. runs repeated trial-level hypothesis tests
4. reports the fraction of trials classified as positive

---

## Reproducibility Notes

- Prefer relative paths such as `./data/...` and `./outputs/...` for datasets and results
- `--vae_path` and `--clip_path` can be either Hugging Face repo IDs or local model paths
- Keep VAE and CLIP checkpoints outside the repository if they are large
- Save the exact training and detector checkpoints used for embedding / extraction / audit
- `train_watermark_model.py` and `train_watermark_detector.py` both store CLI args in checkpoints for reproducibility

---

## Suggested Repository Layout

```text
NotaryMark/
  data/
    images/
    metadata.jsonl
    benign_negatives/
  outputs/
    train_run/
    detector_run/
    predictions/
    audit/
  codebook/
```

This layout is only a suggestion, but it makes the relative-path examples above work cleanly.
