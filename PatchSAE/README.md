# PatchSAE – Video Inference (WLASL / NSLT)

This directory contains the video-level inference pipeline built on top of PatchSAE.

The core Sparse Autoencoder (SAE) implementation is inherited from the original PatchSAE repository.
This module extends it to support frame-wise video inference and temporal aggregation.

---

## Overview

The pipeline follows:

Video (.mp4)
→ Frame Sampling (target FPS)
→ CLIP Vision Encoder
→ Patch Mean Pooling
→ Sparse Autoencoder (SAE)
→ Linear Classifier
→ Temporal Aggregation (Vote / Moving Average / EWA)

This enables explainable video-level classification using frame-level PatchSAE features.

---

## Key Features

- CLIP ViT-B/16 backbone
- Pretrained Sparse Autoencoder (SAE)
- Linear classification head
- Frame sampling by target FPS
- Temporal aggregation methods:
  - Majority Vote
  - Moving Average
  - Exponential Weighted Average (default)

---

## Expected Directory Structure

Training frames (for class order alignment):

train_frames/
  ├── gloss_1/
  │     ├── img_001.png
  │     ├── ...
  ├── gloss_2/
  │     ├── ...

Video evaluation directory:

test_videos/
  ├── gloss_1/
  │     ├── video1.mp4
  │     ├── video2.mp4
  ├── gloss_2/
  │     ├── ...

---

## Usage

Example:

python inference_video_patchsae_wlasl.py \
  --sae_path path/to/sae.pt \
  --linear_path path/to/linear.pth \
  --train_frames_dir path/to/train_frames \
  --video_root path/to/test_videos \
  --output_dir ./output \
  --agg_method ewa

---

## Output

For each video:

- Per-frame probability CSV
- Aggregated video-level prediction
- Accuracy summary (batch mode)

Output directory preserves gloss subfolders.

---

## Notes

- No face cropping or grid reorganization is used for WLASL.
- Class order is automatically aligned using ImageFolder.
- The SAE model architecture is unchanged from the original implementation.

---

## Reference

Original PatchSAE repository:
(Add original repository link here)

CLIP model:
OpenAI CLIP ViT-B/16
