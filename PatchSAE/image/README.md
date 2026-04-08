# PatchSAE -- Image-Based Experimental Pipeline

This directory contains the image-based experimental extension built on
top of the original PatchSAE implementation.

------------------------------------------------------------------------

## Original Implementation

The core PatchSAE implementation is located in:

    patchsae-main/

Original repository: `<INSERT ORIGINAL GITHUB LINK HERE>`{=html}

License: MIT License (see `patchsae-main/LICENSE`)

No modifications were made to the core sparse autoencoder architecture
inside `patchsae-main/src/`.

------------------------------------------------------------------------

## Image Experimental Pipeline

This extension enables PatchSAE to be evaluated in an image
classification setting.

The workflow consists of three stages:

------------------------------------------------------------------------

### 1️⃣ Feature Extraction

`scripts/extract_clip_patches.py`

-   Loads images from an ImageFolder-style dataset
-   Uses a CLIP-ViT backbone
-   Extracts patch-level features
-   Saves patch activations for SAE training

------------------------------------------------------------------------

### 2️⃣ Sparse Autoencoder Training

`patchsae-main/tasks/train_sae_vit.py`

-   Original training script from PatchSAE
-   Used without modifying the core SAE implementation
-   Trains the sparse autoencoder on extracted patch features

------------------------------------------------------------------------

### 3️⃣ Linear Probing

`scripts/linear_probe_sae.py`

-   Added script
-   Trains a linear classifier on SAE sparse codes
-   Used for classification performance evaluation

------------------------------------------------------------------------

## Research Purpose

This image-based pipeline is used to compare:

-   Prototype-based explanations (ProtoPNet)
-   Sparse latent atom representations (PatchSAE)

The goal is to analyze differences in interpretability mechanisms under
consistent experimental conditions.

This work extends the evaluation pipeline, but does not re-implement or
modify the original PatchSAE model.

------------------------------------------------------------------------

## Notes

-   Large files (checkpoints, activations, logs) are excluded via
    `.gitignore`
-   Datasets are not included in this repository
-   Paths should be configured according to the local environment
