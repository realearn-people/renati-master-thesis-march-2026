# ProtoPNet Video Inference

This folder contains **video-level inference** scripts for ProtoPNet.
They perform **frame-wise prediction**, then **temporal aggregation** to produce a video-level label.
Optionally, they can record **top-k prototype activations per frame** for analysis.

> Datasets and checkpoints are **not** included.  
> Edit paths in the **USER CONFIGURATION** section or pass CLI arguments (recommended).

---

## Scripts

- `inference_video.py`  
  Video inference for **RAVDESS-style** datasets (emotion id parsed from filename).

- `inference_video_wlasl.py`  
  Video inference for **WLASL / NSLT-style** datasets (label inferred from parent directory name: `split/gloss/*.mp4`).

---

## Quick Start

Install dependencies (from repo root):

Make sure you have a trained ProtoPNet checkpoint (e.g., `saved_models/your_ckpt.pth`).

---

## 1) RAVDESS (single video)

```bash
python video/inference_video_github.py \
  --weight saved_models/your_ckpt.pth \
  --video path/to/video.mp4 \
  --train_dir path/to/train_frames \
  --stride 1 --max_frames 300 --topk 3
```

### Batch mode (directory)

```bash
python video/inference_video_github.py \
  --batch \
  --weight saved_models/your_ckpt.pth \
  --input_dir path/to/all_videos \
  --output_dir outputs/ravdess \
  --train_dir path/to/train_frames
```

---

## 2) WLASL / NSLT (batch)

Expected directory layout:

```
VIDEO_SPLIT_DIR/
  ├── test/
  │   ├── gloss_001/
  │   │   ├── xxx.mp4
  │   │   └── yyy.mp4
  │   └── gloss_002/
  │       └── zzz.mp4
```

Run:

```bash
python video/inference_video_wlasl_github.py \
  --weight saved_models/your_ckpt.pth \
  --train_dir path/to/train_frames \
  --video_split_dir path/to/VIDEO_SPLIT_DIR/test \
  --output_dir outputs/wlasl \
  --target_fps 8 \
  --topk 3
```

> Note: This script recovers the **class order** from `--train_dir` (ImageFolder `class_to_idx`) to avoid label mismatch.

---

## Outputs

Depending on the script/settings, outputs may include:

- `*_probs.csv` : per-frame class probabilities
- `*_protos.json` : per-frame top-k prototype activations (optional)
- Console logs for predicted label and overall accuracy (batch mode)

---

## Notes

- **Face-only vs original checkpoints**: use the `--face_crop` / `--no_face_crop` switch (if supported in the script) to match your training setting.
- Frame sampling is controlled by `--stride` (RAVDESS) or `--target_fps` (WLASL).
- If you rename scripts (recommended), update the commands above accordingly.