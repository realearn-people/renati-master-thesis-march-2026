"""
inference_video.py (refactored)

Video inference for ProtoPNet with:
- frame-wise prediction
- prototype similarity tracking (top-k per frame)
- video-level aggregation (vote / moving average / EWMA)
- optional face crop and optional 2x2 face grid

This script is intended to be an easy-to-read, easy-to-configure entrypoint.
Edit the "USER CONFIG" section only.
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision import datasets


# ============================================================
# USER CONFIG (edit here)
# ============================================================

# --- Files / Paths ---
WEIGHT_PATH = Path(r'./saved_models/vgg16_bn/Rav_faceonlyv2_vgg16_bn 30 16/90push0.9886.pth')
VIDEO_PATH  = Path("./example.mp4")

# If BATCH_MODE=True, infer all *.mp4 under INPUT_DIR and write results to OUTPUT_DIR
BATCH_MODE = False
INPUT_DIR  = Path("./all")
OUTPUT_DIR = Path("./video_results")

# --- Frame sampling ---
STRIDE     = 1        # use 1 frame every N frames
MAX_FRAMES = 300      # None = no limit; 300 is convenient for debugging

# --- Preprocessing switches ---
FACE_CROP   = True    # True for face-only models; False for original models
FACE_MARGIN = 0.35    # crop margin around detected face
USE_GRID    = False   # optional: re-organize face into a 2x2 grid (eyes/nose/mouth)

# --- Prototype tracking ---
TOPK_PROTOS = 3       # record top-k prototypes per frame

# --- Temporal aggregation ---
MA_WINDOW   = 9       # moving average window size
EWA_ALPHA   = 0.7     # exponential weighted average alpha

# --- Label mapping / class names ---
# Choose one: "3class" or "6class"
LABEL_MODE = "3class"

# For 3-class setting (index order must match training)
CLASS_NAMES_3 = ["negative", "neutral", "positive"]

# For 6-class setting (index order must match training)
# (Example order placeholder. Replace with your training order if needed.)
CLASS_NAMES_6 = ["angry", "calm", "fearful", "happy", "neutral", "sad"]


# ============================================================
# Project modules (best effort import)
# ============================================================

# NOTE:
# - This script expects your project to provide:
#   - settings.py (base_architecture, img_size, prototype_shape, num_classes, etc.)
#   - model.py (construct_PPNet)
#   - preprocess.py (mean, std)
#
# If you share code to your professor, make sure these files are included.

try:
    import settings
except Exception as e:
    raise ImportError(
        "Cannot import 'settings'. Please place this script inside your project root "
        "or ensure settings.py is available on PYTHONPATH."
    ) from e

try:
    from model import construct_PPNet
except Exception as e:
    raise ImportError("Cannot import 'construct_PPNet' from model.py") from e

try:
    from preprocess import mean, std
except Exception as e:
    raise ImportError("Cannot import 'mean, std' from preprocess.py") from e


# ============================================================
# Runtime setup
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

HAAR_PATH = str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
face_detector = cv2.CascadeClassifier(HAAR_PATH)

TARGET_SIZE = int(getattr(settings, "img_size", 224))
PATCH_SIZE  = TARGET_SIZE // 2


# ============================================================
# Model loading
# ============================================================

def build_model_from_settings() -> nn.Module:
    """Build ProtoPNet using the project settings.py."""
    return construct_PPNet(
        base_architecture=settings.base_architecture,
        pretrained=False,
        img_size=settings.img_size,
        prototype_shape=settings.prototype_shape,
        num_classes=settings.num_classes,
        prototype_activation_function=settings.prototype_activation_function,
        add_on_layers_type=settings.add_on_layers_type,
    )

def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Remove 'module.' prefix if the checkpoint was saved from DataParallel."""
    if any(k.startswith("module.") for k in state_dict.keys()):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict

def load_ppnet(weight_path: Path) -> nn.Module:
    """
    Load a ProtoPNet checkpoint.
    Supports:
      - plain state_dict
      - dict with 'state_dict' or 'model_state_dict'
      - (rare) full nn.Module object
    """
    ckpt = torch.load(str(weight_path), map_location=DEVICE, weights_only=False)

    if isinstance(ckpt, dict):
        state = ckpt
        for k in ("state_dict", "model_state_dict"):
            if k in ckpt and isinstance(ckpt[k], dict):
                state = ckpt[k]
                break

        state = _strip_module_prefix(state)
        model = build_model_from_settings().to(DEVICE)
        msg = model.load_state_dict(state, strict=False)

        # Print mismatch info (helps debugging)
        print("[Load] missing keys:", getattr(msg, "missing_keys", []))
        print("[Load] unexpected keys:", getattr(msg, "unexpected_keys", []))

        model.eval()
        return model

    if isinstance(ckpt, nn.Module):
        print("[Load] Loaded a full nn.Module object.")
        return ckpt.to(DEVICE).eval()

    raise RuntimeError(f"Unsupported checkpoint type: {type(ckpt)}")


# ============================================================
# Prototype activation helpers
# ============================================================

@torch.no_grad()
def get_proto_activations(model: nn.Module, x: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """
    Input:
      x: [1, 3, H, W]
    Output:
      sim_maps: [P, Hs, Ws]  similarity heatmaps (higher = more similar)
      cls_of_proto: [P]      prototype's assigned class index (or -1 if unavailable)
    """
    # A) Preferred: use model's built-in proto distance interface
    if hasattr(model, "prototype_distances"):
        dist = model.prototype_distances(x)  # [1, P, Hs, Ws]

        if hasattr(model, "distance_2_similarity"):
            sim = model.distance_2_similarity(dist)        # [1, P, Hs, Ws]
        else:
            sim = -dist

        sim_maps = sim.squeeze(0).detach().cpu().numpy()   # [P, Hs, Ws]

        if hasattr(model, "prototype_class_identity"):
            pci = model.prototype_class_identity           # [P, C] 0/1
            cls_of_proto = pci.argmax(dim=1).detach().cpu().numpy()
        else:
            cls_of_proto = np.full(sim_maps.shape[0], -1, dtype=int)

        return sim_maps, cls_of_proto

    # B) Fallback: approximate cosine similarity by conv2d(features, prototypes)
    if hasattr(model, "push_forward"):
        z = model.push_forward(x)[0]                       # [1, D, Hs, Ws]
    else:
        z = model.add_on_layers(model.features(x))

    z = F.normalize(z, p=2, dim=1)

    proto = model.prototype_vectors                        # [P, D, 1, 1]
    proto = F.normalize(proto, p=2, dim=1)

    sim = torch.conv2d(z, proto)                           # [1, P, Hs, Ws]
    sim_maps = sim.squeeze(0).detach().cpu().numpy()

    if hasattr(model, "prototype_class_identity"):
        cls_of_proto = model.prototype_class_identity.argmax(dim=1).detach().cpu().numpy()
    else:
        cls_of_proto = np.full(sim_maps.shape[0], -1, dtype=int)

    return sim_maps, cls_of_proto


# ============================================================
# Preprocessing
# ============================================================

preprocess = T.Compose([
    T.Resize((TARGET_SIZE, TARGET_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=mean, std=std),
])


# ============================================================
# Aggregation methods
# ============================================================

def majority_vote(argmax_ids: np.ndarray) -> Tuple[int, float]:
    """Return (class_id, confidence≈vote_ratio)."""
    vals, cnts = np.unique(argmax_ids, return_counts=True)
    cls = int(vals[np.argmax(cnts)])
    conf = float((argmax_ids == cls).mean())
    return cls, conf

def moving_average(probs: np.ndarray, win: int = 9) -> Tuple[int, float]:
    """Smooth per-frame probs by local averaging, then average over frames."""
    N, C = probs.shape
    half = win // 2
    out = np.zeros_like(probs)
    for i in range(N):
        L, R = max(0, i - half), min(N, i + half + 1)
        out[i] = probs[L:R].mean(axis=0)
    p = out.mean(axis=0)
    return int(p.argmax()), float(p.max())

def ewa(probs: np.ndarray, alpha: float = 0.7) -> Tuple[int, float]:
    """Exponential Weighted Average (EWMA) over time."""
    agg = probs[0].copy()
    for i in range(1, len(probs)):
        agg = alpha * probs[i] + (1 - alpha) * agg
    return int(agg.argmax()), float(agg.max())


# ============================================================
# Optional face crop / face grid
# ============================================================

def crop_face_bgr(frame_bgr: np.ndarray, margin: float = 0.35) -> np.ndarray:
    """Detect the largest face and crop with margin. If no face, return original frame."""
    h, w = frame_bgr.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
    if len(faces) == 0:
        return frame_bgr

    x, y, fw, fh = max(faces, key=lambda r: r[2] * r[3])
    cx, cy = x + fw // 2, y + fh // 2
    side = int(max(fw, fh) * (1 + 2 * margin))

    x1 = max(0, cx - side // 2)
    y1 = max(0, cy - side // 2)
    x2 = min(w, cx + side // 2)
    y2 = min(h, cy + side // 2)

    return frame_bgr[y1:y2, x1:x2]

def make_face_grid(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Re-organize the face into a 2x2 grid:
      left_eye | right_eye
      nose     | mouth
    """
    img = cv2.resize(frame_bgr, (TARGET_SIZE, TARGET_SIZE))
    h, w, _ = img.shape

    eye_top = int(0.10 * h)
    eye_bottom = int(0.45 * h)
    mid_x = w // 2

    left_eye = img[eye_top:eye_bottom, 0:mid_x]
    right_eye = img[eye_top:eye_bottom, mid_x:w]

    nose_top = int(0.35 * h)
    nose_bottom = int(0.70 * h)
    mouth_top = int(0.55 * h)
    mouth_bottom = int(0.95 * h)

    nose = img[nose_top:nose_bottom, 0:mid_x]
    mouth = img[mouth_top:mouth_bottom, mid_x:w]

    def resize_patch(p: np.ndarray) -> np.ndarray:
        return cv2.resize(p, (PATCH_SIZE, PATCH_SIZE))

    top_row = np.concatenate([resize_patch(left_eye), resize_patch(right_eye)], axis=1)
    bottom_row = np.concatenate([resize_patch(nose), resize_patch(mouth)], axis=1)
    return np.concatenate([top_row, bottom_row], axis=0)


# ============================================================
# Inference
# ============================================================

@torch.no_grad()
def infer_video_with_protos(
    model: nn.Module,
    video_path: str,
    stride: int = 1,
    max_frames: Optional[int] = None,
    topk: int = 3,
    show_progress: bool = True,
) -> Tuple[np.ndarray, np.ndarray, float, List[List[dict]]]:
    """
    Per-frame inference + top-k prototype tracking.

    Returns:
      probs: [N, C]
      argmx: [N]
      fps: float
      proto_records: length N, each item is a list of top-k proto dicts
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))

    probs_list: List[np.ndarray] = []
    argmax_list: List[int] = []
    proto_records: List[List[dict]] = []

    used = 0
    frame_iter = range(total)
    if show_progress:
        frame_iter = tqdm(frame_iter, total=total, desc=Path(video_path).name, unit="frm", leave=False)

    for i in frame_iter:
        ret, frame = cap.read()
        if not ret:
            break
        if i % stride != 0:
            continue

        if FACE_CROP:
            frame = crop_face_bgr(frame, margin=FACE_MARGIN)

        if USE_GRID:
            frame = make_face_grid(frame)

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        x = preprocess(img).unsqueeze(0).to(DEVICE)

        out = model(x)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        prob = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()

        probs_list.append(prob)
        argmax_list.append(int(prob.argmax()))

        # Prototype similarity maps
        sim_maps, cls_of_proto = get_proto_activations(model, x)  # [P, Hs, Ws], [P]
        P, Hs, Ws = sim_maps.shape
        sim_flat = sim_maps.reshape(P, -1)

        max_sim = sim_flat.max(axis=1)         # [P]
        max_idx = sim_flat.argmax(axis=1)      # [P]
        max_h = (max_idx // Ws).astype(int)
        max_w = (max_idx % Ws).astype(int)

        order = np.argsort(-max_sim)[:topk]
        frame_topk: List[dict] = []
        for pid in order:
            frame_topk.append({
                "proto_id": int(pid),
                "proto_cls": int(cls_of_proto[pid]),
                "similarity": float(max_sim[pid]),
                "h": int(max_h[pid]),
                "w": int(max_w[pid]),
                "fmap_h": int(Hs),
                "fmap_w": int(Ws),
            })
        proto_records.append(frame_topk)

        used += 1
        if max_frames is not None and used >= max_frames:
            break

    cap.release()

    probs = np.stack(probs_list) if probs_list else np.zeros((0, int(getattr(settings, "num_classes", 0))))
    argmx = np.array(argmax_list, dtype=int)
    return probs, argmx, fps, proto_records


# ============================================================
# Batch evaluation
# ============================================================

def get_label_mapping(mode: str) -> Tuple[Dict[int, int], List[str]]:
    """
    Return:
      id_to_class_index: mapping from RAVDESS emotion_id -> class_index
      class_names: list of class names in training order
    """
    mode = mode.lower().strip()

    if mode == "6class":
        # Example: original 6-class mapping (adjust indices to your training order if different)
        id_to_class_index = {
            1: 4,  # neutral
            2: 1,  # calm
            3: 3,  # happy
            4: 5,  # sad
            5: 0,  # angry
            6: 2,  # fearful
        }
        return id_to_class_index, CLASS_NAMES_6

    if mode == "3class":
        # Your 3-class mapping:
        # neutral(1) + calm(2) -> neutral
        # happy(3) -> positive
        # sad(4) + angry(5) -> negative
        id_to_class_index = {
            1: 1,  # neutral
            2: 1,  # calm -> neutral
            3: 2,  # happy -> positive
            4: 0,  # sad -> negative
            5: 0,  # angry -> negative
        }
        return id_to_class_index, CLASS_NAMES_3

    raise ValueError(f"Unknown LABEL_MODE: {mode}. Use '3class' or '6class'.")


def batch_infer_and_evaluate(
    model: nn.Module,
    input_dir: Path,
    output_dir: Path,
    stride: int = 1,
    topk: int = 3,
    label_mode: str = "3class",
) -> None:
    """
    Run inference for all videos under input_dir and save:
      - per-frame probabilities (CSV)
      - per-frame top-k prototype records (JSON)
    Also prints overall accuracy using EWMA aggregation.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    id_to_class_index, class_names = get_label_mapping(label_mode)

    total, correct = 0, 0

    for video_path in sorted(input_dir.rglob("*.mp4")):
        print(f"\n▶ Running: {video_path.name}")

        # Parse RAVDESS emotion_id from filename (3rd chunk)
        try:
            emotion_id = int(video_path.stem.split("-")[2])
        except Exception:
            emotion_id = -1

        if emotion_id not in id_to_class_index:
            print(f"Skip unused emotion {emotion_id} ({video_path.name})")
            continue
        true_cls = id_to_class_index[emotion_id]

        probs, argmx, fps, proto_records = infer_video_with_protos(
            model, str(video_path), stride=stride, topk=topk, show_progress=True
        )

        # Aggregation
        pred_vote, conf_vote = majority_vote(argmx)
        pred_ma, conf_ma = moving_average(probs, win=MA_WINDOW)
        pred_ewa, conf_ewa = ewa(probs, alpha=EWA_ALPHA)

        # Save outputs
        csv_path = output_dir / f"{video_path.stem}_probs.csv"
        json_path = output_dir / f"{video_path.stem}_topk.json"

        df = pd.DataFrame(probs, columns=class_names)
        df["argmax_id"] = argmx
        df["argmax_cls"] = [class_names[i] for i in argmx]
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(proto_records, f, ensure_ascii=False, indent=2)

        print(f"Saved: {csv_path.name}, {json_path.name}")
        print(f"[vote] pred={class_names[pred_vote]} conf≈{conf_vote:.3f}")
        print(f"[ m.a] pred={class_names[pred_ma]}   conf≈{conf_ma:.3f}")
        print(f"[ ewa] pred={class_names[pred_ewa]}  conf≈{conf_ewa:.3f}")

        # Overall accuracy uses EWMA by default
        total += 1
        if true_cls == pred_ewa:
            correct += 1

    if total > 0:
        acc = correct / total * 100.0
        print(f"\n✅ Overall Accuracy: {acc:.2f}% ({correct}/{total})")
    else:
        print("⚠️ No videos processed.")


# ============================================================
# Main
# ============================================================

def main() -> None:
    # If you want to recover class order from a training folder (ImageFolder),
    # you can uncomment and set TRAIN_DIR properly.
    #
    # TRAIN_DIR = Path("./ravdess_v2/fo/train")
    # train_ds = datasets.ImageFolder(str(TRAIN_DIR))
    # idx_to_class = {v: k for k, v in train_ds.class_to_idx.items()}
    # class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    #
    # In this refactor, class names come from LABEL_MODE.
    id_to_class_index, class_names = get_label_mapping(LABEL_MODE)
    print("LABEL_MODE:", LABEL_MODE)
    print("class_names:", class_names)

    model = load_ppnet(WEIGHT_PATH)

    if not BATCH_MODE:
        probs, argmx, fps, proto_records = infer_video_with_protos(
            model,
            str(VIDEO_PATH),
            stride=STRIDE,
            max_frames=MAX_FRAMES,
            topk=TOPK_PROTOS,
            show_progress=True,
        )
        print(f"Frames processed: {len(probs)} | FPS: {fps:.1f}")

        vid_vote, conf_vote = majority_vote(argmx)
        vid_ma, conf_ma = moving_average(probs, win=MA_WINDOW)
        vid_ewa, conf_ewa = ewa(probs, alpha=EWA_ALPHA)

        print(f"[vote] {class_names[vid_vote]} conf≈{conf_vote:.3f}")
        print(f"[ m.a] {class_names[vid_ma]}   conf≈{conf_ma:.3f}")
        print(f"[ ewa] {class_names[vid_ewa]}  conf≈{conf_ewa:.3f}")

        out_csv = VIDEO_PATH.with_suffix(".per_frame_probs_proto.csv")
        out_json = VIDEO_PATH.with_suffix(".per_frame_topk_protos.json")

        df = pd.DataFrame(probs, columns=class_names)
        df["argmax_id"] = argmx
        df["argmax_cls"] = [class_names[i] for i in argmx]
        df.to_csv(out_csv, index=False, encoding="utf-8-sig")

        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(proto_records, f, ensure_ascii=False, indent=2)

        print("Saved per-frame probs:", out_csv)
        print("Saved proto tracking:", out_json)

    else:
        batch_infer_and_evaluate(
            model=model,
            input_dir=INPUT_DIR,
            output_dir=OUTPUT_DIR,
            stride=STRIDE,
            topk=TOPK_PROTOS,
            label_mode=LABEL_MODE,
        )


if __name__ == "__main__":
    main()
