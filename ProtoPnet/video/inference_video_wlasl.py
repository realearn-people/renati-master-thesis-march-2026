"""inference_video_wlasl.py

Video inference script for ProtoPNet on WLASL/NSLT-style datasets.

Features:
- Loads a trained ProtoPNet checkpoint (via your project's settings/model code)
- Samples frames at a target FPS
- Runs frame-wise inference and aggregates to a video-level prediction (EWA / moving-average / vote)
- Optionally records top-k prototype activations per frame
- Writes per-frame probabilities (CSV) and prototype traces (JSON)

Notes:
- Edit the 'USER CONFIGURATION' section before running.
- The training frames directory is used ONLY to recover class order (ImageFolder class_to_idx).
"""

import os
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import datasets

# === Project modules (keep consistent with your original project) ===
import settings
from model import construct_PPNet

# If your project defines mean/std in preprocess.py, use them; otherwise fall back to 0.5 normalization.
try:
    from preprocess import mean, std
except Exception:
    mean = [0.5, 0.5, 0.5]
    std  = [0.5, 0.5, 0.5]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==========================================
# USER CONFIGURATION (EDIT BEFORE RUNNING)
# ==========================================
WEIGHT_PATH = "path/to/your/checkpoint.pth"
# Training frames directory used to recover class_to_idx order (IMPORTANT for correct label mapping).
TRAIN_FRAMES_DIR = "path/to/train_frames_dir  # used to recover class order (ImageFolder)"
# Video root directory (expects split/gloss/*.mp4 structure).
VIDEO_SPLIT_DIR = "path/to/video_split_dir    # expects split/gloss/*.mp4 structure"
# Output directory.
OUTPUT_DIR = "path/to/output_dir"
# =========================
# Frame sampling settings
# =========================
TARGET_FPS = 5
MAX_FRAMES = 96

# Temporal aggregation parameters (kept as-is).
MA_WIN = 9
EWA_ALPHA = 0.7

# Whether to record per-frame top-k prototypes (slower).
RECORD_TOPK_PROTOS = True
TOPK_PROTOS = 3


# =========================
# Model construction & loading (kept as-is)
# =========================
def build_model_from_settings():
    return construct_PPNet(
        base_architecture=settings.base_architecture,
        pretrained=False,
        img_size=settings.img_size,
        prototype_shape=settings.prototype_shape,
        num_classes=settings.num_classes,
        prototype_activation_function=settings.prototype_activation_function,
        add_on_layers_type=settings.add_on_layers_type
    )

def _strip_module_prefix(state_dict: dict) -> dict:
    if any(k.startswith("module.") for k in state_dict.keys()):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict

def load_ppnet(weight_path: str) -> nn.Module:
    ckpt = torch.load(weight_path, map_location=DEVICE, weights_only=False)

    # Compatibility: checkpoint can be {'state_dict': ...} or a raw state_dict.
    if isinstance(ckpt, dict):
        state = ckpt
        for k in ("state_dict", "model_state_dict"):
            if k in ckpt and isinstance(ckpt[k], dict):
                state = ckpt[k]
                break
        state = _strip_module_prefix(state)

        model = build_model_from_settings().to(DEVICE)
        msg = model.load_state_dict(state, strict=False)
        print("[Load] missing:", getattr(msg, "missing_keys", []))
        print("[Load] unexpected:", getattr(msg, "unexpected_keys", []))
        model.eval()
        return model

    if isinstance(ckpt, nn.Module):
        print("[Load] Loaded full nn.Module.")
        return ckpt.to(DEVICE).eval()

    raise RuntimeError(f"Unsupported checkpoint type: {type(ckpt)}")


# =========================
# Prototype activations (kept as-is)
# =========================
@torch.no_grad()
def get_proto_activations(model, x):
    """
    输入: x [1,3,H,W]
    输出:
      sim_maps: [P,Hs,Ws]  每个原型的相似度热图(越大越像)
      cls_of_proto: [P]    每个原型归属的类别索引(若未知则为 -1)
    """
    import torch.nn.functional as F

    if hasattr(model, 'prototype_distances'):
        dist = model.prototype_distances(x)  # [1,P,Hs,Ws]
        if hasattr(model, 'distance_2_similarity'):
            sim = model.distance_2_similarity(dist)
        else:
            sim = -dist
        sim_maps = sim.squeeze(0).detach().cpu().numpy()
        if hasattr(model, 'prototype_class_identity'):
            pci = model.prototype_class_identity
            cls_of_proto = pci.argmax(dim=1).detach().cpu().numpy()
        else:
            cls_of_proto = np.full(sim_maps.shape[0], -1, dtype=int)
        return sim_maps, cls_of_proto

    # fallback
    if hasattr(model, 'push_forward'):
        z = model.push_forward(x)[0]
    else:
        z = model.add_on_layers(model.features(x))
    z = F.normalize(z, p=2, dim=1)

    proto = model.prototype_vectors
    proto = F.normalize(proto, p=2, dim=1)

    sim = torch.conv2d(z, proto)
    sim_maps = sim.squeeze(0).detach().cpu().numpy()
    if hasattr(model, 'prototype_class_identity'):
        cls_of_proto = model.prototype_class_identity.argmax(dim=1).detach().cpu().numpy()
    else:
        cls_of_proto = np.full(sim_maps.shape[0], -1, dtype=int)
    return sim_maps, cls_of_proto


# =========================
# Preprocessing (should match training)
# =========================
preprocess = T.Compose([
    T.Resize((settings.img_size, settings.img_size)),
    T.ToTensor(),
    T.Normalize(mean=mean, std=std),
])


# =========================
# Aggregation methods (kept as-is)
# =========================
def majority_vote(argmax_ids: np.ndarray):
    vals, cnts = np.unique(argmax_ids, return_counts=True)
    cls = int(vals[np.argmax(cnts)])
    conf = float((argmax_ids == cls).mean())
    return cls, conf

def moving_average(probs: np.ndarray, win: int = 9):
    N, C = probs.shape
    if N == 0:
        return 0, 0.0
    half = win // 2
    out = np.zeros_like(probs)
    for i in range(N):
        L, R = max(0, i-half), min(N, i+half+1)
        out[i] = probs[L:R].mean(axis=0)
    p = out.mean(axis=0)
    return int(p.argmax()), float(p.max())

def ewa(probs: np.ndarray, alpha: float = 0.7):
    if len(probs) == 0:
        return 0, 0.0
    agg = probs[0].copy()
    for i in range(1, len(probs)):
        agg = alpha * probs[i] + (1 - alpha) * agg
    return int(agg.argmax()), float(agg.max())


# =========================
# WLASL/NSLT frame sampling at a target FPS
# =========================
def iter_sampled_frames(video_path: str, target_fps=5, max_frames=96):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], 0.0

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 25.0

    interval = max(int(round(fps / target_fps)), 1)

    frames = []
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if i % interval == 0:
            frames.append(frame)
            if max_frames and len(frames) >= max_frames:
                break
        i += 1

    cap.release()
    return frames, float(fps)


# =========================
# Single video: frame-wise inference + optional prototype recording
# =========================
@torch.no_grad()
def infer_video_with_protos(
    model,
    video_path: str,
    num_classes: int,
    target_fps=5,
    max_frames=96,
    topk=3,
    record_topk=True,
):
    frames, fps = iter_sampled_frames(video_path, target_fps=target_fps, max_frames=max_frames)

    probs_list, argmax_list = [], []
    proto_records = []

    for frame in frames:
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        x = preprocess(img).unsqueeze(0).to(DEVICE)

        out = model(x)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        prob = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()
        probs_list.append(prob)
        argmax_list.append(int(prob.argmax()))

        if record_topk:
            sim_maps, cls_of_proto = get_proto_activations(model, x)  # [P,Hs,Ws]
            P, Hs, Ws = sim_maps.shape
            sim_flat = sim_maps.reshape(P, -1)
            max_sim = sim_flat.max(axis=1)
            max_idx = sim_flat.argmax(axis=1)
            max_h = (max_idx // Ws).astype(int)
            max_w = (max_idx % Ws).astype(int)

            order = np.argsort(-max_sim)[:topk]
            frame_topk = []
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

    probs = np.stack(probs_list) if probs_list else np.zeros((0, num_classes), dtype=np.float32)
    argmx = np.array(argmax_list, dtype=int)
    return probs, argmx, fps, proto_records


# =========================
# Batch evaluation: run over all *.mp4 under VIDEO_SPLIT_DIR
# Ground-truth label is inferred from the parent directory name (gloss)
# =========================
def batch_infer_and_evaluate(
    model,
    video_root: str,
    output_root: str,
    train_frames_dir: str,
    target_fps=5,
    max_frames=96,
    record_topk=True,
    topk=3,
    agg_method="ewa",  # "ewa" / "ma" / "vote"
):
    video_root = Path(video_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # Recover class order from training frames (CRITICAL to avoid label mismatch).
    train_ds = datasets.ImageFolder(train_frames_dir)
    idx_to_class = {v: k for k, v in train_ds.class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    class_to_idx = train_ds.class_to_idx
    num_classes = len(class_names)

    print("Loaded class order from:", train_frames_dir)
    print("num_classes =", num_classes)
    print("first10 =", class_names[:10])

    total, correct = 0, 0
    bad_open = 0
    skipped_unknown_class = 0

    # Supports both video_root/<gloss>/*.mp4 and video_root/**/<gloss>/*.mp4 layouts.
    mp4_list = sorted(video_root.rglob("*.mp4"))
    print(f"Found {len(mp4_list)} mp4 under {video_root}")

    for video_path in tqdm(mp4_list, desc="videos", unit="vid"):
        # 真值标签来自父文件夹名
        true_gloss = video_path.parent.name
        if true_gloss not in class_to_idx:
            skipped_unknown_class += 1
            continue
        true_cls = class_to_idx[true_gloss]

        try:
            probs, argmx, fps, proto_records = infer_video_with_protos(
                model,
                str(video_path),
                num_classes=num_classes,
                target_fps=target_fps,
                max_frames=max_frames,
                topk=topk,
                record_topk=record_topk,
            )
        except Exception as e:
            bad_open += 1
            continue

        if len(probs) == 0:
            bad_open += 1
            continue

        # 聚合
        pred_vote, conf_vote = majority_vote(argmx)
        pred_ma, conf_ma = moving_average(probs, win=MA_WIN)
        pred_ewa, conf_ewa = ewa(probs, alpha=EWA_ALPHA)

        if agg_method == "vote":
            final_pred = pred_vote
            final_conf = conf_vote
        elif agg_method == "ma":
            final_pred = pred_ma
            final_conf = conf_ma
        else:
            final_pred = pred_ewa
            final_conf = conf_ewa

        # Save per-frame probabilities + top-k prototype traces.
        rel = video_path.relative_to(video_root)  # <gloss>/<video>.mp4
        stem = video_path.stem

        out_dir = output_root / rel.parent  # 保持 gloss 子目录
        out_dir.mkdir(parents=True, exist_ok=True)

        csv_path = out_dir / f"{stem}_per_frame_probs.csv"
        json_path = out_dir / f"{stem}_topk_protos.json"

        df = pd.DataFrame(probs, columns=class_names)
        df["argmax_id"] = argmx
        df["argmax_cls"] = [class_names[i] for i in argmx]
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")

        if record_topk:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(proto_records, f, ensure_ascii=False, indent=2)

        total += 1
        if final_pred == true_cls:
            correct += 1

        # Optional: print limited logs (avoid spamming).
        # print(f"{video_path.name} | true={true_gloss} pred={class_names[final_pred]} conf={final_conf:.3f}")

    acc = (correct / total * 100) if total > 0 else 0.0
    print("\n==== Summary ====")
    print("video_root:", video_root)
    print("agg_method:", agg_method)
    print("target_fps:", target_fps, "max_frames:", max_frames)
    print(f"processed: {total}  correct: {correct}  acc: {acc:.2f}%")
    print("bad_open/empty:", bad_open)
    print("skipped_unknown_class:", skipped_unknown_class)

    return acc


if __name__ == "__main__":
    # 1) Load model
    model = load_ppnet(WEIGHT_PATH)

    # 2) Batch evaluation (frame-wise inference + temporal aggregation).
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    batch_infer_and_evaluate(
        model=model,
        video_root=VIDEO_SPLIT_DIR,
        output_root=OUTPUT_DIR,
        train_frames_dir=TRAIN_FRAMES_DIR,
        target_fps=TARGET_FPS,
        max_frames=MAX_FRAMES,
        record_topk=RECORD_TOPK_PROTOS,
        topk=TOPK_PROTOS,
        agg_method="ewa",   # "ewa" / "ma" / "vote"
    )
