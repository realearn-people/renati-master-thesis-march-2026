import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision import datasets

import argparse

# ==== PatchSAE ====
from transformers import CLIPVisionModel, CLIPImageProcessor
from src.sae_training.sparse_autoencoder import SparseAutoencoder, ViTSAERunnerConfig

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# Default paths (override via CLI)
# =========================
SAE_PATH    = r"E:/PatchSae/sae_ckpt/lcym3qu6/final_sparse_autoencoder_openai/openai/clip-vit-base-patch16_-2_resid_49152.pt"
LINEAR_PATH = r"E:/WLASL_HF/clf_ckpt/linear_wlasl_patchsae_best.pth"
TRAIN_FRAMES_DIR = r"E:/WLASL_HF/nslt10_frames_5fps_upperbody_flat_fix/train"
VIDEO_ROOT = r"E:/WLASL_HF/nslt10_split_gloss"
OUTPUT_DIR = r"E:/WLASL_HF/patchsae_video_infer_out/test"

TARGET_FPS = 5
MAX_FRAMES = 96

MA_WIN = 9
EWA_ALPHA = 0.7
# =========================


def parse_args():
    p = argparse.ArgumentParser("PatchSAE WLASL/NSLT video inference (batch)")
    p.add_argument("--sae_path", type=str, default=SAE_PATH)
    p.add_argument("--linear_path", type=str, default=LINEAR_PATH)
    p.add_argument("--train_frames_dir", type=str, default=TRAIN_FRAMES_DIR,
                   help="ImageFolder train frames dir (train/gloss/*.png) for class order alignment")
    p.add_argument("--video_root", type=str, default=VIDEO_ROOT,
                   help="Video root (test/gloss/*.mp4)")
    p.add_argument("--output_dir", type=str, default=OUTPUT_DIR)

    p.add_argument("--target_fps", type=int, default=TARGET_FPS)
    p.add_argument("--max_frames", type=int, default=MAX_FRAMES)

    p.add_argument("--agg_method", type=str, choices=["ewa", "ma", "vote"], default="ewa")
    p.add_argument("--ma_win", type=int, default=MA_WIN)
    p.add_argument("--ewa_alpha", type=float, default=EWA_ALPHA)
    return p.parse_args()


# ========== vote / smoothing ==========
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

def ewa(probs: np.ndarray, alpha: float = 0.3):
    if len(probs) == 0:
        return 0, 0.0
    agg = probs[0].copy()
    for i in range(1, len(probs)):
        agg = alpha * probs[i] + (1 - alpha) * agg
    return int(agg.argmax()), float(agg.max())


# ========== Load CLIP + SAE + Linear ==========
def load_sae_pipeline(num_classes: int):
    clip_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16").to(DEVICE)
    clip_proc  = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch16")

    ckpt  = torch.load(SAE_PATH, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)

    d_in  = 768
    d_sae = state["W_enc"].shape[1]

    cfg = ViTSAERunnerConfig(d_in=d_in)
    cfg.d_sae = d_sae
    sae = SparseAutoencoder(cfg, device=DEVICE)
    sae.load_state_dict(state, strict=True)
    sae.to(DEVICE).eval()

    lin_ckpt = torch.load(LINEAR_PATH, map_location="cpu", weights_only=False)

    clf = nn.Linear(d_sae, num_classes)
    # compatible: {"state_dict": ...} or direct state_dict
    lin_state = lin_ckpt.get("state_dict", lin_ckpt)
    # compatible with DataParallel
    if any(k.startswith("module.") for k in lin_state.keys()):
        lin_state = {k.replace("module.", "", 1): v for k, v in lin_state.items()}
    clf.load_state_dict(lin_state, strict=True)
    clf.to(DEVICE).eval()

    return clip_model, clip_proc, sae, clf


# ========== Sample frames by target_fps ==========
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


# ========== Single frame: CLIP -> patch mean -> SAE -> Linear ==========
@torch.no_grad()
def predict_frame_sae(clip_model, clip_proc, sae, clf, frame_bgr):
    img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    inputs = clip_proc(images=img, return_tensors="pt").to(DEVICE)

    vit_out = clip_model(**inputs)
    patches = vit_out.last_hidden_state[:, 1:, :]  # [1, n_patches, 768]
    x = patches.mean(dim=1)                        # [1, 768]

    sae_out = sae(x)
    feats = sae_out[1] if isinstance(sae_out, tuple) else sae_out

    logits = clf(feats)  # [1, C]
    probs  = torch.softmax(logits, dim=-1).squeeze(0).detach().cpu().numpy()
    return probs


# ========== Video inference ==========
@torch.no_grad()
def infer_video_with_sae(clip_model, clip_proc, sae, clf, video_path: str,
                         target_fps=5, max_frames=96, show_progress=True):
    frames, fps = iter_sampled_frames(video_path, target_fps=target_fps, max_frames=max_frames)
    probs_list, argmax_list = [], []

    rng = frames
    if show_progress:
        rng = tqdm(frames, total=len(frames), desc=Path(video_path).name, unit="frm", leave=False)

    for frame in rng:
        prob = predict_frame_sae(clip_model, clip_proc, sae, clf, frame)
        probs_list.append(prob)
        argmax_list.append(int(prob.argmax()))

    probs = np.stack(probs_list) if probs_list else np.zeros((0, clf.out_features), dtype=np.float32)
    argmx = np.array(argmax_list, dtype=int)
    return probs, argmx, fps


# ========== Batch eval: WLASL/NSLT directory structure ==========
def batch_infer_and_evaluate_wlasl(
    clip_model, clip_proc, sae, clf,
    video_root: str,
    output_root: str,
    train_frames_dir: str,
    target_fps=5,
    max_frames=96,
    agg_method="ewa",  # "ewa"/"ma"/"vote"
):
    video_root = Path(video_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # Align class order with training ImageFolder
    train_ds = datasets.ImageFolder(train_frames_dir)
    class_to_idx = train_ds.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    print("Loaded class order from:", train_frames_dir)
    print("num_classes =", len(class_names))
    print("first10 =", class_names[:10])

    total, correct = 0, 0
    bad_open_or_empty = 0
    skipped_unknown_class = 0

    mp4_list = sorted(video_root.rglob("*.mp4"))
    print(f"Found {len(mp4_list)} mp4 under {video_root}")

    for video_path in tqdm(mp4_list, desc="videos", unit="vid"):
        true_gloss = video_path.parent.name
        if true_gloss not in class_to_idx:
            skipped_unknown_class += 1
            continue
        true_cls = class_to_idx[true_gloss]

        try:
            probs, argmx, fps = infer_video_with_sae(
                clip_model, clip_proc, sae, clf,
                str(video_path),
                target_fps=target_fps,
                max_frames=max_frames,
                show_progress=False
            )
        except Exception:
            bad_open_or_empty += 1
            continue

        if len(probs) == 0:
            bad_open_or_empty += 1
            continue

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

        # Save per-frame probabilities (keep gloss subfolders)
        rel = video_path.relative_to(video_root)  # gloss/video.mp4
        out_dir = output_root / rel.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        csv_path = out_dir / f"{video_path.stem}_per_frame_probs.csv"
        df = pd.DataFrame(probs, columns=class_names)
        df["argmax_id"]  = argmx
        df["argmax_cls"] = [class_names[i] for i in argmx]
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")

        total += 1
        if final_pred == true_cls:
            correct += 1

    acc = (correct / total * 100.0) if total > 0 else 0.0
    print("\n==== Summary (PatchSAE video inference) ====")
    print("video_root:", video_root)
    print("output_root:", output_root)
    print("agg_method:", agg_method)
    print("target_fps:", target_fps, "max_frames:", max_frames)
    print(f"processed: {total}  correct: {correct}  acc: {acc:.2f}%")
    print("bad_open_or_empty:", bad_open_or_empty)
    print("skipped_unknown_class:", skipped_unknown_class)

    return acc


# ========== main ==========
if __name__ == "__main__":
    args = parse_args()

    # override path globals (paths only; logic unchanged)
    SAE_PATH = args.sae_path
    LINEAR_PATH = args.linear_path
    TRAIN_FRAMES_DIR = args.train_frames_dir
    VIDEO_ROOT = args.video_root
    OUTPUT_DIR = args.output_dir

    TARGET_FPS = args.target_fps
    MAX_FRAMES = args.max_frames

    MA_WIN = args.ma_win
    EWA_ALPHA = args.ewa_alpha

    # get num_classes from ImageFolder training frames
    train_ds = datasets.ImageFolder(TRAIN_FRAMES_DIR)
    num_classes = len(train_ds.classes)

    clip_model, clip_proc, sae, clf = load_sae_pipeline(num_classes=num_classes)

    batch_infer_and_evaluate_wlasl(
        clip_model, clip_proc, sae, clf,
        video_root=VIDEO_ROOT,
        output_root=OUTPUT_DIR,
        train_frames_dir=TRAIN_FRAMES_DIR,
        target_fps=TARGET_FPS,
        max_frames=MAX_FRAMES,
        agg_method=args.agg_method,  # "ewa"/"ma"/"vote"
    )
