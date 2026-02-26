import torch
# from torch._C import T
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import cv2, json, os
from tqdm import tqdm
import argparse

# ==== PatchSAE related ====
from transformers import CLIPVisionModel, CLIPImageProcessor
from src.sae_training.sparse_autoencoder import SparseAutoencoder, ViTSAERunnerConfig

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ====== Default paths (can be overridden via CLI) ======
SAE_PATH    = r"E:/PatchSae/sae_ckpt/lcym3qu6/final_sparse_autoencoder_openai/openai/clip-vit-base-patch16_-2_resid_49152.pt"
LINEAR_PATH = r"E:/WLASL_HF/clf_ckpt/linear_wlasl_patchsae_best.pth"
VIDEO_PATH  = Path(r"E:/ProtoPNet/all/Negative/02-02-04-01-02-01-13.mp4")  # single-video mode
# ======================================================

STRIDE     = 1
MAX_FRAMES = 300

FACE_CROP   = False
FACE_MARGIN = 0.35

TARGET_SIZE = 224
PATCH_SIZE  = TARGET_SIZE // 2

HAAR_PATH   = str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
face_detector = cv2.CascadeClassifier(HAAR_PATH)


def parse_args():
    p = argparse.ArgumentParser("PatchSAE video inference (paths via CLI)")
    p.add_argument("--sae_path", type=str, default=SAE_PATH, help="Path to SAE checkpoint (.pt)")
    p.add_argument("--linear_path", type=str, default=LINEAR_PATH, help="Path to linear head checkpoint (.pth)")
    p.add_argument("--video_path", type=str, default=str(VIDEO_PATH), help="Single video path (.mp4)")

    # Used to align class order (kept for compatibility; you currently hardcode class_names)
    p.add_argument(
        "--train_dir",
        type=str,
        default=r"E:/WLASL_HF/nslt10_frames_5fps_upperbody_flat_fix/train",
        help="ImageFolder train directory (kept for compatibility)"
    )

    # Batch mode paths
    p.add_argument("--batch_mode", action="store_true", help="Enable batch mode")
    p.add_argument("--input_dir", type=str, default=r"E:/WLASL_HF/nslt10_split_gloss",
                   help="Batch mode: directory containing videos")
    p.add_argument("--output_dir", type=str, default="./report_sae/wlasl10",
                   help="Batch mode: output directory")

    # Optional overrides
    p.add_argument("--stride", type=int, default=STRIDE)
    p.add_argument("--max_frames", type=int, default=MAX_FRAMES)
    return p.parse_args()


# ========== Load CLIP + SAE + Linear ==========
def load_sae_pipeline(num_classes: int):
    # CLIP
    clip_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16").to(DEVICE)
    clip_proc  = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch16")

    # SAE
    ckpt  = torch.load(SAE_PATH, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    d_in  = 768
    d_sae = state["W_enc"].shape[1]

    cfg = ViTSAERunnerConfig(d_in=d_in)
    cfg.d_sae = d_sae
    sae = SparseAutoencoder(cfg, device=DEVICE)
    sae.load_state_dict(state, strict=True)
    sae.to(DEVICE).eval()

    # Linear head
    lin_ckpt = torch.load(LINEAR_PATH, map_location="cpu", weights_only=False)
    clf = nn.Linear(d_sae, num_classes)
    clf.load_state_dict(lin_ckpt["state_dict"])
    clf.to(DEVICE).eval()

    return clip_model, clip_proc, sae, clf


# ========== Aggregation utils (keep original behavior) ==========
def majority_vote(argmax_ids: np.ndarray):
    vals, cnts = np.unique(argmax_ids, return_counts=True)
    cls = int(vals[np.argmax(cnts)])
    conf = float((argmax_ids == cls).mean())
    return cls, conf

def moving_average(probs: np.ndarray, win: int = 9):
    N, C = probs.shape
    half = win // 2
    out = np.zeros_like(probs)
    for i in range(N):
        L, R = max(0, i-half), min(N, i+half+1)
        out[i] = probs[L:R].mean(axis=0)
    p = out.mean(axis=0)
    return int(p.argmax()), float(p.max())

def ewa(probs: np.ndarray, alpha: float = 0.3):
    agg = probs[0].copy()
    for i in range(1, len(probs)):
        agg = alpha*probs[i] + (1-alpha)*agg
    return int(agg.argmax()), float(agg.max())


def crop_face_bgr(frame_bgr, margin=0.35):
    h, w = frame_bgr.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.1, 5, minSize=(60,60))
    if len(faces) == 0:
        return frame_bgr
    x,y,fw,fh = max(faces, key=lambda r: r[2]*r[3])
    cx, cy = x + fw//2, y + fh//2
    side = int(max(fw, fh) * (1 + 2*margin))
    x1 = max(0, cx - side//2); y1 = max(0, cy - side//2)
    x2 = min(w, cx + side//2); y2 = min(h, cy + side//2)
    return frame_bgr[y1:y2, x1:x2]


# ========== Frame-level inference ==========
@torch.no_grad()
def predict_frame_sae(clip_model, clip_proc, sae, clf, frame_bgr):
    if FACE_CROP:
        frame_bgr = crop_face_bgr(frame_bgr, margin=FACE_MARGIN)

    # Optional: 2x2 grid
    if USE_GRID:
        frame_bgr = make_face_grid(frame_bgr)

    img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    inputs = clip_proc(images=img, return_tensors="pt").to(DEVICE)

    vit_out = clip_model(**inputs)
    # Remove CLS token; keep patches: [1, n_patches, 768]
    patches = vit_out.last_hidden_state[:, 1:, :]
    # Mean pool patches → [1, 768]
    x = patches.mean(dim=1)

    # SAE forward
    sae_out = sae(x)
    if isinstance(sae_out, tuple):
        _, feats, *_ = sae_out
    else:
        feats = sae_out

    logits = clf(feats)  # [1, num_classes]
    probs  = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()  # [C]
    return probs


# ========== Video-level inference ==========
@torch.no_grad()
def infer_video_with_sae(clip_model, clip_proc, sae, clf, video_path: str,
                         stride=1, max_frames=None, show_progress=True):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)

    probs_list, argmax_list = [], []
    used = 0

    rng = range(total)
    if show_progress:
        rng = tqdm(rng, total=total, desc=Path(video_path).name, unit="frm", leave=False)

    for i in rng:
        ret, frame = cap.read()
        if not ret:
            break
        if i % stride != 0:
            continue

        prob = predict_frame_sae(clip_model, clip_proc, sae, clf, frame)
        probs_list.append(prob)
        argmax_list.append(int(prob.argmax()))

        used += 1
        if max_frames and used >= max_frames:
            break

    cap.release()
    probs = np.stack(probs_list) if probs_list else np.zeros((0, clf.out_features))
    argmx = np.array(argmax_list, dtype=int)
    return probs, argmx, fps


# ========== Batch evaluation ==========
def batch_infer_and_evaluate(clip_model, clip_proc, sae, clf,
                             input_dir, output_dir, class_names,
                             stride=1):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    total, correct = 0, 0

    for video_path in sorted(input_dir.rglob("*.mp4")):
        print(f"\n▶ Running: {video_path.name}")

        # Parse RAVDESS emotion id → 3-class label
        try:
            emotion_id = int(video_path.stem.split("-")[2])
            id_to_class_index = {
                1: 1,  # neutral
                2: 1,  # calm → neutral
                3: 2,  # happy → positive
                4: 0,  # sad → negative
                5: 0,  # angry → negative
            }
            """
            id_to_class_index = {
                1: 4,  # neutral
                2: 1,  # calm
                3: 3,  # happy
                4: 5,  # sad
                5: 0,  # angry
                6: 2,  # fearful
            }
            """
            if emotion_id not in id_to_class_index:
                print(f"Skip unused emotion {emotion_id} ({video_path.name})")
                continue
            true_cls = id_to_class_index[emotion_id]
        except Exception:
            true_cls = -1

        probs, argmx, fps = infer_video_with_sae(
            clip_model, clip_proc, sae, clf,
            str(video_path), stride=stride, max_frames=None, show_progress=True
        )

        if len(probs) == 0:
            print("No frames processed, skip.")
            continue

        # Aggregation (vote + MA + EWA)
        pred_cls, conf      = majority_vote(argmx)
        pred_ma,  conf_ma   = moving_average(probs, win=9)
        pred_ewa, conf_ewa  = ewa(probs, alpha=0.3)

        csv_path = output_dir / f"{video_path.stem}_probs.csv"
        df = pd.DataFrame(probs, columns=class_names)
        df["argmax_id"]  = argmx
        df["argmax_cls"] = [class_names[i] for i in argmx]
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")

        print(f"Saved results to: {csv_path.name}")
        print(f"[vote] pred={class_names[pred_cls]}  conf≈{conf:.3f}")
        print(f"[ m.a] pred={class_names[pred_ma]}   conf≈{conf_ma:.3f}")
        print(f"[ ewa] pred={class_names[pred_ewa]}  conf≈{conf_ewa:.3f}")

        final_pred_cls = pred_ewa  # keep original: use EWMA as final decision
        total += 1
        if true_cls == final_pred_cls:
            correct += 1

    if total > 0:
        acc = correct / total * 100
        print(f"\n✅ Overall Accuracy: {acc:.2f}%  ({correct}/{total})")
    else:
        print("⚠️ No videos processed.")


# ========== main ==========
if __name__ == "__main__":
    from torchvision import datasets

    args = parse_args()

    # Override globals from CLI (paths only; logic unchanged)
    SAE_PATH = args.sae_path
    LINEAR_PATH = args.linear_path
    VIDEO_PATH = Path(args.video_path)
    STRIDE = args.stride
    MAX_FRAMES = args.max_frames

    # ====== Keep class order alignment (same as your original) ======
    TRAIN_DIR = args.train_dir
    train_ds = datasets.ImageFolder(TRAIN_DIR)

    # You manually set 3 classes; keep unchanged
    class_names = ["negative", "neutral", "positive"]
    # class_names = ["neutral", "calm", "happy", "sad", "angry", "fearful"]
    print("class name:", class_names)

    clip_model, clip_proc, sae, clf = load_sae_pipeline(num_classes=len(class_names))

    BATCH_MODE = bool(args.batch_mode)

    if not BATCH_MODE:
        probs, argmx, fps = infer_video_with_sae(
            clip_model, clip_proc, sae, clf,
            str(VIDEO_PATH), stride=STRIDE, max_frames=MAX_FRAMES, show_progress=True
        )
        print(f"Frames processed: {len(probs)} | FPS: {fps:.1f}")

        vid_cls_vote, conf_vote = majority_vote(argmx)
        vid_cls_ma,   conf_ma   = moving_average(probs, win=9)
        vid_cls_ewa,  conf_ewa  = ewa(probs, alpha=0.3)
        print(f"[vote] {class_names[vid_cls_vote]}  conf≈{conf_vote:.3f}")
        print(f"[ m.a] {class_names[vid_cls_ma]}   conf≈{conf_ma:.3f}")
        print(f"[ ewa] {class_names[vid_cls_ewa]}  conf≈{conf_ewa:.3f}")

        out_csv = VIDEO_PATH.with_suffix(".patchsae_probs.csv")
        df = pd.DataFrame(probs, columns=class_names)
        df["argmax_id"]  = argmx
        df["argmax_cls"] = [class_names[i] for i in argmx]
        df.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print("Saved per-frame probabilities:", out_csv)

    else:
        INPUT_DIR  = Path(args.input_dir)
        OUTPUT_DIR = Path(args.output_dir)
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        batch_infer_and_evaluate(
            clip_model, clip_proc, sae, clf,
            INPUT_DIR, OUTPUT_DIR, class_names,
            stride=STRIDE
        )
