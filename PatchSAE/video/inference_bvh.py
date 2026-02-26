# ===============================
# inference_bvh.py
# ===============================

import math
import random
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from tqdm import tqdm

import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPImageProcessor

from src.sae_training.sparse_autoencoder import SparseAutoencoder, ViTSAERunnerConfig

# ===============================
# ======= 配置区域（必改） =======
# ===============================

SAE_PATH = r"D:/PatchSae/sae_ckpt/68nh3iq4/final_sparse_autoencoder_openai/openai/clip-vit-base-patch16_-2_resid_49152.pt"
LINEAR_PATH = r"D:/PatchSae/clf_ckpt/linear_dime_patchsae_best.pth"

BVH_DIR = Path(r"D:/bvh")     # 放 .bvh 的目录（递归扫描）
OUT_DIR = Path(r"D:/DIEM_A_report2")  # 输出 csv 的目录

CLASS_NAMES = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]

N_PER_CLASS = 8      # ⭐ 方案 C：每类抽多少个（8≈48个）
RANDOM_SEED = 42

STRIDE_FRAMES = 5    # 每隔几帧抽一帧
MAX_FRAMES = 300     # 每个 BVH 最多处理多少帧
IMG_SIZE = 224

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===============================
# ======= 模型加载 =======
# ===============================

def load_models():
    clip_model = CLIPVisionModel.from_pretrained(
        "openai/clip-vit-base-patch16"
    ).to(DEVICE).eval()

    clip_proc = CLIPImageProcessor.from_pretrained(
        "openai/clip-vit-base-patch16"
    )

    ckpt = torch.load(SAE_PATH, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)

    d_sae = state["W_enc"].shape[1]
    cfg = ViTSAERunnerConfig(d_in=768)
    cfg.d_sae = d_sae

    sae = SparseAutoencoder(cfg, device=DEVICE)
    sae.load_state_dict(state, strict=True)
    sae.to(DEVICE).eval()

    lin_ckpt = torch.load(LINEAR_PATH, map_location="cpu", weights_only=False)
    clf = nn.Linear(d_sae, len(CLASS_NAMES))
    clf.load_state_dict(lin_ckpt["state_dict"])
    clf.to(DEVICE).eval()

    return clip_model, clip_proc, sae, clf

# ===============================
# ======= BVH 解析 =======
# ===============================

class Joint:
    def __init__(self, name):
        self.name = name
        self.offset = np.zeros(3)
        self.channels = []
        self.channel_idx = []
        self.children = []
        self.parent = None

def parse_bvh(path: Path):
    lines = path.read_text(errors="ignore").splitlines()
    i = 0

    # 找到 HIERARCHY
    while i < len(lines) and "HIERARCHY" not in lines[i].upper():
        i += 1
    if i >= len(lines):
        raise RuntimeError("BVH missing HIERARCHY")

    i += 1  # next line should be ROOT ...

    # ROOT xxx
    while i < len(lines) and not lines[i].strip().upper().startswith("ROOT"):
        i += 1
    if i >= len(lines):
        raise RuntimeError("BVH missing ROOT")

    root = Joint(lines[i].split()[1])
    stack = [root]
    i += 1

    ch_cursor = 0

    def skip_end_site_block(i: int) -> int:
        """
        当前行是 'End Site'，跳过:
          End Site
          {
            OFFSET ...
          }
        返回跳过后新的 i（指向 End Site 块结束后的下一行）
        """
        # End Site
        i += 1
        # skip optional '{'
        if i < len(lines) and lines[i].strip() == "{":
            i += 1
        # skip lines until matching '}'
        while i < len(lines) and lines[i].strip() != "}":
            i += 1
        # skip the closing '}'
        if i < len(lines) and lines[i].strip() == "}":
            i += 1
        return i

    # 解析 hierarchy，直到 MOTION
    while i < len(lines):
        line = lines[i].strip()

        if line.upper().startswith("MOTION"):
            i += 1
            break

        if line == "" or line == "{":
            i += 1
            continue

        if line.startswith("End Site"):
            i = skip_end_site_block(i)
            continue

        if line.startswith("JOINT"):
            j = Joint(line.split()[1])
            j.parent = stack[-1]
            stack[-1].children.append(j)
            stack.append(j)
            i += 1
            continue

        if line.startswith("OFFSET"):
            # OFFSET x y z
            parts = line.split()
            stack[-1].offset = np.array(list(map(float, parts[1:4])), dtype=np.float32)
            i += 1
            continue

        if line.startswith("CHANNELS"):
            parts = line.split()
            n = int(parts[1])
            chs = parts[2:2+n]
            stack[-1].channels = chs
            stack[-1].channel_idx = list(range(ch_cursor, ch_cursor + n))
            ch_cursor += n
            i += 1
            continue

        if line == "}":
            # 关键修复：不要把 root pop 掉
            if len(stack) > 1:
                stack.pop()
            i += 1
            continue

        # 其它行直接跳过（更鲁棒）
        i += 1

    # MOTION header
    # Frames: N
    while i < len(lines) and not lines[i].strip().lower().startswith("frames:"):
        i += 1
    if i >= len(lines):
        raise RuntimeError("BVH missing Frames:")

    frames = int(lines[i].split(":")[1].strip())
    i += 1

    # Frame Time: t
    while i < len(lines) and not lines[i].strip().lower().startswith("frame time:"):
        i += 1
    if i >= len(lines):
        raise RuntimeError("BVH missing Frame Time:")

    i += 1  # motion data begins

    motion = []
    for k in range(frames):
        if i + k >= len(lines):
            break
        row = lines[i + k].strip()
        if not row:
            continue
        motion.append(list(map(float, row.split())))

    motion = np.array(motion, dtype=np.float32)
    return root, motion

# ===============================
# ======= Forward Kinematics =======
# ===============================

def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]])

def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]])

def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]])

def fk(root, frame):
    pos = {}

    def dfs(j, R, t):
        off = j.offset.copy()
        rot = np.eye(3)

        if j.channels:
            vals = frame[j.channel_idx]
            for ch, v in zip(j.channels, vals):
                if ch == "Xposition": off[0] += v
                if ch == "Yposition": off[1] += v
                if ch == "Zposition": off[2] += v
            for ch, v in zip(j.channels, vals):
                a = math.radians(v)
                if ch == "Xrotation": rot = rot @ rot_x(a)
                if ch == "Yrotation": rot = rot @ rot_y(a)
                if ch == "Zrotation": rot = rot @ rot_z(a)

        gt = t + R @ off
        gR = R @ rot
        pos[j.name] = gt
        for c in j.children:
            dfs(c, gR, gt)

    dfs(root, np.eye(3), np.zeros(3))
    return pos

# ===============================
# ======= 渲染骨架 =======
# ===============================

def get_true_label_from_name(filename: str, class_names):
    name = filename.lower()
    for i, emo in enumerate(class_names):
        if f"_{emo}_" in name:   # 你的命名就是这种
            return i
    return -1

def render_skeleton(root, pos):
    pts = np.array([[p[0], p[1]] for p in pos.values()])
    mn, mx = pts.min(0), pts.max(0)
    pts = (pts - mn) / (mx - mn + 1e-6)
    pts = pts * 0.84 + 0.08

    name2pt = dict(zip(pos.keys(), pts))

    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), "black")
    draw = ImageDraw.Draw(img)

    def px(xy):
        return int(xy[0]*223), int((1-xy[1])*223)

    def draw_edges(j):
        for c in j.children:
            draw.line([px(name2pt[j.name]), px(name2pt[c.name])],
                      fill="white", width=2)
            draw_edges(c)

    draw_edges(root)
    return img

# ===============================
# ======= 单帧预测 =======
# ===============================

@torch.no_grad()
def predict_frame(img, clip_model, clip_proc, sae, clf):
    inp = clip_proc(images=img, return_tensors="pt").to(DEVICE)
    out = clip_model(**inp)
    feat = out.last_hidden_state[:,1:,:].mean(1)
    z = sae(feat)[1]
    prob = torch.softmax(clf(z), dim=-1)[0].cpu().numpy()
    return prob

# ===============================
# ======= 主流程 =======
# ===============================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    clip_model, clip_proc, sae, clf = load_models()

    all_bvh = list(BVH_DIR.rglob("*.bvh"))

    total = 0
    correct = 0

    bucket = defaultdict(list)
    for p in all_bvh:
        name = p.name.lower()
        for emo in CLASS_NAMES:
            if f"_{emo}_" in name:
                bucket[emo].append(p)

    random.seed(RANDOM_SEED)
    files = []
    for emo in CLASS_NAMES:
        random.shuffle(bucket[emo])
        files += bucket[emo][:N_PER_CLASS]

    print("=== Sampled BVH files ===")
    for emo in CLASS_NAMES:
        print(emo, sum(emo in f.name.lower() for f in files))
    print("Total:", len(files))

    for bvh in tqdm(files):
        root, motion = parse_bvh(bvh)

        probs = []
        for t in range(0, min(len(motion), MAX_FRAMES), STRIDE_FRAMES):
            pos = fk(root, motion[t])
            img = render_skeleton(root, pos)
            probs.append(predict_frame(img, clip_model, clip_proc, sae, clf))

        probs = np.stack(probs)
        avg = probs.mean(0)
        pred_id = int(avg.argmax())
        pred = CLASS_NAMES[int(avg.argmax())]
        true_id = get_true_label_from_name(bvh.name, CLASS_NAMES)
        if true_id != -1:
            total += 1
        if pred_id == true_id:
            correct += 1

        df = pd.DataFrame(probs, columns=CLASS_NAMES)
        df.to_csv(OUT_DIR / f"{bvh.stem}_probs.csv", index=False)

        print(f"{bvh.name} → {pred}")
    if total > 0:
        acc = correct / total * 100
        print(f"\n✅ File-level Accuracy (mean aggregation): {acc:.2f}%  ({correct}/{total})")
    else:
        print("\n⚠️ No ground-truth labels parsed from filenames, cannot compute accuracy.")

if __name__ == "__main__":
    main()
