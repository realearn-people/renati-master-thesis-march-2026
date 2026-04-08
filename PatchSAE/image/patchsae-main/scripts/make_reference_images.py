import os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

from transformers import CLIPVisionModel, CLIPImageProcessor

# =========================
# 1) 你需要改的 3 个地方
# =========================
SAE_CKPT = r"E:/PatchSae/sae_ckpt/v1_fo/final_sparse_autoencoder_openai/openai/clip-vit-base-patch16_-2_resid_49152.pt"
IMAGE_DIR = r"E:/PatchSae/ravdess_v1/data_face_only_split/train"  # 例: E:\data\FER2013\test
OUT_DIR   = r"E:/PatchSae/report_sae/v1_fo_0.7/1/"

# 选哪些 latent 做展示（先少做：2~3 个）
TARGET_LATENTS = [41897]  # 你可以先随便填，后面第4步教你自动选
TOPK = 6  # 每个 latent 输出几张 reference images

# 用多少图做统计（CPU建议先 200~500）
MAX_IMAGES = 800

# =========================
# 2) 工具函数
# =========================
def list_images(root: str):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    root = Path(root)
    files = [p for p in root.rglob("*") if p.suffix.lower() in exts]
    return sorted(files)

def load_sae_state(ckpt_path: str):
    """
    你的 ckpt 里一般会有 state_dict 或直接是 state_dict
    PyTorch 2.6 的 weights_only 默认 True，会报 UnpicklingError，所以我们显式 weights_only=False
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # 尽量兼容不同保存格式
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and any(k.startswith("W_enc") for k in ckpt.keys()):
        state = ckpt
    else:
        # 兜底：尝试找明显的 state_dict 字段
        for k in ["model_state_dict", "sae_state_dict"]:
            if isinstance(ckpt, dict) and k in ckpt:
                state = ckpt[k]
                break
        else:
            raise RuntimeError("Cannot find SAE state_dict in checkpoint.")

    # 关键参数
    W_enc = state["W_enc"]         # [d_in, d_sae] = [768, 49152]
    b_enc = state["b_enc"]         # [d_sae]
    return W_enc, b_enc

@torch.no_grad()
def sae_encode(patch_acts, W_enc, b_enc):
    """
    patch_acts: [B, N, d_in]  (N=patch tokens数)
    返回 feature_acts: [B, N, d_sae]
    """
    # x @ W + b
    z = torch.einsum("bnd,dk->bnk", patch_acts, W_enc) + b_enc
    # 论文/实现里一般用 ReLU 得到稀疏激活（如果你们实现是别的激活函数，这里再改）
    z = F.relu(z)
    return z

def patch_index_to_box(patch_idx: int, grid=14, patch=16):
    r = patch_idx // grid
    c = patch_idx % grid
    x0, y0 = c * patch, r * patch
    x1, y1 = (c + 1) * patch, (r + 1) * patch
    return (x0, y0, x1, y1)

def save_grid(images, title, out_path, cols=3):
    rows = int(np.ceil(len(images) / cols))
    plt.figure(figsize=(cols * 3.2, rows * 3.2))
    for i, img in enumerate(images):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

# =========================
# 3) 主流程
# =========================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    # CLIP ViT
    model_name = "openai/clip-vit-base-patch16"
    processor = CLIPImageProcessor.from_pretrained(model_name)
    vit = CLIPVisionModel.from_pretrained(model_name).to(device).eval()

    # SAE
    W_enc, b_enc = load_sae_state(SAE_CKPT)
    W_enc = W_enc.to(device)
    b_enc = b_enc.to(device)

    # images
    files = list_images(IMAGE_DIR)[:MAX_IMAGES]
    print("n_images:", len(files))
    if len(files) == 0:
        raise RuntimeError("No images found. Check IMAGE_DIR.")

    # 为每个 latent 维护 topK (score, img_path, patch_idx, preview_img)
    topk = {k: [] for k in TARGET_LATENTS}

    for idx, p in enumerate(files):
        img = Image.open(p).convert("RGB")
        inputs = processor(images=img, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)  # [1,3,224,224]

        out = vit(pixel_values=pixel_values)
        # last_hidden_state: [1, 197, 768]，第0个通常是 CLS
        h = out.last_hidden_state[:, 1:, :]  # 只用 patch tokens: [1,196,768]

        feats = sae_encode(h, W_enc, b_enc)  # [1,196,49152]

        # 对每个目标 latent：找它在 196 个 patch 上的最大激活 + patch位置
        for latent_id in TARGET_LATENTS:
            # feats[0,:,latent_id] -> [196]
            a = feats[0, :, latent_id]
            score = float(a.max().item())
            patch_idx = int(a.argmax().item())

            # 画框：在 224x224 的 resize 图上画 patch box
            vis_img = processor(images=img, return_tensors="pt")["pixel_values"][0]
            # 把 tensor 还原成 PIL（更简单：直接用 processor 的 resize 后图像）
            resized = processor(images=img, return_tensors="pt")  # 这里不取 tensor，重新用内部 resize 的结果太麻烦
            # 简化：我们直接手动 resize 到 224 再画框
            show = img.resize((224, 224))
            draw = ImageDraw.Draw(show)
            box = patch_index_to_box(patch_idx)
            draw.rectangle(box, outline="red", width=3)

            item = (score, str(p), patch_idx, show)

            arr = topk[latent_id]
            arr.append(item)
            arr.sort(key=lambda x: x[0], reverse=True)
            topk[latent_id] = arr[:TOPK]

        if (idx + 1) % 50 == 0:
            print(f"processed {idx+1}/{len(files)}")

    # 保存结果
    for latent_id, items in topk.items():
        imgs = [it[3] for it in items]
        title = f"Reference images | latent {latent_id} | top{TOPK}"
        out_path = os.path.join(OUT_DIR, f"latent_{latent_id}_top{TOPK}.png")
        save_grid(imgs, title, out_path, cols=3)

        # 也存一个文本列表，方便你写报告
        txt_path = os.path.join(OUT_DIR, f"latent_{latent_id}_top{TOPK}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            for score, path, patch_i, _ in items:
                f.write(f"{score:.6f}\tpatch={patch_i}\t{path}\n")

    print("Done. Saved to:", OUT_DIR)

if __name__ == "__main__":
    main()
