import torch
import matplotlib.pyplot as plt
from pathlib import Path

from transformers import CLIPVisionModel, CLIPImageProcessor
from PIL import Image

# =====================
# 1. 加载 SAE checkpoint
# =====================
SAE_CKPT = Path(
    r"E:/PatchSae/sae_ckpt/v1_fo/final_sparse_autoencoder_openai/openai/clip-vit-base-patch16_-2_resid_49152.pt"
)
IMG_PATH = Path(
    r"E:/PatchSae/report_sae/v1_fo_0.7/n/n1/frame.png"
)

ckpt = torch.load(SAE_CKPT, map_location="cpu", weights_only=False)

# 从 ckpt 中取 encoder 参数
W_enc = ckpt["state_dict"]["W_enc"]   # (D, K)
b_enc = ckpt["state_dict"]["b_enc"]   # (K,)

W_enc = W_enc.cpu()
b_enc = b_enc.cpu()

print("Loaded SAE encoder:", W_enc.shape, b_enc.shape)

# =====================
# 2. 用真实图像 → ViT feature
# =====================

device = "cuda" if torch.cuda.is_available() else "cpu"

# ViT (和 train_sae_vit 一致)
vit = CLIPVisionModel.from_pretrained(
    "openai/clip-vit-base-patch16"
).to(device)
vit.eval()

processor = CLIPImageProcessor.from_pretrained(
    "openai/clip-vit-base-patch16"
)

# 读图
img = Image.open(IMG_PATH).convert("RGB")
inputs = processor(images=img, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

# forward
with torch.no_grad():
    outputs = vit(**inputs)
    # outputs.last_hidden_state: (1, n_tokens, 768)

    # 👉 image-level feature（和你现在代码假设的 (1, D) 对齐）
    x_img = outputs.last_hidden_state.mean(dim=1)  # (1, 768)

x_img = x_img.cpu()

# =====================
# 3. 计算 latent activation
# =====================
with torch.no_grad():
    z = torch.relu(x_img @ W_enc + b_enc)  # (1, K)
    z = z.squeeze(0).numpy()               # (K,)

print("Latent dim:", z.shape)

# =====================
# 4. 画折线图 + top neurons
# =====================
import numpy as np

topk = 5
top_idx = np.argsort(z)[-topk:]

plt.figure(figsize=(10, 3))
plt.plot(z, linewidth=1)
plt.scatter(top_idx, z[top_idx], color="red", zorder=3)

for i in top_idx:
    plt.text(i, z[i], str(i), fontsize=8)

plt.xlabel("Latent neuron index")
plt.ylabel("Activation")
plt.title("Image-level SAE latent activations")
plt.tight_layout()
plt.show()
