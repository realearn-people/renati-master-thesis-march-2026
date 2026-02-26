import argparse
import os
import numpy as np
import torch
from transformers import CLIPVisionModel
from torchvision import datasets, transforms

# =============================
# Argument Parser
# =============================
parser = argparse.ArgumentParser(description="Extract CLIP patch features")
parser.add_argument("--dataset_path", type=str, required=True,
                    help="Path to ImageFolder dataset")
parser.add_argument("--save_dir", type=str, default="acts",
                    help="Directory to save extracted features")
parser.add_argument("--save_prefix", type=str, required=True,
                    help="Prefix for saved .npy files")

args = parser.parse_args()

# =============================
# Setup
# =============================
device = "cuda" if torch.cuda.is_available() else "cpu"

model = CLIPVisionModel.from_pretrained(
    "openai/clip-vit-base-patch16"
).to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.48145466, 0.4578275, 0.40821073],
        [0.26862954, 0.26130258, 0.27577711]
    ),
])

dataset = datasets.ImageFolder(
    root=args.dataset_path,
    transform=transform
)

# =============================
# Feature Extraction
# =============================
acts, labels = [], []

with torch.no_grad():
    for i, (img, lab) in enumerate(dataset):
        img = img.unsqueeze(0).to(device)
        out = model(img, output_hidden_states=True)

        patch_features = out.hidden_states[-1][:, 1:, :]
        acts.append(patch_features.cpu().numpy())
        labels.append(lab)

        if i % 100 == 0:
            print(f"{i}/{len(dataset)} images processed")

acts = np.concatenate(acts, axis=0)
labels = np.array(labels)

os.makedirs(args.save_dir, exist_ok=True)

np.save(os.path.join(args.save_dir, f"{args.save_prefix}_acts.npy"), acts)
np.save(os.path.join(args.save_dir, f"{args.save_prefix}_labels.npy"), labels)

print("Saved features:", acts.shape)