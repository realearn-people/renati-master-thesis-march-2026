import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from src.sae_training.sparse_autoencoder import SparseAutoencoder, ViTSAERunnerConfig

# =============================
# Argument Parser
# =============================
parser = argparse.ArgumentParser(description="Linear probe on SAE features")

parser.add_argument("--sae_path", type=str, required=True,
                    help="Path to trained SAE checkpoint")

parser.add_argument("--train_acts", type=str, required=True)
parser.add_argument("--train_labels", type=str, required=True)
parser.add_argument("--test_acts", type=str, required=True)
parser.add_argument("--test_labels", type=str, required=True)

parser.add_argument("--num_classes", type=int, required=True)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--save_path", type=str, default="clf_ckpt/best_linear.pth")

args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

# =============================
# Load features
# =============================
Xtr = np.load(args.train_acts)
ytr = np.load(args.train_labels)
Xte = np.load(args.test_acts)
yte = np.load(args.test_labels)

# Mean pooling over patches if needed
if Xtr.ndim == 3:
    Xtr = Xtr.mean(1)
    Xte = Xte.mean(1)

# =============================
# Load SAE
# =============================
ckpt = torch.load(args.sae_path, map_location="cpu", weights_only=False)
state = ckpt.get("state_dict", ckpt)

d_in = Xtr.shape[1]
d_sae = state["W_enc"].shape[1]

cfg = ViTSAERunnerConfig(d_in=d_in)
cfg.d_sae = d_sae

sae = SparseAutoencoder(cfg, device="cpu")
sae.load_state_dict(state, strict=True)
sae.eval()

def encode(X):
    x = torch.from_numpy(X).float()
    with torch.no_grad():
        out = sae(x)
    if isinstance(out, tuple):
        _, feats, *_ = out
    else:
        feats = out
    return feats.cpu().numpy()

Ztr = encode(Xtr)
Zte = encode(Xte)

# =============================
# Linear classifier
# =============================
clf = nn.Linear(Ztr.shape[1], args.num_classes).to(device)
optimizer = torch.optim.Adam(clf.parameters(), lr=args.lr)
criterion = nn.CrossEntropyLoss()

def run_epoch(X, y, train=True):
    dataset = TensorDataset(
        torch.from_numpy(X).float(),
        torch.from_numpy(y).long()
    )
    loader = DataLoader(dataset,
                        batch_size=args.batch_size,
                        shuffle=train)

    total, correct = 0, 0
    clf.train(train)

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)

        if train:
            optimizer.zero_grad()

        out = clf(xb)
        loss = criterion(out, yb)

        if train:
            loss.backward()
            optimizer.step()

        pred = out.argmax(1)
        correct += (pred == yb).sum().item()
        total += len(yb)

    return correct / total

best_acc = 0.0

for epoch in range(args.epochs):
    train_acc = run_epoch(Ztr, ytr, True)
    test_acc = run_epoch(Zte, yte, False)

    print(f"Epoch {epoch+1:03d} | "
          f"Train Acc: {train_acc:.3f} | "
          f"Test Acc: {test_acc:.3f}")

    if test_acc > best_acc:
        best_acc = test_acc
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        torch.save({"state_dict": clf.state_dict(),
                    "test_acc": best_acc},
                   args.save_path)

print("Best test accuracy:", best_acc)
