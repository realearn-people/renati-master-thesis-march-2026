import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from model import construct_PPNet, PPNet   
import settings


# Path configuration

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#  checkpoint (change this to the model you want to analyze)
checkpoint_path = Path("./saved_models/vgg16_bn/Rav_cutsori_vgg16_bn 60 16/90push_snapshot0.9938.pth")

# test test data path (corresponding to test_dir in settings)
test_dir = Path('./ravdess_ori/Ravdata_Ori_split/test')

# Build PPNet with the same settings as training
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

#  model loading
def load_ppnet(checkpoint_path: Path):
    # first try loading as pure weights
    try:
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if isinstance(state, dict):
            model = build_model_from_settings()
            key = 'state_dict' if 'state_dict' in state else \
                  'model_state_dict' if 'model_state_dict' in state else None
            to_load = state[key] if key is not None else state
            model.load_state_dict(to_load, strict=True)
            print("[Load] Loaded state_dict with weights_only=True.")
            return model.to(device).eval()
    except Exception as e:
        print(f"[Load] weights_only=True failed (likely a full model pickle): {e}")

    # load full model
    import torch.serialization as ts
    ts.add_safe_globals([PPNet])  
    obj = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(obj, nn.Module):
        print("[Load] Loaded full model object (pickle).")
        return obj.to(device).eval()
    if isinstance(obj, dict):
        
        for k in ('state_dict', 'model_state_dict'):
            if k in obj and isinstance(obj[k], dict):
                model = build_model_from_settings()
                model.load_state_dict(obj[k], strict=False)
                print(f"[Load] Loaded dict['{k}'] into PPNet (strict=False).")
                return model.to(device).eval()
        if 'model' in obj and isinstance(obj['model'], nn.Module):
            print("[Load] Loaded obj['model'] which is a full nn.Module.")
            return obj['model'].to(device).eval()
    raise RuntimeError(f"Unexpected checkpoint content type: {type(obj)}")

model = load_ppnet(checkpoint_path)
print("[Info] Model ready.")

# Build test_loader (same normalization as training)
test_tf = transforms.Compose([
    transforms.Resize((settings.img_size, settings.img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])
test_dataset = datasets.ImageFolder(str(test_dir), transform=test_tf)
test_loader = DataLoader(test_dataset, batch_size=settings.test_batch_size, shuffle=False)
class_names = test_dataset.classes
num_classes = len(class_names)
print(f"[Info] Test samples: {len(test_dataset)} | Classes: {class_names}")

# Inference and collect predictions
softmax = nn.Softmax(dim=1)
y_true, y_pred = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        logits, _ = model(xb)  
        pred = logits.argmax(dim=1)
        y_true.append(yb.numpy())
        y_pred.append(pred.cpu().numpy())

y_true = np.concatenate(y_true)
y_pred = np.concatenate(y_pred)
acc = (y_true == y_pred).mean()
print(f"[Summary] Test Accuracy: {acc*100:.2f}%  |  Samples: {len(y_true)}")

# Confusion Matrix + PRF1
def confusion_matrix_np(y_true, y_pred, C):
    cm = np.zeros((C, C), dtype=int)
    for t,p in zip(y_true, y_pred):
        cm[t,p] += 1
    return cm

cm = confusion_matrix_np(y_true, y_pred, num_classes)
def plot_cm(cm, labels, title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(cm, cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    thr = cm.max()/2 if cm.max()>0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i,j]), ha="center", va="center",
                    color="white" if cm[i,j]>thr else "black")
    plt.tight_layout(); plt.show()

plot_cm(cm, class_names, "Confusion Matrix (Test)")

eps = 1e-12
tp = np.diag(cm)
support = cm.sum(axis=1)      
pred_cnt = cm.sum(axis=0)     
precision = tp / (pred_cnt + eps)
recall    = tp / (support + eps)
f1        = 2*precision*recall/(precision+recall+eps)
macro_p, macro_r, macro_f1 = precision.mean(), recall.mean(), f1.mean()

print(f"\n{'Class':<15} {'Prec':>8} {'Rec':>8} {'F1':>8} {'Support':>8}")
for name, p, r, f, s in zip(class_names, precision, recall, f1, support):
    print(f"{name:<15} {p:8.4f} {r:8.4f} {f:8.4f} {s:8d}")
print("-"*55)
print(f"{'Macro Avg':<15} {macro_p:8.4f} {macro_r:8.4f} {macro_f1:8.4f} {support.sum():8d}")