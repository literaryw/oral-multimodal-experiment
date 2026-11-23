"""
口腔亚组输入梯度扩展: G1和G3各15例 + KL concentration统计 + 描述性删除曲线
"""
import sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")

import numpy as np
import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score
from scipy.stats import mannwhitneyu
import copy

BASE_DIR = Path(".").resolve().parent
DATA_DIR = BASE_DIR / "data" / "TCGA-HNSC"
OUTPUT_DIR = DATA_DIR / "experiment_outputs"
FIGURE_DIR = BASE_DIR / "paper"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED); np.random.seed(RANDOM_SEED)
print(f"Device: {DEVICE}")

# ====== Load oral data ======
X_img = np.load(DATA_DIR / "integrated" / "X_img_uni.npy")
X_text = np.load(DATA_DIR / "integrated" / "X_text_biomedbert.npy")
y_4class = np.load(DATA_DIR / "integrated" / "y.npy")
y_binary = np.where(np.isin(y_4class, [0, 1]), 0, 1)
print(f"Oral data: {len(y_binary)} cases, Low={int((y_binary==0).sum())}, High={int((y_binary==1).sum())}")

# ====== Optimized Image Model with hook ======
class OptimizedImageModel(nn.Module):
    def __init__(self, in_dim=1024, num_classes=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 768), nn.ReLU(), nn.BatchNorm1d(768), nn.Dropout(0.4),
            nn.Linear(768, 512), nn.ReLU(), nn.BatchNorm1d(512), nn.Dropout(0.35),
            nn.Linear(512, 256), nn.ReLU(),
        )
        self.head = nn.Linear(256, num_classes)

    def forward(self, x):
        h = self.encoder(x)
        return self.head(h), h  # Return features for external grad computation

# ====== Training ======
def train_model():
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    # Use fold 0's train split for model training
    tr_idx, val_idx = list(kf.split(X_img, y_binary))[0]
    Xt, yt = X_img[tr_idx], y_binary[tr_idx]
    Xv, yv = X_img[val_idx], y_binary[val_idx]

    class_counts = np.bincount(yt)
    weights = 1.0 / class_counts[yt]
    sampler = WeightedRandomSampler(torch.DoubleTensor(weights), len(yt), replacement=True)
    tr_ds = TensorDataset(torch.FloatTensor(Xt), torch.LongTensor(yt))
    tr_loader = DataLoader(tr_ds, batch_size=32, sampler=sampler)
    val_ds = TensorDataset(torch.FloatTensor(Xv), torch.LongTensor(yv))
    val_loader = DataLoader(val_ds, batch_size=64)

    model = OptimizedImageModel().to(DEVICE)
    alpha = torch.FloatTensor([1.0, 5.0]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=alpha)
    opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)

    best_acc, best_state = 0, None
    for epoch in range(120):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            logits, _ = model(xb)
            criterion(logits, yb).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits, _ = model(xb)
                correct += (logits.argmax(1) == yb).sum().item()
                total += yb.size(0)
        val_acc = correct / total
        if val_acc > best_acc:
            best_acc, best_state = val_acc, copy.deepcopy(model.state_dict())
        if (epoch + 1) % 30 == 0:
            print(f"  Epoch {epoch+1}: val_acc={val_acc:.3f} best={best_acc:.3f}")

    model.load_state_dict(best_state)
    print(f"  Final best val_acc={best_acc:.3f}")
    return model, tr_idx, val_idx

# Try loading saved model first
model_path = OUTPUT_DIR / "oral_image_model.pt"
if model_path.exists():
    print("\n=== Loading saved model ===")
    model = OptimizedImageModel().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    model.eval()
    print(f"  Loaded oral_image_model.pt")
else:
    print("\n=== Training OptimizedImageModel ===")
    model, tr_idx, val_idx = train_model()
    torch.save(model.state_dict(), model_path)
    print("  Saved oral_image_model.pt")

# ====== Load oral patches ======
print("\n=== Loading oral patches ===")
patches_data = np.load(DATA_DIR / "integrated" / "oral_patches.npz", allow_pickle=True)
case_ids = patches_data['case_ids']
n_cases = len(case_ids)
print(f"  Oral cases with patches: {n_cases}")

# Build case-level patch lookup
patch_lookup = {}
for i in range(n_cases):
    cid = str(case_ids[i])
    arr = patches_data[f"patches_{i}"]
    patch_lookup[cid] = arr.astype(np.float32)

# ====== Absolute input-gradient patch score ======
def compute_patch_cam(model, patches):
    """Per-patch importance via input gradient magnitude"""
    model.eval()
    n_patches = patches.shape[0]
    cam_scores = np.zeros(n_patches)

    for j in range(n_patches):
        x_j = torch.FloatTensor(patches[j:j+1]).to(DEVICE)
        x_j.requires_grad_(True)
        model.zero_grad()
        logits, _ = model(x_j)
        score = logits[0, 1] - logits[0, 0]  # High - Low logit
        score.backward()
        # Descriptive absolute input-gradient magnitude for this stored patch.
        cam_scores[j] = x_j.grad.abs().mean().item()

    # Min-max normalize
    if cam_scores.max() > cam_scores.min():
        cam_scores = (cam_scores - cam_scores.min()) / (cam_scores.max() - cam_scores.min() + 1e-10)
    return cam_scores

# ====== Process G1 and G3 cases ======
print("\n=== Computing input-gradient patch scores for G1 and G3 cases ===")
# Get case IDs by grade from integrated data
case_labels_df = __import__('pandas').read_csv(DATA_DIR / "integrated" / "case_labels.csv")
g1_cases = case_labels_df[case_labels_df['label'] == 0]['case_id'].tolist()  # G1
g3_cases = case_labels_df[case_labels_df['label'] == 2]['case_id'].tolist()  # G3

# Filter to cases with patch data
g1_with_patches = [c for c in g1_cases if c in patch_lookup]
g3_with_patches = [c for c in g3_cases if c in patch_lookup]
print(f"  G1 with patches: {len(g1_with_patches)}")
print(f"  G3 with patches: {len(g3_with_patches)}")

# Select up to 15 cases each
rng = np.random.RandomState(RANDOM_SEED)
g1_selected = rng.choice(g1_with_patches, min(15, len(g1_with_patches)), replace=False).tolist()
g3_selected = rng.choice(g3_with_patches, min(15, len(g3_with_patches)), replace=False).tolist()

# Compute the input-gradient score for each selected case.
results = {"G1": {}, "G3": {}}

for grade_name, case_list in [("G1", g1_selected), ("G3", g3_selected)]:
    kl_skews = []
    for cid in case_list:
        patches = patch_lookup[cid]
        cam = compute_patch_cam(model, patches)
        n = len(cam)

        # KL skew: how concentrated is attention?
        # P = normalized CAM, Q = uniform. KL(P||Q) = sum(P * log(P/Q))
        # Higher KL = more concentrated
        eps = 1e-10
        p = cam / (cam.sum() + eps) + eps
        q = np.ones(n) / n
        kl_skew = float(np.sum(p * np.log(p / q)))
        kl_skews.append(kl_skew)

        results[grade_name][cid] = {
            "n_patches": int(n),
            "cam_scores": cam.tolist(),
            "kl_skew": kl_skew,
        }

    print(f"  {grade_name}: {len(case_list)} cases, KL skew mean={np.mean(kl_skews):.3f} std={np.std(kl_skews):.3f}")

# ====== Statistical test ======
g1_kl = [results["G1"][c]["kl_skew"] for c in g1_selected]
g3_kl = [results["G3"][c]["kl_skew"] for c in g3_selected]
stat, p_mw = mannwhitneyu(g1_kl, g3_kl, alternative='two-sided')
print(f"\n=== Mann-Whitney U: G1 vs G3 KL skew ===")
print(f"  G1 KL: {np.mean(g1_kl):.3f}+-{np.std(g1_kl):.3f}")
print(f"  G3 KL: {np.mean(g3_kl):.3f}+-{np.std(g3_kl):.3f}")
print(f"  U={stat:.1f}, p={p_mw:.4f}")

# ====== Descriptive High-Grade-probability deletion curve ======
print("\n=== Descriptive deletion-curve area ===")
# Train a simple classifier on the original features to measure AUC
from sklearn.linear_model import LogisticRegression

def compute_delete_auc(case_id, cam_scores, patches, model_orig):
    """Delete patches by score and integrate High-Grade probability.

    Without random/reverse controls this area is descriptive, not a validated
    faithfulness measure. The vertical quantity is prob_high for both grades.
    """
    order = np.argsort(-cam_scores)  # highest importance first
    n = len(order)
    n_steps = min(10, n)
    step_size = max(1, n // n_steps)

    aucs = []
    for k in range(0, n, step_size):
        keep_idx = order[k:]  # keep patches from k onwards
        if len(keep_idx) == 0:
            break
        kept_patches = patches[keep_idx]
        mean_feat = kept_patches.mean(axis=0).reshape(1, -1)
        # Use model prediction confidence
        x = torch.FloatTensor(mean_feat).to(DEVICE)
        with torch.no_grad():
            logits, _ = model_orig(x)
            prob_high = torch.softmax(logits, 1)[0, 1].item()
        aucs.append(prob_high)

    # Area under the prob_high deletion curve; no faithfulness interpretation.
    x_axis = np.linspace(0, 1, len(aucs))
    del_auc = float(np.trapezoid(aucs, x_axis)) if hasattr(np, 'trapezoid') else float(sum(aucs) / len(aucs))
    return del_auc, aucs

# Compute the descriptive curve area for only the first five cases per grade.
for grade_name, case_list in [("G1", g1_selected[:5]), ("G3", g3_selected[:5])]:
    for cid in case_list:
        cam = np.array(results[grade_name][cid]["cam_scores"])
        patches = patch_lookup[cid]
        del_auc, _ = compute_delete_auc(cid, cam, patches, model)
        results[grade_name][cid]["delete_auc"] = del_auc

g1_del_aucs = [results["G1"][c].get("delete_auc", 0) for c in g1_selected[:5] if "delete_auc" in results["G1"][c]]
g3_del_aucs = [results["G3"][c].get("delete_auc", 0) for c in g3_selected[:5] if "delete_auc" in results["G3"][c]]
print(f"  G1 Delete-AUC (5 cases): {np.mean(g1_del_aucs):.3f}+-{np.std(g1_del_aucs):.3f}" if g1_del_aucs else "  No G1 Delete-AUC")
print(f"  G3 Delete-AUC (5 cases): {np.mean(g3_del_aucs):.3f}+-{np.std(g3_del_aucs):.3f}" if g3_del_aucs else "  No G3 Delete-AUC")

# ====== Save all results ======
output = {
    "n_g1_cases": len(g1_selected),
    "n_g3_cases": len(g3_selected),
    "g1_kl_skew": {"mean": float(np.mean(g1_kl)), "std": float(np.std(g1_kl)), "values": g1_kl},
    "g3_kl_skew": {"mean": float(np.mean(g3_kl)), "std": float(np.std(g3_kl)), "values": g3_kl},
    "mann_whitney": {"statistic": float(stat), "p_value": float(p_mw)},
    "delete_auc": {
        "g1_mean": float(np.mean(g1_del_aucs)) if g1_del_aucs else None,
        "g3_mean": float(np.mean(g3_del_aucs)) if g3_del_aucs else None,
    },
    "per_case": results,
    "_interpretation": {
        "kl_skew": (
            f"G1 KL skew {np.mean(g1_kl):.3f}+-{np.std(g1_kl):.3f} vs "
            f"G3 KL skew {np.mean(g3_kl):.3f}+-{np.std(g3_kl):.3f}, "
            f"p={p_mw:.4f} ({'significant' if p_mw < 0.05 else 'not significant'})"
        ),
        "verdict": "single-seed input-gradient comparison; no cohort or mechanism inference",
    },
}
json.dump(output, open(OUTPUT_DIR / "oral_gradcam_results.json", "w"), indent=2)

# Save per-case CAM for heatmap generation
np.savez(OUTPUT_DIR / "oral_gradcam_scores.npz",
         g1_case_ids=np.array(g1_selected),
         g3_case_ids=np.array(g3_selected),
         g1_cam_scores=np.array([results["G1"][c]["cam_scores"] for c in g1_selected], dtype=object),
         g3_cam_scores=np.array([results["G3"][c]["cam_scores"] for c in g3_selected], dtype=object),
         g1_n_patches=np.array([results["G1"][c]["n_patches"] for c in g1_selected]),
         g3_n_patches=np.array([results["G3"][c]["n_patches"] for c in g3_selected]),
         )

print(f"\nSaved oral_gradcam_results.json + oral_gradcam_scores.npz")
print("DONE")
