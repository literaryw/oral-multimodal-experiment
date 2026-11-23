"""
修复 Grad-CAM: 使用真实 WSI patch 特征 + 优化模型
"""
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from pathlib import Path
import copy
import json

_PROJ = Path(__file__).resolve().parent.parent
# Full-cohort data (n=499): pooled features + labels live in integrated_full_legacy499/;
# patch-level features (499 cases) live in integrated/X_img_patches.npz.
DATA_DIR = _PROJ / "data/TCGA-HNSC/integrated_full_legacy499"
PATCH_FILE = _PROJ / "data/TCGA-HNSC/integrated/X_img_patches.npz"
OUT_DIR = _PROJ / "data/TCGA-HNSC/experiment_outputs"
FIG_DIR = _PROJ / "paper"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(RANDOM_SEED); np.random.seed(RANDOM_SEED)

# ============ Optimized Image Model ============
class OptimizedImageModel(nn.Module):
    """Deeper architecture for better G1/G3 discrimination"""
    def __init__(self, in_dim=1024, num_classes=2, dropout=0.3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 768), nn.ReLU(), nn.BatchNorm1d(768), nn.Dropout(dropout),
            nn.Linear(768, 512), nn.ReLU(), nn.BatchNorm1d(512), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(dropout),
        )
        self.head = nn.Linear(256, num_classes)
        self.activations = None
        self.gradients = None

    def forward(self, x):
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if i == len(self.encoder) - 1:  # last encoder layer
                self.activations = x
                if x.requires_grad:
                    x.register_hook(self._save_grad)
        return self.head(x)

    def _save_grad(self, grad):
        self.gradients = grad

# ============ Patch-level activation-gradient score ============
def compute_real_patch_cam(model, patches, target_class):
    """
    Compute a Grad-CAM-like activation-gradient score for stored WSI patches.
    patches: (N_patches, 1024) — actual UNI features per patch location
    Returns a normalized score per stored patch index. The index has no WSI
    coordinate mapping in this archive.
    """
    model.eval()
    model.zero_grad()

    x = patches.clone().detach().requires_grad_(True)
    out = model(x)
    activations = model.activations  # (N_patches, 256)

    # Backprop from target class
    score = out[:, target_class].mean()
    score.backward()

    grads = model.gradients
    if grads is None:
        grads = torch.zeros_like(activations)

    # Per-patch CAM weight
    weights = grads.mean(dim=0)  # (256,)
    cam = (activations * weights.unsqueeze(0)).sum(dim=1)  # (N_patches,)

    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam.detach().cpu().numpy()

def kl_to_uniform(cam):
    """KL(P||uniform) concentration of attention, P = normalized CAM.

    Same definition as gradcam_oral_expanded.py:179-185. Higher = more concentrated.
    """
    n = len(cam)
    eps = 1e-10
    p = cam / (cam.sum() + eps) + eps
    q = np.ones(n) / n
    return float(np.sum(p * np.log(p / q)))

# ============ Contrastive Auxiliary Loss ============
def contrastive_loss(embeddings, labels, temperature=0.5):
    """Pull same-class together, push different-class apart"""
    emb = F.normalize(embeddings, dim=1)
    sim = torch.mm(emb, emb.t()) / temperature
    mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    # Simple NT-Xent
    pos = (sim * mask).sum(1) / mask.sum(1).clamp(min=1)
    neg = (sim * (1 - mask)).sum(1) / (1 - mask).sum(1).clamp(min=1)
    return (-pos + torch.logsumexp(sim, dim=1)).mean()

# ============ Main ============
def generate_real_gradcam():
    print(f"Device: {DEVICE}")

    # Load mean-pooled features for training
    X = np.load(DATA_DIR / "X_img_uni.npy")
    y4 = np.load(DATA_DIR / "y.npy")
    y = np.where(np.isin(y4, [0, 1]), 0, 1)

    # Load raw patch features for CAM
    print("Loading raw patch features...")
    # allow_pickle 安全性说明：X_img_patches.npz 为本项目自产的 UNI patch 特征数值数组，
    # 非外部不可信 pickle 对象。
    patch_data = np.load(PATCH_FILE, allow_pickle=True)
    n_cases = len([k for k in patch_data.keys() if k.startswith('case_')])
    print(f"  {n_cases} cases with raw patches")

    # Train optimized model (5-fold, take fold 1)
    kf = StratifiedKFold(5, shuffle=True, random_state=RANDOM_SEED)
    tr, val = list(kf.split(X, y))[0]
    Xt, Xv = X[tr], X[val]; yt, yv = y[tr], y[val]

    # Weighted sampler + class weights (x5 for High-Grade)
    class_counts = np.bincount(yt)
    weights = 1.0 / class_counts[yt]
    sampler = WeightedRandomSampler(torch.DoubleTensor(weights), len(yt), replacement=True)
    tr_ds = TensorDataset(torch.FloatTensor(Xt), torch.LongTensor(yt))
    tr_loader = DataLoader(tr_ds, batch_size=32, sampler=sampler)

    model = OptimizedImageModel().to(DEVICE)
    ce_weight = torch.FloatTensor([1.0, 5.0]).to(DEVICE)  # High-Grade x5
    ce_criterion = nn.CrossEntropyLoss(weight=ce_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=20, T_mult=2)

    best_acc = 0; best_state = None
    for ep in range(120):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            # Forward through encoder only to get embeddings
            h = model.encoder(xb)
            logits = model.head(h)
            ce_loss = ce_criterion(logits, yb)
            cl_loss = 0.05 * contrastive_loss(h, yb)  # auxiliary contrastive
            loss = ce_loss + cl_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_out = model(torch.FloatTensor(Xv).to(DEVICE))
            acc = (val_out.argmax(1).cpu() == torch.LongTensor(yv)).float().mean()
        if acc > best_acc:
            best_acc = acc; best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    print(f"  Trained optimized model, val acc: {best_acc:.3f}")
    torch.save(model.state_dict(), OUT_DIR / "image_only_model.pt")

    # ===== Compute REAL patch-level CAM for G1 vs G3 =====
    g1_idx = np.where(y4 == 0)[0]
    g3_idx = np.where(y4 == 2)[0]
    print(f"  G1 samples: {len(g1_idx)}, G3 samples: {len(g3_idx)}")

    # Select representative samples (one per class for clean visualization)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for col_idx, (label, sample_idx) in enumerate([("G1", g1_idx[10]), ("G3", g3_idx[10])]):
        if col_idx == 2: break
        case_key = f"case_{sample_idx}"
        if case_key not in patch_data:
            print(f"  WARNING: {case_key} not in patch_data")
            continue

        patches = torch.FloatTensor(patch_data[case_key]).to(DEVICE)  # (N_p, 1024)
        n_p = patches.shape[0]
        tgt = 0 if label == "G1" else 1

        # Compute CAM per real patch
        cam = compute_real_patch_cam(model, patches, tgt)
        print(f"  {label}: {n_p} real patches, CAM range=[{cam.min():.3f}, {cam.max():.3f}], "
              f"top5 idx={np.argsort(cam)[-5:]}, KL(P||uniform)={kl_to_uniform(cam):.3f}, "
              f"frac(cam>0.5)={np.mean(cam > 0.5):.2f}")

        # Reshape to approximate 2D grid
        side = max(2, int(np.ceil(np.sqrt(n_p))))
        heatmap_2d = np.zeros((side, side))
        for i in range(n_p):
            heatmap_2d[i // side, i % side] = cam[i]

        ax = axes[col_idx]
        im = ax.imshow(heatmap_2d, cmap='jet', interpolation='bilinear')
        ax.set_title(f'{label} — Real Patch Grad-CAM ({n_p} patches)', fontsize=13, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, label='Importance')

    # Right panel: MI matrix from measured values (mutual_information_results.json,
    # computed with sklearn mutual_info_regression). Diagonal = trivial self-MI, masked.
    #
    # !! DO NOT RERUN fix_dndf_and_mi.py Part B EXPECTING THE SAME VALUES !!
    # MI estimation is non-deterministic. The paper binds 0.043 / 0.015 / 0.011 from the
    # original measurement stored in mutual_information_results.json. On 2026-09-06 that
    # file was accidentally overwritten by a rerun (0.0713/0.0194/0.0205; archived as
    # mutual_information_results_rerun_2026-09-06.json) and restored from git history.
    # fix_dndf_and_mi.py now writes date-stamped filenames so it can no longer clobber
    # the manuscript-bound json. Regenerating this figure with a re-measured json WOULD
    # silently diverge from the manuscript values.
    ax = axes[2]
    mi_json = json.load(open(OUT_DIR / "mutual_information_results.json"))
    mi_cross = mi_json["cross_mi"]            # image-text
    mi_img_label = mi_json["mi_img_label"]    # image-label
    mi_text_label = mi_json["mi_text_label"]  # text-label
    mi_matrix = np.array([
        [np.nan,       mi_cross,     mi_img_label],
        [mi_cross,     np.nan,       mi_text_label],
        [mi_img_label, mi_text_label, np.nan],
    ])
    labels_mi = ['Image\n(UNI)', 'Text\n(BERT)', 'Label\n(Grade)']
    im = ax.imshow(np.ma.masked_invalid(mi_matrix), cmap='YlOrRd', vmin=0, vmax=0.05)
    for i in range(3):
        for j in range(3):
            if not np.isnan(mi_matrix[i, j]):
                ax.text(j, i, f'{mi_matrix[i,j]:.3f}', ha='center', va='center', fontsize=12, fontweight='bold')
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(labels_mi); ax.set_yticklabels(labels_mi)
    ax.set_title('Mutual Information Matrix', fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax, fraction=0.046, label='MI (nats)')

    plt.suptitle('Grad-CAM: Real Patch-Level Activation from Optimized Model', fontsize=15, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0.02, 0, 1, 0.93])
    plt.savefig(OUT_DIR / 'fig4_gradcam_heatmap_legacy.pdf')
    plt.close()
    print("  Legacy Grad-CAM-plus-MI composite saved outside paper/; use fig4_regenerate.py for the manuscript figure")

    # Save CAM data
    g1_cam = compute_real_patch_cam(model, torch.FloatTensor(patch_data[f"case_{g1_idx[10]}"]).to(DEVICE), 0)
    g3_cam = compute_real_patch_cam(model, torch.FloatTensor(patch_data[f"case_{g3_idx[10]}"]).to(DEVICE), 1)
    np.save(OUT_DIR / "gradcam_g1_scores.npy", g1_cam)
    np.save(OUT_DIR / "gradcam_g3_scores.npy", g3_cam)
    print("  Grad-CAM data saved")

if __name__ == "__main__":
    generate_real_gradcam()
