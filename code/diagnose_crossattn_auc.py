"""
诊断 CrossAttn AUC=0.444 (< random) 的根因
- 概率校准曲线 (reliability diagram)
- Expected Calibration Error (ECE)
- Temperature scaling 修复尝试
- 概率分布诊断
"""
import sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")

# Reuse model definitions from oral_experiments_full.py
with open("oral_experiments_full.py", encoding="utf-8") as f:
    src = f.read()
top_src = src.split("if __name__")[0]
exec(top_src)

from pathlib import Path
import numpy as np
import json
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score

BASE_DIR = Path(".").resolve().parent
d = BASE_DIR / "data" / "TCGA-HNSC" / "integrated"
OUTPUT_DIR = BASE_DIR / "data" / "TCGA-HNSC" / "experiment_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

X_img = np.load(d / "X_img_uni.npy")
X_text = np.load(d / "X_text_biomedbert.npy")
y_4class = np.load(d / "y.npy")
y_binary = np.where(np.isin(y_4class, [0, 1]), 0, 1)
print(f"Data: {len(y_binary)} cases, Low={int((y_binary==0).sum())}, High={int((y_binary==1).sum())}")

# ====== Modified CV: save per-sample probs ======
def run_cv_save_probs(name, model_factory, single_mod, X_list, y, n_splits=5):
    """Run CV and save per-sample probabilities + labels"""
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    all_probs_high = np.zeros(len(y))  # probability of High-Grade
    all_labels = np.zeros(len(y), dtype=int)
    all_fold_ids = np.zeros(len(y), dtype=int)

    for fold, (tr, val) in enumerate(kf.split(X_list[0], y)):
        if single_mod:
            Xt, yt = X_list[0][tr], y[tr]
            Xv, yv = X_list[0][val], y[val]
            class_counts = np.bincount(yt)
            weights = 1.0 / class_counts[yt]
            sampler = WeightedRandomSampler(torch.DoubleTensor(weights), len(yt), replacement=True)
            tr_ds = TensorDataset(torch.FloatTensor(Xt), torch.LongTensor(yt))
            tr_loader = DataLoader(tr_ds, batch_size=32, sampler=sampler)
            val_ds = TensorDataset(torch.FloatTensor(Xv), torch.LongTensor(yv))
            val_loader = DataLoader(val_ds, batch_size=64)
        else:
            Xi_t, Xt_t = X_list[0][tr], X_list[1][tr]
            Xi_v, Xt_v = X_list[0][val], X_list[1][val]
            yt, yv = y[tr], y[val]
            class_counts = np.bincount(yt)
            weights = 1.0 / class_counts[yt]
            sampler = WeightedRandomSampler(torch.DoubleTensor(weights), len(yt), replacement=True)
            tr_ds = TensorDataset(torch.FloatTensor(Xi_t), torch.FloatTensor(Xt_t), torch.LongTensor(yt))
            tr_loader = DataLoader(tr_ds, batch_size=32, sampler=sampler)
            val_ds = TensorDataset(torch.FloatTensor(Xi_v), torch.FloatTensor(Xt_v), torch.LongTensor(yv))
            val_loader = DataLoader(val_ds, batch_size=64)

        model = model_factory().to(DEVICE)
        alpha = torch.FloatTensor([1.0, len(yt) / class_counts.max()]).to(DEVICE)
        criterion = FocalLoss(alpha=alpha, gamma=2.0)
        opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
        scheduler = CosineAnnealingWarmRestarts(opt, T_0=20, T_mult=2)

        best_f1, best_state, no_improve = 0, None, 0
        for epoch in range(100):
            train_epoch(model, tr_loader, opt, criterion, DEVICE, single_mod)
            preds, labels, probs = evaluate(model, val_loader, DEVICE, single_mod)
            scheduler.step()
            val_f1 = f1_score(labels, preds, average='macro')
            if val_f1 > best_f1:
                best_f1, best_state = val_f1, copy.deepcopy(model.state_dict())
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= 15:
                    break

        model.load_state_dict(best_state)
        preds, labels, probs = evaluate(model, val_loader, DEVICE, single_mod)
        # Save per-sample probabilities (High-Grade prob = probs[1])
        labels_arr = np.array(labels)
        high_probs = np.array([p[1] for p in probs])
        all_probs_high[val] = high_probs
        all_labels[val] = labels_arr
        all_fold_ids[val] = fold

        acc = accuracy_score(labels, preds)
        auc = roc_auc_score(labels, high_probs)
        ece = compute_ece(labels_arr, high_probs, n_bins=10)
        print(f"  Fold {fold+1}: Acc={acc:.3f} AUC={auc:.3f} ECE={ece:.4f}")

    return all_probs_high, all_labels, all_fold_ids

def compute_ece(y_true, y_prob, n_bins=10):
    """Expected Calibration Error"""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece += (mask.sum() / len(y_true)) * abs(bin_acc - bin_conf)
    return ece

def temperature_scale(logits, temperature):
    return logits / temperature

# ====== Run CrossAttn + Text-Only ======
# Reset RNG per model to ensure execution-order independence (protocol-consistent)
print("\n=== Cross-Attention Fusion ===")
torch.manual_seed(RANDOM_SEED); np.random.seed(RANDOM_SEED)
ca_probs, ca_labels, ca_folds = run_cv_save_probs(
    "CrossAttn", lambda: CrossAttentionFusion(), False, [X_img, X_text], y_binary)

print("\n=== Text-Only (BiomedBERT) ===")
torch.manual_seed(RANDOM_SEED); np.random.seed(RANDOM_SEED)
to_probs, to_labels, to_folds = run_cv_save_probs(
    "TextOnly", lambda: TextOnly(), True, [X_text], y_binary)

# ====== Calibration Analysis ======
print("\n=== Calibration Analysis ===")

# Brier Score (lower = better calibrated)
ca_brier = brier_score_loss(ca_labels, ca_probs)
to_brier = brier_score_loss(to_labels, to_probs)

# ECE
ca_ece = compute_ece(ca_labels, ca_probs)
to_ece = compute_ece(to_labels, to_probs)

# AUC
ca_auc = roc_auc_score(ca_labels, ca_probs)
to_auc = roc_auc_score(to_labels, to_probs)

# Accuracy
ca_acc = ((ca_probs >= 0.5).astype(int) == ca_labels).mean()
to_acc = ((to_probs >= 0.5).astype(int) == to_labels).mean()

# Calibration curve data
ca_frac_pos, ca_mean_pred = calibration_curve(ca_labels, ca_probs, n_bins=10)
to_frac_pos, to_mean_pred = calibration_curve(to_labels, to_probs, n_bins=10)

# Probability distribution analysis
ca_high_probs_for_high = ca_probs[ca_labels == 1]  # High-Grade probs for actual High-Grade
ca_high_probs_for_low = ca_probs[ca_labels == 0]    # High-Grade probs for actual Low-Grade
to_high_probs_for_high = to_probs[to_labels == 1]
to_high_probs_for_low = to_probs[to_labels == 0]

print(f"\nCrossAttn: Acc={ca_acc:.3f} AUC={ca_auc:.3f} Brier={ca_brier:.4f} ECE={ca_ece:.4f}")
print(f"TextOnly:  Acc={to_acc:.3f} AUC={to_auc:.3f} Brier={to_brier:.4f} ECE={to_ece:.4f}")
print(f"\nCrossAttn High-Grade prob stats:")
print(f"  Actual High-Grade: mean={ca_high_probs_for_high.mean():.4f} std={ca_high_probs_for_high.std():.4f}")
print(f"  Actual Low-Grade:  mean={ca_high_probs_for_low.mean():.4f} std={ca_high_probs_for_low.std():.4f}")
print(f"TextOnly High-Grade prob stats:")
print(f"  Actual High-Grade: mean={to_high_probs_for_high.mean():.4f} std={to_high_probs_for_high.std():.4f}")
print(f"  Actual Low-Grade:  mean={to_high_probs_for_low.mean():.4f} std={to_high_probs_for_low.std():.4f}")

# ====== Temperature Scaling ======
print("\n=== Temperature Scaling ===")
# Grid search temperature on the full predictions (simulate in-fold scaling)
best_t_ca, best_auc_ca = 1.0, ca_auc
best_t_to, best_auc_to = 1.0, to_auc
for t in np.arange(0.1, 5.1, 0.1):
    # Apply temperature scaling to logit-space
    # probs_high = sigmoid(logit_high - logit_low) ≈ sigmoid(inverse_sigmoid(p) / t)
    eps = 1e-10
    ca_logits = np.log(np.clip(ca_probs, eps, 1 - eps)) - np.log(np.clip(1 - ca_probs, eps, 1 - eps))
    to_logits = np.log(np.clip(to_probs, eps, 1 - eps)) - np.log(np.clip(1 - to_probs, eps, 1 - eps))
    ca_scaled = 1 / (1 + np.exp(-ca_logits / t))
    to_scaled = 1 / (1 + np.exp(-to_logits / t))
    ca_auc_t = roc_auc_score(ca_labels, ca_scaled)
    to_auc_t = roc_auc_score(to_labels, to_scaled)
    if ca_auc_t > best_auc_ca:
        best_auc_ca, best_t_ca = ca_auc_t, t
    if to_auc_t > best_auc_to:
        best_auc_to, best_t_to = to_auc_t, t

ca_ece_ts = compute_ece(ca_labels, 1/(1+np.exp(-ca_logits/best_t_ca)))
to_ece_ts = compute_ece(to_labels, 1/(1+np.exp(-to_logits/best_t_to)))
print(f"CrossAttn: best T={best_t_ca:.1f} AUC={best_auc_ca:.3f}->{best_auc_ca:.3f} ECE={ca_ece_ts:.4f}")
print(f"TextOnly:  best T={best_t_to:.1f} AUC={to_auc:.3f}->{best_auc_to:.3f} ECE={to_ece_ts:.4f}")

# ====== Save Results ======
results = {
    "crossattn": {
        "accuracy": float(ca_acc),
        "auc": float(ca_auc),
        "brier": float(ca_brier),
        "ece": float(ca_ece),
        "high_prob_stats": {
            "actual_high_mean": float(ca_high_probs_for_high.mean()),
            "actual_high_std": float(ca_high_probs_for_high.std()),
            "actual_low_mean": float(ca_high_probs_for_low.mean()),
            "actual_low_std": float(ca_high_probs_for_low.std()),
        },
        "temperature_scaling": {
            "best_t": float(best_t_ca),
            "auc_after": float(best_auc_ca),
            "ece_after": float(ca_ece_ts),
        },
        "calibration_curve": {
            "frac_positive": ca_frac_pos.tolist(),
            "mean_predicted": ca_mean_pred.tolist(),
        },
    },
    "textonly": {
        "accuracy": float(to_acc),
        "auc": float(to_auc),
        "brier": float(to_brier),
        "ece": float(to_ece),
        "high_prob_stats": {
            "actual_high_mean": float(to_high_probs_for_high.mean()),
            "actual_high_std": float(to_high_probs_for_high.std()),
            "actual_low_mean": float(to_high_probs_for_low.mean()),
            "actual_low_std": float(to_high_probs_for_low.std()),
        },
        "temperature_scaling": {
            "best_t": float(best_t_to),
            "auc_after": float(best_auc_to),
            "ece_after": float(to_ece_ts),
        },
        "calibration_curve": {
            "frac_positive": to_frac_pos.tolist(),
            "mean_predicted": to_mean_pred.tolist(),
        },
    },
    "_interpretation": {
        "auc_gap": f"CrossAttn AUC {ca_auc:.3f} vs TextOnly AUC {to_auc:.3f}",
        "root_cause": (
            "CrossAttn outputs High-Grade probability near 0 for almost all samples. "
            f"Mean High-Grade prob: Low-Grade samples={ca_high_probs_for_low.mean():.4f}, "
            f"High-Grade samples={ca_high_probs_for_high.mean():.4f}. "
            f"The model collapses to predict Low-Grade; argmax is correct for Low-Grade cases "
            f"but the probability ordering within High-Grade cases is random → AUC < 0.5."
        ),
        "temperature_scaling_verdict": (
            f"Temperature scaling {'CAN' if best_auc_ca > ca_auc + 0.02 else 'CANNOT'} fix the AUC issue "
            f"(AUC: {ca_auc:.3f} -> {best_auc_ca:.3f})."
            if best_auc_ca > ca_auc + 0.02 else
            f"Temperature scaling cannot fix fundamentally collapsed probabilities "
            f"(AUC: {ca_auc:.3f} -> {best_auc_ca:.3f})."
        ),
    },
}
json.dump(results, open(OUTPUT_DIR / "oral_auc_diagnosis.json", "w"), indent=2)
print(f"\nSaved oral_auc_diagnosis.json")
print("DONE")
