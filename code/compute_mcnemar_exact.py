"""
逐样本 McNemar + Bonferroni 校正
重跑 CrossAttn + Text-Only + Image-Only，保存每折逐样本预测
"""
import sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")

with open("oral_experiments_full.py", encoding="utf-8") as f:
    src = f.read()
top_src = src.split("if __name__")[0]
exec(top_src)

from pathlib import Path
import json
# McNemar computed manually via chi2; Bonferroni correction manual

BASE_DIR = Path(".").resolve().parent
d = BASE_DIR / "data" / "TCGA-HNSC" / "integrated"
OUTPUT_DIR = BASE_DIR / "data" / "TCGA-HNSC" / "experiment_outputs"

X_img = np.load(d / "X_img_uni.npy")
X_text = np.load(d / "X_text_biomedbert.npy")
y_4class = np.load(d / "y.npy")
y_binary = np.where(np.isin(y_4class, [0, 1]), 0, 1)
n = len(y_binary)
print(f"Data: {n} cases, Low={int((y_binary==0).sum())}, High={int((y_binary==1).sum())}")

def run_and_save(name, model_factory, single_mod, X_list, y):
    """Run 5-fold CV and return per-sample correctness + probabilities"""
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    all_correct = np.zeros(n, dtype=bool)   # per-sample correctness
    all_probs = np.zeros(n)                  # High-Grade probability
    all_folds = np.zeros(n, dtype=int)

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
        alpha = torch.FloatTensor([1.0, len(yt)/class_counts.max()]).to(DEVICE)
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
                if no_improve >= 15: break

        model.load_state_dict(best_state)
        preds, labels, probs = evaluate(model, val_loader, DEVICE, single_mod)
        labels_arr = np.array(labels)
        preds_arr = np.array(preds)
        correct = (preds_arr == labels_arr)
        high_probs = np.array([p[1] for p in probs])
        all_correct[val] = correct
        all_probs[val] = high_probs
        all_folds[val] = fold

        acc = correct.mean()
        auc_v = roc_auc_score(labels_arr, high_probs)
        print(f"  Fold {fold+1}: Acc={acc:.3f} AUC={auc_v:.3f}")

    return all_correct, all_probs, all_folds

# ====== Run 3 models ======
# Reset RNG per model to ensure execution-order independence (protocol-consistent)
print("\n=== Cross-Attention Fusion ===")
torch.manual_seed(RANDOM_SEED); np.random.seed(RANDOM_SEED)
ca_correct, ca_probs, ca_folds = run_and_save(
    "CrossAttn", lambda: CrossAttentionFusion(), False, [X_img, X_text], y_binary)

print("\n=== Text-Only ===")
torch.manual_seed(RANDOM_SEED); np.random.seed(RANDOM_SEED)
to_correct, to_probs, to_folds = run_and_save(
    "TextOnly", lambda: TextOnly(), True, [X_text], y_binary)

print("\n=== Image-Only ===")
torch.manual_seed(RANDOM_SEED); np.random.seed(RANDOM_SEED)
io_correct, io_probs, io_folds = run_and_save(
    "ImageOnly", lambda: ImageOnly(), True, [X_img], y_binary)

# ====== McNemar tests ======
from scipy.stats import chi2

def do_mcnemar(correct_a, correct_b, name_a, name_b):
    """Exact McNemar from per-sample correctness"""
    # b = B correct, A wrong; c = A correct, B wrong
    b = int((correct_b & ~correct_a).sum())  # B better
    c = int((correct_a & ~correct_b).sum())  # A better
    n_discordant = b + c
    if n_discordant > 0:
        stat = (abs(b - c) - 1)**2 / n_discordant
        p = 1 - chi2.cdf(stat, 1)
    else:
        stat, p = 0.0, 1.0
    return int(b), int(c), float(stat), float(p)

print("\n=== McNemar Tests ===")
pairs = [
    ("CrossAttn", ca_correct, "TextOnly", to_correct),
    ("CrossAttn", ca_correct, "ImageOnly", io_correct),
    ("ImageOnly", io_correct, "TextOnly", to_correct),
]

mcnemar_results = {}
for name_a, corr_a, name_b, corr_b in pairs:
    b, c, stat, p = do_mcnemar(corr_a, corr_b, name_a, name_b)
    mcnemar_results[f"{name_a} vs {name_b}"] = {
        "b": b, "c": c, "n_discordant": b + c,
        "chi2": stat, "p_raw": p,
    }
    print(f"  {name_a} vs {name_b}: b={b} c={c} chi2={stat:.3f} p={p:.4f}")

# Bonferroni correction (manual: multiply p by number of comparisons)
p_vals = [v["p_raw"] for v in mcnemar_results.values()]
n_comparisons = len(p_vals)
for key, val in mcnemar_results.items():
    p_corr = min(val["p_raw"] * n_comparisons, 1.0)
    val["p_bonferroni"] = float(p_corr)
    val["significant_05"] = bool(p_corr < 0.05)
    print(f"  {key}: p_bonf={p_corr:.4f} {'*' if p_corr < 0.05 else 'ns'}")

# Save per-sample predictions + McNemar
save_data = {
    "n_samples": int(n),
    "crossattn": {
        "correct": ca_correct.astype(int).tolist(),
        "high_grade_prob": ca_probs.tolist(),
        "fold": ca_folds.tolist(),
    },
    "textonly": {
        "correct": to_correct.astype(int).tolist(),
        "high_grade_prob": to_probs.tolist(),
        "fold": to_folds.tolist(),
    },
    "imageonly": {
        "correct": io_correct.astype(int).tolist(),
        "high_grade_prob": io_probs.tolist(),
        "fold": io_folds.tolist(),
    },
    "mcnemar": mcnemar_results,
}
json.dump(save_data, open(OUTPUT_DIR / "oral_mcnemar_exact.json", "w"), indent=2)
print(f"\nSaved oral_mcnemar_exact.json")
print("DONE")
