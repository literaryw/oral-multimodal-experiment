"""
全队列 (n=499) 与宽口腔队列 (n=332) 的逐样本精确 McNemar + Bonferroni
- 数据 A (full): integrated_full_legacy499/ (真 499 例特征, r8 从 git 213aaa9 恢复)
- 数据 B (wide): A 的子集, case 属于 0812 宽口径临床表 clinical_data.csv (336 行)
  (嵌套断言: n=332, Low=260/High=72, 且 strict 246 全包含)
- 协议: 与主表一致 —— 每个模型的 CV 前重置 torch/np 种子 (seed=42)
输出: experiment_outputs/main_mcnemar_exact.json, wide_mcnemar_exact.json
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
import pandas as pd

BASE_DIR = Path(".").resolve().parent
DATA_DIR = BASE_DIR / "data" / "TCGA-HNSC"
OUTPUT_DIR = DATA_DIR / "experiment_outputs"

FULL_DIR = DATA_DIR / "integrated_full_legacy499"
STRICT_LABELS = DATA_DIR / "integrated" / "case_labels.csv"
WIDE_CLINICAL = DATA_DIR / "clinical_data.csv"  # 336-row wide-era table (0812 artifact)

X_img_full = np.load(FULL_DIR / "X_img_uni.npy")
X_text_full = np.load(FULL_DIR / "X_text_biomedbert.npy")
y4_full = np.load(FULL_DIR / "y.npy")
y_full = np.where(np.isin(y4_full, [0, 1]), 0, 1)
case_full = pd.read_csv(FULL_DIR / "case_labels.csv")

# ---- wide subset: 499 cases that belong to the 336-row wide-era clinical table ----
wide_clin = pd.read_csv(WIDE_CLINICAL).drop_duplicates(subset=["case_id"], keep="first")
wide_case_set = set(wide_clin["case_id"])
wide_mask = case_full["case_id"].isin(wide_case_set).values

X_img_wide, X_text_wide = X_img_full[wide_mask], X_text_full[wide_mask]
y_wide = y_full[wide_mask]

# ---- nested-cohort assertions ----
assert int(wide_mask.sum()) == 332, f"wide n != 332: {int(wide_mask.sum())}"
assert int((y_wide == 0).sum()) == 260 and int((y_wide == 1).sum()) == 72, "wide binary mismatch"
strict_ids = set(pd.read_csv(STRICT_LABELS)["case_id"])
wide_ids = set(case_full["case_id"][wide_mask])
assert strict_ids <= wide_ids, f"strict not nested in wide: missing {len(strict_ids - wide_ids)}"
print(f"Assertions passed: wide n=332 (Low 260/High 72), strict 246 nested in wide")

def run_and_save(model_factory, single_mod, X_list, y):
    """Run 5-fold CV with per-model seed reset; return per-sample correctness + probs."""
    torch.manual_seed(RANDOM_SEED); np.random.seed(RANDOM_SEED)
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    n = len(y)
    all_correct = np.zeros(n, dtype=bool)
    all_probs = np.zeros(n)
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
        preds_arr, labels_arr = np.array(preds), np.array(labels)
        all_correct[val] = (preds_arr == labels_arr)
        all_probs[val] = np.array([p[1] for p in probs])
        all_folds[val] = fold
        acc = all_correct[val].mean()
        auc_v = roc_auc_score(labels_arr, all_probs[val])
        print(f"  Fold {fold+1}: Acc={acc:.3f} AUC={auc_v:.3f}")
    return all_correct, all_probs, all_folds

from scipy.stats import chi2

def do_mcnemar(correct_a, correct_b):
    b = int((correct_b & ~correct_a).sum())
    c = int((correct_a & ~correct_b).sum())
    nd = b + c
    if nd > 0:
        stat = (abs(b - c) - 1)**2 / nd
        p = 1 - chi2.cdf(stat, 1)
    else:
        stat, p = 0.0, 1.0
    return b, c, float(stat), float(p)

def run_cohort(cohort_name, X_img, X_text, y):
    print(f"\n=== Cohort: {cohort_name} (n={len(y)}) ===")
    print("--- Cross-Attention Fusion ---")
    ca = run_and_save(lambda: CrossAttentionFusion(), False, [X_img, X_text], y)
    print("--- Text-Only ---")
    to = run_and_save(lambda: TextOnly(), True, [X_text], y)
    print("--- Image-Only ---")
    io = run_and_save(lambda: ImageOnly(), True, [X_img], y)
    pairs = [("CrossAttn vs TextOnly", ca[0], to[0]),
             ("CrossAttn vs ImageOnly", ca[0], io[0]),
             ("ImageOnly vs TextOnly", io[0], to[0])]
    mcnemar = {}
    for name, corr_a, corr_b in pairs:
        b, c, stat, p = do_mcnemar(corr_a, corr_b)
        mcnemar[name] = {"b": b, "c": c, "n_discordant": b + c,
                         "chi2": stat, "p_raw": p}
        print(f"  {name}: b={b} c={c} p={p:.4f}")
    n_comp = len(mcnemar)
    for key, val in mcnemar.items():
        p_corr = min(val["p_raw"] * n_comp, 1.0)
        val["p_bonferroni"] = float(p_corr)
        val["significant_05"] = bool(p_corr < 0.05)
        print(f"  {key}: p_bonf={p_corr:.4f} {'*' if p_corr < 0.05 else 'ns'}")
    return {
        "n_samples": int(len(y)),
        "crossattn": {"correct": ca[0].astype(int).tolist(),
                      "high_grade_prob": ca[1].tolist(), "fold": ca[2].tolist(),
                      "n_correct": int(ca[0].sum())},
        "textonly": {"correct": to[0].astype(int).tolist(),
                     "high_grade_prob": to[1].tolist(), "fold": to[2].tolist(),
                     "n_correct": int(to[0].sum())},
        "imageonly": {"correct": io[0].astype(int).tolist(),
                      "high_grade_prob": io[1].tolist(), "fold": io[2].tolist(),
                      "n_correct": int(io[0].sum())},
        "mcnemar": mcnemar,
        "_metadata": {"protocol": "per-model RNG reset (seed=42), 5-fold stratified",
                      "cohort": cohort_name},
    }

# ---- Run both cohorts ----
full_out = run_cohort("Full HNSCC (n=499)", X_img_full, X_text_full, y_full)
wide_out = run_cohort("Wide oral filter (n=332)", X_img_wide, X_text_wide, y_wide)

json.dump(full_out, open(OUTPUT_DIR / "main_mcnemar_exact.json", "w"), indent=2)
json.dump(wide_out, open(OUTPUT_DIR / "wide_mcnemar_exact.json", "w"), indent=2)
print(f"\nSaved main_mcnemar_exact.json / wide_mcnemar_exact.json")

# ---- Cross-validation vs main JSONs ----
main_ref = json.load(open(OUTPUT_DIR / "main_experiment_results.json"))
print("\n=== Consistency check vs main_experiment_results.json (full cohort) ===")
name_map = {"crossattn": "CrossAttention (UNI+BiomedBERT)",
            "textonly": "Text-Only (BiomedBERT)",
            "imageonly": "Image-Only (UNI)"}
for short, full_name in name_map.items():
    cm = main_ref["confusion_matrices"][full_name]
    ref_correct = cm[0][0] + cm[1][1]
    got = full_out[short]["n_correct"]
    flag = "SAME" if ref_correct == got else f"DIFF (ref {ref_correct})"
    print(f"  {short}: n_correct={got} [{flag}]")

wide_ref = json.load(open(OUTPUT_DIR / "wide_oral_main_results.json"))
print("=== Consistency check vs wide_oral_main_results.json (pooled vs fold-mean, loose) ===")
wide_name_map = {"crossattn": "Cross-Attention Fusion",
                 "textonly": "Text-Only (BiomedBERT)",
                 "imageonly": "Image-Only (UNI)"}
for short, full_name in wide_name_map.items():
    pooled = wide_out[short]["n_correct"] / 332.0
    ref_mean = wide_ref[full_name]["aggregate"]["accuracy"]["mean"]
    print(f"  {short}: pooled={pooled:.4f} fold-mean-ref={ref_mean:.4f} delta={pooled-ref_mean:+.4f}")
print("DONE")
