"""
留一亚组交叉验证 (Leave-One-Site-Out / Leave-One-Anatomic-Group-Out)
13个site分4个解剖组 → 4-fold LOSO CV
"""
import sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")

import pandas as pd
import numpy as np
import json
from pathlib import Path

BASE_DIR = Path(".").resolve().parent
DATA_DIR = BASE_DIR / "data" / "TCGA-HNSC"
OUTPUT_DIR = DATA_DIR / "experiment_outputs"

# Load clinical data with site info
clin_df = pd.read_csv(DATA_DIR / "clinical_data_oral.csv")
case_labels = pd.read_csv(DATA_DIR / "integrated" / "case_labels.csv")

# Map sites to 4 anatomical groups
SITE_GROUPS = {
    # Group 0: Tongue
    "Tongue, NOS": 0, "Border of tongue": 0, "Ventral surface of tongue, NOS": 0,
    # Group 1: Floor of Mouth
    "Floor of mouth, NOS": 1, "Anterior floor of mouth": 1,
    # Group 2: Buccal + Mouth
    "Mouth, NOS": 2, "Cheek mucosa": 2,
    # Group 3: Gingiva + Palate + Lip + Retromolar
    "Gum, NOS": 3, "Lower gum": 3, "Upper Gum": 3,
    "Hard palate": 3, "Lip, NOS": 3, "Retromolar area": 3,
}
GROUP_NAMES = ["Tongue", "Floor of Mouth", "Buccal+Mouth", "Gingiva+Palate+Lip"]

clin_df["site_group"] = clin_df["primary_site"].map(SITE_GROUPS)

# Match to integrated cases
matched = clin_df[clin_df["case_id"].isin(case_labels["case_id"])].copy()
matched = matched.merge(case_labels, on="case_id", how="inner")
print(f"Matched cases: {len(matched)}")
for g in range(4):
    n_g = (matched["site_group"] == g).sum()
    n_high = (matched[matched["site_group"] == g]["label"] >= 2).sum()
    print(f"  Group {g} ({GROUP_NAMES[g]}): {n_g} cases, {n_high} High-Grade")

# Get case indices in integrated/ data
# We need to load the integrated data in the same order as case_labels
integrated_order = case_labels["case_id"].tolist()
matched_indices = [integrated_order.index(cid) for cid in matched["case_id"]]

# Load data
X_img = np.load(DATA_DIR / "integrated" / "X_img_uni.npy")
X_text = np.load(DATA_DIR / "integrated" / "X_text_biomedbert.npy")
y_4class = np.load(DATA_DIR / "integrated" / "y.npy")
y_binary = np.where(np.isin(y_4class, [0, 1]), 0, 1)

# Get site groups for each of the 246 cases
all_site_groups = np.full(len(y_binary), -1, dtype=int)
for idx, cid in zip(matched_indices, matched["case_id"]):
    group = matched[matched["case_id"] == cid]["site_group"].values[0]
    all_site_groups[idx] = group

# Verify mapping
for g in range(4):
    mask = all_site_groups == g
    print(f"  Group {g} in integrated: {mask.sum()} cases (binary High: {(y_binary[mask]==1).sum()})")

# ====== Import training utilities ======
with open("oral_experiments_full.py", encoding="utf-8") as f:
    src = f.read()
top_src = src.split("if __name__")[0]
exec(top_src)

# ====== LOSO CV ======
def run_loso(model_name, model_factory, single_mod, X_list, y, site_groups):
    """Leave-one-anatomic-group-out CV"""
    fold_results = {}
    for test_group in range(4):
        tr = site_groups != test_group
        val = site_groups == test_group
        if val.sum() == 0:
            continue

        if single_mod:
            Xt, yt = X_list[0][tr], y[tr]
            Xv, yv = X_list[0][val], y[val]
            class_counts = np.bincount(yt)
            if len(class_counts) < 2:
                continue
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
            if len(class_counts) < 2:
                continue
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
                if no_improve >= 15:
                    break

        model.load_state_dict(best_state)
        preds, labels, probs = evaluate(model, val_loader, DEVICE, single_mod)
        labels_arr = np.array(labels)
        preds_arr = np.array(preds)
        high_probs = np.array([p[1] for p in probs])

        acc = accuracy_score(labels_arr, preds_arr)
        auc_v = roc_auc_score(labels_arr, high_probs) if len(np.unique(labels_arr)) > 1 else 0.5
        high_f1 = f1_score(labels_arr, preds_arr, average=None)[1] if len(np.unique(labels_arr)) > 1 and f1_score(labels_arr, preds_arr, average=None).shape[0] > 1 else 0.0
        low_f1 = f1_score(labels_arr, preds_arr, average=None)[0] if len(np.unique(labels_arr)) > 1 else 0.0

        fold_results[GROUP_NAMES[test_group]] = {
            "n_test": int(val.sum()),
            "n_high_test": int(yv.sum()) if not single_mod else int(yv.sum()),
            "accuracy": float(acc),
            "high_f1": float(high_f1),
            "low_f1": float(low_f1),
            "auc": float(auc_v),
        }
        print(f"  Test={GROUP_NAMES[test_group]}: n={val.sum()} High={int(yv.sum()) if not single_mod else int(yv.sum())} "
              f"Acc={acc:.3f} HighF1={high_f1:.3f} AUC={auc_v:.3f}")

    return fold_results

# ====== Run 4 models ======
print("\n=== Leave-One-Anatomic-Group-Out CV ===")

print("\n--- Image-Only ---")
# Reset RNG per model to ensure execution-order independence (protocol-consistent)
torch.manual_seed(RANDOM_SEED); np.random.seed(RANDOM_SEED)
io_loso = run_loso("ImageOnly", lambda: ImageOnly(), True, [X_img], y_binary, all_site_groups)

print("\n--- Text-Only ---")
torch.manual_seed(RANDOM_SEED); np.random.seed(RANDOM_SEED)
to_loso = run_loso("TextOnly", lambda: TextOnly(), True, [X_text], y_binary, all_site_groups)

print("\n--- Cross-Attention ---")
torch.manual_seed(RANDOM_SEED); np.random.seed(RANDOM_SEED)
ca_loso = run_loso("CrossAttn", lambda: CrossAttentionFusion(), False, [X_img, X_text], y_binary, all_site_groups)

print("\n--- Late Fusion (added r7: M7) ---")
torch.manual_seed(RANDOM_SEED); np.random.seed(RANDOM_SEED)
lf_loso = run_loso("LateFusion", lambda: LateFusion(), False, [X_img, X_text], y_binary, all_site_groups)

# ====== Save ======
results = {
    "method": "Leave-One-Anatomic-Group-Out (4-fold)",
    "groups": {str(i): name for i, name in enumerate(GROUP_NAMES)},
    "group_sizes": {name: int((all_site_groups == i).sum()) for i, name in enumerate(GROUP_NAMES)},
    "Image-Only (UNI)": io_loso,
    "Text-Only (BiomedBERT)": to_loso,
    "Cross-Attention Fusion": ca_loso,
    "Late Fusion": lf_loso,
}
json.dump(results, open(OUTPUT_DIR / "oral_loso_results.json", "w"), indent=2)
print(f"\nSaved oral_loso_results.json")
print("DONE")
