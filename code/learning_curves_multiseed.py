"""
学习曲线多随机种子重跑
5 seeds × 3 subset sizes × 2 models (Image+Text)
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

BASE_DIR = Path(".").resolve().parent
d = BASE_DIR / "data" / "TCGA-HNSC" / "integrated"
OUTPUT_DIR = BASE_DIR / "data" / "TCGA-HNSC" / "experiment_outputs"

X_img = np.load(d / "X_img_uni.npy")
X_text = np.load(d / "X_text_biomedbert.npy")
y_4class = np.load(d / "y.npy")
y_binary = np.where(np.isin(y_4class, [0, 1]), 0, 1)
print(f"Data: {len(y_binary)} cases, Low={int((y_binary==0).sum())}, High={int((y_binary==1).sum())}")

SEEDS = [42, 123, 456, 789, 1024]
SIZES = [100, 150, 200]

results = {}

for size in SIZES:
    results[str(size)] = {"n_samples": size, "image_only": {}, "text_only": {}}
    n_folds = max(3, min(5, size // 25))

    for seed in SEEDS:
        torch.manual_seed(seed); np.random.seed(seed)
        rng = np.random.RandomState(seed)

        # Stratified subset sampling
        idx_low = np.where(y_binary == 0)[0]
        idx_high = np.where(y_binary == 1)[0]
        n_low = int(size * len(idx_low) / len(y_binary))
        n_high = size - n_low
        subset_idx = np.concatenate([
            rng.choice(idx_low, min(n_low, len(idx_low)), replace=False),
            rng.choice(idx_high, min(n_high, len(idx_high)), replace=False)
        ])
        X_img_sub = X_img[subset_idx]
        X_text_sub = X_text[subset_idx]
        y_sub = y_binary[subset_idx]

        # Image-Only (no BatchNorm for small subsets)
        img_res = run_cv_with_stats(f"Img n={size} s={seed}",
            lambda: ImageOnlyNoSE(in_dim=1024, hidden=min(256, size), num_classes=2),
            True, [X_img_sub], y_sub, n_folds)
        # Text-Only
        txt_res = run_cv_with_stats(f"Txt n={size} s={seed}",
            lambda: ImageOnlyNoSE(in_dim=768, hidden=min(256, size), num_classes=2),
            True, [X_text_sub], y_sub, n_folds)

        results[str(size)]["image_only"][str(seed)] = {
            "accuracy": img_res["aggregate"]["accuracy"],
            "macro_f1": img_res["aggregate"]["macro_f1"],
            "auc": img_res["aggregate"]["auc"],
        }
        results[str(size)]["text_only"][str(seed)] = {
            "accuracy": txt_res["aggregate"]["accuracy"],
            "macro_f1": txt_res["aggregate"]["macro_f1"],
            "auc": txt_res["aggregate"]["auc"],
        }
        print(f"  size={size} seed={seed}: Img Acc={img_res['aggregate']['accuracy']['mean']:.3f} Txt Acc={txt_res['aggregate']['accuracy']['mean']:.3f}")

    # Compute mean±std across seeds for each metric
    for model in ["image_only", "text_only"]:
        for metric in ["accuracy", "macro_f1", "auc"]:
            vals = [results[str(size)][model][str(s)][metric]["mean"] for s in SEEDS]
            results[str(size)][model][f"{metric}_cross_seed"] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)),
                "per_seed": vals,
            }
        print(f"  size={size} {model}: Acc={np.mean(vals):.3f}+-{np.std(vals, ddof=1):.3f}")

json.dump(results, open(OUTPUT_DIR / "oral_learning_curves_multiseed.json", "w"), indent=2)
print(f"\nSaved oral_learning_curves_multiseed.json")
print("DONE")
