"""
主表多种子稳健性重跑（G3，review-std v1.3 5.4）
- 队列: strict oral (n=246, canonical integrated/)
- 设计: 对每个 seed in {42,123,456}，完整重跑 4 模型 5-fold 协议
  （折划分与模型初始化均由该 seed 决定——与主表协议一致，只是换种子）；
  每个模型 CV 前重置 RNG 到当前 seed（保持 per-model reset 语义）
- 验证锚点: seed=42 应逐位复现主表 oral_main_results.json
输出: experiment_outputs/oral_main_results_multiseed.json
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
OUT = BASE_DIR / "data" / "TCGA-HNSC" / "experiment_outputs"

X_img = np.load(d / "X_img_uni.npy")
X_text = np.load(d / "X_text_biomedbert.npy")
y_4class = np.load(d / "y.npy")
y_binary = np.where(np.isin(y_4class, [0, 1]), 0, 1)
print(f"Data: n={len(y_binary)}, Low={int((y_binary==0).sum())}, High={int((y_binary==1).sum())}")

SEEDS = [42, 123, 456]
EXPERIMENTS = [
    ("Image-Only (UNI)", lambda: ImageOnly(), True, [X_img]),
    ("Text-Only (BiomedBERT)", lambda: TextOnly(), True, [X_text]),
    ("Late Fusion", lambda: LateFusion(), False, [X_img, X_text]),
    ("Cross-Attention Fusion", lambda: CrossAttentionFusion(), False, [X_img, X_text]),
]

results = {}
for seed in SEEDS:
    print(f"\n===== SEED {seed} =====")
    globals()["RANDOM_SEED"] = seed  # StratifiedKFold inside run_cv_with_stats reads this
    results[str(seed)] = {}
    for name, factory, single, X_list in EXPERIMENTS:
        torch.manual_seed(seed); np.random.seed(seed)
        print(f"\n--- {name} (seed {seed}) ---")
        r = run_cv_with_stats(name, factory, single, X_list, y_binary, n_splits=5)
        results[str(seed)][name] = {"aggregate": r["aggregate"], "per_fold": r["per_fold"]}

# ---- cross-seed aggregation ----
cross = {}
for name, *_ in EXPERIMENTS:
    cross[name] = {}
    for metric in ["accuracy", "macro_f1", "auc", "low_f1", "high_f1"]:
        vals = [results[str(s)][name]["aggregate"][metric]["mean"] for s in SEEDS]
        cross[name][metric] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)),
            "per_seed": [float(v) for v in vals],
        }

out = {
    "_metadata": {
        "protocol": ("full protocol rerun per seed (fold split + initialization both derive "
                      "from the seed); per-model RNG reset preserved within each seed"),
        "seeds": SEEDS,
        "n_samples": int(len(y_binary)),
        "cohort": "strict oral cavity subset (n=246)",
    },
    "per_seed": results,
    "cross_seed": cross,
}
json.dump(out, open(OUT / "oral_main_results_multiseed.json", "w"), indent=2)
print("\nSaved oral_main_results_multiseed.json")

# ---- validation: seed=42 reproduces the canonical main table ----
print("\n=== Validation: seed=42 vs canonical oral_main_results.json ===")
canon = json.load(open(OUT / "oral_main_results.json"))
all_same = True
for name, *_ in EXPERIMENTS:
    for metric in ["accuracy", "macro_f1", "auc", "low_f1", "high_f1"]:
        a = results["42"][name]["aggregate"][metric]["mean"]
        b = canon[name]["aggregate"][metric]["mean"]
        same = abs(a - b) < 1e-9
        all_same &= same
        if not same:
            print(f"  MISMATCH {name} {metric}: multiseed={a:.6f} canonical={b:.6f}")
print(f"  seed=42 bit-identical to canonical: {all_same}")

print("\n=== Cross-seed summary (mean±SD over 3 seeds) ===")
for name in cross:
    c = cross[name]
    print(f"{name}: acc {c['accuracy']['mean']:.4f}±{c['accuracy']['std']:.4f} | "
          f"highF1 {c['high_f1']['mean']:.4f}±{c['high_f1']['std']:.4f} | "
          f"auc {c['auc']['mean']:.4f}±{c['auc']['std']:.4f}")
print("DONE")
