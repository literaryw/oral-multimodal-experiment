"""
R16-fix: emit the LaTeX table bodies for the rewritten results section.

Prints ready-to-paste tabular rows so the manuscript numbers are transcribed by
machine rather than by hand.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

BASE = Path(__file__).resolve().parent.parent
D = BASE / "data" / "TCGA-HNSC"
OUT = D / "experiment_outputs"
aux = json.load(open(OUT / "attention_v2_aux.json"))
SEEDS = ["42", "123", "456"]

LABEL = {
    "image_only": "Image-Only (UNI)",
    "text_only": "Text-Only (BiomedBERT)",
    "late_fusion": "Late Fusion",
    "gated_bilinear": "Single-Token Attention",
    "true_cross_attention": "True Cross-Attention",
}
ORDER = ["image_only", "text_only", "late_fusion", "gated_bilinear", "true_cross_attention"]


def load(coh):
    return json.load(open(OUT / ("attention_v2_%s_results.json" % coh)))[coh]


def seed_stats(res, arm):
    """mean over folds per seed -> mean +- SD across the three seeds."""
    out = {}
    for m in ["accuracy", "macro_f1", "low_f1", "high_f1", "auc"]:
        v = [res[arm][s]["aggregate"][m]["mean"] for s in SEEDS if s in res[arm]]
        out[m] = (float(np.mean(v)), float(np.std(v, ddof=1)))
    return out


def best_flags(rows, keys):
    """mark the best mean per column (higher is better)."""
    best = {k: max(r[1][k][0] for r in rows) for k in keys}
    return best


print("% ================= TABLE 2: strict oral subset, 3-seed mean +- SD =================")
res = load("strict")
rows = [(a, seed_stats(res, a)) for a in ORDER]
best = best_flags(rows, ["accuracy", "macro_f1", "auc", "low_f1", "high_f1"])
for arm, s in rows:
    cells = []
    for k in ["accuracy", "macro_f1", "auc", "low_f1", "high_f1"]:
        m, sd = s[k]
        txt = "%.3f$\\pm$%.3f" % (m, sd)
        if abs(m - best[k]) < 1e-9:
            txt = "\\textbf{%s}" % txt
        cells.append(txt)
    print("%-22s & %s \\\\" % (LABEL[arm], " & ".join(cells)))

print("\n% ================= TABLE 3: cross-cohort, 3-seed mean =================")
cohorts = [("full", "Full HNSCC (n=499)"), ("wide", "Expanded oral-region (n=332)"), ("strict", "Strict oral cavity (n=246)")]
allres = {c: load(c) for c, _ in cohorts}
best3 = {}
for k in ["accuracy", "macro_f1", "high_f1", "auc"]:
    for c, _ in cohorts:
        best3[(c, k)] = max(seed_stats(allres[c], a)[k][0] for a in ORDER)
for arm in ORDER:
    cells = []
    for c, _ in cohorts:
        s = seed_stats(allres[c], arm)
        for k in ["accuracy", "macro_f1", "high_f1", "auc"]:
            m = s[k][0]
            txt = "%.3f" % m
            if abs(m - best3[(c, k)]) < 1e-9:
                txt = "\\textbf{%s}" % txt
            cells.append(txt)
    print("%-22s & %s \\\\" % (LABEL[arm], " & ".join(cells)))

print("\n% ================= exploratory coordinate-level pairwise MI =================")
mi = json.load(open(OUT / "attention_v2_mi_matched.json"))["cohorts"]
for c in ["full", "wide", "strict"]:
    m = mi[c]
    print("%% %-7s mean pairwise coordinate MI (K=10)=%.4f, SD=%.4f (exploratory only)" % (
        c, m["mean_pairwise_coordinate_mi"], m["sd_pairwise_coordinate_mi"]))

print("\n% ================= TABLE: text ablation, seed 42 =================")
for coh in ["full", "strict"]:
    print("%% --- %s cohort ---" % coh)
    for kind, name in [("full", "Full (diagnosis + TNM + demographics)"),
                       ("dx_tnm", "Diagnosis + TNM"),
                       ("tnm_only", "TNM only"),
                       ("dx_only", "Diagnosis only"),
                       ("demo_only", "Demographics only"),
                       ("no_tnm", "No-TNM (diagnosis + demographics)"),
                       ("tnm_demo", "No-diagnosis (TNM + demographics)")]:
        a = aux["text_ablation_" + coh][kind]["aggregate"]
        print("%-42s & %.3f$\\pm$%.3f & %.3f & %.3f & %.3f \\\\" % (
            name, a["accuracy"]["mean"], a["accuracy"]["std"],
            a["macro_f1"]["mean"], a["auc"]["mean"], a["high_f1"]["mean"]))

print("\n% ================= LOSO, seed 42 =================")
lm = json.load(open(OUT / "attention_v2_loso_moddrop.json"))
G = ["Tongue", "Floor of Mouth", "Buccal+Mouth", "Gingiva+Palate+Lip"]
for metric, key in [("Accuracy", "accuracy"), ("High-Grade F1", "high_f1"), ("AUC", "auc")]:
    print("%% --- %s ---" % metric)
    for arm in ORDER:
        pg = lm["loso"][arm]
        print("%-22s & %s \\\\" % (LABEL[arm], " & ".join("%.3f" % pg[g][key] for g in G)))

print("\n% ================= modality dropout, full cohort, seed 42 =================")
for k, v in lm["modality_dropout"].items():
    print("%-26s & %.3f & %.3f & %.3f & %.3f \\\\" % (
        k, v["accuracy"], v["macro_f1"], v["low_f1"], v["high_f1"]))

print("\n% ================= matched structured baseline, strict cohort =================")
structured = json.load(open(OUT / "attention_v2_structured_baseline.json"))
index = json.load(open(D / "attention_v2" / "cohort_index.json"))
strict_mask = np.asarray(index["strict_mask"], dtype=bool)
strict_y = np.asarray(index["y_binary"])[strict_mask]


def text_fold_average_precision():
    """Match the table estimand: per-fold AP, then mean across folds and seeds."""
    values = []
    text = load("strict")["text_only"]
    for seed in [42, 123, 456]:
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        prob = np.asarray(text[str(seed)]["oof_prob"])
        fold_ap = [average_precision_score(strict_y[val], prob[val])
                   for _, val in splitter.split(np.zeros(len(strict_y)), strict_y)]
        values.append(float(np.mean(fold_ap)))
    return float(np.mean(values))


structured_rows = []
for block, suffix in [("structured_all", "with diagnosis"),
                      ("structured_nodx", "no diagnosis")]:
    for model, label in [("logreg", "Logistic regression"),
                         ("gradboost", "Gradient boosting"),
                         ("mlp", "MLP")]:
        record = structured[block]["strict"][model]
        structured_rows.append((f"{label}, {suffix}", {
            "accuracy": record["accuracy"], "macro_f1": record["macro_f1"],
            "auc": record["auc"], "pr_auc": record["pr_auc"],
            "high_f1": record["high_f1"],
        }))
record = structured["dx_onehot"]["strict"]["logreg"]
structured_rows.append(("Logistic regression, diagnosis only", {
    "accuracy": record["accuracy"], "macro_f1": record["macro_f1"],
    "auc": record["auc"], "pr_auc": record["pr_auc"],
    "high_f1": record["high_f1"],
}))
text_stats = seed_stats(load("strict"), "text_only")
structured_rows.append(("Text-Only (BiomedBERT)", {
    "accuracy": text_stats["accuracy"][0], "macro_f1": text_stats["macro_f1"][0],
    "auc": text_stats["auc"][0], "pr_auc": text_fold_average_precision(),
    "high_f1": text_stats["high_f1"][0],
}))
keys = ["accuracy", "macro_f1", "auc", "pr_auc", "high_f1"]
best_structured = {key: max(row[key] for _, row in structured_rows) for key in keys}
for label, record in structured_rows:
    cells = []
    for key in keys:
        value = record[key]
        cell = f"{value:.3f}"
        if abs(value - best_structured[key]) < 5e-4:
            cell = "\\textbf{%s}" % cell
        cells.append(cell)
    print("%-38s & %s \\\\" % (label, " & ".join(cells)))

print("\n% ================= TABLE: reference dependence, strict subset =================")
rep = json.load(open(OUT / "attention_v2_rep_match.json"))["results"]["strict"]


def arm_mean(block, arm, key):
    return float(np.mean([block[arm][s]["aggregate"][key]["mean"] for s in SEEDS]))


def arm_rounded(block, arm, key):
    return round(arm_mean(block, arm, key), 3)


tca = {k: arm_rounded(load("strict"), "true_cross_attention", k)
       for k in ["macro_f1", "auc", "high_f1"]}
reference_rows = [
    ("Text-Only, [CLS] (main grid)", load("strict"), "text_only"),
    ("Text-Only, masked-mean tokens", rep, "text_only_tokmean"),
    ("Image-Only, stored features (main grid)", load("strict"), "image_only"),
    ("Image-Only, mean-pooled patches", rep, "image_only_patchmean"),
]
for label, block, arm in reference_rows:
    vals = [round(tca[k] - arm_rounded(block, arm, k), 3)
            for k in ["macro_f1", "auc", "high_f1"]]
    print("%-44s & %+.3f & %+.3f & %+.3f \\\\" % (label, *vals))

print("\n% ================= strict: per-seed True CA minus Text-Only =================")
st = load("strict")
for seed in SEEDS:
    diffs = [st["true_cross_attention"][seed]["aggregate"][k]["mean"]
             - st["text_only"][seed]["aggregate"][k]["mean"]
             for k in ["macro_f1", "auc", "high_f1"]]
    print("seed %s & %+.3f & %+.3f & %+.3f \\\\" % (seed, *diffs))

print("\n% ================= TABLE: no-diagnosis four cells, strict =================")
fc = json.load(open(OUT / "attention_v2_nodx_fourcell.json"))
for block, label in [("with_diagnosis", "With diagnosis"),
                     ("no_diagnosis", "No diagnosis")]:
    for arm, arm_label in [("text_only", "Text-Only"),
                           ("true_cross_attention", "True Cross-Attn.")]:
        cells = [fc["cells"][block][arm][s] for s in SEEDS]
        vals = {k: float(np.mean([c[k] for c in cells]))
                for k in ["accuracy", "macro_f1", "high_f1",
                          "average_precision", "auc"]}
        print("%s & %s & %.3f & %.3f & %.3f & %.3f & %.3f \\\\"
              % (label, arm_label, vals["accuracy"], vals["macro_f1"],
                 vals["high_f1"], vals["average_precision"], vals["auc"]))
print("% per-seed delta_D = D_nodx - D_full")
for s in SEEDS:
    d = fc["differences"][s]
    print("seed %s & %+.3f & %+.3f & %+.3f \\\\" % (s, d["D_full"], d["D_nodx"],
                                                   d["delta_D"]))

print("\n% ================= reproduction of tab:repmatch =================")
for label, block, arm in [
        ("Text-Only, [CLS]", st, "text_only"),
        ("Text-Only, masked-mean tokens", rep, "text_only_tokmean"),
        ("Image-Only, stored slide features", st, "image_only"),
        ("Image-Only, mean-pooled patch sequence", rep, "image_only_patchmean"),
        ("True Cross-Attn., masked-mean text", st, "true_cross_attention"),
        ("True Cross-Attn., [CLS] text", rep, "true_ca_cls")]:
    vals = [arm_mean(block, arm, k) for k in ["accuracy", "macro_f1", "auc", "high_f1"]]
    print("%-42s & %.3f & %.3f & %.3f & %.3f \\\\" % (label, *vals))

print("\n% ================= reproduction of tab:skip =================")
skip = json.load(open(OUT / "attention_v2_skip_ablation.json"))
for cohort, label in [("full", "Full (n=499)"), ("strict", "Strict (n=246)")]:
    for variant, variant_label in [("ca_full", "Both branches (full)"),
                                   ("ca_patch_only", "Attended patches only"),
                                   ("ca_text_only", "Text summary only")]:
        block = skip[cohort][variant]
        vals = [block[k]["mean"] for k in ["accuracy", "macro_f1", "auc", "high_f1"]]
        print("%-14s & %-24s & %.3f & %.3f & %.3f & %.3f \\\\"
              % (label, variant_label, *vals))
