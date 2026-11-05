"""
R16-fix: emit the LaTeX table bodies for the rewritten results section.

Prints ready-to-paste tabular rows so the manuscript numbers are transcribed by
machine rather than by hand.
"""
import json
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
D = BASE / "data" / "TCGA-HNSC"
OUT = D / "experiment_outputs"
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
cohorts = [("full", "Full HNSCC (n=499)"), ("wide", "Wide Oral (n=332)"), ("strict", "Strict Oral (n=246)")]
allres = {c: load(c) for c, _ in cohorts}
best3 = {}
for k in ["accuracy", "high_f1", "auc"]:
    for c, _ in cohorts:
        best3[(c, k)] = max(seed_stats(allres[c], a)[k][0] for a in ORDER)
for arm in ORDER:
    cells = []
    for c, _ in cohorts:
        s = seed_stats(allres[c], arm)
        for k in ["accuracy", "high_f1", "auc"]:
            m = s[k][0]
            txt = "%.3f" % m
            if abs(m - best3[(c, k)]) < 1e-9:
                txt = "\\textbf{%s}" % txt
            cells.append(txt)
    print("%-22s & %s \\\\" % (LABEL[arm], " & ".join(cells)))

print("\n% ================= MI per cohort =================")
aux = json.load(open(OUT / "attention_v2_aux.json"))
for c in ["full", "wide", "strict"]:
    m = aux["mutual_information"][c]
    print("%% %-7s cross-MI=%.4f  img-label=%.4f  text-label=%.4f  ratio=%.2fx" % (
        c, m["cross_mi"], m["img_label_mi"], m["text_label_mi"], m["ratio"]))

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
