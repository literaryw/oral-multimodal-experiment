"""Four-cell comparison: does the fusion-minus-text-only difference change without diagnosis?

Cells on the strict oral subset (n=246):
    with diagnosis  x {Text-Only, True Cross-Attention}   (reused from the primary grid)
    no diagnosis    x {Text-Only, True Cross-Attention}   (this script)

The no-diagnosis arm is re-encoded from the rendered TNM + demographics template
(text_tokens_nodx.npz), so the encoder never sees the primary-diagnosis field. Folds,
inner selection, training budget, seeds and thresholds are unchanged.

Estimand per seed: D = Macro F1(True CA) - Macro F1(Text-Only), fold mean first, then the
difference between arms. delta_D = D_nodx - D_full. Descriptive only; no hypothesis test.
Every difference is computed from unrounded cell values and rounded only when printed.

The three-seed means and the per-seed differences are emitted as LaTeX fragments so the
manuscript never transcribes them by hand.

Outputs:
    data/TCGA-HNSC/experiment_outputs/attention_v2_nodx_fourcell.json   (cells + per-seed OOF)
    data/TCGA-HNSC/experiment_outputs/attention_v2_nodx_fourcell.tex    (four-cell table body)
    data/TCGA-HNSC/experiment_outputs/attention_v2_nodx_seed.tex        (per-seed Macro F1 differences)
    data/TCGA-HNSC/experiment_outputs/attention_v2_nodx_metric.tex      (per-metric differences)
"""
import datetime
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from attention_v2_experiments import build_cohort, run_cv  # noqa: E402

BASE = Path(__file__).resolve().parent.parent
D = BASE / "data" / "TCGA-HNSC"
BUNDLE = D / "attention_v2"
OUT = D / "experiment_outputs"
SEEDS = [42, 123, 456]
ARMS = ("text_only", "true_cross_attention")
CELL_METRICS = ("accuracy", "macro_f1", "low_f1", "high_f1", "auc", "average_precision")
DIFF_METRICS = ("macro_f1", "high_f1", "average_precision", "auc")
TOL = 1e-12


def fx(v, nd=3):
    """Signed fixed-point with no '-0.000' artefacts."""
    s = "%+.*f" % (nd, v)
    return s if float(s) != 0.0 else "%.*f" % (nd, 0.0)


def folds_for(y, seed):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    return [va for _, va in skf.split(np.zeros(len(y)), y)]


def ap_per_fold(y, prob, seed):
    return float(np.mean([average_precision_score(y[va], prob[va])
                          for va in folds_for(y, seed)]))


def cell_summary(res, y, seed):
    a = res["aggregate"]
    return {
        "accuracy": a["accuracy"]["mean"],
        "macro_f1": a["macro_f1"]["mean"],
        "low_f1": a["low_f1"]["mean"],
        "high_f1": a["high_f1"]["mean"],
        "auc": a["auc"]["mean"],
        "average_precision": ap_per_fold(y, np.asarray(res["oof_prob"]), seed),
        "oof_pred": [int(v) for v in res["oof_pred"]],
        "oof_prob": [float(v) for v in res["oof_prob"]],
    }


def cell_metrics(cell):
    return {k: cell[k] for k in CELL_METRICS}


def check_against_archive(report, path):
    """Refuse to overwrite an archive whose cell values this run does not reproduce."""
    if not path.exists():
        return None
    old = json.load(open(path))
    worst, bad = 0.0, []
    for block in ("with_diagnosis", "no_diagnosis"):
        for arm in ARMS:
            for s in SEEDS:
                new_c, old_c = report["cells"][block][arm][str(s)], old["cells"][block][arm][str(s)]
                for m in CELL_METRICS:
                    delta = abs(new_c[m] - old_c[m])
                    worst = max(worst, delta)
                    if delta > TOL:
                        bad.append("%s/%s/%d/%s: archive %.12f vs rerun %.12f"
                                   % (block, arm, s, m, old_c[m], new_c[m]))
    if bad:
        print("\nREFUSING TO OVERWRITE: rerun does not reproduce the archived cells")
        for line in bad:
            print("  " + line)
        raise SystemExit(1)
    print("archive check: all %d cells reproduced (max |delta| = %.3e)"
          % (4 * 3 * len(CELL_METRICS), worst))
    return old.get("date")


def per_metric_differences(report):
    out = {}
    for m in DIFF_METRICS:
        d_full, d_nodx, d_delta = [], [], []
        for s in SEEDS:
            full = report["cells"]["with_diagnosis"]
            nodx = report["cells"]["no_diagnosis"]
            a = full["true_cross_attention"][str(s)][m] - full["text_only"][str(s)][m]
            b = nodx["true_cross_attention"][str(s)][m] - nodx["text_only"][str(s)][m]
            d_full.append(a)
            d_nodx.append(b)
            d_delta.append(b - a)
        out[m] = {
            "D_full_values": d_full,
            "D_full_mean": float(np.mean(d_full)),
            "D_nodx_values": d_nodx,
            "D_nodx_mean": float(np.mean(d_nodx)),
            "delta_D_values": d_delta,
            "delta_D_mean": float(np.mean(d_delta)),
            "delta_D_sd": float(np.std(d_delta, ddof=1)),
            "delta_D_positive": int(sum(x > 0 for x in d_delta)),
        }
    return out


def write_tex(report):
    lines = ["% Four-cell no-diagnosis comparison (attention_v2_nodx_fourcell.py).",
             "\\begin{tabular}{llccccc}", "\\toprule",
             "Input & Model & Acc & Macro F1 & High F1 & AP & AUC \\\\", "\\midrule"]
    for block, label in [("with_diagnosis", "With diagnosis"),
                         ("no_diagnosis", "No diagnosis")]:
        for arm, arm_label in [("text_only", "Text-Only"),
                               ("true_cross_attention", "True Cross-Attn.")]:
            cells = [report["cells"][block][arm][str(s)] for s in SEEDS]
            vals = {k: float(np.mean([c[k] for c in cells]))
                    for k in ("accuracy", "macro_f1", "high_f1", "average_precision", "auc")}
            lines.append("%s & %s & %.3f & %.3f & %.3f & %.3f & %.3f \\\\"
                         % (label, arm_label, vals["accuracy"], vals["macro_f1"],
                            vals["high_f1"], vals["average_precision"], vals["auc"]))
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}")
    lines.append("")
    (OUT / "attention_v2_nodx_fourcell.tex").write_text("\n".join(lines), encoding="utf-8")

    pm = report["per_metric_differences"]["macro_f1"]
    seed_lines = ["% Per-seed fusion-minus-text Macro F1 difference (attention_v2_nodx_fourcell.py).",
                  "\\begin{tabular}{lccc}", "\\toprule",
                  "Seed & $D_{\\text{with dx}}$ & $D_{\\text{no dx}}$ & $\\Delta D$ \\\\",
                  "\\midrule"]
    for i, s in enumerate(SEEDS):
        seed_lines.append("%d & %s & %s & %s \\\\"
                          % (s, fx(pm["D_full_values"][i]), fx(pm["D_nodx_values"][i]),
                             fx(pm["delta_D_values"][i])))
    seed_lines += ["\\midrule",
                   "Mean (SD) & %s & %s & %s (%.3f) \\\\"
                   % (fx(pm["D_full_mean"]), fx(pm["D_nodx_mean"]),
                      fx(pm["delta_D_mean"]), pm["delta_D_sd"]),
                   "\\bottomrule", "\\end{tabular}", ""]
    (OUT / "attention_v2_nodx_seed.tex").write_text("\n".join(seed_lines), encoding="utf-8")

    labels = {"macro_f1": "Macro F1", "high_f1": "High-Grade F1",
              "average_precision": "Average precision", "auc": "ROC-AUC"}
    metric_lines = ["% Per-metric fusion-minus-text differences (attention_v2_nodx_fourcell.py).",
                    "\\begin{tabular}{lcccc}", "\\toprule",
                    "Metric & $D_{\\text{with dx}}$ & $D_{\\text{no dx}}$ & $\\Delta D$ (SD) & $\\Delta D>0$ \\\\",
                    "\\midrule"]
    for m in DIFF_METRICS:
        v = report["per_metric_differences"][m]
        metric_lines.append("%s & %s & %s & %s (%.3f) & %d/3 \\\\"
                            % (labels[m], fx(v["D_full_mean"]), fx(v["D_nodx_mean"]),
                               fx(v["delta_D_mean"]), v["delta_D_sd"], v["delta_D_positive"]))
    metric_lines += ["\\bottomrule", "\\end{tabular}", ""]
    (OUT / "attention_v2_nodx_metric.tex").write_text("\n".join(metric_lines), encoding="utf-8")


def main():
    index = json.load(open(BUNDLE / "cohort_index.json"))
    y_all = np.asarray(index["y_binary"])
    mask = np.asarray(index["strict_mask"], dtype=bool)
    rows = np.where(mask)[0]
    y = y_all[rows].astype(int)

    X_img_all = np.load(D / "integrated_full_legacy499" / "X_img_uni.npy")
    tokz = np.load(BUNDLE / "text_tokens.npz")
    pat = np.load(BUNDLE / "img_patches.npz")
    patch_list = [pat["case_%d" % i] for i in range(index["n_base"])]
    base = build_cohort(index, mask, X_img_all, tokz["cls"], patch_list)
    assert len(base["y"]) == 246 and np.array_equal(base["y"], y)

    nd = np.load(BUNDLE / "text_tokens_nodx.npz")
    c = dict(base)
    c["tokens"] = nd["tokens"][rows].astype(np.float32)
    c["text_mask"] = nd["mask"][rows].astype(np.int8)
    c["X_text"] = nd["cls"][rows].astype(np.float32)
    n, lmax, dim = c["tokens"].shape
    assert n == 246 and dim == 768 and lmax == c["tokens"].shape[1]
    assert c["text_mask"].min() >= 0 and c["text_mask"].max() <= 1
    assert (c["text_mask"].sum(1) > 0).all() and c["X_text"].shape == (246, 768)

    # --- seed-42 parity gate against the previously archived no-diagnosis text-only run ---
    aux = json.load(open(OUT / "attention_v2_aux.json"))["text_ablation_strict"]["tnm_demo"]
    probe = run_cv("text_only", c, 42)
    pred = np.asarray(probe["oof_pred"])
    prob = np.asarray(probe["oof_prob"])
    parity_pred = float((pred == np.asarray(aux["oof_pred"])).mean())
    parity_delta = float(np.abs(prob - np.asarray(aux["oof_prob"])).max())
    print("seed-42 parity: pred agreement=%.4f  max|dprob|=%.3e" % (parity_pred, parity_delta))
    if parity_pred != 1.0 or parity_delta > 1e-6:
        raise SystemExit("seed-42 no-diagnosis text-only does not reproduce the archive; "
                         "check the encoded text and batching before proceeding")

    main_res = json.load(open(OUT / "attention_v2_strict_results.json"))["strict"]
    report = {"date": "2026-11-22", "cohort": "strict", "n": 246,
              "prevalence": float(y.mean()), "seeds": SEEDS,
              "seed42_parity": {"pred_agreement": parity_pred,
                                "max_abs_prob_delta": parity_delta},
              "cells": {"with_diagnosis": {}, "no_diagnosis": {}}, "differences": {}}

    for arm in ARMS:
        report["cells"]["with_diagnosis"][arm] = {
            str(s): cell_summary(main_res[arm][str(s)], y, s) for s in SEEDS}
    for arm in ARMS:
        report["cells"]["no_diagnosis"][arm] = {}
        for s in SEEDS:
            print("running no-diagnosis %s, seed %d ..." % (arm, s))
            report["cells"]["no_diagnosis"][arm][str(s)] = cell_summary(run_cv(arm, c, s), y, s)

    deltas = []
    for s in SEEDS:
        d_full = (report["cells"]["with_diagnosis"]["true_cross_attention"][str(s)]["macro_f1"]
                  - report["cells"]["with_diagnosis"]["text_only"][str(s)]["macro_f1"])
        d_nodx = (report["cells"]["no_diagnosis"]["true_cross_attention"][str(s)]["macro_f1"]
                  - report["cells"]["no_diagnosis"]["text_only"][str(s)]["macro_f1"])
        deltas.append(float(d_nodx - d_full))
        report["differences"][str(s)] = {"D_full": d_full, "D_nodx": d_nodx,
                                         "delta_D": float(d_nodx - d_full)}
    report["differences"]["summary"] = {
        "mean": float(np.mean(deltas)), "sd": float(np.std(deltas, ddof=1)),
        "sign_consistency": "%d/3 positive" % sum(d > 0 for d in deltas),
        "values": deltas,
    }
    report["per_metric_differences"] = per_metric_differences(report)

    json_path = OUT / "attention_v2_nodx_fourcell.json"
    first_run_date = check_against_archive(report, json_path)
    if first_run_date:
        report["first_run_date"] = first_run_date
        report["rerun_date"] = datetime.date.today().isoformat()

    json.dump(report, open(json_path, "w"), indent=2)
    write_tex(report)

    print(json.dumps({"macro_f1": report["differences"]["summary"],
                      "per_metric": {m: {"D_full": round(v["D_full_mean"], 5),
                                         "D_nodx": round(v["D_nodx_mean"], 5),
                                         "delta_D": round(v["delta_D_mean"], 5),
                                         "positive": "%d/3" % v["delta_D_positive"]}
                                     for m, v in report["per_metric_differences"].items()}},
                     indent=2))


if __name__ == "__main__":
    main()
