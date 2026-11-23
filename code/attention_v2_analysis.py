"""
R16-fix Phase D: aggregate the re-run, compute pooled + per-fold metrics, and run
exact-binomial McNemar calculations on paired OOF correctness with Bonferroni
correction. Exactness refers to the discordant-count calculation and does not
extend to inference over overlapping cross-validation training sets.

Reads  experiment_outputs/attention_v2_{full,wide,strict}_results.json
Writes experiment_outputs/attention_v2_analysis.json  (+ console report)
"""
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score

BASE = Path(__file__).resolve().parent.parent
D = BASE / "data" / "TCGA-HNSC"
OUT = D / "experiment_outputs"

COHORTS = ["full", "wide", "strict"]
ARM_LABEL = {
    "image_only": "Image-Only (UNI)",
    "text_only": "Text-Only (BiomedBERT)",
    "late_fusion": "Late Fusion",
    "gated_bilinear": "Single-Token Attention (ex-'cross-attn')",
    "true_cross_attention": "True Cross-Attention",
}
SEEDS = ["42", "123", "456"]


def pooled(pred, y, prob):
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "low_f1": float(f1_score(y, pred, average=None, labels=[0, 1])[0]),
        "high_f1": float(f1_score(y, pred, average=None, labels=[0, 1])[1]),
        "auc": float(roc_auc_score(y, prob)),
        "pr_auc": float(average_precision_score(y, prob)),
    }


def mcnemar_exact(a_pred, b_pred, y):
    """Two-sided exact-binomial McNemar calculation on paired OOF correctness."""
    a_ok, b_ok = (a_pred == y), (b_pred == y)
    b = int(np.sum(a_ok & ~b_ok))
    c = int(np.sum(~a_ok & b_ok))
    n = b + c
    p = 1.0 if n == 0 else float(binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue)
    return b, c, p


def bootstrap_ci(correct_a, correct_b, n_boot=4000, seed=0):
    """95% percentile bootstrap CI for mean(correct_a) - mean(correct_b), paired."""
    d = (correct_a.astype(np.float64) - correct_b.astype(np.float64))
    rng = np.random.default_rng(seed)
    n = len(d)
    draws = rng.integers(0, n, size=(n_boot, n))
    stats = d[draws].mean(axis=1)
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def majority_baseline(y):
    """Always predict the majority class."""
    maj = int(np.bincount(y).argmax())
    pred = np.full(len(y), maj, dtype=int)
    return {
        "class": maj,
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "high_f1": 0.0,
        "auc": 0.5,
    }


def main():
    index = json.load(open(D / "attention_v2" / "cohort_index.json"))
    y_all = np.array(index["y_binary"])
    masks = {"full": np.ones(len(y_all), bool),
             "wide": np.array(index["wide_mask"], bool),
             "strict": np.array(index["strict_mask"], bool)}

    report, summary = {}, {}
    for coh in COHORTS:
        f = OUT / ("attention_v2_%s_results.json" % coh)
        res = json.load(open(f))[coh]
        y = y_all[masks[coh]]
        print("\n" + "=" * 96)
        print("COHORT %s   n=%d  Low=%d High=%d" % (coh.upper(), len(y), (y == 0).sum(), (y == 1).sum()))
        print("=" * 96)
        print("%-40s %14s %8s %8s %8s %8s" % ("arm", "Acc(3-seed mean)", "MacroF1", "HighF1", "AUC", "pooledAcc"))

        entry = {}
        for arm, label in ARM_LABEL.items():
            if arm not in res:
                continue
            accs, mf1s, hf1s, aucs, pooled_acc = [], [], [], [], []
            for s in SEEDS:
                if s not in res[arm]:
                    continue
                r = res[arm][s]
                accs.append(r["aggregate"]["accuracy"]["mean"])
                mf1s.append(r["aggregate"]["macro_f1"]["mean"])
                hf1s.append(r["aggregate"]["high_f1"]["mean"])
                aucs.append(r["aggregate"]["auc"]["mean"])
                pooled_acc.append(pooled(np.array(r["oof_pred"]), y, np.array(r["oof_prob"]))["accuracy"])
            entry[arm] = {
                "acc_mean": float(np.mean(accs)), "acc_sd": float(np.std(accs, ddof=1)),
                "macro_f1_mean": float(np.mean(mf1s)), "high_f1_mean": float(np.mean(hf1s)),
                "auc_mean": float(np.mean(aucs)), "auc_sd": float(np.std(aucs, ddof=1)),
                "pooled_acc_seed42": float(pooled_acc[0]),
            }
            e = entry[arm]
            print("%-40s %6.3f+-%.3f %8.3f %8.3f %8.3f %8.3f" % (
                label, e["acc_mean"], e["acc_sd"], e["macro_f1_mean"], e["high_f1_mean"],
                e["auc_mean"], e["pooled_acc_seed42"]))

        # ---- P1-5a: majority-class baseline ----
        maj = majority_baseline(y)
        entry["_majority_baseline"] = maj
        print("\n-- majority-class baseline: class=%d  Acc=%.3f  MacroF1=%.3f  HighF1=0.000" % (
            maj["class"], maj["accuracy"], maj["macro_f1"]))

        # ---- P1-5b: pooled PR-AUC (average precision) ----
        pr = {}
        print("\n-- pooled PR-AUC (average precision) --")
        for arm in [a for a in ARM_LABEL if a in res]:
            v = [pooled(np.array(res[arm][s]["oof_pred"]), y,
                        np.array(res[arm][s]["oof_prob"]))["pr_auc"]
                 for s in SEEDS if s in res[arm]]
            pr[arm] = {"mean": float(np.mean(v)), "per_seed": [float(x) for x in v]}
            print("  %-40s PR-AUC=%.3f  (per-seed %s)" % (
                ARM_LABEL[arm], pr[arm]["mean"], " ".join("%.3f" % x for x in v)))
        entry["_pr_auc"] = pr

        # ---- exact McNemar (seed 42, matching the manuscript's protocol) ----
        print("\n-- exact per-sample McNemar, seed 42 (Bonferroni over %d pairs) --" %
              len(list(combinations([a for a in ARM_LABEL if a in res], 2))))
        pairs = list(combinations([a for a in ARM_LABEL if a in res], 2))
        m = len(pairs)
        mcn = {}
        for a, b in pairs:
            pa = np.array(res[a]["42"]["oof_pred"])
            pb = np.array(res[b]["42"]["oof_pred"])
            bb, cc, p = mcnemar_exact(pa, pb, y)
            pb_ = min(1.0, p * m)
            mcn["%s vs %s" % (a, b)] = {"b": bb, "c": cc, "p": p, "p_bonf": pb_}
            flag = " *" if pb_ < 0.05 else ""
            print("  %-34s vs %-24s b=%3d c=%3d p=%.4f  p_bonf=%.4f%s" % (
                a, b, bb, cc, p, pb_, flag))
        entry["_mcnemar"] = mcn

        # ---- P1-5c: the same tests repeated on every seed ----
        print("\n-- exact McNemar repeated across seeds (seed 42 selected post hoc for detailed reporting) --")
        mcn_seeds = {}
        for s in SEEDS:
            if s not in res[list(ARM_LABEL)[0]]:
                continue
            rec = {}
            for a, b in pairs:
                bb, cc, p = mcnemar_exact(np.array(res[a][s]["oof_pred"]),
                                          np.array(res[b][s]["oof_pred"]), y)
                rec["%s vs %s" % (a, b)] = {"b": bb, "c": cc, "p": p,
                                            "p_bonf": min(1.0, p * m)}
            mcn_seeds[s] = rec
            sig = [k for k, v in rec.items() if v["p_bonf"] < 0.05]
            print("  seed %-4s significant after Bonferroni: %s" % (s, ", ".join(sig) or "none"))
        entry["_mcnemar_by_seed"] = mcn_seeds

        # ---- Paired bootstrap: seed-42 pooled-OOF descriptive intervals ----
        # Bootstrap every balanced metric on seed-42 pooled OOF predictions. These are
        # pointwise descriptive intervals, not inference on the three-seed fold means.
        from sklearn.metrics import average_precision_score  # noqa: E402

        best_uni = "text_only"     # primary comparator, selected after inspecting the results
        METRICS = ["accuracy", "macro_f1", "high_f1", "auc", "pr_auc"]
        N_BOOT = 500

        def fast_metrics(yy, pred, prob):
            """Recompute metrics; sklearn handles ROC-AUC ties by average rank."""
            c = np.bincount(2 * yy + pred, minlength=4)
            tn, fp, fn, tp = c[0], c[1], c[2], c[3]
            f0 = 2 * tn / max(2 * tn + fn + fp, 1)
            f1v = 2 * tp / max(2 * tp + fp + fn, 1)
            n1 = int(yy.sum()); n0 = len(yy) - n1
            if n1 and n0:
                auc = float(roc_auc_score(yy, prob))  # sklearn averages ranks for ties
            else:
                auc = 0.5
            return {"accuracy": (pred == yy).mean(), "macro_f1": (f0 + f1v) / 2,
                    "high_f1": f1v, "auc": float(auc),
                    "pr_auc": float(average_precision_score(yy, prob))}

        rng = np.random.default_rng(0)
        n = len(y)
        draws = rng.integers(0, n, size=(N_BOOT, n))
        print("\n-- paired bootstrap (%d draws) 95%% CI vs %s, all balanced metrics --"
              % (N_BOOT, best_uni))

        pred_b = np.array(res[best_uni]["42"]["oof_pred"])
        prob_b = np.array(res[best_uni]["42"]["oof_prob"])
        base_vals = np.array([[fast_metrics(y[d], pred_b[d], prob_b[d])[k] for k in METRICS]
                              for d in draws])
        boot = {}
        for arm in ["late_fusion", "gated_bilinear", "true_cross_attention"]:
            if arm not in res:
                continue
            pred_a = np.array(res[arm]["42"]["oof_pred"])
            prob_a = np.array(res[arm]["42"]["oof_prob"])
            arm_vals = np.array([[fast_metrics(y[d], pred_a[d], prob_a[d])[k] for k in METRICS]
                                 for d in draws])
            point = fast_metrics(y, pred_a, prob_a)
            pointb = fast_metrics(y, pred_b, prob_b)
            rec = {}
            for j, k in enumerate(METRICS):
                lo, hi = np.percentile(arm_vals[:, j] - base_vals[:, j], [2.5, 97.5])
                rec[k] = {"delta": float(point[k] - pointb[k]),
                          "ci95": [float(lo), float(hi)]}
            boot[arm] = {"vs": best_uni, "metrics": rec}
            print("  %-22s " % ARM_LABEL[arm] + "  ".join(
                "%s %+.3f[%+.3f,%+.3f]" % (k, rec[k]["delta"], rec[k]["ci95"][0],
                                           rec[k]["ci95"][1])
                for k in ["macro_f1", "high_f1", "auc", "pr_auc"]))
        entry["_bootstrap_vs_textonly_all_metrics"] = boot

        report[coh] = entry

    json.dump(report, open(OUT / "attention_v2_analysis.json", "w"), indent=2)
    print("\nSaved attention_v2_analysis.json")


if __name__ == "__main__":
    main()
