"""Independent replay of the primary-grid tables from archived patient-level predictions.

Reads only the stored result JSON, re-derives the outer and inner splits from each seed,
and recomputes every per-fold metric and three-seed fold mean. No model is trained.

What this can and cannot establish
----------------------------------
It can show that every stored record is internally consistent with the protocol declared
in the manuscript: one out-of-fold prediction per patient, five outer folds, inner
train/selection rows disjoint and confined to the outer training fold, and per-fold
metrics that reproduce from the OOF vectors. It cannot prove what the historical training
loop touched; the wording in the manuscript must stay at "consistent with the stated
protocol", not "no leakage".

Outputs:
    data/TCGA-HNSC/experiment_outputs/attention_v2_rederive.json
    data/TCGA-HNSC/experiment_outputs/attention_v2_rederive.tex
"""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

BASE = Path(__file__).resolve().parent.parent
D = BASE / "data" / "TCGA-HNSC"
OUT = D / "experiment_outputs"
SEEDS = [42, 123, 456]
COHORTS = ["full", "wide", "strict"]
ARMS = ["image_only", "text_only", "late_fusion", "gated_bilinear", "true_cross_attention"]
METRICS = ["accuracy", "macro_f1", "low_f1", "high_f1", "auc"]
TOL = 1e-12


def load_index():
    return json.load(open(D / "attention_v2" / "cohort_index.json"))


def load_results(cohort):
    return json.load(open(OUT / ("attention_v2_%s_results.json" % cohort)))[cohort]


def cohort_mask(index, cohort):
    if cohort == "full":
        return np.ones(len(index["y_binary"]), dtype=bool)
    return np.asarray(index["%s_mask" % cohort], dtype=bool)


def outer_folds(y, seed):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    return [va for _, va in skf.split(np.zeros(len(y)), y)]


def inner_split(y, tr, seed):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed + 10000)
    a, b = next(skf.split(np.zeros(len(tr)), y[tr]))
    return tr[a], tr[b]


def metric_block(y, pred, prob):
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "low_f1": float(f1_score(y, pred, average=None, labels=[0, 1])[0]),
        "high_f1": float(f1_score(y, pred, average=None, labels=[0, 1])[1]),
        "auc": float(roc_auc_score(y, prob)),
    }


def check_record(rec, y, seed):
    """Recompute one (cohort, arm, seed) record and return a per-record audit."""
    pred = np.asarray(rec["oof_pred"], dtype=int)
    prob = np.asarray(rec["oof_prob"], dtype=float)
    n = len(y)
    problems = []
    if len(pred) != n or len(prob) != n:
        problems.append("OOF length %d/%d != n=%d" % (len(pred), len(prob), n))
    if not np.isfinite(prob).all():
        problems.append("non-finite OOF probability")
    if ((prob < 0) | (prob > 1)).any():
        problems.append("OOF probability outside [0, 1]")
    expected_pred = np.argmax(np.stack([1.0 - prob, prob], axis=0), axis=0)
    pred_mismatches = int((pred != expected_pred).sum())
    if pred_mismatches:
        problems.append("%d OOF predictions differ from argmax(prob)" % pred_mismatches)

    va_list = outer_folds(y, seed)
    covered = np.concatenate(va_list)
    if len(covered) != n or len(np.unique(covered)) != n:
        problems.append("outer folds do not partition the cohort exactly once")

    per_fold = rec["per_fold"]
    if len(per_fold) != 5:
        problems.append("expected 5 outer folds, found %d" % len(per_fold))
    max_metric_delta = 0.0
    ap_folds = []
    inner_problems = 0
    for k, va in enumerate(va_list):
        if k >= len(per_fold):
            break
        got = metric_block(y[va], pred[va], prob[va])
        for m in METRICS:
            max_metric_delta = max(max_metric_delta, abs(got[m] - per_fold[k][m]))
        ap_folds.append(float(average_precision_score(y[va], prob[va])))

        tr = np.setdiff1d(np.arange(n), va)
        fit, sel = inner_split(y, tr, seed)
        fold = per_fold[k]
        if fold["n_fit"] != len(fit) or fold["n_sel"] != len(sel):
            inner_problems += 1
        if np.intersect1d(fit, sel).size or np.setdiff1d(np.union1d(fit, sel), tr).size:
            inner_problems += 1
    if inner_problems:
        problems.append("%d inner-split inconsistencies" % inner_problems)

    agg_delta = 0.0
    for m in METRICS:
        vals = [per_fold[k][m] for k in range(len(per_fold))]
        agg_delta = max(agg_delta,
                        abs(float(np.mean(vals)) - rec["aggregate"][m]["mean"]),
                        abs(float(np.std(vals, ddof=1)) - rec["aggregate"][m]["std"]))
    if agg_delta > TOL:
        problems.append("aggregate mean/SD delta %.3e exceeds tolerance" % agg_delta)
    if max_metric_delta > TOL:
        problems.append("per-fold metric delta %.3e exceeds tolerance" % max_metric_delta)

    return {
        "problems": problems,
        "max_metric_delta": max_metric_delta,
        "aggregate_delta": agg_delta,
        "pred_argmax_mismatches": pred_mismatches,
        "average_precision": float(np.mean(ap_folds)),
        "fold_va": [va.tolist() for va in va_list],
    }


def fold_mean(pred, prob, y, va_list, with_ap=False):
    """Fold-mean metrics recomputed from one OOF vector and the seeded outer partition."""
    per_fold = [metric_block(y[va], pred[va], prob[va]) for va in va_list]
    got = {m: float(np.mean([f[m] for f in per_fold])) for m in METRICS}
    if with_ap:
        got["average_precision"] = float(np.mean(
            [average_precision_score(y[va], prob[va]) for va in va_list]))
    return got


def replay_rep_match(index, failures):
    """Table 3: recompute the representation-matched cells from their OOF vectors."""
    rep = json.load(open(OUT / "attention_v2_rep_match.json"))["results"]["strict"]
    y = np.asarray(index["y_binary"])[cohort_mask(index, "strict")].astype(int)
    worst, cells = 0.0, {}
    for arm, seeds in rep.items():
        cells[arm] = {}
        for seed in SEEDS:
            rec = seeds[str(seed)]
            pred = np.asarray(rec["oof_pred"], dtype=int)
            prob = np.asarray(rec["oof_prob"], dtype=float)
            got = fold_mean(pred, prob, y, outer_folds(y, seed))
            present = [m for m in METRICS if m in rec["aggregate"]]
            deltas = {m: abs(got[m] - rec["aggregate"][m]["mean"]) for m in present}
            worst = max(worst, max(deltas.values()))
            for m, d in deltas.items():
                if d > TOL:
                    failures.append("rep_match/%s/seed %d/%s delta %.3e" % (arm, seed, m, d))
            cells[arm][str(seed)] = got
    return {"max_metric_delta": worst, "cells": cells}


def replay_structured(index, failures):
    """Table 12: recompute every structured-baseline record from its per-seed OOF."""
    sb = json.load(open(OUT / "attention_v2_structured_baseline.json"))
    worst, records = 0.0, 0
    for kind, cohorts in sb.items():
        for cohort, models in cohorts.items():
            y = np.asarray(index["y_binary"])[cohort_mask(index, cohort)].astype(int)
            for name, rec in models.items():
                for i, seed in enumerate(SEEDS):
                    oof = rec["oof_by_seed"][str(seed)]
                    pred = np.asarray(oof["oof_pred"], dtype=int)
                    prob = np.asarray(oof["oof_prob"], dtype=float)
                    fold_id = np.asarray(oof["fold_index"], dtype=int)
                    va_list = [np.where(fold_id == k)[0] for k in range(5)]
                    got = fold_mean(pred, prob, y, va_list, with_ap=True)
                    got["pr_auc"] = got.pop("average_precision")
                    records += 1
                    for m in ["accuracy", "macro_f1", "high_f1", "auc", "pr_auc"]:
                        d = abs(got[m] - rec["per_seed"][i][m])
                        worst = max(worst, d)
                        if d > TOL:
                            failures.append("structured/%s/%s/%s/seed %d/%s delta %.3e"
                                            % (kind, cohort, name, seed, m, d))
    return {"max_metric_delta": worst, "records": records}


def replay_nodx(index, failures):
    """Table 13 and its per-metric difference block, replayed from cell OOF vectors."""
    nd = json.load(open(OUT / "attention_v2_nodx_fourcell.json"))
    y = np.asarray(index["y_binary"])[cohort_mask(index, "strict")].astype(int)
    worst = 0.0
    replayed = {"with_diagnosis": {}, "no_diagnosis": {}}
    for block in ("with_diagnosis", "no_diagnosis"):
        for arm in ("text_only", "true_cross_attention"):
            replayed[block][arm] = {}
            for seed in SEEDS:
                cell = nd["cells"][block][arm][str(seed)]
                pred = np.asarray(cell["oof_pred"], dtype=int)
                prob = np.asarray(cell["oof_prob"], dtype=float)
                got = fold_mean(pred, prob, y, outer_folds(y, seed), with_ap=True)
                for m in ("accuracy", "macro_f1", "low_f1", "high_f1", "auc",
                          "average_precision"):
                    d = abs(got[m] - cell[m])
                    worst = max(worst, d)
                    if d > TOL:
                        failures.append("nodx/%s/%s/seed %d/%s delta %.3e"
                                        % (block, arm, seed, m, d))
                replayed[block][arm][str(seed)] = got

    diffs = {}
    for m in ("macro_f1", "high_f1", "average_precision", "auc"):
        d_full, d_nodx, d_delta = [], [], []
        for seed in SEEDS:
            a = (replayed["with_diagnosis"]["true_cross_attention"][str(seed)][m]
                 - replayed["with_diagnosis"]["text_only"][str(seed)][m])
            b = (replayed["no_diagnosis"]["true_cross_attention"][str(seed)][m]
                 - replayed["no_diagnosis"]["text_only"][str(seed)][m])
            d_full.append(a)
            d_nodx.append(b)
            d_delta.append(b - a)
        stored = nd["per_metric_differences"][m]
        for label, mine, theirs in (("D_full_mean", float(np.mean(d_full)), stored["D_full_mean"]),
                                    ("D_nodx_mean", float(np.mean(d_nodx)), stored["D_nodx_mean"]),
                                    ("delta_D_mean", float(np.mean(d_delta)),
                                     stored["delta_D_mean"])):
            d = abs(mine - theirs)
            worst = max(worst, d)
            if d > TOL:
                failures.append("nodx/%s/%s delta %.3e" % (m, label, d))
        diffs[m] = {"D_full_mean": float(np.mean(d_full)),
                    "D_nodx_mean": float(np.mean(d_nodx)),
                    "delta_D_mean": float(np.mean(d_delta)),
                    "delta_D_sd": float(np.std(d_delta, ddof=1)),
                    "delta_D_positive": int(sum(x > 0 for x in d_delta))}
    return {"max_metric_delta": worst, "differences": diffs}


def reference_dependence():
    """Table 4 values recomputed from unrounded three-seed arm means."""
    strict = load_results("strict")
    rep = json.load(open(OUT / "attention_v2_rep_match.json"))["results"]["strict"]

    def arm_mean(block, arm, key):
        return float(np.mean([block[arm][str(s)]["aggregate"][key]["mean"] for s in SEEDS]))

    keys = ["macro_f1", "auc", "high_f1"]
    base = {k: arm_mean(strict, "true_cross_attention", k) for k in keys}
    rows = {}
    for label, block, arm in (("text_only_cls", strict, "text_only"),
                              ("text_only_tokmean", rep, "text_only_tokmean"),
                              ("image_only_stored", strict, "image_only"),
                              ("image_only_patchmean", rep, "image_only_patchmean")):
        rows[label] = {k: round(base[k] - arm_mean(block, arm, k), 3) for k in keys}
    return rows


def main():
    index = load_index()
    report = {"tolerance": TOL, "date": "2026-11-22", "cohorts": {},
              "partitions": {}, "records": 0, "folds": 0}
    failures = []
    partition_by_cohort_seed = {}

    for cohort in COHORTS:
        mask = cohort_mask(index, cohort)
        y = np.asarray(index["y_binary"])[mask].astype(int)
        res = load_results(cohort)
        report["cohorts"][cohort] = {"n": int(len(y)),
                                     "prevalence": float(y.mean()), "arms": {}}
        for arm in ARMS:
            arm_block = {}
            for seed in SEEDS:
                if arm not in res or str(seed) not in res[arm]:
                    failures.append("%s/%s/seed %d missing" % (cohort, arm, seed))
                    continue
                audit = check_record(res[arm][str(seed)], y, seed)
                report["records"] += 1
                report["folds"] += 5
                if audit["problems"]:
                    failures.extend("%s/%s/seed %d: %s" % (cohort, arm, seed, p)
                                    for p in audit["problems"])
                key = (cohort, seed)
                sig = tuple(tuple(v) for v in audit["fold_va"])
                if key in partition_by_cohort_seed and partition_by_cohort_seed[key] != sig:
                    failures.append("%s/seed %d: fold partition differs across arms"
                                    % (cohort, seed))
                partition_by_cohort_seed.setdefault(key, sig)
                arm_block[str(seed)] = {
                    "max_metric_delta": audit["max_metric_delta"],
                    "aggregate_delta": audit["aggregate_delta"],
                    "pred_argmax_mismatches": audit["pred_argmax_mismatches"],
                    "average_precision": audit["average_precision"],
                }
            if arm_block:
                aps = [arm_block[str(s)]["average_precision"] for s in SEEDS]
                arm_block["average_precision_three_seed_mean"] = float(np.mean(aps))
                report["cohorts"][cohort]["arms"][arm] = arm_block

    # Tables 3, 4, 12 and 13 replayed from archived patient-level predictions.
    report["tables"] = {
        "rep_match": replay_rep_match(index, failures),
        "structured": replay_structured(index, failures),
        "nodx": replay_nodx(index, failures),
        "reference_dependence": reference_dependence(),
    }

    # The manuscript's strict-subset text-only and true-cross-attention AP are 0.293/0.271.
    report["manuscript_ap_check"] = {
        "strict_text_only": report["cohorts"]["strict"]["arms"]["text_only"]
        ["average_precision_three_seed_mean"],
        "strict_true_cross_attention": report["cohorts"]["strict"]["arms"]
        ["true_cross_attention"]["average_precision_three_seed_mean"],
        "reported": {"strict_text_only": 0.293, "strict_true_cross_attention": 0.271},
    }
    report["failures"] = failures
    report["all_checks_passed"] = not failures
    report["max_metric_delta"] = max(
        (rec["max_metric_delta"]
         for cohort in COHORTS
         for block in report["cohorts"][cohort]["arms"].values()
         for seed, rec in block.items() if seed in ("42", "123", "456")), default=0.0)

    json.dump(report, open(OUT / "attention_v2_rederive.json", "w"), indent=2)

    lines = [
        "% Independent replay of the primary grid (attention_v2_rederive.py).",
        "% Each record: 5 outer folds recomputed from OOF vectors.",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Cohort & Records & Folds & Max $|\\Delta|$ & Failures \\\\",
        "\\midrule",
    ]
    for cohort in COHORTS:
        n_rec = sum(1 for arm in report["cohorts"][cohort]["arms"]
                    for seed in report["cohorts"][cohort]["arms"][arm]
                    if seed in ("42", "123", "456"))
        cohort_failures = sum(1 for f in failures if f.startswith(cohort + "/"))
        lines.append("%s & %d & %d & %.1e & %d \\\\"
                     % (cohort, n_rec, 5 * n_rec, report["max_metric_delta"],
                        cohort_failures))
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    (OUT / "attention_v2_rederive.tex").write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({"records": report["records"], "folds": report["folds"],
                      "all_checks_passed": report["all_checks_passed"],
                      "max_metric_delta": report["max_metric_delta"],
                      "failures": failures[:10],
                      "manuscript_ap_check": report["manuscript_ap_check"]}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
