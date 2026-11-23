"""
R18 fix (P1-2): a structured-data baseline for the clinical-text results.

The text arm is a BiomedBERT encoder over a fixed narrative template.  If a plain
tabular model on the same underlying fields reaches comparable balanced metrics, then
whatever the text arm contributes comes from simple tabular associations rather than
from clinical semantics, and the paper should say so.

The design matrix encodes the same structured fields as the text arm using one-hot and numeric features.
It uses the same patient-level outer folds and inner-train fit rows as attention_v2_experiments.py,
with fold-mean metrics averaged across seeds. Representation and model family differ.

Variants:
    structured_all    age + gender + T/N/M/stage one-hot + diagnosis one-hot
    structured_nodx   age + gender + T/N/M/stage one-hot          (no diagnosis)
    dx_onehot         diagnosis one-hot only

Outputs -> data/TCGA-HNSC/experiment_outputs/attention_v2_structured_baseline.json
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder

BASE = Path(__file__).resolve().parent.parent
D = BASE / "data" / "TCGA-HNSC"
OUT = D / "experiment_outputs"
SEEDS = [42, 123, 456]

TXT_RE = re.compile(
    r"Diagnosis:\s*(?P<dx>.*?)\.\s*"
    r"(?:TNM staging:\s*(?P<tnm>.*?)\.\s*)?"
    r"Patient:\s*(?P<demo>.*?)\.\s*$", re.S)
TNM_RE = re.compile(r"T:\s*(?P<t>[^;]+);\s*N:\s*(?P<n>[^;]+);\s*(?:M:\s*(?P<m>[^;]+);\s*)?"
                    r"Overall stage:\s*(?P<s>.+)$")
AGE_RE = re.compile(r"\bage (\d+)\b")


def parse(text):
    m = TXT_RE.match(str(text).strip())
    if not m:
        return None
    g = m.groupdict()
    tnm = TNM_RE.match((g["tnm"] or "").strip()) if g["tnm"] else None
    demo = (g["demo"] or "").strip()
    age = AGE_RE.search(demo)
    gender = ""
    if "," in demo:
        gender = demo.split(",", 1)[1].strip()
    return {
        "dx": (g["dx"] or "").strip(),
        "t": (tnm.group("t").strip() if tnm else ""),
        "n": (tnm.group("n").strip() if tnm else ""),
        "m": ((tnm.group("m") or "").strip() if tnm else ""),
        "stage": (tnm.group("s").strip() if tnm else ""),
        "age": (int(age.group(1)) if age else np.nan),
        "gender": gender,
    }


def build(rows, kind):
    df = pd.DataFrame(rows)
    num = df[["age"]].to_numpy(dtype=float) if kind != "dx_onehot" else np.zeros((len(df), 0))
    cats = []
    if kind in ("structured_all", "structured_nodx"):
        cats = ["t", "n", "m", "stage", "gender"]
    if kind in ("structured_all", "dx_onehot"):
        cats = cats + ["dx"]
    if cats:
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        oh = enc.fit_transform(df[cats].astype(str))
    else:
        oh = np.zeros((len(df), 0))
    X = np.hstack([num, oh])
    X = np.nan_to_num(X, nan=0.0)
    return X


def metrics(y, pred, prob):
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "high_f1": float(f1_score(y, pred, average=None, labels=[0, 1])[1]),
        "auc": float(roc_auc_score(y, prob)),
        "pr_auc": float(average_precision_score(y, prob)),
    }


def make_models():
    return {
        "logreg": lambda: LogisticRegression(max_iter=5000, class_weight="balanced"),
        "gradboost": lambda: GradientBoostingClassifier(random_state=0),
        "mlp": lambda: MLPClassifier(hidden_layer_sizes=(64,), max_iter=2000,
                                     random_state=0),
    }


def inner_fit_idx(y, outer_train, seed):
    """Use the primary grid's 80/20 inner split and the same fit rows."""
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed + 10000)
    fit_rel, _ = next(splitter.split(np.zeros(len(outer_train)), y[outer_train]))
    return outer_train[fit_rel]


def check_against_archive(results, path, tol=1e-12):
    """Refuse to overwrite an archive whose fold-mean metrics this run does not reproduce."""
    if not path.exists():
        return
    old = json.load(open(path))
    bad, worst = [], 0.0
    for kind, cohorts in results.items():
        for cohort, models in cohorts.items():
            for name, rec in models.items():
                for m in ["accuracy", "macro_f1", "high_f1", "auc", "pr_auc"]:
                    delta = abs(rec[m] - old[kind][cohort][name][m])
                    worst = max(worst, delta)
                    if delta > tol:
                        bad.append("%s/%s/%s/%s: archive %.12f vs rerun %.12f"
                                   % (kind, cohort, name, m,
                                      old[kind][cohort][name][m], rec[m]))
    if bad:
        print("\nREFUSING TO OVERWRITE: rerun does not reproduce the archived metrics")
        for line in bad:
            print("  " + line)
        raise SystemExit(1)
    print("archive check: all fold-mean metrics reproduced (max |delta| = %.3e)" % worst)


def main():
    index = json.load(open(D / "attention_v2" / "cohort_index.json"))
    case_ids = index["case_ids"]
    y_all = np.array(index["y_binary"])
    masks = {"full": np.ones(len(y_all), bool),
             "wide": np.array(index["wide_mask"], bool),
             "strict": np.array(index["strict_mask"], bool)}

    meta = pd.read_csv(D / "integrated_full_legacy499" / "text_metadata_499era_ageyears.csv")
    meta = meta.drop_duplicates("case_id", keep="first").set_index("case_id")
    rows = [parse(meta.loc[c, "clinical_text"]) for c in case_ids]
    assert all(r is not None for r in rows), "unparsed narrative"

    results = {}
    for kind in ["structured_all", "structured_nodx", "dx_onehot"]:
        X_all = build(rows, kind)
        results[kind] = {}
        print("\n=== variant %s  (n_features=%d) ===" % (kind, X_all.shape[1]))
        for cohort in ["full", "wide", "strict"]:
            mask = masks[cohort]
            X, y = X_all[mask], y_all[mask]
            results[kind][cohort] = {}
            for name, factory in make_models().items():
                per_seed, oof_by_seed, seed42_store = [], {}, {}
                for seed in SEEDS:
                    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
                    pred = np.zeros(len(y), dtype=int)
                    prob = np.zeros(len(y), dtype=float)
                    fold_id = np.full(len(y), -1, dtype=int)
                    fold_metrics, fit_sizes = [], []
                    for fold, (outer_train, outer_test) in enumerate(kf.split(X, y)):
                        fit = inner_fit_idx(y, outer_train, seed)
                        model = factory()
                        model.fit(X[fit], y[fit])
                        pred[outer_test] = model.predict(X[outer_test])
                        prob[outer_test] = model.predict_proba(X[outer_test])[:, 1]
                        fold_id[outer_test] = fold
                        fold_metrics.append(metrics(y[outer_test], pred[outer_test], prob[outer_test]))
                        fit_sizes.append(int(len(fit)))
                    assert (fold_id >= 0).all(), "outer folds do not cover every patient"
                    seed_result = {k: float(np.mean([m[k] for m in fold_metrics]))
                                   for k in fold_metrics[0]}
                    seed_result["n_fit_mean"] = float(np.mean(fit_sizes))
                    per_seed.append(seed_result)
                    oof_by_seed[str(seed)] = {
                        "oof_pred": pred.tolist(),
                        "oof_prob": prob.tolist(),
                        "fold_index": fold_id.tolist(),
                        "folds": fold_metrics,
                        "n_fit_by_fold": fit_sizes,
                    }
                    if seed == 42:
                        seed42_store = {
                            "oof_pred": pred.tolist(),
                            "oof_prob": prob.tolist(),
                            "folds": fold_metrics,
                            "n_fit_by_fold": fit_sizes,
                        }
                agg = {k: float(np.mean([r[k] for r in per_seed])) for k in per_seed[0]}
                agg["sd_by_seed"] = {k: float(np.std([r[k] for r in per_seed], ddof=1))
                                      for k in ["accuracy", "macro_f1", "high_f1", "auc", "pr_auc"]}
                agg["per_seed"] = per_seed
                agg["oof_by_seed"] = oof_by_seed
                results[kind][cohort][name] = dict(agg, **seed42_store)
                print("  %-7s %-8s Acc=%.3f MacroF1=%.3f HighF1=%.3f AUC=%.3f PR-AUC=%.3f n_fit≈%.1f" % (
                    cohort, name, agg["accuracy"], agg["macro_f1"], agg["high_f1"],
                    agg["auc"], agg["pr_auc"], agg["n_fit_mean"]))

    out_path = OUT / "attention_v2_structured_baseline.json"
    check_against_archive(results, out_path)
    json.dump(results, open(out_path, "w"), indent=2)
    print("\nSaved attention_v2_structured_baseline.json")
    print("DONE")


if __name__ == "__main__":
    main()
