"""Replay Table 17 from archived predictions. Requires numpy and pyarrow."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from io import BytesIO

import numpy as np
import pyarrow.parquet as pq

SPLITS = (42, 123, 456)
MEMBERS = (20261026, 20261027, 20261105)
OOF_SHA = "56ea17833fdb41990f3234a5f6467f12d14cdd960b5df046ac7d3b543751961c"
ARCH_SHA = "4025b43f82f8d083550498b9e8fa4945a2057dbe2b33c907f6cb1f577f0e5f21"


def verified(path, expected):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f"Archive digest mismatch: {path}")
    return data


def macro_f1(labels, prediction, weights):
    positive = labels == 1
    tp = weights[:, positive & prediction].sum(1)
    tn = weights[:, ~positive & ~prediction].sum(1)
    errors = weights[:, positive != prediction].sum(1)
    return tp / np.maximum(2 * tp + errors, 1) + tn / np.maximum(2 * tn + errors, 1)


def replay(run, label_file):
    import pyarrow as pa

    archive = json.loads(verified(run / "architecture_outer_evaluation_v2/metrics.json", ARCH_SHA))
    labels_csv = verified(label_file, archive["patient_label_source_sha256"])
    rows = [r for r in csv.DictReader(labels_csv.decode("utf-8").splitlines()) if r["cohort"] == "strict"]
    ids = [r["case_id"] for r in rows]
    y = np.array([int(r["label"]) for r in rows])
    if len(set(ids)) != 233 or len(ids) != 233 or set(y) != {0, 1} or y.sum() != 48:
        raise ValueError("Unexpected strict cohort")
    positions = {case: i for i, case in enumerate(ids)}
    oof = pq.read_table(pa.BufferReader(verified(run / "outer_evaluation_v4/strict_outer_oof_v4.parquet", OOF_SHA)),
                        columns=["role", "case_id", "outer_seed", "member_seed", "high_probability"]).to_pylist()
    arrays = {role: np.full((3, 3, 233), np.nan) for role in ("B-main", "B-v4", "A07", "A08")}

    def put(role, split, member, case, probability):
        key = (SPLITS.index(split), MEMBERS.index(member), positions[case])
        if not np.isnan(arrays[role][key]) or not 0 <= probability <= 1:
            raise ValueError("Duplicate or invalid prediction")
        arrays[role][key] = probability

    for r in oof:
        if r["role"] in ("B-main", "B-v4"):
            put(r["role"], r["outer_seed"], r["member_seed"], r["case_id"], r["high_probability"])
    for r in archive["prediction_records"]:
        if r["model"] not in ("A07", "A08"):
            continue
        split, fold, role, member = (r[k] for k in ("outer_seed", "outer_fold", "model", "training_seed"))
        path = run / "architecture_outer_predictions_v2" / f"seed{split}/fold{fold}/{role}/seed{member}/prediction.npz"
        with np.load(BytesIO(verified(path, r["prediction_sha256"])), allow_pickle=False) as z:
            for case, p in zip(z["case_ids"], z["probability"], strict=True):
                put(role, split, member, case, p)
    if any(not np.isfinite(a).all() for a in arrays.values()):
        raise ValueError("Incomplete patient/member coverage")
    predictions = {role: a.mean(1) > .5 for role, a in arrays.items()}
    unit = np.ones((1, len(y)), dtype=int)
    scores = {role: np.array([macro_f1(y, p, unit)[0] for p in splits])
              for role, splits in predictions.items()}
    rng = np.random.default_rng(20261109)
    weights = np.zeros((4000, len(y)), dtype=int)
    for group in (0, 1):
        indices = np.flatnonzero(y == group)
        weights[:, indices] = rng.multinomial(len(indices), np.full(len(indices), 1 / len(indices)), size=4000)
    delta = scores["A08"] - scores["A07"]
    draws = np.mean([macro_f1(y, a, weights) - macro_f1(y, b, weights)
                     for a, b in zip(predictions["A08"], predictions["A07"], strict=True)], axis=0)
    return {"B-main": float(scores["B-main"].mean()), "B-v4": float(scores["B-v4"].mean()),
            "C-A2": {"mean": float(delta.mean()), "per_split": delta.tolist(),
                     "conditional_ci95": np.percentile(draws, [2.5, 97.5]).tolist(),
                     "all_splits_nonnegative": bool((delta >= 0).all())}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Archived Grade run directory")
    parser.add_argument("--labels", type=Path, required=True, help="Archived patient_join.csv")
    args = parser.parse_args()
    print(json.dumps(replay(args.run, args.labels), indent=2))
