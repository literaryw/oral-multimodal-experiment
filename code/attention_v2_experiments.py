"""
R16-fix Phase C: full-grid re-run with a genuine cross-attention arm.

Grid: 3 cohorts (full 499 / wide 332 / strict 246)
    x 5 arms (image-only, text-only, late fusion, gated-bilinear, true cross-attention)
    x 5 folds x 3 seeds (42 / 123 / 456)

Training protocol is unchanged from Section 3.7 of the manuscript so the new arm is
directly comparable with the published rows: Focal Loss gamma=2 with class-specific
alpha, WeightedRandomSampler, AdamW(lr=1e-3, wd=1e-3), CosineAnnealingWarmRestarts(T0=20),
<=100 epochs, early stopping patience=15 on validation macro F1, and torch/numpy seeds
reset before every model's cross-validation loop.

R18 fix (P0-1): the early-stopping validation set is now drawn from *inside* the training
fold (nested CV).  Previously the outer held-out fold served both as the early-stopping
signal and as the set the reported metrics come from, so model selection saw the test
data.  The inner split is deterministic per (fold, seed) and the outer fold is untouched
until the final evaluation.

Usage:
    python code/attention_v2_experiments.py --cohort strict --seeds 42      # smoke
    python code/attention_v2_experiments.py                                # full grid
"""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from attention_v2_models import (FocalLoss, GatedBilinearFusion, ImageOnly, LateFusion,
                                 TextOnly, TrueCrossAttentionFusion)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "TCGA-HNSC"
BUNDLE = DATA_DIR / "attention_v2"
OUT_DIR = DATA_DIR / "experiment_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS, PATIENCE, BATCH = 100, 15, 32
INNER_SPLITS = 5          # inner CV folds carved out of each training fold (R18 P0-1)

# Determinism: the reported numbers must be reproducible on re-run, so the cuDNN
# autotuner is switched off (it picks different kernels run to run) and every seed is
# set explicitly.
if DEVICE.type == "cuda":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_everything(seed):
    torch.manual_seed(seed)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

ARMS = ["image_only", "text_only", "late_fusion", "gated_bilinear", "true_cross_attention"]


def build_cohort(index, mask, slide_img, cls, patches):
    """Slice the base bundle down to one cohort and pad the variable-length sequences."""
    mask = np.asarray(mask, dtype=bool)
    rows = np.where(mask)[0]
    n = len(rows)

    # patch sequences -> (n, Nmax, 1024)
    plist = [patches[r] for r in rows]
    nmax = max(p.shape[0] for p in plist)
    P = np.zeros((n, nmax, 1024), dtype=np.float32)
    PM = np.zeros((n, nmax), dtype=np.int8)
    for i, p in enumerate(plist):
        P[i, :p.shape[0]] = p
        PM[i, :p.shape[0]] = 1

    tok = np.load(BUNDLE / "text_tokens.npz")
    T, TM = tok["tokens"][rows], tok["mask"][rows]

    return {
        "rows": rows,
        "X_img": slide_img[rows].astype(np.float32),
        "X_text": cls[rows].astype(np.float32),
        "y": np.array(index["y_binary"])[rows].astype(np.int64),
        "patches": P, "patch_mask": PM,
        "tokens": T.astype(np.float32), "text_mask": TM.astype(np.int8),
        "nmax": nmax, "lmax": T.shape[1],
    }


def make_loader(c, idx, train):
    tensors = [
        torch.from_numpy(c["X_img"][idx]),
        torch.from_numpy(c["X_text"][idx]),
        torch.from_numpy(c["patches"][idx]),
        torch.from_numpy(c["patch_mask"][idx]).long(),
        torch.from_numpy(c["tokens"][idx]),
        torch.from_numpy(c["text_mask"][idx]).long(),
        torch.from_numpy(c["y"][idx]),
    ]
    ds = TensorDataset(*tensors)
    if not train:
        return DataLoader(ds, batch_size=64)
    yt = c["y"][idx]
    counts = np.bincount(yt, minlength=2)
    w = 1.0 / np.maximum(counts[yt], 1)
    sampler = WeightedRandomSampler(torch.DoubleTensor(w), len(yt), replacement=True)
    return DataLoader(ds, batch_size=BATCH, sampler=sampler)


def forward_arm(arm, model, batch):
    xi, xt, p, pm, tk, tm, _ = batch
    if arm == "image_only":
        return model(xi.to(DEVICE))
    if arm == "text_only":
        return model(xt.to(DEVICE))
    if arm in ("late_fusion", "gated_bilinear"):
        return model(xi.to(DEVICE), xt.to(DEVICE))
    if arm == "true_cross_attention":
        return model(p.to(DEVICE), pm.to(DEVICE), tk.to(DEVICE), tm.to(DEVICE))
    raise ValueError(arm)


def make_model(arm):
    return {
        "image_only": lambda: ImageOnly(),
        "text_only": lambda: TextOnly(),
        "late_fusion": lambda: LateFusion(),
        "gated_bilinear": lambda: GatedBilinearFusion(),
        "true_cross_attention": lambda: TrueCrossAttentionFusion(),
    }[arm]()


@torch.no_grad()
def evaluate(arm, model, loader):
    model.eval()
    preds, labels, probs = [], [], []
    for batch in loader:
        logits, _ = forward_arm(arm, model, batch)
        preds.extend(logits.argmax(1).cpu().tolist())
        labels.extend(batch[-1].tolist())
        probs.extend(torch.softmax(logits, 1).cpu().tolist())
    return np.array(preds), np.array(labels), np.array(probs)


def inner_split(y, tr, seed):
    """Carve a stratified inner-train / inner-validation split out of the training fold.

    The inner validation set is the only signal early stopping and best-epoch selection
    are allowed to see; the outer fold `va` stays untouched until the final evaluation.
    """
    skf = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=seed + 10000)
    a, b = next(skf.split(np.zeros(len(tr)), y[tr]))
    return tr[a], tr[b]


def run_fold(arm, c, tr, va, seed):
    fit_idx, sel_idx = inner_split(c["y"], tr, seed)
    tr_loader = make_loader(c, fit_idx, True)
    sel_loader = make_loader(c, sel_idx, False)
    test_loader = make_loader(c, va, False)

    model = make_model(arm).to(DEVICE)
    yt = c["y"][fit_idx]
    counts = np.bincount(yt, minlength=2)
    alpha = torch.FloatTensor([1.0, len(yt) / max(counts.max(), 1)]).to(DEVICE)
    criterion = FocalLoss(alpha=alpha, gamma=2.0)
    opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    sched = CosineAnnealingWarmRestarts(opt, T_0=20, T_mult=2)

    best_f1, best_state, no_improve = -1.0, None, 0
    for epoch in range(EPOCHS):
        model.train()
        for batch in tr_loader:
            opt.zero_grad()
            logits, _ = forward_arm(arm, model, batch)
            criterion(logits, batch[-1].to(DEVICE)).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        p, l, _ = evaluate(arm, model, sel_loader)
        vf1 = f1_score(l, p, average="macro")
        if vf1 > best_f1:
            best_f1, best_state, no_improve = vf1, copy.deepcopy(model.state_dict()), 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                break

    model.load_state_dict(best_state)
    preds, labels, probs = evaluate(arm, model, test_loader)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro")),
        "low_f1": float(f1_score(labels, preds, average=None, labels=[0, 1])[0]),
        "high_f1": float(f1_score(labels, preds, average=None, labels=[0, 1])[1]),
        "auc": float(roc_auc_score(labels, probs[:, 1])),
        "epochs": int(epoch - no_improve),
        "n_fit": int(len(fit_idx)),
        "n_sel": int(len(sel_idx)),
    }, preds, probs


def run_cv(arm, c, seed):
    """Per-model seed reset, then 5-fold stratified CV with fold splits from the seed."""
    seed_everything(seed)
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    folds, oof_pred, oof_prob = [], np.zeros(len(c["y"]), dtype=int), np.zeros(len(c["y"]))
    for tr, va in kf.split(c["X_img"], c["y"]):
        res, preds, probs = run_fold(arm, c, tr, va, seed)
        folds.append(res)
        oof_pred[va] = preds
        oof_prob[va] = probs[:, 1]
    agg = {}
    for k in ["accuracy", "macro_f1", "low_f1", "high_f1", "auc"]:
        vals = [f[k] for f in folds]
        agg[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)),
                  "per_fold": [float(v) for v in vals]}
    return {"aggregate": agg, "per_fold": folds,
            "oof_pred": oof_pred.tolist(), "oof_prob": oof_prob.tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="all", choices=["all", "full", "wide", "strict"])
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--seeds", default="42,123,456")
    ap.add_argument("--tag", default="attention_v2")
    args = ap.parse_args()

    index = json.load(open(BUNDLE / "cohort_index.json"))
    slide_img = np.load(DATA_DIR / "integrated_full_legacy499" / "X_img_uni.npy")
    tok = np.load(BUNDLE / "text_tokens.npz")
    cls = tok["cls"]
    patches = np.load(BUNDLE / "img_patches.npz")
    patch_list = [patches["case_%d" % i] for i in range(index["n_base"])]

    mask_by_cohort = {
        "full": np.ones(index["n_base"], dtype=bool),
        "wide": np.array(index["wide_mask"], dtype=bool),
        "strict": np.array(index["strict_mask"], dtype=bool),
    }
    want = ["full", "wide", "strict"] if args.cohort == "all" else [args.cohort]
    arms = [a for a in args.arms.split(",") if a]
    seeds = [int(s) for s in args.seeds.split(",")]

    print("Device: %s | cohorts=%s | arms=%s | seeds=%s" % (DEVICE, want, arms, seeds))
    out = {}
    for cname in want:
        c = build_cohort(index, mask_by_cohort[cname], slide_img, cls, patch_list)
        print("\n=== cohort %s: n=%d (Low=%d High=%d) patches Nmax=%d tokens Lmax=%d ===" % (
            cname, len(c["y"]), int((c["y"] == 0).sum()), int((c["y"] == 1).sum()),
            c["nmax"], c["lmax"]))
        out[cname] = {}
        for arm in arms:
            out[cname][arm] = {}
            for seed in seeds:
                t0 = time.time()
                r = run_cv(arm, c, seed)
                dt = time.time() - t0
                out[cname][arm][str(seed)] = r
                a = r["aggregate"]
                print("  %-22s seed=%-4d Acc=%.3f+-%.3f MacroF1=%.3f HighF1=%.3f AUC=%.3f  (%.0fs)" % (
                    arm, seed, a["accuracy"]["mean"], a["accuracy"]["std"],
                    a["macro_f1"]["mean"], a["high_f1"]["mean"], a["auc"]["mean"], dt))
                json.dump(out, open(OUT_DIR / ("%s_results.json" % args.tag), "w"), indent=2)

    json.dump(out, open(OUT_DIR / ("%s_results.json" % args.tag), "w"), indent=2)
    print("\nSaved %s_results.json" % args.tag)


if __name__ == "__main__":
    main()
