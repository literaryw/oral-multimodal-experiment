"""
R19 P1 fix: ablate the text-summary skip path of the true cross-attention head.

`TrueCrossAttentionFusion` concatenates the attended image pooling with a *direct*
masked-mean text summary before the classifier.  That skip path gives the head a
text-only route to the logits, so the full-head result does not by itself show that the
attended image representation carries the signal.  This script separates the three
paths on the same folds and seeds:

    ca_full        concat(pooled attention over patches, masked-mean text)   [current]
    ca_patch_only  pooled attention over patches only
    ca_text_only   masked-mean text summary only

Outputs -> data/TCGA-HNSC/experiment_outputs/attention_v2_skip_ablation.json
"""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

sys.path.insert(0, str(Path(__file__).resolve().parent))
import attention_v2_experiments as E  # noqa: E402
from attention_v2_models import FocalLoss, TrueCrossAttentionFusion  # noqa: E402

D = E.DATA_DIR
OUT = D / "experiment_outputs"
SEEDS = [42, 123, 456]


def _head(hidden, num_classes=2, dropout=0.4):
    return nn.Sequential(
        nn.Linear(hidden, hidden), nn.ReLU(), nn.BatchNorm1d(hidden), nn.Dropout(dropout),
        nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(hidden // 2, num_classes))


class CAPatchOnly(TrueCrossAttentionFusion):
    """Full head with the text branch zero-masked (parameter-matched).

    R20 P1-5: the first version replaced the head with a 256-input MLP, so it changed
    capacity as well as removing the branch.  Zeroing the branch keeps the original
    512-input head, so the only difference is which branch carries information.
    """

    forward = TrueCrossAttentionFusion.forward  # overridden below

    def forward(self, patches, patch_mask, text_tokens, text_mask):  # noqa: F811
        q = self.norm_q(self.img_proj(patches))
        kv = self.norm_t(self.text_proj(text_tokens))
        attn_out, _ = self.cross_attn(q, kv, kv, key_padding_mask=(text_mask == 0),
                                      need_weights=False)
        m = patch_mask.unsqueeze(-1).to(attn_out.dtype)
        pooled = (attn_out * m).sum(1) / m.sum(1).clamp(min=1.0)
        return self.head(torch.cat([pooled, torch.zeros_like(pooled)], -1)), None


class CATextOnly(TrueCrossAttentionFusion):
    """Full head with the patch branch zero-masked (parameter-matched)."""

    def forward(self, patches, patch_mask, text_tokens, text_mask):
        kv = self.norm_t(self.text_proj(text_tokens))
        mt = text_mask.unsqueeze(-1).to(kv.dtype)
        summary = (kv * mt).sum(1) / mt.sum(1).clamp(min=1.0)
        return self.head(torch.cat([torch.zeros_like(summary), summary], -1)), None


ARMS = {
    "ca_full": (lambda: TrueCrossAttentionFusion(),
                lambda m, b: m(b[2].to(E.DEVICE), b[3].to(E.DEVICE), b[4].to(E.DEVICE), b[5].to(E.DEVICE))),
    "ca_patch_only": (lambda: CAPatchOnly(),
                      lambda m, b: m(b[2].to(E.DEVICE), b[3].to(E.DEVICE), b[4].to(E.DEVICE), b[5].to(E.DEVICE))),
    "ca_text_only": (lambda: CATextOnly(),
                     lambda m, b: m(b[2].to(E.DEVICE), b[3].to(E.DEVICE), b[4].to(E.DEVICE), b[5].to(E.DEVICE))),
}


@torch.no_grad()
def evaluate(fn, model, loader):
    model.eval()
    preds, labels, probs = [], [], []
    for batch in loader:
        logits, _ = fn(model, batch)
        preds.extend(logits.argmax(1).cpu().tolist())
        labels.extend(batch[-1].tolist())
        probs.extend(torch.softmax(logits, 1).cpu().tolist())
    return np.array(preds), np.array(labels), np.array(probs)


def run_fold(fn, factory, c, tr, va, seed):
    fit_idx, sel_idx = E.inner_split(c["y"], tr, seed)
    tr_loader = E.make_loader(c, fit_idx, True)
    sel_loader = E.make_loader(c, sel_idx, False)
    test_loader = E.make_loader(c, va, False)
    model = factory().to(E.DEVICE)
    yt = c["y"][fit_idx]
    counts = np.bincount(yt, minlength=2)
    alpha = torch.FloatTensor([1.0, len(yt) / max(counts.max(), 1)]).to(E.DEVICE)
    crit = FocalLoss(alpha=alpha, gamma=2.0)
    opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    sched = CosineAnnealingWarmRestarts(opt, T_0=20, T_mult=2)
    best_f1, best_state, no_improve = -1.0, None, 0
    for _ in range(E.EPOCHS):
        model.train()
        for batch in tr_loader:
            opt.zero_grad()
            logits, _ = fn(model, batch)
            crit(logits, batch[-1].to(E.DEVICE)).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        p, l, _ = evaluate(fn, model, sel_loader)
        vf1 = f1_score(l, p, average="macro")
        if vf1 > best_f1:
            best_f1, best_state, no_improve = vf1, copy.deepcopy(model.state_dict()), 0
        else:
            no_improve += 1
            if no_improve >= E.PATIENCE:
                break
    model.load_state_dict(best_state)
    preds, labels, probs = evaluate(fn, model, test_loader)
    return {"accuracy": float(accuracy_score(labels, preds)),
            "macro_f1": float(f1_score(labels, preds, average="macro")),
            "high_f1": float(f1_score(labels, preds, average=None, labels=[0, 1])[1]),
            "auc": float(roc_auc_score(labels, probs[:, 1]))}, preds, probs


def run_cv(fn, factory, c, seed):
    E.seed_everything(seed)
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    folds = []
    for tr, va in kf.split(c["X_img"], c["y"]):
        r, _, _ = run_fold(fn, factory, c, tr, va, seed)
        folds.append(r)
    return {k: {"mean": float(np.mean([f[k] for f in folds])),
                "std": float(np.std([f[k] for f in folds], ddof=1))} for k in folds[0]}


def main():
    index = json.load(open(E.BUNDLE / "cohort_index.json"))
    slide_img = np.load(D / "integrated_full_legacy499" / "X_img_uni.npy")
    cls = np.load(E.BUNDLE / "text_tokens.npz")["cls"]
    pat = np.load(E.BUNDLE / "img_patches.npz")
    patch_list = [pat["case_%d" % i] for i in range(index["n_base"])]
    masks = {"full": np.ones(index["n_base"], bool),
             "strict": np.array(index["strict_mask"], bool)}

    results = {}
    for cname, mask in masks.items():
        c = E.build_cohort(index, mask, slide_img, cls, patch_list)
        print("\n=== cohort %s (n=%d) ===" % (cname, len(c["y"])))
        results[cname] = {}
        for arm, (factory, fn) in ARMS.items():
            per_seed = [run_cv(fn, factory, c, s) for s in SEEDS]
            agg = {k: {"mean": float(np.mean([p[k]["mean"] for p in per_seed])),
                       "std": float(np.std([p[k]["mean"] for p in per_seed], ddof=1))}
                   for k in per_seed[0]}
            results[cname][arm] = agg
            print("  %-14s Acc=%.3f+-%.3f MacroF1=%.3f HighF1=%.3f AUC=%.3f" % (
                arm, agg["accuracy"]["mean"], agg["accuracy"]["std"],
                agg["macro_f1"]["mean"], agg["high_f1"]["mean"], agg["auc"]["mean"]))
        json.dump(results, open(OUT / "attention_v2_skip_ablation.json", "w"), indent=2)
    print("\nSaved attention_v2_skip_ablation.json")


if __name__ == "__main__":
    main()
