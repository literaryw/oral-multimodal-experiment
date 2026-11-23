"""
R18 fix (P1-1): representation-matched controls and parameter counts.

The main grid is not representation-matched: text-only reads the [CLS] embedding while
the genuine cross-attention reads masked-mean text tokens; image-only reads stored
mean-pooled slide features while the cross-attention reads patch sequences.  A reader
cannot tell whether a difference between arms is the fusion mechanism or the pooling.

This script adds the four missing cells plus parameter counts:
    text_only_tokmean    TextOnly on masked-mean text tokens   (matches true CA's text side)
    image_only_patchmean ImageOnly on mean-pooled patch sequence (matches true CA's image side)
    true_ca_cls          TrueCrossAttentionFusion summarised by [CLS] (matches TextOnly)
    (parameter-matched late fusion is reported as a count; its hidden width already
     equals the fusion heads' 256, so no extra run is needed)

Training protocol, folds and seeds are identical to attention_v2_experiments.py.

Outputs -> data/TCGA-HNSC/experiment_outputs/attention_v2_rep_match.json
"""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

sys.path.insert(0, str(Path(__file__).resolve().parent))
import attention_v2_experiments as E  # noqa: E402
from attention_v2_models import (FocalLoss, GatedBilinearFusion, ImageOnly, LateFusion,  # noqa: E402
                                 TextOnly, TrueCrossAttentionFusion)

D = E.DATA_DIR
OUT = D / "experiment_outputs"
SEEDS = [42, 123, 456]


class CLSCrossAttention(TrueCrossAttentionFusion):
    """Genuine cross-attention whose text summary is the [CLS] token (token 0).

    Matches the representation TextOnly consumes, isolating the pooling choice from the
    attention mechanism.
    """

    def forward(self, patches, patch_mask, text_tokens, text_mask):
        q = self.norm_q(self.img_proj(patches))
        kv = self.norm_t(self.text_proj(text_tokens))
        attn_out, _ = self.cross_attn(q, kv, kv, key_padding_mask=(text_mask == 0),
                                      need_weights=False)
        m = patch_mask.unsqueeze(-1).to(attn_out.dtype)
        pooled = (attn_out * m).sum(1) / m.sum(1).clamp(min=1.0)
        cls = kv[:, 0, :]
        return self.head(torch.cat([pooled, cls], -1)), None


def masked_mean(x, mask):
    m = mask.unsqueeze(-1).to(x.dtype)
    return (x * m).sum(1) / m.sum(1).clamp(min=1.0)


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
    criterion = FocalLoss(alpha=alpha, gamma=2.0)
    opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    sched = CosineAnnealingWarmRestarts(opt, T_0=20, T_mult=2)

    best_f1, best_state, no_improve = -1.0, None, 0
    for _ in range(E.EPOCHS):
        model.train()
        for batch in tr_loader:
            opt.zero_grad()
            logits, _ = fn(model, batch)
            criterion(logits, batch[-1].to(E.DEVICE)).backward()
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
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro")),
        "high_f1": float(f1_score(labels, preds, average=None, labels=[0, 1])[1]),
        "auc": float(roc_auc_score(labels, probs[:, 1])),
    }, preds, probs


def run_cv(fn, factory, c, seed):
    E.seed_everything(seed)
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    folds, oof_pred, oof_prob = [], np.zeros(len(c["y"]), int), np.zeros(len(c["y"]))
    for tr, va in kf.split(c["X_img"], c["y"]):
        r, p, pr = run_fold(fn, factory, c, tr, va, seed)
        folds.append(r)
        oof_pred[va] = p
        oof_prob[va] = pr[:, 1]
    agg = {k: {"mean": float(np.mean([f[k] for f in folds])),
               "std": float(np.std([f[k] for f in folds], ddof=1))}
           for k in folds[0]}
    return {"aggregate": agg, "oof_pred": oof_pred.tolist(), "oof_prob": oof_prob.tolist()}


def param_count(factory):
    return int(sum(p.numel() for p in factory().parameters()))


def main():
    index = json.load(open(E.BUNDLE / "cohort_index.json"))
    slide_img = np.load(D / "integrated_full_legacy499" / "X_img_uni.npy")
    tok = np.load(E.BUNDLE / "text_tokens.npz")
    cls = tok["cls"]
    patches = np.load(E.BUNDLE / "img_patches.npz")
    patch_list = [patches["case_%d" % i] for i in range(index["n_base"])]
    masks = {"full": np.ones(index["n_base"], bool),
             "wide": np.array(index["wide_mask"], bool),
             "strict": np.array(index["strict_mask"], bool)}

    # ---- parameter counts -------------------------------------------------
    counts = {
        "image_only": param_count(ImageOnly),
        "text_only": param_count(TextOnly),
        "late_fusion": param_count(LateFusion),
        "gated_bilinear": param_count(GatedBilinearFusion),
        "true_cross_attention": param_count(TrueCrossAttentionFusion),
    }
    print("=== parameter counts ===")
    for k, v in counts.items():
        print("  %-22s %d" % (k, v))

    # ---- arms under test ---------------------------------------------------
    # make_loader yields (X_img, X_text, patches, patch_mask, tokens, text_mask, y),
    # so text tokens are batch index 4 and their mask index 5.
    ARMS = {
        "text_only_tokmean": (
            lambda: TextOnly(),
            lambda m, b: m(masked_mean(b[4], b[5]).to(E.DEVICE))),
        "image_only_patchmean": (
            lambda: ImageOnly(),
            lambda m, b: m(masked_mean(b[2], b[3]).to(E.DEVICE))),
        "true_ca_cls": (
            lambda: CLSCrossAttention(),
            lambda m, b: m(b[2].to(E.DEVICE), b[3].to(E.DEVICE),
                           b[4].to(E.DEVICE), b[5].to(E.DEVICE))),
    }

    results = {}
    for cname, mask in masks.items():
        c = E.build_cohort(index, mask, slide_img, cls, patch_list)
        print("\n=== cohort %s: n=%d ===" % (cname, len(c["y"])))
        results[cname] = {}
        for arm, (factory, fn) in ARMS.items():
            results[cname][arm] = {}
            for seed in SEEDS:
                r = run_cv(fn, factory, c, seed)
                results[cname][arm][str(seed)] = r
                a = r["aggregate"]
                print("  %-22s seed=%-4d Acc=%.3f MacroF1=%.3f HighF1=%.3f AUC=%.3f" % (
                    arm, seed, a["accuracy"]["mean"], a["macro_f1"]["mean"],
                    a["high_f1"]["mean"], a["auc"]["mean"]))
        json.dump({"params": counts, "results": results},
                  open(OUT / "attention_v2_rep_match.json", "w"), indent=2)

    print("\nSaved attention_v2_rep_match.json")
    print("DONE")


if __name__ == "__main__":
    main()
