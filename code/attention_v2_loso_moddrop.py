"""
R16-fix: leave-one-anatomic-group-out validation and the modality-dropout ablation,
re-run on the unified features and with the genuine cross-attention arm.

Outputs -> experiment_outputs/attention_v2_loso_moddrop.json
"""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import attention_v2_experiments as E  # noqa: E402
from attention_v2_models import (GatedBilinearFusion, TrueCrossAttentionFusion)  # noqa: E402

BASE = Path(__file__).resolve().parent.parent
D = BASE / "data" / "TCGA-HNSC"
BUNDLE = D / "attention_v2"
OUT = D / "experiment_outputs"

SITE_GROUPS = {
    "Tongue, NOS": 0, "Border of tongue": 0, "Ventral surface of tongue, NOS": 0,
    "Floor of mouth, NOS": 1, "Anterior floor of mouth": 1,
    "Mouth, NOS": 2, "Cheek mucosa": 2,
    "Gum, NOS": 3, "Lower gum": 3, "Upper Gum": 3,
    "Hard palate": 3, "Lip, NOS": 3, "Retromolar area": 3,
}
GROUP_NAMES = ["Tongue", "Floor of Mouth", "Buccal+Mouth", "Gingiva+Palate+Lip"]


class ModDropCA(TrueCrossAttentionFusion):
    """True cross-attention with modality dropout (training only).

    R18 fix (P2-3): the dropout decision is drawn per *sample*, not per batch.  The
    earlier implementation used a single `torch.rand(1)` for the whole batch, so an
    entire mini-batch either dropped a modality together or kept both, which is not
    the data-augmentation behaviour modality dropout is meant to implement.
    """

    def __init__(self, drop_rate=0.3, **kw):
        super().__init__(**kw)
        self.drop_rate = drop_rate

    def forward(self, patches, patch_mask, text_tokens, text_mask):
        if self.training:
            b = patches.shape[0]
            dev = patches.device
            drop = torch.rand(b, device=dev) < self.drop_rate
            if bool(drop.any()):
                drop_img = drop & (torch.rand(b, device=dev) < 0.5)
                drop_txt = drop & ~drop_img
                if bool(drop_img.any()):
                    patches, patch_mask = patches.clone(), patch_mask.clone()
                    patches[drop_img] = 0
                    patch_mask[drop_img] = 0
                    patch_mask[drop_img, 0] = 1   # keep one dummy query so pooling stays defined
                if bool(drop_txt.any()):
                    text_tokens, text_mask = text_tokens.clone(), text_mask.clone()
                    text_tokens[drop_txt] = 0
                    text_mask[drop_txt] = 0
                    text_mask[drop_txt, 0] = 1
        return super().forward(patches, patch_mask, text_tokens, text_mask)


def metrics(pred, labels, prob):
    return {"accuracy": float(accuracy_score(labels, pred)),
            "macro_f1": float(f1_score(labels, pred, average="macro")),
            "low_f1": float(f1_score(labels, pred, average=None, labels=[0, 1])[0]),
            "high_f1": float(f1_score(labels, pred, average=None, labels=[0, 1])[1]),
            "auc": float(roc_auc_score(labels, prob))}


def fit_predict(arm, c, tr, va, model=None, seed=42):
    """Train on `tr`, return (predictions, labels, high-class prob) on `va`.

    R18 P0-1: early stopping selects the best epoch on a stratified inner split of `tr`;
    `va` is only ever touched by the final evaluation.
    """
    fit_idx, sel_idx = E.inner_split(c["y"], tr, seed)
    tr_loader = E.make_loader(c, fit_idx, True)
    sel_loader = E.make_loader(c, sel_idx, False)
    va_loader = E.make_loader(c, va, False)
    model = (model if model is not None else E.make_model(arm)).to(E.DEVICE)
    yt = c["y"][fit_idx]
    counts = np.bincount(yt, minlength=2)
    alpha = torch.FloatTensor([1.0, len(yt) / max(counts.max(), 1)]).to(E.DEVICE)
    criterion = E.FocalLoss(alpha=alpha, gamma=2.0)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=20, T_mult=2)

    best_f1, best_state, no_improve = -1.0, None, 0
    for _ in range(E.EPOCHS):
        model.train()
        for batch in tr_loader:
            opt.zero_grad()
            logits, _ = E.forward_arm(arm, model, batch)
            criterion(logits, batch[-1].to(E.DEVICE)).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        p, l, _ = E.evaluate(arm, model, sel_loader)
        vf1 = f1_score(l, p, average="macro")
        if vf1 > best_f1:
            best_f1, best_state, no_improve = vf1, copy.deepcopy(model.state_dict()), 0
        else:
            no_improve += 1
            if no_improve >= E.PATIENCE:
                break
    model.load_state_dict(best_state)
    pred, labels, prob = E.evaluate(arm, model, va_loader)
    return pred, labels, prob[:, 1]


def main():
    index = json.load(open(BUNDLE / "cohort_index.json"))
    X_img_all = np.load(D / "integrated_full_legacy499" / "X_img_uni.npy")
    cls_all = np.load(BUNDLE / "text_tokens.npz")["cls"]
    pat = np.load(BUNDLE / "img_patches.npz")
    patch_list = [pat["case_%d" % i] for i in range(index["n_base"])]
    strict_mask = np.array(index["strict_mask"], bool)
    results = {}

    # ================= LOSO on the strict cohort =================
    rows = np.where(strict_mask)[0]
    c = E.build_cohort(index, strict_mask, X_img_all, cls_all, patch_list)
    site = pd.read_csv(D / "clinical_data_oral.csv").drop_duplicates("case_id", keep="first")
    site_map = dict(zip(site["case_id"], site["primary_site"]))
    groups = np.array([SITE_GROUPS.get(site_map.get(index["case_ids"][r]), -1) for r in rows])
    bad = {site_map.get(index["case_ids"][r]) for r in rows if groups[list(rows).index(r)] < 0}
    assert (groups >= 0).all(), "unmapped subsites: %s" % bad
    print("LOSO group sizes: %s" % [int((groups == g).sum()) for g in range(4)])

    print("\n=== leave-one-anatomic-group-out (strict cohort, seed 42) ===")
    loso = {}
    for arm in E.ARMS:
        E.seed_everything(42)
        per_group = {}
        for g in range(4):
            tr, va = np.where(groups != g)[0], np.where(groups == g)[0]
            pred, labels, prob = fit_predict(arm, c, tr, va)
            per_group[GROUP_NAMES[g]] = metrics(pred, labels, prob)
        loso[arm] = per_group
        print("  %-22s " % arm + "  ".join(
            "%s=%.3f" % (GROUP_NAMES[g][:6], per_group[GROUP_NAMES[g]]["accuracy"]) for g in range(4)))
    results["loso"] = loso

    # ================= modality dropout (full cohort) =================
    print("\n=== modality dropout, full cohort (n=499, seed 42) ===")
    cf = E.build_cohort(index, np.ones(index["n_base"], bool), X_img_all, cls_all, patch_list)
    md = {}
    for tag, arm, factory in [
            ("true_ca_baseline", "true_cross_attention", TrueCrossAttentionFusion),
            ("true_ca_moddrop", "true_cross_attention", lambda: ModDropCA(drop_rate=0.3)),
            ("gated_bilinear_baseline", "gated_bilinear", GatedBilinearFusion)]:
        E.seed_everything(42)
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        P, L, R = [], [], []
        for tr, va in kf.split(cf["X_img"], cf["y"]):
            pred, labels, prob = fit_predict(arm, cf, tr, va, model=factory())
            P.extend(pred); L.extend(labels); R.extend(prob)
        md[tag] = metrics(np.array(P), np.array(L), np.array(R))
        print("  %-26s Acc=%.3f MacroF1=%.3f HighF1=%.3f AUC=%.3f" % (
            tag, md[tag]["accuracy"], md[tag]["macro_f1"], md[tag]["high_f1"], md[tag]["auc"]))
    results["modality_dropout"] = md

    json.dump(results, open(OUT / "attention_v2_loso_moddrop.json", "w"), indent=2)
    print("\nSaved attention_v2_loso_moddrop.json")


if __name__ == "__main__":
    main()
