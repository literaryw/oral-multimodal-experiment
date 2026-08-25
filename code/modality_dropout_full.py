"""
Full-cohort (n=499) modality dropout ablation -- unified protocol rerun.

Replaces the legacy `modality_dropout_results.json` (old seed protocol, batch-
level coin-flip dropout) with a per-model seed-reset rerun on the restored
full HNSCC cohort (n=499, Low=370 / High=129):

- Baseline: CrossAttentionFusion trained with the exact same pooled-CV
  protocol as multimodal_fusion_v2.py (StratifiedKFold seed 42, weighted
  sampler, FocalLoss, AdamW, cosine restarts, early stopping on val macro-F1),
  so the baseline is bit-identical to the full-cohort main-table
  cross-attention entry.
- ModDrop: identical protocol plus per-sample modality dropout (p=0.3)
  during training, matching the oral-subset implementation
  (modality_dropout_oral.py) reviewed in round 7.
- RNG is reset before each configuration (execution-order independent).
"""
import json
from pathlib import Path

import numpy as np
import torch

BASE = Path(__file__).resolve().parent

# Reuse model definitions + pooled CV protocol from multimodal_fusion_v2
src = (BASE / "multimodal_fusion_v2.py").read_text(encoding="utf-8")
top_src = src.split('if __name__ == "__main__"')[0]
ns = {"__name__": "mf2_lib", "__file__": str(BASE / "multimodal_fusion_v2.py")}
exec(compile(top_src, "multimodal_fusion_v2.py", "exec"), ns)

DATA_DIR = ns["DATA_DIR"]
out_dir = BASE.parent / "data" / "TCGA-HNSC" / "experiment_outputs"

from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score

X_img = np.load(DATA_DIR / "X_img_uni.npy")
X_text = np.load(DATA_DIR / "X_text_biomedbert.npy")
y_4class = np.load(DATA_DIR / "y.npy")
y_binary = np.where(np.isin(y_4class, [0, 1]), 0, 1)
print(f"Full cohort: n={len(y_binary)}, Low={int((y_binary==0).sum())}, High={int((y_binary==1).sum())}")


class ModalityDropoutFusion(ns["CrossAttentionFusion"]):
    """Cross-attention fusion with per-sample modality dropout during training."""

    def __init__(self, drop_rate=0.3, **kwargs):
        super().__init__(**kwargs)
        self.drop_rate = drop_rate

    def forward(self, img, text, training=True):
        if training:
            bs = img.size(0)
            drop_img = torch.rand(bs, 1, device=img.device) < self.drop_rate
            drop_text = torch.rand(bs, 1, device=img.device) < self.drop_rate
            img = img.clone(); text = text.clone()
            img[drop_img.expand_as(img)] = 0.0
            text[drop_text.expand_as(text)] = 0.0
        return super().forward(img, text)


def summarize(preds, labels, probs):
    p, l = np.array(preds), np.array(labels)
    rep = classification_report(l, p, target_names=["Low-Grade", "High-Grade"],
                                zero_division=0, output_dict=True)
    return {
        "accuracy": round(float((p == l).mean()), 4),
        "macro_f1": round(float(f1_score(l, p, average="macro")), 4),
        "auc": round(float(roc_auc_score(l, [q[1] for q in probs])), 4),
        "low_f1": round(float(rep["Low-Grade"]["f1-score"]), 4),
        "high_f1": round(float(rep["High-Grade"]["f1-score"]), 4),
        "recall_high": round(float(rep["High-Grade"]["recall"]), 4),
        "precision_high": round(float(rep["High-Grade"]["precision"]), 4),
        "confusion_matrix": confusion_matrix(l, p).tolist(),
    }


results = {}

print("\n=== Cross-Attention baseline (per-model seed reset) ===")
torch.manual_seed(ns["RANDOM_SEED"]); np.random.seed(ns["RANDOM_SEED"])
preds, labels, probs = ns["run_cv"]("CrossAttn baseline (full)", ns["CrossAttentionFusion"],
                                    False, [X_img, X_text], y_binary)
results["Cross-Attention (baseline)"] = summarize(preds, labels, probs)

print("\n=== Cross-Attention + Modality Dropout p=0.3 (per-model seed reset) ===")
torch.manual_seed(ns["RANDOM_SEED"]); np.random.seed(ns["RANDOM_SEED"])
preds, labels, probs = ns["run_cv"]("CrossAttn + ModDrop (full)",
                                    lambda: ModalityDropoutFusion(drop_rate=0.3),
                                    False, [X_img, X_text], y_binary)
results["Cross-Attention + ModDrop"] = summarize(preds, labels, probs)

results["_metadata"] = {
    "dataset": "TCGA-HNSC full cohort (n=499, Low=370/High=129)",
    "protocol": "per-configuration RNG reset (seed=42), pooled 5-fold CV",
    "dropout": "per-sample modality dropout p=0.3 (training only)",
    "baseline_equals_main_table": "identical protocol/data as multimodal_fusion_v2.py CrossAttention",
}

json.dump(results, open(out_dir / "modality_dropout_results.json", "w"), indent=2)
print("\n=== Full-cohort moddrop results ===")
for k in ["Cross-Attention (baseline)", "Cross-Attention + ModDrop"]:
    r = results[k]
    print(f"{k}: Acc={r['accuracy']:.3f} MacroF1={r['macro_f1']:.3f} "
          f"LowF1={r['low_f1']:.2f} HighF1={r['high_f1']:.2f} HighRecall={r['recall_high']:.2f} AUC={r['auc']:.3f}")
print(f"\nSaved modality_dropout_results.json (full cohort, unified protocol)")
