"""
Full-cohort (n=499) text ablation -- unified protocol rerun with REAL
text-content reconstruction (no feature-space proxy).

Replaces the legacy `text_ablation_results.json` (old seed protocol, method
no longer traceable) with a per-variant seed-reset rerun on the restored full
HNSCC cohort (n=499):

- Full (diagnosis + TNM + demographics): reuses X_text_biomedbert.npy itself,
  so this row is bit-identical to the full-cohort main-table Text-Only entry.
- TNM only / Demographics only: extracted as verbatim substrings of the
  cohort-matched clinical text that produced the original features (recovered
  from the archived text_metadata.csv, same 506-row encoding run), re-encoded
  with the same BiomedBERT checkpoint / CLS pooling / max_length=256 used by
  the original pipeline (extract_text_features.py), then trained with the
  exact same pooled-CV protocol as multimodal_fusion_v2.py.
- RNG is reset before each variant (execution-order independent).
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE.parent / "data" / "TCGA-HNSC" / "integrated_full_legacy499"
out_dir = BASE.parent / "data" / "TCGA-HNSC" / "experiment_outputs"

# Reuse model definitions + pooled CV protocol from multimodal_fusion_v2
src = (BASE / "multimodal_fusion_v2.py").read_text(encoding="utf-8")
top_src = src.split('if __name__ == "__main__"')[0]
ns = {"__name__": "mf2_lib", "__file__": str(BASE / "multimodal_fusion_v2.py")}
exec(compile(top_src, "multimodal_fusion_v2.py", "exec"), ns)

from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score

X_text_full = np.load(DATA_DIR / "X_text_biomedbert.npy")
y_4class = np.load(DATA_DIR / "y.npy")
y_binary = np.where(np.isin(y_4class, [0, 1]), 0, 1)
case_labels = pd.read_csv(DATA_DIR / "case_labels.csv")
assert (case_labels["label"].values == y_4class).all(), "case_labels order mismatch vs y.npy"

# Recover the exact clinical text that produced the original features.
# text_metadata.csv holds the 506 texts encoded by extract_text_features.py;
# the original integration used a dict lookup where the LAST occurrence of a
# duplicated case_id wins, so we deduplicate with keep='last' to mirror it.
tm = pd.read_csv(DATA_DIR / "text_metadata_499era.csv").drop_duplicates("case_id", keep="last")
missing = case_labels["case_id"].values[~np.isin(case_labels["case_id"].values, tm["case_id"].values)]
print(f"Full cohort: n={len(y_binary)}; texts recovered for {len(case_labels) - len(missing)}/{len(case_labels)} cases"
      + (f" (missing: {list(missing[:5])})" if len(missing) else ""))
texts_full = tm.set_index("case_id").loc[case_labels["case_id"].values, "clinical_text"].tolist()


def extract_substring(texts, pattern, label):
    out, n_fb = [], 0
    for t in texts:
        m = re.search(pattern, str(t))
        if m:
            out.append(m.group(0))
        else:
            out.append("Oral squamous cell carcinoma.")
            n_fb += 1
    print(f"  {label}: {len(out) - n_fb} extracted, {n_fb} fallback")
    return out


print("Extracting ablation text variants from the recovered clinical text...")
texts_tnm = extract_substring(texts_full, r"TNM staging: [^.]*\.", "TNM only")
texts_demo = extract_substring(texts_full, r"Patient: [^.]*\.", "Demographics only")
print(f"Full example    : {texts_full[0][:100]}")
print(f"TNM-only example: {texts_tnm[0][:100]}")
print(f"Demo-only example: {texts_demo[0][:100]}")

DEVICE = ns["DEVICE"]
from transformers import AutoTokenizer, AutoModel
model_name = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
encoder = AutoModel.from_pretrained(model_name, local_files_only=True)
encoder.eval().to(DEVICE)


def encode(texts, label):
    feats = []
    bs = 32
    for i in range(0, len(texts), bs):
        inputs = tokenizer(texts[i:i + bs], padding=True, truncation=True,
                           max_length=256, return_tensors="pt")  # 256 = original pipeline setting
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            feats.append(encoder(**inputs).last_hidden_state[:, 0, :].cpu().numpy())
    arr = np.concatenate(feats, axis=0).astype(np.float32)
    print(f"  encoded {label}: {arr.shape}")
    return arr


print("\nEncoding ablation text variants with BiomedBERT (CLS pooling, max_length=256)...")
X_tnm = encode(texts_tnm, "TNM only")
X_demo = encode(texts_demo, "Demographics only")
assert X_tnm.shape == X_text_full.shape and X_demo.shape == X_text_full.shape

# Sanity check: re-encoding the recovered full texts must reproduce the
# original features (cosine similarity ~ 1.0 per case).
X_full_check = encode(texts_full, "Full (reconstruction check)")
sims = (X_full_check * X_text_full).sum(1) / (np.linalg.norm(X_full_check, axis=1) * np.linalg.norm(X_text_full, axis=1))
print(f"  reconstruction cosine vs original X_text: mean={sims.mean():.6f} min={sims.min():.6f}")
assert sims.min() > 0.999, "BiomedBERT reconstruction deviates from original features"


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
        "confusion_matrix": confusion_matrix(l, p).tolist(),
    }


variants = [
    ("Full (diagnosis + TNM + demographics)", X_text_full),
    ("TNM only", X_tnm),
    ("Demographics only (age, gender)", X_demo),
]

results = {}
for name, X in variants:
    print(f"\n--- {name} (per-variant seed reset) ---")
    torch.manual_seed(ns["RANDOM_SEED"]); np.random.seed(ns["RANDOM_SEED"])
    preds, labels, probs = ns["run_cv"](name, ns["TextOnly"], True, [X], y_binary)
    results[name] = summarize(preds, labels, probs)
    r = results[name]
    print(f"  Acc={r['accuracy']:.3f} MacroF1={r['macro_f1']:.3f} AUC={r['auc']:.3f} "
          f"HighF1={r['high_f1']:.3f}")

results["_metadata"] = {
    "dataset": "TCGA-HNSC full cohort (n=499, Low=370/High=129)",
    "protocol": "per-variant RNG reset (seed=42), pooled 5-fold CV",
    "method": "TNM-only / demographics-only texts are verbatim substrings of the recovered original clinical text (text_metadata.csv), re-encoded with the original BiomedBERT settings (CLS, max_length=256); Full row reuses the main-table text features",
    "full_reconstruction_cosine_mean": round(float(sims.mean()), 6),
}

json.dump(results, open(out_dir / "text_ablation_results.json", "w"), indent=2)
print("\nSaved text_ablation_results.json (full cohort, unified protocol)")
