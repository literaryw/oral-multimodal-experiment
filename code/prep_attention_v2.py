"""
R16-fix Phase A: build the feature bundle for the true-cross-attention re-run.

Why this exists
---------------
The original `CrossAttentionFusion` fed `nn.MultiheadAttention` sequences of length 1
(`.unsqueeze(1)`), so the softmax over a single key is identically 1.0 and the module
degenerates into a linear projection of the text vector.  A genuine cross-attention
needs real sequences, so this script re-extracts:

  * token-level BiomedBERT embeddings for the clinical text (K = V side)
  * per-case patch-level UNI features (Q side), aligned to the same row order

Cohorts (all derived from the same 499-case base set):
  full  HNSCC n=499
  wide  oral n=332   (subset whose case_id is in clinical_data.csv)
  strict oral n=246  (subset listed in integrated/case_labels.csv)

Outputs -> data/TCGA-HNSC/attention_v2/
  cohort_index.json          row order + cohort masks + labels for all three cohorts
  text_tokens.npz            padded token embeddings (N, Lmax, 768) + attention mask
  img_patches.npz            per-case patch features keyed by base-set row index

Verification gate: re-encoding the stored clinical text must reproduce the archived
CLS embeddings (cosine ~ 1.000), which is what Section 4.5 of the manuscript claims.

R18 fix (P0-2): the text source is now `text_metadata_499era_ageyears.csv`, in which the
age field is expressed in years rather than TCGA's `age_at_diagnosis` days.  The
"reproduce the archived CLS" gate therefore no longer applies to the age-bearing rows --
it is replaced by a change-confinement gate (the new text must differ from the legacy text
on exactly the rows that carry an age field) plus a reported cosine delta versus the
legacy archive.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "TCGA-HNSC"
FULL_DIR = DATA_DIR / "integrated_full_legacy499"
INT_DIR = DATA_DIR / "integrated"
OUT_DIR = DATA_DIR / "attention_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
MAX_LEN = 256
BATCH = 32


def cos(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


# ---------------------------------------------------------------- cohort index
print("=== cohort index ===")
full_cases = pd.read_csv(FULL_DIR / "case_labels.csv")
full_ids = full_cases["case_id"].tolist()
y4 = np.load(FULL_DIR / "y.npy").ravel()
y_bin = np.where(np.isin(y4, [0, 1]), 0, 1)
assert len(full_ids) == 499 and len(y4) == 499
print("  full  n=%d  labels=%s" % (len(full_ids), np.bincount(y4, minlength=4).tolist()))

wide_clin = pd.read_csv(DATA_DIR / "clinical_data.csv").drop_duplicates(subset=["case_id"], keep="first")
wide_set = set(wide_clin["case_id"])
wide_mask = np.array([c in wide_set for c in full_ids])
assert int(wide_mask.sum()) == 332, int(wide_mask.sum())
print("  wide  n=%d  Low=%d High=%d" % (wide_mask.sum(), int((y_bin[wide_mask] == 0).sum()),
                                        int((y_bin[wide_mask] == 1).sum())))

strict_ids = pd.read_csv(INT_DIR / "case_labels.csv")["case_id"].tolist()
strict_set = set(strict_ids)
strict_mask = np.array([c in strict_set for c in full_ids])
assert int(strict_mask.sum()) == 246, int(strict_mask.sum())
assert strict_set <= set(np.array(full_ids)[wide_mask]), "strict not nested in wide"
print("  strict n=%d  Low=%d High=%d" % (strict_mask.sum(), int((y_bin[strict_mask] == 0).sum()),
                                         int((y_bin[strict_mask] == 1).sum())))

# ------------------------------------------------------------- image patches
print("\n=== image patches ===")
xp = np.load(INT_DIR / "X_img_patches.npz")
patch_keys = ["case_%d" % i for i in range(len(full_ids))]
missing = [k for k in patch_keys if k not in xp]
assert not missing, "missing patch keys: %s" % missing[:5]
counts = np.array([xp[k].shape[0] for k in patch_keys])
print("  n=%d  patches min=%d median=%d max=%d mean=%.1f" % (
    len(patch_keys), counts.min(), int(np.median(counts)), counts.max(), counts.mean()))

# alignment check: mean-pooled patches should resemble the stored slide features
slide_img = np.load(FULL_DIR / "X_img_uni.npy")
pooled = np.stack([xp[k].mean(axis=0) for k in patch_keys])
align = [cos(pooled[i], slide_img[i]) for i in range(len(patch_keys))]
print("  pooled vs stored slide features: cosine min=%.4f mean=%.4f" % (min(align), float(np.mean(align))))
assert np.mean(align) > 0.95, "patch rows are not aligned with the slide-level feature rows"

# ---------------------------------------------------------------- text tokens
print("\n=== token-level text features ===")
meta = pd.read_csv(FULL_DIR / "text_metadata_499era_ageyears.csv").drop_duplicates(subset=["case_id"], keep="first")
text_by_id = dict(zip(meta["case_id"], meta["clinical_text"]))
missing_txt = [c for c in full_ids if c not in text_by_id]
assert not missing_txt, "no clinical text for: %s" % missing_txt[:5]
texts = [str(text_by_id[c]) for c in full_ids]
print("  texts resolved: %d  (sample: %s)" % (len(texts), texts[0][:110]))

# --- R18 P0-2 change-confinement gate: only the age token may differ --------------
legacy = pd.read_csv(FULL_DIR / "text_metadata_499era.csv").drop_duplicates(
    subset=["case_id"], keep="first")
legacy_by_id = dict(zip(legacy["case_id"], legacy["clinical_text"]))
import re as _re
AGE = _re.compile(r"\bage (\d+)\b")
n_diff = 0
for c, new in zip(full_ids, texts):
    old = str(legacy_by_id[c])
    if old != new:
        n_diff += 1
        assert AGE.sub("age X", old) == AGE.sub("age X", new), (
            "text change is not confined to the age token for %s:\n  old=%s\n  new=%s"
            % (c, old[:160], new[:160]))
n_age = sum(1 for c in full_ids if AGE.search(str(legacy_by_id[c])))
print("  age-unit rewrite: %d rows differ from legacy; %d rows carry an age field" % (n_diff, n_age))
assert n_diff == n_age, "expected exactly the age-bearing rows to change (%d vs %d)" % (n_diff, n_age)

tok = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()

all_tok, all_mask, all_cls = [], [], []
with torch.no_grad():
    for i in range(0, len(texts), BATCH):
        batch = texts[i:i + BATCH]
        enc = tok(batch, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
        out = model(**enc)
        all_tok.append(out.last_hidden_state.numpy())
        all_mask.append(enc["attention_mask"].numpy())
        all_cls.append(out.last_hidden_state[:, 0, :].numpy())
        if (i // BATCH) % 5 == 0:
            print("    %d/%d" % (i, len(texts)))

# The tokenizer pads each batch to its own longest member, so batches differ in
# width.  Pad every batch up to the global maximum before stacking.
Lmax = max(b.shape[1] for b in all_tok)
tokens = np.concatenate([np.pad(b, ((0, 0), (0, Lmax - b.shape[1]), (0, 0))) for b in all_tok], axis=0)
mask = np.concatenate([np.pad(b, ((0, 0), (0, Lmax - b.shape[1]))) for b in all_mask], axis=0)
cls = np.concatenate(all_cls, axis=0)             # (N, 768)
Lmax = tokens.shape[1]
print("  tokens %s  mask %s  Lmax=%d" % (tokens.shape, mask.shape, Lmax))
print("  real token lengths: min=%d median=%d max=%d" % (
    mask.sum(1).min(), int(np.median(mask.sum(1))), mask.sum(1).max()))

# --------------------------------------------- legacy-CLS comparison (informational)
print("\n=== legacy CLS comparison (informational; R18 P0-2) ===")
stored_text = np.load(FULL_DIR / "X_text_biomedbert.npy")
gate = [cos(cls[i], stored_text[i]) for i in range(len(full_ids))]
print("  age-corrected CLS vs legacy (days) archive: cosine min=%.6f mean=%.6f" % (
    min(gate), float(np.mean(gate))))
print("  NOTE: <1.0 is expected and correct -- the age token changed for %d / %d cases."
      % (n_diff, len(full_ids)))
worst = int(np.argmin(gate))
print("  largest shift: %s cosine=%.6f" % (full_ids[worst], gate[worst]))
print("    legacy: %s" % str(legacy_by_id[full_ids[worst]])[:160])
print("    new   : %s" % texts[worst][:160])
np.save(FULL_DIR / "X_text_biomedbert_ageyears.npy", cls.astype(np.float32))

# -------------------------------------------------------------------- save
print("\n=== saving bundle ===")
np.savez_compressed(OUT_DIR / "text_tokens.npz",
                    tokens=tokens.astype(np.float32), mask=mask.astype(np.int8),
                    cls=cls.astype(np.float32))
np.savez_compressed(OUT_DIR / "img_patches.npz", **{k: xp[k] for k in patch_keys})

index = {
    "model_name": MODEL_NAME,
    "max_len": MAX_LEN,
    "n_base": int(len(full_ids)),
    "case_ids": full_ids,
    "y4": y4.tolist(),
    "y_binary": y_bin.tolist(),
    "wide_mask": wide_mask.tolist(),
    "strict_mask": strict_mask.tolist(),
    "n_patches": counts.tolist(),
    "text_source": "integrated_full_legacy499/text_metadata_499era_ageyears.csv",
    "text_source_note": "R18 P0-2: age rendered in years (round(days/365.25)); legacy days-based file kept as text_metadata_499era.csv",
    "note": "row i of every array corresponds to case_ids[i]; cohorts are boolean masks over this base order",
}
json.dump(index, open(OUT_DIR / "cohort_index.json", "w"), indent=2)
print("  wrote text_tokens.npz / img_patches.npz / cohort_index.json -> %s" % OUT_DIR)
print("DONE")
