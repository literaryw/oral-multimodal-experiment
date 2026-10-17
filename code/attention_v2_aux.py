"""
R16-fix: re-run the auxiliary experiments on the unified (gender-inclusive) text.

Affected by the cohort-consistency fix: the strict-cohort text previously omitted the
gender field (its source table had a 100%-null gender column), so the published
"Demographics-only (Age + Gender)" variant on the strict subset actually contained age
only.  Everything here is rebuilt from one text source for all three cohorts.

Structured fields are recovered by parsing the stored clinical narrative, which follows
a fixed template:
    Diagnosis: <d>. TNM staging: T: <t>; N: <n>; M: <m>; Overall stage: <s>. Patient: age <a>, <g>.

Outputs -> experiment_outputs/attention_v2_aux.json
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from attention_v2_experiments import build_cohort, run_cv  # noqa: E402

BASE = Path(__file__).resolve().parent.parent
D = BASE / "data" / "TCGA-HNSC"
BUNDLE = D / "attention_v2"
OUT = D / "experiment_outputs"
MODEL_NAME = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"

TXT_RE = re.compile(
    r"Diagnosis:\s*(?P<dx>.*?)\.\s*"
    r"(?:TNM staging:\s*(?P<tnm>.*?)\.\s*)?"
    r"Patient:\s*(?P<demo>.*?)\.\s*$", re.S)


def parse(text):
    m = TXT_RE.match(str(text).strip())
    if not m:
        return None
    g = m.groupdict()
    return {"dx": (g["dx"] or "").strip(),
            "tnm": (g["tnm"] or "").strip(),
            "demo": (g["demo"] or "").strip()}


# R18 fix (P0-3): the variant previously labelled "TNM only" also carried the diagnosis,
# so it could not support a TNM-only claim.  The set is now explicit and separates the
# diagnosis from the staging fields.
#   full      Diagnosis + TNM + demographics
#   dx_tnm    Diagnosis + TNM          (what the old "tnm_only" actually was)
#   tnm_only  TNM staging only         (true TNM-only)
#   dx_only   Diagnosis only
#   demo_only Age + gender
#   no_tnm    Diagnosis + demographics
#   tnm_demo  TNM + demographics       (R19 P0-4: clinical-only, no diagnosis string)
VARIANTS = ["full", "dx_tnm", "tnm_only", "dx_only", "demo_only", "no_tnm", "tnm_demo"]


def variant_text(p, kind):
    parts = []
    if kind in ("full", "dx_tnm", "dx_only", "no_tnm"):
        if p["dx"]:
            parts.append("Diagnosis: %s." % p["dx"])
    if kind in ("full", "dx_tnm", "tnm_only", "tnm_demo"):
        if p["tnm"]:
            parts.append("TNM staging: %s." % p["tnm"])
    if kind in ("full", "no_tnm", "demo_only", "tnm_demo"):
        if p["demo"]:
            parts.append("Patient: %s." % p["demo"])
    return " ".join(parts) if parts else "Oral squamous cell carcinoma."


def encode(texts):
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    mdl = AutoModel.from_pretrained(MODEL_NAME)
    mdl.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), 32):
            enc = tok(texts[i:i + 32], padding=True, truncation=True,
                      max_length=256, return_tensors="pt")
            out.append(mdl(**enc).last_hidden_state[:, 0, :].numpy())
    return np.concatenate(out, axis=0)


def main():
    index = json.load(open(BUNDLE / "cohort_index.json"))
    y_all = np.array(index["y_binary"])
    masks = {"full": np.ones(len(y_all), bool),
             "wide": np.array(index["wide_mask"], bool),
             "strict": np.array(index["strict_mask"], bool)}
    X_img_all = np.load(D / "integrated_full_legacy499" / "X_img_uni.npy")
    tokz = np.load(BUNDLE / "text_tokens.npz")
    cls_all = tokz["cls"]
    pat = np.load(BUNDLE / "img_patches.npz")
    patch_list = [pat["case_%d" % i] for i in range(index["n_base"])]

    # ---- recover structured fields and check the parser round-trips the full text ----
    import pandas as pd
    meta = pd.read_csv(D / "integrated_full_legacy499" / "text_metadata_499era_ageyears.csv")
    meta = meta.drop_duplicates("case_id", keep="first").set_index("case_id")
    parsed, ok = [], 0
    for c in index["case_ids"]:
        raw = str(meta.loc[c, "clinical_text"])
        p = parse(raw)
        assert p is not None, "unparsed text for %s: %s" % (c, raw[:120])
        if variant_text(p, "full") == raw.strip():
            ok += 1
        parsed.append(p)
    print("parser round-trip: %d / %d texts reconstructed exactly" % (ok, len(parsed)))
    n_gen = sum(1 for p in parsed if "," in p["demo"])
    print("gender present in parsed demographics: %d / %d" % (n_gen, len(parsed)))

    results = {}

    # ---- encode every variant ONCE at base-499 level, then slice per cohort ----
    # BiomedBERT's CLS output is batch/padding dependent at the ~5e-06 level, and that
    # difference alone shifts downstream accuracy by several points.  Encoding the whole
    # base set in one fixed pass keeps every cohort on an identical encoding scheme.
    print("\n=== encoding text variants at base level (n=%d) ===" % len(parsed))
    variant_feats = {}
    for kind in VARIANTS:
        base_texts = [variant_text(p, kind) for p in parsed]
        variant_feats[kind] = encode(base_texts)
        print("  %-9s encoded %s | %s" % (kind, variant_feats[kind].shape, base_texts[0][:70]))
    dev = float(np.abs(variant_feats["full"] - cls_all).max())
    print("  consistency check: 'full' re-encoding vs stored CLS max|d| = %.3e" % dev)
    assert dev < 1e-5, "variant 'full' must reproduce the stored text features"

    # ================= text ablation + TNM sensitivity =================
    for coh in ["full", "strict"]:
        rows = np.where(masks[coh])[0]
        y = y_all[rows]
        base_c = build_cohort(index, masks[coh], X_img_all, cls_all, patch_list)
        assert np.array_equal(base_c["y"], y)
        print("\n=== %s cohort text variants (n=%d) ===" % (coh, len(y)))
        entry = {}
        for kind in VARIANTS:
            texts = [variant_text(parsed[r], kind) for r in rows]
            c = dict(base_c)
            c["X_text"] = variant_feats[kind][rows].astype(np.float32)
            r = run_cv("text_only", c, 42)
            a = r["aggregate"]
            entry[kind] = {"aggregate": a, "sample_text": texts[0],
                           "oof_pred": r["oof_pred"], "oof_prob": r["oof_prob"]}
            print("  %-9s Acc=%.3f+-%.3f MacroF1=%.3f HighF1=%.3f AUC=%.3f | %s" % (
                kind, a["accuracy"]["mean"], a["accuracy"]["std"],
                a["macro_f1"]["mean"], a["high_f1"]["mean"], a["auc"]["mean"],
                texts[0][:78]))
        results["text_ablation_" + coh] = entry
        json.dump(results, open(OUT / "attention_v2_aux.json", "w"), indent=2)

    # ================= mutual information =================
    from sklearn.feature_selection import mutual_info_regression
    from sklearn.preprocessing import StandardScaler
    print("\n=== cross-modality MI (unified features) ===")
    mi = {}
    for coh in ["full", "wide", "strict"]:
        rows = np.where(masks[coh])[0]
        y = y_all[rows]
        Xi = StandardScaler().fit_transform(X_img_all[rows])
        Xt = StandardScaler().fit_transform(cls_all[rows])
        K = 10
        cross = float(np.mean([mutual_info_regression(Xt[:, :K], Xi[:, i],
                                                      random_state=42).mean()
                               for i in range(K)]))
        mi_img = float(mutual_info_regression(Xi, y, random_state=42).mean())
        mi_txt = float(mutual_info_regression(Xt, y, random_state=42).mean())
        # R19: label terms estimated on the SAME first-K dimensions as the cross term,
        # so the ratio compares quantities computed on matched feature counts.
        mi_img_m = float(mutual_info_regression(Xi[:, :K], y, random_state=42).mean())
        mi_txt_m = float(mutual_info_regression(Xt[:, :K], y, random_state=42).mean())
        mi[coh] = {"cross_mi": cross,
                   "img_label_mi": mi_img, "text_label_mi": mi_txt,
                   "img_label_mi_matched": mi_img_m, "text_label_mi_matched": mi_txt_m,
                   "ratio_all_dims": cross / mi_img if mi_img else float("nan"),
                   "ratio": cross / mi_img_m if mi_img_m else float("nan"),
                   "matched_dimensions": K}
        print("  %-7s cross=%.4f  img-label=%.4f (matched %.4f)  text-label=%.4f (matched %.4f)  ratio=%.2fx" % (
            coh, cross, mi_img, mi_img_m, mi_txt, mi_txt_m, mi[coh]["ratio"]))
    results["mutual_information"] = mi

    json.dump(results, open(OUT / "attention_v2_aux.json", "w"), indent=2)
    print("\nSaved attention_v2_aux.json")


if __name__ == "__main__":
    main()
