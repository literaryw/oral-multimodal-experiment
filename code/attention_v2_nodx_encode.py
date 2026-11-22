"""Encode the no-diagnosis clinical template for the four-cell comparison.

The manuscript's "No-diagnosis (TNM + demographics)" text is rebuilt from the stored
narrative with the primary-diagnosis block removed, then re-encoded from scratch. The
diagnosis is not merely masked at the classifier: removing it changes the rendered token
sequence, so the encoder never sees the field.

Gates
-----
* the full template reproduced from the parsed fields must match the stored narrative for
  every case (499/499);
* no case may fall back to the generic string, which mentions carcinoma and would reinsert
  a diagnosis-like token;
* re-encoding the full template must reproduce the stored token mask exactly and the stored
  token values within 1e-5.

Outputs -> data/TCGA-HNSC/attention_v2/
    text_tokens_nodx.npz        tokens (499, Lmax, 768), mask (499, Lmax), cls (499, 768)
    text_tokens_nodx_meta.json  encoding settings, shapes, gates, sha256
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from attention_v2_aux import parse, variant_text  # noqa: E402

BASE = Path(__file__).resolve().parent.parent
D = BASE / "data" / "TCGA-HNSC"
BUNDLE = D / "attention_v2"
MODEL_NAME = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
MAX_LEN, BATCH = 256, 32


def encode(texts):
    """Token-level encoding with one global right-pad width (as in prep_attention_v2)."""
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()
    all_tok, all_mask, all_cls = [], [], []
    with torch.no_grad():
        for i in range(0, len(texts), BATCH):
            enc = tok(texts[i:i + BATCH], padding=True, truncation=True,
                      max_length=MAX_LEN, return_tensors="pt")
            out = model(**enc)
            all_tok.append(out.last_hidden_state.numpy())
            all_mask.append(enc["attention_mask"].numpy())
            all_cls.append(out.last_hidden_state[:, 0, :].numpy())
    lmax = max(b.shape[1] for b in all_tok)
    tokens = np.concatenate([np.pad(b, ((0, 0), (0, lmax - b.shape[1]), (0, 0)))
                             for b in all_tok], axis=0)
    mask = np.concatenate([np.pad(b, ((0, 0), (0, lmax - b.shape[1])))
                           for b in all_mask], axis=0)
    return (tokens.astype(np.float32), mask.astype(np.int8),
            np.concatenate(all_cls, axis=0).astype(np.float32))


def main():
    index = json.load(open(BUNDLE / "cohort_index.json"))
    ids = index["case_ids"]
    meta = pd.read_csv(D / "integrated_full_legacy499" / "text_metadata_499era_ageyears.csv")
    meta = meta.drop_duplicates("case_id", keep="first").set_index("case_id")

    parsed, roundtrip = [], 0
    for case in ids:
        raw = str(meta.loc[case, "clinical_text"]).strip()
        p = parse(raw)
        assert p is not None, "unparsed text for %s: %s" % (case, raw[:120])
        if variant_text(p, "full") == raw:
            roundtrip += 1
        parsed.append(p)
    print("parser round-trip: %d / %d" % (roundtrip, len(ids)))
    assert roundtrip == len(ids), "template does not reconstruct every stored text"

    empty = sum(1 for p in parsed if not p["tnm"] and not p["demo"])
    print("cases with no TNM and no demographics: %d" % empty)
    assert empty == 0, "generic fallback would reinsert a diagnosis-like string"

    full_texts = [variant_text(p, "full") for p in parsed]
    nodx_texts = [variant_text(p, "tnm_demo") for p in parsed]
    print("example no-diagnosis text: %s" % nodx_texts[0])

    print("\n=== encoding full template (gate) ===")
    tokens, mask, _ = encode(full_texts)
    stored = np.load(BUNDLE / "text_tokens.npz")
    assert mask.shape == stored["mask"].shape and np.array_equal(mask, stored["mask"]), \
        "token mask differs from the stored bundle"
    delta = float(np.abs(tokens - stored["tokens"]).max())
    print("  mask identical; token max|delta| = %.3e" % delta)
    assert delta < 1e-5, "full-template tokens do not reproduce the stored bundle"

    print("\n=== encoding no-diagnosis template ===")
    nd_tokens, nd_mask, nd_cls = encode(nodx_texts)
    lengths = nd_mask.sum(1)
    print("  tokens %s  mask %s  Lmax=%d  length min/median/max = %d/%d/%d"
          % (nd_tokens.shape, nd_mask.shape, nd_tokens.shape[1], lengths.min(),
             int(np.median(lengths)), lengths.max()))
    assert not (nd_mask.sum(1) == 0).any(), "empty no-diagnosis sequence"

    out = BUNDLE / "text_tokens_nodx.npz"
    np.savez_compressed(out, tokens=nd_tokens, mask=nd_mask, cls=nd_cls)
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    meta_out = {
        "model_name": MODEL_NAME,
        "max_len": MAX_LEN,
        "batch": BATCH,
        "n": int(len(ids)),
        "row_order": "cohort_index.case_ids",
        "text_template": "TNM staging: <tnm>. Patient: <demo>.",
        "source": "integrated_full_legacy499/text_metadata_499era_ageyears.csv",
        "roundtrip": "%d/%d" % (roundtrip, len(ids)),
        "empty_tnm_and_demo": empty,
        "full_gate_max_abs_delta": delta,
        "full_gate_mask_identical": True,
        "nodx_length_min_median_max": [int(lengths.min()), int(np.median(lengths)),
                                       int(lengths.max())],
        "sha256": digest,
    }
    json.dump(meta_out, open(BUNDLE / "text_tokens_nodx_meta.json", "w"), indent=2)
    print("\nwrote %s  sha256=%s" % (out, digest))


if __name__ == "__main__":
    main()
