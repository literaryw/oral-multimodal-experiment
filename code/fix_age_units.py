"""
R18 fix (P0-2): the clinical narrative carried TCGA's `age_at_diagnosis`, which is
recorded in *days*.  The template rendered it verbatim ("Patient: age 24322, male."),
so BiomedBERT was fed a semantically meaningless number instead of a human age.

This script rewrites the age field in the stored clinical text to whole years
(round(days / 365.25)) and writes a NEW metadata file; the legacy CSV is left in place
as the audit artefact so the change is diffable.

Input  -> data/TCGA-HNSC/integrated_full_legacy499/text_metadata_499era.csv
Output -> data/TCGA-HNSC/integrated_full_legacy499/text_metadata_499era_ageyears.csv

Downstream consumers (prep_attention_v2.py, attention_v2_aux.py) must point at the new
file; everything else about the narrative is byte-identical.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
FULL_DIR = BASE / "data" / "TCGA-HNSC" / "integrated_full_legacy499"
SRC = FULL_DIR / "text_metadata_499era.csv"
DST = FULL_DIR / "text_metadata_499era_ageyears.csv"

DAYS_PER_YEAR = 365.25
AGE_RE = re.compile(r"\bage (\d+)\b")


def convert(text):
    """Return (new_text, days_or_None, years_or_None)."""
    m = AGE_RE.search(str(text))
    if not m:
        return str(text), None, None
    days = int(m.group(1))
    years = int(round(days / DAYS_PER_YEAR))
    return AGE_RE.sub("age %d" % years, str(text), count=1), days, years


def main():
    df = pd.read_csv(SRC)
    print("source: %s  rows=%d  unique case_id=%d" % (SRC.name, len(df), df["case_id"].nunique()))

    days_seen, years_seen, n_changed, n_noage = [], [], 0, 0
    out_text = []
    for raw in df["clinical_text"]:
        new, days, years = convert(raw)
        if days is None:
            n_noage += 1
        else:
            days_seen.append(days)
            years_seen.append(years)
            if new != raw:
                n_changed += 1
        out_text.append(new)

    df["clinical_text"] = out_text

    days_seen = np.array(days_seen)
    years_seen = np.array(years_seen)
    print("rows with an age field : %d / %d  (%d without)" % (len(days_seen), len(df), n_noage))
    print("rows rewritten         : %d" % n_changed)
    print("days  range: min=%d median=%d max=%d" % (
        days_seen.min(), int(np.median(days_seen)), days_seen.max()))
    print("years range: min=%d median=%d max=%d" % (
        years_seen.min(), int(np.median(years_seen)), years_seen.max()))

    # Plausibility gate: human ages at OSCC diagnosis.
    assert years_seen.min() >= 10 and years_seen.max() <= 110, (
        "implausible age range after conversion: %d..%d" % (years_seen.min(), years_seen.max()))

    # Every rewritten text must differ from the original only in the age token.
    for raw, new in zip(pd.read_csv(SRC)["clinical_text"], df["clinical_text"]):
        assert AGE_RE.sub("age X", str(raw)) == AGE_RE.sub("age X", str(new)), (
            "conversion changed more than the age token: %r -> %r" % (raw, new))

    df.to_csv(DST, index=False)
    print("\nwrote %s" % DST)
    print("sample: %s" % df["clinical_text"].iloc[0])
    print("DONE")


if __name__ == "__main__":
    main()
