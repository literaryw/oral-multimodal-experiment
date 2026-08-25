"""
TNM 泄漏敏感性分析
构建 3 种文本变体 → BiomedBERT 编码 → Text-Only 5-fold CV
- Full: Diagnosis + TNM + demographics (现有)
- No-TNM: Diagnosis + demographics (去掉所有 T/N/M/stage)
- Demographics-only: 仅 age + gender
"""
import sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
import json

BASE_DIR = Path(".").resolve().parent
DATA_DIR = BASE_DIR / "data" / "TCGA-HNSC"
OUTPUT_DIR = DATA_DIR / "experiment_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED); np.random.seed(RANDOM_SEED)

# Load clinical data
clin_df = pd.read_csv(DATA_DIR / "clinical_data_oral.csv")
print(f"Clinical data: {len(clin_df)} cases")

# Build 3 text variants
def build_text_full(row):
    parts = []
    if pd.notna(row.get("primary_diagnosis")) and row["primary_diagnosis"]:
        parts.append(f"Diagnosis: {row['primary_diagnosis']}.")
    stage_parts = []
    for field, label in [("ajcc_t","T"),("ajcc_n","N"),("ajcc_m","M")]:
        if pd.notna(row.get(field)) and row[field]:
            stage_parts.append(f"{label}: {row[field]}")
    if pd.notna(row.get("ajcc_stage")) and row["ajcc_stage"]:
        stage_parts.append(f"Overall stage: {row['ajcc_stage']}")
    if stage_parts:
        parts.append("TNM staging: " + "; ".join(stage_parts) + ".")
    demo = []
    if pd.notna(row.get("age_at_diagnosis")) and row["age_at_diagnosis"]:
        try: demo.append(f"age {int(row['age_at_diagnosis'])}")
        except: pass
    if pd.notna(row.get("gender")) and row["gender"]:
        demo.append(str(row["gender"]).lower())
    if demo:
        parts.append("Patient: " + ", ".join(demo) + ".")
    return " ".join(parts) if parts else "Oral squamous cell carcinoma."

def build_text_no_tnm(row):
    parts = []
    if pd.notna(row.get("primary_diagnosis")) and row["primary_diagnosis"]:
        parts.append(f"Diagnosis: {row['primary_diagnosis']}.")
    demo = []
    if pd.notna(row.get("age_at_diagnosis")) and row["age_at_diagnosis"]:
        try: demo.append(f"age {int(row['age_at_diagnosis'])}")
        except: pass
    if pd.notna(row.get("gender")) and row["gender"]:
        demo.append(str(row["gender"]).lower())
    if demo:
        parts.append("Patient: " + ", ".join(demo) + ".")
    return " ".join(parts) if parts else "Oral squamous cell carcinoma."

def build_text_demo_only(row):
    parts = []
    demo = []
    if pd.notna(row.get("age_at_diagnosis")) and row["age_at_diagnosis"]:
        try: demo.append(f"age {int(row['age_at_diagnosis'])}")
        except: pass
    if pd.notna(row.get("gender")) and row["gender"]:
        demo.append(str(row["gender"]).lower())
    if demo:
        parts.append("Patient: " + ", ".join(demo) + ".")
    return " ".join(parts) if parts else "Oral squamous cell carcinoma."

texts_full = [build_text_full(row) for _, row in clin_df.iterrows()]
texts_notnm = [build_text_no_tnm(row) for _, row in clin_df.iterrows()]
texts_demo = [build_text_demo_only(row) for _, row in clin_df.iterrows()]

print(f"Text samples:")
print(f"  Full:    {texts_full[0][:100]}...")
print(f"  No-TNM:  {texts_notnm[0][:100]}...")
print(f"  Demo:    {texts_demo[0][:100]}...")

# Encode with BiomedBERT
model_name = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
model = AutoModel.from_pretrained(model_name, local_files_only=True)
model.eval().to(DEVICE)

def encode_texts(texts, label):
    features = []
    bs = 32
    for i in range(0, len(texts), bs):
        batch = texts[i:i+bs]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            cls_emb = model(**inputs).last_hidden_state[:, 0, :].cpu().numpy()
            features.append(cls_emb)
    feats = np.concatenate(features, axis=0).astype(np.float32)
    print(f"  {label}: {feats.shape}")
    return feats

print("\nEncoding text variants...")
X_full = encode_texts(texts_full, "Full")
X_notnm = encode_texts(texts_notnm, "No-TNM")
X_demo = encode_texts(texts_demo, "Demo-only")

# Load oral labels and filter to matched cases
# Use the same case matching logic as refilter_oral_subsite.py
# Match by case_id to the integrated labels
integrated_labels = pd.read_csv(DATA_DIR / "integrated" / "case_labels.csv")
# clinical_data_oral has case_id; integrated case_labels has case_id + label
matched = clin_df[clin_df["case_id"].isin(integrated_labels["case_id"])].copy()
matched = matched.merge(integrated_labels, on="case_id", how="inner")
print(f"\nMatched cases: {len(matched)}")

# Get indices of matched cases in the original dataframe
clin_indices = clin_df[clin_df["case_id"].isin(matched["case_id"])].index.tolist()
y_4class = matched["label"].values
y_binary = np.where(np.isin(y_4class, [0, 1]), 0, 1)

X_full_m = X_full[clin_indices]
X_notnm_m = X_notnm[clin_indices]
X_demo_m = X_demo[clin_indices]
print(f"Binary: Low={int((y_binary==0).sum())}, High={int((y_binary==1).sum())}")

# ====== Run Text-Only CV for each variant ======
# Import training utilities from oral_experiments_full
with open("oral_experiments_full.py", encoding="utf-8") as f:
    src = f.read()
top_src = src.split("if __name__")[0]
exec(top_src)

from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

def run_text_cv(name, X, y):
    print(f"\n--- {name} ---")
    result = run_cv_with_stats(name, lambda: TextOnly(), True, [X], y, 5)
    agg = result["aggregate"]
    print(f"  Acc={agg['accuracy']['mean']:.3f}+-{agg['accuracy']['std']:.3f} "
          f"F1={agg['macro_f1']['mean']:.3f} AUC={agg['auc']['mean']:.3f} "
          f"HighF1={agg['high_f1']['mean']:.3f}")
    return result

results = {}
results["Full (Diagnosis + TNM + Demographics)"] = run_text_cv("Full", X_full_m, y_binary)
results["No-TNM (Diagnosis + Demographics only)"] = run_text_cv("NoTNM", X_notnm_m, y_binary)
results["Demographics-only (Age + Gender)"] = run_text_cv("Demo", X_demo_m, y_binary)

# Save
output = {}
for name, result in results.items():
    output[name] = {"aggregate": result["aggregate"]}

# Add interpretation
agg_full = results["Full (Diagnosis + TNM + Demographics)"]["aggregate"]
agg_notnm = results["No-TNM (Diagnosis + Demographics only)"]["aggregate"]
agg_demo = results["Demographics-only (Age + Gender)"]["aggregate"]

highf1_drop = agg_full["high_f1"]["mean"] - agg_notnm["high_f1"]["mean"]
output["_interpretation"] = {
    "high_f1_drop_without_tnm": float(highf1_drop),
    "verdict": (
        f"Dropping TNM reduces High F1 by {abs(highf1_drop):.3f}. "
        f"{'GRADE LEAKAGE SUSPECTED: TNM contributes substantially to grading' if abs(highf1_drop) > 0.08 else 'NO LEAKAGE: grading signal persists without TNM'}"
    ),
    "full_high_f1": float(agg_full["high_f1"]["mean"]),
    "notnm_high_f1": float(agg_notnm["high_f1"]["mean"]),
    "demo_high_f1": float(agg_demo["high_f1"]["mean"]),
}

json.dump(output, open(OUTPUT_DIR / "oral_tnm_sensitivity.json", "w"), indent=2)
print(f"\nSaved oral_tnm_sensitivity.json")
print("DONE")
