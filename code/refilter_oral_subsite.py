"""
TCGA-HNSCC 口腔亚组过滤脚本
1. 从 GDC API 下载含 tissue_or_organ_of_origin 的完整临床数据
2. 过滤出口腔亚组(oral cavity)病例
3. 更新 clinical_data.csv
4. 重提取文本特征 + 重整合多模态特征
"""
import json, requests, time
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
import struct

import os
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "TCGA-HNSC"
FEATURE_DIR = DATA_DIR / "features"
PRETRAINED_DIR = DATA_DIR / "pretrained_features"
OUTPUT_DIR = DATA_DIR / "integrated"
for d in [DATA_DIR, FEATURE_DIR, PRETRAINED_DIR, OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)
GDC_API = "https://api.gdc.cancer.gov"

# === Oral cavity site definitions ===
# === Strict oral cavity site definitions ===
# Excludes: Base of tongue (oropharyngeal), Overlapping lesion (cross-region),
#           Palate NOS (may include soft palate), oropharynx, hypopharynx, larynx
ORAL_SITES = {
    "Tongue, NOS", "Border of tongue", "Ventral surface of tongue, NOS",
    "Floor of mouth, NOS", "Anterior floor of mouth",
    "Mouth, NOS",
    "Cheek mucosa",
    "Gum, NOS", "Lower gum", "Upper Gum",
    "Lip, NOS",
    "Hard palate",
    "Retromolar area",
}

def download_clinical_with_site():
    """下载含 primary_site 的完整临床数据"""
    print("=== Step 1: Download clinical data with site info ===")
    endpoint = f"{GDC_API}/cases"
    params = {
        "filters": json.dumps({
            "op": "in",
            "content": {"field": "project.project_id", "value": ["TCGA-HNSC"]}
        }),
        "fields": "submitter_id,diagnoses.primary_diagnosis,diagnoses.tumor_grade,"
                  "diagnoses.ajcc_pathologic_t,diagnoses.ajcc_pathologic_n,"
                  "diagnoses.ajcc_pathologic_m,diagnoses.ajcc_pathologic_stage,"
                  "diagnoses.tissue_or_organ_of_origin,"
                  "demographic.gender,demographic.race,"
                  "diagnoses.age_at_diagnosis",
        "format": "JSON",
        "size": "1000"
    }
    resp = requests.get(endpoint, params=params)
    data = resp.json()
    hits = data.get("data", {}).get("hits", [])
    print(f"  Total cases from API: {len(hits)}")

    records = []
    for hit in hits:
        case_id = hit.get("submitter_id")
        demo = hit.get("demographic", {}) or {}
        for diag in hit.get("diagnoses", []):
            site = diag.get("tissue_or_organ_of_origin", "Not Reported")
            records.append({
                "case_id": case_id,
                "primary_diagnosis": diag.get("primary_diagnosis"),
                "tumor_grade": diag.get("tumor_grade"),
                "ajcc_t": diag.get("ajcc_pathologic_t"),
                "ajcc_n": diag.get("ajcc_pathologic_n"),
                "ajcc_m": diag.get("ajcc_pathologic_m"),
                "ajcc_stage": diag.get("ajcc_pathologic_stage"),
                "age_at_diagnosis": diag.get("age_at_diagnosis"),
                "gender": demo.get("gender"),
                "race": demo.get("race"),
                "primary_site": site,
            })

    df = pd.DataFrame(records)
    print(f"  Total diagnosis records: {len(df)}")
    print(f"  Site distribution:")
    for site, count in df["primary_site"].value_counts().items():
        marker = " [ORAL]" if site in ORAL_SITES else ""
        print(f"    {site}: {count}{marker}")
    return df

def filter_oral_cavity(df):
    """过滤口腔亚组"""
    print("\n=== Step 2: Filter oral cavity cases ===")
    oral_df = df[df["primary_site"].isin(ORAL_SITES)].copy()
    print(f"  Oral cavity cases: {len(oral_df)}")

    # Further filter: only cases with valid grade G1-G4
    graded = oral_df[oral_df["tumor_grade"].isin(["G1", "G2", "G3", "G4"])].copy()
    print(f"  With valid grade (G1-G4): {len(graded)}")

    # Deduplicate by case_id (keep first diagnosis per case)
    graded = graded.drop_duplicates(subset=["case_id"], keep="first")
    print(f"  After dedup: {len(graded)} unique cases")

    # Grade distribution
    for g in ["G1", "G2", "G3", "G4"]:
        n = (graded["tumor_grade"] == g).sum()
        print(f"    {g}: {n} ({100*n/len(graded):.1f}%)")

    return graded

def reextract_text_features(clinical_df):
    """用过滤后的临床数据重提取文本特征"""
    print("\n=== Step 3: Re-extract text features ===")
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)

    def build_clinical_text(row):
        parts = []
        if pd.notna(row.get("primary_diagnosis")) and row["primary_diagnosis"]:
            parts.append(f"Diagnosis: {row['primary_diagnosis']}.")
        stage_parts = []
        if pd.notna(row.get("ajcc_t")) and row["ajcc_t"]:
            stage_parts.append(f"T: {row['ajcc_t']}")
        if pd.notna(row.get("ajcc_n")) and row["ajcc_n"]:
            stage_parts.append(f"N: {row['ajcc_n']}")
        if pd.notna(row.get("ajcc_m")) and row["ajcc_m"]:
            stage_parts.append(f"M: {row['ajcc_m']}")
        if pd.notna(row.get("ajcc_stage")) and row["ajcc_stage"]:
            stage_parts.append(f"Overall stage: {row['ajcc_stage']}")
        if stage_parts:
            parts.append("TNM staging: " + "; ".join(stage_parts) + ".")
        demo_parts = []
        if pd.notna(row.get("age_at_diagnosis")) and row["age_at_diagnosis"]:
            try:
                demo_parts.append(f"age {int(row['age_at_diagnosis'])}")
            except:
                pass
        if pd.notna(row.get("gender")) and row["gender"]:
            demo_parts.append(str(row["gender"]).lower())
        if demo_parts:
            parts.append("Patient: " + ", ".join(demo_parts) + ".")
        text = " ".join(parts)
        return text if text.strip() else "Oral squamous cell carcinoma."

    clinical_df = clinical_df.reset_index(drop=True)
    texts = [build_clinical_text(row) for _, row in clinical_df.iterrows()]
    print(f"  Generated {len(texts)} clinical text narratives")
    print(f"  Sample: {texts[0][:120]}...")

    # BiomedBERT encoding
    model_name = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModel.from_pretrained(model_name, local_files_only=True)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    features = []
    batch_size = 32
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True,
                          max_length=512, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            features.append(cls_emb)
        if (i // batch_size) % 10 == 0:
            print(f"  Encoded {i}/{len(texts)}")

    features = np.concatenate(features, axis=0)
    print(f"  Text features: {features.shape}")

    # Labels
    grade_to_label = {"G1": 0, "G2": 1, "G3": 2, "G4": 3}
    labels = np.array([grade_to_label[g] for g in clinical_df["tumor_grade"]])

    np.save(FEATURE_DIR / "text_features.npy", features.astype(np.float32))
    np.save(FEATURE_DIR / "labels.npy", labels)
    clinical_df[["case_id"]].to_csv(FEATURE_DIR / "text_metadata.csv", index=False)
    print(f"  Saved text features and labels to {FEATURE_DIR}")
    return features, labels

def reintegrate_features(clinical_df):
    """整合过滤后的图像+文本特征"""
    print("\n=== Step 4: Re-integrate features ===")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load UNI features
    wsi_df = pd.read_parquet(PRETRAINED_DIR / "Slide_Image_UNI.parquet")
    print(f"  Total UNI slides: {len(wsi_df)}")

    def deserialize_and_pool(emb_bytes, emb_shape):
        arr = np.frombuffer(emb_bytes, dtype=np.float32)
        return arr.reshape(emb_shape[0], emb_shape[1]).mean(axis=0)

    # Deserialize all
    all_features = []
    all_case_ids = []
    for _, row in wsi_df.iterrows():
        try:
            feat = deserialize_and_pool(row['embedding'], row['embedding_shape'])
            all_features.append(feat)
            all_case_ids.append(row['gdc_case_id'])
        except:
            continue
    wsi_features = np.array(all_features)
    print(f"  Deserialized {len(wsi_features)} slide features")

    # Case-level aggregation
    wsi_case_df = pd.DataFrame({'gdc_case_id': all_case_ids})
    wsi_case_df['feat_idx'] = range(len(wsi_case_df))
    case_groups = wsi_case_df.groupby('gdc_case_id')['feat_idx'].apply(list)
    case_features = {}
    for case_id, indices in case_groups.items():
        case_features[case_id] = wsi_features[indices].mean(axis=0)

    # ID mapping
    clin_gatortron = pd.read_parquet(PRETRAINED_DIR / "Clinical_GatorTron.parquet")
    hnsc_clin = clin_gatortron[clin_gatortron['project_id'] == 'TCGA-HNSC']
    id_map = dict(zip(hnsc_clin['case_submitter_id'], hnsc_clin['case_id']))

    # Match oral cavity cases
    img_feats = []
    text_feats_list = []
    labels_list = []
    matched_cases = []

    grade_to_label = {"G1": 0, "G2": 1, "G3": 2, "G4": 3}
    text_features_all = np.load(FEATURE_DIR / "text_features.npy")
    text_meta = pd.read_csv(FEATURE_DIR / "text_metadata.csv")
    text_lookup = {}
    for i, row in text_meta.iterrows():
        if i < len(text_features_all):
            text_lookup[row['case_id']] = text_features_all[i]

    for _, row in clinical_df.iterrows():
        submitter_id = row['case_id']
        grade = row['tumor_grade']
        gdc_id = id_map.get(submitter_id)
        if gdc_id and gdc_id in case_features and submitter_id in text_lookup:
            img_feats.append(case_features[gdc_id])
            text_feats_list.append(text_lookup[submitter_id])
            labels_list.append(grade_to_label[grade])
            matched_cases.append(submitter_id)

    print(f"  Matched cases (oral + image + text + grade): {len(matched_cases)}")

    X_img = np.array(img_feats).astype(np.float32)
    X_text = np.array(text_feats_list).astype(np.float32)
    y = np.array(labels_list)

    print(f"  X_img: {X_img.shape}")
    print(f"  X_text: {X_text.shape}")
    print(f"  y: {y.shape}")
    print(f"  Label dist: {dict(zip(*np.unique(y, return_counts=True)))}")

    np.save(OUTPUT_DIR / "X_img_uni.npy", X_img)
    np.save(OUTPUT_DIR / "X_text_biomedbert.npy", X_text)
    np.save(OUTPUT_DIR / "y.npy", y)
    pd.DataFrame({'case_id': matched_cases, 'label': y}).to_csv(
        OUTPUT_DIR / "case_labels.csv", index=False)

    # Save dataset summary
    summary = {
        "total_cases": len(matched_cases),
        "oral_subsites": clinical_df["primary_site"].value_counts().to_dict(),
        "grade_distribution": {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
        "binary_distribution": {
            "Low-Grade": int((y <= 1).sum()),
            "High-Grade": int((y >= 2).sum())
        }
    }
    json.dump(summary, open(OUTPUT_DIR / "oral_dataset_summary.json", "w"), indent=2)
    print(f"\n  Saved to {OUTPUT_DIR}")
    return X_img, X_text, y

if __name__ == "__main__":
    # Step 1-2: Download + filter
    full_df = download_clinical_with_site()
    oral_df = filter_oral_cavity(full_df)

    # Save filtered clinical data
    oral_df.to_csv(DATA_DIR / "clinical_data_oral.csv", index=False)
    # Also overwrite as main clinical_data.csv for downstream compatibility
    oral_df.to_csv(DATA_DIR / "clinical_data.csv", index=False)
    print(f"\n  Saved filtered clinical data ({len(oral_df)} oral cavity cases)")

    # Step 3: Re-extract text features
    text_feats, labels = reextract_text_features(oral_df)

    # Step 4: Re-integrate (image + text)
    X_img, X_text, y = reintegrate_features(oral_df)

    print(f"\n=== COMPLETE: Oral subsite pipeline ===")
    print(f"  Oral cavity cases with full multimodal data: {len(y)}")
    bin_y = (y >= 2).astype(int)
    print(f"  Binary: Low-Grade={int((bin_y==0).sum())}, High-Grade={int((bin_y==1).sum())}")
