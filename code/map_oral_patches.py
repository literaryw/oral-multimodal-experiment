"""
口腔 case_id → X_img_patches.npz 索引映射
重建原始全数据集 matched_cases 顺序，找到口腔亚组对应的 patch 索引
"""
import json, requests
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "TCGA-HNSC"
GDC_API = "https://api.gdc.cancer.gov"

# Step 1: Download full HNSC clinical data from GDC API
print("=== Step 1: Download full HNSC clinical data ===")
endpoint = f"{GDC_API}/cases"
params = {
    "filters": json.dumps({
        "op": "in",
        "content": {"field": "project.project_id", "value": ["TCGA-HNSC"]}
    }),
    "fields": "submitter_id,diagnoses.primary_diagnosis,diagnoses.tumor_grade,"
              "diagnoses.tissue_or_organ_of_origin,"
              "demographic.gender,demographic.race,diagnoses.age_at_diagnosis",
    "format": "JSON", "size": "1000"
}
resp = requests.get(endpoint, params=params)
data = resp.json()
hits = data.get("data", {}).get("hits", [])
print(f"  Total cases: {len(hits)}")

records = []
for hit in hits:
    case_id = hit.get("submitter_id")
    demo = hit.get("demographic", {}) or {}
    for diag in hit.get("diagnoses", []):
        records.append({
            "case_id": case_id,
            "tumor_grade": diag.get("tumor_grade"),
            "primary_site": diag.get("tissue_or_organ_of_origin", "Not Reported"),
        })
full_df = pd.DataFrame(records)
# Filter to valid grades
full_graded = full_df[full_df["tumor_grade"].isin(["G1","G2","G3","G4"])].copy()
full_graded = full_graded.drop_duplicates(subset=["case_id"], keep="first")
print(f"  Full graded cases: {len(full_graded)}")

# Step 2: Build ID mapping (submitter_id → gdc_case_id) via GatorTron
print("\n=== Step 2: Build ID mapping ===")
clin_gatortron = pd.read_parquet(DATA_DIR / "pretrained_features" / "Clinical_GatorTron.parquet")
hnsc_clin = clin_gatortron[clin_gatortron['project_id'] == 'TCGA-HNSC']
id_map = dict(zip(hnsc_clin['case_submitter_id'], hnsc_clin['case_id']))
print(f"  ID map entries: {len(id_map)}")

# Step 3: Reconstruct full matched_cases order
print("\n=== Step 3: Reconstruct original matched_cases order ===")
# Load UNI features to get gdc_case_id order
wsi_df = pd.read_parquet(DATA_DIR / "pretrained_features" / "Slide_Image_UNI.parquet")
# Get all unique gdc_case_ids in order of first appearance
all_gdc_ids = wsi_df['gdc_case_id'].unique()
print(f"  Unique gdc_case_ids in UNI: {len(all_gdc_ids)}")

# Build reverse map: gdc_case_id → list of submitter_ids
gdc_to_submitter = {}
for submitter_id, gdc_id in id_map.items():
    if gdc_id not in gdc_to_submitter:
        gdc_to_submitter[gdc_id] = []
    gdc_to_submitter[gdc_id].append(submitter_id)

# Reconstruct the integration order
# integrate_features.py iterates clin_df (full_graded) and matches via id_map
# matched_cases order = clin_df order filtered by (id_map hit + case_features hit)
case_features_set = set()  # gdc_case_ids that have UNI features
for _, row in wsi_df.iterrows():
    case_features_set.add(row['gdc_case_id'])

full_matched = []
for _, row in full_graded.iterrows():
    submitter_id = row['case_id']
    gdc_id = id_map.get(submitter_id)
    if gdc_id and gdc_id in case_features_set:
        full_matched.append(submitter_id)

print(f"  Reconstructed full matched_cases: {len(full_matched)}")

# Step 4: Load oral case_ids and find their patch indices
print("\n=== Step 4: Map oral cases to patch indices ===")
oral_labels = pd.read_csv(DATA_DIR / "integrated" / "case_labels.csv")
oral_case_ids = set(oral_labels['case_id'].tolist())
print(f"  Oral cases: {len(oral_case_ids)}")

# Build mapping: oral case_id → patch index (case_{i})
oral_to_patch = {}
for i, submitter_id in enumerate(full_matched):
    if submitter_id in oral_case_ids:
        oral_to_patch[submitter_id] = i

print(f"  Oral cases mapped to patches: {len(oral_to_patch)}")

# Step 5: Load patches for oral cases and save
print("\n=== Step 5: Extract oral patch features ===")
patches_all = np.load(DATA_DIR / "integrated" / "X_img_patches.npz")

# Also get the oral labels (4-class)
oral_labels_dict = dict(zip(oral_labels['case_id'], oral_labels['label']))

oral_patches = {}  # case_id → (n_patches, 1024)
oral_grades = {}   # case_id → grade label
for case_id, patch_idx in oral_to_patch.items():
    key = f"case_{patch_idx}"
    if key in patches_all:
        p = patches_all[key]
        if p.shape[0] > 1:  # Only cases with >1 patch (meaningful spatial analysis)
            oral_patches[case_id] = p
            oral_grades[case_id] = oral_labels_dict.get(case_id, -1)

print(f"  Oral cases with >1 patch: {len(oral_patches)}")
# Grade distribution
from collections import Counter
grade_dist = Counter(oral_grades.values())
print(f"  G1(0): {grade_dist.get(0,0)}, G2(1): {grade_dist.get(1,0)}, G3(2): {grade_dist.get(2,0)}")

# Save mapping
np.savez(DATA_DIR / "integrated" / "oral_patches.npz",
         case_ids=np.array(list(oral_patches.keys())),
         **{f"patches_{i}": v for i, (k, v) in enumerate(oral_patches.items())})
# Also save a simple lookup
mapping = {
    "total_oral_with_patches": len(oral_patches),
    "grade_counts": {str(k): v for k, v in grade_dist.items()},
    "case_to_original_idx": {k: int(v) for k, v in oral_to_patch.items() if k in oral_patches},
}
json.dump(mapping, open(DATA_DIR / "integrated" / "oral_patch_mapping.json", "w"), indent=2)
print(f"\nSaved oral_patches.npz + oral_patch_mapping.json")
print("DONE")
