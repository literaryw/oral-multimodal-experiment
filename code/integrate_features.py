"""
整合: UNI WSI图像特征 + 临床数据 + 标签 -> 多模态数据集
"""
import pandas as pd
import numpy as np
from pathlib import Path
import struct

FEATURE_DIR = Path("../data/TCGA-HNSC/features")
PRETRAINED_DIR = Path("../data/TCGA-HNSC/pretrained_features")
OUTPUT_DIR = Path("../data/TCGA-HNSC/integrated")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=== Step 1: Load Slide Image UNI features ===")
wsi_df = pd.read_parquet(PRETRAINED_DIR / "Slide_Image_UNI.parquet")
print(f"  Total slides: {len(wsi_df)}")
print(f"  Unique gdc_case_ids: {wsi_df['gdc_case_id'].nunique()}")
print(f"  Embedding shape: {wsi_df['embedding_shape'].iloc[0]}")

# Deserialize bytes -> mean-pooled 1024-dim vectors
def deserialize_and_pool(emb_bytes, emb_shape):
    """Convert serialized bytes to numpy array and mean-pool patches"""
    arr = np.frombuffer(emb_bytes, dtype=np.float32)
    n_patches = emb_shape[0]
    dim = emb_shape[1]
    arr = arr.reshape(n_patches, dim)  # (n_patches, 1024)
    return arr.mean(axis=0)  # mean pool -> (1024,)

def deserialize_raw(emb_bytes, emb_shape):
    """Convert serialized bytes to raw patch features (no pooling)"""
    arr = np.frombuffer(emb_bytes, dtype=np.float32)
    n_patches = emb_shape[0]
    dim = emb_shape[1]
    return arr.reshape(n_patches, dim)  # (n_patches, 1024)

print("\n=== Step 2: Deserialize embeddings ===")
# Process in batches
batch_size = 1000
all_features = []
all_case_ids = []
raw_patches = {}  # case_id -> list of (n_patches, 1024)

for i in range(0, len(wsi_df), batch_size):
    batch = wsi_df.iloc[i:i+batch_size]
    for _, row in batch.iterrows():
        try:
            feat = deserialize_and_pool(row['embedding'], row['embedding_shape'])
            raw = deserialize_raw(row['embedding'], row['embedding_shape'])
            all_features.append(feat)
            cid = row['gdc_case_id']
            all_case_ids.append(cid)
            if cid not in raw_patches:
                raw_patches[cid] = []
            raw_patches[cid].append(raw)
        except Exception as e:
            continue
    if i % 5000 == 0:
        print(f"  Processed {i}/{len(wsi_df)}")

wsi_features = np.array(all_features)
print(f"  WSI features: {wsi_features.shape}")
print(f"  Raw patch data for {len(raw_patches)} cases")

# Create case_id -> feature mapping (average multiple slides per case)
wsi_case_df = pd.DataFrame({'gdc_case_id': all_case_ids})
wsi_case_df['feat_idx'] = range(len(wsi_case_df))
case_groups = wsi_case_df.groupby('gdc_case_id')['feat_idx'].apply(list)

print(f"  Unique cases with WSI: {len(case_groups)}")

# Mean pool multiple slides per case
case_features = {}
for case_id, indices in case_groups.items():
    feats = wsi_features[indices]
    case_features[case_id] = feats.mean(axis=0)

print(f"  Case-level features: {len(case_features)} cases x {list(case_features.values())[0].shape[0]} dim")

print("\n=== Step 3: Load clinical data with labels ===")
clin_df = pd.read_csv(FEATURE_DIR.parent / "clinical_data.csv")
clin_df = clin_df[clin_df["tumor_grade"].isin(["G1", "G2", "G3", "G4"])].copy()
print(f"  Graded cases: {len(clin_df)}")

# We need gdc_case_id mapping from case_submitter_id
# The clinical data from GDC API doesn't have gdc_case_id
# Use the pretrained Clinical GatorTron file for mapping
clin_gatortron = pd.read_parquet(PRETRAINED_DIR / "Clinical_GatorTron.parquet")
hnsc_clin = clin_gatortron[clin_gatortron['project_id'] == 'TCGA-HNSC']
print(f"  HNSC cases in GatorTron: {len(hnsc_clin)}")

# Create submitter_id -> case_id mapping
id_map = dict(zip(hnsc_clin['case_submitter_id'], hnsc_clin['case_id']))
print(f"  ID map entries: {len(id_map)}")

print("\n=== Step 4: Merge image + clinical + labels ===")
grade_to_label = {"G1": 0, "G2": 1, "G3": 2, "G4": 3}

img_feats_list = []
text_feats_list = []
labels_list = []
matched_cases = []

for _, row in clin_df.iterrows():
    submitter_id = row['case_id']
    grade = row['tumor_grade']

    gdc_id = id_map.get(submitter_id)
    if gdc_id and gdc_id in case_features:
        img_feats_list.append(case_features[gdc_id])
        labels_list.append(grade_to_label[grade])
        matched_cases.append(submitter_id)

print(f"  Matched cases (image + clinical + grade): {len(matched_cases)}")

# Also load text features for matched cases
text_features_all = np.load(FEATURE_DIR / "text_features.npy")
labels_all = np.load(FEATURE_DIR / "labels.npy")
text_meta = pd.read_csv(FEATURE_DIR / "text_metadata.csv")
text_case_ids = text_meta['case_id'].tolist()

# Build text feature lookup
text_lookup = {cid: feat for cid, feat in zip(text_case_ids, text_features_all)}

for cid in matched_cases:
    if cid in text_lookup:
        text_feats_list.append(text_lookup[cid])
    else:
        # Mean text feature as fallback
        text_feats_list.append(text_features_all.mean(axis=0))

X_img = np.array(img_feats_list).astype(np.float32)
X_text = np.array(text_feats_list).astype(np.float32)
y = np.array(labels_list)

print(f"\n=== Final Dataset ===")
print(f"  X_img: {X_img.shape}")
print(f"  X_text: {X_text.shape}")
print(f"  y: {y.shape}")
print(f"  Label distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

# Save
np.save(OUTPUT_DIR / "X_img_uni.npy", X_img)

# Save raw patch features (map submitter_id → gdc_case_id → patches)
print(f"\n=== Save raw patch features ===")
patch_list = []
n_found = 0
for submitter_id in matched_cases:
    gdc_id = id_map.get(submitter_id)
    if gdc_id and gdc_id in raw_patches:
        patches = np.array(raw_patches[gdc_id][0]).astype(np.float32)  # first slide
        patch_list.append(patches)
        n_found += 1
    else:
        # Use a dummy all-zero patch
        patch_list.append(np.zeros((1, 1024), dtype=np.float32))

# Save as dict of individual arrays (each case may have different n_patches)
np.savez(OUTPUT_DIR / "X_img_patches.npz",
         **{f"case_{i}": p for i, p in enumerate(patch_list)})
print(f"  Saved raw patches: {n_found}/{len(matched_cases)} cases with real patch data")
np.save(OUTPUT_DIR / "X_text_biomedbert.npy", X_text)
np.save(OUTPUT_DIR / "y.npy", y)
pd.DataFrame({'case_id': matched_cases, 'label': y}).to_csv(OUTPUT_DIR / "case_labels.csv", index=False)
print(f"\nSaved to {OUTPUT_DIR}")
