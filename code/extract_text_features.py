"""
临床文本特征提取脚本
1. 将结构化临床数据转换为自然语言文本
2. 使用 PubMedBERT 提取文本特征向量
"""
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
import numpy as np

DATA_DIR = Path("../data/TCGA-HNSC")
OUTPUT_DIR = Path("../data/TCGA-HNSC/features")
BATCH_SIZE = 32

def build_clinical_text(row):
    """将结构化临床数据拼接为类病理报告的文本"""
    parts = []

    # 主诊断
    if pd.notna(row.get("primary_diagnosis")) and row["primary_diagnosis"]:
        parts.append(f"Diagnosis: {row['primary_diagnosis']}.")

    # 病理分级 (REMOVED from text to prevent label leakage)
    # grade is the prediction target, should NOT be in input features

    # AJCC分期
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

    # 人口学
    demo_parts = []
    if pd.notna(row.get("age_at_diagnosis")) and row["age_at_diagnosis"]:
        demo_parts.append(f"age {int(row['age_at_diagnosis'])}")
    if pd.notna(row.get("gender")) and row["gender"]:
        demo_parts.append(row["gender"].lower())
    if demo_parts:
        parts.append("Patient: " + ", ".join(demo_parts) + ".")

    text = " ".join(parts)
    return text if text.strip() else "Oral squamous cell carcinoma."

def extract_features():
    print("=== 临床文本特征提取 ===")

    # 加载数据
    df = pd.read_csv(DATA_DIR / "clinical_data.csv")
    df = df[df["tumor_grade"].isin(["G1", "G2", "G3", "G4"])].copy()
    print(f"  有效病例: {len(df)}")

    # 构建文本
    df["clinical_text"] = df.apply(build_clinical_text, axis=1)
    print(f"  文本样例: {df['clinical_text'].iloc[0][:200]}...")

    # 加载PubMedBERT
    print("  加载 PubMedBERT...")
    model_name = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
    # 备用: "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    # 提取特征
    print(f"  提取特征中...")
    all_features = []

    for i in range(0, len(df), BATCH_SIZE):
        batch_texts = df["clinical_text"].iloc[i:i+BATCH_SIZE].tolist()
        inputs = tokenizer(batch_texts, padding=True, truncation=True,
                          max_length=256, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].numpy()
            all_features.append(cls_embeddings)

        if (i // BATCH_SIZE) % 10 == 0:
            print(f"    {i}/{len(df)}")

    features = np.vstack(all_features)
    print(f"  特征形状: {features.shape}")

    # 保存
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_DIR / "text_features.npy", features)
    df[["case_id", "tumor_grade", "clinical_text"]].to_csv(
        OUTPUT_DIR / "text_metadata.csv", index=False
    )

    # 标签
    grade_to_label = {"G1": 0, "G2": 1, "G3": 2, "G4": 3}
    labels = df["tumor_grade"].map(grade_to_label).values
    np.save(OUTPUT_DIR / "labels.npy", labels)

    print(f"  标签分布: {dict(zip(*np.unique(labels, return_counts=True)))}")
    print(f"  特征已保存: {OUTPUT_DIR}")

if __name__ == "__main__":
    extract_features()
