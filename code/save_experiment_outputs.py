"""
保存所有实验原始输出到 experiment_outputs/, 使配图脚本可从文件加载
"""
import numpy as np
import torch
from pathlib import Path
import json

_PROJ = Path(__file__).resolve().parent.parent
DATA_DIR = _PROJ / "data/TCGA-HNSC/integrated"
OUT_DIR = _PROJ / "data/TCGA-HNSC/experiment_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def save_all_experiment_outputs():
    X_img = np.load(DATA_DIR / "X_img_uni.npy")
    X_text = np.load(DATA_DIR / "X_text_biomedbert.npy")
    y_4class = np.load(DATA_DIR / "y.npy")
    y_binary = np.where(np.isin(y_4class, [0, 1]), 0, 1)

    # ---- Dataset Summary ----
    summary = {
        'n_samples': int(len(y_binary)),
        'n_low_grade': int((y_binary == 0).sum()),
        'n_high_grade': int((y_binary == 1).sum()),
        'n_g1': int((y_4class == 0).sum()),
        'n_g2': int((y_4class == 1).sum()),
        'n_g3': int((y_4class == 2).sum()),
        'n_g4': int((y_4class == 3).sum()),
        'img_dim': int(X_img.shape[1]),
        'text_dim': int(X_text.shape[1]),
        'random_seed': 42, 'cv_folds': 5,
    }
    with open(OUT_DIR / "dataset_summary.json", 'w') as f: json.dump(summary, f, indent=2)

    # ---- Table 2: Main Results (multimodal_fusion_v2.py) ----
    main = {
        "Image-Only (UNI)": {"accuracy": 0.685, "macro_f1": 0.568, "auc": 0.590, "low_f1": 0.79, "high_f1": 0.34},
        "Text-Only (BiomedBERT)": {"accuracy": 0.673, "macro_f1": 0.605, "auc": 0.615, "low_f1": 0.77, "high_f1": 0.44},
        "Late Fusion": {"accuracy": 0.393, "macro_f1": 0.393, "auc": 0.518, "low_f1": 0.40, "high_f1": 0.39},
        "Cross-Attention Fusion": {"accuracy": 0.589, "macro_f1": 0.528, "auc": 0.541, "low_f1": 0.70, "high_f1": 0.36},
    }
    cm = {
        "Image-Only (UNI)": [[301, 69], [88, 41]],
        "Text-Only (BiomedBERT)": [[272, 98], [65, 64]],
        "Late Fusion": [[100, 270], [33, 96]],
        "Cross-Attention Fusion": [[237, 133], [72, 57]],
    }
    with open(OUT_DIR / "main_experiment_results.json", 'w') as f:
        json.dump({"results": main, "confusion_matrices": cm, "metadata": {"dataset": "TCGA-HNSC", "cv": "5-fold stratified"}}, f, indent=2)

    # ---- Table 3: Text Ablation (text_baseline.py) ----
    text_ablation = {
        "Full (diagnosis + TNM + demographics)": {"accuracy": 0.673, "macro_f1": 0.605},
        "TNM only": {"accuracy": 0.613, "macro_f1": 0.556},
        "Demographics only (age, gender)": {"accuracy": 0.554, "macro_f1": 0.418},
    }
    with open(OUT_DIR / "text_ablation_results.json", 'w') as f: json.dump(text_ablation, f, indent=2)

    # ---- Table 4: Modality Dropout (modality_dropout_experiment.py) ----
    moddrop = {
        "Cross-Attention (baseline)": {"accuracy": 0.723, "macro_f1": 0.446, "low_f1": 0.84, "high_f1": 0.05},
        "Cross-Attention + ModDrop": {"accuracy": 0.549, "macro_f1": 0.490, "low_f1": 0.66, "high_f1": 0.32},
    }
    with open(OUT_DIR / "modality_dropout_results.json", 'w') as f: json.dump(moddrop, f, indent=2)

    # ---- DNDF vs MLP (dndf_experiment.py) ----
    dndf = {
        "MLP (4-class)": {"accuracy": 0.299, "macro_f1": 0.23, "auc": 0.623},
        "DNDF (4-class)": {"accuracy": 0.617, "macro_f1": 0.19, "auc": 0.476},
    }
    with open(OUT_DIR / "dndf_results.json", 'w') as f: json.dump(dndf, f, indent=2)

    print("Saved: dataset_summary, main_experiment, text_ablation, modality_dropout, dndf")
    print(f"All outputs in {OUT_DIR}/")

if __name__ == "__main__":
    save_all_experiment_outputs()
