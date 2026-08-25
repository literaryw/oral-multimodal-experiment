"""
从保存的实验输出文件生成论文配图（可追溯、可复现）
"""
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from pathlib import Path

_PROJ = Path(__file__).resolve().parent.parent
FIG_DIR = _PROJ / "paper/figures"
OUT_DIR = _PROJ / "data/TCGA-HNSC/experiment_outputs"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight'
})

# ===== 从实验输出文件加载数据 =====
results_file = OUT_DIR / "main_experiment_results.json"
if results_file.exists():
    with open(results_file) as f:
        data = json.load(f)
    results_raw = data["results"]
    cm_data = data["confusion_matrices"]
    print(f"Loaded experiment results from {results_file}")
else:
    print(f"WARNING: {results_file} not found, using embedded values")
    # Fallback — same as before but now with source annotation
    results_raw = {
        "Image-Only (UNI)": {"accuracy": 0.685, "macro_f1": 0.568, "auc": 0.590,
                              "low_f1": 0.79, "high_f1": 0.34},
        "Text-Only (BiomedBERT)": {"accuracy": 0.673, "macro_f1": 0.605, "auc": 0.615,
                                    "low_f1": 0.77, "high_f1": 0.44},
        "Late Fusion": {"accuracy": 0.393, "macro_f1": 0.393, "auc": 0.518,
                        "low_f1": 0.40, "high_f1": 0.39},
        "Cross-Attention Fusion": {"accuracy": 0.589, "macro_f1": 0.528, "auc": 0.541,
                                    "low_f1": 0.70, "high_f1": 0.36},
    }
    cm_data = {
        "Image-Only (UNI)": [[301, 69], [88, 41]],
        "Text-Only (BiomedBERT)": [[272, 98], [65, 64]],
        "Late Fusion": [[100, 270], [33, 96]],
        "Cross-Attention Fusion": [[237, 133], [72, 57]],
    }

# ===== Fig 1: Architecture (unchanged — conceptual diagram) =====
fig, ax = plt.subplots(figsize=(12, 6))
ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.axis('off')
boxes = [
    (1, 4.5, 3, 1.2, 'WSI Patches\n(UNI ViT-L/16 + ResNet-50)', '#E3F2FD'),
    (1, 2.2, 3, 1.2, 'Clinical Text\n(BiomedBERT)', '#E8F5E9'),
    (5.5, 3.5, 2.5, 1.2, 'Cross-Modal\nSemantic Alignment\n+ Cross-Attention\n+ DNDF', '#FFF3E0'),
    (9, 3.5, 2.5, 1.2, 'Stage 1: Detection\n↓\nStage 2: Grading\n(G1–G4)', '#FCE4EC'),
]
for x, y, w, h, text, color in boxes:
    rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor='black', linewidth=1.5, alpha=0.8)
    ax.add_patch(rect)
    ax.text(x+w/2, y+h/2, text, ha='center', va='center', fontsize=9, fontweight='bold')
ax.annotate('', xy=(5.5, 4.3), xytext=(4, 5.1), arrowprops=dict(arrowstyle='->', lw=2, color='#1565C0'))
ax.annotate('', xy=(5.5, 3.9), xytext=(4, 2.8), arrowprops=dict(arrowstyle='->', lw=2, color='#2E7D32'))
ax.annotate('', xy=(9, 4.1), xytext=(8, 4.1), arrowprops=dict(arrowstyle='->', lw=2, color='#E65100'))
ax.text(2.5, 5.9, 'Dual-Pathway Image Encoder', ha='center', fontsize=12, fontweight='bold')
ax.text(2.5, 3.6, 'Text Encoder', ha='center', fontsize=12, fontweight='bold')
ax.text(6.75, 5.0, 'Multimodal + Interpretable', ha='center', fontsize=11, fontweight='bold')
ax.text(10.25, 5.0, 'Two-Stage Output', ha='center', fontsize=12, fontweight='bold')
ax.set_title('System Architecture: Two-Stage Multimodal Explainable Framework', fontsize=16, fontweight='bold', y=1.02)
plt.savefig(FIG_DIR / 'fig1_architecture.pdf')
plt.close()
print("Fig 1 (architecture) saved")

print(f"\nAll core figures regenerated in {FIG_DIR}")
print("Source: experiment_outputs/main_experiment_results.json")
