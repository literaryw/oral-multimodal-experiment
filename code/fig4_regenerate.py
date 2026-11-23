"""Regenerate two illustrative activation-gradient patch-score panels.

The classifier is loaded from its archived checkpoint. The panels are qualitative
one-dimensional score sequences over stored patch indices, not spatial heatmaps,
input-gradient explanations, or held-out performance estimates.
"""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fix_real_gradcam import OptimizedImageModel, compute_real_patch_cam, kl_to_uniform  # noqa: E402

PROJ = Path(__file__).resolve().parent.parent
DATA = PROJ / "data/TCGA-HNSC"
OUT = DATA / "experiment_outputs"
FIG = PROJ / "paper"
DEVICE = torch.device("cpu")

y4 = np.load(DATA / "integrated_full_legacy499" / "y.npy").ravel()
patches = np.load(DATA / "integrated" / "X_img_patches.npz")
model = OptimizedImageModel().to(DEVICE)
model.load_state_dict(torch.load(OUT / "image_only_model.pt", map_location=DEVICE,
                                 weights_only=True))
model.eval()

fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
selected = [("G1", np.where(y4 == 0)[0][10]), ("G3", np.where(y4 == 2)[0][10])]
for col, (label, sample_idx) in enumerate(selected):
    patch_array = torch.FloatTensor(patches["case_%d" % sample_idx]).to(DEVICE)
    n_patches = patch_array.shape[0]
    importance = compute_real_patch_cam(model, patch_array, 0 if label == "G1" else 1)
    print("  %s: %d patches  KL=%.3f" % (label, n_patches, kl_to_uniform(importance)))
    ax = axes[col]
    patch_index = np.arange(n_patches)
    top = np.argsort(importance)[-5:]
    colours = np.full(n_patches, "#5B8DB8", dtype=object)
    colours[top] = "#C65D47"
    ax.bar(patch_index, importance, color=colours, width=0.85)
    ax.set_title("%s case (%d stored patches)" % (label, n_patches),
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Stored patch index (no WSI coordinate mapping)")
    ax.set_ylabel("Normalized activation-gradient score")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.25)

fig.suptitle("Illustrative activation-gradient patch scores (not held-out evaluation)",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0.02, 0, 1, 0.92])
fig.savefig(FIG / "fig4_gradcam_heatmap.pdf")
preview = PROJ / "codex-review" / "fig4_preview.png"
preview.parent.mkdir(exist_ok=True)
fig.savefig(preview, dpi=140)
plt.close(fig)
print("saved paper/fig4_gradcam_heatmap.pdf")
