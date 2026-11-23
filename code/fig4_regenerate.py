"""
Regenerate Figure 4 with the re-measured mutual-information panel.

The Grad-CAM panels are produced from the *saved* classifier (image_only_model.pt) and
the same patch file and sample indices as the original script, so the left/centre
panels are unchanged; only the MI panel is refreshed to the values measured on the
unified feature set (code/attention_v2_aux.py), which is what the manuscript now reports.
"""
import json
import sys
from pathlib import Path

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

# MI measured on the unified features (full HNSCC cohort, n=499)
aux = json.load(open(OUT / "attention_v2_aux.json"))
mi = aux["mutual_information"]["full"]
mi_cross, mi_img_label, mi_text_label = mi["cross_mi"], mi["img_label_mi"], mi["text_label_mi"]
print("MI panel (full cohort): cross=%.4f img-label=%.4f text-label=%.4f"
      % (mi_cross, mi_img_label, mi_text_label))

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for col, (label, sidx) in enumerate([("G1", np.where(y4 == 0)[0][10]), ("G3", np.where(y4 == 2)[0][10])]):
    key = "case_%d" % sidx
    p = torch.FloatTensor(patches[key]).to(DEVICE)
    n_p = p.shape[0]
    cam = compute_real_patch_cam(model, p, 0 if label == "G1" else 1)
    print("  %s: %d patches  KL=%.3f" % (label, n_p, kl_to_uniform(cam)))
    side = max(2, int(np.ceil(np.sqrt(n_p))))
    grid = np.zeros((side, side))
    for i in range(n_p):
        grid[i // side, i % side] = cam[i]
    ax = axes[col]
    im = ax.imshow(grid, cmap="jet", interpolation="bilinear")
    ax.set_title("%s — Real Patch Grad-CAM (%d patches)" % (label, n_p), fontsize=13, fontweight="bold")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, label="Importance")

ax = axes[2]
M = np.array([[np.nan, mi_cross, mi_img_label],
              [mi_cross, np.nan, mi_text_label],
              [mi_img_label, mi_text_label, np.nan]])
im = ax.imshow(np.ma.masked_invalid(M), cmap="YlOrRd", vmin=0, vmax=0.05)
for i in range(3):
    for j in range(3):
        if not np.isnan(M[i, j]):
            ax.text(j, i, "%.3f" % M[i, j], ha="center", va="center", fontsize=12, fontweight="bold")
ax.set_xticks(range(3)); ax.set_yticks(range(3))
ax.set_xticklabels(["Image\n(UNI)", "Text\n(BERT)", "Label\n(Grade)"])
ax.set_yticklabels(["Image\n(UNI)", "Text\n(BERT)", "Label\n(Grade)"])
ax.set_title("Mutual Information Matrix", fontsize=13, fontweight="bold")
plt.colorbar(im, ax=ax, fraction=0.046, label="MI (nats)")

plt.suptitle("Grad-CAM: Real Patch-Level Activation from Optimized Model", fontsize=15, fontweight="bold", y=0.99)
plt.tight_layout(rect=[0.02, 0, 1, 0.93])
plt.savefig(FIG / "fig4_gradcam_heatmap.pdf")
plt.close()
print("saved fig4_gradcam_heatmap.pdf")
