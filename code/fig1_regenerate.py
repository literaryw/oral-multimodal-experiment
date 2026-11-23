"""
R18 (P2-1): replace Figure 1.

The previous Figure 1 was a system-architecture diagram of the *project blueprint*
(dual-pathway encoder, cross-modal alignment, two-stage detection-grading, DNDF),
almost none of which the paper evaluates.  A reader met the unvalidated design before
meeting the experiment.  This figure shows instead what the paper actually runs: the
The five configurations in the primary grid and its evaluation protocol are shown below; auxiliary analyses use their own protocols.

Output -> paper/fig1_architecture.pdf  (filename kept so the manuscript is unchanged)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJ = Path(__file__).resolve().parent.parent
FIG_DIR = PROJ / "paper"

plt.rcParams.update({"font.size": 10, "savefig.dpi": 300, "savefig.bbox": "tight"})

fig, ax = plt.subplots(figsize=(13, 6.4))
ax.set_xlim(0, 13)
ax.set_ylim(-0.75, 6.6)
ax.axis("off")

# ------------------------------------------------------------------ panel A
ax.text(0.15, 6.35, "A.  Configurations compared", fontsize=13, fontweight="bold")
ax.text(0.15, 6.0, "Primary grid only; auxiliary protocols are separate",
        fontsize=9, style="italic", color="#444444")

ROWS = [
    ("Image-Only (UNI)", "mean-pooled slide features (1024)  →  3-layer MLP", "#E3F2FD"),
    ("Text-Only (BiomedBERT)", "[CLS] embedding (768)  →  3-layer MLP", "#E8F5E9"),
    ("Late Fusion", "image and text each projected to 256  →  concat  →  MLP", "#FFF8E1"),
    ("Single-Token Attention", "single-token attention output is query-independent\n"
                             "image and text projections still enter a joint MLP", "#FFEBEE"),
    ("True Cross-Attention", "Q = patch tokens (N×1024), K = V = text tokens (L×768)\n"
                             "masked multi-head attention → pooled ⊕ text summary → MLP", "#EDE7F6"),
]
y = 5.15
for name, desc, color in ROWS:
    h = 0.86
    ax.add_patch(plt.Rectangle((0.15, y - h), 5.6, h, facecolor=color,
                               edgecolor="black", linewidth=1.2))
    ax.text(0.32, y - 0.28, name, fontsize=10.5, fontweight="bold", va="center")
    ax.text(0.32, y - 0.62, desc, fontsize=8.4, va="center", color="#222222")
    y -= 0.98

# ------------------------------------------------------------------ panel B
ax.text(6.45, 6.35, "B.  Evaluation protocol", fontsize=13, fontweight="bold")
ax.text(6.45, 6.0, "primary five-arm grid only", fontsize=9,
        style="italic", color="#444444")

STEPS = [
    ("Three nested cohorts", "full HNSCC n=499  ·  expanded oral-region n=332  ·  strict oral n=246", "#E1F5FE"),
    ("5-fold stratified CV per seed", "splits derived from the seed (42 / 123 / 456)", "#E0F2F1"),
    ("Nested split inside each training fold", "inner-train 80%  /  inner-validation 20%\n"
                                               "early stopping selects the epoch on the inner split", "#FFF3E0"),
    ("Held-out fold", "touched only to compute the reported metrics\n"
                      "(model selection never sees it)", "#F3E5F5"),
    ("Primary statistics", "McNemar tests correctness (seed 42; Bonferroni)\n"
                          "pointwise bootstrap on seed-42 pooled OOF; tables use 3-seed fold means ± SD", "#F1F8E9"),
]
y = 5.15
for i, (name, desc, color) in enumerate(STEPS):
    h = 0.78
    ax.add_patch(plt.Rectangle((6.45, y - h), 6.4, h, facecolor=color,
                               edgecolor="black", linewidth=1.2))
    ax.text(6.62, y - 0.25, name, fontsize=10.5, fontweight="bold", va="center")
    ax.text(6.62, y - 0.58, desc, fontsize=8.4, va="center", color="#222222")
    if i < len(STEPS) - 1:
        ax.annotate("", xy=(9.65, y - h - 0.20), xytext=(9.65, y - h),
                    arrowprops=dict(arrowstyle="-|>", lw=1.6, color="#37474F"))
    y -= 0.98

ax.text(0.15, -0.42, "Primary-grid protocol does not cover auxiliary analyses: LOSO (seed 42), saliency probes, "
                      "legacy learning curves, and the excluded DNDF run use separate protocols. "
                      "Project components outside the five-arm comparison remain unvalidated.",
        fontsize=8.2, style="italic", color="#555555")

plt.savefig(FIG_DIR / "fig1_architecture.pdf")
# PNG preview for visual review (gitignored scratch dir)
(FIG_DIR.parent / "codex-review").mkdir(exist_ok=True)
plt.savefig(FIG_DIR.parent / "codex-review" / "fig1_preview.png", dpi=140)
plt.close()
print("Fig 1 regenerated -> paper/fig1_architecture.pdf")
