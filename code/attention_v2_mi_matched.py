"""
R19 P1 fix: dimension-matched mutual information.

The original MI block compared a cross-modality term computed on the first 10 text and
10 image dimensions against label terms computed on all 768 text and 1024 image
dimensions.  The ratio of two quantities estimated on different numbers of features is
not interpretable.  This script recomputes both label terms on the same first-10
dimensions used by the cross term, and reports the matched ratio alongside the original.

Outputs -> data/TCGA-HNSC/experiment_outputs/attention_v2_mi_matched.json
"""
import json
from pathlib import Path

import numpy as np
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parent.parent
D = BASE / "data" / "TCGA-HNSC"
OUT = D / "experiment_outputs"
K = 10

index = json.load(open(D / "attention_v2" / "cohort_index.json"))
y_all = np.array(index["y_binary"])
X_img_all = np.load(D / "integrated_full_legacy499" / "X_img_uni.npy")
cls_all = np.load(D / "attention_v2" / "text_tokens.npz")["cls"]
masks = {"full": np.ones(len(y_all), bool),
         "wide": np.array(index["wide_mask"], bool),
         "strict": np.array(index["strict_mask"], bool)}

out = {}
print("%-8s %-9s %-9s %-9s %-9s %-9s %-8s %-8s" % (
    "cohort", "cross", "img-lab", "txt-lab", "img-lab*", "txt-lab*", "ratio", "ratio*"))
for coh in ["full", "wide", "strict"]:
    rows = np.where(masks[coh])[0]
    y = y_all[rows]
    Xi = StandardScaler().fit_transform(X_img_all[rows])
    Xt = StandardScaler().fit_transform(cls_all[rows])
    cross = float(np.mean([mutual_info_regression(Xt[:, :K], Xi[:, i],
                                                  random_state=42).mean()
                           for i in range(K)]))
    mi_img = float(mutual_info_regression(Xi, y, random_state=42).mean())
    mi_txt = float(mutual_info_regression(Xt, y, random_state=42).mean())
    # dimension-matched label terms: same first-K columns as the cross term
    mi_img_m = float(mutual_info_regression(Xi[:, :K], y, random_state=42).mean())
    mi_txt_m = float(mutual_info_regression(Xt[:, :K], y, random_state=42).mean())
    out[coh] = {
        "K": K,
        "cross_mi": cross,
        "img_label_mi_all_dims": mi_img,
        "text_label_mi_all_dims": mi_txt,
        "img_label_mi_matched": mi_img_m,
        "text_label_mi_matched": mi_txt_m,
        "ratio_all_dims": cross / mi_img if mi_img else float("nan"),
        "ratio_matched": cross / mi_img_m if mi_img_m else float("nan"),
    }
    o = out[coh]
    print("%-8s %-9.4f %-9.4f %-9.4f %-9.4f %-9.4f %-8.2f %-8.2f" % (
        coh, cross, mi_img, mi_txt, mi_img_m, mi_txt_m, o["ratio_all_dims"], o["ratio_matched"]))

json.dump(out, open(OUT / "attention_v2_mi_matched.json", "w"), indent=2)
print("\nSaved attention_v2_mi_matched.json")
