"""
Exploratory mean pairwise coordinate MI for the first ten standardized features.

This is not joint vector MI, conditional redundancy, or a model-selection rule.
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
y_all = np.asarray(index["y_binary"])
X_img_all = np.load(D / "integrated_full_legacy499" / "X_img_uni.npy")
X_text_all = np.load(D / "attention_v2" / "text_tokens.npz")["cls"]
masks = {"full": np.ones(len(y_all), bool),
         "wide": np.asarray(index["wide_mask"], bool),
         "strict": np.asarray(index["strict_mask"], bool)}

result = {"estimator": "mean univariate pairwise MI; sklearn mutual_info_regression",
          "coordinates": K, "interpretation": "exploratory only; not joint MI or conditional redundancy",
          "cohorts": {}}
for cohort, mask in masks.items():
    Xi = StandardScaler().fit_transform(X_img_all[mask])[:, :K]
    Xt = StandardScaler().fit_transform(X_text_all[mask])[:, :K]
    pairs = np.empty((K, K), dtype=float)
    for i in range(K):
        pairs[i, :] = mutual_info_regression(Xt, Xi[:, i], random_state=42)
    values = pairs.reshape(-1)
    result["cohorts"][cohort] = {
        "n": int(mask.sum()),
        "mean_pairwise_coordinate_mi": float(values.mean()),
        "sd_pairwise_coordinate_mi": float(values.std(ddof=1)),
        "pairwise_estimates": pairs.tolist(),
    }
    print("%-7s mean pairwise MI=%.4f (SD %.4f)" %
          (cohort, values.mean(), values.std(ddof=1)))

json.dump(result, open(OUT / "attention_v2_mi_matched.json", "w"), indent=2)
print("Saved exploratory coordinate-MI summary.")
