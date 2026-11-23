# Manuscript-version reproduction workflows

This guide describes the current manuscript-version scripts and the archived outputs that let a reader replay every reported table without training. The `publish` branch of `https://github.com/literaryw/oral-multimodal-experiment` carries these scripts, the patient-level predictions and the regenerated table fragments. Datasets and pre-extracted features are not distributed there: TCGA-HNSC is available from the GDC and the UNI features from the Lab-Rasool/TCGA dataset, so a full retrain starts from those public downloads plus the retained feature bundle described in Section 4 of this guide.

Run commands from the repository root in the project environment (Python 3.12, environment tcga-oscc). The verified workflow begins with the retained manuscript feature bundle. Public raw data and pretrained assets alone do not yet reconstruct that bundle.

## 1. Verify archived outputs without training

This path reads per-sample out-of-fold predictions from the three primary-grid files and regenerates paired correctness tests, descriptive bootstrap summaries, and table rows. It does not refit the neural models.

```powershell
python code\attention_v2_analysis.py
python code\attention_v2_rederive.py
python code\attention_v2_tables.py
```

The analysis script requires data/TCGA-HNSC/attention_v2/cohort_index.json and data/TCGA-HNSC/experiment_outputs/attention_v2_{full,wide,strict}_results.json. Table generation additionally reads attention_v2_aux.json, attention_v2_loso_moddrop.json, attention_v2_mi_matched.json, and attention_v2_structured_baseline.json. The analysis JSON records exact-binomial McNemar calculations on seed-specific OOF correctness. Binomial exactness does not extend to inference over overlapping cross-validation training sets, and these calculations do not test Macro F1, High-Grade F1, AUC, or average precision. The paired metric bootstrap uses seed-42 pooled OOF predictions and reports conditional pointwise intervals. Main-table estimates remain three-seed means of fold means.

`code\attention_v2_rederive.py` re-derives each seed's outer and inner splits from `cohort_index.json`, recomputes every per-fold metric and three-seed fold mean for the 45 (cohort, arm, seed) records, and checks one out-of-fold prediction per patient, five-fold coverage, inner fit/selection disjointness, aggregate mean/SD, and `pred == argmax(prob)`. It then replays the representation-matched cells (manuscript Table 3), the reference-dependence differences (Table 4), every structured-baseline record from its per-seed out-of-fold vector (Table 12), and the four no-diagnosis cells with their per-metric differences (Table 13). It writes `attention_v2_rederive.json` and `.tex`. These checks establish consistency with the stated protocol; they do not prove what the historical training loop touched.

## 1b. Replay the follow-up table

```powershell
python code\replay_grade_followup.py --run data\TCGA-HNSC\followup_archive --labels data\TCGA-HNSC\followup_archive\patient_join.csv
```

This script needs NumPy and PyArrow; everything else it reads is in `data/TCGA-HNSC/followup_archive/` on the `publish` branch — `outer_evaluation_v4/strict_outer_oof_v4.parquet`, `architecture_outer_evaluation_v2/metrics.json`, the A07/A08 `prediction.npz` files under `architecture_outer_predictions_v2/`, and `patient_join.csv`. It verifies the archived SHA-256 hashes, requires complete patient/member prediction coverage, and recomputes the B-main/B-v4 references, the three A08-A07 partition differences, and the conditional 95% interval from the original 4,000 paired bootstrap draws. The request bundle carries the same layout with the surrounding run record; `results/main-grade-improvement-2026-11-09-v1/REPRODUCE.md` describes it. The replay covers Section 4.10 and Table 17 only; C1, C2 and C-A1 remain not estimable because their per-fold eligibility records are not archived.

## 2. Rerun the primary grid from retained manuscript features

The feature-bundle builder requires the retained 499-case integrated arrays and corrected metadata listed in manuscript_bundle_manifest.json. It creates token sequences, patch sequences, and cohort masks. The training script runs all three cohorts, five arms, and seeds 42/123/456; each cohort is saved separately for the analysis and table scripts.

```powershell
python code\prep_attention_v2.py
python code\attention_v2_experiments.py --cohort all --seeds 42,123,456 --tag attention_v2
python code\attention_v2_analysis.py
```

Training overwrites the named primary-grid JSON files. Preserve the archived outputs before running a new analysis. The primary grid uses outer five-fold stratified splits and an 80/20 inner split of each outer-training fold; models are fitted on the inner-training rows and selected by inner-validation Macro F1. This command sequence does not rerun the separately configured auxiliary analyses in Section 5. Run the required auxiliary jobs before regenerating the full table output with attention_v2_tables.py.

## 2b. No-diagnosis four-cell comparison

```powershell
python code\attention_v2_nodx_encode.py
python code\attention_v2_nodx_fourcell.py
```

The encoder re-renders the stored narrative without the primary-diagnosis block and re-encodes it at the base-499 level. Gates: 499/499 template round-trip, no case with empty TNM and demographics, and a full-template re-encoding that reproduces the stored token mask exactly and token values within 1e-5. The comparison script replaces the token, mask and [CLS] inputs, checks that the seed-42 no-diagnosis Text-Only prediction reproduces the archived auxiliary run, then runs Text-Only and True Cross-Attention on the strict subset for seeds 42/123/456. The revision-stage analysis plan and the frozen input hashes are in `data/TCGA-HNSC/experiment_outputs/analysis_plan_nodx_fourcell.md`. That plan postdates the with-diagnosis cells and one seed-42 auxiliary no-diagnosis Text-Only run, and shares a revision commit with the results, so it records an analysis plan rather than a time-stamped public registration. Outputs are `attention_v2_nodx_fourcell.json` (four cells plus per-seed out-of-fold predictions and fold indices), `attention_v2_nodx_fourcell.tex` (table body), `attention_v2_nodx_seed.tex` (per-seed differences) and `attention_v2_nodx_metric.tex` (per-metric differences). All differences are computed from unrounded cell values and rounded only for display. The script refuses to overwrite the archived JSON unless every cell reproduces to 1e-12.

## 3. Assemble and verify the request-based bundle

```powershell
python code\build_manuscript_bundle_manifest.py
python code\verify_manuscript_bundle.py --bundle-root codex-review\bundle-stage --stage-from . --report codex-review\bundle-verification.json
```

The first command writes file roles, sizes, and SHA-256 hashes for the retained inputs, analysis code, OOF outputs, and figure sources, including the source-parquet columns needed to audit slide selection. The second command hard-links or copies only manifest-listed files into an independent staging directory and validates them there. Checks cover 499 unique ordered case IDs, 499/332/246 nested masks, binary-label derivation, slide-manifest IDs and order against the source parquet, pooled and token feature shapes, patch-array order, five neural arms, three seeds, five folds, OOF lengths, and the $499\times4\times4$ latent-attention archive. The staging directory must be absent or empty.

Not every listed file is published: the retained feature inputs are not redistributed, so the verifier runs against the request bundle. Text files that are published also record `size_lf` and `sha256_lf`, which are the size and digest of the Git blob, so a published entry can be checked without the bundle: `git cat-file blob <rev>:<path>` piped through `sha256sum` reproduces `sha256_lf`. `--line-ending auto` accepts either the working-tree bytes or their LF-normalized form, `raw` requires the former and `lf` the latter. The manifest marks each entry `public` or `request_only`; a plain clone without the retained inputs checks the 61 published entries with `python code\verify_manuscript_bundle.py --bundle-root . --public-only --public-rev HEAD`, which fails on any byte difference or on an entry whose classification does not match the tree.

## 4. Public raw-source reconstruction boundary

The current repository does not provide a validated end-to-end path from public downloads to the retained manuscript feature bundle. The older download and integration scripts use inconsistent data roots and clinical-field schemas, do not recreate the historical 499-case inputs or the 332/246 masks, and do not pin the Lab-Rasool/TCGA or BiomedBERT revisions used by the historical encodings. They remain historical utilities, not the authoritative manuscript reproduction entry.

Until those upstream revisions and transformations are reconstructed and verified, a request-based bundle must include the files labelled `retained_feature_inputs` in `manuscript_bundle_manifest.json`. Archive verification and retraining from those retained inputs are supported; public raw-source reconstruction is not claimed.

## 5. Auxiliary analyses and scope

To rebuild the archived table inputs after a new primary-grid run, execute the auxiliary jobs with their documented scopes, then emit the table text:

```powershell
python code\attention_v2_aux.py
python code\attention_v2_loso_moddrop.py
python code\attention_v2_mi_matched.py
python code\attention_v2_rep_match.py
python code\attention_v2_structured_baseline.py
python code\attention_v2_skip_ablation.py
python code\attention_v2_tables.py
```

The auxiliary scripts use separate protocols noted below; their outputs must not be described as primary-grid estimates unless the manuscript states the relevant seed and aggregation.

- code/attention_v2_structured_baseline.py uses the same patient folds and inner-train rows as the primary grid; its metrics are fold means aggregated across three seeds. It archives per-seed out-of-fold predictions and fold indices, so Table 12 can be replayed without refitting, and it refuses to overwrite its JSON when a rerun does not reproduce the stored metrics.
- code/attention_v2_skip_ablation.py keeps the nominal 512-dimensional classifier input, but zero-masking changes active gradient paths. Its archived patch-only path used a different need_weights setting from the full path, so the output is exploratory and does not isolate branch information alone.
- code/attention_v2_mi_matched.py reports an exploratory mean pairwise coordinate MI for ten standardized features. It is not joint MI, conditional redundancy, or a fusion-selection rule.
- code/attention_v2_loso_moddrop.py contains seed-42 leave-one-group-out and modality-dropout analyses; these are not three-seed primary-grid results.
- code/learning_curves_multiseed.py is a legacy exploratory workflow and is excluded from the paper's primary evidence and sample-size conclusions.
- The archived DNDF-vs-MLP run is objective-mismatched: the model returns probabilities that are passed to CrossEntropyLoss as logits. It is excluded from performance claims. code/fig3_regenerate.py only renders the saved exploratory parameters and does not train a model.
- Manuscript Figure 1: `code/fig1_regenerate.py` writes `paper/fig1_architecture.pdf`. The older `code/generate_figures.py` writes a separate legacy blueprint file.
- Manuscript Figure 2 (`fig:gradcam`): `code/fig4_regenerate.py` reads `experiment_outputs/image_only_model.pt`, `integrated_full_legacy499/y.npy`, and `integrated/X_img_patches.npz`, then writes `paper/fig4_gradcam_heatmap.pdf`. It displays one-dimensional activation-gradient scores by stored patch index; there is no WSI coordinate mapping.
- Manuscript Figure 3 (`fig:dndfrouting`): `code/fig3_regenerate.py` reads `experiment_outputs/dndf_learned_params.npz` and writes `paper/fig3_dndf_routing.pdf`.
- Manuscript Figure 4 (`fig:textweights`): `code/fig2_regenerate.py` reads `experiment_outputs/text_latent_attention_archive.npz` and writes `paper/fig2_text_weights.pdf`. The archive contains real $4\times4$ latent-chunk attention only. No TNM, diagnosis, age, gender, or other clinical-field mapping is present.
- `code/fix_real_attention.py` is the historical auxiliary trainer for the latent-chunk proxy and requires the retained ordered 499-case inputs in `integrated_full_legacy499/`. It retrains and overwrites outputs, so it is not the Figure 4 reproduction command. `code/fig2_regenerate.py` is the authoritative non-training renderer.
- `experiment_outputs/biomedbert_attention_weights.npy` is retained only as a legacy audit artifact. Its clinical-group keys were derived from unsupported semantic mappings and are excluded from the request-bundle manifest and manuscript evidence.

## 6. Release checklist

The zero-training path is public: the `publish` branch of `https://github.com/literaryw/oral-multimodal-experiment` contains the manifest-listed scripts, the patient-level predictions and the regenerated table fragments, and the manuscript cites the branch commit. A tagged release can be added later without changing that path. Do not claim public raw-source reconstruction until upstream dataset/model revisions, clinical schemas, IDs, masks, ordering, and all feature-building transformations have been independently verified.
