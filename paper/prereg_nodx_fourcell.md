# Preregistration: no-diagnosis four-cell comparison (strict oral subset)

Written before the no-diagnosis neural arms were run. Scope: strict oral cavity subset
(n=246). This is a revision-stage addition, not part of the original study plan.

## Question

Does the True Cross-Attention minus Text-Only difference change when the
primary-diagnosis field is removed from the structured clinical input?

## Cells

| Input | Text-Only | True Cross-Attention |
|---|---|---|
| Diagnosis + TNM + demographics | reused from the primary grid | reused from the primary grid |
| TNM + demographics (no diagnosis) | new run | new run |

The no-diagnosis text is re-rendered from the stored narrative with the diagnosis block
removed (`TNM staging: <tnm>. Patient: <demo>.`) and re-encoded from scratch; the encoder
never sees the field. The diagnosis is not masked at the classifier.

## Fixed protocol

- Seeds 42/123/456; outer `StratifiedKFold(5, shuffle=True, random_state=seed)`.
- Inner `StratifiedKFold(5, shuffle=True, random_state=seed+10000)`, first fold, 80/20.
- 100 epochs, patience 15 on inner Macro F1; unchanged model classes and hyperparameters.
- Threshold 0.5 (argmax over the two logits); same rows and labels as the primary grid.

## Estimands

- Per seed: `D_full = MacroF1(True CA) - MacroF1(Text-Only)` with the diagnosis;
  `D_nodx` the same difference without it. Each metric is a mean over the five outer folds
  (fold first, then arms).
- `delta_D = D_nodx - D_full`, reported per seed with mean and SD across the three seeds
  and a sign count.
- Primary metric: Macro F1. Secondary: High-Grade F1, average precision (per-fold AP
  averaged), AUC, accuracy, Low-Grade F1.

## Analysis commitments

- No directional hypothesis and no significance or equivalence test. Three seeds are
  descriptive; the seed also controls the split and initialization.
- Report all four cells and all three per-seed differences, not only the mean.
- If the seed-42 no-diagnosis Text-Only run does not reproduce the archived auxiliary run
  (prediction agreement 100%, max probability delta <= 1e-6), stop and check the text and
  batching; do not tune.
- Interpretation follows the observed pattern: a stable positive delta, an unchanged
  difference with lower absolute scores, a ranking reversal, or a single-metric change.
  The manuscript states which one occurred.

## Frozen inputs

| File | SHA-256 |
|---|---|
| `code/attention_v2_nodx_encode.py` | `cbe82516b006ebced4e9db25adc2735d688090765785feb082a36cfd7cea46f8` |
| `code/attention_v2_nodx_fourcell.py` | `6fad4b3bef6a4f24edb871121bc45cf4ec0a1c812a9b13107f3a44c335be6e74` |
| `data/TCGA-HNSC/attention_v2/text_tokens_nodx.npz` | `df88d0b0811b24a8e97b19311e13dad7ee109456bc437a7f12e218334985a7ba` |
| `data/TCGA-HNSC/attention_v2/cohort_index.json` | `24d4093d2dcc65b750025be93352c39f171ee902383d2859697a7307e9f29536` |
| `data/TCGA-HNSC/attention_v2/text_tokens.npz` | `5d159d5b559998d4da788c6009da6c7d921da9590969047a6f3a87228171d3fa` |
| `data/TCGA-HNSC/experiment_outputs/attention_v2_aux.json` | `49ddecdc233a56a3a4cb2423c63171b27a0396672a76125e97ce6459b39edc9e` |
| `data/TCGA-HNSC/experiment_outputs/attention_v2_strict_results.json` | `064da660e46886073466e737a22682d22540a404847d6ba750e8bfbcff5e325c` |

Encoding gates already passed: 499/499 template round-trip; no case with empty TNM and
demographics; full-template re-encoding reproduces stored tokens (max |delta| 0.0) and mask
exactly; no-diagnosis sequences 9-33 tokens (median 29).
