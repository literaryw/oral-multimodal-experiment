# Replaying the 233-patient follow-up

Run from the repository root with Python 3.12, NumPy and PyArrow (the project uses the tcga-oscc environment).

```powershell
python code/replay_grade_followup.py --run <archived-run-directory> --labels <patient_join.csv>
```

Use the sealed main OOF Parquet file under outer_evaluation_v4, the archived architecture_outer_evaluation_v2/metrics.json, and the A07/A08 prediction.npz files listed in its prediction records under architecture_outer_predictions_v2. The labels are the archived patient_join.csv from the controlled comparison. Retain the original file bytes and relative directory layout. These inputs are held in the research archive and can be requested from the corresponding author.

The JSON output supplies the estimates for Section 4.9 and Table 12. The script verifies archive hashes and complete prediction coverage, then recomputes the two baseline scores, three A08–A07 differences, and the conditional 95% interval using the original 4,000 paired stratified bootstrap draws. The interval is conditional on fixed predictions.

C1, C2 and C-A1 require the candidate eligibility records described in grade_followup_sources.json and grade_followup_results.csv. The replay does not estimate these incomplete comparisons or retrain any model.
