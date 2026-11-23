"""Build the request-based manuscript bundle manifest.

This manifest covers archive verification from the retained feature bundle and
outputs. It does not claim reconstruction of those features from public raw data.
"""

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = (
    ROOT / "data" / "TCGA-HNSC" / "experiment_outputs" / "manuscript_bundle_manifest.json"
)

FILES = {
    "analysis_code": [
        "code/README.md",
        "code/fix_age_units.py",
        "code/prep_attention_v2.py",
        "code/attention_v2_models.py",
        "code/attention_v2_experiments.py",
        "code/attention_v2_aux.py",
        "code/attention_v2_loso_moddrop.py",
        "code/attention_v2_analysis.py",
        "code/attention_v2_rederive.py",
        "code/attention_v2_tables.py",
        "code/attention_v2_nodx_encode.py",
        "code/attention_v2_nodx_fourcell.py",
        "code/replay_grade_followup.py",
        "code/attention_v2_rep_match.py",
        "code/attention_v2_structured_baseline.py",
        "code/attention_v2_mi_matched.py",
        "code/attention_v2_skip_ablation.py",
        "code/build_case_slide_manifest.py",
        "code/build_manuscript_bundle_manifest.py",
        "code/verify_manuscript_bundle.py",
        "code/fig1_regenerate.py",
        "code/fig2_regenerate.py",
        "code/fig3_regenerate.py",
        "code/fig4_regenerate.py",
        "code/fix_real_attention.py",
        "code/fix_real_gradcam.py",
        "code/gradcam_oral_expanded.py",
        "code/requirements_full.txt",
    ],
    "retained_feature_inputs": [
        "data/TCGA-HNSC/clinical_data.csv",
        "data/TCGA-HNSC/clinical_data_oral.csv",
        "data/TCGA-HNSC/pretrained_features/Slide_Image_UNI.parquet",
        "data/TCGA-HNSC/attention_v2/cohort_index.json",
        "data/TCGA-HNSC/attention_v2/img_patches.npz",
        "data/TCGA-HNSC/attention_v2/text_tokens.npz",
        "data/TCGA-HNSC/attention_v2/text_tokens_nodx.npz",
        "data/TCGA-HNSC/attention_v2/text_tokens_nodx_meta.json",
        "data/TCGA-HNSC/integrated_full_legacy499/X_img_uni.npy",
        "data/TCGA-HNSC/integrated_full_legacy499/X_text_biomedbert.npy",
        "data/TCGA-HNSC/integrated_full_legacy499/X_text_biomedbert_ageyears.npy",
        "data/TCGA-HNSC/integrated_full_legacy499/y.npy",
        "data/TCGA-HNSC/integrated_full_legacy499/text_metadata_499era.csv",
        "data/TCGA-HNSC/integrated_full_legacy499/case_labels.csv",
        "data/TCGA-HNSC/integrated_full_legacy499/text_metadata_499era_ageyears.csv",
        "data/TCGA-HNSC/integrated/X_img_patches.npz",
        "data/TCGA-HNSC/integrated/X_img_uni.npy",
        "data/TCGA-HNSC/integrated/X_text_biomedbert.npy",
        "data/TCGA-HNSC/integrated/y.npy",
        "data/TCGA-HNSC/integrated/case_labels.csv",
        "data/TCGA-HNSC/integrated/oral_patches.npz",
    ],
    "primary_and_auxiliary_outputs": [
        "data/TCGA-HNSC/experiment_outputs/attention_v2_full_results.json",
        "data/TCGA-HNSC/experiment_outputs/attention_v2_wide_results.json",
        "data/TCGA-HNSC/experiment_outputs/attention_v2_strict_results.json",
        "data/TCGA-HNSC/experiment_outputs/attention_v2_analysis.json",
        "data/TCGA-HNSC/experiment_outputs/attention_v2_aux.json",
        "data/TCGA-HNSC/experiment_outputs/attention_v2_loso_moddrop.json",
        "data/TCGA-HNSC/experiment_outputs/attention_v2_mi_matched.json",
        "data/TCGA-HNSC/experiment_outputs/attention_v2_rep_match.json",
        "data/TCGA-HNSC/experiment_outputs/attention_v2_skip_ablation.json",
        "data/TCGA-HNSC/experiment_outputs/attention_v2_structured_baseline.json",
        "data/TCGA-HNSC/experiment_outputs/attention_v2_nodx_fourcell.json",
        "data/TCGA-HNSC/experiment_outputs/attention_v2_nodx_fourcell.tex",
        "data/TCGA-HNSC/experiment_outputs/attention_v2_nodx_seed.tex",
        "data/TCGA-HNSC/experiment_outputs/attention_v2_nodx_metric.tex",
        "data/TCGA-HNSC/experiment_outputs/attention_v2_nodx_analysis_plan.json",
        "data/TCGA-HNSC/experiment_outputs/attention_v2_rederive.json",
        "data/TCGA-HNSC/experiment_outputs/attention_v2_rederive.tex",
        "data/TCGA-HNSC/experiment_outputs/case_slide_manifest.csv",
        "data/TCGA-HNSC/experiment_outputs/main_grade_replay_verification_2026_11_22.json",
        "data/TCGA-HNSC/experiment_outputs/analysis_plan_nodx_fourcell.md",
    ],
    "figure_sources": [
        "data/TCGA-HNSC/experiment_outputs/text_latent_attention_archive.npz",
        "data/TCGA-HNSC/experiment_outputs/text_attention_model.pt",
        "data/TCGA-HNSC/experiment_outputs/dndf_learned_params.npz",
        "data/TCGA-HNSC/experiment_outputs/image_only_model.pt",
        "data/TCGA-HNSC/experiment_outputs/gradcam_g1_scores.npy",
        "data/TCGA-HNSC/experiment_outputs/gradcam_g3_scores.npy",
        "data/TCGA-HNSC/experiment_outputs/oral_gradcam_results.json",
        "data/TCGA-HNSC/experiment_outputs/oral_gradcam_scores.npz",
        "data/TCGA-HNSC/experiment_outputs/oral_image_model.pt",
    ],
}


TEXT_SUFFIXES = {".py", ".md", ".json", ".csv", ".tex", ".txt", ".yaml", ".yml"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_lf(path: Path) -> tuple[str, int]:
    """Hash the same bytes with CRLF collapsed to LF, which is what Git stores."""
    digest = hashlib.sha256()
    size = 0
    carry = b""
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            data = carry + block
            carry = b""
            if data.endswith(b"\r"):
                data, carry = data[:-1], b"\r"
            data = data.replace(b"\r\n", b"\n")
            size += len(data)
            digest.update(data)
    size += len(carry)
    digest.update(carry)
    return digest.hexdigest(), size


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    root = args.root.resolve()

    entries = []
    for role, paths in FILES.items():
        for relative in paths:
            absolute = (root / relative).resolve()
            if not absolute.is_file():
                raise FileNotFoundError(relative)
            record = {
                "path": relative.replace("\\", "/"),
                "role": role,
                "size": absolute.stat().st_size,
                "sha256": sha256(absolute),
            }
            lf_digest, lf_size = sha256_lf(absolute)
            if absolute.suffix.lower() in TEXT_SUFFIXES and lf_digest != record["sha256"]:
                record["size_lf"] = lf_size
                record["sha256_lf"] = lf_digest
            entries.append(record)

    manifest = {
        "schema_version": 1,
        "scope": "request-based archive verification; not public raw-source reconstruction",
        "line_ending_note": (
            "size and sha256 are those of the working-tree bytes, which include CRLF on a "
            "Windows checkout. Text files whose Git blob stores LF instead also record "
            "size_lf and sha256_lf, so a clone with either checkout setting can be verified. "
            "Binary files are hashed as-is."
        ),
        "expected_cohorts": {"full": 499, "wide": 332, "strict": 246},
        "ordering_authority": "data/TCGA-HNSC/attention_v2/cohort_index.json:case_ids",
        "unresolved_upstream_revisions": [
            "Lab-Rasool/TCGA feature dataset revision",
            "BiomedBERT model revision used for historical encodings",
        ],
        "files": entries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output} with {len(entries)} files")


if __name__ == "__main__":
    main()
