"""Verify the request-based manuscript bundle in an isolated directory.

The verifier reads only files listed in manuscript_bundle_manifest.json. With
``--stage-from``, it first hard-links (or copies) the listed files into a separate
bundle root and then performs every content check against that staged root.
"""

import argparse
import csv
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


MANIFEST = Path("data/TCGA-HNSC/experiment_outputs/manuscript_bundle_manifest.json")
EXPECTED_ARMS = {
    "image_only",
    "text_only",
    "late_fusion",
    "gated_bilinear",
    "true_cross_attention",
}
EXPECTED_SEEDS = {"42", "123", "456"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def contained(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    path.relative_to(root.resolve())
    return path


def stage_bundle(source: Path, destination: Path) -> None:
    source_manifest = json.loads((source / MANIFEST).read_text(encoding="utf-8"))
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Staging directory is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    for entry in source_manifest["files"]:
        src = contained(source, entry["path"])
        dst = contained(destination, entry["path"])
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    target_manifest = contained(destination, MANIFEST.as_posix())
    target_manifest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / MANIFEST, target_manifest)


def validate(root: Path) -> dict:
    errors = []
    manifest_path = contained(root, MANIFEST.as_posix())
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    for entry in manifest["files"]:
        path = contained(root, entry["path"])
        if not path.is_file():
            errors.append(f"missing: {entry['path']}")
            continue
        if path.stat().st_size != entry["size"]:
            errors.append(f"size mismatch: {entry['path']}")
        if sha256(path) != entry["sha256"]:
            errors.append(f"sha256 mismatch: {entry['path']}")

    data = root / "data" / "TCGA-HNSC"
    index = json.loads((data / "attention_v2" / "cohort_index.json").read_text(encoding="utf-8"))
    case_ids = [str(value) for value in index["case_ids"]]
    y4 = np.asarray(index["y4"])
    y_binary = np.asarray(index["y_binary"])
    wide = np.asarray(index["wide_mask"], dtype=bool)
    strict = np.asarray(index["strict_mask"], dtype=bool)
    n_base = int(index["n_base"])

    if n_base != 499 or len(case_ids) != n_base or len(set(case_ids)) != n_base:
        errors.append("base case IDs are not 499 unique ordered IDs")
    if any(len(values) != n_base for values in [y4, y_binary, wide, strict]):
        errors.append("cohort-index arrays do not all have length 499")
    if int(wide.sum()) != 332 or int(strict.sum()) != 246:
        errors.append("cohort mask counts are not 499/332/246")
    if np.any(strict & ~wide):
        errors.append("strict cohort is not a subset of wide cohort")
    expected_binary = np.where(np.isin(y4, [0, 1]), 0, 1)
    if not np.array_equal(y_binary, expected_binary):
        errors.append("binary labels do not match the declared G1+G2 versus G3+G4 rule")

    image = np.load(data / "integrated_full_legacy499" / "X_img_uni.npy", mmap_mode="r")
    text = np.load(
        data / "integrated_full_legacy499" / "X_text_biomedbert_ageyears.npy", mmap_mode="r"
    )
    archived_y4 = np.load(data / "integrated_full_legacy499" / "y.npy", mmap_mode="r")
    if image.shape != (499, 1024) or text.shape != (499, 768):
        errors.append(f"pooled feature shapes are {image.shape} and {text.shape}")
    if not np.array_equal(np.asarray(archived_y4).reshape(-1), y4):
        errors.append("integrated y.npy order differs from cohort_index y4")

    with (data / "integrated_full_legacy499" / "case_labels.csv").open(
        encoding="utf-8-sig", newline=""
    ) as stream:
        label_rows = list(csv.DictReader(stream))
    if list(label_rows[0]) != ["case_id", "label"]:
        errors.append("case_labels.csv schema mismatch")
    label_ids = [row["case_id"] for row in label_rows]
    if label_ids != case_ids or len(set(label_ids)) != 499:
        errors.append("case_labels.csv does not contain 499 unique IDs in cohort order")
    if [int(row["label"]) for row in label_rows] != y4.tolist():
        errors.append("case_labels.csv labels differ from cohort_index y4")

    with (data / "integrated_full_legacy499" / "text_metadata_499era_ageyears.csv").open(
        encoding="utf-8-sig", newline=""
    ) as stream:
        metadata_rows = list(csv.DictReader(stream))
    required_metadata = ["case_id", "tumor_grade", "clinical_text"]
    if list(metadata_rows[0]) != required_metadata:
        errors.append("text metadata schema mismatch")
    metadata_ids = [row["case_id"] for row in metadata_rows]
    selected_metadata_ids = [value for value in metadata_ids if value in set(case_ids)]
    if len(metadata_rows) != 506 or len(set(metadata_ids)) != 506:
        errors.append("text metadata is not 506 unique case rows")
    if selected_metadata_ids != case_ids:
        errors.append("text metadata does not preserve the 499-case cohort order")

    with np.load(data / "attention_v2" / "text_tokens.npz") as tokens:
        if tokens["tokens"].shape != (499, 48, 768):
            errors.append(f"text token shape is {tokens['tokens'].shape}")
        if tokens["mask"].shape != (499, 48) or tokens["cls"].shape != (499, 768):
            errors.append("text mask or CLS shape mismatch")
    with np.load(data / "attention_v2" / "img_patches.npz") as patches:
        expected_keys = [f"case_{i}" for i in range(499)]
        if patches.files != expected_keys:
            errors.append("patch keys do not preserve case_0 ... case_498 order")
        elif any(patches[key].ndim != 2 or patches[key].shape[1] != 1024 for key in patches.files):
            errors.append("one or more patch arrays are not (n_patches, 1024)")

    with (data / "experiment_outputs" / "case_slide_manifest.csv").open(
        encoding="utf-8-sig", newline=""
    ) as stream:
        slide_rows = list(csv.DictReader(stream))
    slide_ids = [row["case_id"] for row in slide_rows]
    if slide_ids != case_ids or len(set(slide_ids)) != 499:
        errors.append("slide manifest does not match cohort_index case order")

    parquet_path = data / "pretrained_features" / "Slide_Image_UNI.parquet"
    slides = pd.read_parquet(parquet_path, columns=["PatientID", "id", "file_name"])
    slides["PatientID"] = slides["PatientID"].astype(str)
    slide_rows_by_case = {
        case_id: group[["id", "file_name"]].astype(str).values.tolist()
        for case_id, group in slides.groupby("PatientID", sort=False)
    }
    for row in slide_rows:
        source_rows = slide_rows_by_case.get(row["case_id"], [])
        source_ids = [item[0] for item in source_rows]
        source_files = [item[1] for item in source_rows]
        if not source_rows:
            errors.append(f"slide manifest case is absent from source parquet: {row['case_id']}")
            continue
        if int(row["slide_count"]) != len(source_rows):
            errors.append(f"slide count mismatch for {row['case_id']}")
        if row["selected_patch_slide_id"] != source_ids[0]:
            errors.append(f"selected first slide ID mismatch for {row['case_id']}")
        if row["selected_patch_slide_file"] != source_files[0]:
            errors.append(f"selected first slide file mismatch for {row['case_id']}")
        if row["all_slide_ids_in_source_order"].split(";") != source_ids:
            errors.append(f"source slide-ID order mismatch for {row['case_id']}")
        if row["all_slide_files_in_source_order"].split(";") != source_files:
            errors.append(f"source slide-file order mismatch for {row['case_id']}")

    for cohort, count in manifest["expected_cohorts"].items():
        path = data / "experiment_outputs" / f"attention_v2_{cohort}_results.json"
        result = json.loads(path.read_text(encoding="utf-8"))[cohort]
        if set(result) != EXPECTED_ARMS:
            errors.append(f"{cohort}: unexpected primary-grid arms")
            continue
        for arm, records in result.items():
            if set(records) != EXPECTED_SEEDS:
                errors.append(f"{cohort}/{arm}: unexpected seeds")
                continue
            for seed, record in records.items():
                if len(record["oof_pred"]) != count or len(record["oof_prob"]) != count:
                    errors.append(f"{cohort}/{arm}/{seed}: OOF length mismatch")
                if len(record["per_fold"]) != 5:
                    errors.append(f"{cohort}/{arm}/{seed}: expected five folds")
                pred = np.asarray(record["oof_pred"], dtype=int)
                prob = np.asarray(record["oof_prob"], dtype=float)
                if not np.isfinite(prob).all() or ((prob < 0) | (prob > 1)).any():
                    errors.append(f"{cohort}/{arm}/{seed}: invalid OOF probability")
                expected = np.argmax(np.stack([1.0 - prob, prob], axis=0), axis=0)
                if not np.array_equal(pred, expected):
                    errors.append(f"{cohort}/{arm}/{seed}: OOF prediction is not argmax(prob)")
                for fold in record["per_fold"]:
                    if fold.get("n_fit", 0) <= 0 or fold.get("n_sel", 0) <= 0:
                        errors.append(f"{cohort}/{arm}/{seed}: invalid inner split size")
                        break

    structured = json.loads(
        (data / "experiment_outputs" / "attention_v2_structured_baseline.json").read_text(
            encoding="utf-8"))
    cohort_counts = {"full": 499, "wide": 332, "strict": 246}
    for kind, cohorts in structured.items():
        for cohort, models in cohorts.items():
            for name, record in models.items():
                if set(record.get("oof_by_seed", {})) != EXPECTED_SEEDS:
                    errors.append(f"structured/{kind}/{cohort}/{name}: missing per-seed OOF")
                    continue
                for seed, oof in record["oof_by_seed"].items():
                    pred = np.asarray(oof["oof_pred"], dtype=int)
                    prob = np.asarray(oof["oof_prob"], dtype=float)
                    folds = np.asarray(oof["fold_index"], dtype=int)
                    count = cohort_counts[cohort]
                    if len(pred) != count or len(prob) != count or len(folds) != count:
                        errors.append(
                            f"structured/{kind}/{cohort}/{name}/{seed}: OOF length mismatch")
                        continue
                    expected = np.argmax(np.stack([1.0 - prob, prob], axis=0), axis=0)
                    if not np.array_equal(pred, expected):
                        errors.append(
                            f"structured/{kind}/{cohort}/{name}/{seed}: prediction is not argmax(prob)")
                    if sorted(set(folds.tolist())) != [0, 1, 2, 3, 4]:
                        errors.append(
                            f"structured/{kind}/{cohort}/{name}/{seed}: folds do not cover 0-4")

    nodx = json.loads(
        (data / "experiment_outputs" / "attention_v2_nodx_fourcell.json").read_text(
            encoding="utf-8"))
    for block, arms in nodx["cells"].items():
        for arm, seeds in arms.items():
            for seed, cell in seeds.items():
                pred = np.asarray(cell["oof_pred"], dtype=int)
                prob = np.asarray(cell["oof_prob"], dtype=float)
                if len(pred) != 246 or len(prob) != 246:
                    errors.append(f"nodx/{block}/{arm}/{seed}: OOF length mismatch")
                    continue
                expected = np.argmax(np.stack([1.0 - prob, prob], axis=0), axis=0)
                if not np.array_equal(pred, expected):
                    errors.append(f"nodx/{block}/{arm}/{seed}: prediction is not argmax(prob)")
    for metric, block in nodx["per_metric_differences"].items():
        values = np.asarray(block["delta_D_values"], dtype=float)
        if len(values) != 3 or abs(float(values.mean()) - block["delta_D_mean"]) > 1e-12:
            errors.append(f"nodx/{metric}: per-seed differences do not reproduce their mean")

    with np.load(data / "experiment_outputs" / "text_latent_attention_archive.npz") as attn:
        raw = attn["raw_attn"]
        avg = attn["avg_per_key"]
        if raw.shape != (499, 4, 4):
            errors.append(f"latent attention shape is {raw.shape}")
        if not np.allclose(raw.mean(axis=0).mean(axis=0), avg, rtol=0, atol=1e-7):
            errors.append("latent attention average does not match raw matrices")

    return {
        "passed": not errors,
        "scope": manifest["scope"],
        "files_verified": len(manifest["files"]),
        "case_ids": len(case_ids),
        "unique_case_ids": len(set(case_ids)),
        "cohort_counts": {"full": n_base, "wide": int(wide.sum()), "strict": int(strict.sum())},
        "strict_subset_of_wide": bool(np.all(~strict | wide)),
        "slide_manifest_order_match": slide_ids == case_ids,
        "slide_manifest_matches_source_parquet": not any(
            "slide " in error or "source parquet" in error for error in errors
        ),
        "case_label_order_match": label_ids == case_ids,
        "text_metadata_rows": len(metadata_rows),
        "text_metadata_unique_ids": len(set(metadata_ids)),
        "text_metadata_selected_order_match": selected_metadata_ids == case_ids,
        "unresolved_upstream_revisions": manifest["unresolved_upstream_revisions"],
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--stage-from", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    bundle_root = args.bundle_root.resolve()
    if args.stage_from:
        stage_bundle(args.stage_from.resolve(), bundle_root)
    report = validate(bundle_root)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
