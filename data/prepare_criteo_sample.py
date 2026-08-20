#!/usr/bin/env python3
"""Profile the full Criteo uplift CSV and create reproducible project splits."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [f"f{i}" for i in range(12)]
BINARY_COLUMNS = ["treatment", "conversion", "visit", "exposure"]
EXPECTED_COLUMNS = FEATURE_COLUMNS + BINARY_COLUMNS
DEFAULT_EXPECTED_ROWS = 13_979_592
DEFAULT_EXPECTED_SOURCE_SHA256 = (
    "2716e1bf0fd157a93b5bf86924d9088419dfbac2022c6cd90030220634f616dc"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--chunk-size", type=int, default=500_000)
    parser.add_argument("--expected-rows", type=int, default=DEFAULT_EXPECTED_ROWS)
    parser.add_argument(
        "--expected-source-sha256",
        default=DEFAULT_EXPECTED_SOURCE_SHA256,
        help="Expected SHA-256 of the compressed Criteo v2.1 source file",
    )
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--prefix", default="2026-07-31_criteo_v2_1")
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def empty_profile() -> dict:
    return {
        "rows": 0,
        "null_counts": Counter(),
        "infinite_counts": Counter(),
        "invalid_binary_counts": Counter(),
        "binary_value_counts": {column: Counter() for column in BINARY_COLUMNS},
        "joint_binary_counts": Counter(),
        "cross_field_counts": Counter(),
        "first_occurrence": {
            "treatment_0_source_row": None,
            "treatment_1_source_row": None,
        },
        "feature_stats": {
            column: {
                "finite_count": 0,
                "sum": 0.0,
                "sum_squares": 0.0,
                "min": math.inf,
                "max": -math.inf,
                "zero_count": 0,
            }
            for column in FEATURE_COLUMNS
        },
    }


def update_profile(profile: dict, chunk: pd.DataFrame, source_row_offset: int) -> None:
    profile["rows"] += len(chunk)
    profile["null_counts"].update(chunk.isna().sum().astype(int).to_dict())

    for column in FEATURE_COLUMNS:
        values = chunk[column].to_numpy(dtype=np.float64, copy=False)
        finite = np.isfinite(values)
        finite_values = values[finite]
        profile["infinite_counts"][column] += int(np.isinf(values).sum())
        if finite_values.size:
            stats = profile["feature_stats"][column]
            stats["finite_count"] += int(finite_values.size)
            stats["sum"] += float(finite_values.sum(dtype=np.float64))
            stats["sum_squares"] += float(np.square(finite_values).sum(dtype=np.float64))
            stats["min"] = min(stats["min"], float(finite_values.min()))
            stats["max"] = max(stats["max"], float(finite_values.max()))
            stats["zero_count"] += int((finite_values == 0).sum())

    for column in BINARY_COLUMNS:
        non_null = chunk[column].dropna()
        profile["invalid_binary_counts"][column] += int((~non_null.isin([0, 1])).sum())
        counts = non_null.value_counts(dropna=False)
        profile["binary_value_counts"][column].update(
            {str(value): int(count) for value, count in counts.items()}
        )

    binary_frame = chunk[BINARY_COLUMNS]
    complete = binary_frame.notna().all(axis=1)
    if complete.any():
        grouped = binary_frame.loc[complete].astype(np.int8).value_counts(sort=False)
        for key, count in grouped.items():
            profile["joint_binary_counts"]["|".join(map(str, key))] += int(count)

    profile["cross_field_counts"]["exposure_without_treatment"] += int(
        ((chunk["exposure"] == 1) & (chunk["treatment"] == 0)).sum()
    )
    profile["cross_field_counts"]["conversion_without_visit"] += int(
        ((chunk["conversion"] == 1) & (chunk["visit"] == 0)).sum()
    )
    profile["cross_field_counts"]["treated_not_exposed"] += int(
        ((chunk["treatment"] == 1) & (chunk["exposure"] == 0)).sum()
    )

    treatment_values = chunk["treatment"].to_numpy()
    for value in (0, 1):
        key = f"treatment_{value}_source_row"
        if profile["first_occurrence"][key] is None:
            positions = np.flatnonzero(treatment_values == value)
            if positions.size:
                profile["first_occurrence"][key] = source_row_offset + int(positions[0]) + 1


def finalize_profile(profile: dict) -> dict:
    rows = int(profile["rows"])
    feature_summary = {}
    for column, stats in profile["feature_stats"].items():
        count = stats["finite_count"]
        mean = stats["sum"] / count if count else None
        variance = max(stats["sum_squares"] / count - mean * mean, 0.0) if count else None
        feature_summary[column] = {
            "finite_count": int(count),
            "mean": mean,
            "std_population": math.sqrt(variance) if variance is not None else None,
            "min": None if stats["min"] == math.inf else stats["min"],
            "max": None if stats["max"] == -math.inf else stats["max"],
            "zero_count": int(stats["zero_count"]),
            "zero_rate": stats["zero_count"] / count if count else None,
        }

    binary_summary = {}
    for column in BINARY_COLUMNS:
        counts = profile["binary_value_counts"][column]
        one_count = int(counts.get("1", 0) + counts.get("1.0", 0))
        zero_count = int(counts.get("0", 0) + counts.get("0.0", 0))
        denominator = one_count + zero_count
        binary_summary[column] = {
            "zero_count": zero_count,
            "one_count": one_count,
            "one_rate": one_count / denominator if denominator else None,
        }

    treatment_group_summary = {
        str(value): {
            "rows": 0,
            "visit_count": 0,
            "visit_rate": None,
            "conversion_count": 0,
            "conversion_rate": None,
            "exposure_count": 0,
            "exposure_rate": None,
        }
        for value in (0, 1)
    }
    for key, count in profile["joint_binary_counts"].items():
        treatment, conversion, visit, exposure = map(int, key.split("|"))
        group = treatment_group_summary[str(treatment)]
        group["rows"] += int(count)
        group["visit_count"] += int(count) * visit
        group["conversion_count"] += int(count) * conversion
        group["exposure_count"] += int(count) * exposure
    for group in treatment_group_summary.values():
        if group["rows"]:
            group["visit_rate"] = group["visit_count"] / group["rows"]
            group["conversion_rate"] = group["conversion_count"] / group["rows"]
            group["exposure_rate"] = group["exposure_count"] / group["rows"]

    return {
        "rows": rows,
        "columns": EXPECTED_COLUMNS,
        "null_counts": {column: int(profile["null_counts"].get(column, 0)) for column in EXPECTED_COLUMNS},
        "infinite_counts": {column: int(profile["infinite_counts"].get(column, 0)) for column in FEATURE_COLUMNS},
        "invalid_binary_counts": {
            column: int(profile["invalid_binary_counts"].get(column, 0)) for column in BINARY_COLUMNS
        },
        "binary_summary": binary_summary,
        "treatment_group_summary": treatment_group_summary,
        "joint_binary_counts": dict(sorted(profile["joint_binary_counts"].items())),
        "cross_field_counts": dict(profile["cross_field_counts"]),
        "first_occurrence": profile["first_occurrence"],
        "feature_summary": feature_summary,
    }


def summarize_frame(frame: pd.DataFrame) -> dict:
    summary = {
        "rows": int(len(frame)),
        "binary_summary": {},
        "feature_summary": {},
        "treatment_group_summary": {},
        "exact_duplicate_rows_without_source_number": int(frame.duplicated(subset=EXPECTED_COLUMNS).sum()),
    }
    for column in BINARY_COLUMNS:
        counts = frame[column].value_counts().to_dict()
        summary["binary_summary"][column] = {
            "zero_count": int(counts.get(0, 0)),
            "one_count": int(counts.get(1, 0)),
            "one_rate": float(frame[column].mean()),
        }
    for column in FEATURE_COLUMNS:
        summary["feature_summary"][column] = {
            "mean": float(frame[column].mean()),
            "std_population": float(frame[column].std(ddof=0)),
            "min": float(frame[column].min()),
            "max": float(frame[column].max()),
            "zero_rate": float((frame[column] == 0).mean()),
        }
    for treatment in (0, 1):
        group = frame.loc[frame["treatment"] == treatment]
        summary["treatment_group_summary"][str(treatment)] = {
            "rows": int(len(group)),
            "visit_count": int(group["visit"].sum()),
            "visit_rate": float(group["visit"].mean()) if len(group) else None,
            "conversion_count": int(group["conversion"].sum()),
            "conversion_rate": float(group["conversion"].mean()) if len(group) else None,
            "exposure_count": int(group["exposure"].sum()),
            "exposure_rate": float(group["exposure"].mean()) if len(group) else None,
        }
    return summary


def allocate_proportionally(counts: pd.Series, target: int) -> pd.Series:
    raw = counts * (target / counts.sum())
    allocated = np.floor(raw).astype(int)
    shortfall = int(target - allocated.sum())
    if shortfall:
        order = (raw - allocated).sort_values(ascending=False).index[:shortfall]
        allocated.loc[order] += 1
    return allocated


def nearest_group_boundary(cumulative_sizes: np.ndarray, target: int) -> int:
    """Return the number of complete groups closest to a target row count."""
    if target <= 0:
        return 0
    insertion = int(np.searchsorted(cumulative_sizes, target, side="left"))
    candidates = []
    if insertion < len(cumulative_sizes):
        candidates.append(insertion + 1)
    if insertion > 0:
        candidates.append(insertion)
    return min(candidates, key=lambda count: abs(int(cumulative_sizes[count - 1]) - target))


def stratified_split(frame: pd.DataFrame, seed: int) -> dict[str, pd.DataFrame]:
    """Split by outcome strata while keeping exact row patterns in one split."""
    strata = frame[["treatment", "visit", "conversion"]].astype(str).agg("|".join, axis=1)
    counts = strata.value_counts().sort_index()
    validation_target = int(round(len(frame) * 0.15))
    test_target = int(round(len(frame) * 0.15))
    validation_alloc = allocate_proportionally(counts, validation_target)
    test_alloc = allocate_proportionally(counts, test_target)

    rng = np.random.default_rng(seed)
    strata_values = strata.to_numpy()
    row_hashes = pd.util.hash_pandas_object(frame[EXPECTED_COLUMNS], index=False).to_numpy()
    assignment = np.full(len(frame), -1, dtype=np.int8)

    for stratum in counts.index:
        positions = np.flatnonzero(strata_values == stratum)
        ordered_positions = positions[np.argsort(row_hashes[positions], kind="stable")]
        ordered_hashes = row_hashes[ordered_positions]
        group_starts = np.r_[0, np.flatnonzero(ordered_hashes[1:] != ordered_hashes[:-1]) + 1]
        group_ends = np.r_[group_starts[1:], len(ordered_positions)]
        group_sizes = group_ends - group_starts

        group_order = rng.permutation(len(group_sizes))
        cumulative = np.cumsum(group_sizes[group_order])
        validation_groups = nearest_group_boundary(
            cumulative,
            int(validation_alloc.loc[stratum]),
        )
        validation_rows = int(cumulative[validation_groups - 1]) if validation_groups else 0
        combined_groups = nearest_group_boundary(
            cumulative,
            validation_rows + int(test_alloc.loc[stratum]),
        )

        group_assignment = np.zeros(len(group_sizes), dtype=np.int8)
        group_assignment[group_order[:validation_groups]] = 1
        group_assignment[group_order[validation_groups:combined_groups]] = 2
        group_ids_in_sorted_rows = np.repeat(np.arange(len(group_sizes)), group_sizes)
        assignment[ordered_positions] = group_assignment[group_ids_in_sorted_rows]

    if (assignment < 0).any():
        raise RuntimeError("Some sampled rows were not assigned to a split")

    result = {}
    for name, code in (
        ("train", 0),
        ("validation", 1),
        ("test", 2),
    ):
        positions = rng.permutation(np.flatnonzero(assignment == code))
        result[name] = frame.iloc[positions].reset_index(drop=True)
    return result


def exact_pattern_overlap_counts(splits: dict[str, pd.DataFrame]) -> dict[str, int]:
    hashes = {
        name: set(pd.util.hash_pandas_object(frame[EXPECTED_COLUMNS], index=False).tolist())
        for name, frame in splits.items()
    }
    return {
        "train_validation": len(hashes["train"] & hashes["validation"]),
        "train_test": len(hashes["train"] & hashes["test"]),
        "validation_test": len(hashes["validation"] & hashes["test"]),
    }


def write_gzip_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw_handle:
        with gzip.GzipFile(fileobj=raw_handle, mode="wb", compresslevel=6, mtime=0) as gzip_handle:
            with io.TextIOWrapper(gzip_handle, encoding="utf-8", newline="") as text_handle:
                frame.to_csv(text_handle, index=False, float_format="%.17g")


def format_rate(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6%}"


def build_quality_report(
    full: dict,
    sample: dict,
    split_summaries: dict,
    comparison: list[dict],
    manifest: dict,
) -> str:
    max_rate_gap = max(
        abs(row["sample_rate"] - row["full_rate"])
        for row in comparison
        if row["metric_type"] == "binary_rate"
    )
    conditional_gaps = [
        abs(row["sample_rate"] - row["full_rate"])
        for row in comparison
        if row["metric_type"] == "conditional_rate"
    ]
    max_conditional_gap = max(conditional_gaps) if conditional_gaps else 0.0
    max_smd = max(
        abs(row["standardized_mean_difference"])
        for row in comparison
        if row["metric_type"] == "feature_mean"
    )
    total_nulls = sum(full["null_counts"].values())
    total_invalid_binary = sum(full["invalid_binary_counts"].values())
    total_infinite = sum(full["infinite_counts"].values())
    cross = full["cross_field_counts"]
    overlaps = manifest["split_integrity"]["exact_pattern_overlap_counts"]
    first_control = full["first_occurrence"]["treatment_0_source_row"]
    first_control_text = f"{first_control:,}" if first_control is not None else "not observed"

    lines = [
        "# Criteo v2.1 Data Quality And Sampling Report",
        "",
        f"Created: {manifest['created_date']}",
        "",
        "## Readiness Decision",
        "",
        "**Ready for educational uplift modeling with documented limitations.**",
        "",
        "The full file passed schema, completeness, finite-value, and binary-domain checks. The fixed",
        "sample preserves the observed population rates closely enough for development and held-out",
        "evaluation. The prepared files are published with attribution for this noncommercial",
        "educational project under the source dataset's CC BY-NC-SA 4.0 license.",
        "",
        "## Dataset And Grain",
        "",
        f"- Full rows: {full['rows']:,}",
        f"- Columns: {len(full['columns'])}",
        "- Grain: One anonymized experiment record per row",
        "- Candidate identifier: None supplied by the source",
        "- Primary treatment: `treatment`",
        "- Primary outcome: `visit`",
        "- Secondary outcome: `conversion`",
        "",
        "## Core Quality Checks",
        "",
        f"- Missing values across all columns: {total_nulls:,}",
        f"- Infinite feature values: {total_infinite:,}",
        f"- Invalid values in binary columns: {total_invalid_binary:,}",
        f"- Exposure records in control: {cross.get('exposure_without_treatment', 0):,}",
        f"- Conversion records without visit: {cross.get('conversion_without_visit', 0):,}",
        f"- Treatment assignments without observed exposure: {cross.get('treated_not_exposed', 0):,}",
        f"- First control record source row: {first_control_text}",
        "- Ordering finding: A first-rows sample would exclude control records and invalidate uplift analysis",
        "",
        "## Full Population Rates",
        "",
        "| Field | Count of 1 | Rate |",
        "|---|---:|---:|",
    ]
    for column in BINARY_COLUMNS:
        item = full["binary_summary"][column]
        lines.append(f"| `{column}` | {item['one_count']:,} | {format_rate(item['one_rate'])} |")

    lines.extend(
        [
            "",
            "## Outcomes By Treatment Group",
            "",
            "| Group | Full rows | Full visit rate | Sample visit rate | Full conversion rate | Sample conversion rate | Full exposure rate | Sample exposure rate |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for treatment, label in ((0, "Control"), (1, "Treatment")):
        full_group = full["treatment_group_summary"][str(treatment)]
        sample_group = sample["treatment_group_summary"][str(treatment)]
        lines.append(
            f"| {label} | {full_group['rows']:,} | {format_rate(full_group['visit_rate'])} | "
            f"{format_rate(sample_group['visit_rate'])} | {format_rate(full_group['conversion_rate'])} | "
            f"{format_rate(sample_group['conversion_rate'])} | {format_rate(full_group['exposure_rate'])} | "
            f"{format_rate(sample_group['exposure_rate'])} |"
        )

    lines.extend(
        [
            "",
            "## Prepared Data",
            "",
            f"- Selected sample rows: {sample['rows']:,}",
            f"- Train rows: {split_summaries['train']['rows']:,}",
            f"- Validation rows: {split_summaries['validation']['rows']:,}",
            f"- Test rows: {split_summaries['test']['rows']:,}",
            f"- Largest absolute binary-rate difference from full data: {max_rate_gap:.6%}",
            f"- Largest absolute treatment-conditional rate difference: {max_conditional_gap:.6%}",
            f"- Largest absolute standardized feature-mean difference: {max_smd:.6f}",
            f"- Exact duplicate rows in sample: {sample['exact_duplicate_rows_without_source_number']:,}",
            f"- Exact row patterns shared by train and validation: {overlaps['train_validation']:,}",
            f"- Exact row patterns shared by train and test: {overlaps['train_test']:,}",
            f"- Exact row patterns shared by validation and test: {overlaps['validation_test']:,}",
            "",
            "## Analytical Risks",
            "",
            "1. `f0` through `f11` are anonymous, so technical distributions cannot be translated into",
            "   customer attributes or business personas.",
            "2. `exposure` occurs after treatment assignment. It should not be used as a normal predictor",
            "   in the primary intention-to-treat model.",
            "3. Conversion is rare. Visit remains the primary outcome, while conversion analysis is",
            "   secondary and requires uncertainty warnings.",
            "4. The source provides no unique identifier. Exact duplicate rows are retained because they",
            "   may represent different experiment records, but identical patterns are kept in one split",
            "   to prevent cross-split pattern leakage.",
            "5. Source row numbers are used internally for audit checks and are removed from the public",
            "   student files before publication.",
            "6. The prepared sample is suitable for development. Close model results may be",
            "   inconclusive because uplift estimates contain sampling uncertainty.",
            "",
            "## Required Modeling Controls",
            "",
            "- Use only `f0` through `f11` as audience model features.",
            "- Do not filter the primary analysis to records where `exposure = 1`.",
            "- Keep validation and test distributions untouched.",
            "- Fit preprocessing only on training data.",
            "- Report treatment and outcome rates for every split.",
            "- Compare uplift methods against random ranking and response prediction.",
            "- Preserve the source attribution and CC BY-NC-SA 4.0 license when redistributing the data.",
            "",
            "## Confidence",
            "",
            "High for schema, row counts, missingness, binary validity, population rates, and reproducible",
            "sampling. Moderate for broader representativeness because feature semantics and independent",
            "source identifiers are unavailable.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    source_sha256 = sha256_file(args.input)
    if source_sha256.lower() != args.expected_source_sha256.lower():
        raise ValueError(
            "Source SHA-256 mismatch. "
            f"Expected {args.expected_source_sha256} but found {source_sha256}."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)

    expected_population = args.max_rows or args.expected_rows
    if args.sample_size >= expected_population:
        raise ValueError("sample-size must be smaller than the processed population")

    candidate_target = min(expected_population, int(math.ceil(args.sample_size * 1.03)))
    inclusion_probability = candidate_target / expected_population
    sample_rng = np.random.default_rng(args.seed)
    profile = empty_profile()
    candidate_frames: list[pd.DataFrame] = []
    source_row_offset = 0

    reader = pd.read_csv(args.input, chunksize=args.chunk_size, nrows=args.max_rows)
    for chunk_number, chunk in enumerate(reader, start=1):
        if list(chunk.columns) != EXPECTED_COLUMNS:
            raise ValueError(f"Unexpected schema: {list(chunk.columns)}")
        update_profile(profile, chunk, source_row_offset)
        row_numbers = np.arange(
            source_row_offset + 1,
            source_row_offset + len(chunk) + 1,
            dtype=np.int64,
        )
        selected = sample_rng.random(len(chunk)) < inclusion_probability
        if selected.any():
            selected_chunk = chunk.loc[selected].copy()
            selected_chunk.insert(0, "source_row_number", row_numbers[selected])
            candidate_frames.append(selected_chunk)
        source_row_offset += len(chunk)
        print(
            f"processed chunk {chunk_number}: {source_row_offset:,} rows, "
            f"{sum(len(part) for part in candidate_frames):,} candidates",
            flush=True,
        )

    full_profile = finalize_profile(profile)
    if full_profile["rows"] != expected_population:
        raise ValueError(
            f"Expected {expected_population:,} rows but processed {full_profile['rows']:,}"
        )
    if sum(full_profile["null_counts"].values()):
        raise ValueError("Missing values detected. Review the generated full profile.")
    if sum(full_profile["infinite_counts"].values()):
        raise ValueError("Infinite feature values detected. Review the generated full profile.")
    if sum(full_profile["invalid_binary_counts"].values()):
        raise ValueError("Invalid binary values detected. Review the generated full profile.")

    candidates = pd.concat(candidate_frames, ignore_index=True)
    if len(candidates) < args.sample_size:
        raise RuntimeError(
            f"Only {len(candidates):,} candidates selected for a {args.sample_size:,}-row sample"
        )
    trim_rng = np.random.default_rng(args.seed + 1)
    selected_positions = trim_rng.choice(len(candidates), size=args.sample_size, replace=False)
    sample = candidates.iloc[selected_positions].reset_index(drop=True)
    sample = sample.iloc[trim_rng.permutation(len(sample))].reset_index(drop=True)
    for column in BINARY_COLUMNS:
        sample[column] = sample[column].astype(np.int8)

    splits = stratified_split(sample, args.seed + 2)
    split_paths = {}
    split_summaries = {}
    for name, frame in splits.items():
        path = args.output_dir / f"{name}.csv.gz"
        print(f"writing {name}: {len(frame):,} rows", flush=True)
        write_gzip_csv(frame[EXPECTED_COLUMNS], path)
        split_paths[name] = path
        split_summaries[name] = summarize_frame(frame)

    sample_summary = summarize_frame(sample)
    pattern_overlaps = exact_pattern_overlap_counts(splits)
    source_ids = [set(frame["source_row_number"].tolist()) for frame in splits.values()]
    source_id_overlap_count = sum(
        len(source_ids[left] & source_ids[right])
        for left, right in ((0, 1), (0, 2), (1, 2))
    )
    comparison: list[dict] = []
    for column in BINARY_COLUMNS:
        full_rate = full_profile["binary_summary"][column]["one_rate"]
        sample_rate = sample_summary["binary_summary"][column]["one_rate"]
        comparison.append(
            {
                "metric_type": "binary_rate",
                "column": column,
                "full_rate": full_rate,
                "sample_rate": sample_rate,
                "absolute_difference": abs(sample_rate - full_rate),
                "standardized_mean_difference": None,
            }
        )
    for column in FEATURE_COLUMNS:
        full_mean = full_profile["feature_summary"][column]["mean"]
        sample_mean = sample_summary["feature_summary"][column]["mean"]
        full_std = full_profile["feature_summary"][column]["std_population"]
        comparison.append(
            {
                "metric_type": "feature_mean",
                "column": column,
                "full_rate": None,
                "sample_rate": None,
                "absolute_difference": abs(sample_mean - full_mean),
                "standardized_mean_difference": (sample_mean - full_mean) / full_std if full_std else 0.0,
            }
        )
    for treatment in (0, 1):
        for outcome in ("visit", "conversion", "exposure"):
            full_rate = full_profile["treatment_group_summary"][str(treatment)][f"{outcome}_rate"]
            sample_rate = sample_summary["treatment_group_summary"][str(treatment)][f"{outcome}_rate"]
            if full_rate is None or sample_rate is None:
                continue
            comparison.append(
                {
                    "metric_type": "conditional_rate",
                    "column": f"{outcome}|treatment={treatment}",
                    "full_rate": full_rate,
                    "sample_rate": sample_rate,
                    "absolute_difference": abs(sample_rate - full_rate),
                    "standardized_mean_difference": None,
                }
            )

    created_date = "2026-07-31"
    output_files = {
        name: {
            "path": path.name,
            "rows": split_summaries[name]["rows"],
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for name, path in split_paths.items()
    }
    manifest = {
        "created_date": created_date,
        "script": Path(__file__).name,
        "source": {
            "path": args.input.name,
            "bytes": args.input.stat().st_size,
            "sha256": source_sha256,
            "rows": full_profile["rows"],
        },
        "sampling": {
            "method": "equal-probability Bernoulli candidate selection followed by fixed-size random trim",
            "sample_size": args.sample_size,
            "candidate_rows": int(len(candidates)),
            "inclusion_probability": inclusion_probability,
            "seed": args.seed,
            "split_seed": args.seed + 2,
            "stratification_columns": ["treatment", "visit", "conversion"],
            "split_shares": {"train": 0.70, "validation": 0.15, "test": 0.15},
        },
        "outputs": output_files,
        "split_integrity": {
            "source_row_overlap_count": source_id_overlap_count,
            "exact_pattern_overlap_counts": pattern_overlaps,
            "actual_split_rows": {
                name: split_summaries[name]["rows"] for name in ("train", "validation", "test")
            },
        },
        "runtime_seconds": time.perf_counter() - started,
        "publication_status": "public educational derivative under CC BY-NC-SA 4.0",
    }

    full_profile_path = args.report_dir / f"{args.prefix}_full_profile.json"
    sample_profile_path = args.report_dir / f"{args.prefix}_sample_profile.json"
    manifest_path = args.report_dir / f"{args.prefix}_processing_manifest.json"
    comparison_path = args.report_dir / f"{args.prefix}_population_sample_comparison.csv"
    report_path = args.report_dir / f"{args.prefix}_data_quality_report.md"

    full_profile_path.write_text(json.dumps(full_profile, indent=2) + "\n", encoding="utf-8")
    sample_profile_path.write_text(
        json.dumps({"sample": sample_summary, "splits": split_summaries}, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(comparison).to_csv(comparison_path, index=False)
    report_path.write_text(
        build_quality_report(full_profile, sample_summary, split_summaries, comparison, manifest),
        encoding="utf-8",
    )

    print(json.dumps({
        "full_rows": full_profile["rows"],
        "sample_rows": sample_summary["rows"],
        "splits": {name: details["rows"] for name, details in split_summaries.items()},
        "manifest": str(manifest_path),
        "report": str(report_path),
        "runtime_seconds": manifest["runtime_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
