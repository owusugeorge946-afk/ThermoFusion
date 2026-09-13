# =====================================================================
# ThermoFusion reproducibility code
# Source notebook cell(s): 50
# Audit status: KEEP
# Extracted without scientific-logic modification.
# Replace local Google Drive paths as described in README.md when needed.
# =====================================================================

# =============================================================================
# THERMOFUSION STAGE 22B — CORRECTED EXPANDED TEMPORAL-TEST SELECTION
# Unique thermal scenes + ≥50% valid-pixel coverage + strict temporal holdout
# =============================================================================

from google.colab import drive
from pathlib import Path
import pandas as pd
import numpy as np

drive.mount("/content/drive")

ROOT = Path("/content/drive/MyDrive")

VERIFIED_MANIFEST = (
    ROOT / "ThermoFusion_Stage06_Match_Verification"
    / "07_verified_match_manifest.csv"
)

ORIGINAL_SPLIT = (
    ROOT / "ThermoFusion_Stage12_ModelReady_Pilot"
    / "01_scene_split_manifest.csv"
)

REVISED_PILOT = ROOT / "ThermoFusion_Stage10_Revised_Pilot_Manifest.csv"

OUT_DIR = ROOT / "ThermoFusion_Stage22B_CorrectedExpandedTest"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_PER_STRATUM = 2
MIN_VALID_FRACTION_PCT = 50.0


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def find_column(df, possible_names, file_label, required=True):
    """Find a column using exact then partial case-insensitive matching."""
    normalized = {
        str(column).lower().strip(): column
        for column in df.columns
    }

    for name in possible_names:
        if name.lower() in normalized:
            return normalized[name.lower()]

    for column in df.columns:
        column_text = str(column).lower().strip()

        if any(
            name.lower() in column_text
            for name in possible_names
        ):
            return column

    if required:
        raise KeyError(
            f"{file_label}: could not find any of {possible_names}.\n"
            f"Available columns: {df.columns.tolist()}"
        )

    return None


def standardize_core_fields(df, file_label):
    """Create common record_id, city, thermal_sensor and acquisition_date."""
    out = df.copy()

    record_col = find_column(
        out,
        ["record_id", "scene_id", "tfp_id"],
        file_label
    )

    city_col = find_column(out, ["city"], file_label)

    sensor_col = find_column(
        out,
        ["thermal_sensor", "thermal sensor", "sensor"],
        file_label
    )

    date_col = find_column(
        out,
        [
            "acquisition_date",
            "thermal_datetime_utc",
            "thermal_datetime",
            "scene_date",
            "date"
        ],
        file_label
    )

    out = out.rename(columns={
        record_col: "record_id",
        city_col: "city",
        sensor_col: "thermal_sensor",
        date_col: "acquisition_date"
    })

    out["record_id"] = out["record_id"].astype(str).str.strip()
    out["city"] = out["city"].astype(str).str.strip()
    out["thermal_sensor"] = out["thermal_sensor"].astype(str).str.strip()

    out["acquisition_date"] = pd.to_datetime(
        out["acquisition_date"],
        errors="coerce",
        utc=True
    )

    return out


def true_values(series):
    """Convert common Boolean encodings to True/False."""
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes", "y"])
    )


# -----------------------------------------------------------------------------
# Read and standardize files
# -----------------------------------------------------------------------------

assert VERIFIED_MANIFEST.exists(), f"Missing: {VERIFIED_MANIFEST}"
assert ORIGINAL_SPLIT.exists(), f"Missing: {ORIGINAL_SPLIT}"
assert REVISED_PILOT.exists(), f"Missing: {REVISED_PILOT}"

verified_raw = pd.read_csv(VERIFIED_MANIFEST)
split_raw = pd.read_csv(ORIGINAL_SPLIT)
pilot_raw = pd.read_csv(REVISED_PILOT)

verified = standardize_core_fields(verified_raw, "Verified inventory")
split = standardize_core_fields(split_raw, "Original split manifest")
pilot = standardize_core_fields(pilot_raw, "Revised pilot manifest")

# Keep the thermal-scene identity for strict de-duplication.
thermal_scene_col = find_column(
    verified_raw,
    ["thermal_scene_id", "thermal scene id"],
    "Verified inventory",
    required=False
)

if thermal_scene_col is not None:
    verified["thermal_scene_id"] = (
        verified_raw[thermal_scene_col]
        .astype(str)
        .str.strip()
    )
else:
    verified["thermal_scene_id"] = (
        verified["city"].str.lower()
        + "__"
        + verified["thermal_sensor"].str.lower()
        + "__"
        + verified["acquisition_date"].astype(str)
    )

valid_fraction_col = find_column(
    verified_raw,
    ["valid_fraction_pct", "valid fraction", "valid_pixel_fraction"],
    "Verified inventory"
)

verified["valid_fraction_pct"] = pd.to_numeric(
    verified_raw[valid_fraction_col],
    errors="coerce"
)

# Convert fractions in 0–1 form to percentages if required.
if verified["valid_fraction_pct"].dropna().max() <= 1.0:
    verified["valid_fraction_pct"] *= 100.0


# -----------------------------------------------------------------------------
# Protect all original 64 scenes and identify training-date cut-offs
# -----------------------------------------------------------------------------

protected_ids = set(split["record_id"]) | set(pilot["record_id"])

split_role_col = find_column(
    split_raw,
    ["model_split", "split", "role"],
    "Original split manifest"
)

split["model_role"] = split_raw[split_role_col].astype(str).str.lower()

training = split.loc[
    split["model_role"].str.contains("train", na=False)
].copy()

latest_training_dates = (
    training
    .groupby(["city", "thermal_sensor"])["acquisition_date"]
    .max()
    .rename("latest_training_date")
    .reset_index()
)


# -----------------------------------------------------------------------------
# Apply the fixed eligibility criteria
# -----------------------------------------------------------------------------

candidates = verified.loc[
    ~verified["record_id"].isin(protected_ids)
].copy()

candidates = candidates.merge(
    latest_training_dates,
    on=["city", "thermal_sensor"],
    how="left"
)

candidates["is_later_than_training"] = (
    candidates["acquisition_date"]
    > candidates["latest_training_date"]
)

candidates["quarter"] = candidates["acquisition_date"].dt.quarter

# Preserve the original strict matching requirements.
for audit_field in [
    "all_three_found",
    "s2_within_3_days",
    "s1_within_6_days",
    "strict_temporal_match"
]:
    if audit_field in verified_raw.columns:
        candidates[audit_field] = verified_raw.loc[
            candidates.index, audit_field
        ].values

        candidates = candidates.loc[
            true_values(candidates[audit_field])
        ].copy()

candidates = candidates.loc[
    candidates["is_later_than_training"]
    & candidates["acquisition_date"].notna()
    & candidates["quarter"].notna()
    & candidates["valid_fraction_pct"].ge(MIN_VALID_FRACTION_PCT)
].copy()


# -----------------------------------------------------------------------------
# Retain one best record for each distinct thermal scene
# -----------------------------------------------------------------------------

for field in [
    "s2_time_difference_hours",
    "s1_time_difference_hours",
    "era5_time_difference_hours",
    "s2_cloudy_pixel_percentage"
]:
    if field in verified_raw.columns:
        candidates[field] = pd.to_numeric(
            verified_raw.loc[candidates.index, field].values,
            errors="coerce"
        )

sort_columns = ["valid_fraction_pct"]

for field in [
    "s2_time_difference_hours",
    "s1_time_difference_hours",
    "era5_time_difference_hours",
    "s2_cloudy_pixel_percentage"
]:
    if field in candidates.columns:
        sort_columns.append(field)

candidates = candidates.sort_values(
    sort_columns,
    ascending=[False] + [True] * (len(sort_columns) - 1),
    na_position="last"
).copy()

unique_candidates = candidates.drop_duplicates(
    subset=["city", "thermal_sensor", "thermal_scene_id"],
    keep="first"
).copy()


# -----------------------------------------------------------------------------
# Select two scenes per city × sensor × quarter where available
# -----------------------------------------------------------------------------

unique_candidates = unique_candidates.sort_values(
    [
        "city",
        "thermal_sensor",
        "quarter",
        "valid_fraction_pct",
        "acquisition_date"
    ],
    ascending=[True, True, True, False, True]
).copy()

selected = (
    unique_candidates
    .groupby(
        ["city", "thermal_sensor", "quarter"],
        group_keys=False
    )
    .head(TARGET_PER_STRATUM)
    .copy()
)

selected["stage22_role"] = "expanded_temporal_test"
selected["selection_rank"] = (
    selected
    .groupby(["city", "thermal_sensor", "quarter"])
    .cumcount() + 1
)


# -----------------------------------------------------------------------------
# Integrity checks
# -----------------------------------------------------------------------------

assert not selected["record_id"].isin(protected_ids).any(), (
    "FAIL: selected records overlap with the original 64 scenes."
)

assert not selected.duplicated(
    subset=["city", "thermal_sensor", "thermal_scene_id"]
).any(), (
    "FAIL: duplicate thermal scenes remain in the selection."
)

assert selected["valid_fraction_pct"].ge(
    MIN_VALID_FRACTION_PCT
).all(), (
    "FAIL: at least one selected scene is below 50% valid-pixel coverage."
)


# -----------------------------------------------------------------------------
# Reports and saved files
# -----------------------------------------------------------------------------

balance = (
    selected
    .groupby(["city", "thermal_sensor", "quarter"])
    .agg(
        selected_scenes=("record_id", "size"),
        mean_valid_fraction_pct=("valid_fraction_pct", "mean"),
        minimum_valid_fraction_pct=("valid_fraction_pct", "min"),
        earliest_date=("acquisition_date", "min"),
        latest_date=("acquisition_date", "max")
    )
    .reset_index()
    .sort_values(["city", "thermal_sensor", "quarter"])
)

shortfalls = balance.loc[
    balance["selected_scenes"] < TARGET_PER_STRATUM
].copy()

selection_path = OUT_DIR / "01_corrected_expanded_test_manifest.csv"
balance_path = OUT_DIR / "02_corrected_balance.csv"
shortfall_path = OUT_DIR / "03_stratum_shortfalls.csv"
candidate_path = OUT_DIR / "04_unique_eligible_candidates.csv"

selected.to_csv(selection_path, index=False)
balance.to_csv(balance_path, index=False)
shortfalls.to_csv(shortfall_path, index=False)
unique_candidates.to_csv(candidate_path, index=False)

print("=" * 100)
print("THERMOFUSION STAGE 22B — CORRECTED EXPANDED TEMPORAL-TEST SELECTION")
print("=" * 100)
print(f"Original protected scenes: {len(protected_ids)}")
print(f"Unique high-coverage later candidates: {len(unique_candidates)}")
print(f"Final selected independent thermal scenes: {len(selected)}")
print("Leakage check: PASS")
print("Unique thermal-scene check: PASS")
print("≥50% valid-pixel coverage check: PASS")
print()

print("Final city × sensor × quarter balance:")
display(balance)

if shortfalls.empty:
    print("Balanced design: PASS — two scenes in every stratum.")
else:
    print("Strata with fewer than two eligible scenes:")
    display(shortfalls)

print()
print("Saved files:")
print(selection_path)
print(balance_path)
print(shortfall_path)
print(candidate_path)


# ==================== MERGED INTEGRITY CHECKS FROM CELL 51 ====================

from pathlib import Path
import pandas as pd

ROOT = Path("/content/drive/MyDrive")

manifest_path = (
    ROOT / "ThermoFusion_Stage22B_CorrectedExpandedTest"
    / "01_corrected_expanded_test_manifest.csv"
)

assert manifest_path.exists(), f"Missing: {manifest_path}"

df = pd.read_csv(manifest_path)

print("Selected scenes:", len(df))
print("Unique record IDs:", df["record_id"].nunique())
print(
    "Unique thermal scenes:",
    df.drop_duplicates(
        ["city", "thermal_sensor", "thermal_scene_id"]
    ).shape[0]
)
print("Minimum valid fraction:", df["valid_fraction_pct"].min())

assert len(df) == 38, "Expected 38 selected scenes."
assert df["record_id"].nunique() == 38, "Duplicate record IDs found."
assert (
    df.drop_duplicates(
        ["city", "thermal_sensor", "thermal_scene_id"]
    ).shape[0] == 38
), "Duplicate thermal scenes found."
assert df["valid_fraction_pct"].min() >= 50, "A scene is below 50% coverage."

print("\nSTAGE 23 PREFLIGHT: PASS")
display(
    df[
        [
            "record_id", "city", "thermal_sensor",
            "acquisition_date", "valid_fraction_pct"
        ]
    ].head(10)
)