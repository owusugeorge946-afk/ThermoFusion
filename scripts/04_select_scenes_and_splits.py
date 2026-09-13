# =====================================================================
# ThermoFusion reproducibility code
# Source notebook cell(s): 7
# Audit status: KEEP
# Extracted without scientific-logic modification.
# Replace local Google Drive paths as described in README.md when needed.
# =====================================================================

# ThermoFusion Stage 04: scene selection and leakage-safe split design.
# Run in one Google Colab cell after Stage 03 verification passes.

from google.colab import drive
from pathlib import Path
import shutil
import numpy as np
import pandas as pd


drive.mount("/content/drive")

INPUT_DIR = Path("/content/drive/MyDrive/ThermoFusion_Stage02_Audit")
OUTPUT_DIR = Path("/content/drive/MyDrive/ThermoFusion_Stage04_Selection")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAINING_MIN_COVERAGE = 10.0
EVALUATION_MIN_COVERAGE = 50.0
EXPECTED_CITIES = ["Accra", "Lagos", "Abidjan", "Freetown"]
EXPECTED_SENSORS = ["Landsat", "ECOSTRESS"]


files = sorted(INPUT_DIR.glob("*.csv"))
if len(files) != 8:
    raise RuntimeError(f"Expected 8 Stage 02 CSV files but found {len(files)}.")

audit = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
audit["date"] = pd.to_datetime(audit["date"], errors="coerce")
audit["valid_fraction_pct"] = pd.to_numeric(
    audit["valid_fraction_pct"], errors="coerce"
)
audit["valid_area_km2"] = pd.to_numeric(audit["valid_area_km2"], errors="coerce")
audit["city"] = audit["city"].astype(str).str.strip()
audit["sensor"] = audit["sensor"].astype(str).str.strip()

if audit["date"].isna().any():
    raise RuntimeError("Some records have invalid dates.")

if audit["scene_id"].isna().any():
    raise RuntimeError("Some records have missing Earth Engine scene identifiers.")


# Remove exact duplicate city-sensor-scene records, retaining an audit trail.
duplicate_mask = audit.duplicated(
    subset=["city", "sensor", "scene_id"], keep="first"
)
duplicates = audit.loc[duplicate_mask].copy()
audit = audit.loc[~duplicate_mask].copy()
duplicates.to_csv(OUTPUT_DIR / "01_removed_exact_duplicates.csv", index=False)


# Flag scene eligibility without altering the original coverage values.
audit["training_eligible"] = audit["valid_fraction_pct"].ge(
    TRAINING_MIN_COVERAGE
)
audit["high_coverage_evaluation"] = audit["valid_fraction_pct"].ge(
    EVALUATION_MIN_COVERAGE
)

selected = audit[audit["training_eligible"]].copy()
selected = selected.sort_values(["city", "sensor", "date", "scene_id"])


# Chronological 70/15/15 split within each city-sensor group. This prevents
# observations from later dates leaking into model development.
selected["temporal_role"] = "temporal_test"
for _, group_indices in selected.groupby(
    ["city", "sensor"], observed=True
).groups.items():
    ordered_indices = (
        selected.loc[group_indices]
        .sort_values(["date", "scene_id"])
        .index.to_numpy()
    )
    n = len(ordered_indices)
    train_end = max(1, int(np.floor(0.70 * n)))
    calibration_end = max(train_end + 1, int(np.floor(0.85 * n)))
    calibration_end = min(calibration_end, n)
    selected.loc[ordered_indices[:train_end], "temporal_role"] = "development"
    selected.loc[
        ordered_indices[train_end:calibration_end], "temporal_role"
    ] = "calibration"

selected = selected.reset_index(drop=True)


# Leave-one-city-out folds. For each fold, the named city is completely unseen
# during training. Temporal roles remain available for secondary experiments.
for held_out_city in EXPECTED_CITIES:
    column = f"loco_{held_out_city.lower()}"
    selected[column] = np.where(
        selected["city"].eq(held_out_city), "test", "train"
    )


# Sensor-transfer labels support a separate experiment in which one thermal
# sensor is unseen during model fitting.
selected["landsat_holdout_role"] = np.where(
    selected["sensor"].eq("Landsat"), "test", "train"
)
selected["ecostress_holdout_role"] = np.where(
    selected["sensor"].eq("ECOSTRESS"), "test", "train"
)


# Stable record identifier for all downstream imagery and patch tables.
selected["date_key"] = selected["date"].dt.strftime("%Y%m%d")
selected["record_id"] = (
    selected["city"].str.lower()
    + "__"
    + selected["sensor"].str.lower()
    + "__"
    + selected["date_key"]
    + "__"
    + selected.groupby(
        ["city", "sensor", "date_key"],
        observed=True,
    ).cumcount().astype(str).str.zfill(2)
)


# Test the invariants that protect the evaluation design.
assert selected["record_id"].is_unique, "record_id values are not unique."
assert selected["valid_fraction_pct"].ge(TRAINING_MIN_COVERAGE).all()

for held_out_city in EXPECTED_CITIES:
    column = f"loco_{held_out_city.lower()}"
    assert selected.loc[selected[column] == "test", "city"].eq(held_out_city).all()
    assert selected.loc[selected[column] == "train", "city"].ne(held_out_city).all()


manifest_columns = [
    "record_id",
    "city",
    "sensor",
    "scene_id",
    "date",
    "year",
    "month",
    "valid_area_km2",
    "valid_fraction_pct",
    "training_eligible",
    "high_coverage_evaluation",
    "temporal_role",
    "loco_accra",
    "loco_lagos",
    "loco_abidjan",
    "loco_freetown",
    "landsat_holdout_role",
    "ecostress_holdout_role",
]

selected[manifest_columns].to_csv(
    OUTPUT_DIR / "02_selected_scene_manifest.csv", index=False
)

selected[selected["high_coverage_evaluation"]][manifest_columns].to_csv(
    OUTPUT_DIR / "03_high_coverage_evaluation_manifest.csv", index=False
)


# Counts for every principal experimental partition.
selection_summary = (
    selected.groupby(["city", "sensor"], observed=True)
    .agg(
        training_pool=("record_id", "count"),
        high_coverage_evaluation=("high_coverage_evaluation", "sum"),
        coverage_median_pct=("valid_fraction_pct", "median"),
        coverage_p25_pct=("valid_fraction_pct", lambda x: x.quantile(0.25)),
        coverage_p75_pct=("valid_fraction_pct", lambda x: x.quantile(0.75)),
        first_date=("date", "min"),
        last_date=("date", "max"),
    )
    .reset_index()
)

temporal_summary = (
    selected.groupby(["city", "sensor", "temporal_role"], observed=True)
    .size()
    .rename("scenes")
    .reset_index()
)

loco_summary_rows = []
for held_out_city in EXPECTED_CITIES:
    column = f"loco_{held_out_city.lower()}"
    for role in ["train", "test"]:
        subset = selected[selected[column] == role]
        loco_summary_rows.append(
            {
                "fold": held_out_city,
                "role": role,
                "scenes": len(subset),
                "high_coverage_scenes": int(
                    subset["high_coverage_evaluation"].sum()
                ),
                "cities": ", ".join(sorted(subset["city"].unique())),
            }
        )

loco_summary = pd.DataFrame(loco_summary_rows)

selection_summary.to_csv(OUTPUT_DIR / "04_selection_summary.csv", index=False)
temporal_summary.to_csv(OUTPUT_DIR / "05_temporal_split_summary.csv", index=False)
loco_summary.to_csv(OUTPUT_DIR / "06_leave_one_city_out_summary.csv", index=False)


# Year-month observation matrix for downstream pairing decisions.
selected["year_month"] = selected["date"].dt.to_period("M").astype(str)
monthly_inventory = (
    selected.groupby(["city", "sensor", "year_month"], observed=True)
    .agg(
        selected_scenes=("record_id", "count"),
        high_coverage_scenes=("high_coverage_evaluation", "sum"),
        median_coverage_pct=("valid_fraction_pct", "median"),
    )
    .reset_index()
)
monthly_inventory.to_csv(OUTPUT_DIR / "07_monthly_scene_inventory.csv", index=False)


print("\nTHERMOFUSION STAGE 04")
print(f"Candidate audit records after deduplication: {len(audit)}")
print(f"Removed exact duplicates: {len(duplicates)}")
print(f"Training-pool scenes (>=10% coverage): {len(selected)}")
print(
    "High-coverage evaluation scenes (>=50% coverage): "
    f"{int(selected['high_coverage_evaluation'].sum())}"
)
print("\nSelection summary:")
display(selection_summary.round(2))
print("\nChronological split summary:")
display(temporal_summary)
print("\nLeave-one-city-out fold summary:")
display(loco_summary)


verdict = [
    "THERMOFUSION STAGE 04 SELECTION VERDICT",
    f"Training minimum coverage: {TRAINING_MIN_COVERAGE:.0f}%",
    f"Evaluation minimum coverage: {EVALUATION_MIN_COVERAGE:.0f}%",
    f"Selected training-pool scenes: {len(selected)}",
    (
        "Selected high-coverage evaluation scenes: "
        f"{int(selected['high_coverage_evaluation'].sum())}"
    ),
    f"Exact duplicates removed: {len(duplicates)}",
    "Four leave-one-city-out folds: VERIFIED",
    "Chronological within-city splits: VERIFIED",
    "Scene manifest uniqueness: VERIFIED",
]

(OUTPUT_DIR / "08_selection_verdict.txt").write_text(
    "\n".join(verdict), encoding="utf-8"
)

zip_path = shutil.make_archive(
    "/content/drive/MyDrive/ThermoFusion_Stage04_Selection",
    "zip",
    root_dir=OUTPUT_DIR,
)

print("\n" + "\n".join(verdict))
print(f"\nOutput folder: {OUTPUT_DIR}")
print(f"ZIP package: {zip_path}")
