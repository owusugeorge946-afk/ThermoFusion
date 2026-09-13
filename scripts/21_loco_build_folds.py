# =====================================================================
# ThermoFusion reproducibility code
# Source notebook cell(s): 43
# Audit status: KEEP
# Extracted without scientific-logic modification.
# Replace local Google Drive paths as described in README.md when needed.
# =====================================================================

# ============================================================
# THERMOFUSION LEAVE-ONE-CITY-OUT: STAGE 02
# Construct leakage-safe geographical hold-out folds
# ============================================================

from pathlib import Path
import pandas as pd
import numpy as np
import json

# ------------------------------------------------------------
# 1. Input and output locations
# ------------------------------------------------------------
ROOT = Path("/content/drive/MyDrive/ThermoFusion_Stage12_ModelReady_Pilot")
ARRAY_DIR = ROOT / "arrays"
ARRAY_MANIFEST = ROOT / "04_model_ready_array_manifest.csv"

OUTPUT_ROOT = Path(
    "/content/drive/MyDrive/ThermoFusion_Stage20_LeaveOneCityOut"
)
MANIFEST_DIR = OUTPUT_ROOT / "fold_manifests"
REPORT_DIR = OUTPUT_ROOT / "reports"

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# 2. Load and validate the array manifest
# ------------------------------------------------------------
df = pd.read_csv(ARRAY_MANIFEST)

required_columns = [
    "pilot_id",
    "record_id",
    "city",
    "thermal_sensor",
    "thermal_datetime_utc",
    "model_split",
    "array_path",
    "channels",
    "height",
    "width",
    "status",
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Required columns are missing: {missing_columns}"
    )

df["city"] = df["city"].astype(str).str.strip()
df["thermal_sensor"] = (
    df["thermal_sensor"].astype(str).str.strip()
)
df["model_split"] = (
    df["model_split"].astype(str).str.strip().str.lower()
)
df["thermal_datetime_utc"] = pd.to_datetime(
    df["thermal_datetime_utc"],
    utc=True,
    errors="raise",
)

# Reconstruct array paths from the verified directory.
# This avoids problems if stored paths have changed.
df["array_path"] = df["pilot_id"].apply(
    lambda pilot_id:
    str(ARRAY_DIR / f"{pilot_id}_model_ready.npz")
)

print("=" * 100)
print("THERMOFUSION LEAVE-ONE-CITY-OUT FOLD CONSTRUCTION")
print("=" * 100)
print(f"Input scenes: {len(df)}")
print(f"Cities: {sorted(df['city'].unique().tolist())}")
print(
    "Sensors:",
    sorted(df["thermal_sensor"].unique().tolist())
)

# ------------------------------------------------------------
# 3. Global integrity checks
# ------------------------------------------------------------
assert len(df) == 64, f"Expected 64 scenes, found {len(df)}"
assert df["pilot_id"].is_unique, "pilot_id values are not unique"
assert df["record_id"].is_unique, "record_id values are not unique"
assert (df["status"] == "PASS").all(), "A scene did not pass Stage 12"
assert (df["channels"] == 16).all(), "Unexpected channel count"
assert (df["height"] == 256).all(), "Unexpected array height"
assert (df["width"] == 256).all(), "Unexpected array width"

missing_arrays = [
    path for path in df["array_path"]
    if not Path(path).exists()
]

if missing_arrays:
    raise FileNotFoundError(
        f"{len(missing_arrays)} array files are missing. "
        f"First missing path: {missing_arrays[0]}"
    )

expected_cities = {
    "Abidjan",
    "Accra",
    "Freetown",
    "Lagos",
}

observed_cities = set(df["city"].unique())

assert observed_cities == expected_cities, (
    f"Unexpected cities: {observed_cities}"
)

expected_sensors = {
    "Landsat",
    "ECOSTRESS",
}

observed_sensors = set(df["thermal_sensor"].unique())

assert observed_sensors == expected_sensors, (
    f"Unexpected sensors: {observed_sensors}"
)

expected_splits = {"train", "validation", "temporal_test"}

# Allow common alternative labels and convert them consistently.
split_replacements = {
    "val": "validation",
    "valid": "validation",
    "test": "temporal_test",
    "temporal-test": "temporal_test",
    "temporal test": "temporal_test",
}

df["model_split"] = df["model_split"].replace(split_replacements)

print("\nOriginal model-split distribution:")
print(df["model_split"].value_counts(dropna=False))

unexpected_splits = (
    set(df["model_split"].unique()) - expected_splits
)

if unexpected_splits:
    raise ValueError(
        f"Unexpected model_split values: {unexpected_splits}"
    )

# ------------------------------------------------------------
# 4. Confirm the original balanced design
# ------------------------------------------------------------
original_balance = (
    df.groupby(
        ["city", "thermal_sensor", "model_split"]
    )
    .size()
    .rename("scene_count")
    .reset_index()
)

print("\nOriginal city × sensor × split balance:")
display(original_balance)

expected_original_counts = {
    "train": 6,
    "validation": 1,
    "temporal_test": 1,
}

for _, row in original_balance.iterrows():
    expected_count = expected_original_counts[row["model_split"]]

    assert row["scene_count"] == expected_count, (
        f"Unexpected count for {row['city']} / "
        f"{row['thermal_sensor']} / {row['model_split']}: "
        f"{row['scene_count']} instead of {expected_count}"
    )

# ------------------------------------------------------------
# 5. Construct four leave-one-city-out folds
# ------------------------------------------------------------
fold_tables = []
fold_summary = []

for held_out_city in sorted(expected_cities):

    fold_name = (
        "holdout_" +
        held_out_city.lower().replace(" ", "_")
    )

    fold_df = df.copy()

    conditions = [
        fold_df["city"].eq(held_out_city),
        (
            fold_df["city"].ne(held_out_city)
            & fold_df["model_split"].eq("train")
        ),
        (
            fold_df["city"].ne(held_out_city)
            & fold_df["model_split"].eq("validation")
        ),
    ]

    choices = [
        "test",
        "train",
        "validation",
    ]

    fold_df["loco_role"] = np.select(
        conditions,
        choices,
        default="excluded",
    )

    fold_df["fold"] = fold_name
    fold_df["held_out_city"] = held_out_city

    # Sort records consistently.
    role_order = pd.CategoricalDtype(
        categories=[
            "train",
            "validation",
            "test",
            "excluded",
        ],
        ordered=True,
    )

    fold_df["loco_role"] = fold_df["loco_role"].astype(
        role_order
    )

    fold_df = fold_df.sort_values(
        [
            "loco_role",
            "city",
            "thermal_sensor",
            "thermal_datetime_utc",
            "pilot_id",
        ]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # 5a. Leakage checks
    # --------------------------------------------------------
    train_ids = set(
        fold_df.loc[
            fold_df["loco_role"] == "train",
            "pilot_id",
        ]
    )

    validation_ids = set(
        fold_df.loc[
            fold_df["loco_role"] == "validation",
            "pilot_id",
        ]
    )

    test_ids = set(
        fold_df.loc[
            fold_df["loco_role"] == "test",
            "pilot_id",
        ]
    )

    assert train_ids.isdisjoint(validation_ids)
    assert train_ids.isdisjoint(test_ids)
    assert validation_ids.isdisjoint(test_ids)

    training_cities = set(
        fold_df.loc[
            fold_df["loco_role"] == "train",
            "city",
        ]
    )

    validation_cities = set(
        fold_df.loc[
            fold_df["loco_role"] == "validation",
            "city",
        ]
    )

    testing_cities = set(
        fold_df.loc[
            fold_df["loco_role"] == "test",
            "city",
        ]
    )

    assert held_out_city not in training_cities
    assert held_out_city not in validation_cities
    assert testing_cities == {held_out_city}

    # --------------------------------------------------------
    # 5b. Expected fold sizes
    # --------------------------------------------------------
    role_counts = (
        fold_df["loco_role"]
        .value_counts()
        .reindex(
            ["train", "validation", "test", "excluded"],
            fill_value=0,
        )
    )

    assert role_counts["train"] == 36
    assert role_counts["validation"] == 6
    assert role_counts["test"] == 16
    assert role_counts["excluded"] == 6

    # Held-out test city must contain 8 scenes per sensor.
    test_sensor_counts = (
        fold_df.loc[
            fold_df["loco_role"] == "test"
        ]
        .groupby("thermal_sensor")
        .size()
        .to_dict()
    )

    assert test_sensor_counts == {
        "ECOSTRESS": 8,
        "Landsat": 8,
    }

    # Training contains three cities × two sensors × six scenes.
    train_sensor_counts = (
        fold_df.loc[
            fold_df["loco_role"] == "train"
        ]
        .groupby("thermal_sensor")
        .size()
        .to_dict()
    )

    assert train_sensor_counts == {
        "ECOSTRESS": 18,
        "Landsat": 18,
    }

    # Validation contains three scenes per sensor.
    validation_sensor_counts = (
        fold_df.loc[
            fold_df["loco_role"] == "validation"
        ]
        .groupby("thermal_sensor")
        .size()
        .to_dict()
    )

    assert validation_sensor_counts == {
        "ECOSTRESS": 3,
        "Landsat": 3,
    }

    # --------------------------------------------------------
    # 5c. Save fold manifest
    # --------------------------------------------------------
    output_columns = [
        "fold",
        "held_out_city",
        "loco_role",
        "pilot_id",
        "record_id",
        "city",
        "thermal_sensor",
        "thermal_datetime_utc",
        "model_split",
        "array_path",
        "target_valid_pixels",
        "target_valid_fraction",
        "s2_valid_fraction",
        "s1_valid_fraction",
        "status",
    ]

    fold_path = MANIFEST_DIR / f"{fold_name}.csv"

    fold_df[output_columns].to_csv(
        fold_path,
        index=False,
    )

    fold_tables.append(fold_df[output_columns].copy())

    fold_summary.append({
        "fold": fold_name,
        "held_out_city": held_out_city,
        "training_scenes": int(role_counts["train"]),
        "validation_scenes": int(
            role_counts["validation"]
        ),
        "test_scenes": int(role_counts["test"]),
        "excluded_scenes": int(role_counts["excluded"]),
        "training_cities": ", ".join(
            sorted(training_cities)
        ),
        "test_landsat_scenes": int(
            test_sensor_counts["Landsat"]
        ),
        "test_ecostress_scenes": int(
            test_sensor_counts["ECOSTRESS"]
        ),
        "leakage_check": "PASS",
    })

# ------------------------------------------------------------
# 6. Save combined manifests and summaries
# ------------------------------------------------------------
combined_folds = pd.concat(
    fold_tables,
    ignore_index=True,
)

summary_df = pd.DataFrame(fold_summary)

combined_manifest_path = (
    OUTPUT_ROOT / "01_all_loco_fold_assignments.csv"
)

summary_path = (
    OUTPUT_ROOT / "02_loco_fold_summary.csv"
)

json_path = (
    OUTPUT_ROOT / "03_loco_fold_configuration.json"
)

combined_folds.to_csv(
    combined_manifest_path,
    index=False,
)

summary_df.to_csv(
    summary_path,
    index=False,
)

configuration = {
    "experiment": "ThermoFusion leave-one-city-out",
    "source_scene_count": 64,
    "cities": sorted(expected_cities),
    "thermal_sensors": sorted(expected_sensors),
    "fold_count": 4,
    "fold_design": {
        "training": (
            "Original training scenes from the three "
            "non-held-out cities"
        ),
        "validation": (
            "Original validation scenes from the three "
            "non-held-out cities"
        ),
        "testing": (
            "All 16 scenes from the completely held-out city"
        ),
        "excluded": (
            "Original temporal-test scenes from the three "
            "non-held-out cities"
        ),
    },
    "expected_fold_sizes": {
        "train": 36,
        "validation": 6,
        "test": 16,
        "excluded": 6,
    },
    "random_seed": 20260908,
    "held_out_data_used_for_training": False,
    "held_out_data_used_for_validation": False,
    "status": "PASS",
}

with open(json_path, "w", encoding="utf-8") as file:
    json.dump(
        configuration,
        file,
        indent=2,
    )

# ------------------------------------------------------------
# 7. Display final fold summary
# ------------------------------------------------------------
print("\n" + "=" * 100)
print("LEAVE-ONE-CITY-OUT FOLD SUMMARY")
print("=" * 100)

display(summary_df)

print("\nRole counts across folds:")
display(
    combined_folds.groupby(
        ["fold", "loco_role"]
    )
    .size()
    .unstack(fill_value=0)
)

print("\nTest sensor balance:")
display(
    combined_folds.loc[
        combined_folds["loco_role"] == "test"
    ]
    .groupby(
        ["fold", "held_out_city", "thermal_sensor"]
    )
    .size()
    .unstack(fill_value=0)
)

print("\nSaved files:")
print(combined_manifest_path)
print(summary_path)
print(json_path)

print("\n" + "=" * 100)
print("FINAL VERDICT: PASS")
print("Four leakage-safe leave-one-city-out folds are ready.")
print("=" * 100)
