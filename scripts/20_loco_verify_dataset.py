# =====================================================================
# ThermoFusion reproducibility code
# Source notebook cell(s): 42
# Audit status: KEEP
# Extracted without scientific-logic modification.
# Replace local Google Drive paths as described in README.md when needed.
# =====================================================================

# ============================================================
# THERMOFUSION LEAVE-ONE-CITY-OUT: STAGE 01
# Verify the existing model-ready dataset
# ============================================================

from google.colab import drive
drive.mount("/content/drive")

from pathlib import Path
import pandas as pd
import numpy as np
import json
import os

# ------------------------------------------------------------
# 1. Existing ThermoFusion locations
# ------------------------------------------------------------
ROOT = Path("/content/drive/MyDrive/ThermoFusion_Stage12_ModelReady_Pilot")
ARRAY_DIR = ROOT / "arrays"

SPLIT_MANIFEST = ROOT / "01_scene_split_manifest.csv"
ARRAY_MANIFEST = ROOT / "04_model_ready_array_manifest.csv"
VERDICT_FILE = ROOT / "07_model_ready_verdict.txt"

print("=" * 90)
print("THERMOFUSION MODEL-READY DATA AUDIT")
print("=" * 90)

required_paths = {
    "Stage 12 root": ROOT,
    "Array directory": ARRAY_DIR,
    "Split manifest": SPLIT_MANIFEST,
    "Array manifest": ARRAY_MANIFEST,
    "Verification verdict": VERDICT_FILE,
}

all_required_exist = True

for name, path in required_paths.items():
    exists = path.exists()
    status = "FOUND" if exists else "MISSING"
    print(f"{name:<24}: {status}")
    print(f"  {path}")

    if not exists:
        all_required_exist = False

# ------------------------------------------------------------
# 2. Locate all NPZ model-ready arrays
# ------------------------------------------------------------
npz_files = sorted(ARRAY_DIR.glob("*.npz")) if ARRAY_DIR.exists() else []

print("\n" + "=" * 90)
print("MODEL-READY ARRAYS")
print("=" * 90)
print(f"Number of .npz arrays found: {len(npz_files)}")
print(f"Expected number:             64")

if npz_files:
    print("\nFirst five arrays:")
    for file in npz_files[:5]:
        print(f"  {file.name}")

# ------------------------------------------------------------
# 3. Read the verification verdict
# ------------------------------------------------------------
print("\n" + "=" * 90)
print("STAGE 12 VERDICT")
print("=" * 90)

if VERDICT_FILE.exists():
    verdict_text = VERDICT_FILE.read_text(
        encoding="utf-8",
        errors="replace"
    )
    print(verdict_text)
else:
    print("Verdict file was not found.")

# ------------------------------------------------------------
# 4. Inspect manifest files
# ------------------------------------------------------------
def inspect_manifest(path, label):
    print("\n" + "=" * 90)
    print(label)
    print("=" * 90)

    if not path.exists():
        print(f"Missing: {path}")
        return None

    df = pd.read_csv(path)

    print(f"Rows:    {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nFirst five records:")
    display(df.head())

    return df


split_df = inspect_manifest(
    SPLIT_MANIFEST,
    "SCENE-SPLIT MANIFEST"
)

array_df = inspect_manifest(
    ARRAY_MANIFEST,
    "MODEL-READY ARRAY MANIFEST"
)

# ------------------------------------------------------------
# 5. Detect important manifest columns
# ------------------------------------------------------------
def find_column(df, candidates):
    if df is None:
        return None

    normalized = {
        str(col).strip().lower().replace("-", "_").replace(" ", "_"): col
        for col in df.columns
    }

    for candidate in candidates:
        key = candidate.lower().replace("-", "_").replace(" ", "_")
        if key in normalized:
            return normalized[key]

    return None


if split_df is not None:
    city_col = find_column(
        split_df,
        ["city", "study_city", "site", "location"]
    )

    sensor_col = find_column(
        split_df,
        ["sensor", "thermal_sensor", "target_sensor"]
    )

    split_col = find_column(
        split_df,
        ["split", "partition", "data_split", "set"]
    )

    scene_col = find_column(
        split_df,
        ["scene_id", "scene", "sample_id", "record_id"]
    )

    print("\n" + "=" * 90)
    print("DETECTED MANIFEST FIELDS")
    print("=" * 90)
    print(f"Scene column:  {scene_col}")
    print(f"City column:   {city_col}")
    print(f"Sensor column: {sensor_col}")
    print(f"Split column:  {split_col}")

    if city_col:
        print("\nScenes by city:")
        print(split_df[city_col].value_counts(dropna=False))

    if sensor_col:
        print("\nScenes by thermal sensor:")
        print(split_df[sensor_col].value_counts(dropna=False))

    if split_col:
        print("\nScenes by original partition:")
        print(split_df[split_col].value_counts(dropna=False))

    if city_col and sensor_col:
        print("\nCity × sensor distribution:")
        print(pd.crosstab(split_df[city_col], split_df[sensor_col]))

# ------------------------------------------------------------
# 6. Inspect one NPZ file safely
# ------------------------------------------------------------
print("\n" + "=" * 90)
print("NPZ STRUCTURE")
print("=" * 90)

npz_structure_valid = False

if npz_files:
    example_file = npz_files[0]

    with np.load(example_file, allow_pickle=True) as sample:
        print(f"Example file: {example_file.name}")
        print(f"Stored keys:  {sample.files}")

        print("\nArray structure:")
        for key in sample.files:
            value = sample[key]
            print(
                f"{key:<25} "
                f"shape={str(value.shape):<20} "
                f"dtype={value.dtype}"
            )

        npz_structure_valid = len(sample.files) > 0
else:
    print("No NPZ files were available for inspection.")

# ------------------------------------------------------------
# 7. Check every NPZ file for readability
# ------------------------------------------------------------
readable = []
failed = []
structures = {}

for file in npz_files:
    try:
        with np.load(file, allow_pickle=True) as data:
            structure = tuple(
                (key, tuple(data[key].shape), str(data[key].dtype))
                for key in data.files
            )

        readable.append(file.name)
        structures.setdefault(structure, []).append(file.name)

    except Exception as error:
        failed.append({
            "file": file.name,
            "error": str(error)
        })

print("\n" + "=" * 90)
print("ARRAY-INTEGRITY SUMMARY")
print("=" * 90)
print(f"Readable arrays:       {len(readable)}")
print(f"Failed arrays:         {len(failed)}")
print(f"Unique NPZ structures: {len(structures)}")

if failed:
    print("\nFailed files:")
    for item in failed:
        print(f"  {item['file']}: {item['error']}")

# ------------------------------------------------------------
# 8. Final readiness decision
# ------------------------------------------------------------
ready = (
    all_required_exist
    and len(npz_files) == 64
    and len(readable) == 64
    and len(failed) == 0
    and split_df is not None
    and array_df is not None
    and npz_structure_valid
)

print("\n" + "=" * 90)

if ready:
    print("FINAL VERDICT: PASS")
    print("The dataset is ready for leave-one-city-out experiments.")
else:
    print("FINAL VERDICT: REVIEW REQUIRED")
    print("One or more required files or checks did not pass.")

print("=" * 90)
