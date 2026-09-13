# =====================================================================
# ThermoFusion reproducibility code
# Source notebook cell(s): 60
# Audit status: KEEP
# Extracted without scientific-logic modification.
# Replace local Google Drive paths as described in README.md when needed.
# =====================================================================

# ==========================================================================================
# THERMOFUSION STAGE 25 — EXPANDED TEMPORAL-TEST MODEL-READY ARRAY PREPARATION
# Uses the original Stage 12 training normalization unchanged.
# No retraining, no recalibration, and no new normalization statistics.
# ==========================================================================================

from google.colab import drive
from pathlib import Path
import json
import shutil
import subprocess
import sys
import numpy as np
import pandas as pd
from IPython.display import display

try:
    import rasterio
    from rasterio.warp import reproject, Resampling
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "rasterio"])
    import rasterio
    from rasterio.warp import reproject, Resampling

drive.mount("/content/drive")

# ------------------------------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------------------------------
BASE = Path("/content/drive/MyDrive")

# Original frozen normalization from the 64-scene development experiment.
ORIGINAL_NORMALIZATION = (
    BASE / "ThermoFusion_Stage12_ModelReady_Pilot"
    / "02_training_normalization.csv"
)

ORIGINAL_TARGET_NORMALIZATION = (
    BASE / "ThermoFusion_Stage12_ModelReady_Pilot"
    / "03_target_normalization.json"
)

# Stage 23R2/24R products.
EXPORT_DIR = (
    BASE / "ThermoFusion_Stage23R2_FinalReplacement"
    / "Stage23R2_Export"
)

TASK_REGISTER = EXPORT_DIR / "03_export_task_register.csv"

CHIP_FOLDER = BASE / "ThermoFusion_Stage23R2_ExpandedTemporalTest_Chips"

STAGE24_VERDICT = (
    EXPORT_DIR / "08_corrected_chip_pair_verification_summary.csv"
)

# New supplementary-test output only.
OUTPUT_DIR = BASE / "ThermoFusion_Stage25_ExpandedTest_ModelReady"
ARRAY_DIR = OUTPUT_DIR / "arrays"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ARRAY_DIR.mkdir(parents=True, exist_ok=True)

OUT_MANIFEST = OUTPUT_DIR / "01_expanded_test_array_manifest.csv"
OUT_NORMALIZATION_AUDIT = OUTPUT_DIR / "02_frozen_normalization_audit.csv"
OUT_TARGET_AUDIT = OUTPUT_DIR / "03_frozen_target_normalization.json"
OUT_INTEGRITY = OUTPUT_DIR / "04_array_integrity_verification.csv"
OUT_VERDICT = OUTPUT_DIR / "05_model_ready_verdict.txt"

# ------------------------------------------------------------------------------------------
# Fixed original model-input contract
# ------------------------------------------------------------------------------------------
EXPECTED_SCENES = 38
TARGET_SIZE = 256
NODATA = -9999.0
NORMALIZATION_CLIP = 8.0

PREDICTOR_BANDS = [
    "s2_b2", "s2_b3", "s2_b4", "s2_b8", "s2_b11", "s2_b12",
    "ndvi", "ndbi", "ndmi", "s2_valid",
    "s1_vv_db", "s1_vh_db", "s1_vv_minus_vh_db", "s1_valid",
    "elevation_m", "slope_deg",
]

CONTINUOUS_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 14, 15]
S2_CONTINUOUS_INDICES = set(range(0, 9))
S1_CONTINUOUS_INDICES = {10, 11, 12}
S2_VALID_INDEX = 9
S1_VALID_INDEX = 13

# ------------------------------------------------------------------------------------------
# Input validation
# ------------------------------------------------------------------------------------------
required_paths = [
    ORIGINAL_NORMALIZATION,
    ORIGINAL_TARGET_NORMALIZATION,
    TASK_REGISTER,
    STAGE24_VERDICT,
]

for path in required_paths:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found:\n{path}")

if not CHIP_FOLDER.exists():
    raise FileNotFoundError(f"Exported-chip folder not found:\n{CHIP_FOLDER}")

stage24 = pd.read_csv(STAGE24_VERDICT)

if (
    len(stage24) != EXPECTED_SCENES
    or not stage24["pair_validation_pass"].eq(True).all()
):
    raise RuntimeError(
        "Stage 24R did not record a clean 38/38 PASS. "
        "Do not prepare model-ready arrays."
    )

tasks = pd.read_csv(TASK_REGISTER)

if len(tasks) != EXPECTED_SCENES or tasks["record_id"].nunique() != EXPECTED_SCENES:
    raise RuntimeError("Task register must contain exactly 38 unique scenes.")

# ------------------------------------------------------------------------------------------
# Load frozen original predictor normalization
# ------------------------------------------------------------------------------------------
normalization = pd.read_csv(ORIGINAL_NORMALIZATION)

required_norm_columns = [
    "band_index", "band_name", "mean", "std", "active_channel",
]

missing_norm = [
    column for column in required_norm_columns
    if column not in normalization.columns
]

if missing_norm:
    raise KeyError(
        f"Original normalization file is missing: {missing_norm}"
    )

normalization["band_index"] = pd.to_numeric(
    normalization["band_index"], errors="raise"
).astype(int)

normalization["mean"] = pd.to_numeric(
    normalization["mean"], errors="raise"
)

normalization["std"] = pd.to_numeric(
    normalization["std"], errors="raise"
)

normalization["active_channel"] = (
    normalization["active_channel"]
    .astype(str)
    .str.strip()
    .str.lower()
    .isin(["true", "1", "yes"])
)

normalization = normalization.sort_values("band_index").reset_index(drop=True)

if normalization["band_index"].tolist() != CONTINUOUS_INDICES:
    raise RuntimeError(
        "The original normalization does not match the required "
        "continuous predictor-channel contract."
    )

expected_continuous_names = [
    PREDICTOR_BANDS[index] for index in CONTINUOUS_INDICES
]

if normalization["band_name"].tolist() != expected_continuous_names:
    raise RuntimeError(
        "The original normalization band names do not match the "
        "ThermoFusion predictor order."
    )

if (normalization.loc[normalization["active_channel"], "std"] <= 0).any():
    raise RuntimeError("Original normalization contains non-positive standard deviations.")

normalization.to_csv(OUT_NORMALIZATION_AUDIT, index=False)

with ORIGINAL_TARGET_NORMALIZATION.open("r", encoding="utf-8") as handle:
    target_normalization = json.load(handle)

target_mean = float(target_normalization["mean_c"])
target_std = float(target_normalization["std_c"])

if not np.isfinite(target_mean) or not np.isfinite(target_std) or target_std <= 0:
    raise RuntimeError("Original target normalization is invalid.")

target_audit = {
    "target_name": "land_surface_temperature_c",
    "mean_c": target_mean,
    "std_c": target_std,
    "statistics_source": (
        "Original Stage 12 training scenes only; reused unchanged "
        "for the Stage 25 expanded temporal test."
    ),
}

with OUT_TARGET_AUDIT.open("w", encoding="utf-8") as handle:
    json.dump(target_audit, handle, indent=2)

print("=" * 100)
print("THERMOFUSION STAGE 25 — EXPANDED TEMPORAL-TEST MODEL-READY ARRAYS")
print("=" * 100)
print(f"Verified expanded-test scene pairs: {len(tasks)}")
print(f"Frozen original target mean: {target_mean:.4f} °C")
print(f"Frozen original target standard deviation: {target_std:.4f} °C")

# ------------------------------------------------------------------------------------------
# File matching
# ------------------------------------------------------------------------------------------
all_tifs = sorted(set(
    list(CHIP_FOLDER.rglob("*.tif")) +
    list(CHIP_FOLDER.rglob("*.tiff"))
))

def get_pair_paths(prefix):
    predictor_paths = [
        path for path in all_tifs
        if path.name.startswith(f"{prefix}_predictors_10m")
    ]

    thermal_paths = [
        path for path in all_tifs
        if path.name.startswith(f"{prefix}_thermal_native")
    ]

    if len(predictor_paths) != 1 or len(thermal_paths) != 1:
        raise RuntimeError(
            f"{prefix}: expected one predictor and one thermal file; "
            f"found predictors={len(predictor_paths)}, thermal={len(thermal_paths)}."
        )

    return predictor_paths[0], thermal_paths[0]

# ------------------------------------------------------------------------------------------
# Geometry helper: centre crop or pad to the fixed 256 × 256 model grid.
# Padding uses nodata for predictors and invalid pixels for targets.
# ------------------------------------------------------------------------------------------
def centre_crop_or_pad(x, y, target_mask):
    _, height, width = x.shape

    if height < 250 or width < 250:
        raise RuntimeError(
            f"Predictor chip is too small ({height}×{width}); expected near 256×256."
        )

    output_x = np.full(
        (x.shape[0], TARGET_SIZE, TARGET_SIZE),
        NODATA,
        dtype=np.float32,
    )

    output_y = np.full(
        (TARGET_SIZE, TARGET_SIZE),
        np.nan,
        dtype=np.float32,
    )

    output_mask = np.zeros(
        (TARGET_SIZE, TARGET_SIZE),
        dtype=bool,
    )

    source_row_start = max(0, (height - TARGET_SIZE) // 2)
    source_col_start = max(0, (width - TARGET_SIZE) // 2)

    source_row_end = min(height, source_row_start + TARGET_SIZE)
    source_col_end = min(width, source_col_start + TARGET_SIZE)

    copied_height = source_row_end - source_row_start
    copied_width = source_col_end - source_col_start

    destination_row_start = max(0, (TARGET_SIZE - copied_height) // 2)
    destination_col_start = max(0, (TARGET_SIZE - copied_width) // 2)

    destination_row_end = destination_row_start + copied_height
    destination_col_end = destination_col_start + copied_width

    src_rows = slice(source_row_start, source_row_end)
    src_cols = slice(source_col_start, source_col_end)

    dst_rows = slice(destination_row_start, destination_row_end)
    dst_cols = slice(destination_col_start, destination_col_end)

    output_x[:, dst_rows, dst_cols] = x[:, src_rows, src_cols]
    output_y[dst_rows, dst_cols] = y[src_rows, src_cols]
    output_mask[dst_rows, dst_cols] = target_mask[src_rows, src_cols]

    return (
        output_x,
        output_y,
        output_mask,
        height,
        width,
        source_row_start,
        source_col_start,
        destination_row_start,
        destination_col_start,
    )

# ------------------------------------------------------------------------------------------
# Read a pair, align the native thermal target to the 10 m predictor grid,
# then apply the frozen Stage 12 input contract.
# ------------------------------------------------------------------------------------------
def aligned_scene(predictor_path, thermal_path):
    with rasterio.open(predictor_path) as predictor, rasterio.open(thermal_path) as thermal:
        x = predictor.read().astype(np.float32)

        if x.shape[0] != 16:
            raise RuntimeError(
                f"{predictor_path.name} has {x.shape[0]} predictor bands; expected 16."
            )

        thermal_data = thermal.read()
        y_native = thermal_data[0].astype(np.float32)

        # Prefer the exported thermal-valid band. If it contains no positive
        # values, derive the mask from valid LST values because the LST band was
        # already exported with the original sensor QA mask applied.
        exported_mask = (
            thermal_data[1] > 0.5
            if thermal_data.shape[0] >= 2
            else np.zeros_like(y_native, dtype=bool)
        )

        lst_valid = (
            np.isfinite(y_native)
            & (y_native != NODATA)
            & (y_native > -30.0)
            & (y_native < 80.0)
        )

        if exported_mask.sum() > 0:
            native_mask = (exported_mask & lst_valid).astype(np.uint8)
            mask_source = "exported_thermal_valid_band"
        else:
            native_mask = lst_valid.astype(np.uint8)
            mask_source = "lst_nodata_and_physical_range_fallback"

        y = np.full(
            (predictor.height, predictor.width),
            np.nan,
            dtype=np.float32,
        )

        target_mask = np.zeros(
            (predictor.height, predictor.width),
            dtype=np.uint8,
        )

        reproject(
            source=y_native,
            destination=y,
            src_transform=thermal.transform,
            src_crs=thermal.crs,
            src_nodata=NODATA,
            dst_transform=predictor.transform,
            dst_crs=predictor.crs,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )

        reproject(
            source=native_mask,
            destination=target_mask,
            src_transform=thermal.transform,
            src_crs=thermal.crs,
            src_nodata=0,
            dst_transform=predictor.transform,
            dst_crs=predictor.crs,
            dst_nodata=0,
            resampling=Resampling.nearest,
        )

    (
        x,
        y,
        target_mask,
        original_height,
        original_width,
        source_row_start,
        source_col_start,
        destination_row_start,
        destination_col_start,
    ) = centre_crop_or_pad(
        x,
        y,
        target_mask > 0,
    )

    s2_valid = (
        np.isfinite(x[S2_VALID_INDEX])
        & (x[S2_VALID_INDEX] != NODATA)
        & (x[S2_VALID_INDEX] > 0.5)
    )

    s1_valid = (
        np.isfinite(x[S1_VALID_INDEX])
        & (x[S1_VALID_INDEX] != NODATA)
        & (x[S1_VALID_INDEX] > 0.5)
    )

    target_valid = (
        target_mask
        & np.isfinite(y)
        & (y != NODATA)
        & (y > -30.0)
        & (y < 80.0)
    )

    band_valid = {}

    for band_index in CONTINUOUS_INDICES:
        valid = np.isfinite(x[band_index]) & (x[band_index] != NODATA)

        if band_index in S2_CONTINUOUS_INDICES:
            valid &= s2_valid
        elif band_index in S1_CONTINUOUS_INDICES:
            valid &= s1_valid

        band_valid[band_index] = valid

    return {
        "x": x,
        "y": y,
        "target_valid": target_valid,
        "band_valid": band_valid,
        "s2_valid": s2_valid,
        "s1_valid": s1_valid,
        "mask_source": mask_source,
        "original_height": original_height,
        "original_width": original_width,
        "source_row_start": source_row_start,
        "source_col_start": source_col_start,
        "destination_row_start": destination_row_start,
        "destination_col_start": destination_col_start,
    }

# ------------------------------------------------------------------------------------------
# Prepare and save the 38 frozen-normalization model-ready arrays
# ------------------------------------------------------------------------------------------
array_rows = []

for number, (_, row) in enumerate(tasks.iterrows(), start=1):
    predictor_path, thermal_path = get_pair_paths(str(row["file_prefix"]))

    scene = aligned_scene(predictor_path, thermal_path)

    x = scene["x"]
    y = scene["y"]
    target_valid = scene["target_valid"]
    band_valid = scene["band_valid"]
    s2_valid = scene["s2_valid"]
    s1_valid = scene["s1_valid"]

    if target_valid.sum() == 0:
        raise RuntimeError(
            f"{row['record_id']} has no valid target pixels after alignment."
        )

    x_normalized = np.zeros_like(x, dtype=np.float32)

    for norm_row in normalization.itertuples(index=False):
        band_index = int(norm_row.band_index)

        if not bool(norm_row.active_channel):
            continue

        standardized = (
            (x[band_index] - float(norm_row.mean))
            / float(norm_row.std)
        )

        standardized = np.clip(
            standardized,
            -NORMALIZATION_CLIP,
            NORMALIZATION_CLIP,
        )

        standardized[~band_valid[band_index]] = 0.0

        x_normalized[band_index] = standardized.astype(np.float32)

    # Explicit modality masks remain unstandardized, as in the original model arrays.
    x_normalized[S2_VALID_INDEX] = s2_valid.astype(np.float32)
    x_normalized[S1_VALID_INDEX] = s1_valid.astype(np.float32)

    y_c = np.where(target_valid, y, 0.0).astype(np.float32)

    y_standardized = np.where(
        target_valid,
        (y - target_mean) / target_std,
        0.0,
    ).astype(np.float32)

    valid_mask = target_valid.astype(np.uint8)

    array_path = ARRAY_DIR / f"{row['export_id']}_model_ready.npz"

    np.savez_compressed(
        array_path,
        x=x_normalized,
        y_c=y_c,
        y_standardized=y_standardized,
        valid_mask=valid_mask,
        predictor_band_names=np.asarray(PREDICTOR_BANDS),
        pilot_id=np.asarray(row["export_id"]),
        record_id=np.asarray(row["record_id"]),
        city=np.asarray(row["city"]),
        thermal_sensor=np.asarray(row["thermal_sensor"]),
        model_split=np.asarray("expanded_temporal_test"),
    )

    array_rows.append({
        "export_id": row["export_id"],
        "record_id": row["record_id"],
        "city": row["city"],
        "thermal_sensor": row["thermal_sensor"],
        "model_split": "expanded_temporal_test",
        "predictor_path": str(predictor_path),
        "thermal_path": str(thermal_path),
        "array_path": str(array_path),
        "array_size_mb": array_path.stat().st_size / 1e6,
        "channels": x_normalized.shape[0],
        "height": x_normalized.shape[1],
        "width": x_normalized.shape[2],
        "target_valid_pixels": int(target_valid.sum()),
        "target_valid_fraction": float(target_valid.mean()),
        "s2_valid_fraction": float(s2_valid.mean()),
        "s1_valid_fraction": float(s1_valid.mean()),
        "target_min_c": float(y[target_valid].min()),
        "target_median_c": float(np.median(y[target_valid])),
        "target_max_c": float(y[target_valid].max()),
        "thermal_mask_source": scene["mask_source"],
        "original_predictor_height": scene["original_height"],
        "original_predictor_width": scene["original_width"],
        "source_row_start": scene["source_row_start"],
        "source_col_start": scene["source_col_start"],
        "destination_row_start": scene["destination_row_start"],
        "destination_col_start": scene["destination_col_start"],
        "status": "PASS",
    })

    print(
        f"Prepared {number:02d}/{EXPECTED_SCENES}: {row['record_id']} | "
        f"thermal={target_valid.mean():.1%} | "
        f"S2={s2_valid.mean():.1%} | "
        f"S1={s1_valid.mean():.1%}"
    )

array_manifest = pd.DataFrame(array_rows)
array_manifest.to_csv(OUT_MANIFEST, index=False)

# ------------------------------------------------------------------------------------------
# Reload every saved array and verify the frozen model-input contract
# ------------------------------------------------------------------------------------------
verification_rows = []

for row in array_manifest.itertuples(index=False):
    with np.load(row.array_path, allow_pickle=False) as item:
        x = item["x"]
        y_c = item["y_c"]
        y_standardized = item["y_standardized"]
        mask = item["valid_mask"]

        checks = {
            "shape_check": (
                x.shape == (16, TARGET_SIZE, TARGET_SIZE)
                and y_c.shape == (TARGET_SIZE, TARGET_SIZE)
                and y_standardized.shape == (TARGET_SIZE, TARGET_SIZE)
                and mask.shape == (TARGET_SIZE, TARGET_SIZE)
            ),
            "dtype_check": (
                x.dtype == np.float32
                and y_c.dtype == np.float32
                and y_standardized.dtype == np.float32
                and mask.dtype == np.uint8
            ),
            "finite_check": (
                np.isfinite(x).all()
                and np.isfinite(y_c).all()
                and np.isfinite(y_standardized).all()
            ),
            "mask_check": (
                set(np.unique(mask)).issubset({0, 1})
                and mask.sum() > 0
            ),
            "identity_check": (
                str(item["pilot_id"]) == str(row.export_id)
                and str(item["record_id"]) == str(row.record_id)
            ),
            "channel_check": (
                list(item["predictor_band_names"]) == PREDICTOR_BANDS
            ),
            "frozen_target_scale_check": (
                np.isclose(
                    y_standardized[mask > 0].mean(),
                    (
                        y_c[mask > 0].mean() - target_mean
                    ) / target_std,
                    atol=1e-5,
                )
            ),
        }

        verification_rows.append({
            "export_id": row.export_id,
            "record_id": row.record_id,
            **checks,
            "status": "PASS" if all(checks.values()) else "REVIEW",
        })

integrity = pd.DataFrame(verification_rows)
integrity.to_csv(OUT_INTEGRITY, index=False)

all_arrays_pass = integrity["status"].eq("PASS").all()

verdict_lines = [
    "THERMOFUSION STAGE 25 EXPANDED TEMPORAL-TEST MODEL-READY ARRAYS",
    f"Verified exported scene pairs: {EXPECTED_SCENES}/38",
    f"Model-ready arrays prepared: {len(array_manifest)}/38",
    f"Arrays passing reload integrity checks: {integrity['status'].eq('PASS').sum()}/38",
    "Predictor channels: 16",
    "Array dimensions: 256 x 256",
    f"Frozen original training target mean (°C): {target_mean:.4f}",
    f"Frozen original training target standard deviation (°C): {target_std:.4f}",
    "Predictor normalization source: original Stage 12 training scenes only",
    "Target normalization source: original Stage 12 training scenes only",
    "Expanded-test model split: independent supplementary temporal test",
    "No model fitting, retraining, or recalibration performed",
    (
        "Overall model-ready verdict: PASS"
        if all_arrays_pass and len(array_manifest) == EXPECTED_SCENES
        else "Overall model-ready verdict: REVIEW REQUIRED"
    ),
]

OUT_VERDICT.write_text("\n".join(verdict_lines), encoding="utf-8")

zip_path = shutil.make_archive(
    str(OUTPUT_DIR),
    "zip",
    root_dir=OUTPUT_DIR,
)

print()
print("=" * 100)
print("\n".join(verdict_lines))
print(f"\nArray manifest: {OUT_MANIFEST}")
print(f"Integrity report: {OUT_INTEGRITY}")
print(f"Output package: {zip_path}")

if not all_arrays_pass:
    print("\nARRAYS REQUIRING REVIEW:")
    display(integrity.loc[integrity["status"].ne("PASS")])
