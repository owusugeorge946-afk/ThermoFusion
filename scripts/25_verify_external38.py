# =====================================================================
# ThermoFusion reproducibility code
# Source notebook cell(s): 59
# Audit status: KEEP
# Extracted without scientific-logic modification.
# Replace local Google Drive paths as described in README.md when needed.
# =====================================================================

# ==========================================================================================
# THERMOFUSION STAGE 24R — FINAL CORRECTED EXPORTED-CHIP VERIFICATION
# Checks all 38 predictor–thermal pairs without modifying the exported GeoTIFF files.
# ==========================================================================================

from google.colab import drive
from pathlib import Path
import pandas as pd
import numpy as np
import rasterio
from IPython.display import display

drive.mount("/content/drive")

# ------------------------------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------------------------------
BASE = Path("/content/drive/MyDrive")

EXPORT_DIR = BASE / "ThermoFusion_Stage23R2_FinalReplacement" / "Stage23R2_Export"
TASK_REGISTER = EXPORT_DIR / "03_export_task_register.csv"
CHIP_FOLDER = BASE / "ThermoFusion_Stage23R2_ExpandedTemporalTest_Chips"

OUT_SUMMARY = EXPORT_DIR / "08_corrected_chip_pair_verification_summary.csv"
OUT_DETAILS = EXPORT_DIR / "09_corrected_chip_file_details.csv"
OUT_WARNINGS = EXPORT_DIR / "10_chip_verification_warnings.csv"

EXPECTED_SCENES = 38

if not TASK_REGISTER.exists():
    raise FileNotFoundError(f"Task register not found:\n{TASK_REGISTER}")

if not CHIP_FOLDER.exists():
    raise FileNotFoundError(f"Chip folder not found:\n{CHIP_FOLDER}")

tasks = pd.read_csv(TASK_REGISTER)

if len(tasks) != EXPECTED_SCENES:
    raise RuntimeError(
        f"Expected {EXPECTED_SCENES} exported scenes; found {len(tasks)}."
    )

if tasks["record_id"].nunique() != EXPECTED_SCENES:
    raise RuntimeError("Task register contains duplicate record IDs.")

# Find all exported GeoTIFF files.
all_tifs = sorted(set(
    list(CHIP_FOLDER.rglob("*.tif")) +
    list(CHIP_FOLDER.rglob("*.tiff"))
))

print("=" * 100)
print("THERMOFUSION STAGE 24R — FINAL EXPORTED-CHIP VERIFICATION")
print("=" * 100)
print(f"Expected GeoTIFF files: {EXPECTED_SCENES * 2}")
print(f"GeoTIFF files found: {len(all_tifs)}")
print(f"Chip folder: {CHIP_FOLDER}")

details = []
scene_summary = []
warnings = []

# ------------------------------------------------------------------------------------------
# Verify each predictor–thermal pair
# ------------------------------------------------------------------------------------------
for number, (_, row) in enumerate(tasks.iterrows(), start=1):
    prefix = str(row["file_prefix"])

    predictor_files = [
        path for path in all_tifs
        if path.name.startswith(f"{prefix}_predictors_10m")
    ]

    thermal_files = [
        path for path in all_tifs
        if path.name.startswith(f"{prefix}_thermal_native")
    ]

    pair_pass = True
    messages = []

    # --------------------------------------------------------------------------------------
    # Predictor stack validation
    # --------------------------------------------------------------------------------------
    if len(predictor_files) != 1:
        pair_pass = False
        messages.append(f"Predictor files found: {len(predictor_files)}")

        details.append({
            "export_id": row["export_id"],
            "record_id": row["record_id"],
            "city": row["city"],
            "thermal_sensor": row["thermal_sensor"],
            "export_type": "predictors",
            "file_path": "",
            "bands": np.nan,
            "width": np.nan,
            "height": np.nan,
            "crs": "",
            "nodata": np.nan,
            "usable_pixels": np.nan,
            "validation_pass": False,
            "message": "Predictor file missing or duplicated",
        })

    else:
        predictor_path = predictor_files[0]

        try:
            with rasterio.open(predictor_path) as src:
                predictor_data = src.read(masked=True)

                valid_predictor_pixels = int(
                    np.any(~np.ma.getmaskarray(predictor_data), axis=0).sum()
                )

                # Grid alignment at the boundary can produce 255–257 pixels
                # rather than exactly 256 for a nominal 2,560 m chip.
                normal_predictor_size = (
                    250 <= src.width <= 260
                    and 250 <= src.height <= 260
                )

                predictor_pass = (
                    src.count == 16
                    and src.crs is not None
                    and normal_predictor_size
                    and valid_predictor_pixels > 0
                )

                if not predictor_pass:
                    pair_pass = False
                    messages.append(
                        f"Predictor invalid: bands={src.count}, "
                        f"size={src.width}×{src.height}, "
                        f"usable_pixels={valid_predictor_pixels}"
                    )

                details.append({
                    "export_id": row["export_id"],
                    "record_id": row["record_id"],
                    "city": row["city"],
                    "thermal_sensor": row["thermal_sensor"],
                    "export_type": "predictors",
                    "file_path": str(predictor_path),
                    "bands": src.count,
                    "width": src.width,
                    "height": src.height,
                    "crs": str(src.crs),
                    "nodata": src.nodata,
                    "usable_pixels": valid_predictor_pixels,
                    "validation_pass": predictor_pass,
                    "message": "",
                })

        except Exception as exc:
            pair_pass = False
            messages.append(f"Predictor unreadable: {exc}")

            details.append({
                "export_id": row["export_id"],
                "record_id": row["record_id"],
                "city": row["city"],
                "thermal_sensor": row["thermal_sensor"],
                "export_type": "predictors",
                "file_path": str(predictor_path),
                "bands": np.nan,
                "width": np.nan,
                "height": np.nan,
                "crs": "",
                "nodata": np.nan,
                "usable_pixels": np.nan,
                "validation_pass": False,
                "message": str(exc),
            })

    # --------------------------------------------------------------------------------------
    # Thermal target validation
    # --------------------------------------------------------------------------------------
    if len(thermal_files) != 1:
        pair_pass = False
        messages.append(f"Thermal files found: {len(thermal_files)}")

        details.append({
            "export_id": row["export_id"],
            "record_id": row["record_id"],
            "city": row["city"],
            "thermal_sensor": row["thermal_sensor"],
            "export_type": "thermal",
            "file_path": "",
            "bands": np.nan,
            "width": np.nan,
            "height": np.nan,
            "crs": "",
            "nodata": np.nan,
            "usable_pixels": np.nan,
            "positive_valid_mask_pixels": np.nan,
            "thermal_valid_is_binary": False,
            "validation_pass": False,
            "message": "Thermal file missing or duplicated",
        })

    else:
        thermal_path = thermal_files[0]

        try:
            with rasterio.open(thermal_path) as src:
                thermal_lst = src.read(1, masked=True)
                thermal_valid = src.read(2, masked=True) if src.count >= 2 else None

                lst_values = np.asarray(
                    thermal_lst.filled(np.nan),
                    dtype=np.float32,
                )

                plausible_lst = (
                    np.isfinite(lst_values)
                    & (lst_values > -30.0)
                    & (lst_values < 80.0)
                )

                plausible_lst_pixels = int(plausible_lst.sum())

                if thermal_valid is not None:
                    valid_values = np.asarray(
                        thermal_valid.filled(np.nan),
                        dtype=np.float32,
                    )

                    positive_valid_pixels = int(
                        (np.isfinite(valid_values) & (valid_values > 0.5)).sum()
                    )

                    finite_valid_values = valid_values[np.isfinite(valid_values)]

                    mask_is_binary = bool(
                        np.all(
                            np.isin(
                                np.round(finite_valid_values, 6),
                                [0.0, 1.0],
                            )
                        )
                    ) if finite_valid_values.size > 0 else True

                else:
                    positive_valid_pixels = 0
                    mask_is_binary = False

                # Native target chips are approximately:
                # Landsat: 85–87 pixels at 30 m
                # ECOSTRESS: 36–39 pixels at 70 m
                normal_thermal_size = (
                    25 <= src.width <= 100
                    and 25 <= src.height <= 100
                )

                thermal_pass = (
                    src.count == 2
                    and src.crs is not None
                    and normal_thermal_size
                    and plausible_lst_pixels > 0
                )

                if not thermal_pass:
                    pair_pass = False
                    messages.append(
                        f"Thermal invalid: bands={src.count}, "
                        f"size={src.width}×{src.height}, "
                        f"plausible_LST_pixels={plausible_lst_pixels}"
                    )

                # These are warnings, not automatic failures. The model-ready
                # conversion derives its final target mask from valid LST pixels.
                if not mask_is_binary:
                    warnings.append({
                        "export_id": row["export_id"],
                        "record_id": row["record_id"],
                        "city": row["city"],
                        "thermal_sensor": row["thermal_sensor"],
                        "warning": "Thermal-valid band is not strictly binary.",
                    })

                if positive_valid_pixels == 0:
                    warnings.append({
                        "export_id": row["export_id"],
                        "record_id": row["record_id"],
                        "city": row["city"],
                        "thermal_sensor": row["thermal_sensor"],
                        "warning": (
                            "No positive pixels in thermal-valid band. "
                            "Stage 25 will derive the target mask from valid LST values."
                        ),
                    })

                details.append({
                    "export_id": row["export_id"],
                    "record_id": row["record_id"],
                    "city": row["city"],
                    "thermal_sensor": row["thermal_sensor"],
                    "export_type": "thermal",
                    "file_path": str(thermal_path),
                    "bands": src.count,
                    "width": src.width,
                    "height": src.height,
                    "crs": str(src.crs),
                    "nodata": src.nodata,
                    "usable_pixels": plausible_lst_pixels,
                    "positive_valid_mask_pixels": positive_valid_pixels,
                    "thermal_valid_is_binary": mask_is_binary,
                    "validation_pass": thermal_pass,
                    "message": "",
                })

        except Exception as exc:
            pair_pass = False
            messages.append(f"Thermal unreadable: {exc}")

            details.append({
                "export_id": row["export_id"],
                "record_id": row["record_id"],
                "city": row["city"],
                "thermal_sensor": row["thermal_sensor"],
                "export_type": "thermal",
                "file_path": str(thermal_path),
                "bands": np.nan,
                "width": np.nan,
                "height": np.nan,
                "crs": "",
                "nodata": np.nan,
                "usable_pixels": np.nan,
                "positive_valid_mask_pixels": np.nan,
                "thermal_valid_is_binary": False,
                "validation_pass": False,
                "message": str(exc),
            })

    scene_summary.append({
        "export_id": row["export_id"],
        "record_id": row["record_id"],
        "city": row["city"],
        "thermal_sensor": row["thermal_sensor"],
        "pair_validation_pass": pair_pass,
        "message": " | ".join(messages),
    })

    print(
        f"Checked {number:02d}/{EXPECTED_SCENES}: {row['record_id']} — "
        f"{'PASS' if pair_pass else 'FAIL'}"
    )

# ------------------------------------------------------------------------------------------
# Save reports and give final status
# ------------------------------------------------------------------------------------------
details_df = pd.DataFrame(details)
summary_df = pd.DataFrame(scene_summary)
warnings_df = pd.DataFrame(warnings)

details_df.to_csv(OUT_DETAILS, index=False)
summary_df.to_csv(OUT_SUMMARY, index=False)
warnings_df.to_csv(OUT_WARNINGS, index=False)

failed = summary_df.loc[
    ~summary_df["pair_validation_pass"]
].copy()

print()
print("=" * 100)
print("STAGE 24R FINAL VERIFICATION SUMMARY")
print("=" * 100)
print(f"Valid predictor–thermal pairs: {int(summary_df['pair_validation_pass'].sum())}/{EXPECTED_SCENES}")
print(f"Failed pairs: {len(failed)}")
print(f"Summary: {OUT_SUMMARY}")
print(f"Details: {OUT_DETAILS}")
print(f"Warnings: {OUT_WARNINGS}")

if failed.empty:
    print("\nFINAL VERDICT: PASS — all 38 predictor–thermal pairs are ready for Stage 25.")
else:
    print("\nFAILED PAIRS:")
    display(failed)
    print("\nFINAL VERDICT: STOP — send the failed-pair table before continuing.")
