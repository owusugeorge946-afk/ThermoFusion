# =====================================================================
# ThermoFusion reproducibility code
# Source notebook cell(s): 58
# Audit status: ARCHIVE
# Extracted without scientific-logic modification.
# Replace local Google Drive paths as described in README.md when needed.
# =====================================================================

# ==========================================================================================
# THERMOFUSION STAGE 24R — CORRECTED CHIP VERIFICATION
# Accepts normal edge-grid differences from Earth Engine exports.
# Validates file pairs, band counts, readable data, plausible LST values, and native grids.
# ==========================================================================================

from google.colab import drive
from pathlib import Path
import pandas as pd
import numpy as np
import rasterio
from IPython.display import display

drive.mount("/content/drive")

BASE = Path("/content/drive/MyDrive")

EXPORT_DIR = BASE / "ThermoFusion_Stage23R2_FinalReplacement" / "Stage23R2_Export"
TASK_REGISTER = EXPORT_DIR / "03_export_task_register.csv"
CHIP_FOLDER = BASE / "ThermoFusion_Stage23R2_ExpandedTemporalTest_Chips"

OUT_SUMMARY = EXPORT_DIR / "08_corrected_chip_pair_verification_summary.csv"
OUT_DETAILS = EXPORT_DIR / "09_corrected_chip_file_details.csv"
OUT_WARNINGS = EXPORT_DIR / "10_chip_verification_warnings.csv"

EXPECTED_SCENES = 38
NODATA = -9999.0

if not TASK_REGISTER.exists():
    raise FileNotFoundError(f"Task register not found:\n{TASK_REGISTER}")

if not CHIP_FOLDER.exists():
    raise FileNotFoundError(f"Chip folder not found:\n{CHIP_FOLDER}")

tasks = pd.read_csv(TASK_REGISTER)

if len(tasks) != EXPECTED_SCENES or tasks["record_id"].nunique() != EXPECTED_SCENES:
    raise RuntimeError("Task register does not contain exactly 38 unique scenes.")

all_tifs = sorted(set(
    list(CHIP_FOLDER.rglob("*.tif")) +
    list(CHIP_FOLDER.rglob("*.tiff"))
))

print("=" * 100)
print("THERMOFUSION STAGE 24R — CORRECTED EXPORTED-CHIP VERIFICATION")
print("=" * 100)
print(f"Expected files: {EXPECTED_SCENES * 2}")
print(f"GeoTIFF files found: {len(all_tifs)}")

details = []
scene_summary = []
warnings = []

for number, (_, row) in enumerate(tasks.iterrows(), start=1):
    prefix = str(row["file_prefix"])

    predictor_files = [
        p for p in all_tifs
        if p.name.startswith(f"{prefix}_predictors_10m")
    ]
    thermal_files = [
        p for p in all_tifs
        if p.name.startswith(f"{prefix}_thermal_native")
    ]

    pair_pass = True
    messages = []

    # ------------------------------------------------------------------
    # Predictor stack: expected 16 bands; normal GEE edge dimensions accepted.
    # ------------------------------------------------------------------
    if len(predictor_files) != 1:
        pair_pass = False
        messages.append(f"Predictor files found: {len(predictor_files)}")
    else:
        predictor_path = predictor_files[0]

        try:
            with rasterio.open(predictor_path) as src:
                arr = src.read(masked=True)

                valid_predictor_pixels = int(
                    np.any(~np.ma.getmaskarray(arr), axis=0).sum()
                )

                # A 2,560 m chip at 10 m may be 255, 256, or 257 pixels,
                # depending on grid alignment at the export boundary.
                normal_size = (
                    250 <= src.width <= 260
                    and 250 <= src.height <= 260
                )

                predictor_pass = (
                    src.count == 16
                    and src.crs is not None
                    and normal_size
                    and valid_predictor_pixels > 0
                )

                if not predictor_pass:
                    pair_pass = False
                    messages.append(
                        f"Predictor check failed: bands={src.count}, "
                        f"size={src.width}×{src.height}, "
                        f"valid_pixels={valid_predictor_pixels}"
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
                })

        except Exception as exc:
            pair_pass = False
            messages.append(f"Predictor unreadable: {exc}")

    # ------------------------------------------------------------------
    # Thermal target: expected 2 bands; native Landsat and ECOSTRESS grids differ.
    # ------------------------------------------------------------------
    if len(thermal_files) != 1:
        pair_pass = False
        messages.append(f"Thermal files found: {len(thermal_files)}")
    else:
        thermal_path = thermal_files[0]

        try:
            with rasterio.open(thermal_path) as src:
                lst = src.read(1, masked=True)
                valid_band = src.read(2, masked=True) if src.count >= 2 else None

                lst_values = np.asarray(lst.filled(np.nan), dtype=float)

                plausible_lst = (
                    np.isfinite(lst_values)
                    & (lst_values > -30.0)
                    & (lst_values < 80.0)
                )

                plausible_lst_pixels = int(plausible_lst.sum())

                if valid_band is not None:
                    valid_values = np.asarray(
                        valid_band.filled(np.nan),
                        dtype=float,
                    )

                    positive_valid_pixels = int(
                        np.isfinite(valid_values)
                        & (valid_values > 0.5)
                    ).sum()

                    non_binary_values = np.isfinite(valid_values) & ~np.isin(
                        np.round(valid_values, 6),
                        [0.0, 1.0],
                    )

                    mask_binary = not bool(non_binary_values.any())
                else:
                    positive_valid_pixels = 0
                    mask_binary = False

                # Landsat targets are ~30 m, ECOSTRESS targets are ~70 m.
                # Their native export dimensions therefore should not be 256×256.
                normal_native_size = (
                    25 <= src.width <= 100
                    and 25 <= src.height <= 100
                )

                thermal_pass = (
                    src.count == 2
                    and src.crs is not None
                    and normal_native_size
                    and plausible_lst_pixels > 0
                )

                if not thermal_pass:
                    pair_pass = False
                    messages.append(
                        f"Thermal check failed: bands={src.count}, "
                        f"size={src.width}×{src.height}, "
                        f"plausible_LST_pixels={plausible_lst_pixels}"
                    )

                # This is reported as a warning only. Stage 25 derives the
                # final target mask from the LST value and nodata convention.
                if not mask_binary:
                    warnings.append({
                        "export_id": row["export_id"],
                        "record_id": row["record_id"],
                        "export_type": "thermal",
                        "warning": "Thermal-valid band contains values other than 0 and 1.",
                    })

                if positive_valid_pixels == 0:
                    warnings.append({
                        "export_id": row["export_id"],
                        "record_id": row["record_id"],
                        "export_type": "thermal",
                        "warning": (
                            "No positive pixels in exported thermal-valid band; "
                            "Stage 25 will derive validity from LST nodata/range."
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
                    "thermal_valid_is_binary": mask_binary,
                    "validation_pass": thermal_pass,
                })

        except Exception as exc:
            pair_pass = False
            messages.append(f"Thermal unreadable: {exc}")

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

details_df = pd.DataFrame(details)
summary_df = pd.DataFrame(scene_summary)
warnings_df = pd.DataFrame(warnings)

details_df.to_csv(OUT_DETAILS, index=False)
summary_df.to_csv(OUT_SUMMARY, index=False)
warnings_df.to_csv(OUT_WARNINGS, index=False)

failed = summary_df.loc[~summary_df["pair_validation_pass"]].copy()

print()
print("=" * 100)
print("STAGE 24R VERIFICATION SUMMARY")
print("=" * 100)
print(f"Valid predictor–thermal pairs: {summary_df['pair_validation_pass'].sum()}/{EXPECTED_SCENES}")
print(f"Failed pairs: {len(failed)}")
print(f"Summary file: {OUT_SUMMARY}")
print(f"Detailed file report: {OUT_DETAILS}")
print(f"Warnings file: {OUT_WARNINGS}")

if failed.empty:
    print("\nFINAL VERDICT: PASS — all 38 exported predictor–thermal pairs are usable.")
else:
    print("\nFAILED PAIRS:")
    display(failed)
    print("\nFINAL VERDICT: STOP — send me this table before continuing.")
