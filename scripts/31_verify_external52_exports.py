# =====================================================================
# ThermoFusion reproducibility code
# Source notebook cell(s): 66, 67
# Audit status: MERGE
# Extracted without scientific-logic modification.
# Replace local Google Drive paths as described in README.md when needed.
# =====================================================================

# ======================================================================================
# THERMOFUSION — LARGE EXTERNAL TEMPORAL EVALUATION
# STAGE 04C: VERIFY EXTERNAL52 BATCH 01 EXPORTS
# ======================================================================================

from google.colab import drive
drive.mount('/content/drive')

from pathlib import Path
import pandas as pd
import numpy as np
import ee
import rasterio

PROJECT_ID = "nana213"

ROOT = Path("/content/drive/MyDrive")

TASK_REGISTER = (
    ROOT
    / "ThermoFusion_External_Evaluation"
    / "Stage04_Export"
    / "04_batch01_export_task_register.csv"
)

CHIP_DIR = (
    ROOT
    / "ThermoFusion_External52_Chips"
)

OUTDIR = (
    ROOT
    / "ThermoFusion_External_Evaluation"
    / "Stage04_Export"
)

NODATA = -9999

print("=" * 110)
print("THERMOFUSION STAGE 04C — BATCH 01 EXPORT VERIFICATION")
print("=" * 110)

# ======================================================================================
# 1. LOAD TASK REGISTER
# ======================================================================================

if not TASK_REGISTER.exists():
    raise FileNotFoundError(TASK_REGISTER)

tasks = pd.read_csv(TASK_REGISTER)

if len(tasks) != 20:
    raise RuntimeError(
        f"Expected 20 Batch-1 scenes, found {len(tasks)}."
    )

print(f"\nBatch scenes: {len(tasks)}")
print(f"Expected predictor files: {len(tasks)}")
print(f"Expected thermal files:   {len(tasks)}")

# ======================================================================================
# 2. CHECK EARTH ENGINE TASK STATUS
# ======================================================================================

ee.Authenticate()
ee.Initialize(project=PROJECT_ID)

task_rows = []

for row in tasks.itertuples(index=False):

    predictor_status = ee.data.getTaskStatus(
        str(row.predictor_task_id)
    )[0]

    thermal_status = ee.data.getTaskStatus(
        str(row.thermal_task_id)
    )[0]

    task_rows.append({
        "export_id": row.export_id,
        "record_id": row.record_id,

        "predictor_state":
            predictor_status.get("state"),

        "predictor_error":
            predictor_status.get(
                "error_message", ""
            ),

        "thermal_state":
            thermal_status.get("state"),

        "thermal_error":
            thermal_status.get(
                "error_message", ""
            ),
    })

task_status = pd.DataFrame(task_rows)

print("\nEARTH ENGINE TASK STATES")
print("-" * 110)

print("\nPredictor tasks:")
print(
    task_status["predictor_state"]
    .value_counts(dropna=False)
    .to_string()
)

print("\nThermal tasks:")
print(
    task_status["thermal_state"]
    .value_counts(dropna=False)
    .to_string()
)

# ======================================================================================
# 3. LOCATE PHYSICAL FILES IN DRIVE
# ======================================================================================

records = []

for row in tasks.itertuples(index=False):

    prefix = str(row.prefix)

    predictor_matches = sorted(
        CHIP_DIR.glob(
            f"{prefix}_predictors_10m*.tif"
        )
    )

    thermal_matches = sorted(
        CHIP_DIR.glob(
            f"{prefix}_thermal_native*.tif"
        )
    )

    predictor_path = (
        predictor_matches[0]
        if len(predictor_matches) == 1
        else None
    )

    thermal_path = (
        thermal_matches[0]
        if len(thermal_matches) == 1
        else None
    )

    rec = {
        "export_id": row.export_id,
        "external_id": row.external_id,
        "record_id": row.record_id,
        "city": row.city,
        "thermal_sensor": row.thermal_sensor,

        "predictor_copies":
            len(predictor_matches),

        "thermal_copies":
            len(thermal_matches),

        "predictor_path":
            str(predictor_path)
            if predictor_path
            else "",

        "thermal_path":
            str(thermal_path)
            if thermal_path
            else "",
    }

    # ------------------------------------------------------------------
    # 4. VERIFY PREDICTOR
    # ------------------------------------------------------------------

    if predictor_path is not None:

        try:

            with rasterio.open(
                predictor_path
            ) as src:

                predictor_data = src.read()

                rec.update({
                    "predictor_bands":
                        src.count,

                    "predictor_width":
                        src.width,

                    "predictor_height":
                        src.height,

                    "predictor_crs":
                        str(src.crs),

                    "predictor_xres":
                        abs(src.res[0]),

                    "predictor_yres":
                        abs(src.res[1]),

                    "predictor_dtype":
                        "|".join(
                            src.dtypes
                        ),

                    "predictor_band_check":
                        src.count == 16,

                    "predictor_resolution_check":
                        (
                            abs(
                                abs(src.res[0])
                                - 10
                            ) < 0.1
                            and
                            abs(
                                abs(src.res[1])
                                - 10
                            ) < 0.1
                        ),

                    "predictor_finite_check":
                        np.isfinite(
                            predictor_data[
                                predictor_data
                                != NODATA
                            ]
                        ).all()
                        if np.any(
                            predictor_data
                            != NODATA
                        )
                        else False,
                })

                # S2 validity is band 10
                s2_valid = (
                    np.isfinite(
                        predictor_data[9]
                    )
                    &
                    (
                        predictor_data[9]
                        != NODATA
                    )
                    &
                    (
                        predictor_data[9]
                        > 0.5
                    )
                )

                # S1 validity is band 14
                s1_valid = (
                    np.isfinite(
                        predictor_data[13]
                    )
                    &
                    (
                        predictor_data[13]
                        != NODATA
                    )
                    &
                    (
                        predictor_data[13]
                        > 0.5
                    )
                )

                rec[
                    "s2_valid_fraction"
                ] = float(
                    s2_valid.mean()
                )

                rec[
                    "s1_valid_fraction"
                ] = float(
                    s1_valid.mean()
                )

        except Exception as e:

            rec[
                "predictor_read_error"
            ] = str(e)

    # ------------------------------------------------------------------
    # 5. VERIFY THERMAL
    # ------------------------------------------------------------------

    if thermal_path is not None:

        try:

            with rasterio.open(
                thermal_path
            ) as src:

                thermal_data = src.read()

                rec.update({
                    "thermal_bands":
                        src.count,

                    "thermal_width":
                        src.width,

                    "thermal_height":
                        src.height,

                    "thermal_crs":
                        str(src.crs),

                    "thermal_dtype":
                        "|".join(
                            src.dtypes
                        ),

                    "thermal_band_check":
                        src.count == 2,
                })

                lst = (
                    thermal_data[0]
                    .astype("float32")
                )

                valid = (
                    np.isfinite(lst)
                    &
                    (lst != NODATA)
                    &
                    (
                        thermal_data[1]
                        > 0.5
                    )
                )

                values = lst[valid]

                rec[
                    "thermal_valid_fraction"
                ] = float(
                    valid.mean()
                )

                rec[
                    "thermal_valid_pixels"
                ] = int(
                    valid.sum()
                )

                if values.size:

                    rec[
                        "thermal_min_c"
                    ] = float(
                        values.min()
                    )

                    rec[
                        "thermal_median_c"
                    ] = float(
                        np.median(values)
                    )

                    rec[
                        "thermal_max_c"
                    ] = float(
                        values.max()
                    )

                    rec[
                        "thermal_range_check"
                    ] = bool(
                        values.min() > -30
                        and
                        values.max() < 80
                    )

                else:

                    rec[
                        "thermal_range_check"
                    ] = False

        except Exception as e:

            rec[
                "thermal_read_error"
            ] = str(e)

    records.append(rec)

verification = pd.DataFrame(records)

# ======================================================================================
# 6. FINAL PASS/FAIL LOGIC
# ======================================================================================

def scene_status(row):

    required = [
        row.get(
            "predictor_copies", 0
        ) == 1,

        row.get(
            "thermal_copies", 0
        ) == 1,

        row.get(
            "predictor_band_check",
            False
        ),

        row.get(
            "thermal_band_check",
            False
        ),

        row.get(
            "predictor_resolution_check",
            False
        ),

        row.get(
            "predictor_finite_check",
            False
        ),

        row.get(
            "thermal_range_check",
            False
        ),

        row.get(
            "thermal_valid_fraction",
            0
        ) >= 0.10,
    ]

    return (
        "PASS"
        if all(required)
        else "REVIEW"
    )

verification[
    "verification_status"
] = verification.apply(
    scene_status,
    axis=1
)

# ======================================================================================
# 7. MERGE EE TASK STATUS
# ======================================================================================

verification = verification.merge(
    task_status,
    on=[
        "export_id",
        "record_id"
    ],
    how="left",
    validate="one_to_one"
)

# ======================================================================================
# 8. SAVE
# ======================================================================================

VERIFY_PATH = (
    OUTDIR
    / "05_batch01_export_verification.csv"
)

STATUS_PATH = (
    OUTDIR
    / "06_batch01_task_status.csv"
)

verification.to_csv(
    VERIFY_PATH,
    index=False
)

task_status.to_csv(
    STATUS_PATH,
    index=False
)

# ======================================================================================
# 9. REPORT
# ======================================================================================

print("\n" + "=" * 110)
print("BATCH 01 PHYSICAL FILE VERIFICATION")
print("=" * 110)

print(
    verification[
        "verification_status"
    ]
    .value_counts(
        dropna=False
    )
    .to_string()
)

print("\nFILES FOUND")

print(
    f"Predictor GeoTIFFs: "
    f"{verification['predictor_copies'].eq(1).sum()}/20"
)

print(
    f"Thermal GeoTIFFs:   "
    f"{verification['thermal_copies'].eq(1).sum()}/20"
)

print("\nMean validity:")

if (
    "s2_valid_fraction"
    in verification.columns
):
    print(
        f"S2 mean valid fraction: "
        f"{verification['s2_valid_fraction'].mean():.3f}"
    )

if (
    "s1_valid_fraction"
    in verification.columns
):
    print(
        f"S1 mean valid fraction: "
        f"{verification['s1_valid_fraction'].mean():.3f}"
    )

if (
    "thermal_valid_fraction"
    in verification.columns
):
    print(
        f"Thermal mean valid fraction: "
        f"{verification['thermal_valid_fraction'].mean():.3f}"
    )

print("\nSaved:")
print(VERIFY_PATH)
print(STATUS_PATH)

print("\n" + "=" * 110)

n_pass = (
    verification[
        "verification_status"
    ]
    .eq("PASS")
    .sum()
)

if n_pass == 20:

    print(
        "PASS — Batch 01 is complete and valid."
    )

    print(
        "Batch 02 may now be submitted."
    )

else:

    print(
        f"STOP — only {n_pass}/20 scenes "
        f"passed verification."
    )

    print(
        "Do not submit Batch 02 until "
        "the REVIEW scenes are resolved."
    )

print("=" * 110)


# ==================== MERGED CELL BOUNDARY ====================

# ======================================================================================
# THERMOFUSION — LARGE EXTERNAL TEMPORAL EVALUATION
# STAGE 04C: VERIFY EXTERNAL52 BATCH 02 EXPORTS
# ======================================================================================

from google.colab import drive
drive.mount('/content/drive')

from pathlib import Path
import pandas as pd
import numpy as np
import ee
import rasterio

PROJECT_ID = "nana213"

ROOT = Path("/content/drive/MyDrive")

TASK_REGISTER = (
    ROOT
    / "ThermoFusion_External_Evaluation"
    / "Stage04_Export"
    / "04_batch02_export_task_register.csv"
)

CHIP_DIR = (
    ROOT
    / "ThermoFusion_External52_Chips"
)

OUTDIR = (
    ROOT
    / "ThermoFusion_External_Evaluation"
    / "Stage04_Export"
)

NODATA = -9999

print("=" * 110)
print("THERMOFUSION STAGE 04C — BATCH 02 EXPORT VERIFICATION")
print("=" * 110)

# ======================================================================================
# 1. LOAD TASK REGISTER
# ======================================================================================

if not TASK_REGISTER.exists():
    raise FileNotFoundError(TASK_REGISTER)

tasks = pd.read_csv(TASK_REGISTER)

if len(tasks) != 20:
    raise RuntimeError(
        f"Expected 20 Batch-2 scenes, found {len(tasks)}."
    )

print(f"\nBatch scenes: {len(tasks)}")
print(f"Expected predictor files: {len(tasks)}")
print(f"Expected thermal files:   {len(tasks)}")

# ======================================================================================
# 2. CHECK EARTH ENGINE TASK STATUS
# ======================================================================================

ee.Authenticate()
ee.Initialize(project=PROJECT_ID)

task_rows = []

for row in tasks.itertuples(index=False):

    predictor_status = ee.data.getTaskStatus(
        str(row.predictor_task_id)
    )[0]

    thermal_status = ee.data.getTaskStatus(
        str(row.thermal_task_id)
    )[0]

    task_rows.append({
        "export_id": row.export_id,
        "record_id": row.record_id,

        "predictor_state":
            predictor_status.get("state"),

        "predictor_error":
            predictor_status.get(
                "error_message", ""
            ),

        "thermal_state":
            thermal_status.get("state"),

        "thermal_error":
            thermal_status.get(
                "error_message", ""
            ),
    })

task_status = pd.DataFrame(task_rows)

print("\nEARTH ENGINE TASK STATES")
print("-" * 110)

print("\nPredictor tasks:")
print(
    task_status["predictor_state"]
    .value_counts(dropna=False)
    .to_string()
)

print("\nThermal tasks:")
print(
    task_status["thermal_state"]
    .value_counts(dropna=False)
    .to_string()
)

# ======================================================================================
# 3. LOCATE PHYSICAL FILES IN DRIVE
# ======================================================================================

records = []

for row in tasks.itertuples(index=False):

    prefix = str(row.prefix)

    predictor_matches = sorted(
        CHIP_DIR.glob(
            f"{prefix}_predictors_10m*.tif"
        )
    )

    thermal_matches = sorted(
        CHIP_DIR.glob(
            f"{prefix}_thermal_native*.tif"
        )
    )

    predictor_path = (
        predictor_matches[0]
        if len(predictor_matches) == 1
        else None
    )

    thermal_path = (
        thermal_matches[0]
        if len(thermal_matches) == 1
        else None
    )

    rec = {
        "export_id": row.export_id,
        "external_id": row.external_id,
        "record_id": row.record_id,
        "city": row.city,
        "thermal_sensor": row.thermal_sensor,

        "predictor_copies":
            len(predictor_matches),

        "thermal_copies":
            len(thermal_matches),

        "predictor_path":
            str(predictor_path)
            if predictor_path
            else "",

        "thermal_path":
            str(thermal_path)
            if thermal_path
            else "",
    }

    # ==================================================================================
    # 4. VERIFY PREDICTOR
    # ==================================================================================

    if predictor_path is not None:

        try:

            with rasterio.open(
                predictor_path
            ) as src:

                predictor_data = src.read()

                rec.update({
                    "predictor_bands":
                        src.count,

                    "predictor_width":
                        src.width,

                    "predictor_height":
                        src.height,

                    "predictor_crs":
                        str(src.crs),

                    "predictor_xres":
                        abs(src.res[0]),

                    "predictor_yres":
                        abs(src.res[1]),

                    "predictor_dtype":
                        "|".join(
                            src.dtypes
                        ),

                    "predictor_band_check":
                        src.count == 16,

                    "predictor_resolution_check":
                        (
                            abs(
                                abs(src.res[0])
                                - 10
                            ) < 0.1
                            and
                            abs(
                                abs(src.res[1])
                                - 10
                            ) < 0.1
                        ),

                    "predictor_finite_check":
                        np.isfinite(
                            predictor_data[
                                predictor_data
                                != NODATA
                            ]
                        ).all()
                        if np.any(
                            predictor_data
                            != NODATA
                        )
                        else False,
                })

                # Sentinel-2 validity band = band 10
                s2_valid = (
                    np.isfinite(
                        predictor_data[9]
                    )
                    &
                    (
                        predictor_data[9]
                        != NODATA
                    )
                    &
                    (
                        predictor_data[9]
                        > 0.5
                    )
                )

                # Sentinel-1 validity band = band 14
                s1_valid = (
                    np.isfinite(
                        predictor_data[13]
                    )
                    &
                    (
                        predictor_data[13]
                        != NODATA
                    )
                    &
                    (
                        predictor_data[13]
                        > 0.5
                    )
                )

                rec[
                    "s2_valid_fraction"
                ] = float(
                    s2_valid.mean()
                )

                rec[
                    "s1_valid_fraction"
                ] = float(
                    s1_valid.mean()
                )

        except Exception as e:

            rec[
                "predictor_read_error"
            ] = str(e)

    # ==================================================================================
    # 5. VERIFY THERMAL
    # ==================================================================================

    if thermal_path is not None:

        try:

            with rasterio.open(
                thermal_path
            ) as src:

                thermal_data = src.read()

                rec.update({
                    "thermal_bands":
                        src.count,

                    "thermal_width":
                        src.width,

                    "thermal_height":
                        src.height,

                    "thermal_crs":
                        str(src.crs),

                    "thermal_dtype":
                        "|".join(
                            src.dtypes
                        ),

                    "thermal_band_check":
                        src.count == 2,
                })

                lst = (
                    thermal_data[0]
                    .astype("float32")
                )

                valid = (
                    np.isfinite(lst)
                    &
                    (lst != NODATA)
                    &
                    (
                        thermal_data[1]
                        > 0.5
                    )
                )

                values = lst[valid]

                rec[
                    "thermal_valid_fraction"
                ] = float(
                    valid.mean()
                )

                rec[
                    "thermal_valid_pixels"
                ] = int(
                    valid.sum()
                )

                if values.size:

                    rec[
                        "thermal_min_c"
                    ] = float(
                        values.min()
                    )

                    rec[
                        "thermal_median_c"
                    ] = float(
                        np.median(values)
                    )

                    rec[
                        "thermal_max_c"
                    ] = float(
                        values.max()
                    )

                    rec[
                        "thermal_range_check"
                    ] = bool(
                        values.min() > -30
                        and
                        values.max() < 80
                    )

                else:

                    rec[
                        "thermal_range_check"
                    ] = False

        except Exception as e:

            rec[
                "thermal_read_error"
            ] = str(e)

    records.append(rec)

verification = pd.DataFrame(records)

# ======================================================================================
# 6. FINAL PASS / REVIEW LOGIC
# ======================================================================================

def scene_status(row):

    required = [
        row.get(
            "predictor_copies", 0
        ) == 1,

        row.get(
            "thermal_copies", 0
        ) == 1,

        row.get(
            "predictor_band_check",
            False
        ),

        row.get(
            "thermal_band_check",
            False
        ),

        row.get(
            "predictor_resolution_check",
            False
        ),

        row.get(
            "predictor_finite_check",
            False
        ),

        row.get(
            "thermal_range_check",
            False
        ),

        row.get(
            "thermal_valid_fraction",
            0
        ) >= 0.10,
    ]

    return (
        "PASS"
        if all(required)
        else "REVIEW"
    )

verification[
    "verification_status"
] = verification.apply(
    scene_status,
    axis=1
)

# ======================================================================================
# 7. MERGE EARTH ENGINE TASK STATUS
# ======================================================================================

verification = verification.merge(
    task_status,
    on=[
        "export_id",
        "record_id"
    ],
    how="left",
    validate="one_to_one"
)

# ======================================================================================
# 8. SAVE RESULTS
# ======================================================================================

VERIFY_PATH = (
    OUTDIR
    / "05_batch02_export_verification.csv"
)

STATUS_PATH = (
    OUTDIR
    / "06_batch02_task_status.csv"
)

verification.to_csv(
    VERIFY_PATH,
    index=False
)

task_status.to_csv(
    STATUS_PATH,
    index=False
)

# ======================================================================================
# 9. REPORT
# ======================================================================================

print("\n" + "=" * 110)
print("BATCH 02 PHYSICAL FILE VERIFICATION")
print("=" * 110)

print(
    verification[
        "verification_status"
    ]
    .value_counts(
        dropna=False
    )
    .to_string()
)

print("\nFILES FOUND")

print(
    f"Predictor GeoTIFFs: "
    f"{verification['predictor_copies'].eq(1).sum()}/20"
)

print(
    f"Thermal GeoTIFFs:   "
    f"{verification['thermal_copies'].eq(1).sum()}/20"
)

print("\nMean validity:")

if (
    "s2_valid_fraction"
    in verification.columns
):
    print(
        f"S2 mean valid fraction: "
        f"{verification['s2_valid_fraction'].mean():.3f}"
    )

if (
    "s1_valid_fraction"
    in verification.columns
):
    print(
        f"S1 mean valid fraction: "
        f"{verification['s1_valid_fraction'].mean():.3f}"
    )

if (
    "thermal_valid_fraction"
    in verification.columns
):
    print(
        f"Thermal mean valid fraction: "
        f"{verification['thermal_valid_fraction'].mean():.3f}"
    )

print("\nSaved:")
print(VERIFY_PATH)
print(STATUS_PATH)

print("\n" + "=" * 110)

n_pass = (
    verification[
        "verification_status"
    ]
    .eq("PASS")
    .sum()
)

if n_pass == 20:

    print(
        "PASS — Batch 02 is complete and valid."
    )

    print(
        "Batch 03 may now be submitted."
    )

else:

    print(
        f"STOP — only {n_pass}/20 scenes "
        f"passed verification."
    )

    print(
        "Do not submit Batch 03 until "
        "the REVIEW scenes are resolved."
    )

print("=" * 110)
