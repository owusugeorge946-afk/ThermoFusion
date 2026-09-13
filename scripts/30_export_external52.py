# =====================================================================
# ThermoFusion reproducibility code
# Source notebook cell(s): 65
# Audit status: KEEP
# Extracted without scientific-logic modification.
# Replace local Google Drive paths as described in README.md when needed.
# =====================================================================

# ======================================================================================
# THERMOFUSION — LARGE EXTERNAL TEMPORAL EVALUATION
# STAGE 04B: EXPORT 52 SPATIALLY VERIFIED EXTERNAL SCENES
# ======================================================================================

from google.colab import drive
drive.mount('/content/drive')

from pathlib import Path
import math
import ast
import pandas as pd
import numpy as np
import ee

# ======================================================================================
# 1. CONFIGURATION
# ======================================================================================

PROJECT_ID = "nana213"

ROOT = Path("/content/drive/MyDrive")

MANIFEST_PATH = (
    ROOT
    / "ThermoFusion_External_Evaluation"
    / "Stage02_final_new_external_test_manifest.csv"
)

PREFLIGHT_PATH = (
    ROOT
    / "ThermoFusion_External_Evaluation"
    / "Stage04_Export"
    / "01_external65_spatial_preflight.csv"
)

OUTDIR = (
    ROOT
    / "ThermoFusion_External_Evaluation"
    / "Stage04_Export"
)

OUTDIR.mkdir(parents=True, exist_ok=True)

DRIVE_FOLDER = "ThermoFusion_External52_Chips"

CHIP_SIZE_M = 2560
NODATA = -9999

# ----------------------------------------------------------------------
# CHANGE ONLY THIS BETWEEN RUNS
# ----------------------------------------------------------------------

BATCH_NUMBER = 3

BATCH_SIZE = 20

# Batch 1 = scenes 1–20
# Batch 2 = scenes 21–40
# Batch 3 = scenes 41–52

# ======================================================================================
# 2. ORIGINAL THERMOFUSION DATA CONTRACT
# ======================================================================================

CITY_ASSETS = {
    "Accra": "projects/nana213/assets/Accra",
    "Lagos": "projects/nana213/assets/lagos",
    "Abidjan": "projects/nana213/assets/Abijan",
    "Freetown": "projects/nana213/assets/Freetown",
}

LANDSAT_8 = "LANDSAT/LC08/C02/T1_L2"
LANDSAT_9 = "LANDSAT/LC09/C02/T1_L2"
ECOSTRESS = "NASA/ECOSTRESS/L2T_LSTE/V2"

SENTINEL_2 = "COPERNICUS/S2_SR_HARMONIZED"
SENTINEL_1 = "COPERNICUS/S1_GRD"

S2_WINDOW_DAYS = 10
S1_WINDOW_DAYS = 12

# ======================================================================================
# 3. LOAD ORIGINAL 65 + PREFLIGHT
# ======================================================================================

manifest = pd.read_csv(MANIFEST_PATH)
preflight = pd.read_csv(PREFLIGHT_PATH)

manifest["record_id"] = (
    manifest["record_id"]
    .astype(str)
    .str.strip()
)

preflight["record_id"] = (
    preflight["record_id"]
    .astype(str)
    .str.strip()
)

# ======================================================================================
# 4. CREATE FINAL 52-SCENE EXTERNAL MANIFEST
# ======================================================================================

passed_preflight = preflight[
    preflight["status"].eq("PASS")
].copy()

failed_preflight = preflight[
    ~preflight["status"].eq("PASS")
].copy()

if len(passed_preflight) != 52:
    raise RuntimeError(
        f"Expected 52 PASS scenes, found {len(passed_preflight)}."
    )

final52 = manifest.merge(
    passed_preflight[
        [
            "record_id",
            "centre_longitude",
            "centre_latitude",
            "thermal_crs",
            "thermal_transform",
            "thermal_nominal_scale_m",
            "s2_band_count",
            "s1_band_count",
            "status",
        ]
    ],
    on="record_id",
    how="inner",
    validate="one_to_one"
)

# Preserve original external IDs
final52 = final52.sort_values(
    "external_id"
).reset_index(drop=True)

# Add new sequential export identifier
final52["export_id"] = [
    f"TFXNEW{i:03d}"
    for i in range(1, len(final52) + 1)
]

final52["evaluation_role"] = (
    "new_external_frozen_model_test"
)

# ======================================================================================
# 5. SAVE FINAL 52 + EXCLUDED 13
# ======================================================================================

FINAL52_PATH = (
    OUTDIR
    / "02_final_external52_manifest.csv"
)

EXCLUDED13_PATH = (
    OUTDIR
    / "03_external13_spatial_exclusions.csv"
)

final52.to_csv(
    FINAL52_PATH,
    index=False
)

failed_preflight.to_csv(
    EXCLUDED13_PATH,
    index=False
)

print("=" * 110)
print("THERMOFUSION STAGE 04B — VERIFIED EXTERNAL EXPORT")
print("=" * 110)

print(f"Original selected scenes : {len(manifest)}")
print(f"Spatial preflight PASS   : {len(final52)}")
print(f"Spatial exclusions       : {len(failed_preflight)}")

print("\nFinal external manifest:")
print(FINAL52_PATH)

print("\nSpatial exclusion audit:")
print(EXCLUDED13_PATH)

# ======================================================================================
# 6. FIXED LEAKAGE / INTEGRITY CHECKS
# ======================================================================================

assert len(final52) == 52

assert final52["record_id"].is_unique

assert final52["external_id"].is_unique

assert final52["centre_longitude"].notna().all()

assert final52["centre_latitude"].notna().all()

assert final52["s2_band_count"].eq(10).all()

assert final52["s1_band_count"].eq(4).all()

print("\nFinal-52 integrity checks: PASS")

# ======================================================================================
# 7. SELECT CURRENT BATCH
# ======================================================================================

start = (BATCH_NUMBER - 1) * BATCH_SIZE
stop = min(
    start + BATCH_SIZE,
    len(final52)
)

if start >= len(final52):
    raise ValueError(
        f"BATCH_NUMBER={BATCH_NUMBER} exceeds available scenes."
    )

batch = final52.iloc[
    start:stop
].copy()

print("\n" + "=" * 110)
print(f"BATCH {BATCH_NUMBER}")
print("=" * 110)

print(
    f"Rows {start + 1}–{stop} "
    f"of {len(final52)}"
)

print(
    f"Scenes in this batch: {len(batch)}"
)

print(
    f"Expected EE tasks: {len(batch) * 2}"
)

print("\nBatch city × sensor composition:")

print(
    batch.groupby(
        ["city", "thermal_sensor"]
    )
    .size()
    .reset_index(name="scenes")
    .to_string(index=False)
)

# ======================================================================================
# 8. CONNECT TO EARTH ENGINE
# ======================================================================================

ee.Authenticate()
ee.Initialize(project=PROJECT_ID)

print(
    f"\nConnected to Earth Engine project: "
    f"{PROJECT_ID}"
)

# ======================================================================================
# 9. HELPERS
# ======================================================================================

def source_index(scene_id):

    index = str(scene_id).split("/")[-1]

    while (
        index.startswith("1_")
        or index.startswith("2_")
    ):
        index = index[2:]

    return index


def normalized_ee_date(value):

    timestamp = pd.to_datetime(
        value,
        utc=True,
        errors="raise"
    )

    return ee.Date(
        timestamp.strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
    )


# ======================================================================================
# 10. THERMAL TARGET
# ======================================================================================

def thermal_image_and_mask(row):

    index = source_index(
        row.thermal_scene_id
    )

    if row.thermal_sensor == "Landsat":

        collection = (
            LANDSAT_9
            if index.startswith("LC09")
            else LANDSAT_8
        )

        image = ee.Image(
            f"{collection}/{index}"
        )

        qa = image.select(
            "QA_PIXEL"
        )

        clear = (
            qa.bitwiseAnd(1 << 0).eq(0)
            .And(
                qa.bitwiseAnd(
                    1 << 1
                ).eq(0)
            )
            .And(
                qa.bitwiseAnd(
                    1 << 2
                ).eq(0)
            )
            .And(
                qa.bitwiseAnd(
                    1 << 3
                ).eq(0)
            )
            .And(
                qa.bitwiseAnd(
                    1 << 4
                ).eq(0)
            )
        )

        unsaturated = (
            image.select(
                "QA_RADSAT"
            ).eq(0)
        )

        lst = (
            image.select("ST_B10")
            .multiply(0.00341802)
            .add(149.0)
            .subtract(273.15)
            .rename(
                "thermal_lst_c"
            )
        )

        valid = (
            clear
            .And(unsaturated)
            .And(lst.gt(5))
            .And(lst.lt(70))
        )

        return (
            image,
            lst.updateMask(valid),
            valid.rename(
                "thermal_valid"
            ),
        )

    # ----------------------------------------------------------
    # ECOSTRESS
    # ----------------------------------------------------------

    image = ee.Image(
        f"{ECOSTRESS}/{index}"
    )

    lst = (
        image.select("LST")
        .subtract(273.15)
        .rename(
            "thermal_lst_c"
        )
    )

    valid = (
        image.select("QC")
        .bitwiseAnd(31)
        .eq(0)
        .And(
            image.select("cloud")
            .bitwiseAnd(1)
            .eq(0)
        )
        .And(
            image.select("water")
            .bitwiseAnd(1)
            .eq(0)
        )
        .And(
            lst.gt(-23.15)
        )
        .And(
            lst.lt(76.85)
        )
    )

    return (
        image,
        lst.updateMask(valid),
        valid.rename(
            "thermal_valid"
        ),
    )


# ======================================================================================
# 11. SENTINEL-2
# ======================================================================================

def sentinel2_predictors(
    acquisition_time,
    geometry
):

    target_time = normalized_ee_date(
        acquisition_time
    )

    def prepare(image):

        image = ee.Image(image)

        offset = (
            ee.Number(
                image.get(
                    "system:time_start"
                )
            )
            .subtract(
                target_time.millis()
            )
            .abs()
        )

        scl = image.select(
            "SCL"
        )

        valid = (
            scl.neq(0)
            .And(scl.neq(1))
            .And(scl.neq(3))
            .And(scl.neq(7))
            .And(scl.neq(8))
            .And(scl.neq(9))
            .And(scl.neq(10))
            .And(scl.neq(11))
        )

        refl = (
            image.select(
                [
                    "B2",
                    "B3",
                    "B4",
                    "B8",
                    "B11",
                    "B12",
                ]
            )
            .multiply(0.0001)
            .rename(
                [
                    "s2_b2",
                    "s2_b3",
                    "s2_b4",
                    "s2_b8",
                    "s2_b11",
                    "s2_b12",
                ]
            )
            .updateMask(valid)
        )

        ndvi = (
            refl.normalizedDifference(
                ["s2_b8", "s2_b4"]
            )
            .rename("ndvi")
        )

        ndbi = (
            refl.normalizedDifference(
                ["s2_b11", "s2_b8"]
            )
            .rename("ndbi")
        )

        ndmi = (
            refl.normalizedDifference(
                ["s2_b8", "s2_b11"]
            )
            .rename("ndmi")
        )

        return (
            ee.Image.cat(
                refl,
                ndvi,
                ndbi,
                ndmi,
                valid.selfMask()
                .rename(
                    "s2_valid"
                ),
            )
            .set(
                "time_offset_ms",
                offset
            )
        )

    return (
        ee.ImageCollection(
            SENTINEL_2
        )
        .filterBounds(
            geometry
        )
        .filterDate(
            target_time.advance(
                -S2_WINDOW_DAYS,
                "day"
            ),
            target_time.advance(
                S2_WINDOW_DAYS + 1,
                "day"
            ),
        )
        .filter(
            ee.Filter.lt(
                "CLOUDY_PIXEL_PERCENTAGE",
                90
            )
        )
        .map(prepare)
        .sort(
            "time_offset_ms",
            False
        )
        .mosaic()
    )


# ======================================================================================
# 12. SENTINEL-1
# ======================================================================================

def sentinel1_predictors(
    acquisition_time,
    geometry
):

    target_time = normalized_ee_date(
        acquisition_time
    )

    def prepare(image):

        image = ee.Image(image)

        offset = (
            ee.Number(
                image.get(
                    "system:time_start"
                )
            )
            .subtract(
                target_time.millis()
            )
            .abs()
        )

        vv = (
            image.select("VV")
            .rename("s1_vv_db")
        )

        vh = (
            image.select("VH")
            .rename("s1_vh_db")
        )

        difference = (
            vv.subtract(vh)
            .rename(
                "s1_vv_minus_vh_db"
            )
        )

        valid = (
            vv.mask()
            .And(vh.mask())
        )

        return (
            ee.Image.cat(
                vv,
                vh,
                difference,
                valid.selfMask()
                .rename(
                    "s1_valid"
                ),
            )
            .set(
                "time_offset_ms",
                offset
            )
        )

    return (
        ee.ImageCollection(
            SENTINEL_1
        )
        .filterBounds(
            geometry
        )
        .filterDate(
            target_time.advance(
                -S1_WINDOW_DAYS,
                "day"
            ),
            target_time.advance(
                S1_WINDOW_DAYS + 1,
                "day"
            ),
        )
        .filter(
            ee.Filter.eq(
                "instrumentMode",
                "IW"
            )
        )
        .filter(
            ee.Filter.listContains(
                "transmitterReceiverPolarisation",
                "VV"
            )
        )
        .filter(
            ee.Filter.listContains(
                "transmitterReceiverPolarisation",
                "VH"
            )
        )
        .filter(
            ee.Filter.eq(
                "resolution_meters",
                10
            )
        )
        .map(prepare)
        .sort(
            "time_offset_ms",
            False
        )
        .mosaic()
    )


# ======================================================================================
# 13. TERRAIN
# ======================================================================================

dem = (
    ee.ImageCollection(
        "COPERNICUS/DEM/GLO30"
    )
    .select("DEM")
    .mosaic()
    .rename(
        "elevation_m"
    )
)

slope = (
    ee.Terrain.slope(dem)
    .rename(
        "slope_deg"
    )
)


# ======================================================================================
# 14. SUBMIT CURRENT BATCH
# ======================================================================================

task_rows = []

for number, row in enumerate(
    batch.itertuples(index=False),
    start=1
):

    print(
        f"\nPreparing "
        f"{number:02d}/{len(batch)}: "
        f"{row.record_id}"
    )

    # ----------------------------------------------------------
    # Fixed centre from Stage 04A
    # ----------------------------------------------------------

    centre = ee.Geometry.Point(
        [
            float(
                row.centre_longitude
            ),
            float(
                row.centre_latitude
            ),
        ]
    )

    region = (
        centre
        .buffer(
            CHIP_SIZE_M / 2
        )
        .bounds()
    )

    # ----------------------------------------------------------
    # UTM predictor CRS
    # ----------------------------------------------------------

    longitude = float(
        row.centre_longitude
    )

    latitude = float(
        row.centre_latitude
    )

    zone = int(
        math.floor(
            (longitude + 180) / 6
        ) + 1
    )

    epsg = (
        32600 + zone
        if latitude >= 0
        else 32700 + zone
    )

    predictor_crs = (
        f"EPSG:{epsg}"
    )

    # ----------------------------------------------------------
    # Reconstruct exact sources
    # ----------------------------------------------------------

    (
        thermal_source,
        thermal_lst,
        thermal_valid
    ) = thermal_image_and_mask(
        row
    )

    city_geometry = (
        ee.FeatureCollection(
            CITY_ASSETS[row.city]
        )
        .geometry()
    )

    s2 = sentinel2_predictors(
        row.acquisition_date,
        city_geometry
    )

    s1 = sentinel1_predictors(
        row.acquisition_date,
        city_geometry
    )

    predictors = (
        ee.Image.cat(
            s2,
            s1,
            dem,
            slope
        )
        .toFloat()
        .unmask(
            NODATA
        )
    )

    thermal_export = (
        ee.Image.cat(
            thermal_lst
            .unmask(
                NODATA
            )
            .toFloat(),

            thermal_valid
            .unmask(0)
            .toFloat(),
        )
    )

    # ----------------------------------------------------------
    # Native thermal projection
    # Re-read directly from thermal source for safety
    # ----------------------------------------------------------

    thermal_band = (
        "ST_B10"
        if row.thermal_sensor
        == "Landsat"
        else "LST"
    )

    thermal_projection = (
        thermal_source
        .select(
            thermal_band
        )
        .projection()
    )

    projection_info = (
        thermal_projection
        .getInfo()
    )

    native_scale = (
        thermal_projection
        .nominalScale()
        .getInfo()
    )

    # ----------------------------------------------------------
    # Prefix
    # ----------------------------------------------------------

    prefix = (
        f"{row.export_id}_"
        f"{row.city.lower()}_"
        f"{row.thermal_sensor.lower()}"
    )

    # ----------------------------------------------------------
    # Predictor export
    # ----------------------------------------------------------

    predictor_task = (
        ee.batch.Export.image.toDrive(
            image=predictors,

            description=(
                f"{prefix}_predictors"
            ),

            folder=DRIVE_FOLDER,

            fileNamePrefix=(
                f"{prefix}_predictors_10m"
            ),

            region=region,

            crs=predictor_crs,

            scale=10,

            maxPixels=1e9,

            fileFormat="GeoTIFF",

            formatOptions={
                "cloudOptimized": True,
                "noData": NODATA,
            },
        )
    )

    # ----------------------------------------------------------
    # Thermal export
    # ----------------------------------------------------------

    thermal_task = (
        ee.batch.Export.image.toDrive(
            image=thermal_export,

            description=(
                f"{prefix}_thermal"
            ),

            folder=DRIVE_FOLDER,

            fileNamePrefix=(
                f"{prefix}_thermal_native"
            ),

            region=region,

            crs=projection_info[
                "crs"
            ],

            crsTransform=projection_info[
                "transform"
            ],

            maxPixels=1e9,

            fileFormat="GeoTIFF",

            formatOptions={
                "cloudOptimized": True,
                "noData": NODATA,
            },
        )
    )

    predictor_task.start()
    thermal_task.start()

    task_rows.append(
        {
            "batch_number":
                BATCH_NUMBER,

            "batch_position":
                number,

            "export_id":
                row.export_id,

            "external_id":
                row.external_id,

            "record_id":
                row.record_id,

            "city":
                row.city,

            "thermal_sensor":
                row.thermal_sensor,

            "acquisition_date":
                row.acquisition_date,

            "centre_longitude":
                longitude,

            "centre_latitude":
                latitude,

            "predictor_crs":
                predictor_crs,

            "predictor_scale_m":
                10,

            "thermal_crs":
                projection_info[
                    "crs"
                ],

            "thermal_transform":
                str(
                    projection_info[
                        "transform"
                    ]
                ),

            "thermal_nominal_scale_m":
                native_scale,

            "predictor_task_id":
                predictor_task.id,

            "thermal_task_id":
                thermal_task.id,

            "prefix":
                prefix,
        }
    )

    print(
        f"Submitted "
        f"{number:02d}/{len(batch)}: "
        f"{prefix}"
    )


# ======================================================================================
# 15. SAVE BATCH TASK REGISTER
# ======================================================================================

task_df = pd.DataFrame(
    task_rows
)

task_path = (
    OUTDIR
    / (
        f"04_batch{BATCH_NUMBER:02d}_"
        f"export_task_register.csv"
    )
)

task_df.to_csv(
    task_path,
    index=False
)

# ======================================================================================
# 16. FINAL REPORT
# ======================================================================================

print("\n" + "=" * 110)
print(
    f"THERMOFUSION EXTERNAL52 — "
    f"BATCH {BATCH_NUMBER} SUBMITTED"
)
print("=" * 110)

print(
    f"Scenes submitted       : "
    f"{len(task_df)}"
)

print(
    f"Predictor tasks        : "
    f"{len(task_df)}"
)

print(
    f"Thermal tasks          : "
    f"{len(task_df)}"
)

print(
    f"Total EE tasks         : "
    f"{len(task_df) * 2}"
)

print(
    f"Drive folder           : "
    f"{DRIVE_FOLDER}"
)

print(
    f"Task register          : "
    f"{task_path}"
)

print("=" * 110)
