# =====================================================================
# ThermoFusion reproducibility code
# Source notebook cell(s): 64
# Audit status: KEEP
# Extracted without scientific-logic modification.
# Replace local Google Drive paths as described in README.md when needed.
# =====================================================================

# ======================================================================================
# THERMOFUSION — LARGE EXTERNAL TEMPORAL EVALUATION
# STAGE 04A: EXACT 65-SCENE SPATIAL PREFLIGHT
# ======================================================================================

from google.colab import drive
drive.mount('/content/drive')

from pathlib import Path
import math
import pandas as pd
import numpy as np
import ee

# ======================================================================================
# 1. CONFIGURATION — PRESERVE ORIGINAL THERMOFUSION CONTRACT
# ======================================================================================

PROJECT_ID = "nana213"

MANIFEST_PATH = Path(
    "/content/drive/MyDrive/ThermoFusion_External_Evaluation/"
    "Stage02_final_new_external_test_manifest.csv"
)

OUTDIR = Path(
    "/content/drive/MyDrive/ThermoFusion_External_Evaluation/Stage04_Export"
)
OUTDIR.mkdir(parents=True, exist_ok=True)

CHIP_SIZE_M = 2560
EROSION_RADIUS_M = 640
NODATA = -9999

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
# 2. LOAD FIXED 65-SCENE MANIFEST
# ======================================================================================

if not MANIFEST_PATH.exists():
    raise FileNotFoundError(MANIFEST_PATH)

df = pd.read_csv(MANIFEST_PATH)

if len(df) != 65:
    raise RuntimeError(
        f"Expected 65 external scenes, found {len(df)}."
    )

df["acquisition_date"] = pd.to_datetime(
    df["acquisition_date"],
    errors="raise",
    utc=True
)

if df["record_id"].duplicated().any():
    raise RuntimeError("Duplicate record_id detected.")

if df["external_id"].duplicated().any():
    raise RuntimeError("Duplicate external_id detected.")

print("=" * 110)
print("THERMOFUSION STAGE 04A — 65-SCENE PREFLIGHT")
print("=" * 110)
print(f"Fixed external scenes: {len(df)}")
print(f"Unique record IDs: {df['record_id'].nunique()}")
print(f"Minimum valid coverage: {df['valid_fraction_pct'].min():.2f}%")

# ======================================================================================
# 3. EARTH ENGINE
# ======================================================================================

ee.Authenticate()
ee.Initialize(project=PROJECT_ID)

print(f"Connected to Earth Engine project: {PROJECT_ID}")

# ======================================================================================
# 4. HELPERS
# ======================================================================================

def source_index(scene_id):
    index = str(scene_id).split("/")[-1]

    while index.startswith("1_") or index.startswith("2_"):
        index = index[2:]

    return index


def normalized_ee_date(value):
    timestamp = pd.to_datetime(
        value,
        utc=True,
        errors="raise"
    )

    return ee.Date(
        timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")
    )


# ======================================================================================
# 5. THERMAL TARGET
# ======================================================================================

def thermal_image_and_mask(row):

    index = source_index(row.thermal_scene_id)

    if row.thermal_sensor == "Landsat":

        collection = (
            LANDSAT_9
            if index.startswith("LC09")
            else LANDSAT_8
        )

        image = ee.Image(
            f"{collection}/{index}"
        )

        qa = image.select("QA_PIXEL")

        clear = (
            qa.bitwiseAnd(1 << 0).eq(0)
            .And(qa.bitwiseAnd(1 << 1).eq(0))
            .And(qa.bitwiseAnd(1 << 2).eq(0))
            .And(qa.bitwiseAnd(1 << 3).eq(0))
            .And(qa.bitwiseAnd(1 << 4).eq(0))
        )

        unsaturated = (
            image.select("QA_RADSAT").eq(0)
        )

        lst = (
            image.select("ST_B10")
            .multiply(0.00341802)
            .add(149.0)
            .subtract(273.15)
            .rename("thermal_lst_c")
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
            valid.rename("thermal_valid")
        )

    # ECOSTRESS
    image = ee.Image(
        f"{ECOSTRESS}/{index}"
    )

    lst = (
        image.select("LST")
        .subtract(273.15)
        .rename("thermal_lst_c")
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
        .And(lst.gt(-23.15))
        .And(lst.lt(76.85))
    )

    return (
        image,
        lst.updateMask(valid),
        valid.rename("thermal_valid")
    )


# ======================================================================================
# 6. SENTINEL-2 — EXACT ORIGINAL PREDICTOR CONSTRUCTION
# ======================================================================================

def sentinel2_predictors(acquisition_time, geometry):

    target_time = normalized_ee_date(
        acquisition_time
    )

    def prepare(image):

        image = ee.Image(image)

        offset = (
            ee.Number(
                image.get("system:time_start")
            )
            .subtract(target_time.millis())
            .abs()
        )

        scl = image.select("SCL")

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

        reflectance = (
            image.select(
                ["B2", "B3", "B4", "B8", "B11", "B12"]
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
            reflectance
            .normalizedDifference(
                ["s2_b8", "s2_b4"]
            )
            .rename("ndvi")
        )

        ndbi = (
            reflectance
            .normalizedDifference(
                ["s2_b11", "s2_b8"]
            )
            .rename("ndbi")
        )

        ndmi = (
            reflectance
            .normalizedDifference(
                ["s2_b8", "s2_b11"]
            )
            .rename("ndmi")
        )

        prepared = ee.Image.cat(
            reflectance,
            ndvi,
            ndbi,
            ndmi,
            valid.selfMask().rename("s2_valid"),
        )

        return prepared.set(
            "time_offset_ms",
            offset
        )

    return (
        ee.ImageCollection(SENTINEL_2)
        .filterBounds(geometry)
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
        .sort("time_offset_ms", False)
        .mosaic()
    )


# ======================================================================================
# 7. SENTINEL-1 — EXACT ORIGINAL PREDICTOR CONSTRUCTION
# ======================================================================================

def sentinel1_predictors(acquisition_time, geometry):

    target_time = normalized_ee_date(
        acquisition_time
    )

    def prepare(image):

        image = ee.Image(image)

        offset = (
            ee.Number(
                image.get("system:time_start")
            )
            .subtract(target_time.millis())
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
            .rename("s1_vv_minus_vh_db")
        )

        valid = (
            vv.mask()
            .And(vh.mask())
        )

        prepared = ee.Image.cat(
            vv,
            vh,
            difference,
            valid.selfMask().rename("s1_valid"),
        )

        return prepared.set(
            "time_offset_ms",
            offset
        )

    return (
        ee.ImageCollection(SENTINEL_1)
        .filterBounds(geometry)
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
# 8. CHIP CENTRE
# ======================================================================================

def select_chip_centre(
    valid_mask,
    city_geometry,
    native_scale
):

    eroded = (
        valid_mask
        .selfMask()
        .focal_min(
            radius=EROSION_RADIUS_M,
            units="meters"
        )
    )

    sample = eroded.sample(
        region=city_geometry,
        scale=native_scale,
        numPixels=1,
        seed=946,
        geometries=True,
        tileScale=4,
    )

    count = sample.size().getInfo()

    if count == 0:

        sample = (
            valid_mask
            .selfMask()
            .sample(
                region=city_geometry,
                scale=native_scale,
                numPixels=1,
                seed=946,
                geometries=True,
                tileScale=4,
            )
        )

        count = sample.size().getInfo()

    if count == 0:
        return None

    return (
        ee.Feature(
            sample.first()
        )
        .geometry()
    )


# ======================================================================================
# 9. PREFLIGHT ALL 65 FIXED SCENES
# ======================================================================================

preflight_rows = []

for number, row in enumerate(
    df.itertuples(index=False),
    start=1
):

    city_geometry = (
        ee.FeatureCollection(
            CITY_ASSETS[row.city]
        )
        .geometry()
    )

    try:

        thermal_source, thermal_lst, thermal_valid = (
            thermal_image_and_mask(row)
        )

        thermal_band = (
            "ST_B10"
            if row.thermal_sensor == "Landsat"
            else "LST"
        )

        thermal_projection = (
            thermal_source
            .select(thermal_band)
            .projection()
        )

        projection_info = (
            thermal_projection.getInfo()
        )

        native_scale = (
            thermal_projection
            .nominalScale()
            .getInfo()
        )

        s2 = sentinel2_predictors(
            row.acquisition_date,
            city_geometry
        )

        s1 = sentinel1_predictors(
            row.acquisition_date,
            city_geometry
        )

        # Trigger band existence now, not during export
        s2_bands = s2.bandNames().getInfo()
        s1_bands = s1.bandNames().getInfo()

        s2_ok = len(s2_bands) == 10
        s1_ok = len(s1_bands) == 4

        if not s2_ok or not s1_ok:

            centre = None

        else:

            joint_valid = (
                thermal_valid
                .unmask(0)
                .And(
                    s2.select("s2_valid")
                    .unmask(0)
                    .gt(0.5)
                )
                .And(
                    s1.select("s1_valid")
                    .unmask(0)
                    .gt(0.5)
                )
                .rename("joint_valid")
            )

            centre = select_chip_centre(
                joint_valid,
                city_geometry,
                native_scale
            )

        passed = (
            s2_ok
            and s1_ok
            and centre is not None
        )

        if centre is not None:

            coordinates = (
                centre.coordinates()
                .getInfo()
            )

            lon = coordinates[0]
            lat = coordinates[1]

        else:

            lon = np.nan
            lat = np.nan

        preflight_rows.append(
            {
                "external_id": row.external_id,
                "record_id": row.record_id,
                "city": row.city,
                "thermal_sensor": row.thermal_sensor,
                "acquisition_date": row.acquisition_date,
                "thermal_scene_id": row.thermal_scene_id,
                "valid_fraction_pct": row.valid_fraction_pct,

                "s2_band_count": len(s2_bands),
                "s1_band_count": len(s1_bands),

                "joint_valid_centre_found":
                    centre is not None,

                "centre_longitude": lon,
                "centre_latitude": lat,

                "thermal_crs":
                    projection_info["crs"],

                "thermal_transform":
                    str(projection_info["transform"]),

                "thermal_nominal_scale_m":
                    native_scale,

                "status":
                    "PASS" if passed else "FAIL",
            }
        )

        print(
            f"Preflight {number:02d}/65: "
            f"{row.record_id} — "
            f"{'PASS' if passed else 'FAIL'}"
        )

    except Exception as e:

        preflight_rows.append(
            {
                "external_id": row.external_id,
                "record_id": row.record_id,
                "city": row.city,
                "thermal_sensor": row.thermal_sensor,
                "acquisition_date": row.acquisition_date,
                "thermal_scene_id": row.thermal_scene_id,
                "valid_fraction_pct": row.valid_fraction_pct,

                "status": "ERROR",
                "error": str(e),
            }
        )

        print(
            f"Preflight {number:02d}/65: "
            f"{row.record_id} — ERROR"
        )

        print("   ", str(e))


# ======================================================================================
# 10. SAVE PREFLIGHT
# ======================================================================================

preflight = pd.DataFrame(
    preflight_rows
)

preflight_path = (
    OUTDIR
    / "01_external65_spatial_preflight.csv"
)

preflight.to_csv(
    preflight_path,
    index=False
)

# ======================================================================================
# 11. REPORT
# ======================================================================================

print("\n" + "=" * 110)
print("STAGE 04A PREFLIGHT SUMMARY")
print("=" * 110)

print(
    preflight["status"]
    .value_counts(
        dropna=False
    )
    .to_string()
)

passed = (
    preflight["status"]
    .eq("PASS")
    .sum()
)

failed = len(preflight) - passed

print(f"\nPASS : {passed}/65")
print(f"FAIL : {failed}/65")

print(f"\nSaved:\n{preflight_path}")

print("\n" + "=" * 110)

if passed == 65:

    print(
        "PASS — all 65 fixed external scenes are spatially exportable."
    )

    print(
        "Proceed to Stage 04B batch export."
    )

else:

    print(
        "STOP — do not export yet. "
        "Failed scenes must be reviewed first."
    )

print("=" * 110)
