# =====================================================================
# ThermoFusion reproducibility code
# Source notebook cell(s): 52
# Audit status: ARCHIVE
# Extracted without scientific-logic modification.
# Replace local Google Drive paths as described in README.md when needed.
# =====================================================================

# =============================================================================
# THERMOFUSION STAGE 23 — EXPORT 38 EXPANDED TEMPORAL-TEST SCENES
# Exact Stage 10 processing contract; no retraining and no scene replacement.
# =============================================================================

!pip -q install earthengine-api

from google.colab import drive
from pathlib import Path
import math
import pandas as pd
import ee

drive.mount("/content/drive")

PROJECT_ID = "nana213"

SELECTION_PATH = Path(
    "/content/drive/MyDrive/"
    "ThermoFusion_Stage22B_CorrectedExpandedTest/"
    "01_corrected_expanded_test_manifest.csv"
)

OUT_DIR = Path(
    "/content/drive/MyDrive/"
    "ThermoFusion_Stage23_ExpandedTemporalTest"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

DRIVE_FOLDER = "ThermoFusion_Stage23_ExpandedTemporalTest_Chips"

CHIP_SIZE_M = 2560
EROSION_RADIUS_M = 640
NODATA = -9999
EXPECTED_SCENES = 38
S2_WINDOW_DAYS = 10
S1_WINDOW_DAYS = 12

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

assert SELECTION_PATH.exists(), f"Missing manifest: {SELECTION_PATH}"

data = pd.read_csv(SELECTION_PATH)
data["thermal_datetime_utc"] = pd.to_datetime(
    data["acquisition_date"],
    errors="raise",
    utc=True
)

assert len(data) == EXPECTED_SCENES, "Expected exactly 38 selected scenes."
assert data["record_id"].nunique() == EXPECTED_SCENES, "Duplicate record IDs."
assert not data.duplicated(
    ["city", "thermal_sensor", "thermal_scene_id"]
).any(), "Duplicate thermal scenes."
assert data["valid_fraction_pct"].min() >= 50, "Coverage threshold failed."

ee.Authenticate()
ee.Initialize(project=PROJECT_ID)

print(f"Connected to Earth Engine project: {PROJECT_ID}")
print(f"Fixed scenes to export: {len(data)}")


def source_index(scene_id):
    index = str(scene_id).split("/")[-1]

    while index.startswith("1_") or index.startswith("2_"):
        index = index[2:]

    return index


def normalized_ee_date(value):
    timestamp = pd.to_datetime(value, utc=True, errors="raise")
    return ee.Date(timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"))


def thermal_image_and_mask(row):
    index = source_index(row.thermal_scene_id)

    if row.thermal_sensor == "Landsat":
        collection = LANDSAT_9 if index.startswith("LC09") else LANDSAT_8
        image = ee.Image(f"{collection}/{index}")

        qa = image.select("QA_PIXEL")

        clear = (
            qa.bitwiseAnd(1 << 0).eq(0)
            .And(qa.bitwiseAnd(1 << 1).eq(0))
            .And(qa.bitwiseAnd(1 << 2).eq(0))
            .And(qa.bitwiseAnd(1 << 3).eq(0))
            .And(qa.bitwiseAnd(1 << 4).eq(0))
        )

        unsaturated = image.select("QA_RADSAT").eq(0)

        lst = (
            image.select("ST_B10")
            .multiply(0.00341802)
            .add(149.0)
            .subtract(273.15)
            .rename("thermal_lst_c")
        )

        valid = clear.And(unsaturated).And(lst.gt(5)).And(lst.lt(70))

        return image, lst.updateMask(valid), valid.rename("thermal_valid")

    image = ee.Image(f"{ECOSTRESS}/{index}")

    lst = image.select("LST").subtract(273.15).rename("thermal_lst_c")

    valid = (
        image.select("QC").bitwiseAnd(31).eq(0)
        .And(image.select("cloud").bitwiseAnd(1).eq(0))
        .And(image.select("water").bitwiseAnd(1).eq(0))
        .And(lst.gt(-23.15))
        .And(lst.lt(76.85))
    )

    return image, lst.updateMask(valid), valid.rename("thermal_valid")


def sentinel2_predictors(acquisition_time, geometry):
    target_time = normalized_ee_date(acquisition_time)

    def prepare(image):
        image = ee.Image(image)

        offset = ee.Number(
            image.get("system:time_start")
        ).subtract(target_time.millis()).abs()

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
            image.select(["B2", "B3", "B4", "B8", "B11", "B12"])
            .multiply(0.0001)
            .rename(
                [
                    "s2_b2", "s2_b3", "s2_b4",
                    "s2_b8", "s2_b11", "s2_b12"
                ]
            )
            .updateMask(valid)
        )

        ndvi = reflectance.normalizedDifference(
            ["s2_b8", "s2_b4"]
        ).rename("ndvi")

        ndbi = reflectance.normalizedDifference(
            ["s2_b11", "s2_b8"]
        ).rename("ndbi")

        ndmi = reflectance.normalizedDifference(
            ["s2_b8", "s2_b11"]
        ).rename("ndmi")

        return ee.Image.cat(
            reflectance,
            ndvi,
            ndbi,
            ndmi,
            valid.selfMask().rename("s2_valid")
        ).set("time_offset_ms", offset)

    return (
        ee.ImageCollection(SENTINEL_2)
        .filterBounds(geometry)
        .filterDate(
            target_time.advance(-S2_WINDOW_DAYS, "day"),
            target_time.advance(S2_WINDOW_DAYS + 1, "day")
        )
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 90))
        .map(prepare)
        .sort("time_offset_ms", False)
        .mosaic()
    )


def sentinel1_predictors(acquisition_time, geometry):
    target_time = normalized_ee_date(acquisition_time)

    def prepare(image):
        image = ee.Image(image)

        offset = ee.Number(
            image.get("system:time_start")
        ).subtract(target_time.millis()).abs()

        vv = image.select("VV").rename("s1_vv_db")
        vh = image.select("VH").rename("s1_vh_db")

        difference = vv.subtract(vh).rename("s1_vv_minus_vh_db")
        valid = vv.mask().And(vh.mask())

        return ee.Image.cat(
            vv,
            vh,
            difference,
            valid.selfMask().rename("s1_valid")
        ).set("time_offset_ms", offset)

    return (
        ee.ImageCollection(SENTINEL_1)
        .filterBounds(geometry)
        .filterDate(
            target_time.advance(-S1_WINDOW_DAYS, "day"),
            target_time.advance(S1_WINDOW_DAYS + 1, "day")
        )
        .filter(ee.Filter.eq("instrumentMode", "IW"))
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
        .filter(ee.Filter.eq("resolution_meters", 10))
        .map(prepare)
        .sort("time_offset_ms", False)
        .mosaic()
    )


def utm_crs(point):
    longitude, latitude = point.coordinates().getInfo()
    zone = int(math.floor((longitude + 180) / 6) + 1)

    epsg = 32600 + zone if latitude >= 0 else 32700 + zone

    return f"EPSG:{epsg}", longitude, latitude


def select_chip_centre(valid_mask, city_geometry, native_scale):
    eroded = valid_mask.selfMask().focal_min(
        radius=EROSION_RADIUS_M,
        units="meters"
    )

    sample = eroded.sample(
        region=city_geometry,
        scale=native_scale,
        numPixels=1,
        seed=946,
        geometries=True,
        tileScale=4
    )

    count = sample.size().getInfo()

    if count == 0:
        sample = valid_mask.selfMask().sample(
            region=city_geometry,
            scale=native_scale,
            numPixels=1,
            seed=946,
            geometries=True,
            tileScale=4
        )

        count = sample.size().getInfo()

    if count == 0:
        return None

    return ee.Feature(sample.first()).geometry()


# -----------------------------------------------------------------------------
# Phase 1: spatial preflight of the fixed 38 selected scenes
# -----------------------------------------------------------------------------

preflight_rows = []
resolved_bundles = []

for rank, row in enumerate(data.itertuples(index=False), start=1):
    city_geometry = ee.FeatureCollection(
        CITY_ASSETS[row.city]
    ).geometry()

    thermal_source, thermal_lst, thermal_valid = thermal_image_and_mask(row)

    thermal_projection = thermal_source.select(
        "ST_B10" if row.thermal_sensor == "Landsat" else "LST"
    ).projection()

    projection_info = thermal_projection.getInfo()
    native_scale = thermal_projection.nominalScale().getInfo()

    s2 = sentinel2_predictors(
        row.thermal_datetime_utc,
        city_geometry
    )

    s1 = sentinel1_predictors(
        row.thermal_datetime_utc,
        city_geometry
    )

    joint_valid = (
        thermal_valid.unmask(0)
        .And(s2.select("s2_valid").unmask(0).gt(0.5))
        .And(s1.select("s1_valid").unmask(0).gt(0.5))
        .rename("joint_valid")
    )

    centre = select_chip_centre(
        joint_valid,
        city_geometry,
        native_scale
    )

    passed = centre is not None

    preflight_rows.append({
        "record_id": row.record_id,
        "city": row.city,
        "thermal_sensor": row.thermal_sensor,
        "acquisition_date": row.acquisition_date,
        "valid_fraction_pct": row.valid_fraction_pct,
        "joint_valid_centre_found": passed
    })

    print(
        f"Preflight {rank:02d}/{EXPECTED_SCENES}: "
        f"{row.record_id} — {'PASS' if passed else 'FAIL'}"
    )

    if passed:
        export_id = f"TFE{len(resolved_bundles) + 1:04d}"

        resolved_bundles.append(
            (
                export_id,
                row,
                thermal_lst,
                thermal_valid,
                projection_info,
                native_scale,
                s2,
                s1,
                centre
            )
        )

preflight_df = pd.DataFrame(preflight_rows)

preflight_path = OUT_DIR / "01_spatial_preflight.csv"
preflight_df.to_csv(preflight_path, index=False)

if len(resolved_bundles) != EXPECTED_SCENES:
    failed = preflight_df.loc[
        ~preflight_df["joint_valid_centre_found"]
    ]

    display(failed)

    raise RuntimeError(
        f"Only {len(resolved_bundles)} of {EXPECTED_SCENES} scenes "
        "passed spatial preflight. No Earth Engine exports were submitted."
    )

print(
    f"\nSpatial preflight passed: "
    f"{len(resolved_bundles)}/{EXPECTED_SCENES}"
)


# -----------------------------------------------------------------------------
# Phase 2: export predictor and thermal bundles
# -----------------------------------------------------------------------------

dem = (
    ee.ImageCollection("COPERNICUS/DEM/GLO30")
    .select("DEM")
    .mosaic()
    .rename("elevation_m")
)

slope = ee.Terrain.slope(dem).rename("slope_deg")

task_rows = []

for export_number, bundle in enumerate(resolved_bundles, start=1):
    (
        export_id,
        row,
        thermal_lst,
        thermal_valid,
        projection_info,
        native_scale,
        s2,
        s1,
        centre
    ) = bundle

    predictor_crs, longitude, latitude = utm_crs(centre)
    region = centre.buffer(CHIP_SIZE_M / 2).bounds()

    predictors = (
        ee.Image.cat(s2, s1, dem, slope)
        .toFloat()
        .unmask(NODATA)
    )

    thermal_export = ee.Image.cat(
        thermal_lst.unmask(NODATA).toFloat(),
        thermal_valid.unmask(0).toFloat()
    )

    prefix = (
        f"{export_id}_"
        f"{row.city.lower()}_"
        f"{row.thermal_sensor.lower()}"
    )

    predictor_task = ee.batch.Export.image.toDrive(
        image=predictors,
        description=f"{prefix}_predictors",
        folder=DRIVE_FOLDER,
        fileNamePrefix=f"{prefix}_predictors_10m",
        region=region,
        crs=predictor_crs,
        scale=10,
        maxPixels=1e9,
        fileFormat="GeoTIFF",
        formatOptions={
            "cloudOptimized": True,
            "noData": NODATA
        }
    )

    thermal_task = ee.batch.Export.image.toDrive(
        image=thermal_export,
        description=f"{prefix}_thermal",
        folder=DRIVE_FOLDER,
        fileNamePrefix=f"{prefix}_thermal_native",
        region=region,
        crs=projection_info["crs"],
        crsTransform=projection_info["transform"],
        maxPixels=1e9,
        fileFormat="GeoTIFF",
        formatOptions={
            "cloudOptimized": True,
            "noData": NODATA
        }
    )

    predictor_task.start()
    thermal_task.start()

    task_rows.append({
        "export_id": export_id,
        "record_id": row.record_id,
        "city": row.city,
        "thermal_sensor": row.thermal_sensor,
        "acquisition_date": row.acquisition_date,
        "valid_fraction_pct": row.valid_fraction_pct,
        "centre_longitude": longitude,
        "centre_latitude": latitude,
        "predictor_crs": predictor_crs,
        "predictor_scale_m": 10,
        "thermal_crs": projection_info["crs"],
        "thermal_transform": str(projection_info["transform"]),
        "thermal_nominal_scale_m": native_scale,
        "predictor_task_id": predictor_task.id,
        "thermal_task_id": thermal_task.id,
        "prefix": prefix
    })

    print(
        f"Submitted {export_number:02d}/{EXPECTED_SCENES}: {prefix}"
    )

task_df = pd.DataFrame(task_rows)

manifest_out = OUT_DIR / "02_exported_scene_manifest.csv"
task_register = OUT_DIR / "03_export_task_register.csv"

data.to_csv(manifest_out, index=False)
task_df.to_csv(task_register, index=False)

print("\n" + "=" * 100)
print("THERMOFUSION STAGE 23 EXPORT — SUBMISSION COMPLETE")
print("=" * 100)
print(f"Scenes exported: {len(task_df)}")
print(f"Predictor tasks: {len(task_df)}")
print(f"Thermal tasks: {len(task_df)}")
print(f"Total Earth Engine tasks: {len(task_df) * 2}")
print(f"Google Drive folder: {DRIVE_FOLDER}")
print(f"Preflight report: {preflight_path}")
print(f"Task register: {task_register}")
display(task_df)
