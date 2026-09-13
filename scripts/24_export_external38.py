# =====================================================================
# ThermoFusion reproducibility code
# Source notebook cell(s): 55
# Audit status: KEEP
# Extracted without scientific-logic modification.
# Replace local Google Drive paths as described in README.md when needed.
# =====================================================================

# ==========================================================================================
# THERMOFUSION STAGE 23R2 — FINAL EXPANDED TEMPORAL-TEST PREFLIGHT AND GEE EXPORT
# Fixed 38-scene supplementary test. No reselection and no model retraining.
# Exports run only if all 38 scenes pass spatial preflight.
# ==========================================================================================

!pip -q install earthengine-api

from google.colab import drive
from pathlib import Path
import math
import pandas as pd
import ee
from IPython.display import display

# ------------------------------------------------------------------------------------------
# 1. Paths and fixed export contract
# ------------------------------------------------------------------------------------------
drive.mount("/content/drive")

PROJECT_ID = "nana213"

REVISED_MANIFEST = Path(
    "/content/drive/MyDrive/ThermoFusion_Stage23R2_FinalReplacement/"
    "01_final_expanded_test_manifest.csv"
)

OUT_DIR = Path(
    "/content/drive/MyDrive/ThermoFusion_Stage23R2_FinalReplacement/"
    "Stage23R2_Export"
)

DRIVE_EXPORT_FOLDER = "ThermoFusion_Stage23R2_ExpandedTemporalTest_Chips"

OUT_DIR.mkdir(parents=True, exist_ok=True)

PREFLIGHT_CSV = OUT_DIR / "01_spatial_preflight.csv"
EXPORTED_MANIFEST_CSV = OUT_DIR / "02_exported_scene_manifest.csv"
TASK_REGISTER_CSV = OUT_DIR / "03_export_task_register.csv"

EXPECTED_SCENES = 38
CHIP_SIZE_M = 2560
EROSION_RADIUS_M = 640
NODATA = -9999

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

PREDICTOR_BANDS = [
    "s2_b2", "s2_b3", "s2_b4", "s2_b8", "s2_b11", "s2_b12",
    "ndvi", "ndbi", "ndmi", "s2_valid",
    "s1_vv_db", "s1_vh_db", "s1_vv_minus_vh_db", "s1_valid",
    "elevation_m", "slope_deg",
]

# ------------------------------------------------------------------------------------------
# 2. Load and validate the final fixed manifest
# ------------------------------------------------------------------------------------------
if not REVISED_MANIFEST.exists():
    raise FileNotFoundError(f"Final Stage 23R2 manifest not found:\n{REVISED_MANIFEST}")

manifest = pd.read_csv(REVISED_MANIFEST)

required_columns = [
    "record_id", "city", "thermal_sensor", "thermal_scene_id",
    "acquisition_date", "valid_fraction_pct",
]

missing = [column for column in required_columns if column not in manifest.columns]
if missing:
    raise KeyError(
        f"Manifest is missing required columns: {missing}\n"
        f"Available columns: {manifest.columns.tolist()}"
    )

manifest["acquisition_date"] = pd.to_datetime(
    manifest["acquisition_date"], utc=True, errors="raise"
)
manifest["valid_fraction_pct"] = pd.to_numeric(
    manifest["valid_fraction_pct"], errors="raise"
)

if len(manifest) != EXPECTED_SCENES:
    raise RuntimeError(
        f"Expected {EXPECTED_SCENES} fixed scenes, found {len(manifest)}."
    )

if manifest["record_id"].nunique() != EXPECTED_SCENES:
    raise RuntimeError("Duplicate record IDs found in the final manifest.")

if "thermal_scene_id" in manifest.columns:
    if manifest["thermal_scene_id"].nunique() != EXPECTED_SCENES:
        raise RuntimeError("Duplicate thermal scenes found in the final manifest.")

if manifest["valid_fraction_pct"].min() < 50.0:
    raise RuntimeError(
        "At least one selected scene is below the required 50% valid-pixel coverage."
    )

if "temporal_eligibility" in manifest.columns:
    invalid_temporal = manifest.loc[
        ~manifest["temporal_eligibility"].astype(str).str.upper().eq("PASS")
    ]
    if not invalid_temporal.empty:
        raise RuntimeError("One or more scenes do not satisfy temporal eligibility.")

manifest = manifest.sort_values(
    ["city", "thermal_sensor", "acquisition_date", "record_id"],
    kind="mergesort"
).reset_index(drop=True)

print("=" * 100)
print("THERMOFUSION STAGE 23R2 — FINAL EXPANDED TEMPORAL-TEST EXPORT")
print("=" * 100)
print(f"Fixed selected scenes: {len(manifest)}")
print(f"Unique record IDs: {manifest['record_id'].nunique()}")
print(f"Minimum valid fraction: {manifest['valid_fraction_pct'].min():.2f}%")

# ------------------------------------------------------------------------------------------
# 3. Connect to Google Earth Engine
# ------------------------------------------------------------------------------------------
ee.Authenticate()
ee.Initialize(project=PROJECT_ID)
print(f"Connected to Earth Engine project: {PROJECT_ID}")

# ------------------------------------------------------------------------------------------
# 4. Processing functions — identical predictor/target construction to the original pipeline
# ------------------------------------------------------------------------------------------
def source_index(scene_id):
    index = str(scene_id).split("/")[-1]
    while index.startswith("1_") or index.startswith("2_"):
        index = index[2:]
    return index


def normalized_ee_date(value):
    timestamp = pd.to_datetime(value, utc=True, errors="raise")
    return ee.Date(timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"))


def thermal_image_and_mask(row):
    index = source_index(row["thermal_scene_id"])

    if row["thermal_sensor"] == "Landsat":
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

        return (
            image,
            lst.updateMask(valid),
            valid.rename("thermal_valid"),
        )

    image = ee.Image(f"{ECOSTRESS}/{index}")

    lst = image.select("LST").subtract(273.15).rename("thermal_lst_c")

    valid = (
        image.select("QC").bitwiseAnd(31).eq(0)
        .And(image.select("cloud").bitwiseAnd(1).eq(0))
        .And(image.select("water").bitwiseAnd(1).eq(0))
        .And(lst.gt(-23.15))
        .And(lst.lt(76.85))
    )

    return (
        image,
        lst.updateMask(valid),
        valid.rename("thermal_valid"),
    )


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
            .rename(["s2_b2", "s2_b3", "s2_b4", "s2_b8", "s2_b11", "s2_b12"])
            .updateMask(valid)
        )

        ndvi = reflectance.normalizedDifference(["s2_b8", "s2_b4"]).rename("ndvi")
        ndbi = reflectance.normalizedDifference(["s2_b11", "s2_b8"]).rename("ndbi")
        ndmi = reflectance.normalizedDifference(["s2_b8", "s2_b11"]).rename("ndmi")

        return ee.Image.cat(
            reflectance,
            ndvi,
            ndbi,
            ndmi,
            valid.selfMask().rename("s2_valid"),
        ).set("time_offset_ms", offset)

    return (
        ee.ImageCollection(SENTINEL_2)
        .filterBounds(geometry)
        .filterDate(
            target_time.advance(-S2_WINDOW_DAYS, "day"),
            target_time.advance(S2_WINDOW_DAYS + 1, "day"),
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
            valid.selfMask().rename("s1_valid"),
        ).set("time_offset_ms", offset)

    return (
        ee.ImageCollection(SENTINEL_1)
        .filterBounds(geometry)
        .filterDate(
            target_time.advance(-S1_WINDOW_DAYS, "day"),
            target_time.advance(S1_WINDOW_DAYS + 1, "day"),
        )
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .filter(ee.Filter.eq("resolution_meters", 10))
        .map(prepare)
        .sort("time_offset_ms", False)
        .mosaic()
    )


def select_chip_centre(valid_mask, city_geometry, native_scale):
    eroded = valid_mask.selfMask().focal_min(
        radius=EROSION_RADIUS_M,
        units="meters",
    )

    sample = eroded.sample(
        region=city_geometry,
        scale=native_scale,
        numPixels=1,
        seed=946,
        geometries=True,
        tileScale=4,
    )

    if sample.size().getInfo() == 0:
        sample = valid_mask.selfMask().sample(
            region=city_geometry,
            scale=native_scale,
            numPixels=1,
            seed=946,
            geometries=True,
            tileScale=4,
        )

    if sample.size().getInfo() == 0:
        return None

    return ee.Feature(sample.first()).geometry()


def utm_crs(point):
    longitude, latitude = point.coordinates().getInfo()
    zone = int(math.floor((longitude + 180) / 6) + 1)
    epsg = 32600 + zone if latitude >= 0 else 32700 + zone
    return f"EPSG:{epsg}", longitude, latitude


# ------------------------------------------------------------------------------------------
# 5. Phase 1 — strict preflight of every fixed selected scene
# ------------------------------------------------------------------------------------------
preflight_rows = []
resolved_bundles = []

for position, (_, row) in enumerate(manifest.iterrows(), start=1):
    city = row["city"]

    if city not in CITY_ASSETS:
        raise KeyError(f"No city asset configured for: {city}")

    try:
        city_geometry = ee.FeatureCollection(CITY_ASSETS[city]).geometry()

        thermal_source, thermal_lst, thermal_valid = thermal_image_and_mask(row)

        thermal_band = "ST_B10" if row["thermal_sensor"] == "Landsat" else "LST"
        thermal_projection = thermal_source.select(thermal_band).projection()

        projection_info = thermal_projection.getInfo()
        native_scale = thermal_projection.nominalScale().getInfo()

        s2 = sentinel2_predictors(row["acquisition_date"], city_geometry)
        s1 = sentinel1_predictors(row["acquisition_date"], city_geometry)

        joint_valid = (
            thermal_valid.unmask(0)
            .And(s2.select("s2_valid").unmask(0).gt(0.5))
            .And(s1.select("s1_valid").unmask(0).gt(0.5))
            .rename("joint_valid")
        )

        centre = select_chip_centre(
            joint_valid,
            city_geometry,
            native_scale,
        )

        passed = centre is not None
        error_message = ""

    except Exception as exc:
        thermal_source = None
        thermal_lst = None
        thermal_valid = None
        projection_info = None
        native_scale = None
        s2 = None
        s1 = None
        centre = None
        passed = False
        error_message = str(exc)

    preflight_rows.append({
        "record_id": row["record_id"],
        "city": row["city"],
        "thermal_sensor": row["thermal_sensor"],
        "acquisition_date": row["acquisition_date"],
        "valid_fraction_pct": row["valid_fraction_pct"],
        "joint_valid_centre_found": passed,
        "error_message": error_message,
    })

    print(
        f"Preflight {position:02d}/{EXPECTED_SCENES}: "
        f"{row['record_id']} — {'PASS' if passed else 'FAIL'}"
    )

    if passed:
        resolved_bundles.append({
            "export_id": f"TFX{position:03d}",
            "row": row.copy(),
            "thermal_lst": thermal_lst,
            "thermal_valid": thermal_valid,
            "projection_info": projection_info,
            "native_scale": native_scale,
            "s2": s2,
            "s1": s1,
            "centre": centre,
        })

preflight_df = pd.DataFrame(preflight_rows)
preflight_df.to_csv(PREFLIGHT_CSV, index=False)

failed = preflight_df.loc[
    ~preflight_df["joint_valid_centre_found"]
].copy()

if len(resolved_bundles) != EXPECTED_SCENES:
    print()
    print("FAILED SCENES:")
    display(failed)

    raise RuntimeError(
        f"Only {len(resolved_bundles)} of {EXPECTED_SCENES} scenes passed "
        "spatial preflight. No Earth Engine exports were submitted."
    )

print()
print(f"STAGE 23R2 PREFLIGHT: PASS — {len(resolved_bundles)}/{EXPECTED_SCENES} scenes")

# ------------------------------------------------------------------------------------------
# 6. Phase 2 — submit 38 predictor and 38 thermal exports
# ------------------------------------------------------------------------------------------
dem = (
    ee.ImageCollection("COPERNICUS/DEM/GLO30")
    .select("DEM")
    .mosaic()
    .rename("elevation_m")
)

slope = ee.Terrain.slope(dem).rename("slope_deg")

task_rows = []
exported_rows = []

for export_number, bundle in enumerate(resolved_bundles, start=1):
    row = bundle["row"]

    predictor_crs, longitude, latitude = utm_crs(bundle["centre"])
    region = bundle["centre"].buffer(CHIP_SIZE_M / 2).bounds()

    predictors = (
        ee.Image.cat(bundle["s2"], bundle["s1"], dem, slope)
        .select(PREDICTOR_BANDS)
        .toFloat()
        .unmask(NODATA)
    )

    thermal_export = ee.Image.cat(
        bundle["thermal_lst"].unmask(NODATA).toFloat(),
        bundle["thermal_valid"].unmask(0).toFloat(),
    )

    prefix = (
        f"{bundle['export_id']}_"
        f"{row['city'].lower()}_"
        f"{row['thermal_sensor'].lower()}"
    )

    predictor_task = ee.batch.Export.image.toDrive(
        image=predictors,
        description=f"{prefix}_predictors",
        folder=DRIVE_EXPORT_FOLDER,
        fileNamePrefix=f"{prefix}_predictors_10m",
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

    thermal_task = ee.batch.Export.image.toDrive(
        image=thermal_export,
        description=f"{prefix}_thermal",
        folder=DRIVE_EXPORT_FOLDER,
        fileNamePrefix=f"{prefix}_thermal_native",
        region=region,
        crs=bundle["projection_info"]["crs"],
        crsTransform=bundle["projection_info"]["transform"],
        maxPixels=1e9,
        fileFormat="GeoTIFF",
        formatOptions={
            "cloudOptimized": True,
            "noData": NODATA,
        },
    )

    predictor_task.start()
    thermal_task.start()

    exported_rows.append({
        "export_id": bundle["export_id"],
        "record_id": row["record_id"],
        "city": row["city"],
        "thermal_sensor": row["thermal_sensor"],
        "acquisition_date": row["acquisition_date"],
        "valid_fraction_pct": row["valid_fraction_pct"],
        "file_prefix": prefix,
    })

    task_rows.append({
        "export_id": bundle["export_id"],
        "record_id": row["record_id"],
        "city": row["city"],
        "thermal_sensor": row["thermal_sensor"],
        "centre_longitude": longitude,
        "centre_latitude": latitude,
        "predictor_crs": predictor_crs,
        "predictor_scale_m": 10,
        "thermal_crs": bundle["projection_info"]["crs"],
        "thermal_transform": str(bundle["projection_info"]["transform"]),
        "thermal_nominal_scale_m": bundle["native_scale"],
        "s2_composite_window_days": S2_WINDOW_DAYS,
        "s1_composite_window_days": S1_WINDOW_DAYS,
        "predictor_task_id": predictor_task.id,
        "thermal_task_id": thermal_task.id,
        "file_prefix": prefix,
    })

    print(f"Submitted {export_number:02d}/{EXPECTED_SCENES}: {prefix}")

exported_df = pd.DataFrame(exported_rows)
task_df = pd.DataFrame(task_rows)

exported_df.to_csv(EXPORTED_MANIFEST_CSV, index=False)
task_df.to_csv(TASK_REGISTER_CSV, index=False)

print()
print("=" * 100)
print("THERMOFUSION STAGE 23R2 EXPORT SUBMISSION COMPLETE")
print("=" * 100)
print(f"Spatial preflight: {len(resolved_bundles)}/{EXPECTED_SCENES} PASS")
print(f"Predictor exports submitted: {len(task_df)}")
print(f"Thermal exports submitted: {len(task_df)}")
print(f"Total Earth Engine tasks submitted: {len(task_df) * 2}")
print(f"Drive chip folder: {DRIVE_EXPORT_FOLDER}")
print(f"Preflight register: {PREFLIGHT_CSV}")
print(f"Exported-scene manifest: {EXPORTED_MANIFEST_CSV}")
print(f"Task register: {TASK_REGISTER_CSV}")

display(task_df)
