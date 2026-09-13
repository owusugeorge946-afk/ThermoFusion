# =====================================================================
# ThermoFusion reproducibility code
# Source notebook cell(s): 24
# Audit status: ARCHIVE
# Extracted without scientific-logic modification.
# Replace local Google Drive paths as described in README.md when needed.
# =====================================================================

# ThermoFusion Stage 10R: repair Stage 11 pilot bundles that failed raster QA.
# Checks chip-level validity before exporting only replacement predictor/thermal pairs.

from google.colab import drive
from pathlib import Path
import math
import numpy as np
import pandas as pd
import ee


PROJECT_ID = "nana213"
VERIFIED_MATCH_PATH = Path(
    "/content/drive/MyDrive/ThermoFusion_Stage06_Match_Verification/"
    "07_verified_match_manifest.csv"
)
DRIVE_FOLDER = "ThermoFusion_Stage10_Pilot_Chips_Repair"
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
SCENES_PER_QUARTER = 2
MIN_THERMAL_COVERAGE = 25.0
MAX_S2_SCENE_CLOUD = 80.0
EXPECTED_CITIES = ["Accra", "Lagos", "Abidjan", "Freetown"]
EXPECTED_SENSORS = ["Landsat", "ECOSTRESS"]


drive.mount("/content/drive")
if not VERIFIED_MATCH_PATH.exists():
    raise FileNotFoundError(
        f"Stage 06 verified manifest not found: {VERIFIED_MATCH_PATH}"
    )

data = pd.read_csv(VERIFIED_MATCH_PATH)
data["thermal_datetime_utc"] = pd.to_datetime(
    data["thermal_datetime_utc"], errors="raise", utc=True
)
numeric_columns = [
    "valid_fraction_pct", "s2_cloudy_pixel_percentage",
    "s2_time_difference_hours", "s1_time_difference_hours",
    "era5_time_difference_hours", "all_three_found", "strict_temporal_match",
]
for column in numeric_columns:
    data[column] = pd.to_numeric(data[column], errors="coerce")
data["year"] = data["thermal_datetime_utc"].dt.year
data["quarter"] = data["thermal_datetime_utc"].dt.quarter

eligible = data[
    data["all_three_found"].eq(1)
    & data["strict_temporal_match"].eq(1)
    & data["valid_fraction_pct"].ge(MIN_THERMAL_COVERAGE)
    & data["s2_cloudy_pixel_percentage"].le(MAX_S2_SCENE_CLOUD)
].copy()

score_groups = eligible.groupby(["city", "thermal_sensor"], observed=True)
coverage_rank = score_groups["valid_fraction_pct"].rank(pct=True, method="average")
cloud_rank = 1.0 - score_groups["s2_cloudy_pixel_percentage"].rank(
    pct=True, method="average"
)
s2_rank = 1.0 - score_groups["s2_time_difference_hours"].rank(
    pct=True, method="average"
)
s1_rank = 1.0 - score_groups["s1_time_difference_hours"].rank(
    pct=True, method="average"
)
era5_rank = 1.0 - score_groups["era5_time_difference_hours"].rank(
    pct=True, method="average"
)
eligible["pilot_quality_score"] = (
    0.35 * coverage_rank + 0.25 * cloud_rank + 0.20 * s2_rank
    + 0.15 * s1_rank + 0.05 * era5_rank
)

ee.Authenticate()
ee.Initialize(project=PROJECT_ID)
print(f"Connected to Earth Engine project: {PROJECT_ID}")


def source_index(scene_id):
    index = str(scene_id).split("/")[-1]
    while index.startswith("1_") or index.startswith("2_"):
        index = index[2:]
    return index


def collection_image(scene_id, collection_id):
    value = str(scene_id)
    if value.startswith(collection_id + "/"):
        return ee.Image(value)
    return ee.Image(f"{collection_id}/{value.split('/')[-1]}")


def normalized_ee_date(value):
    """Convert CSV timestamps, including timezone offsets, to EE-safe UTC ISO."""
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
        offset = ee.Number(image.get("system:time_start")).subtract(
            target_time.millis()
        ).abs()
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
        prepared = ee.Image.cat(
            reflectance,
            ndvi,
            ndbi,
            ndmi,
            valid.selfMask().rename("s2_valid"),
        )
        return prepared.set("time_offset_ms", offset)

    # Mosaic all intersecting granules in the accepted temporal window. Sorting
    # from farthest to nearest makes the closest valid observation take priority.
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
        offset = ee.Number(image.get("system:time_start")).subtract(
            target_time.millis()
        ).abs()
        vv = image.select("VV").rename("s1_vv_db")
        vh = image.select("VH").rename("s1_vh_db")
        difference = vv.subtract(vh).rename("s1_vv_minus_vh_db")
        valid = vv.mask().And(vh.mask())
        prepared = ee.Image.cat(
            vv,
            vh,
            difference,
            valid.selfMask().rename("s1_valid"),
        )
        return prepared.set("time_offset_ms", offset)

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


def utm_crs(point):
    longitude, latitude = point.coordinates().getInfo()
    zone = int(math.floor((longitude + 180) / 6) + 1)
    epsg = 32600 + zone if latitude >= 0 else 32700 + zone
    return f"EPSG:{epsg}", longitude, latitude


def select_chip_centre(valid_mask, city_geometry, native_scale):
    eroded = valid_mask.selfMask().focal_min(
        radius=EROSION_RADIUS_M, units="meters"
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
        sample = valid_mask.selfMask().sample(
            region=city_geometry,
            scale=native_scale,
            numPixels=1,
            seed=946,
            geometries=True,
            tileScale=4,
        )
        count = sample.size().getInfo()

    if count == 0:
        return None

    return ee.Feature(sample.first()).geometry()


STAGE11_CSV = Path(
    "/content/drive/MyDrive/ThermoFusion_Stage11_Pilot_Verification/"
    "01_pilot_bundle_verification.csv"
)
CURRENT_MANIFEST = Path(
    "/content/drive/MyDrive/ThermoFusion_Stage10_Revised_Pilot_Manifest.csv"
)
REPAIR_REGISTER = Path(
    "/content/drive/MyDrive/ThermoFusion_Stage10_Repair_Tasks.csv"
)
REPAIR_PREFLIGHT = Path(
    "/content/drive/MyDrive/ThermoFusion_Stage10_Repair_Preflight.csv"
)

for required_path in [STAGE11_CSV, CURRENT_MANIFEST]:
    if not required_path.exists():
        raise FileNotFoundError(f"Required verification input not found: {required_path}")

verification = pd.read_csv(STAGE11_CSV)
current_manifest = pd.read_csv(CURRENT_MANIFEST)
failed = verification.loc[verification["status"].ne("PASS")].copy()

if failed.empty:
    raise RuntimeError("Stage 11 contains no failed bundles; no repair is required.")
if len(current_manifest) != 64 or current_manifest["record_id"].nunique() != 64:
    raise RuntimeError("Current 64-scene manifest failed its integrity check.")
if failed["pilot_id"].duplicated().any() or failed["record_id"].duplicated().any():
    raise RuntimeError("Stage 11 failure list contains duplicate identities.")

print(f"Bundles requiring replacement: {len(failed)}")
display(
    failed[
        [
            "pilot_id", "record_id", "city", "thermal_sensor",
            "s2_valid_fraction", "s1_valid_fraction",
            "thermal_valid_fraction", "status",
        ]
    ]
)


def fraction(mask, region, scale):
    result = (
        mask.unmask(0)
        .reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=scale,
            maxPixels=1e8,
            tileScale=4,
        )
        .values()
        .get(0)
    )
    return float(ee.Number(result).getInfo())


def temperature_limits(lst, valid, region, scale):
    result = (
        lst.updateMask(valid)
        .reduceRegion(
            reducer=ee.Reducer.minMax(),
            geometry=region,
            scale=scale,
            maxPixels=1e8,
            tileScale=4,
        )
        .getInfo()
    )
    values = [float(value) for value in result.values() if value is not None]
    return (min(values), max(values)) if values else (np.nan, np.nan)


selected_record_ids = set(current_manifest["record_id"].astype(str))
replacement_bundles = []
audit_rows = []

for failed_row in failed.itertuples(index=False):
    old_manifest_row = current_manifest.loc[
        current_manifest["record_id"].astype(str).eq(str(failed_row.record_id))
    ]
    if len(old_manifest_row) != 1:
        raise RuntimeError(
            f"Could not uniquely resolve failed record {failed_row.record_id}."
        )
    old_quarter = int(old_manifest_row.iloc[0]["quarter"])
    city_geometry = ee.FeatureCollection(CITY_ASSETS[failed_row.city]).geometry()

    candidates = eligible[
        eligible["city"].eq(failed_row.city)
        & eligible["thermal_sensor"].eq(failed_row.thermal_sensor)
        & ~eligible["record_id"].astype(str).isin(selected_record_ids)
    ].copy()
    candidates["same_quarter_priority"] = candidates["quarter"].eq(old_quarter).astype(int)
    candidates = candidates.sort_values(
        ["same_quarter_priority", "pilot_quality_score", "valid_fraction_pct"],
        ascending=[False, False, False],
    )

    chosen = None
    for candidate_rank, row in enumerate(candidates.itertuples(index=False), start=1):
        thermal_source, thermal_lst, thermal_valid = thermal_image_and_mask(row)
        thermal_projection = thermal_source.select(
            "ST_B10" if row.thermal_sensor == "Landsat" else "LST"
        ).projection()
        projection_info = thermal_projection.getInfo()
        native_scale = float(thermal_projection.nominalScale().getInfo())
        s2 = sentinel2_predictors(row.thermal_datetime_utc, city_geometry)
        s1 = sentinel1_predictors(row.thermal_datetime_utc, city_geometry)
        joint_valid = (
            thermal_valid.unmask(0)
            .And(s2.select("s2_valid").unmask(0).gt(0.5))
            .And(s1.select("s1_valid").unmask(0).gt(0.5))
            .rename("joint_valid")
        )
        centre = select_chip_centre(joint_valid, city_geometry, native_scale)

        if centre is None:
            audit_rows.append(
                {
                    "pilot_id": failed_row.pilot_id,
                    "candidate_record_id": row.record_id,
                    "candidate_rank": candidate_rank,
                    "candidate_quarter": row.quarter,
                    "joint_centre_found": False,
                    "status": "REJECT_NO_JOINT_CENTRE",
                }
            )
            continue

        predictor_crs, longitude, latitude = utm_crs(centre)
        region = centre.buffer(CHIP_SIZE_M / 2).bounds()
        s2_fraction = fraction(s2.select("s2_valid"), region, 10)
        s1_fraction = fraction(s1.select("s1_valid"), region, 10)
        thermal_fraction = fraction(thermal_valid, region, native_scale)
        minimum_c, maximum_c = temperature_limits(
            thermal_lst, thermal_valid, region, native_scale
        )

        passes = (
            s2_fraction >= 0.25
            and s1_fraction >= 0.80
            and thermal_fraction >= 0.10
            and np.isfinite(minimum_c)
            and np.isfinite(maximum_c)
            and minimum_c > -30
            and maximum_c < 80
        )
        audit_rows.append(
            {
                "pilot_id": failed_row.pilot_id,
                "candidate_record_id": row.record_id,
                "candidate_rank": candidate_rank,
                "candidate_quarter": row.quarter,
                "same_quarter_as_rejected": row.quarter == old_quarter,
                "joint_centre_found": True,
                "s2_valid_fraction": s2_fraction,
                "s1_valid_fraction": s1_fraction,
                "thermal_valid_fraction": thermal_fraction,
                "thermal_min_c": minimum_c,
                "thermal_max_c": maximum_c,
                "status": "PASS" if passes else "REJECT_THRESHOLDS",
            }
        )
        print(
            f"{failed_row.pilot_id} candidate {candidate_rank}: "
            f"S2={s2_fraction:.1%}, S1={s1_fraction:.1%}, "
            f"thermal={thermal_fraction:.1%} | {'PASS' if passes else 'REJECT'}"
        )

        if passes:
            chosen = (
                failed_row.pilot_id, row, thermal_lst, thermal_valid,
                projection_info, native_scale, s2, s1, centre,
                predictor_crs, longitude, latitude, region,
            )
            selected_record_ids.add(str(row.record_id))
            break

    if chosen is None:
        pd.DataFrame(audit_rows).to_csv(REPAIR_PREFLIGHT, index=False)
        raise RuntimeError(
            f"No fully valid replacement was found for {failed_row.pilot_id}. "
            "No repair export tasks were submitted."
        )
    replacement_bundles.append(chosen)

pd.DataFrame(audit_rows).to_csv(REPAIR_PREFLIGHT, index=False)
if len(replacement_bundles) != len(failed):
    raise RuntimeError("Repair preflight was incomplete; no tasks were submitted.")

print(
    f"\nRepair preflight passed for {len(replacement_bundles)}/"
    f"{len(failed)} failed bundles. Submitting {len(replacement_bundles) * 2} tasks."
)

dem = (
    ee.ImageCollection("COPERNICUS/DEM/GLO30")
    .select("DEM")
    .mosaic()
    .rename("elevation_m")
)
slope = ee.Terrain.slope(dem).rename("slope_deg")

task_rows = []
updated_manifest = current_manifest.copy()

for bundle in replacement_bundles:
    (
        pilot_id, row, thermal_lst, thermal_valid, projection_info,
        native_scale, s2, s1, centre, predictor_crs,
        longitude, latitude, region,
    ) = bundle

    predictors = ee.Image.cat(s2, s1, dem, slope).toFloat().unmask(NODATA)
    thermal_export = ee.Image.cat(
        thermal_lst.unmask(NODATA).toFloat(),
        thermal_valid.unmask(0).toFloat(),
    )
    prefix = f"{pilot_id}_{row.city.lower()}_{row.thermal_sensor.lower()}"

    predictor_task = ee.batch.Export.image.toDrive(
        image=predictors,
        description=f"repair_{prefix}_predictors",
        folder=DRIVE_FOLDER,
        fileNamePrefix=f"{prefix}_predictors_10m",
        region=region,
        crs=predictor_crs,
        scale=10,
        maxPixels=1e9,
        fileFormat="GeoTIFF",
        formatOptions={"cloudOptimized": True, "noData": NODATA},
    )
    thermal_task = ee.batch.Export.image.toDrive(
        image=thermal_export,
        description=f"repair_{prefix}_thermal",
        folder=DRIVE_FOLDER,
        fileNamePrefix=f"{prefix}_thermal_native",
        region=region,
        crs=projection_info["crs"],
        crsTransform=projection_info["transform"],
        maxPixels=1e9,
        fileFormat="GeoTIFF",
        formatOptions={"cloudOptimized": True, "noData": NODATA},
    )
    predictor_task.start()
    thermal_task.start()

    task_rows.append(
        {
            "pilot_id": pilot_id,
            "record_id": row.record_id,
            "city": row.city,
            "thermal_sensor": row.thermal_sensor,
            "centre_longitude": longitude,
            "centre_latitude": latitude,
            "predictor_crs": predictor_crs,
            "predictor_scale_m": 10,
            "thermal_crs": projection_info["crs"],
            "thermal_transform": str(projection_info["transform"]),
            "thermal_nominal_scale_m": native_scale,
            "s2_composite_window_days": S2_WINDOW_DAYS,
            "s1_composite_window_days": S1_WINDOW_DAYS,
            "predictor_task_id": predictor_task.id,
            "thermal_task_id": thermal_task.id,
            "prefix": prefix,
        }
    )

    replacement_values = row._asdict()
    replacement_values["pilot_id"] = pilot_id
    mask = updated_manifest["pilot_id"].eq(pilot_id)
    for column, value in replacement_values.items():
        if column in updated_manifest.columns:
            updated_manifest.loc[mask, column] = value

task_df = pd.DataFrame(task_rows)
task_df.to_csv(REPAIR_REGISTER, index=False)
updated_manifest.to_csv(CURRENT_MANIFEST, index=False)

print("\nTHERMOFUSION STAGE 10R TARGETED REPAIR")
print(f"Rejected bundles replaced: {len(task_df)}")
print(f"Predictor repair exports: {len(task_df)}")
print(f"Thermal repair exports: {len(task_df)}")
print(f"Total submitted repair tasks: {len(task_df) * 2}")
print(f"Repair folder: {DRIVE_FOLDER}")
print(f"Repair preflight: {REPAIR_PREFLIGHT}")
print(f"Repair task register: {REPAIR_REGISTER}")
display(task_df)
