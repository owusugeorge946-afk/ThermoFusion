# =====================================================================
# ThermoFusion reproducibility code
# Source notebook cell(s): 22
# Audit status: KEEP
# Extracted without scientific-logic modification.
# Replace local Google Drive paths as described in README.md when needed.
# =====================================================================

# ThermoFusion Stage 10: export the complete 64-scene balanced pilot dataset.
# Runs a no-partial-export preflight, then exports 64 predictor/thermal pairs.

from google.colab import drive
from pathlib import Path
import math
import pandas as pd
import ee


PROJECT_ID = "nana213"
VERIFIED_MATCH_PATH = Path(
    "/content/drive/MyDrive/ThermoFusion_Stage06_Match_Verification/"
    "07_verified_match_manifest.csv"
)
DRIVE_FOLDER = "ThermoFusion_Stage10_Pilot_Chips"
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


# Phase 1: choose two spatially valid scenes per city-sensor-quarter stratum.
# Candidates follow the Stage 07 quality score and prefer different years.
preflight_rows = []
resolved_bundles = []
selected_manifest_rows = []
selection_audit = []

for city in EXPECTED_CITIES:
    for thermal_sensor in EXPECTED_SENSORS:
        city_geometry = ee.FeatureCollection(CITY_ASSETS[city]).geometry()
        for quarter in range(1, 5):
            candidates = eligible[
                eligible["city"].eq(city)
                & eligible["thermal_sensor"].eq(thermal_sensor)
                & eligible["quarter"].eq(quarter)
            ].sort_values(
                ["pilot_quality_score", "valid_fraction_pct"],
                ascending=[False, False],
            )
            diverse = candidates.drop_duplicates("year", keep="first")
            remainder = candidates[
                ~candidates["record_id"].isin(diverse["record_id"])
            ]
            ordered = pd.concat([diverse, remainder], ignore_index=False)
            selected_in_stratum = 0

            for candidate_rank, row in enumerate(
                ordered.itertuples(index=False), start=1
            ):
                thermal_source, thermal_lst, thermal_valid = thermal_image_and_mask(row)
                thermal_projection = thermal_source.select(
                    "ST_B10" if row.thermal_sensor == "Landsat" else "LST"
                ).projection()
                projection_info = thermal_projection.getInfo()
                native_scale = thermal_projection.nominalScale().getInfo()
                s2 = sentinel2_predictors(row.thermal_datetime_utc, city_geometry)
                s1 = sentinel1_predictors(row.thermal_datetime_utc, city_geometry)
                joint_valid = (
                    thermal_valid.unmask(0)
                    .And(s2.select("s2_valid").unmask(0).gt(0.5))
                    .And(s1.select("s1_valid").unmask(0).gt(0.5))
                    .rename("joint_valid")
                )
                centre = select_chip_centre(
                    joint_valid, city_geometry, native_scale
                )
                passed = centre is not None
                preflight_rows.append(
                    {
                        "record_id": row.record_id,
                        "city": city,
                        "thermal_sensor": thermal_sensor,
                        "quarter": quarter,
                        "year": row.year,
                        "candidate_rank": candidate_rank,
                        "joint_valid_centre_found": passed,
                        "selected": passed and selected_in_stratum < SCENES_PER_QUARTER,
                    }
                )
                print(
                    f"Tested {city} {thermal_sensor} Q{quarter} candidate "
                    f"{candidate_rank}: {'PASS' if passed else 'FAIL'}"
                )
                if passed:
                    pilot_id = f"TFP{len(resolved_bundles) + 1:04d}"
                    resolved_bundles.append(
                        (
                            pilot_id, row, thermal_lst, thermal_valid,
                            projection_info, native_scale, s2, s1, centre,
                        )
                    )
                    selected_row = row._asdict()
                    selected_row["pilot_id"] = pilot_id
                    selected_manifest_rows.append(selected_row)
                    selected_in_stratum += 1
                if selected_in_stratum == SCENES_PER_QUARTER:
                    break

            selection_audit.append(
                {
                    "city": city,
                    "thermal_sensor": thermal_sensor,
                    "quarter": quarter,
                    "eligible_candidates": len(candidates),
                    "tested_candidates": candidate_rank if len(ordered) else 0,
                    "selected": selected_in_stratum,
                    "status": "PASS" if selected_in_stratum == 2 else "INSUFFICIENT",
                }
            )

# Some quarters have no jointly valid candidates. Fill only those shortfalls
# from the best untested candidates in the same city-sensor group, retaining
# eight scenes per group without weakening any validity threshold.
group_audit = []
for city in EXPECTED_CITIES:
    for thermal_sensor in EXPECTED_SENSORS:
        city_geometry = ee.FeatureCollection(CITY_ASSETS[city]).geometry()
        already_selected = sum(
            item[1].city == city and item[1].thermal_sensor == thermal_sensor
            for item in resolved_bundles
        )
        deficit = 8 - already_selected
        tested_ids = {
            item["record_id"] for item in preflight_rows
            if item["city"] == city and item["thermal_sensor"] == thermal_sensor
        }
        fill_candidates = eligible[
            eligible["city"].eq(city)
            & eligible["thermal_sensor"].eq(thermal_sensor)
            & ~eligible["record_id"].isin(tested_ids)
        ].sort_values(
            ["pilot_quality_score", "valid_fraction_pct"],
            ascending=[False, False],
        )
        fill_tested = 0

        if deficit > 0:
            print(
                f"Redistributing {deficit} quarter-shortfall slot(s) within "
                f"{city} {thermal_sensor}."
            )

        for fill_rank, row in enumerate(
            fill_candidates.itertuples(index=False), start=1
        ):
            if deficit == 0:
                break
            fill_tested += 1
            thermal_source, thermal_lst, thermal_valid = thermal_image_and_mask(row)
            thermal_projection = thermal_source.select(
                "ST_B10" if row.thermal_sensor == "Landsat" else "LST"
            ).projection()
            projection_info = thermal_projection.getInfo()
            native_scale = thermal_projection.nominalScale().getInfo()
            s2 = sentinel2_predictors(row.thermal_datetime_utc, city_geometry)
            s1 = sentinel1_predictors(row.thermal_datetime_utc, city_geometry)
            joint_valid = (
                thermal_valid.unmask(0)
                .And(s2.select("s2_valid").unmask(0).gt(0.5))
                .And(s1.select("s1_valid").unmask(0).gt(0.5))
                .rename("joint_valid")
            )
            centre = select_chip_centre(joint_valid, city_geometry, native_scale)
            passed = centre is not None
            preflight_rows.append(
                {
                    "record_id": row.record_id,
                    "city": city,
                    "thermal_sensor": thermal_sensor,
                    "quarter": row.quarter,
                    "year": row.year,
                    "candidate_rank": fill_rank,
                    "selection_phase": "within_group_redistribution",
                    "joint_valid_centre_found": passed,
                    "selected": passed,
                }
            )
            print(
                f"Fill test {city} {thermal_sensor} Q{row.quarter} "
                f"candidate {fill_rank}: {'PASS' if passed else 'FAIL'}"
            )
            if passed:
                pilot_id = f"TFP{len(resolved_bundles) + 1:04d}"
                resolved_bundles.append(
                    (
                        pilot_id, row, thermal_lst, thermal_valid,
                        projection_info, native_scale, s2, s1, centre,
                    )
                )
                selected_row = row._asdict()
                selected_row["pilot_id"] = pilot_id
                selected_manifest_rows.append(selected_row)
                deficit -= 1

        final_count = sum(
            item[1].city == city and item[1].thermal_sensor == thermal_sensor
            for item in resolved_bundles
        )
        group_audit.append(
            {
                "city": city,
                "thermal_sensor": thermal_sensor,
                "initially_selected": already_selected,
                "redistributed": final_count - already_selected,
                "fill_candidates_tested": fill_tested,
                "final_selected": final_count,
                "status": "PASS" if final_count == 8 else "INSUFFICIENT",
            }
        )

preflight_df = pd.DataFrame(preflight_rows)
preflight_path = Path(
    "/content/drive/MyDrive/ThermoFusion_Stage10_Pilot_Preflight.csv"
)
preflight_df.to_csv(preflight_path, index=False)

selection_audit_df = pd.DataFrame(selection_audit)
selection_audit_df.to_csv(
    "/content/drive/MyDrive/ThermoFusion_Stage10_Replacement_Selection_Audit.csv",
    index=False,
)
group_audit_df = pd.DataFrame(group_audit)
group_audit_df.to_csv(
    "/content/drive/MyDrive/ThermoFusion_Stage10_Group_Balance_Audit.csv",
    index=False,
)
selected_ids = preflight_df.loc[preflight_df["selected"], "record_id"]
if (
    len(resolved_bundles) != 64
    or (group_audit_df["status"] != "PASS").any()
    or selected_ids.duplicated().any()
):
    display(group_audit_df)
    raise RuntimeError(
        f"Only {len(resolved_bundles)} of 64 spatially valid scenes were selected. "
        "No export tasks were submitted."
    )

selected_manifest = pd.DataFrame(selected_manifest_rows)
selected_manifest.to_csv(
    "/content/drive/MyDrive/ThermoFusion_Stage10_Revised_Pilot_Manifest.csv",
    index=False,
)

print("\nBalanced replacement preflight passed: 64/64. Submitting 128 exports.")

# Static terrain predictors are identical in construction for every bundle.
dem = (
    ee.ImageCollection("COPERNICUS/DEM/GLO30")
    .select("DEM")
    .mosaic()
    .rename("elevation_m")
)
slope = ee.Terrain.slope(dem).rename("slope_deg")

# Phase 2: submit predictor and native-grid thermal exports.
task_rows = []
for export_number, bundle in enumerate(resolved_bundles, start=1):
    (
        pilot_id, row, thermal_lst, thermal_valid, projection_info,
        native_scale, s2, s1, centre,
    ) = bundle

    predictor_crs, longitude, latitude = utm_crs(centre)
    region = centre.buffer(CHIP_SIZE_M / 2).bounds()
    predictors = ee.Image.cat(s2, s1, dem, slope).toFloat().unmask(NODATA)
    thermal_export = ee.Image.cat(
        thermal_lst.unmask(NODATA).toFloat(),
        thermal_valid.unmask(0).toFloat(),
    )

    prefix = f"{pilot_id}_{row.city.lower()}_{row.thermal_sensor.lower()}"
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
        formatOptions={"cloudOptimized": True, "noData": NODATA},
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
    print(f"Submitted {export_number:02d}/64: {prefix}")

task_df = pd.DataFrame(task_rows)
task_register = Path(
    "/content/drive/MyDrive/ThermoFusion_Stage10_Pilot_Export_Tasks.csv"
)
task_df.to_csv(task_register, index=False)

print("\nTHERMOFUSION STAGE 10 PILOT EXPORT")
print(f"Spatially valid balanced bundles: {len(resolved_bundles)}/64 PASS")
print(f"Candidates tested during preflight: {len(preflight_df)}")
print(f"Submitted predictor exports: {len(task_df)}")
print(f"Submitted thermal exports: {len(task_df)}")
print(f"Total submitted tasks: {len(task_df) * 2}")
print(f"Drive folder: {DRIVE_FOLDER}")
print(f"Preflight register: {preflight_path}")
print(f"Task register: {task_register}")
display(task_df)
