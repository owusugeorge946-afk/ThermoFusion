# =====================================================================
# ThermoFusion reproducibility code
# Source notebook cell(s): 18
# Audit status: KEEP
# Extracted without scientific-logic modification.
# Replace local Google Drive paths as described in README.md when needed.
# =====================================================================

# ThermoFusion Stage 08: export eight alignment-test chip bundles.
# Each bundle contains a 10 m predictor chip and a native-grid thermal target.

from google.colab import drive
from pathlib import Path
import math
import pandas as pd
import ee


PROJECT_ID = "nana213"
PILOT_PATH = Path(
    "/content/drive/MyDrive/ThermoFusion_Stage07_Pilot/"
    "01_balanced_pilot_manifest.csv"
)
DRIVE_FOLDER = "ThermoFusion_Stage08_Alignment_Chips"
CHIP_SIZE_M = 2560
EROSION_RADIUS_M = 640
NODATA = -9999
EXPORT_PREDICTORS = True  # Regenerate bundles using joint-valid chip centres.

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


drive.mount("/content/drive")
if not PILOT_PATH.exists():
    raise FileNotFoundError(f"Stage 07 pilot manifest not found: {PILOT_PATH}")

pilot = pd.read_csv(PILOT_PATH)
pilot["pilot_quality_score"] = pd.to_numeric(
    pilot["pilot_quality_score"], errors="raise"
)

# Candidate records remain ordered by quality. The script will test them until
# it finds a genuine three-source spatial intersection for each group.
pilot = pilot.sort_values(
    ["city", "thermal_sensor", "pilot_quality_score"],
    ascending=[True, True, False],
).reset_index(drop=True)

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


task_rows = []
candidate_audit = []

for (city, thermal_sensor), group in pilot.groupby(
    ["city", "thermal_sensor"], sort=True
):
    city_geometry = ee.FeatureCollection(CITY_ASSETS[city]).geometry()
    selected_bundle = None

    for candidate_rank, row in enumerate(group.itertuples(index=False), start=1):
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
        candidate_audit.append(
            {
                "city": city,
                "thermal_sensor": thermal_sensor,
                "pilot_id": row.pilot_id,
                "candidate_rank": candidate_rank,
                "joint_valid_centre_found": centre is not None,
            }
        )
        print(
            f"Tested {city} {thermal_sensor} candidate {candidate_rank}: "
            f"{row.pilot_id} | joint centre: {centre is not None}"
        )
        if centre is not None:
            selected_bundle = (
                row,
                thermal_lst,
                thermal_valid,
                projection_info,
                native_scale,
                s2,
                s1,
                centre,
                candidate_rank,
            )
            break

    if selected_bundle is None:
        raise RuntimeError(
            f"No pilot candidate for {city} {thermal_sensor} contained a "
            "jointly valid thermal, Sentinel-2 and Sentinel-1 location."
        )

    (
        row,
        thermal_lst,
        thermal_valid,
        projection_info,
        native_scale,
        s2,
        s1,
        centre,
        candidate_rank,
    ) = selected_bundle

    predictor_crs, longitude, latitude = utm_crs(centre)
    region = centre.buffer(CHIP_SIZE_M / 2).bounds()

    dem = (
        ee.ImageCollection("COPERNICUS/DEM/GLO30")
        .select("DEM")
        .mosaic()
        .rename("elevation_m")
    )
    slope = ee.Terrain.slope(dem).rename("slope_deg")

    predictors = ee.Image.cat(s2, s1, dem, slope).toFloat().unmask(NODATA)
    thermal_export = ee.Image.cat(
        thermal_lst.unmask(NODATA).toFloat(),
        thermal_valid.unmask(0).toFloat(),
    )

    prefix = f"{row.pilot_id}_{row.city.lower()}_{row.thermal_sensor.lower()}"

    predictor_task_id = "already_completed"
    if EXPORT_PREDICTORS:
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
        predictor_task.start()
        predictor_task_id = predictor_task.id

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
    thermal_task.start()

    task_rows.append(
        {
            "pilot_id": row.pilot_id,
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
            "selected_candidate_rank": candidate_rank,
            "s2_composite_window_days": S2_WINDOW_DAYS,
            "s1_composite_window_days": S1_WINDOW_DAYS,
            "predictor_task_id": predictor_task_id,
            "thermal_task_id": thermal_task.id,
            "prefix": prefix,
        }
    )
    print(f"Submitted joint-valid predictor and thermal bundle: {prefix}")


task_df = pd.DataFrame(task_rows)
pd.DataFrame(candidate_audit).to_csv(
    "/content/drive/MyDrive/ThermoFusion_Stage08_Candidate_Audit.csv",
    index=False,
)
task_register = Path(
    "/content/drive/MyDrive/ThermoFusion_Stage08_Alignment_Task_Register.csv"
)
task_df.to_csv(task_register, index=False)

submitted_count = len(task_df) * (1 + int(EXPORT_PREDICTORS))
print(f"\nSubmitted {submitted_count} alignment-test exports.")
print(f"Drive folder: {DRIVE_FOLDER}")
print(f"Task register: {task_register}")
display(task_df)
