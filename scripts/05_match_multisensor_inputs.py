# =====================================================================
# ThermoFusion reproducibility code
# Source notebook cell(s): 12
# Audit status: KEEP
# Extracted without scientific-logic modification.
# Replace local Google Drive paths as described in README.md when needed.
# =====================================================================

# ThermoFusion Stage 05: match selected thermal scenes to S2, S1 and ERA5-Land.
# Run in one Google Colab cell after Stage 04 completes.

from google.colab import drive
from pathlib import Path
import pandas as pd
import ee


PROJECT_ID = "nana213"
MANIFEST = Path(
    "/content/drive/MyDrive/ThermoFusion_Stage04_Selection/"
    "02_selected_scene_manifest.csv"
)
DRIVE_FOLDER = "ThermoFusion_Stage05_Matches"
S2_WINDOW_DAYS = 10
S1_WINDOW_DAYS = 12
ERA5_WINDOW_HOURS = 2

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
ERA5_LAND = "ECMWF/ERA5_LAND/HOURLY"


drive.mount("/content/drive")
if not MANIFEST.exists():
    raise FileNotFoundError(f"Stage 04 manifest not found: {MANIFEST}")

manifest = pd.read_csv(MANIFEST)
manifest["date"] = pd.to_datetime(manifest["date"], errors="raise")
manifest["system_index"] = manifest["scene_id"].astype(str).str.split("/").str[-1]

# ImageCollection.merge prefixes exported image IDs with 1_ or 2_. Remove
# those prefixes only for Landsat so the scenes can be recovered from their
# original source collections.
manifest["source_index"] = manifest["system_index"]
landsat_rows = manifest["sensor"].eq("Landsat")
manifest.loc[landsat_rows, "source_index"] = (
    manifest.loc[landsat_rows, "system_index"]
    .astype(str)
    .str.replace(r"^(?:(?:1|2)_)+", "", regex=True)
)

required = {
    "record_id",
    "city",
    "sensor",
    "scene_id",
    "system_index",
    "source_index",
    "valid_fraction_pct",
    "high_coverage_evaluation",
    "temporal_role",
}
missing = required.difference(manifest.columns)
if missing:
    raise RuntimeError(f"Manifest is missing columns: {sorted(missing)}")

ee.Authenticate()
ee.Initialize(project=PROJECT_ID)
print(f"Connected to Earth Engine project: {PROJECT_ID}")
print(f"Selected thermal records: {len(manifest)}")


def filtered_thermal_collection(sensor, indices, geometry):
    index_filter = ee.Filter.inList("system:index", indices)
    if sensor == "Landsat":
        l8 = (
            ee.ImageCollection(LANDSAT_8)
            .filterBounds(geometry)
            .filter(index_filter)
            .map(lambda image: image.set("source_index", image.get("system:index")))
        )
        l9 = (
            ee.ImageCollection(LANDSAT_9)
            .filterBounds(geometry)
            .filter(index_filter)
            .map(lambda image: image.set("source_index", image.get("system:index")))
        )
        return l8.merge(l9)
    return (
        ee.ImageCollection(ECOSTRESS)
        .filterBounds(geometry)
        .filter(index_filter)
        .map(lambda image: image.set("source_index", image.get("system:index")))
    )


def nearest_join(primary, secondary, match_key, difference_key, difference_ms):
    condition = ee.Filter.maxDifference(
        difference=difference_ms,
        leftField="system:time_start",
        rightField="system:time_start",
    )
    join = ee.Join.saveBest(
        matchKey=match_key,
        measureKey=difference_key,
        outer=True,
    )
    return ee.ImageCollection(join.apply(primary, secondary, condition))


def null_safe_image(parent, property_name):
    value = parent.get(property_name)
    exists = ee.Algorithms.If(
        ee.Algorithms.IsEqual(value, None), False, True
    )
    fallback = ee.Image.constant(0).set("match_missing", 1)
    return ee.Image(ee.Algorithms.If(exists, value, fallback)), exists


def safe_time_difference_hours(parent, matched_image, match_exists):
    # Derive the offset from acquisition timestamps instead of relying on the
    # join measure property, which can become null after nested outer joins.
    parent_time = ee.Number(parent.get("system:time_start"))
    matched_time = ee.Number(
        ee.Algorithms.If(
            match_exists,
            matched_image.get("system:time_start"),
            parent_time,
        )
    )
    difference = matched_time.subtract(parent_time).abs().divide(3600000)
    return ee.Algorithms.If(match_exists, difference, None)


def joined_image_to_feature(image):
    image = ee.Image(image)
    s2, has_s2 = null_safe_image(image, "s2_match")
    s1, has_s1 = null_safe_image(image, "s1_match")
    era5, has_era5 = null_safe_image(image, "era5_match")

    acquisition = ee.Date(image.get("system:time_start"))

    return ee.Feature(
        None,
        {
            "record_id": image.get("record_id"),
            "city": image.get("city"),
            "thermal_sensor": image.get("thermal_sensor"),
            "thermal_scene_id": image.id(),
            "thermal_datetime_utc": acquisition.format("YYYY-MM-dd HH:mm:ss"),
            "valid_fraction_pct": image.get("valid_fraction_pct"),
            "high_coverage_evaluation": image.get("high_coverage_evaluation"),
            "temporal_role": image.get("temporal_role"),
            "s2_found": ee.Number(ee.Algorithms.If(has_s2, 1, 0)),
            "s2_scene_id": ee.Algorithms.If(has_s2, s2.id(), None),
            "s2_time_difference_hours": safe_time_difference_hours(
                image, s2, has_s2
            ),
            "s2_cloudy_pixel_percentage": ee.Algorithms.If(
                has_s2, s2.get("CLOUDY_PIXEL_PERCENTAGE"), None
            ),
            "s2_solar_azimuth_deg": ee.Algorithms.If(
                has_s2, s2.get("MEAN_SOLAR_AZIMUTH_ANGLE"), None
            ),
            "s2_solar_zenith_deg": ee.Algorithms.If(
                has_s2, s2.get("MEAN_SOLAR_ZENITH_ANGLE"), None
            ),
            "s1_found": ee.Number(ee.Algorithms.If(has_s1, 1, 0)),
            "s1_scene_id": ee.Algorithms.If(has_s1, s1.id(), None),
            "s1_time_difference_hours": safe_time_difference_hours(
                image, s1, has_s1
            ),
            "s1_orbit_pass": ee.Algorithms.If(
                has_s1, s1.get("orbitProperties_pass"), None
            ),
            "s1_relative_orbit": ee.Algorithms.If(
                has_s1, s1.get("relativeOrbitNumber_start"), None
            ),
            "era5_found": ee.Number(ee.Algorithms.If(has_era5, 1, 0)),
            "era5_scene_id": ee.Algorithms.If(has_era5, era5.id(), None),
            "era5_time_difference_hours": safe_time_difference_hours(
                image, era5, has_era5
            ),
        },
    )


task_records = []

for (city, sensor), group in manifest.groupby(["city", "sensor"], sort=True):
    geometry = ee.FeatureCollection(CITY_ASSETS[city]).geometry()
    indices = group["source_index"].drop_duplicates().tolist()

    record_lookup = dict(zip(group["source_index"], group["record_id"]))
    coverage_lookup = dict(
        zip(group["source_index"], group["valid_fraction_pct"].astype(float))
    )
    evaluation_lookup = dict(
        zip(
            group["source_index"],
            group["high_coverage_evaluation"].astype(int),
        )
    )
    role_lookup = dict(zip(group["source_index"], group["temporal_role"]))

    record_dictionary = ee.Dictionary(record_lookup)
    coverage_dictionary = ee.Dictionary(coverage_lookup)
    evaluation_dictionary = ee.Dictionary(evaluation_lookup)
    role_dictionary = ee.Dictionary(role_lookup)

    recovered_thermal = filtered_thermal_collection(sensor, indices, geometry)
    recovered_count = recovered_thermal.size().getInfo()
    expected_count = len(indices)

    if recovered_count != expected_count:
        raise RuntimeError(
            f"Preflight failed for {city} {sensor}: recovered "
            f"{recovered_count} of {expected_count} unique source scenes. "
            "No new tasks were submitted after this failure."
        )

    print(
        f"Preflight passed: {city} {sensor}: "
        f"{recovered_count}/{expected_count} scenes recovered"
    )

    thermal = recovered_thermal.map(
        lambda image: image.set(
            {
                "record_id": record_dictionary.get(image.get("source_index")),
                "city": city,
                "thermal_sensor": sensor,
                "valid_fraction_pct": coverage_dictionary.get(
                    image.get("source_index")
                ),
                "high_coverage_evaluation": evaluation_dictionary.get(
                    image.get("source_index")
                ),
                "temporal_role": role_dictionary.get(image.get("source_index")),
            }
        )
    )

    minimum_date = group["date"].min() - pd.Timedelta(days=S1_WINDOW_DAYS)
    maximum_date = group["date"].max() + pd.Timedelta(days=S1_WINDOW_DAYS + 1)
    start = minimum_date.strftime("%Y-%m-%d")
    end = maximum_date.strftime("%Y-%m-%d")

    s2 = (
        ee.ImageCollection(SENTINEL_2)
        .filterBounds(geometry)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 90))
    )

    s1 = (
        ee.ImageCollection(SENTINEL_1)
        .filterBounds(geometry)
        .filterDate(start, end)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .filter(ee.Filter.eq("resolution_meters", 10))
    )

    era5 = (
        ee.ImageCollection(ERA5_LAND)
        .filterBounds(geometry)
        .filterDate(start, end)
    )

    joined = nearest_join(
        thermal,
        s2,
        "s2_match",
        "s2_time_difference_ms",
        S2_WINDOW_DAYS * 24 * 60 * 60 * 1000,
    )
    joined = nearest_join(
        joined,
        s1,
        "s1_match",
        "s1_time_difference_ms",
        S1_WINDOW_DAYS * 24 * 60 * 60 * 1000,
    )
    joined = nearest_join(
        joined,
        era5,
        "era5_match",
        "era5_time_difference_ms",
        ERA5_WINDOW_HOURS * 60 * 60 * 1000,
    )

    joined_list = joined.toList(joined.size())
    output = ee.FeatureCollection(
        joined_list.map(lambda item: joined_image_to_feature(ee.Image(item)))
    )

    description = f"thermofusion_{city.lower()}_{sensor.lower()}_matches"
    export_task = ee.batch.Export.table.toDrive(
        collection=output,
        description=description,
        folder=DRIVE_FOLDER,
        fileNamePrefix=description,
        fileFormat="CSV",
    )
    export_task.start()

    task_records.append(
        {
            "city": city,
            "sensor": sensor,
            "manifest_records": len(group),
            "unique_scene_indices": len(indices),
            "task_id": export_task.id,
            "description": description,
        }
    )
    print(
        f"Started {description}: {len(group)} records | task {export_task.id}"
    )


task_df = pd.DataFrame(task_records)
task_path = Path(
    "/content/drive/MyDrive/ThermoFusion_Stage05_Matching_Tasks.csv"
)
task_df.to_csv(task_path, index=False)

print("\nStage 05 submitted eight temporal-matching exports.")
print(f"Drive output folder: {DRIVE_FOLDER}")
print(f"Task register: {task_path}")
display(task_df)
