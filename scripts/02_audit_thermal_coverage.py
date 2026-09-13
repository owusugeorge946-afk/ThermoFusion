# =====================================================================
# ThermoFusion reproducibility code
# Source notebook cell(s): 4
# Audit status: KEEP
# Extracted without scientific-logic modification.
# Replace local Google Drive paths as described in README.md when needed.
# =====================================================================

# ThermoFusion Stage 02: pixel-level thermal coverage audit.
# Run in Google Colab. The script creates eight Google Drive CSV export tasks:
# one Landsat and one ECOSTRESS audit for each study city.

import ee


PROJECT_ID = "nana213"
START_DATE = "2018-07-01"
END_DATE = "2026-01-01"  # Exclusive
AUDIT_SCALE_M = 300
DRIVE_FOLDER = "ThermoFusion_Stage02_Audit"

CITY_ASSETS = {
    "Accra": "projects/nana213/assets/Accra",
    "Lagos": "projects/nana213/assets/lagos",
    "Abidjan": "projects/nana213/assets/Abijan",
    "Freetown": "projects/nana213/assets/Freetown",
}

LANDSAT_8 = "LANDSAT/LC08/C02/T1_L2"
LANDSAT_9 = "LANDSAT/LC09/C02/T1_L2"
ECOSTRESS = "NASA/ECOSTRESS/L2T_LSTE/V2"


ee.Authenticate()
ee.Initialize(project=PROJECT_ID)
print(f"Connected to Earth Engine project: {PROJECT_ID}")


def aoi_area_image(geometry):
    # Approximate AOI area represented on the audit grid.
    return ee.Number(
        ee.Image.pixelArea()
        .reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=geometry,
            scale=AUDIT_SCALE_M,
            maxPixels=1e9,
            bestEffort=True,
            tileScale=4,
        )
        .get("area")
    )


def landsat_valid_mask(image):
    qa = image.select("QA_PIXEL")
    clear = (
        qa.bitwiseAnd(1 << 0).eq(0)  # Fill
        .And(qa.bitwiseAnd(1 << 1).eq(0))  # Dilated cloud
        .And(qa.bitwiseAnd(1 << 2).eq(0))  # Cirrus
        .And(qa.bitwiseAnd(1 << 3).eq(0))  # Cloud
        .And(qa.bitwiseAnd(1 << 4).eq(0))  # Cloud shadow
    )
    unsaturated = image.select("QA_RADSAT").eq(0)
    lst_c = image.select("ST_B10").multiply(0.00341802).add(149).subtract(273.15)
    plausible = lst_c.gt(5).And(lst_c.lt(70))
    return clear.And(unsaturated).And(plausible).rename("valid")


def ecostress_valid_mask(image):
    # Require good LST quality/error, no temporal interpolation, clear sky,
    # no outlier, cloud-free land, and a physically plausible temperature.
    qc_good = image.select("QC").bitwiseAnd(31).eq(0)
    cloud_free = image.select("cloud").bitwiseAnd(1).eq(0)
    land = image.select("water").bitwiseAnd(1).eq(0)
    lst_k = image.select("LST")
    plausible = lst_k.gt(250).And(lst_k.lt(350))
    return qc_good.And(cloud_free).And(land).And(plausible).rename("valid")


def image_audit_feature(image, geometry, city, sensor, total_area_m2, mask_fn):
    valid_area = ee.Number(
        ee.Image.pixelArea()
        .updateMask(mask_fn(image))
        .reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=geometry,
            scale=AUDIT_SCALE_M,
            maxPixels=1e9,
            bestEffort=True,
            tileScale=4,
        )
        .get("area", 0)
    )

    valid_fraction = valid_area.divide(total_area_m2).multiply(100)
    date = ee.Date(image.get("system:time_start"))

    return ee.Feature(
        None,
        {
            "city": city,
            "sensor": sensor,
            "scene_id": image.id(),
            "date": date.format("YYYY-MM-dd"),
            "year": date.get("year"),
            "month": date.get("month"),
            "valid_area_km2": valid_area.divide(1e6),
            "valid_fraction_pct": valid_fraction,
            "catalog_cloud_cover_pct": image.get("CLOUD_COVER"),
        },
    )


def export_audit(collection, geometry, city, sensor, mask_fn):
    total_area_m2 = aoi_area_image(geometry)
    image_list = collection.toList(collection.size())
    audit = ee.FeatureCollection(
        image_list.map(
            lambda item: image_audit_feature(
                ee.Image(item), geometry, city, sensor, total_area_m2, mask_fn
            )
        )
    )

    safe_sensor = sensor.lower().replace(" ", "_")
    description = f"thermofusion_{city.lower()}_{safe_sensor}_coverage"

    task = ee.batch.Export.table.toDrive(
        collection=audit,
        description=description,
        folder=DRIVE_FOLDER,
        fileNamePrefix=description,
        fileFormat="CSV",
        selectors=[
            "city",
            "sensor",
            "scene_id",
            "date",
            "year",
            "month",
            "valid_area_km2",
            "valid_fraction_pct",
            "catalog_cloud_cover_pct",
        ],
    )
    task.start()
    print(f"Started: {description} | task ID: {task.id}")


for city, asset_id in CITY_ASSETS.items():
    geometry = ee.FeatureCollection(asset_id).geometry()

    common_filter = ee.Filter.And(
        ee.Filter.eq("PROCESSING_LEVEL", "L2SP"),
        ee.Filter.lt("CLOUD_COVER", 70),
    )

    landsat = (
        ee.ImageCollection(LANDSAT_8)
        .filterBounds(geometry)
        .filterDate(START_DATE, END_DATE)
        .filter(common_filter)
        .merge(
            ee.ImageCollection(LANDSAT_9)
            .filterBounds(geometry)
            .filterDate(START_DATE, END_DATE)
            .filter(common_filter)
        )
        .sort("system:time_start")
    )

    ecostress = (
        ee.ImageCollection(ECOSTRESS)
        .filterBounds(geometry)
        .filterDate(START_DATE, END_DATE)
        .sort("system:time_start")
    )

    export_audit(landsat, geometry, city, "Landsat", landsat_valid_mask)
    export_audit(ecostress, geometry, city, "ECOSTRESS", ecostress_valid_mask)


print("\nStage 02 submitted eight export tasks.")
print(f"Google Drive folder: {DRIVE_FOLDER}")
print("Monitor the tasks in Earth Engine or Colab before continuing.")
