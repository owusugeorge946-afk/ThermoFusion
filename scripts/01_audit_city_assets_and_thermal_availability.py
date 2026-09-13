# =====================================================================
# ThermoFusion reproducibility code
# Source notebook cell(s): 3
# Audit status: KEEP
# Extracted without scientific-logic modification.
# Replace local Google Drive paths as described in README.md when needed.
# =====================================================================

"""ThermoFusion Stage 01: verify city assets and audit thermal-scene availability.

Designed for Google Colab with the Earth Engine Python API.
"""

import ee
import pandas as pd


PROJECT_ID = "nana213"
START_DATE = "2018-07-01"
END_DATE = "2026-01-01"  # Exclusive

CITY_ASSETS = {
    "Accra": "projects/nana213/assets/Accra",
    "Lagos": "projects/nana213/assets/lagos",
    "Abidjan": "projects/nana213/assets/Abijan",
    "Freetown": "projects/nana213/assets/Freetown",
}

LANDSAT_8 = "LANDSAT/LC08/C02/T1_L2"
LANDSAT_9 = "LANDSAT/LC09/C02/T1_L2"
# Earth Engine catalogue identifier. As of September 2026, this collection
# contains only Los Angeles tiles; global ECOSTRESS data must be obtained
# separately from NASA LP DAAC/AppEEARS.
ECOSTRESS = "NASA/ECOSTRESS/L2T_LSTE/V2"


ee.Authenticate()
ee.Initialize(project=PROJECT_ID)
print(f"Connected to Earth Engine project: {PROJECT_ID}")


city_geometries = {}
asset_records = []

for city, asset_id in CITY_ASSETS.items():
    try:
        collection = ee.FeatureCollection(asset_id)
        geometry = collection.geometry()
        feature_count = collection.size().getInfo()
        area_km2 = geometry.area(maxError=10).divide(1e6).getInfo()
        city_geometries[city] = geometry
        asset_records.append(
            {
                "City": city,
                "Asset_ID": asset_id,
                "Features": feature_count,
                "Area_km2": round(area_km2, 2),
                "Status": "Valid",
            }
        )
        print(f"Valid: {city}: {feature_count} feature(s), {area_km2:,.2f} km2")
    except Exception as error:
        asset_records.append(
            {
                "City": city,
                "Asset_ID": asset_id,
                "Features": None,
                "Area_km2": None,
                "Status": f"Error: {error}",
            }
        )
        print(f"Error: {city}: {error}")

asset_df = pd.DataFrame(asset_records)
display(asset_df)

invalid = asset_df[asset_df["Status"] != "Valid"]
if not invalid.empty:
    raise RuntimeError("At least one city asset is invalid. Fix it before continuing.")


availability_records = []

for city, geometry in city_geometries.items():
    common_landsat_filters = (
        ee.Filter.And(
            ee.Filter.eq("PROCESSING_LEVEL", "L2SP"),
            ee.Filter.lt("CLOUD_COVER", 70),
        )
    )

    l8_count = (
        ee.ImageCollection(LANDSAT_8)
        .filterBounds(geometry)
        .filterDate(START_DATE, END_DATE)
        .filter(common_landsat_filters)
        .size()
        .getInfo()
    )

    l9_count = (
        ee.ImageCollection(LANDSAT_9)
        .filterBounds(geometry)
        .filterDate(START_DATE, END_DATE)
        .filter(common_landsat_filters)
        .size()
        .getInfo()
    )

    ecostress_count = (
        ee.ImageCollection(ECOSTRESS)
        .filterBounds(geometry)
        .filterDate(START_DATE, END_DATE)
        .size()
        .getInfo()
    )

    availability_records.append(
        {
            "City": city,
            "Landsat_8": l8_count,
            "Landsat_9": l9_count,
            "Landsat_Total": l8_count + l9_count,
            "ECOSTRESS_Total": ecostress_count,
        }
    )
    print(
        f"{city}: Landsat 8={l8_count}, Landsat 9={l9_count}, "
        f"ECOSTRESS={ecostress_count}"
    )

availability_df = pd.DataFrame(availability_records)
display(availability_df)

asset_df.to_csv("thermofusion_city_asset_audit.csv", index=False)
availability_df.to_csv("thermofusion_scene_availability.csv", index=False)

print("Stage 01 completed.")
print("Saved: thermofusion_city_asset_audit.csv")
print("Saved: thermofusion_scene_availability.csv")
