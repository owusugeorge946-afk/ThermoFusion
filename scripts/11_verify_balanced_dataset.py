# =====================================================================
# ThermoFusion reproducibility code
# Source notebook cell(s): 25
# Audit status: KEEP
# Extracted without scientific-logic modification.
# Replace local Google Drive paths as described in README.md when needed.
# =====================================================================

# ThermoFusion Stage 11: verify all 64 Stage 10 pilot chip bundles.
# Run in one Google Colab cell after every Stage 10 Earth Engine task completes.

from google.colab import drive
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import rasterio
    from rasterio.warp import transform_bounds, reproject, Resampling
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "rasterio"])
    import rasterio
    from rasterio.warp import transform_bounds, reproject, Resampling


drive.mount("/content/drive")

MY_DRIVE = Path("/content/drive/MyDrive")
CHIP_DIR = MY_DRIVE / "ThermoFusion_Stage10_Pilot_Chips"
REPAIR_DIR = MY_DRIVE / "ThermoFusion_Stage10_Pilot_Chips_Repair"
TASK_REGISTER = MY_DRIVE / "ThermoFusion_Stage10_Pilot_Export_Tasks.csv"
REPAIR_REGISTER = MY_DRIVE / "ThermoFusion_Stage10_Repair_Tasks.csv"
REVISED_MANIFEST = MY_DRIVE / "ThermoFusion_Stage10_Revised_Pilot_Manifest.csv"
OUTPUT_DIR = MY_DRIVE / "ThermoFusion_Stage11_Pilot_Verification"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_BUNDLES = 64
EXPECTED_FILES = 128
NODATA = -9999.0
PREDICTOR_BANDS = (
    "s2_b2", "s2_b3", "s2_b4", "s2_b8", "s2_b11", "s2_b12",
    "ndvi", "ndbi", "ndmi", "s2_valid",
    "s1_vv_db", "s1_vh_db", "s1_vv_minus_vh_db", "s1_valid",
    "elevation_m", "slope_deg",
)
THERMAL_BANDS = ("thermal_lst_c", "thermal_valid")


for required_path in [
    CHIP_DIR, REPAIR_DIR, TASK_REGISTER, REPAIR_REGISTER, REVISED_MANIFEST
]:
    if not required_path.exists():
        raise FileNotFoundError(f"Required Stage 10 output not found: {required_path}")

base_register = pd.read_csv(TASK_REGISTER)
repair_register = pd.read_csv(REPAIR_REGISTER)
manifest = pd.read_csv(REVISED_MANIFEST)

if repair_register.empty:
    raise RuntimeError("The Stage 10 repair register contains no replacements.")
if repair_register["pilot_id"].duplicated().any():
    raise RuntimeError("The Stage 10 repair register has duplicate pilot IDs.")
if not set(repair_register["pilot_id"]).issubset(set(base_register["pilot_id"])):
    raise RuntimeError("The repair register contains an unknown pilot ID.")

base_register["chip_directory"] = str(CHIP_DIR)
repair_register["chip_directory"] = str(REPAIR_DIR)
register = pd.concat(
    [
        base_register[~base_register["pilot_id"].isin(repair_register["pilot_id"])],
        repair_register,
    ],
    ignore_index=True,
).sort_values("pilot_id").reset_index(drop=True)

register_checks = {
    "register_rows": len(register) == EXPECTED_BUNDLES,
    "unique_pilot_ids": register["pilot_id"].nunique() == EXPECTED_BUNDLES,
    "unique_record_ids": register["record_id"].nunique() == EXPECTED_BUNDLES,
    "unique_prefixes": register["prefix"].nunique() == EXPECTED_BUNDLES,
    "manifest_rows": len(manifest) == EXPECTED_BUNDLES,
    "manifest_unique_records": manifest["record_id"].nunique() == EXPECTED_BUNDLES,
    "register_manifest_match": set(register["record_id"]) == set(manifest["record_id"]),
}
if not all(register_checks.values()):
    raise RuntimeError(f"Stage 10 register/manifest integrity failed: {register_checks}")

balance = (
    register.groupby(["city", "thermal_sensor"], observed=True)
    .size().rename("bundles").reset_index()
)
balance["status"] = np.where(balance["bundles"].eq(8), "PASS", "REVIEW")
balance.to_csv(OUTPUT_DIR / "02_city_sensor_balance.csv", index=False)

all_tifs = sorted(CHIP_DIR.glob("*.tif")) + sorted(REPAIR_DIR.glob("*.tif"))
unexpected_tifs = []
expected_paths = set()
superseded_paths = set()
records = []


def matching_files(directory, pattern):
    return sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)


for repair_row in repair_register.itertuples(index=False):
    superseded_paths.update(CHIP_DIR.glob(f"{repair_row.prefix}_predictors_10m*.tif"))
    superseded_paths.update(CHIP_DIR.glob(f"{repair_row.prefix}_thermal_native*.tif"))


def overlap_fraction(predictor, thermal):
    pb = transform_bounds(
        predictor.crs, thermal.crs, *predictor.bounds, densify_pts=21
    )
    left = max(pb[0], thermal.bounds.left)
    bottom = max(pb[1], thermal.bounds.bottom)
    right = min(pb[2], thermal.bounds.right)
    top = min(pb[3], thermal.bounds.top)
    intersection = max(0.0, right - left) * max(0.0, top - bottom)
    predictor_area = max(0.0, pb[2] - pb[0]) * max(0.0, pb[3] - pb[1])
    thermal_area = max(0.0, thermal.bounds.right - thermal.bounds.left) * max(
        0.0, thermal.bounds.top - thermal.bounds.bottom
    )
    denominator = min(predictor_area, thermal_area)
    return intersection / denominator if denominator > 0 else 0.0


def description_check(descriptions, expected):
    # Earth Engine normally preserves band names. Some GDAL builds expose no
    # descriptions, so absence is recorded but does not invalidate band order.
    populated = tuple(value for value in descriptions if value)
    return True if not populated else tuple(descriptions) == tuple(expected)


for number, row in enumerate(register.itertuples(index=False), start=1):
    selected_directory = Path(row.chip_directory)
    predictor_matches = matching_files(
        selected_directory, f"{row.prefix}_predictors_10m*.tif"
    )
    thermal_matches = matching_files(
        selected_directory, f"{row.prefix}_thermal_native*.tif"
    )
    predictor_path = predictor_matches[0] if predictor_matches else None
    thermal_path = thermal_matches[0] if thermal_matches else None

    expected_paths.update(predictor_matches)
    expected_paths.update(thermal_matches)

    if predictor_path is None or thermal_path is None:
        records.append(
            {
                "pilot_id": row.pilot_id,
                "record_id": row.record_id,
                "city": row.city,
                "thermal_sensor": row.thermal_sensor,
                "predictor_copies": len(predictor_matches),
                "thermal_copies": len(thermal_matches),
                "status": "MISSING_FILE",
            }
        )
        print(f"Verified {number:02d}/64: {row.prefix} | MISSING_FILE")
        continue

    with rasterio.open(predictor_path) as predictor, rasterio.open(thermal_path) as thermal:
        predictor_data = predictor.read()
        thermal_data = thermal.read()

        s2_valid = (
            np.isfinite(predictor_data[9])
            & (predictor_data[9] != NODATA)
            & (predictor_data[9] > 0.5)
        )
        s1_valid = (
            np.isfinite(predictor_data[13])
            & (predictor_data[13] != NODATA)
            & (predictor_data[13] > 0.5)
        )
        thermal_lst = thermal_data[0].astype("float32")
        thermal_valid = (
            np.isfinite(thermal_lst)
            & (thermal_lst != NODATA)
            & (thermal_data[1] > 0.5)
        )
        temperatures = thermal_lst[thermal_valid]
        overlap = overlap_fraction(predictor, thermal)

        checks = {
            "single_predictor_check": len(predictor_matches) == 1,
            "single_thermal_check": len(thermal_matches) == 1,
            "predictor_band_check": predictor.count == 16,
            "thermal_band_check": thermal.count == 2,
            "predictor_dtype_check": all(dtype == "float32" for dtype in predictor.dtypes),
            "thermal_dtype_check": all(dtype == "float32" for dtype in thermal.dtypes),
            "predictor_description_check": description_check(
                predictor.descriptions, PREDICTOR_BANDS
            ),
            "thermal_description_check": description_check(
                thermal.descriptions, THERMAL_BANDS
            ),
            "predictor_crs_check": predictor.crs is not None,
            "thermal_crs_check": thermal.crs is not None,
            "predictor_resolution_check": (
                abs(abs(predictor.res[0]) - 10) < 0.1
                and abs(abs(predictor.res[1]) - 10) < 0.1
            ),
            "spatial_overlap_check": overlap >= 0.90,
            "s2_validity_check": s2_valid.mean() >= 0.25,
            "s1_validity_check": s1_valid.mean() >= 0.80,
            "thermal_validity_check": thermal_valid.mean() >= 0.10,
            "temperature_range_check": (
                temperatures.size > 0
                and temperatures.min() > -30
                and temperatures.max() < 80
            ),
        }
        status = "PASS" if all(checks.values()) else "REVIEW"

        records.append(
            {
                "pilot_id": row.pilot_id,
                "record_id": row.record_id,
                "city": row.city,
                "thermal_sensor": row.thermal_sensor,
                "predictor_path": str(predictor_path),
                "thermal_path": str(thermal_path),
                "predictor_copies": len(predictor_matches),
                "thermal_copies": len(thermal_matches),
                "predictor_size_mb": predictor_path.stat().st_size / 1e6,
                "thermal_size_mb": thermal_path.stat().st_size / 1e6,
                "predictor_bands": predictor.count,
                "thermal_bands": thermal.count,
                "predictor_width": predictor.width,
                "predictor_height": predictor.height,
                "thermal_width": thermal.width,
                "thermal_height": thermal.height,
                "predictor_crs": str(predictor.crs),
                "thermal_crs": str(thermal.crs),
                "predictor_xres": abs(predictor.res[0]),
                "thermal_xres": abs(thermal.res[0]),
                "overlap_fraction": overlap,
                "s2_valid_fraction": float(s2_valid.mean()),
                "s1_valid_fraction": float(s1_valid.mean()),
                "thermal_valid_fraction": float(thermal_valid.mean()),
                "thermal_valid_pixels": int(thermal_valid.sum()),
                "thermal_min_c": temperatures.min() if temperatures.size else np.nan,
                "thermal_median_c": (
                    np.median(temperatures) if temperatures.size else np.nan
                ),
                "thermal_max_c": temperatures.max() if temperatures.size else np.nan,
                "predictor_band_descriptions": "|".join(
                    value or "" for value in predictor.descriptions
                ),
                "thermal_band_descriptions": "|".join(
                    value or "" for value in thermal.descriptions
                ),
                **checks,
                "status": status,
            }
        )
    print(f"Verified {number:02d}/64: {row.prefix} | {status}")

unexpected_tifs = [
    path for path in all_tifs
    if path not in expected_paths and path not in superseded_paths
]
verification = pd.DataFrame(records)
verification.to_csv(OUTPUT_DIR / "01_pilot_bundle_verification.csv", index=False)
display(verification)

complete = verification[verification["status"].isin(["PASS", "REVIEW"])].copy()
if len(complete) != EXPECTED_BUNDLES:
    raise RuntimeError(
        f"Only {len(complete)} complete bundles were found; expected 64."
    )

# Diagnostic summary covering every bundle.
sns.set_theme(style="whitegrid", context="notebook")
plot_data = complete.copy()
plot_data["label"] = (
    plot_data["pilot_id"] + " " + plot_data["city"].str[:3]
    + "-" + plot_data["thermal_sensor"].str[:3]
)
validity = plot_data.set_index("label")[
    ["s2_valid_fraction", "s1_valid_fraction", "thermal_valid_fraction"]
] * 100
validity.columns = ["Sentinel-2", "Sentinel-1", "Thermal"]

fig, axes = plt.subplots(1, 3, figsize=(22, 14), constrained_layout=True)
sns.heatmap(
    validity, cmap="viridis", vmin=0, vmax=100, ax=axes[0],
    cbar_kws={"label": "Valid pixels (%)"}, yticklabels=True,
)
axes[0].set_title("(a) Bundle-level valid-pixel coverage")
axes[0].set_xlabel("")
axes[0].set_ylabel("")

sns.scatterplot(
    data=plot_data,
    x="thermal_valid_fraction",
    y="thermal_median_c",
    hue="city",
    style="thermal_sensor",
    s=90,
    alpha=0.8,
    ax=axes[1],
)
axes[1].set_title("(b) Thermal coverage and temperature")
axes[1].set_xlabel("Valid thermal fraction")
axes[1].set_ylabel("Median land-surface temperature (°C)")
axes[1].legend(frameon=False, fontsize=9)

balance_matrix = balance.pivot(
    index="city", columns="thermal_sensor", values="bundles"
)
sns.heatmap(
    balance_matrix, annot=True, fmt=".0f", cmap="crest", vmin=0, vmax=8,
    linewidths=0.7, cbar_kws={"label": "Verified bundles"}, ax=axes[2],
)
axes[2].set_title("(c) City–sensor balance")
axes[2].set_xlabel("")
axes[2].set_ylabel("")

figure_path = OUTPUT_DIR / "thermofusion_stage11_pilot_diagnostics.png"
fig.savefig(figure_path, dpi=800, bbox_inches="tight", facecolor="white")
plt.show()

# Worst-case visual atlas: one lowest-validity bundle per city-sensor group.
atlas_rows = []
plot_data["minimum_valid_fraction"] = plot_data[
    ["s2_valid_fraction", "s1_valid_fraction", "thermal_valid_fraction"]
].min(axis=1)
for _, group in plot_data.groupby(["city", "thermal_sensor"], observed=True):
    atlas_rows.append(group.nsmallest(1, "minimum_valid_fraction").iloc[0])

fig, axes = plt.subplots(8, 3, figsize=(15, 32), constrained_layout=True)
thermal_artists = []
for index, item in enumerate(atlas_rows):
    with rasterio.open(item["predictor_path"]) as predictor, rasterio.open(
        item["thermal_path"]
    ) as thermal:
        pdata = predictor.read()
        tdata = thermal.read()
        rgb = np.stack([pdata[2], pdata[1], pdata[0]], axis=-1).astype("float32")
        rgb_valid = (pdata[9] > 0.5) & np.isfinite(rgb).all(axis=2)
        shown_rgb = np.full_like(rgb, np.nan)
        for band in range(3):
            values = rgb[:, :, band][rgb_valid]
            if values.size:
                low, high = np.percentile(values, [2, 98])
                shown_rgb[:, :, band] = np.clip(
                    (rgb[:, :, band] - low) / max(high - low, 1e-6), 0, 1
                )
        shown_rgb[~rgb_valid] = np.nan

        lst = tdata[0].astype("float32")
        native_valid = (
            (tdata[1] > 0.5) & np.isfinite(lst) & (lst != NODATA)
        )
        displayed_lst = np.full(
            (predictor.height, predictor.width), np.nan, dtype="float32"
        )
        displayed_valid = np.zeros(
            (predictor.height, predictor.width), dtype="uint8"
        )
        reproject(
            source=lst,
            destination=displayed_lst,
            src_transform=thermal.transform,
            src_crs=thermal.crs,
            src_nodata=NODATA,
            dst_transform=predictor.transform,
            dst_crs=predictor.crs,
            dst_nodata=np.nan,
            resampling=Resampling.nearest,
        )
        reproject(
            source=native_valid.astype("uint8"),
            destination=displayed_valid,
            src_transform=thermal.transform,
            src_crs=thermal.crs,
            src_nodata=0,
            dst_transform=predictor.transform,
            dst_crs=predictor.crs,
            dst_nodata=0,
            resampling=Resampling.nearest,
        )
        displayed_lst[displayed_valid == 0] = np.nan

    axes[index, 0].imshow(shown_rgb)
    artist = axes[index, 1].imshow(displayed_lst, cmap="inferno")
    thermal_artists.append(artist)
    axes[index, 2].imshow(displayed_valid, cmap="gray", vmin=0, vmax=1)
    axes[index, 0].set_ylabel(
        f"{item['city']}-{item['thermal_sensor']}", fontweight="bold"
    )
    for axis in axes[index]:
        axis.set_xticks([])
        axis.set_yticks([])

axes[0, 0].set_title("Sentinel-2 RGB")
axes[0, 1].set_title("Thermal LST on predictor grid")
axes[0, 2].set_title("Thermal-validity footprint")
atlas_path = OUTPUT_DIR / "thermofusion_stage11_worst_case_atlas.png"
fig.savefig(atlas_path, dpi=600, bbox_inches="tight", facecolor="white")
plt.show()

passed = int(verification["status"].eq("PASS").sum())
selected_file_count = len(expected_paths)
expected_physical_files = EXPECTED_FILES + 2 * len(repair_register)
file_count_check = (
    selected_file_count == EXPECTED_FILES
    and len(all_tifs) == expected_physical_files
)
repair_replacement_check = (
    len(superseded_paths) == 2 * len(repair_register)
    and len(repair_register) == 3
)
unexpected_check = len(unexpected_tifs) == 0
balance_check = len(balance) == 8 and balance["bundles"].eq(8).all()
overall_pass = (
    passed == EXPECTED_BUNDLES
    and file_count_check
    and repair_replacement_check
    and unexpected_check
    and balance_check
    and all(register_checks.values())
)

verdict = [
    "THERMOFUSION STAGE 11 PILOT-BUNDLE VERIFICATION",
    f"Selected GeoTIFF files: {selected_file_count}/{EXPECTED_FILES}",
    f"Repair bundles incorporated: {len(repair_register)}/3",
    f"Superseded rejected files retained: {len(superseded_paths)}/6",
    f"Physical GeoTIFF files across original and repair folders: "
    f"{len(all_tifs)}/{expected_physical_files}",
    f"Complete predictor-thermal bundles: {len(complete)}/{EXPECTED_BUNDLES}",
    f"Bundles passing every check: {passed}/{EXPECTED_BUNDLES}",
    f"Unique selected records: {register['record_id'].nunique()}/{EXPECTED_BUNDLES}",
    f"City-sensor groups with 8 bundles: {balance['bundles'].eq(8).sum()}/8",
    f"Predictors with 16 Float32 bands: "
    f"{(verification['predictor_band_check'] & verification['predictor_dtype_check']).sum()}/64",
    f"Thermal targets with 2 Float32 bands: "
    f"{(verification['thermal_band_check'] & verification['thermal_dtype_check']).sum()}/64",
    f"Bundles with >=90% spatial overlap: "
    f"{verification['spatial_overlap_check'].sum()}/64",
    f"Bundles with >=25% valid Sentinel-2 pixels: "
    f"{verification['s2_validity_check'].sum()}/64",
    f"Bundles with >=80% valid Sentinel-1 pixels: "
    f"{verification['s1_validity_check'].sum()}/64",
    f"Bundles with >=10% valid thermal pixels: "
    f"{verification['thermal_validity_check'].sum()}/64",
    f"Bundles with plausible temperature ranges: "
    f"{verification['temperature_range_check'].sum()}/64",
    f"Unexpected GeoTIFF files across Stage 10 folders: {len(unexpected_tifs)}",
    "Overall pilot verification: PASS" if overall_pass
    else "Overall pilot verification: REVIEW REQUIRED",
]
(OUTPUT_DIR / "03_pilot_verification_verdict.txt").write_text(
    "\n".join(verdict), encoding="utf-8"
)
if unexpected_tifs:
    pd.DataFrame({"unexpected_path": [str(path) for path in unexpected_tifs]}).to_csv(
        OUTPUT_DIR / "04_unexpected_files.csv", index=False
    )

zip_path = shutil.make_archive(
    "/content/drive/MyDrive/ThermoFusion_Stage11_Pilot_Verification",
    "zip",
    root_dir=OUTPUT_DIR,
)
print("\n" + "\n".join(verdict))
print(f"\nOutput folder: {OUTPUT_DIR}")
print(f"ZIP package: {zip_path}")
