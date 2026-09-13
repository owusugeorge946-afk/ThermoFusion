# =====================================================================
# ThermoFusion reproducibility code
# Source notebook cell(s): 19
# Audit status: KEEP
# Extracted without scientific-logic modification.
# Replace local Google Drive paths as described in README.md when needed.
# =====================================================================

# ThermoFusion Stage 09: verify eight predictor/thermal alignment bundles.

from google.colab import drive
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import rasterio
    from rasterio.warp import transform_bounds, reproject, Resampling
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "rasterio"])
    import rasterio
    from rasterio.warp import transform_bounds, reproject, Resampling

drive.mount("/content/drive")

MY_DRIVE = Path("/content/drive/MyDrive")
TASK_REGISTER = MY_DRIVE / "ThermoFusion_Stage08_Alignment_Task_Register.csv"
OUTPUT_DIR = MY_DRIVE / "ThermoFusion_Stage09_Alignment_Verification"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
NODATA = -9999

if not TASK_REGISTER.exists():
    raise FileNotFoundError(f"Task register not found: {TASK_REGISTER}")

register = pd.read_csv(TASK_REGISTER)
if len(register) != 8:
    raise RuntimeError(f"Expected 8 task-register rows but found {len(register)}.")

def newest_match(pattern):
    paths = sorted(MY_DRIVE.rglob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return (paths[0] if paths else None), len(paths)

def make_rgb(data):
    rgb = np.stack([data[2], data[1], data[0]], axis=-1).astype("float32")
    valid = (data[9] > 0.5) & np.isfinite(rgb).all(axis=2) & (rgb != NODATA).all(axis=2)
    output = np.zeros_like(rgb)
    for band in range(3):
        values = rgb[:, :, band][valid]
        if values.size:
            low, high = np.percentile(values, [2, 98])
            output[:, :, band] = np.clip((rgb[:, :, band] - low) / max(high - low, 1e-6), 0, 1)
    output[~valid] = np.nan
    return output

records = []
visuals = []

for row in register.itertuples(index=False):
    predictor_path, predictor_copies = newest_match(f"{row.prefix}_predictors_10m*.tif")
    thermal_path, thermal_copies = newest_match(f"{row.prefix}_thermal_native*.tif")
    if predictor_path is None or thermal_path is None:
        records.append({"pilot_id": row.pilot_id, "city": row.city, "thermal_sensor": row.thermal_sensor, "status": "MISSING_FILE"})
        continue

    with rasterio.open(predictor_path) as predictor, rasterio.open(thermal_path) as thermal:
        predictor_data = predictor.read()
        thermal_data = thermal.read()
        thermal_lst = thermal_data[0].astype("float32")
        thermal_valid = (thermal_data[1] > 0.5) & np.isfinite(thermal_lst) & (thermal_lst != NODATA)
        temperatures = thermal_lst[thermal_valid]
        s2_valid_fraction = float((predictor_data[9] > 0.5).mean())
        s1_valid_fraction = float((predictor_data[13] > 0.5).mean())

        transformed = transform_bounds(predictor.crs, thermal.crs, *predictor.bounds, densify_pts=21)
        overlap = (
            min(transformed[2], thermal.bounds.right) > max(transformed[0], thermal.bounds.left)
            and min(transformed[3], thermal.bounds.top) > max(transformed[1], thermal.bounds.bottom)
        )
        checks = {
            "predictor_band_check": predictor.count == 16,
            "thermal_band_check": thermal.count == 2,
            "predictor_crs_check": predictor.crs is not None,
            "thermal_crs_check": thermal.crs is not None,
            "predictor_resolution_check": abs(abs(predictor.res[0]) - 10) < 0.1,
            "spatial_overlap_check": overlap,
            "thermal_valid_check": temperatures.size > 0,
            "s2_validity_check": s2_valid_fraction >= 0.25,
            "s1_validity_check": s1_valid_fraction >= 0.80,
            "thermal_fraction_check": thermal_valid.mean() >= 0.10,
            "temperature_range_check": temperatures.size > 0 and temperatures.min() > -30 and temperatures.max() < 80,
        }

        records.append({
            "pilot_id": row.pilot_id, "record_id": row.record_id, "city": row.city,
            "thermal_sensor": row.thermal_sensor, "predictor_path": str(predictor_path),
            "thermal_path": str(thermal_path), "predictor_copies": predictor_copies,
            "thermal_copies": thermal_copies, "predictor_bands": predictor.count,
            "thermal_bands": thermal.count, "predictor_width": predictor.width,
            "predictor_height": predictor.height, "thermal_width": thermal.width,
            "thermal_height": thermal.height, "predictor_crs": str(predictor.crs),
            "thermal_crs": str(thermal.crs), "predictor_xres": abs(predictor.res[0]),
            "thermal_xres": abs(thermal.res[0]), "s2_valid_fraction": s2_valid_fraction,
            "s1_valid_fraction": s1_valid_fraction, "thermal_valid_fraction": thermal_valid.mean(),
            "thermal_min_c": temperatures.min() if temperatures.size else np.nan,
            "thermal_median_c": np.median(temperatures) if temperatures.size else np.nan,
            "thermal_max_c": temperatures.max() if temperatures.size else np.nan,
            **checks, "status": "PASS" if all(checks.values()) else "REVIEW"
        })

        displayed_thermal = np.full((predictor.height, predictor.width), np.nan, dtype="float32")
        reproject(
            source=thermal_lst, destination=displayed_thermal,
            src_transform=thermal.transform, src_crs=thermal.crs, src_nodata=NODATA,
            dst_transform=predictor.transform, dst_crs=predictor.crs, dst_nodata=np.nan,
            resampling=Resampling.nearest,
        )
        visuals.append({
            "label": f"{row.city}-{row.thermal_sensor}", "rgb": make_rgb(predictor_data),
            "thermal": displayed_thermal,
            "valid": np.isfinite(displayed_thermal) & (displayed_thermal != NODATA),
        })

verification = pd.DataFrame(records)
verification.to_csv(OUTPUT_DIR / "01_alignment_verification.csv", index=False)
display(verification)

if len(visuals) != 8:
    raise RuntimeError(f"Only {len(visuals)} complete bundles were found; expected 8.")

all_temperatures = np.concatenate([v["thermal"][v["valid"]] for v in visuals])
vmin, vmax = np.percentile(all_temperatures, [2, 98])
fig, axes = plt.subplots(8, 3, figsize=(15, 32), constrained_layout=True)
for index, item in enumerate(visuals):
    axes[index, 0].imshow(item["rgb"])
    axes[index, 0].set_ylabel(item["label"], fontsize=11, fontweight="bold")
    shown = np.ma.masked_where(~item["valid"], item["thermal"])
    thermal_artist = axes[index, 1].imshow(shown, cmap="inferno", vmin=vmin, vmax=vmax)
    axes[index, 2].imshow(item["valid"], cmap="gray", vmin=0, vmax=1)
    for axis in axes[index]:
        axis.set_xticks([]); axis.set_yticks([])
axes[0, 0].set_title("Sentinel-2 RGB predictor grid")
axes[0, 1].set_title("Thermal LST displayed on predictor grid")
axes[0, 2].set_title("Thermal-validity footprint")
colourbar = fig.colorbar(thermal_artist, ax=axes[:, 1], fraction=0.02, pad=0.02)
colourbar.set_label("Land-surface temperature (degrees C)")
fig.savefig(OUTPUT_DIR / "thermofusion_stage09_alignment_atlas.png", dpi=600, bbox_inches="tight", facecolor="white")
plt.show()

passed = int(verification["status"].eq("PASS").sum())
verdict = [
    "THERMOFUSION STAGE 09 ALIGNMENT VERIFICATION",
    f"Bundles located: {len(verification)}/8",
    f"Bundles passing all checks: {passed}/8",
    f"Predictor files with 16 bands: {verification['predictor_band_check'].sum()}/8",
    f"Thermal files with 2 bands: {verification['thermal_band_check'].sum()}/8",
    f"Spatially overlapping bundles: {verification['spatial_overlap_check'].sum()}/8",
    f"Bundles with >=25% valid Sentinel-2 pixels: {verification['s2_validity_check'].sum()}/8",
    f"Bundles with >=80% valid Sentinel-1 pixels: {verification['s1_validity_check'].sum()}/8",
    f"Bundles with >=10% valid thermal pixels: {verification['thermal_fraction_check'].sum()}/8",
    f"Plausible thermal ranges: {verification['temperature_range_check'].sum()}/8",
    "Overall alignment verdict: PASS" if passed == 8 else "Overall alignment verdict: REVIEW REQUIRED",
]
(OUTPUT_DIR / "02_alignment_verdict.txt").write_text("\n".join(verdict), encoding="utf-8")
zip_path = shutil.make_archive(
    "/content/drive/MyDrive/ThermoFusion_Stage09_Alignment_Verification", "zip", root_dir=OUTPUT_DIR
)
print("\n" + "\n".join(verdict))
print(f"\nZIP package: {zip_path}")
