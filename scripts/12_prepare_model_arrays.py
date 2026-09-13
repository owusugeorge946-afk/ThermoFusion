# =====================================================================
# ThermoFusion reproducibility code
# Source notebook cell(s): 26
# Audit status: KEEP
# Extracted without scientific-logic modification.
# Replace local Google Drive paths as described in README.md when needed.
# =====================================================================

# ThermoFusion Stage 12: build leakage-safe model-ready pilot arrays.
# Run in one Google Colab cell after Stage 11 reports an overall PASS.

from google.colab import drive
from pathlib import Path
import json
import shutil
import subprocess
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import rasterio
    from rasterio.warp import reproject, Resampling
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "rasterio"])
    import rasterio
    from rasterio.warp import reproject, Resampling


drive.mount("/content/drive")

MY_DRIVE = Path("/content/drive/MyDrive")
VERIFICATION_CSV = (
    MY_DRIVE / "ThermoFusion_Stage11_Pilot_Verification"
    / "01_pilot_bundle_verification.csv"
)
VERDICT_PATH = (
    MY_DRIVE / "ThermoFusion_Stage11_Pilot_Verification"
    / "03_pilot_verification_verdict.txt"
)
MANIFEST_PATH = MY_DRIVE / "ThermoFusion_Stage10_Revised_Pilot_Manifest.csv"
OUTPUT_DIR = MY_DRIVE / "ThermoFusion_Stage12_ModelReady_Pilot"
ARRAY_DIR = OUTPUT_DIR / "arrays"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ARRAY_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_SCENES = 64
TARGET_SIZE = 256
NODATA = -9999.0
NORMALIZATION_CLIP = 8.0
CONTINUOUS_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 14, 15]
VALIDITY_INDICES = [9, 13]
S2_CONTINUOUS_INDICES = set(range(0, 9))
S1_CONTINUOUS_INDICES = {10, 11, 12}
PREDICTOR_BANDS = [
    "s2_b2", "s2_b3", "s2_b4", "s2_b8", "s2_b11", "s2_b12",
    "ndvi", "ndbi", "ndmi", "s2_valid",
    "s1_vv_db", "s1_vh_db", "s1_vv_minus_vh_db", "s1_valid",
    "elevation_m", "slope_deg",
]


for path in [VERIFICATION_CSV, VERDICT_PATH, MANIFEST_PATH]:
    if not path.exists():
        raise FileNotFoundError(f"Required input not found: {path}")
if "Overall pilot verification: PASS" not in VERDICT_PATH.read_text(encoding="utf-8"):
    raise RuntimeError("Stage 11 did not record an overall PASS.")

verification = pd.read_csv(VERIFICATION_CSV)
manifest = pd.read_csv(MANIFEST_PATH)
if (
    len(verification) != EXPECTED_SCENES
    or not verification["status"].eq("PASS").all()
    or verification["record_id"].nunique() != EXPECTED_SCENES
):
    raise RuntimeError("Stage 11 verification table is not a clean 64/64 PASS.")
if len(manifest) != EXPECTED_SCENES or manifest["record_id"].nunique() != EXPECTED_SCENES:
    raise RuntimeError("The revised pilot manifest is not a unique 64-record table.")

keep_metadata = [
    "record_id", "thermal_datetime_utc", "quarter", "year",
    "valid_fraction_pct", "temporal_role", "pilot_quality_score",
]
available_metadata = [column for column in keep_metadata if column in manifest.columns]
data = verification.merge(
    manifest[available_metadata], on="record_id", how="left", validate="one_to_one"
)
data["thermal_datetime_utc"] = pd.to_datetime(
    data["thermal_datetime_utc"], utc=True, errors="raise"
)
if data["thermal_datetime_utc"].isna().any():
    raise RuntimeError("At least one acquisition timestamp is missing.")

# Scene-level chronological partition: six development, one validation and one
# temporal test scene in every city-sensor group. No pixels from one scene can
# occur in more than one partition.
split_parts = []
for (_, _), group in data.groupby(["city", "thermal_sensor"], observed=True):
    ordered = group.sort_values(["thermal_datetime_utc", "record_id"]).copy()
    if len(ordered) != 8:
        raise RuntimeError("Every city-sensor group must contain exactly 8 scenes.")
    ordered["model_split"] = "train"
    ordered.loc[ordered.index[-2], "model_split"] = "validation"
    ordered.loc[ordered.index[-1], "model_split"] = "temporal_test"
    split_parts.append(ordered)

data = pd.concat(split_parts, ignore_index=True).sort_values("pilot_id").reset_index(drop=True)
split_counts = (
    data.groupby(["city", "thermal_sensor", "model_split"], observed=True)
    .size().rename("scenes").reset_index()
)
expected_split = {"train": 6, "validation": 1, "temporal_test": 1}
for (city, sensor), group in split_counts.groupby(["city", "thermal_sensor"]):
    observed = dict(zip(group["model_split"], group["scenes"]))
    if observed != expected_split:
        raise RuntimeError(f"Invalid split for {city} {sensor}: {observed}")


def aligned_scene(row):
    predictor_path = Path(row.predictor_path)
    thermal_path = Path(row.thermal_path)
    with rasterio.open(predictor_path) as predictor, rasterio.open(thermal_path) as thermal:
        x = predictor.read().astype("float32")
        thermal_data = thermal.read()
        y_native = thermal_data[0].astype("float32")
        native_mask = (
            (thermal_data[1] > 0.5)
            & np.isfinite(y_native)
            & (y_native != NODATA)
        ).astype("uint8")

        y = np.full((predictor.height, predictor.width), np.nan, dtype="float32")
        target_mask = np.zeros((predictor.height, predictor.width), dtype="uint8")
        reproject(
            source=y_native,
            destination=y,
            src_transform=thermal.transform,
            src_crs=thermal.crs,
            src_nodata=NODATA,
            dst_transform=predictor.transform,
            dst_crs=predictor.crs,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
        reproject(
            source=native_mask,
            destination=target_mask,
            src_transform=thermal.transform,
            src_crs=thermal.crs,
            src_nodata=0,
            dst_transform=predictor.transform,
            dst_crs=predictor.crs,
            dst_nodata=0,
            resampling=Resampling.nearest,
        )

    height, width = x.shape[1:]
    if height < TARGET_SIZE or width < TARGET_SIZE:
        raise RuntimeError(
            f"{row.pilot_id} predictor grid is {height}x{width}; "
            f"at least {TARGET_SIZE}x{TARGET_SIZE} is required."
        )
    row_offset = (height - TARGET_SIZE) // 2
    col_offset = (width - TARGET_SIZE) // 2
    rows = slice(row_offset, row_offset + TARGET_SIZE)
    cols = slice(col_offset, col_offset + TARGET_SIZE)
    x = x[:, rows, cols]
    y = y[rows, cols]
    target_mask = target_mask[rows, cols] > 0

    s2_valid = np.isfinite(x[9]) & (x[9] != NODATA) & (x[9] > 0.5)
    s1_valid = np.isfinite(x[13]) & (x[13] != NODATA) & (x[13] > 0.5)
    target_valid = (
        target_mask & np.isfinite(y) & (y != NODATA) & (y > -30) & (y < 80)
    )

    # Predictor modalities are allowed to be missing at different pixels. Their
    # explicit validity channels let the model distinguish missing values from
    # real standardized zeros. Requiring a thermal/S2/S1 intersection here would
    # incorrectly reject bundles that passed Stage 11's modality-wise checks.
    band_valid = {}
    for index in CONTINUOUS_INDICES:
        valid = np.isfinite(x[index]) & (x[index] != NODATA)
        if index in S2_CONTINUOUS_INDICES:
            valid &= s2_valid
        elif index in S1_CONTINUOUS_INDICES:
            valid &= s1_valid
        band_valid[index] = valid

    return (
        x, y, target_valid, band_valid, s2_valid, s1_valid,
        height, width, row_offset, col_offset,
    )


# First pass: statistics are calculated from training scenes only.
band_count = np.zeros(len(CONTINUOUS_INDICES), dtype="int64")
band_sum = np.zeros(len(CONTINUOUS_INDICES), dtype="float64")
band_sum_sq = np.zeros(len(CONTINUOUS_INDICES), dtype="float64")
target_count = 0
target_sum = 0.0
target_sum_sq = 0.0

for row in data.loc[data["model_split"].eq("train")].itertuples(index=False):
    x, y, target_valid, band_valid, *_ = aligned_scene(row)
    if target_valid.sum() == 0:
        raise RuntimeError(f"Training scene {row.pilot_id} has no valid thermal targets.")
    for position, band_index in enumerate(CONTINUOUS_INDICES):
        values = x[band_index][band_valid[band_index]].astype("float64")
        band_count[position] += values.size
        band_sum[position] += values.sum()
        band_sum_sq[position] += np.square(values).sum()
    targets = y[target_valid].astype("float64")
    target_count += targets.size
    target_sum += targets.sum()
    target_sum_sq += np.square(targets).sum()

band_active = band_count > 0
band_mean = np.zeros(len(CONTINUOUS_INDICES), dtype="float64")
band_std = np.ones(len(CONTINUOUS_INDICES), dtype="float64")
band_mean[band_active] = band_sum[band_active] / band_count[band_active]
band_variance = np.zeros(len(CONTINUOUS_INDICES), dtype="float64")
band_variance[band_active] = np.maximum(
    band_sum_sq[band_active] / band_count[band_active]
    - np.square(band_mean[band_active]),
    1e-12,
)
band_std[band_active] = np.sqrt(band_variance[band_active])
target_mean = target_sum / target_count
target_std = np.sqrt(max(target_sum_sq / target_count - target_mean ** 2, 1e-12))

if (
    target_count == 0
    or not np.isfinite(band_mean).all()
    or not np.isfinite(band_std).all()
    or np.any(band_std <= 0)
    or not np.isfinite(target_mean)
    or not np.isfinite(target_std)
    or target_std <= 0
):
    raise RuntimeError("Training-only normalization statistics are invalid.")

normalization = pd.DataFrame(
    {
        "band_index": CONTINUOUS_INDICES,
        "band_name": [PREDICTOR_BANDS[index] for index in CONTINUOUS_INDICES],
        "training_pixels": band_count,
        "mean": band_mean,
        "std": band_std,
        "active_channel": band_active,
        "normalization_status": np.where(
            band_active, "TRAINING_STATS", "INACTIVE_NO_TRAINING_PIXELS"
        ),
    }
)
normalization.to_csv(OUTPUT_DIR / "02_training_normalization.csv", index=False)
inactive_names = normalization.loc[
    ~normalization["active_channel"], "band_name"
].tolist()
if inactive_names:
    print(
        "Inactive continuous channels (zero usable training pixels; stored as 0): "
        + ", ".join(inactive_names)
    )
with (OUTPUT_DIR / "03_target_normalization.json").open("w", encoding="utf-8") as handle:
    json.dump(
        {
            "target_name": "land_surface_temperature_c",
            "training_pixels": int(target_count),
            "mean_c": float(target_mean),
            "std_c": float(target_std),
            "statistics_source": "training scenes only",
        },
        handle,
        indent=2,
    )

# Second pass: save standardized inputs and targets with explicit validity masks.
array_rows = []
for number, row in enumerate(data.itertuples(index=False), start=1):
    (
        x, y, target_valid, band_valid, s2_valid, s1_valid,
        original_height, original_width, row_offset, col_offset,
    ) = aligned_scene(row)
    if target_valid.sum() == 0:
        raise RuntimeError(f"Scene {row.pilot_id} has no valid thermal targets.")
    x_normalized = np.zeros_like(x, dtype="float32")

    for position, band_index in enumerate(CONTINUOUS_INDICES):
        if not band_active[position]:
            continue
        standardized = (x[band_index] - band_mean[position]) / band_std[position]
        standardized = np.clip(
            standardized, -NORMALIZATION_CLIP, NORMALIZATION_CLIP
        )
        standardized[~band_valid[band_index]] = 0.0
        x_normalized[band_index] = standardized.astype("float32")

    x_normalized[9] = s2_valid.astype("float32")
    x_normalized[13] = s1_valid.astype("float32")
    y_c = np.where(target_valid, y, 0.0).astype("float32")
    y_standardized = np.where(
        target_valid, (y - target_mean) / target_std, 0.0
    ).astype("float32")
    valid_mask = target_valid.astype("uint8")

    output_path = ARRAY_DIR / f"{row.pilot_id}_model_ready.npz"
    np.savez_compressed(
        output_path,
        x=x_normalized,
        y_c=y_c,
        y_standardized=y_standardized,
        valid_mask=valid_mask,
        predictor_band_names=np.asarray(PREDICTOR_BANDS),
        pilot_id=np.asarray(row.pilot_id),
        record_id=np.asarray(row.record_id),
        city=np.asarray(row.city),
        thermal_sensor=np.asarray(row.thermal_sensor),
        model_split=np.asarray(row.model_split),
    )
    array_rows.append(
        {
            "pilot_id": row.pilot_id,
            "record_id": row.record_id,
            "city": row.city,
            "thermal_sensor": row.thermal_sensor,
            "thermal_datetime_utc": row.thermal_datetime_utc,
            "model_split": row.model_split,
            "array_path": str(output_path),
            "array_size_mb": output_path.stat().st_size / 1e6,
            "channels": x_normalized.shape[0],
            "height": x_normalized.shape[1],
            "width": x_normalized.shape[2],
            "target_valid_pixels": int(target_valid.sum()),
            "target_valid_fraction": float(target_valid.mean()),
            "s2_valid_fraction": float(s2_valid.mean()),
            "s1_valid_fraction": float(s1_valid.mean()),
            "target_min_c": float(y[target_valid].min()),
            "target_median_c": float(np.median(y[target_valid])),
            "target_max_c": float(y[target_valid].max()),
            "original_predictor_height": original_height,
            "original_predictor_width": original_width,
            "crop_row_offset": row_offset,
            "crop_column_offset": col_offset,
            "status": "PASS",
        }
    )
    print(
        f"Prepared {number:02d}/64: {row.pilot_id} | {row.model_split} | "
        f"thermal={target_valid.mean():.1%} | S2={s2_valid.mean():.1%} | "
        f"S1={s1_valid.mean():.1%}"
    )

array_manifest = pd.DataFrame(array_rows)
data.to_csv(OUTPUT_DIR / "01_scene_split_manifest.csv", index=False)
array_manifest.to_csv(OUTPUT_DIR / "04_model_ready_array_manifest.csv", index=False)
split_counts.to_csv(OUTPUT_DIR / "05_split_balance.csv", index=False)

# Reload every saved array to verify shape, dtype, values and identity.
verification_rows = []
for row in array_manifest.itertuples(index=False):
    with np.load(row.array_path, allow_pickle=False) as item:
        x = item["x"]
        y_c = item["y_c"]
        y_standardized = item["y_standardized"]
        mask = item["valid_mask"]
        checks = {
            "shape_check": (
                x.shape == (16, TARGET_SIZE, TARGET_SIZE)
                and y_c.shape == (TARGET_SIZE, TARGET_SIZE)
                and y_standardized.shape == (TARGET_SIZE, TARGET_SIZE)
                and mask.shape == (TARGET_SIZE, TARGET_SIZE)
            ),
            "dtype_check": (
                x.dtype == np.float32
                and y_c.dtype == np.float32
                and y_standardized.dtype == np.float32
                and mask.dtype == np.uint8
            ),
            "finite_check": (
                np.isfinite(x).all()
                and np.isfinite(y_c).all()
                and np.isfinite(y_standardized).all()
            ),
            "mask_check": set(np.unique(mask)).issubset({0, 1}) and mask.sum() > 0,
            "identity_check": str(item["pilot_id"]) == row.pilot_id,
            "channel_check": len(item["predictor_band_names"]) == 16,
        }
        verification_rows.append(
            {
                "pilot_id": row.pilot_id,
                **checks,
                "status": "PASS" if all(checks.values()) else "REVIEW",
            }
        )

array_verification = pd.DataFrame(verification_rows)
array_verification.to_csv(OUTPUT_DIR / "06_array_integrity_verification.csv", index=False)

# Compact diagnostics.
sns.set_theme(style="whitegrid", context="notebook")
fig, axes = plt.subplots(1, 3, figsize=(20, 6), constrained_layout=True)

split_matrix = split_counts.pivot_table(
    index=["city", "thermal_sensor"],
    columns="model_split",
    values="scenes",
    fill_value=0,
)
split_matrix = split_matrix.reindex(
    columns=["train", "validation", "temporal_test"], fill_value=0
)
sns.heatmap(
    split_matrix, annot=True, fmt=".0f", cmap="crest", linewidths=0.7,
    cbar_kws={"label": "Scenes"}, ax=axes[0],
)
axes[0].set_title("(a) Leakage-safe scene splits")
axes[0].set_xlabel("")
axes[0].set_ylabel("")

sns.boxplot(
    data=array_manifest,
    x="thermal_sensor",
    y="target_valid_fraction",
    hue="city",
    ax=axes[1],
)
axes[1].set_title("(b) Model-ready thermal-target coverage")
axes[1].set_xlabel("")
axes[1].set_ylabel("Valid thermal-target fraction")
axes[1].legend(frameon=False, fontsize=8)

sensor_colours = {"Landsat": "#3569a8", "ECOSTRESS": "#b42c3b"}
for sensor, group in array_manifest.groupby("thermal_sensor", observed=True):
    axes[2].hist(
        group["target_median_c"], bins=10, alpha=0.65,
        label=sensor, color=sensor_colours[sensor],
    )
axes[2].set_title("(c) Scene-level target distributions")
axes[2].set_xlabel("Median land-surface temperature (°C)")
axes[2].set_ylabel("Scenes")
axes[2].legend(frameon=False)

figure_path = OUTPUT_DIR / "thermofusion_stage12_model_ready_diagnostics.png"
fig.savefig(figure_path, dpi=800, bbox_inches="tight", facecolor="white")
plt.show()

split_totals = data["model_split"].value_counts().to_dict()
all_arrays_pass = array_verification["status"].eq("PASS").all()
overall_pass = (
    len(array_manifest) == EXPECTED_SCENES
    and array_manifest["record_id"].nunique() == EXPECTED_SCENES
    and len(list(ARRAY_DIR.glob("*.npz"))) == EXPECTED_SCENES
    and split_totals == {"train": 48, "validation": 8, "temporal_test": 8}
    and all_arrays_pass
)

verdict = [
    "THERMOFUSION STAGE 12 MODEL-READY PILOT",
    f"Verified input bundles: {len(data)}/64",
    f"Model-ready arrays: {len(array_manifest)}/64",
    f"Training scenes: {split_totals.get('train', 0)}/48",
    f"Validation scenes: {split_totals.get('validation', 0)}/8",
    f"Temporal-test scenes: {split_totals.get('temporal_test', 0)}/8",
    f"Arrays passing reload integrity checks: "
    f"{array_verification['status'].eq('PASS').sum()}/64",
    f"Predictor channels: 16",
    f"Active continuous predictor channels: {int(band_active.sum())}/{len(CONTINUOUS_INDICES)}",
    "Inactive continuous channels: " + (
        ", ".join(inactive_names) if inactive_names else "none"
    ),
    f"Array dimensions: 256 x 256",
    f"Training-only target mean (°C): {target_mean:.4f}",
    f"Training-only target standard deviation (°C): {target_std:.4f}",
    "Scene-level leakage check: PASS",
    "Modality-aware missing-data handling: PASS",
    "Overall model-ready verdict: PASS" if overall_pass
    else "Overall model-ready verdict: REVIEW REQUIRED",
]
(OUTPUT_DIR / "07_model_ready_verdict.txt").write_text(
    "\n".join(verdict), encoding="utf-8"
)

zip_path = shutil.make_archive(
    "/content/drive/MyDrive/ThermoFusion_Stage12_ModelReady_Pilot",
    "zip",
    root_dir=OUTPUT_DIR,
)
print("\n" + "\n".join(verdict))
print(f"\nOutput folder: {OUTPUT_DIR}")
print(f"ZIP package: {zip_path}")
