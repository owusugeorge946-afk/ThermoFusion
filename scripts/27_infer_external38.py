# =====================================================================
# ThermoFusion reproducibility code
# Source notebook cell(s): 61
# Audit status: KEEP
# Extracted without scientific-logic modification.
# Replace local Google Drive paths as described in README.md when needed.
# =====================================================================

# ==========================================================================================
# THERMOFUSION STAGE 26 — FROZEN FINAL-MODEL INFERENCE ON EXPANDED TEMPORAL TEST
# No training, tuning, architecture selection, normalization fitting, or interval recalibration.
# ==========================================================================================

from google.colab import drive
from pathlib import Path
import json
import shutil
import subprocess
import sys
import os
import numpy as np
import pandas as pd
from IPython.display import display

try:
    import torch
    import torch.nn as nn
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "torch"])
    import torch
    import torch.nn as nn

try:
    from scipy.stats import spearmanr
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "scipy"])
    from scipy.stats import spearmanr

drive.mount("/content/drive")

# ------------------------------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------------------------------
BASE = Path("/content/drive/MyDrive")

STAGE25 = BASE / "ThermoFusion_Stage25_ExpandedTest_ModelReady"

EXPANDED_ARRAY_MANIFEST = (
    STAGE25 / "01_expanded_test_array_manifest.csv"
)

STAGE25_VERDICT = (
    STAGE25 / "05_model_ready_verdict.txt"
)

# Original final-model materials. These were derived before the expanded test existed.
STAGE14 = BASE / "ThermoFusion_Stage14_Refined_Benchmark"
STAGE16 = BASE / "ThermoFusion_Stage16_Modality_Ablation"
STAGE17 = BASE / "ThermoFusion_Stage17_Final_Uncertainty"

TRAINING_STATS = STAGE14 / "01_training_only_target_statistics.csv"

FINAL_CHECKPOINT = (
    STAGE16 / "checkpoints" / "without_terrain_best.pt"
)

STAGE16_VERDICT = STAGE16 / "05_stage16_verdict.txt"
STAGE17_VERDICT = STAGE17 / "04_stage17_verdict.txt"
STAGE17_CONFIG = STAGE17 / "05_configuration.json"

# New outputs only.
OUT = BASE / "ThermoFusion_Stage26_ExpandedTemporalTest_Inference"
PRED_DIR = OUT / "predictions"

OUT.mkdir(parents=True, exist_ok=True)
PRED_DIR.mkdir(parents=True, exist_ok=True)

SCENE_METRICS_CSV = OUT / "01_scene_accuracy_uncertainty.csv"
AGGREGATE_METRICS_CSV = OUT / "02_aggregate_accuracy_uncertainty.csv"
SAMPLED_PIXELS_CSV = OUT / "03_sampled_pixel_results.csv"
CONFIG_JSON = OUT / "04_frozen_inference_configuration.json"
VERDICT_TXT = OUT / "05_stage26_verdict.txt"

# ------------------------------------------------------------------------------------------
# Fixed final-model contract
# ------------------------------------------------------------------------------------------
SEED = 20260908
BASE_CHANNELS = 24
EXPECTED_SCENES = 38
TARGET_SIZE = 256
TARGET_COVERAGE = 0.90
CITIES = ["Abidjan", "Accra", "Freetown", "Lagos"]
SENSORS = ["ECOSTRESS", "Landsat"]

# The final Stage 17 selected model excluded terrain channels.
TERRAIN_CHANNELS = [14, 15]

# Eight dihedral test-time transformations.
TRANSFORMS = [(rotation, flip) for rotation in range(4) for flip in [False, True]]

np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    torch.set_num_threads(min(4, os.cpu_count() or 1))

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print("=" * 100)
print("THERMOFUSION STAGE 26 — FROZEN EXPANDED TEMPORAL-TEST INFERENCE")
print("=" * 100)
print(f"Compute device: {device}")

# ------------------------------------------------------------------------------------------
# Validate required original and expanded-test inputs
# ------------------------------------------------------------------------------------------
required_paths = [
    EXPANDED_ARRAY_MANIFEST,
    STAGE25_VERDICT,
    TRAINING_STATS,
    FINAL_CHECKPOINT,
    STAGE16_VERDICT,
    STAGE17_VERDICT,
    STAGE17_CONFIG,
]

for path in required_paths:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found:\n{path}")

if "Overall model-ready verdict: PASS" not in STAGE25_VERDICT.read_text(encoding="utf-8"):
    raise RuntimeError("Stage 25 did not report PASS.")

if "Overall Stage 16 execution: PASS" not in STAGE16_VERDICT.read_text(encoding="utf-8"):
    raise RuntimeError("Stage 16 did not report PASS.")

if "Overall Stage 17 execution: PASS" not in STAGE17_VERDICT.read_text(encoding="utf-8"):
    raise RuntimeError("Stage 17 did not report PASS.")

scenes = pd.read_csv(EXPANDED_ARRAY_MANIFEST)

if len(scenes) != EXPECTED_SCENES:
    raise RuntimeError(
        f"Expected {EXPECTED_SCENES} expanded-test arrays; found {len(scenes)}."
    )

if scenes["record_id"].nunique() != EXPECTED_SCENES:
    raise RuntimeError("Expanded-test array manifest contains duplicate record IDs.")

if not scenes["status"].eq("PASS").all():
    raise RuntimeError("At least one Stage 25 array is not marked PASS.")

# Acquisition date is needed for the original metadata context channels.
date_source = (
    BASE / "ThermoFusion_Stage23R2_FinalReplacement"
    / "Stage23R2_Export"
    / "02_exported_scene_manifest.csv"
)

if not date_source.exists():
    raise FileNotFoundError(f"Exported-scene manifest not found:\n{date_source}")

dates = pd.read_csv(date_source)[["export_id", "record_id", "acquisition_date"]]
dates["acquisition_date"] = pd.to_datetime(
    dates["acquisition_date"],
    utc=True,
    errors="raise",
)

scenes = scenes.merge(
    dates,
    on=["export_id", "record_id"],
    how="left",
    validate="one_to_one",
)

if scenes["acquisition_date"].isna().any():
    raise RuntimeError("At least one expanded-test scene has no acquisition date.")

scenes = scenes.sort_values(
    ["city", "thermal_sensor", "acquisition_date", "export_id"],
    kind="mergesort",
).reset_index(drop=True)

# ------------------------------------------------------------------------------------------
# Load training-only city–sensor statistics used by the frozen model
# ------------------------------------------------------------------------------------------
stats_table = pd.read_csv(TRAINING_STATS)

group_stats = {}

for row in stats_table.query("level == 'GROUP'").itertuples(index=False):
    city, sensor = row.group.split("|")

    group_stats[(city, sensor)] = {
        "mean_c": float(row.mean_c),
        "std_c": float(row.std_c),
    }

if len(group_stats) != 8:
    raise RuntimeError("Expected eight frozen city–sensor training-statistic groups.")

for city in CITIES:
    for sensor in SENSORS:
        if (city, sensor) not in group_stats:
            raise RuntimeError(f"Missing frozen training statistic: {city} × {sensor}")

global_row = stats_table.loc[
    (stats_table["level"] == "GLOBAL")
    & (stats_table["group"] == "ALL")
]

if len(global_row) != 1:
    raise RuntimeError("Could not locate the frozen global training climatology.")

global_mean_c = float(global_row.iloc[0]["mean_c"])

sensor_mean_c = {}

for sensor in SENSORS:
    match = stats_table.loc[
        (stats_table["level"] == "SENSOR")
        & (stats_table["group"] == sensor)
    ]

    if len(match) != 1:
        raise RuntimeError(f"Could not locate frozen sensor climatology: {sensor}")

    sensor_mean_c[sensor] = float(match.iloc[0]["mean_c"])

# ------------------------------------------------------------------------------------------
# Load original Stage 17 uncertainty calibration unchanged
# ------------------------------------------------------------------------------------------
with STAGE17_CONFIG.open("r", encoding="utf-8") as handle:
    stage17_config = json.load(handle)

conformal_q = float(stage17_config["conformal_multiplier"])
uncertainty_floor_c = float(stage17_config["uncertainty_floor_c"])
stage17_target_coverage = float(stage17_config["target_coverage"])

if not np.isclose(stage17_target_coverage, TARGET_COVERAGE):
    raise RuntimeError(
        "The stored Stage 17 target coverage is inconsistent with the final protocol."
    )

if not np.isfinite(conformal_q) or conformal_q <= 0:
    raise RuntimeError("Stored Stage 17 conformal multiplier is invalid.")

# ------------------------------------------------------------------------------------------
# Original metadata-conditioning function
# ------------------------------------------------------------------------------------------
def context(row):
    stamp = pd.Timestamp(row.acquisition_date)

    day_phase = 2 * np.pi * (stamp.dayofyear - 1) / 365.25
    hour_phase = 2 * np.pi * (stamp.hour + stamp.minute / 60.0) / 24.0

    values = [
        float(row.thermal_sensor == "ECOSTRESS"),
        np.sin(day_phase),
        np.cos(day_phase),
        np.sin(hour_phase),
        np.cos(hour_phase),
    ]

    values += [float(row.city == city) for city in CITIES]

    return np.asarray(values, dtype=np.float32)[:, None, None]

# ------------------------------------------------------------------------------------------
# Original final U-Net architecture
# ------------------------------------------------------------------------------------------
def conv_block(inputs, outputs):
    return nn.Sequential(
        nn.Conv2d(inputs, outputs, 3, padding=1, bias=False),
        nn.GroupNorm(min(8, outputs), outputs),
        nn.SiLU(inplace=True),
        nn.Conv2d(outputs, outputs, 3, padding=1, bias=False),
        nn.GroupNorm(min(8, outputs), outputs),
        nn.SiLU(inplace=True),
    )

class ResidualUNet(nn.Module):
    def __init__(self):
        super().__init__()

        b = BASE_CHANNELS

        self.pool = nn.MaxPool2d(2)

        self.e1 = conv_block(25, b)
        self.e2 = conv_block(b, 2 * b)
        self.e3 = conv_block(2 * b, 4 * b)
        self.e4 = conv_block(4 * b, 8 * b)

        self.bridge = conv_block(8 * b, 16 * b)

        self.u4 = nn.ConvTranspose2d(16 * b, 8 * b, 2, 2)
        self.d4 = conv_block(16 * b, 8 * b)

        self.u3 = nn.ConvTranspose2d(8 * b, 4 * b, 2, 2)
        self.d3 = conv_block(8 * b, 4 * b)

        self.u2 = nn.ConvTranspose2d(4 * b, 2 * b, 2, 2)
        self.d2 = conv_block(4 * b, 2 * b)

        self.u1 = nn.ConvTranspose2d(2 * b, b, 2, 2)
        self.d1 = conv_block(2 * b, b)

        self.output = nn.Conv2d(b, 1, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))

        z = self.bridge(self.pool(e4))

        z = self.d4(torch.cat([self.u4(z), e4], dim=1))
        z = self.d3(torch.cat([self.u3(z), e3], dim=1))
        z = self.d2(torch.cat([self.u2(z), e2], dim=1))
        z = self.d1(torch.cat([self.u1(z), e1], dim=1))

        return self.output(z)

# ------------------------------------------------------------------------------------------
# Load the frozen final model
# ------------------------------------------------------------------------------------------
checkpoint = torch.load(
    FINAL_CHECKPOINT,
    map_location=device,
    weights_only=False,
)

if checkpoint.get("variant") != "without_terrain":
    raise RuntimeError(
        "The selected checkpoint is not the validation-selected without-terrain model."
    )

model = ResidualUNet().to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("Frozen final model: metadata-conditioned residual U-Net without terrain")
print(f"TTA transformations: {len(TRANSFORMS)}")
print(f"Validation-only conformal multiplier: {conformal_q:.4f}")

# ------------------------------------------------------------------------------------------
# Array loading and eight-view test-time augmentation
# ------------------------------------------------------------------------------------------
def load_scene(row):
    with np.load(row.array_path, allow_pickle=False) as item:
        x = item["x"].astype(np.float32)
        observed = item["y_c"].astype(np.float32)
        valid_mask = item["valid_mask"].astype(bool)

    if (
        x.shape != (16, TARGET_SIZE, TARGET_SIZE)
        or observed.shape != (TARGET_SIZE, TARGET_SIZE)
        or valid_mask.shape != (TARGET_SIZE, TARGET_SIZE)
        or valid_mask.sum() == 0
    ):
        raise RuntimeError(f"Invalid Stage 25 array: {row.export_id}")

    metadata = np.broadcast_to(
        context(row),
        (9, TARGET_SIZE, TARGET_SIZE),
    ).copy()

    model_input = np.concatenate([x, metadata], axis=0)

    # Final selected Stage 16/17 model had terrain masked to zero.
    model_input[TERRAIN_CHANNELS] = 0.0

    if model_input.shape != (25, TARGET_SIZE, TARGET_SIZE):
        raise RuntimeError(f"Invalid final-model input shape: {model_input.shape}")

    return model_input, observed, valid_mask


def transform_array(array, rotation, flip):
    result = np.rot90(array, rotation, axes=(-2, -1))

    if flip:
        result = result[..., :, ::-1]

    return np.ascontiguousarray(result)


def inverse_array(array, rotation, flip):
    result = array[..., :, ::-1] if flip else array
    return np.ascontiguousarray(np.rot90(result, -rotation, axes=(-2, -1)))


def tta_predict(row):
    model_input, observed, valid_mask = load_scene(row)

    transformed = np.stack([
        transform_array(model_input, rotation, flip)
        for rotation, flip in TRANSFORMS
    ]).astype(np.float32)

    group = group_stats[(row.city, row.thermal_sensor)]

    with torch.no_grad():
        tensor = torch.from_numpy(transformed).to(device)
        standardized_residuals = model(tensor).cpu().numpy()[:, 0]

    predictions = []

    for residual, (rotation, flip) in zip(
        standardized_residuals,
        TRANSFORMS,
    ):
        residual = inverse_array(residual, rotation, flip)

        predictions.append(
            residual * group["std_c"] + group["mean_c"]
        )

    prediction_stack = np.stack(predictions).astype(np.float32)

    return (
        observed,
        valid_mask,
        prediction_stack.mean(axis=0).astype(np.float32),
        prediction_stack.std(axis=0, ddof=1).astype(np.float32),
    )

# ------------------------------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------------------------------
def regression_metrics(observed, predicted):
    observed = np.asarray(observed, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)

    residual = predicted - observed
    denominator = np.sum((observed - observed.mean()) ** 2)

    return {
        "pixels": int(len(observed)),
        "mae_c": float(np.mean(np.abs(residual))),
        "rmse_c": float(np.sqrt(np.mean(residual ** 2))),
        "bias_c": float(np.mean(residual)),
        "r2": (
            float(1 - np.sum(residual ** 2) / denominator)
            if denominator > 0
            else np.nan
        ),
    }


def safe_spearman(x, y):
    if len(x) < 3 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return np.nan, np.nan

    result = spearmanr(x, y)

    return float(result.statistic), float(result.pvalue)

# ------------------------------------------------------------------------------------------
# Frozen inference, interval calculation, predictions, and scene-level metrics
# ------------------------------------------------------------------------------------------
scene_rows = []
pixel_samples = []

for number, (_, row) in enumerate(scenes.iterrows(), start=1):
    observed, valid, predicted, tta_std = tta_predict(row)

    interval_half_width = conformal_q * (
        tta_std + uncertainty_floor_c
    )

    interval_lower = predicted - interval_half_width
    interval_upper = predicted + interval_half_width

    y = observed[valid]
    prediction = predicted[valid]
    uncertainty = tta_std[valid]
    width = 2 * interval_half_width[valid]

    absolute_error = np.abs(prediction - y)
    covered = (y >= interval_lower[valid]) & (y <= interval_upper[valid])

    rho, rho_p = safe_spearman(uncertainty, absolute_error)

    model_metrics = regression_metrics(y, prediction)

    scene_rows.append({
        "model": "frozen_without_terrain_unet_tta",
        "export_id": row["export_id"],
        "record_id": row["record_id"],
        "city": row["city"],
        "thermal_sensor": row["thermal_sensor"],
        "acquisition_date": row["acquisition_date"],
        **model_metrics,
        "interval_coverage": float(covered.mean()),
        "mean_interval_width_c": float(width.mean()),
        "median_tta_std_c": float(np.median(uncertainty)),
        "uncertainty_error_spearman": rho,
        "uncertainty_error_p": rho_p,
    })

    # Frozen training-only climatology baselines for interpretation.
    baselines = {
        "global_training_climatology": np.full_like(y, global_mean_c),
        "sensor_training_climatology": np.full_like(
            y,
            sensor_mean_c[row["thermal_sensor"]],
        ),
        "city_sensor_training_climatology": np.full_like(
            y,
            group_stats[(row["city"], row["thermal_sensor"])]["mean_c"],
        ),
    }

    for baseline_name, baseline_prediction in baselines.items():
        scene_rows.append({
            "model": baseline_name,
            "export_id": row["export_id"],
            "record_id": row["record_id"],
            "city": row["city"],
            "thermal_sensor": row["thermal_sensor"],
            "acquisition_date": row["acquisition_date"],
            **regression_metrics(y, baseline_prediction),
            "interval_coverage": np.nan,
            "mean_interval_width_c": np.nan,
            "median_tta_std_c": np.nan,
            "uncertainty_error_spearman": np.nan,
            "uncertainty_error_p": np.nan,
        })

    # Save full-resolution inference products for each scene.
    np.savez_compressed(
        PRED_DIR / f"{row['export_id']}_expanded_test_prediction.npz",
        observed_c=observed.astype(np.float32),
        predicted_c=predicted.astype(np.float32),
        tta_std_c=tta_std.astype(np.float32),
        interval_lower_c=interval_lower.astype(np.float32),
        interval_upper_c=interval_upper.astype(np.float32),
        valid_mask=valid.astype(np.uint8),
        record_id=np.asarray(row["record_id"]),
        city=np.asarray(row["city"]),
        thermal_sensor=np.asarray(row["thermal_sensor"]),
    )

    # Equal scene-level sample size avoids high-coverage scenes dominating
    # pooled diagnostics.
    rng = np.random.default_rng(SEED + sum(map(ord, str(row["export_id"]))))
    valid_indices = np.flatnonzero(valid)

    chosen = rng.choice(
        valid_indices,
        size=min(20_000, len(valid_indices)),
        replace=False,
    )

    pixel_samples.append(pd.DataFrame({
        "export_id": row["export_id"],
        "record_id": row["record_id"],
        "city": row["city"],
        "thermal_sensor": row["thermal_sensor"],
        "observed_c": observed.ravel()[chosen],
        "predicted_c": predicted.ravel()[chosen],
        "absolute_error_c": np.abs(
            predicted.ravel()[chosen] - observed.ravel()[chosen]
        ),
        "tta_std_c": tta_std.ravel()[chosen],
        "covered": (
            (observed.ravel()[chosen] >= interval_lower.ravel()[chosen])
            & (observed.ravel()[chosen] <= interval_upper.ravel()[chosen])
        ),
    }))

    print(
        f"Inference {number:02d}/{EXPECTED_SCENES}: {row['record_id']} | "
        f"MAE={model_metrics['mae_c']:.3f} °C | "
        f"RMSE={model_metrics['rmse_c']:.3f} °C | "
        f"coverage={covered.mean():.1%}"
    )

scene_metrics = pd.DataFrame(scene_rows)
scene_metrics.to_csv(SCENE_METRICS_CSV, index=False)

sampled_pixels = pd.concat(pixel_samples, ignore_index=True)
sampled_pixels.to_csv(SAMPLED_PIXELS_CSV, index=False)

# ------------------------------------------------------------------------------------------
# Aggregate results: pooled pixels and equal-weighted scene macro results
# ------------------------------------------------------------------------------------------
aggregate_rows = []

for model_name, model_scene in scene_metrics.groupby("model", observed=True):
    for group_name, group_columns in {
        "ALL": [],
        "CITY": ["city"],
        "SENSOR": ["thermal_sensor"],
        "CITY_SENSOR": ["city", "thermal_sensor"],
    }.items():

        if not group_columns:
            groups = [("ALL", model_scene)]
        else:
            groups = [
                (
                    " | ".join(map(str, key if isinstance(key, tuple) else (key,))),
                    subset,
                )
                for key, subset in model_scene.groupby(
                    group_columns,
                    observed=True,
                )
            ]

        for label, subset in groups:
            aggregate_rows.append({
                "model": model_name,
                "aggregation": group_name,
                "group": label,
                "scenes": int(subset["export_id"].nunique()),
                "scene_macro_mae_c": float(subset["mae_c"].mean()),
                "scene_macro_rmse_c": float(subset["rmse_c"].mean()),
                "median_scene_rmse_c": float(subset["rmse_c"].median()),
                "mean_scene_bias_c": float(subset["bias_c"].mean()),
                "mean_scene_r2": float(subset["r2"].mean()),
                "mean_interval_coverage": float(
                    subset["interval_coverage"].mean()
                ),
                "mean_interval_width_c": float(
                    subset["mean_interval_width_c"].mean()
                ),
                "mean_uncertainty_error_spearman": float(
                    subset["uncertainty_error_spearman"].mean()
                ),
            })

# Pooled-pixel metrics for the final U-Net from equal-size scene samples.
for group_name, group_columns in {
    "ALL": [],
    "CITY": ["city"],
    "SENSOR": ["thermal_sensor"],
    "CITY_SENSOR": ["city", "thermal_sensor"],
}.items():

    if not group_columns:
        groups = [("ALL", sampled_pixels)]
    else:
        groups = [
            (
                " | ".join(map(str, key if isinstance(key, tuple) else (key,))),
                subset,
            )
            for key, subset in sampled_pixels.groupby(
                group_columns,
                observed=True,
            )
        ]

    for label, subset in groups:
        metrics = regression_metrics(
            subset["observed_c"].to_numpy(),
            subset["predicted_c"].to_numpy(),
        )

        rho, rho_p = safe_spearman(
            subset["tta_std_c"].to_numpy(),
            subset["absolute_error_c"].to_numpy(),
        )

        aggregate_rows.append({
            "model": "frozen_without_terrain_unet_tta",
            "aggregation": f"PIXEL_SAMPLE_{group_name}",
            "group": label,
            "scenes": int(subset["export_id"].nunique()),
            "scene_macro_mae_c": metrics["mae_c"],
            "scene_macro_rmse_c": metrics["rmse_c"],
            "median_scene_rmse_c": np.nan,
            "mean_scene_bias_c": metrics["bias_c"],
            "mean_scene_r2": metrics["r2"],
            "mean_interval_coverage": float(subset["covered"].mean()),
            "mean_interval_width_c": np.nan,
            "mean_uncertainty_error_spearman": rho,
            "uncertainty_error_p": rho_p,
        })

aggregate = pd.DataFrame(aggregate_rows)
aggregate.to_csv(AGGREGATE_METRICS_CSV, index=False)

# ------------------------------------------------------------------------------------------
# Save configuration and final verdict
# ------------------------------------------------------------------------------------------
config = {
    "evaluation_name": "expanded independent temporal test",
    "scenes": EXPECTED_SCENES,
    "checkpoint": str(FINAL_CHECKPOINT),
    "checkpoint_variant": "without_terrain",
    "model_input_channels": 25,
    "raster_channels": 16,
    "metadata_channels": 9,
    "terrain_channels_zeroed": TERRAIN_CHANNELS,
    "test_time_transformations": len(TRANSFORMS),
    "conformal_multiplier_source": "original Stage 17 validation-only calibration",
    "conformal_multiplier": conformal_q,
    "uncertainty_floor_c": uncertainty_floor_c,
    "nominal_interval_coverage": TARGET_COVERAGE,
    "retraining_performed": False,
    "normalization_refit_performed": False,
    "interval_recalibration_performed": False,
    "device": str(device),
}

with CONFIG_JSON.open("w", encoding="utf-8") as handle:
    json.dump(config, handle, indent=2)

final_scene_results = scene_metrics.loc[
    scene_metrics["model"].eq("frozen_without_terrain_unet_tta")
].copy()

final_all = aggregate.loc[
    (aggregate["model"] == "frozen_without_terrain_unet_tta")
    & (aggregate["aggregation"] == "ALL")
    & (aggregate["group"] == "ALL")
].iloc[0]

execution_pass = (
    len(final_scene_results) == EXPECTED_SCENES
    and final_scene_results["export_id"].nunique() == EXPECTED_SCENES
    and len(list(PRED_DIR.glob("*_expanded_test_prediction.npz"))) == EXPECTED_SCENES
    and np.isfinite(
        final_scene_results[
            ["mae_c", "rmse_c", "bias_c", "r2", "interval_coverage"]
        ]
    ).all().all()
)

verdict_lines = [
    "THERMOFUSION STAGE 26 EXPANDED TEMPORAL-TEST INFERENCE",
    f"Expanded independent temporal-test scenes: {EXPECTED_SCENES}/38",
    "Model: validation-selected metadata-conditioned residual U-Net without terrain",
    "Model fitting on expanded test: NO",
    "Model selection using expanded test: NO",
    "Predictor normalization refitted on expanded test: NO",
    "Interval calibration refitted on expanded test: NO",
    f"Test-time transformations: {len(TRANSFORMS)}",
    f"Nominal interval coverage: {TARGET_COVERAGE:.0%}",
    f"Frozen validation-only conformal multiplier: {conformal_q:.4f}",
    f"Scene-macro MAE (°C): {final_all['scene_macro_mae_c']:.4f}",
    f"Scene-macro RMSE (°C): {final_all['scene_macro_rmse_c']:.4f}",
    f"Mean scene bias (°C): {final_all['mean_scene_bias_c']:.4f}",
    f"Mean scene R²: {final_all['mean_scene_r2']:.4f}",
    f"Mean observed interval coverage: {final_all['mean_interval_coverage']:.4f}",
    (
        "Overall Stage 26 inference verdict: PASS"
        if execution_pass
        else "Overall Stage 26 inference verdict: REVIEW REQUIRED"
    ),
]

VERDICT_TXT.write_text("\n".join(verdict_lines), encoding="utf-8")

zip_path = shutil.make_archive(
    str(OUT),
    "zip",
    root_dir=OUT,
)

print()
print("=" * 100)
print("\n".join(verdict_lines))
print(f"\nScene metrics: {SCENE_METRICS_CSV}")
print(f"Aggregate metrics: {AGGREGATE_METRICS_CSV}")
print(f"Prediction directory: {PRED_DIR}")
print(f"Output package: {zip_path}")

if not execution_pass:
    print("\nSCENES REQUIRING REVIEW:")
    display(
        final_scene_results.loc[
            ~np.isfinite(
                final_scene_results[
                    ["mae_c", "rmse_c", "bias_c", "r2", "interval_coverage"]
                ]
            ).all(axis=1)
        ]
    )
