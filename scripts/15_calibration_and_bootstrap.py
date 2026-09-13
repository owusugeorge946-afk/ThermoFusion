# =====================================================================
# ThermoFusion reproducibility code
# Source notebook cell(s): 29
# Audit status: KEEP
# Extracted without scientific-logic modification.
# Replace local Google Drive paths as described in README.md when needed.
# =====================================================================


# Run in one Google Colab cell after Stage 14 reports PASS. GPU is not required.

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
    from sklearn.linear_model import HuberRegressor
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "scikit-learn"])
    from sklearn.linear_model import HuberRegressor


drive.mount("/content/drive")

ROOT = Path("/content/drive/MyDrive")
S12 = ROOT / "ThermoFusion_Stage12_ModelReady_Pilot"
S14 = ROOT / "ThermoFusion_Stage14_Refined_Benchmark"
ARRAY_MANIFEST = S12 / "04_model_ready_array_manifest.csv"
SPLIT_MANIFEST = S12 / "01_scene_split_manifest.csv"
S12_VERDICT = S12 / "07_model_ready_verdict.txt"
S14_VERDICT = S14 / "06_stage14_verdict.txt"
S14_PREDICTIONS = S14 / "predictions"
TRAIN_STATS = S14 / "01_training_only_target_statistics.csv"
OUTPUT_DIR = ROOT / "ThermoFusion_Stage15_Calibration_Bootstrap"
CALIBRATED_DIR = OUTPUT_DIR / "calibrated_predictions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CALIBRATED_DIR.mkdir(parents=True, exist_ok=True)

SEED = 20260908
SAMPLE_PER_SCENE = 10000
BOOTSTRAPS = 20000
ALPHA = 0.05
rng = np.random.default_rng(SEED)

for path in [ARRAY_MANIFEST, SPLIT_MANIFEST, S12_VERDICT, S14_VERDICT,
             S14_PREDICTIONS, TRAIN_STATS]:
    if not path.exists():
        raise FileNotFoundError(f"Required input not found: {path}")
if "Overall model-ready verdict: PASS" not in S12_VERDICT.read_text(encoding="utf-8"):
    raise RuntimeError("Stage 12 did not pass.")
if "Overall Stage 14 execution: PASS" not in S14_VERDICT.read_text(encoding="utf-8"):
    raise RuntimeError("Stage 14 did not pass.")

arrays = pd.read_csv(ARRAY_MANIFEST)
meta = pd.read_csv(SPLIT_MANIFEST)[[
    "pilot_id", "record_id", "city", "thermal_sensor",
    "thermal_datetime_utc", "model_split",
]]
arrays = arrays.drop(columns=[c for c in ["city", "thermal_sensor", "thermal_datetime_utc"] if c in arrays])
scenes = arrays.merge(meta, on=["pilot_id", "record_id", "model_split"], validate="one_to_one")
scenes["thermal_datetime_utc"] = pd.to_datetime(scenes.thermal_datetime_utc, utc=True, errors="raise")
evaluation = scenes.query("model_split in ['validation', 'temporal_test']").copy()
if len(evaluation) != 16 or evaluation.pilot_id.nunique() != 16:
    raise RuntimeError("Expected 8 validation and 8 temporal-test scenes.")

training_stats = pd.read_csv(TRAIN_STATS)
group_means = {}
for row in training_stats.query("level == 'GROUP'").itertuples(index=False):
    city, sensor = row.group.split("|")
    group_means[(city, sensor)] = float(row.mean_c)
if len(group_means) != 8:
    raise RuntimeError("Expected eight training-only city-sensor climatologies.")


def load_scene(row):
    prediction_path = S14_PREDICTIONS / f"{row.pilot_id}_prediction.npz"
    if not prediction_path.exists():
        raise FileNotFoundError(f"Missing Stage 14 prediction: {prediction_path}")
    with np.load(prediction_path, allow_pickle=False) as item:
        predicted = item["predicted_c"].astype("float32")
        observed = item["observed_c"].astype("float32")
        valid = item["valid_mask"].astype(bool)
    if predicted.shape != (256, 256) or observed.shape != (256, 256):
        raise RuntimeError(f"Unexpected prediction shape for {row.pilot_id}.")
    finite = valid & np.isfinite(predicted) & np.isfinite(observed)
    if finite.sum() == 0:
        raise RuntimeError(f"No valid evaluation pixels for {row.pilot_id}.")
    return observed, predicted, finite


scene_cache = {}
for row in evaluation.itertuples(index=False):
    scene_cache[row.pilot_id] = load_scene(row)


def balanced_sample(frame, excluded=None, sensor=None):
    xs, ys = [], []
    subset = frame if sensor is None else frame.loc[frame.thermal_sensor.eq(sensor)]
    for row in subset.itertuples(index=False):
        if row.pilot_id == excluded:
            continue
        observed, predicted, valid = scene_cache[row.pilot_id]
        indices = np.flatnonzero(valid)
        sample_rng = np.random.default_rng(SEED + sum(map(ord, row.pilot_id)))
        chosen = sample_rng.choice(indices, size=min(SAMPLE_PER_SCENE, len(indices)), replace=False)
        xs.append(predicted.ravel()[chosen]); ys.append(observed.ravel()[chosen])
    if not xs:
        raise RuntimeError("No scenes were available to fit calibration.")
    return np.concatenate(xs), np.concatenate(ys)


def fit_huber(predicted, observed):
    model = HuberRegressor(epsilon=1.35, alpha=0.01, max_iter=500)
    model.fit(predicted.reshape(-1, 1), observed)
    return float(model.intercept_), float(model.coef_[0])


def apply_calibration(predicted, intercept, slope):
    return intercept + slope * predicted


def metrics(observed, predicted):
    observed = np.asarray(observed, dtype="float64")
    predicted = np.asarray(predicted, dtype="float64")
    error = predicted - observed
    denominator = np.sum((observed - observed.mean()) ** 2)
    return {
        "pixels": len(observed),
        "mae_c": float(np.mean(np.abs(error))),
        "rmse_c": float(np.sqrt(np.mean(error ** 2))),
        "bias_c": float(np.mean(error)),
        "r2": float(1 - np.sum(error ** 2) / denominator) if denominator > 0 else np.nan,
    }


# Leave-one-scene-out validation estimates calibration performance without
# evaluating a scene using calibration coefficients fitted to that same scene.
validation = evaluation.query("model_split == 'validation'").copy()
candidate_rows = []
for held_out in validation.itertuples(index=False):
    observed, raw, valid = scene_cache[held_out.pilot_id]
    global_x, global_y = balanced_sample(validation, excluded=held_out.pilot_id)
    global_intercept, global_slope = fit_huber(global_x, global_y)
    sensor_x, sensor_y = balanced_sample(
        validation, excluded=held_out.pilot_id, sensor=held_out.thermal_sensor
    )
    sensor_intercept, sensor_slope = fit_huber(sensor_x, sensor_y)
    predictions = {
        "raw_refined_unet": raw,
        "global_huber_calibration": apply_calibration(raw, global_intercept, global_slope),
        "sensor_huber_calibration": apply_calibration(raw, sensor_intercept, sensor_slope),
    }
    for method, prediction in predictions.items():
        candidate_rows.append({
            "pilot_id": held_out.pilot_id, "city": held_out.city,
            "thermal_sensor": held_out.thermal_sensor, "method": method,
            **metrics(observed[valid], prediction[valid]),
        })

cv_results = pd.DataFrame(candidate_rows)
cv_results.to_csv(OUTPUT_DIR / "01_leave_one_scene_out_calibration.csv", index=False)
cv_summary = (
    cv_results.groupby("method", observed=True)
    .agg(scene_macro_mae_c=("mae_c", "mean"), scene_macro_rmse_c=("rmse_c", "mean"),
         median_scene_rmse_c=("rmse_c", "median"))
    .reset_index()
    .sort_values(["scene_macro_rmse_c", "scene_macro_mae_c"])
)
selected_method = cv_summary.iloc[0].method
cv_summary["selected"] = cv_summary.method.eq(selected_method)
cv_summary.to_csv(OUTPUT_DIR / "02_calibration_selection.csv", index=False)
print("Validation-only selected method:", selected_method)

# Refit the selected calibration using all validation scenes, then freeze it.
global_x, global_y = balanced_sample(validation)
global_coefficients = fit_huber(global_x, global_y)
sensor_coefficients = {}
for sensor in sorted(validation.thermal_sensor.unique()):
    sensor_x, sensor_y = balanced_sample(validation, sensor=sensor)
    sensor_coefficients[sensor] = fit_huber(sensor_x, sensor_y)

coefficient_rows = [{
    "method": "global_huber_calibration", "thermal_sensor": "ALL",
    "intercept_c": global_coefficients[0], "slope": global_coefficients[1],
}]
coefficient_rows += [{
    "method": "sensor_huber_calibration", "thermal_sensor": sensor,
    "intercept_c": values[0], "slope": values[1],
} for sensor, values in sensor_coefficients.items()]
pd.DataFrame(coefficient_rows).to_csv(OUTPUT_DIR / "03_frozen_calibration_coefficients.csv", index=False)


def selected_prediction(raw, sensor):
    if selected_method == "raw_refined_unet":
        return raw.copy()
    if selected_method == "global_huber_calibration":
        return apply_calibration(raw, *global_coefficients)
    return apply_calibration(raw, *sensor_coefficients[sensor])


evaluation_rows = []
pixel_tables = []
for row in evaluation.itertuples(index=False):
    observed, raw, valid = scene_cache[row.pilot_id]
    calibrated = selected_prediction(raw, row.thermal_sensor)
    climatology = np.full_like(raw, group_means[(row.city, row.thermal_sensor)])
    predictions = {
        "raw_refined_unet": raw,
        "selected_calibrated_unet": calibrated,
        "city_sensor_climatology": climatology,
    }
    for method, prediction in predictions.items():
        evaluation_rows.append({
            "model_split": row.model_split, "pilot_id": row.pilot_id,
            "city": row.city, "thermal_sensor": row.thermal_sensor,
            "method": method, **metrics(observed[valid], prediction[valid]),
        })
        pixel_tables.append(pd.DataFrame({
            "model_split": row.model_split, "pilot_id": row.pilot_id,
            "city": row.city, "thermal_sensor": row.thermal_sensor,
            "method": method, "observed_c": observed[valid],
            "predicted_c": prediction[valid],
        }))
    np.savez_compressed(
        CALIBRATED_DIR / f"{row.pilot_id}_calibrated_prediction.npz",
        selected_method=np.asarray(selected_method),
        predicted_c=calibrated.astype("float32"),
        observed_c=observed,
        valid_mask=valid.astype("uint8"),
    )

scene_metrics = pd.DataFrame(evaluation_rows)
scene_metrics.to_csv(OUTPUT_DIR / "04_scene_evaluation.csv", index=False)
pixels = pd.concat(pixel_tables, ignore_index=True)
aggregate_rows = []
for (split, method), group in pixels.groupby(["model_split", "method"], observed=True):
    aggregate_rows.append({
        "model_split": split, "method": method, "group": "ALL",
        **metrics(group.observed_c, group.predicted_c),
    })
    for (city, sensor), sub in group.groupby(["city", "thermal_sensor"], observed=True):
        aggregate_rows.append({
            "model_split": split, "method": method,
            "group": f"{city}-{sensor}", **metrics(sub.observed_c, sub.predicted_c),
        })
aggregate = pd.DataFrame(aggregate_rows)
aggregate.to_csv(OUTPUT_DIR / "05_aggregate_evaluation.csv", index=False)

# Paired bootstrap resamples the eight temporal-test scenes, not individual pixels.
test_scene = scene_metrics.query("model_split == 'temporal_test'")
wide_rmse = test_scene.pivot(index="pilot_id", columns="method", values="rmse_c")
wide_mae = test_scene.pivot(index="pilot_id", columns="method", values="mae_c")
test_ids = wide_rmse.index.to_numpy()
bootstrap_rows = []
for metric_name, wide in [("scene_macro_rmse_c", wide_rmse), ("scene_macro_mae_c", wide_mae)]:
    delta = wide["selected_calibrated_unet"] - wide["city_sensor_climatology"]
    draws = np.empty(BOOTSTRAPS, dtype="float64")
    for number in range(BOOTSTRAPS):
        sampled = rng.choice(test_ids, size=len(test_ids), replace=True)
        draws[number] = delta.loc[sampled].mean()
    bootstrap_rows.append({
        "metric": metric_name,
        "calibrated_minus_climatology": float(delta.mean()),
        "ci_lower": float(np.quantile(draws, ALPHA / 2)),
        "ci_upper": float(np.quantile(draws, 1 - ALPHA / 2)),
        "probability_calibrated_better": float(np.mean(draws < 0)),
        "test_scenes": len(test_ids), "bootstrap_replicates": BOOTSTRAPS,
    })
bootstrap = pd.DataFrame(bootstrap_rows)
bootstrap.to_csv(OUTPUT_DIR / "06_paired_scene_bootstrap.csv", index=False)

sns.set_theme(style="whitegrid", context="notebook")
fig, axes = plt.subplots(1, 3, figsize=(20, 6), constrained_layout=True)
sns.barplot(data=cv_summary, x="method", y="scene_macro_rmse_c", ax=axes[0], color="#4c78a8")
axes[0].set(title="(a) Validation-only calibration selection", xlabel="", ylabel="LOSO scene-macro RMSE (°C)")
axes[0].tick_params(axis="x", rotation=22)

test_plot = test_scene[test_scene.method.isin([
    "selected_calibrated_unet", "city_sensor_climatology"
])]
sns.boxplot(data=test_plot, x="method", y="rmse_c", ax=axes[1])
sns.stripplot(data=test_plot, x="method", y="rmse_c", color="black", size=5, ax=axes[1])
axes[1].set(title="(b) Paired temporal-test scenes", xlabel="", ylabel="Scene RMSE (°C)")
axes[1].tick_params(axis="x", rotation=15)

scatter = pixels.query(
    "model_split == 'temporal_test' and method == 'selected_calibrated_unet'"
)
if len(scatter) > 100000:
    scatter = scatter.sample(100000, random_state=SEED)
sns.scatterplot(data=scatter, x="observed_c", y="predicted_c", hue="thermal_sensor",
                alpha=0.16, s=10, linewidth=0, ax=axes[2])
limits = [min(scatter.observed_c.min(), scatter.predicted_c.min()),
          max(scatter.observed_c.max(), scatter.predicted_c.max())]
axes[2].plot(limits, limits, "k--", lw=1)
axes[2].set(xlim=limits, ylim=limits, title="(c) Calibrated temporal-test predictions",
            xlabel="Observed LST (°C)", ylabel="Predicted LST (°C)")
axes[2].legend(frameon=False)
fig.savefig(OUTPUT_DIR / "thermofusion_stage15_calibration_bootstrap.png",
            dpi=800, bbox_inches="tight", facecolor="white")
plt.show()

test_overall = aggregate.query("model_split == 'temporal_test' and group == 'ALL'").set_index("method")
calibrated = test_overall.loc["selected_calibrated_unet"]
climatology = test_overall.loc["city_sensor_climatology"]
rmse_bootstrap = bootstrap.query("metric == 'scene_macro_rmse_c'").iloc[0]
execution_pass = (
    len(scene_metrics) == 48
    and scene_metrics.pilot_id.nunique() == 16
    and len(list(CALIBRATED_DIR.glob("*.npz"))) == 16
    and np.isfinite(aggregate[["mae_c", "rmse_c", "bias_c", "r2"]]).all().all()
    and len(bootstrap) == 2
)

verdict = [
    "THERMOFUSION STAGE 15 CALIBRATION AND BOOTSTRAP",
    f"Validation scenes used for calibration selection: {len(validation)}/8",
    f"Temporal-test scenes used for calibration fitting: 0/8",
    f"Selected calibration method: {selected_method}",
    f"Temporal-test calibrated MAE (°C): {calibrated.mae_c:.4f}",
    f"Temporal-test calibrated RMSE (°C): {calibrated.rmse_c:.4f}",
    f"Temporal-test calibrated bias (°C): {calibrated.bias_c:.4f}",
    f"Temporal-test calibrated R²: {calibrated.r2:.4f}",
    f"Temporal-test city-sensor climatology RMSE (°C): {climatology.rmse_c:.4f}",
    f"Paired scene-macro RMSE difference, calibrated minus climatology (°C): {rmse_bootstrap.calibrated_minus_climatology:.4f}",
    f"95% scene-bootstrap CI (°C): [{rmse_bootstrap.ci_lower:.4f}, {rmse_bootstrap.ci_upper:.4f}]",
    f"Bootstrap probability calibrated model is better: {rmse_bootstrap.probability_calibrated_better:.3f}",
    "Overall Stage 15 execution: PASS" if execution_pass else "Overall Stage 15 execution: REVIEW REQUIRED",
]
(OUTPUT_DIR / "07_stage15_verdict.txt").write_text("\n".join(verdict), encoding="utf-8")
with (OUTPUT_DIR / "08_stage15_configuration.json").open("w", encoding="utf-8") as handle:
    json.dump({
        "seed": SEED, "sample_per_validation_scene": SAMPLE_PER_SCENE,
        "bootstrap_replicates": BOOTSTRAPS, "confidence_level": 1 - ALPHA,
        "selection_rule": "lowest leave-one-scene-out validation scene-macro RMSE",
        "selected_method": selected_method,
    }, handle, indent=2)

zip_path = shutil.make_archive(
    str(ROOT / "ThermoFusion_Stage15_Calibration_Bootstrap"),
    "zip", root_dir=OUTPUT_DIR,
)
print("\n" + "\n".join(verdict))
print(f"\nOutput folder: {OUTPUT_DIR}")
print(f"ZIP package: {zip_path}")
