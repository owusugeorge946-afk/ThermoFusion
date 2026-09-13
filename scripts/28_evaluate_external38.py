# =====================================================================
# ThermoFusion reproducibility code
# Source notebook cell(s): 62
# Audit status: KEEP
# Extracted without scientific-logic modification.
# Replace local Google Drive paths as described in README.md when needed.
# =====================================================================

# ==========================================================================================
# THERMOFUSION STAGE 27 — EXPANDED TEMPORAL-TEST RESULTS, PAIRED EFFECTS, AND FIGURES
# Analyses the completed frozen-inference outputs only. No fitting or recalibration.
# ==========================================================================================

from google.colab import drive
from pathlib import Path
import json
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

drive.mount("/content/drive")

# ------------------------------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------------------------------
BASE = Path("/content/drive/MyDrive")

STAGE26 = BASE / "ThermoFusion_Stage26_ExpandedTemporalTest_Inference"

SCENE_METRICS = STAGE26 / "01_scene_accuracy_uncertainty.csv"
SAMPLED_PIXELS = STAGE26 / "03_sampled_pixel_results.csv"
STAGE26_VERDICT = STAGE26 / "05_stage26_verdict.txt"

OUT = BASE / "ThermoFusion_Stage27_ExpandedTemporalTest_Analysis"
OUT.mkdir(parents=True, exist_ok=True)

MODEL_SUMMARY_CSV = OUT / "01_model_scene_macro_summary.csv"
PAIRED_EFFECTS_CSV = OUT / "02_paired_scene_bootstrap_effects.csv"
SUBGROUP_CSV = OUT / "03_city_sensor_subgroup_results.csv"
UNCERTAINTY_CSV = OUT / "04_uncertainty_reliability.csv"
SCENE_TABLE_CSV = OUT / "05_scene_level_comparison_table.csv"
VERDICT_TXT = OUT / "06_stage27_analysis_verdict.txt"

FIGURE_MAIN = OUT / "thermofusion_stage27_expanded_temporal_test_results.png"
FIGURE_RELIABILITY = OUT / "thermofusion_stage27_uncertainty_reliability.png"

EXPECTED_SCENES = 38
SEED = 20260909
BOOTSTRAP_REPLICATES = 10_000

FINAL_MODEL = "frozen_without_terrain_unet_tta"
BASELINES = [
    "global_training_climatology",
    "sensor_training_climatology",
    "city_sensor_training_climatology",
]

# ------------------------------------------------------------------------------------------
# Load and validate completed Stage 26 results
# ------------------------------------------------------------------------------------------
for path in [SCENE_METRICS, SAMPLED_PIXELS, STAGE26_VERDICT]:
    if not path.exists():
        raise FileNotFoundError(f"Required Stage 26 output not found:\n{path}")

if "Overall Stage 26 inference verdict: PASS" not in STAGE26_VERDICT.read_text(
    encoding="utf-8"
):
    raise RuntimeError("Stage 26 did not report PASS.")

scene_metrics = pd.read_csv(SCENE_METRICS)
pixels = pd.read_csv(SAMPLED_PIXELS)

expected_models = [FINAL_MODEL] + BASELINES

if not set(expected_models).issubset(set(scene_metrics["model"])):
    raise RuntimeError(
        "One or more expected frozen-model or baseline outputs are missing."
    )

final_scene = scene_metrics.loc[
    scene_metrics["model"].eq(FINAL_MODEL)
].copy()

if (
    len(final_scene) != EXPECTED_SCENES
    or final_scene["record_id"].nunique() != EXPECTED_SCENES
):
    raise RuntimeError("Expected 38 unique frozen U-Net scene results.")

if pixels["record_id"].nunique() != EXPECTED_SCENES:
    raise RuntimeError("Sampled-pixel table does not contain all 38 scenes.")

print("=" * 100)
print("THERMOFUSION STAGE 27 — EXPANDED TEMPORAL-TEST RESULTS ANALYSIS")
print("=" * 100)
print(f"Independent supplementary temporal-test scenes: {EXPECTED_SCENES}")
print("Model fitting, model selection, and interval recalibration: not performed")

# ------------------------------------------------------------------------------------------
# 1. Overall scene-macro comparison
# ------------------------------------------------------------------------------------------
summary_rows = []

for model_name, group in scene_metrics.groupby("model", observed=True):
    summary_rows.append({
        "model": model_name,
        "scenes": int(group["record_id"].nunique()),
        "scene_macro_mae_c": float(group["mae_c"].mean()),
        "scene_macro_rmse_c": float(group["rmse_c"].mean()),
        "median_scene_rmse_c": float(group["rmse_c"].median()),
        "mean_scene_bias_c": float(group["bias_c"].mean()),
        "mean_scene_r2": float(group["r2"].mean()),
        "mean_interval_coverage": float(group["interval_coverage"].mean()),
        "mean_interval_width_c": float(group["mean_interval_width_c"].mean()),
        "mean_uncertainty_error_spearman": float(
            group["uncertainty_error_spearman"].mean()
        ),
    })

model_summary = pd.DataFrame(summary_rows).sort_values(
    "scene_macro_rmse_c"
).reset_index(drop=True)

model_summary.to_csv(MODEL_SUMMARY_CSV, index=False)

print()
print("SCENE-MACRO MODEL COMPARISON:")
display(model_summary)

# ------------------------------------------------------------------------------------------
# 2. Paired scene bootstrap: positive delta means the U-Net has lower error.
# ------------------------------------------------------------------------------------------
wide_rmse = scene_metrics.pivot_table(
    index=["export_id", "record_id", "city", "thermal_sensor"],
    columns="model",
    values="rmse_c",
    aggfunc="first",
)

wide_mae = scene_metrics.pivot_table(
    index=["export_id", "record_id", "city", "thermal_sensor"],
    columns="model",
    values="mae_c",
    aggfunc="first",
)

if len(wide_rmse) != EXPECTED_SCENES:
    raise RuntimeError("Paired scene table does not contain 38 scenes.")

rng = np.random.default_rng(SEED)
paired_rows = []

for baseline in BASELINES:
    if baseline not in wide_rmse.columns:
        continue

    rmse_delta = (
        wide_rmse[baseline].to_numpy()
        - wide_rmse[FINAL_MODEL].to_numpy()
    )

    mae_delta = (
        wide_mae[baseline].to_numpy()
        - wide_mae[FINAL_MODEL].to_numpy()
    )

    n = len(rmse_delta)

    bootstrap_rmse = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    bootstrap_mae = np.empty(BOOTSTRAP_REPLICATES, dtype=float)

    for draw in range(BOOTSTRAP_REPLICATES):
        sampled = rng.integers(0, n, size=n)
        bootstrap_rmse[draw] = rmse_delta[sampled].mean()
        bootstrap_mae[draw] = mae_delta[sampled].mean()

    paired_rows.append({
        "comparison": f"{baseline} minus {FINAL_MODEL}",
        "baseline": baseline,
        "scenes": n,
        "mean_paired_rmse_advantage_c": float(rmse_delta.mean()),
        "rmse_ci_lower_c": float(np.quantile(bootstrap_rmse, 0.025)),
        "rmse_ci_upper_c": float(np.quantile(bootstrap_rmse, 0.975)),
        "probability_unet_lower_rmse": float(np.mean(bootstrap_rmse > 0)),
        "mean_paired_mae_advantage_c": float(mae_delta.mean()),
        "mae_ci_lower_c": float(np.quantile(bootstrap_mae, 0.025)),
        "mae_ci_upper_c": float(np.quantile(bootstrap_mae, 0.975)),
        "probability_unet_lower_mae": float(np.mean(bootstrap_mae > 0)),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
    })

paired_effects = pd.DataFrame(paired_rows)
paired_effects.to_csv(PAIRED_EFFECTS_CSV, index=False)

print()
print("PAIRED SCENE EFFECTS:")
print("Positive advantage means lower error for the frozen U-Net.")
display(paired_effects)

# ------------------------------------------------------------------------------------------
# 3. City × sensor subgroup results
# ------------------------------------------------------------------------------------------
unet_groups = final_scene.groupby(
    ["city", "thermal_sensor"],
    observed=True,
)

baseline_city_sensor = scene_metrics.loc[
    scene_metrics["model"].eq("city_sensor_training_climatology")
].copy()

baseline_lookup = baseline_city_sensor.set_index(
    ["export_id", "record_id"]
)[["rmse_c", "mae_c"]]

subgroup_rows = []

for (city, sensor), group in unet_groups:
    group_ids = group.set_index(["export_id", "record_id"]).index

    paired_baseline = baseline_lookup.reindex(group_ids)

    subgroup_rows.append({
        "city": city,
        "thermal_sensor": sensor,
        "scenes": int(len(group)),
        "scene_macro_mae_c": float(group["mae_c"].mean()),
        "scene_macro_rmse_c": float(group["rmse_c"].mean()),
        "median_scene_rmse_c": float(group["rmse_c"].median()),
        "mean_scene_bias_c": float(group["bias_c"].mean()),
        "mean_scene_r2": float(group["r2"].mean()),
        "mean_interval_coverage": float(group["interval_coverage"].mean()),
        "mean_interval_width_c": float(group["mean_interval_width_c"].mean()),
        "mean_tta_uncertainty_error_spearman": float(
            group["uncertainty_error_spearman"].mean()
        ),
        "city_sensor_climatology_rmse_c": float(
            paired_baseline["rmse_c"].mean()
        ),
        "unet_rmse_advantage_vs_city_sensor_climatology_c": float(
            paired_baseline["rmse_c"].mean() - group["rmse_c"].mean()
        ),
    })

subgroups = pd.DataFrame(subgroup_rows).sort_values(
    ["city", "thermal_sensor"]
).reset_index(drop=True)

subgroups.to_csv(SUBGROUP_CSV, index=False)

print()
print("CITY × SENSOR RESULTS:")
display(subgroups)

# ------------------------------------------------------------------------------------------
# 4. Uncertainty-error reliability from equal-size per-scene pixel samples
# ------------------------------------------------------------------------------------------
reliability_rows = []

for sensor, group in pixels.groupby("thermal_sensor", observed=True):
    group = group.copy()

    # Quantile bins are calculated within sensor, so the error trend is not
    # driven by different ECOSTRESS and Landsat uncertainty scales.
    try:
        group["uncertainty_bin"] = pd.qcut(
            group["tta_std_c"],
            q=10,
            duplicates="drop",
        )
    except ValueError:
        group["uncertainty_bin"] = pd.cut(
            group["tta_std_c"],
            bins=10,
            duplicates="drop",
        )

    summary = group.groupby(
        "uncertainty_bin",
        observed=True,
    ).agg(
        pixels=("absolute_error_c", "size"),
        mean_tta_std_c=("tta_std_c", "mean"),
        mean_absolute_error_c=("absolute_error_c", "mean"),
        median_absolute_error_c=("absolute_error_c", "median"),
        observed_coverage=("covered", "mean"),
    ).reset_index()

    summary["thermal_sensor"] = sensor
    summary["uncertainty_bin"] = summary["uncertainty_bin"].astype(str)

    reliability_rows.append(summary)

reliability = pd.concat(reliability_rows, ignore_index=True)
reliability.to_csv(UNCERTAINTY_CSV, index=False)

# ------------------------------------------------------------------------------------------
# 5. Assemble a scene-level comparison table for manuscript/supplement use
# ------------------------------------------------------------------------------------------
scene_table = final_scene.merge(
    scene_metrics.loc[
        scene_metrics["model"].eq("city_sensor_training_climatology"),
        ["export_id", "record_id", "mae_c", "rmse_c"],
    ].rename(columns={
        "mae_c": "city_sensor_climatology_mae_c",
        "rmse_c": "city_sensor_climatology_rmse_c",
    }),
    on=["export_id", "record_id"],
    how="left",
    validate="one_to_one",
)

scene_table["rmse_advantage_vs_city_sensor_climatology_c"] = (
    scene_table["city_sensor_climatology_rmse_c"]
    - scene_table["rmse_c"]
)

scene_table = scene_table.sort_values(
    ["city", "thermal_sensor", "acquisition_date", "export_id"]
).reset_index(drop=True)

scene_table.to_csv(SCENE_TABLE_CSV, index=False)

# ------------------------------------------------------------------------------------------
# 6. Publication-quality diagnostic figures
# ------------------------------------------------------------------------------------------
sns.set_theme(style="whitegrid", context="notebook")

final_label = "Frozen U-Net + TTA"
baseline_label = "City–sensor climatology"

plot_pairs = wide_rmse.reset_index()

fig, axes = plt.subplots(
    2,
    3,
    figsize=(19, 11),
    constrained_layout=True,
)

# (a) Paired RMSE against frozen city–sensor climatology.
axes[0, 0].scatter(
    plot_pairs[FINAL_MODEL],
    plot_pairs["city_sensor_training_climatology"],
    s=50,
    alpha=0.80,
    color="#2a9d8f",
    edgecolor="white",
    linewidth=0.5,
)

limits = [
    min(
        plot_pairs[FINAL_MODEL].min(),
        plot_pairs["city_sensor_training_climatology"].min(),
    ),
    max(
        plot_pairs[FINAL_MODEL].max(),
        plot_pairs["city_sensor_training_climatology"].max(),
    ),
]

axes[0, 0].plot(limits, limits, "--", color="black", linewidth=1)
axes[0, 0].set(
    title="(a) Paired scene RMSE",
    xlabel=f"{final_label} RMSE (°C)",
    ylabel=f"{baseline_label} RMSE (°C)",
    xlim=limits,
    ylim=limits,
)

# (b (b) Scene-level RMSE distributions.
rmse_plot = scene_metrics.loc[
    scene_metrics["model"].isin([
        FINAL_MODEL,
        "city_sensor_training_climatology",
    ])
].copy()

rmse_plot["model_label"] = rmse_plot["model"].map({
    FINAL_MODEL: final_label,
    "city_sensor_training_climatology": baseline_label,
})

sns.boxplot(
    data=rmse_plot,
    x="model_label",
    y="rmse_c",
    palette=["#2a9d8f", "#4c78a8"],
    ax=axes[0, 1],
)

sns.stripplot(
    data=rmse_plot,
    x="model_label",
    y="rmse_c",
    color="black",
    alpha=0.6,
    size=4,
    ax=axes[0, 1],
)

axes[0, 1].set(
    title="(b) Scene-level error distribution",
    xlabel="",
    ylabel="RMSE (°C)",
)

# (c) Interval coverage by sensor.
sns.boxplot(
    data=final_scene,
    x="thermal_sensor",
    y="interval_coverage",
    palette=["#7f3c8d", "#e76f51"],
    ax=axes[0, 2],
)

sns.stripplot(
    data=final_scene,
    x="thermal_sensor",
    y="interval_coverage",
    color="black",
    alpha=0.65,
    size=4,
    ax=axes[0, 2],
)

axes[0, 2].axhline(
    0.90,
    color="black",
    linestyle="--",
    linewidth=1,
    label="Nominal 90%",
)

axes[0, 2].set(
    title="(c) Interval coverage under temporal shift",
    xlabel="",
    ylabel="Observed coverage",
    ylim=(0, 1.05),
)

axes[0, 2].legend(frameon=False, loc="lower left")

# (d) City × sensor scene-macro RMSE.
matrix = subgroups.pivot(
    index="city",
    columns="thermal_sensor",
    values="scene_macro_rmse_c",
).reindex(index=CITIES, columns=SENSORS)

sns.heatmap(
    matrix,
    annot=True,
    fmt=".2f",
    cmap="mako_r",
    linewidths=0.8,
    cbar_kws={"label": "Scene-macro RMSE (°C)"},
    ax=axes[1, 0],
)

axes[1, 0].set(
    title="(d) City–sensor transfer matrix",
    xlabel="",
    ylabel="",
)

# (e) City × sensor interval coverage.
coverage_matrix = subgroups.pivot(
    index="city",
    columns="thermal_sensor",
    values="mean_interval_coverage",
).reindex(index=CITIES, columns=SENSORS)

sns.heatmap(
    coverage_matrix,
    annot=True,
    fmt=".2f",
    cmap="RdYlGn",
    vmin=0,
    vmax=1,
    linewidths=0.8,
    cbar_kws={"label": "Observed 90% interval coverage"},
    ax=axes[1, 1],
)

axes[1, 1].set(
    title="(e) Coverage by city and sensor",
    xlabel="",
    ylabel="",
)

# (f) RMSE advantage over city–sensor climatology.
advantage_plot = subgroups.copy()
advantage_plot["city_sensor"] = (
    advantage_plot["city"]
    + " — "
    + advantage_plot["thermal_sensor"]
)

advantage_plot = advantage_plot.sort_values(
    "unet_rmse_advantage_vs_city_sensor_climatology_c"
)

bar_colours = np.where(
    advantage_plot["unet_rmse_advantage_vs_city_sensor_climatology_c"] >= 0,
    "#2a9d8f",
    "#c44e52",
)

axes[1, 2].barh(
    advantage_plot["city_sensor"],
    advantage_plot["unet_rmse_advantage_vs_city_sensor_climatology_c"],
    color=bar_colours,
)

axes[1, 2].axvline(0, color="black", linewidth=1)

axes[1, 2].set(
    title="(f) U-Net advantage over climatology",
    xlabel="Climatology RMSE − U-Net RMSE (°C)",
    ylabel="",
)

fig.savefig(
    FIGURE_MAIN,
    dpi=800,
    bbox_inches="tight",
    facecolor="white",
)

plt.show()

# Separate uncertainty-reliability figure.
fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)

sensor_colours = {
    "ECOSTRESS": "#7f3c8d",
    "Landsat": "#e76f51",
}

for sensor, group in reliability.groupby("thermal_sensor", observed=True):
    group = group.sort_values("mean_tta_std_c")

    ax.plot(
        group["mean_tta_std_c"],
        group["mean_absolute_error_c"],
        "o-",
        linewidth=2,
        markersize=5,
        label=sensor,
        color=sensor_colours.get(sensor, None),
    )

ax.set(
    title="Expanded temporal-test uncertainty–error reliability",
    xlabel="Mean TTA standard deviation (°C)",
    ylabel="Mean absolute error (°C)",
)

ax.legend(frameon=False)

fig.savefig(
    FIGURE_RELIABILITY,
    dpi=800,
    bbox_inches="tight",
    facecolor="white",
)

plt.show()

# ------------------------------------------------------------------------------------------
# Final analysis verdict
# ------------------------------------------------------------------------------------------
final_model_summary = model_summary.loc[
    model_summary["model"].eq(FINAL_MODEL)
].iloc[0]

best_baseline = model_summary.loc[
    model_summary["model"].isin(BASELINES)
].sort_values("scene_macro_rmse_c").iloc[0]

verdict_lines = [
    "THERMOFUSION STAGE 27 EXPANDED TEMPORAL-TEST ANALYSIS",
    f"Independent supplementary temporal-test scenes: {EXPECTED_SCENES}/38",
    "Model fitting on expanded test: NO",
    "Model selection on expanded test: NO",
    "Normalization refitting on expanded test: NO",
    "Interval recalibration on expanded test: NO",
    f"Frozen U-Net scene-macro MAE (°C): {final_model_summary['scene_macro_mae_c']:.4f}",
    f"Frozen U-Net scene-macro RMSE (°C): {final_model_summary['scene_macro_rmse_c']:.4f}",
    f"Frozen U-Net mean scene bias (°C): {final_model_summary['mean_scene_bias_c']:.4f}",
    f"Frozen U-Net observed interval coverage: {final_model_summary['mean_interval_coverage']:.4f}",
    f"Best frozen climatology baseline: {best_baseline['model']}",
    f"Best climatology scene-macro RMSE (°C): {best_baseline['scene_macro_rmse_c']:.4f}",
    "Interpretation: this is a supplementary independent temporal stress test,",
    "not a replacement for the original balanced 8-scene temporal test or LOCO analysis.",
    "Overall Stage 27 analysis verdict: PASS",
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
print(f"\nModel summary: {MODEL_SUMMARY_CSV}")
print(f"Paired bootstrap effects: {PAIRED_EFFECTS_CSV}")
print(f"Subgroup results: {SUBGROUP_CSV}")
print(f"Main figure: {FIGURE_MAIN}")
print(f"Reliability figure: {FIGURE_RELIABILITY}")
print(f"Output package: {zip_path}")
