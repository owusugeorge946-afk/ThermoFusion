# =====================================================================
# ThermoFusion reproducibility code
# Source notebook cell(s): 63
# Audit status: KEEP
# Extracted without scientific-logic modification.
# Replace local Google Drive paths as described in README.md when needed.
# =====================================================================

# ==========================================================================================
# THERMOFUSION STAGE 28 — ADVANCED EXPANDED TEMPORAL-TEST TRANSFER DIAGNOSTICS ATLAS
# Creates one high-resolution manuscript figure and an exact caption.
# ==========================================================================================

from google.colab import drive
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import seaborn as sns

drive.mount("/content/drive")

# ------------------------------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------------------------------
BASE = Path("/content/drive/MyDrive")

STAGE26 = BASE / "ThermoFusion_Stage26_ExpandedTemporalTest_Inference"
STAGE27 = BASE / "ThermoFusion_Stage27_ExpandedTemporalTest_Analysis"

SCENE_METRICS = STAGE26 / "01_scene_accuracy_uncertainty.csv"
SAMPLED_PIXELS = STAGE26 / "03_sampled_pixel_results.csv"

PAIRED_EFFECTS = STAGE27 / "02_paired_scene_bootstrap_effects.csv"
SUBGROUP_RESULTS = STAGE27 / "03_city_sensor_subgroup_results.csv"
UNCERTAINTY_RELIABILITY = STAGE27 / "04_uncertainty_reliability.csv"
STAGE27_VERDICT = STAGE27 / "06_stage27_analysis_verdict.txt"

OUT = BASE / "ThermoFusion_Stage28_AdvancedFigures"
OUT.mkdir(parents=True, exist_ok=True)

FIGURE_PNG = OUT / "thermofusion_expanded_temporal_transfer_diagnostics_atlas.png"
FIGURE_PDF = OUT / "thermofusion_expanded_temporal_transfer_diagnostics_atlas.pdf"
CAPTION_TXT = OUT / "thermofusion_expanded_temporal_transfer_diagnostics_caption.txt"
SOURCE_TABLE_CSV = OUT / "source_scene_transfer_table.csv"

FINAL_MODEL = "frozen_without_terrain_unet_tta"
CITY_SENSOR_BASELINE = "city_sensor_training_climatology"

CITIES = ["Abidjan", "Accra", "Freetown", "Lagos"]
SENSORS = ["ECOSTRESS", "Landsat"]

for path in [
    SCENE_METRICS,
    SAMPLED_PIXELS,
    PAIRED_EFFECTS,
    SUBGROUP_RESULTS,
    UNCERTAINTY_RELIABILITY,
    STAGE27_VERDICT,
]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found:\n{path}")

if "Overall Stage 27 analysis verdict: PASS" not in STAGE27_VERDICT.read_text(
    encoding="utf-8"
):
    raise RuntimeError("Stage 27 analysis did not report PASS.")

scene_metrics = pd.read_csv(SCENE_METRICS)
paired = pd.read_csv(PAIRED_EFFECTS)
subgroups = pd.read_csv(SUBGROUP_RESULTS)
reliability = pd.read_csv(UNCERTAINTY_RELIABILITY)

# ------------------------------------------------------------------------------------------
# Create paired U-Net versus city–sensor-climatology table
# ------------------------------------------------------------------------------------------
unet = scene_metrics.loc[
    scene_metrics["model"].eq(FINAL_MODEL)
].copy()

climatology = scene_metrics.loc[
    scene_metrics["model"].eq(CITY_SENSOR_BASELINE),
    ["export_id", "record_id", "mae_c", "rmse_c"],
].copy()

climatology = climatology.rename(columns={
    "mae_c": "climatology_mae_c",
    "rmse_c": "climatology_rmse_c",
})

paired_scene = unet.merge(
    climatology,
    on=["export_id", "record_id"],
    how="left",
    validate="one_to_one",
)

if len(paired_scene) != 38:
    raise RuntimeError("Expected 38 paired scene comparisons.")

paired_scene["rmse_advantage_c"] = (
    paired_scene["climatology_rmse_c"]
    - paired_scene["rmse_c"]
)

paired_scene["short_scene_id"] = (
    paired_scene["city"].str.slice(0, 3)
    + " | "
    + paired_scene["thermal_sensor"].map({
        "ECOSTRESS": "ECO",
        "Landsat": "LAN",
    })
    + " | "
    + paired_scene["export_id"]
)

paired_scene = paired_scene.sort_values(
    "rmse_advantage_c",
    ascending=True,
).reset_index(drop=True)

paired_scene.to_csv(SOURCE_TABLE_CSV, index=False)

# ------------------------------------------------------------------------------------------
# Style
# ------------------------------------------------------------------------------------------
sns.set_theme(style="whitegrid", context="notebook")

COLOURS = {
    "ECOSTRESS": "#7B3F98",
    "Landsat": "#D66A4A",
    "unet": "#159A8A",
    "climatology": "#436FA6",
    "positive": "#159A8A",
    "negative": "#C44E52",
    "dark": "#17365D",
    "neutral": "#6B7280",
}

marker_map = {
    "ECOSTRESS": "o",
    "Landsat": "s",
}

# ------------------------------------------------------------------------------------------
# Figure layout
# ------------------------------------------------------------------------------------------
fig = plt.figure(
    figsize=(23, 16),
    constrained_layout=False,
    facecolor="white",
)

gs = GridSpec(
    nrows=3,
    ncols=4,
    figure=fig,
    height_ratios=[1.28, 1.05, 1.05],
    width_ratios=[1.45, 1.15, 1.10, 1.20],
    hspace=0.42,
    wspace=0.42,
)

ax_a = fig.add_subplot(gs[0, 0:2])   # Paired scene slope chart
ax_b = fig.add_subplot(gs[0, 2])     # Skill–coverage plane
ax_c = fig.add_subplot(gs[0, 3])     # Forest plot
ax_d = fig.add_subplot(gs[1, 0:2])   # Bivariate city-sensor matrix
ax_e = fig.add_subplot(gs[1, 2:4])   # Reliability curve
ax_f = fig.add_subplot(gs[2, 0:2])   # Coverage distributions
ax_g = fig.add_subplot(gs[2, 2:4])   # Paired effect distribution

# ------------------------------------------------------------------------------------------
# (a) Paired per-scene RMSE slope chart
# ------------------------------------------------------------------------------------------
y_positions = np.arange(len(paired_scene))

for y, row in paired_scene.iterrows():
    better = row["rmse_advantage_c"] >= 0
    line_colour = COLOURS["positive"] if better else COLOURS["negative"]

    ax_a.plot(
        [row["rmse_c"], row["climatology_rmse_c"]],
        [y, y],
        color=line_colour,
        linewidth=1.7,
        alpha=0.72,
        zorder=1,
    )

    ax_a.scatter(
        row["rmse_c"],
        y,
        color=COLOURS["unet"],
        marker=marker_map[row["thermal_sensor"]],
        edgecolor="white",
        linewidth=0.6,
        s=55,
        zorder=3,
    )

    ax_a.scatter(
        row["climatology_rmse_c"],
        y,
        color=COLOURS["climatology"],
        marker=marker_map[row["thermal_sensor"]],
        edgecolor="white",
        linewidth=0.6,
        s=55,
        zorder=3,
    )

ax_a.set_yticks(y_positions)
ax_a.set_yticklabels(paired_scene["short_scene_id"], fontsize=8.2)
ax_a.set(
    title="(a) Paired scene-level transfer response",
    xlabel="Scene RMSE (°C)",
    ylabel="Independent temporal-test scene",
)

ax_a.invert_yaxis()

legend_elements = [
    Line2D(
        [0], [0],
        marker="o",
        color="none",
        markerfacecolor=COLOURS["unet"],
        markeredgecolor="white",
        markersize=8,
        label="Frozen U-Net + TTA",
    ),
    Line2D(
        [0], [0],
        marker="o",
        color="none",
        markerfacecolor=COLOURS["climatology"],
        markeredgecolor="white",
        markersize=8,
        label="City–sensor climatology",
    ),
    Line2D(
        [0], [0],
        color=COLOURS["positive"],
        linewidth=2,
        label="U-Net lower RMSE",
    ),
    Line2D(
        [0], [0],
        color=COLOURS["negative"],
        linewidth=2,
        label="Climatology lower RMSE",
    ),
]

ax_a.legend(
    handles=legend_elements,
    frameon=False,
    ncol=2,
    fontsize=8.6,
    loc="lower right",
)

# ------------------------------------------------------------------------------------------
# (b) Skill–coverage plane: accuracy, calibration and width in one panel
# ------------------------------------------------------------------------------------------
for sensor in SENSORS:
    group = unet.loc[unet["thermal_sensor"].eq(sensor)]

    ax_b.scatter(
        group["rmse_c"],
        group["interval_coverage"],
        s=np.clip(group["mean_interval_width_c"] * 19, 30, 360),
        c=COLOURS[sensor],
        marker=marker_map[sensor],
        alpha=0.80,
        edgecolor="white",
        linewidth=0.7,
        label=sensor,
    )

for _, row in unet.iterrows():
    ax_b.annotate(
        row["city"][0],
        (row["rmse_c"], row["interval_coverage"]),
        xytext=(3, 3),
        textcoords="offset points",
        fontsize=7.5,
        color="#222222",
    )

ax_b.axhline(
    0.90,
    color="black",
    linestyle="--",
    linewidth=1.1,
    label="Nominal 90%",
)

ax_b.axvline(
    unet["rmse_c"].median(),
    color="#8A8A8A",
    linestyle=":",
    linewidth=1,
)

ax_b.set(
    title="(b) Accuracy–coverage transfer plane",
    xlabel="Scene RMSE (°C)",
    ylabel="Observed interval coverage",
    ylim=(-0.02, 1.05),
)

ax_b.legend(
    frameon=False,
    fontsize=8.5,
    loc="upper right",
)

ax_b.text(
    0.02,
    0.02,
    "Marker area = mean interval width",
    transform=ax_b.transAxes,
    fontsize=8,
    color="#4B5563",
)

# ------------------------------------------------------------------------------------------
# (c) Paired bootstrap forest plot
# ------------------------------------------------------------------------------------------
forest = paired.copy()

forest = forest.sort_values(
    "mean_paired_rmse_advantage_c",
    ascending=True,
).reset_index(drop=True)

forest_y = np.arange(len(forest))

for y, (_, row) in zip(forest_y, forest.iterrows()):
    value = row["mean_paired_rmse_advantage_c"]
    left = value - row["rmse_ci_lower_c"]
    right = row["rmse_ci_upper_c"] - value

    colour = (
        COLOURS["positive"]
        if row["rmse_ci_lower_c"] > 0
        else COLOURS["neutral"]
    )

    ax_c.errorbar(
        value,
        y,
        xerr=np.array([[left], [right]]),
        fmt="o",
        color=colour,
        ecolor=colour,
        capsize=4,
        markersize=7,
        linewidth=2,
    )

ax_c.axvline(0, color="black", linestyle="--", linewidth=1)

forest_labels = forest["baseline"].map({
    "global_training_climatology": "Global climatology",
    "sensor_training_climatology": "Sensor climatology",
    "city_sensor_training_climatology": "City–sensor climatology",
})

ax_c.set_yticks(forest_y)
ax_c.set_yticklabels(forest_labels, fontsize=9)
ax_c.set(
    title="(c) Paired bootstrap evidence",
    xlabel="Baseline RMSE − U-Net RMSE (°C)",
    ylabel="",
)

ax_c.text(
    0.02,
    0.02,
    "Positive values favour the U-Net\nPoints: mean; bars: 95% bootstrap CI",
    transform=ax_c.transAxes,
    fontsize=8,
    color="#4B5563",
)

# ------------------------------------------------------------------------------------------
# (d) Bivariate city–sensor matrix: RMSE colour + coverage + paired advantage
# ------------------------------------------------------------------------------------------
rmse_matrix = subgroups.pivot(
    index="city",
    columns="thermal_sensor",
    values="scene_macro_rmse_c",
).reindex(index=CITIES, columns=SENSORS)

coverage_matrix = subgroups.pivot(
    index="city",
    columns="thermal_sensor",
    values="mean_interval_coverage",
).reindex(index=CITIES, columns=SENSORS)

advantage_matrix = subgroups.pivot(
    index="city",
    columns="thermal_sensor",
    values="unet_rmse_advantage_vs_city_sensor_climatology_c",
).reindex(index=CITIES, columns=SENSORS)

cmap = mpl.cm.viridis_r.copy()
cmap.set_bad("#F2F2F2")

im = ax_d.imshow(
    rmse_matrix.to_numpy(dtype=float),
    cmap=cmap,
    aspect="auto",
)

for i, city in enumerate(CITIES):
    for j, sensor in enumerate(SENSORS):
        rmse = rmse_matrix.loc[city, sensor]

        if pd.isna(rmse):
            ax_d.text(
                j,
                i,
                "Not\navailable",
                ha="center",
                va="center",
                fontsize=10,
                color="#6B7280",
            )
            continue

        coverage = coverage_matrix.loc[city, sensor]
        advantage = advantage_matrix.loc[city, sensor]

        text_colour = "white" if rmse > 4.5 else "#1F2937"
        sign = "+" if advantage >= 0 else "−"

        ax_d.text(
            j,
            i,
            (
                f"RMSE {rmse:.2f}°C\n"
                f"Cov. {coverage:.0%}\n"
                f"Δ {sign}{abs(advantage):.2f}°C"
            ),
            ha="center",
            va="center",
            fontsize=10.5,
            color=text_colour,
            fontweight="medium",
        )

ax_d.set_xticks(range(len(SENSORS)))
ax_d.set_xticklabels(SENSORS)

ax_d.set_yticks(range(len(CITIES)))
ax_d.set_yticklabels(CITIES)

ax_d.set_title("(d) City–sensor transfer mosaic")
ax_d.set_xlabel("")
ax_d.set_ylabel("")

colour_bar = fig.colorbar(
    im,
    ax=ax_d,
    fraction=0.046,
    pad=0.04,
)

colour_bar.set_label("Scene-macro U-Net RMSE (°C)")

ax_d.text(
    0.01,
    -0.18,
    "Cell labels: RMSE; observed 90% coverage; Δ = climatology RMSE − U-Net RMSE.",
    transform=ax_d.transAxes,
    fontsize=8.2,
    color="#4B5563",
)

# ------------------------------------------------------------------------------------------
# (e) Sensor-stratified uncertainty reliability
# ------------------------------------------------------------------------------------------
for sensor in SENSORS:
    group = reliability.loc[
        reliability["thermal_sensor"].eq(sensor)
    ].sort_values("mean_tta_std_c")

    ax_e.plot(
        group["mean_tta_std_c"],
        group["mean_absolute_error_c"],
        "o-",
        color=COLOURS[sensor],
        linewidth=2.5,
        markersize=5.5,
        label=sensor,
    )

ax_e.set(
    title="(e) Uncertainty–error reliability by thermal sensor",
    xlabel="Mean TTA standard deviation (°C)",
    ylabel="Mean absolute error (°C)",
)

ax_e.legend(frameon=False)

ax_e.text(
    0.02,
    0.03,
    "Each curve uses ten within-sensor uncertainty quantiles.",
    transform=ax_e.transAxes,
    fontsize=8.3,
    color="#4B5563",
)

# ------------------------------------------------------------------------------------------
# (f) Distribution of observed coverage, with nominal threshold
# ------------------------------------------------------------------------------------------
coverage_plot = unet.copy()

sns.violinplot(
    data=coverage_plot,
    x="thermal_sensor",
    y="interval_coverage",
    hue="thermal_sensor",
    palette={
        "ECOSTRESS": COLOURS["ECOSTRESS"],
        "Landsat": COLOURS["Landsat"],
    },
    inner=None,
    linewidth=0,
    legend=False,
    alpha=0.7,
    ax=ax_f,
)

sns.boxplot(
    data=coverage_plot,
    x="thermal_sensor",
    y="interval_coverage",
    width=0.27,
    showfliers=False,
    boxprops={"facecolor": "white", "zorder": 3},
    whiskerprops={"linewidth": 1.2},
    medianprops={"color": "black", "linewidth": 1.5},
    ax=ax_f,
)

sns.stripplot(
    data=coverage_plot,
    x="thermal_sensor",
    y="interval_coverage",
    hue="thermal_sensor",
    palette={
        "ECOSTRESS": "#3D1C4F",
        "Landsat": "#8D341F",
    },
    dodge=False,
    size=4.2,
    alpha=0.78,
    legend=False,
    ax=ax_f,
)

ax_f.axhline(
    0.90,
    color="black",
    linestyle="--",
    linewidth=1.2,
)

ax_f.set(
    title="(f) Interval coverage distribution",
    xlabel="",
    ylabel="Observed 90% interval coverage",
    ylim=(-0.02, 1.05),
)

ax_f.text(
    1.03,
    0.905,
    "Nominal 90%",
    fontsize=8.5,
    va="bottom",
)

# ------------------------------------------------------------------------------------------
# (g) Distribution of paired U-Net RMSE advantage
# ------------------------------------------------------------------------------------------
advantage = paired_scene["rmse_advantage_c"].to_numpy()

sns.histplot(
    advantage,
    bins=12,
    stat="count",
    color=COLOURS["dark"],
    alpha=0.75,
    edgecolor="white",
    ax=ax_g,
)

ax_g.axvline(0, color="black", linestyle="--", linewidth=1.2)
ax_g.axvline(
    advantage.mean(),
    color=COLOURS["positive"],
    linewidth=2.2,
    label=f"Mean Δ = {advantage.mean():.2f} °C",
)

ax_g.set(
    title="(g) Distribution of paired RMSE advantage",
    xlabel="City–sensor climatology RMSE − U-Net RMSE (°C)",
    ylabel="Scenes",
)

ax_g.legend(frameon=False)

ax_g.text(
    0.02,
    0.92,
    "Positive values favour the U-Net.",
    transform=ax_g.transAxes,
    fontsize=8.5,
    va="top",
    color="#4B5563",
)

# ------------------------------------------------------------------------------------------
# Global title and export
# ------------------------------------------------------------------------------------------
fig.suptitle(
    "Independent temporal-transfer diagnostics for ThermoFusion",
    fontsize=20,
    fontweight="bold",
    y=0.995,
)

fig.text(
    0.5,
    0.004,
    (
        "Frozen validation-selected without-terrain U-Net; 38 later, high-coverage "
        "city–sensor scenes; eight-view test-time augmentation; "
        "interval calibration fixed from the original validation set."
    ),
    ha="center",
    fontsize=10,
    color="#4B5563",
)

fig.savefig(
    FIGURE_PNG,
    dpi=900,
    bbox_inches="tight",
    facecolor="white",
)

fig.savefig(
    FIGURE_PDF,
    bbox_inches="tight",
    facecolor="white",
)

plt.show()

# ------------------------------------------------------------------------------------------
# Exact manuscript caption
# ------------------------------------------------------------------------------------------
unet_macro_rmse = unet["rmse_c"].mean()
unet_macro_mae = unet["mae_c"].mean()
mean_coverage = unet["interval_coverage"].mean()

city_sensor_effect = paired.loc[
    paired["baseline"].eq("city_sensor_training_climatology")
].iloc[0]

global_effect = paired.loc[
    paired["baseline"].eq("global_training_climatology")
].iloc[0]

caption = (
    "Figure Sx. Independent expanded temporal-transfer diagnostics for the frozen "
    "ThermoFusion model. (a) Paired scene-level RMSE comparison between the "
    "validation-selected without-terrain U-Net with eight-view test-time augmentation "
    "and the training-derived city–sensor climatology; connecting lines are teal where "
    "the U-Net has lower RMSE and red where climatology is lower. (b) The accuracy–coverage "
    "plane relates scene RMSE to observed coverage of nominal 90% prediction intervals; "
    "marker area denotes mean interval width and letters identify cities. (c) Paired "
    "scene-bootstrap effects against fixed training climatologies; positive differences "
    "favour the U-Net. (d) City–sensor transfer mosaic reporting scene-macro RMSE, observed "
    "interval coverage and RMSE difference relative to the city–sensor climatology. "
    "(e) Sensor-stratified uncertainty–error reliability across within-sensor uncertainty "
    "quantiles. (f) Distribution of observed coverage by sensor, with the nominal 90% "
    "reference. (g) Distribution of paired RMSE differences relative to the city–sensor "
    "climatology. Across 38 later high-coverage scenes, the frozen U-Net attained a "
    f"scene-macro MAE of {unet_macro_mae:.2f} °C and RMSE of {unet_macro_rmse:.2f} °C, "
    f"while nominal 90% intervals achieved {mean_coverage:.1%} observed coverage. "
    "The U-Net improved over the global climatology "
    f"(mean RMSE advantage {global_effect['mean_paired_rmse_advantage_c']:.2f} °C; "
    f"95% bootstrap CI {global_effect['rmse_ci_lower_c']:.2f} to "
    f"{global_effect['rmse_ci_upper_c']:.2f} °C), but its advantage over the stronger "
    "city–sensor climatology was not conclusive "
    f"(mean {city_sensor_effect['mean_paired_rmse_advantage_c']:.2f} °C; "
    f"95% bootstrap CI {city_sensor_effect['rmse_ci_lower_c']:.2f} to "
    f"{city_sensor_effect['rmse_ci_upper_c']:.2f} °C)."
)

CAPTION_TXT.write_text(caption, encoding="utf-8")

print("=" * 100)
print("THERMOFUSION STAGE 28 — ADVANCED FIGURE COMPLETE")
print("=" * 100)
print(f"High-resolution PNG: {FIGURE_PNG}")
print(f"Vector PDF: {FIGURE_PDF}")
print(f"Figure caption: {CAPTION_TXT}")
print(f"Source scene table: {SOURCE_TABLE_CSV}")
print()
print(caption)
