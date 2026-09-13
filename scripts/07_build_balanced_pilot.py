# =====================================================================
# ThermoFusion reproducibility code
# Source notebook cell(s): 14
# Audit status: KEEP
# Extracted without scientific-logic modification.
# Replace local Google Drive paths as described in README.md when needed.
# =====================================================================


# Run in one Google Colab cell after Stage 06 verification passes.

from google.colab import drive
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


drive.mount("/content/drive")

INPUT_PATH = Path(
    "/content/drive/MyDrive/ThermoFusion_Stage06_Match_Verification/"
    "07_verified_match_manifest.csv"
)
OUTPUT_DIR = Path("/content/drive/MyDrive/ThermoFusion_Stage07_Pilot")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SCENES_PER_QUARTER = 2
MIN_THERMAL_COVERAGE = 25.0
MAX_S2_SCENE_CLOUD = 80.0
EXPECTED_CITIES = ["Accra", "Lagos", "Abidjan", "Freetown"]
EXPECTED_SENSORS = ["Landsat", "ECOSTRESS"]


if not INPUT_PATH.exists():
    raise FileNotFoundError(f"Verified Stage 06 manifest not found: {INPUT_PATH}")

data = pd.read_csv(INPUT_PATH)
data["thermal_datetime_utc"] = pd.to_datetime(
    data["thermal_datetime_utc"], errors="raise", utc=True
)

numeric_columns = [
    "valid_fraction_pct",
    "s2_cloudy_pixel_percentage",
    "s2_time_difference_hours",
    "s1_time_difference_hours",
    "era5_time_difference_hours",
    "all_three_found",
    "strict_temporal_match",
]
for column in numeric_columns:
    data[column] = pd.to_numeric(data[column], errors="coerce")

data["year"] = data["thermal_datetime_utc"].dt.year
data["month"] = data["thermal_datetime_utc"].dt.month
data["quarter"] = data["thermal_datetime_utc"].dt.quarter


# Pilot scenes must have all auxiliary sources, tight temporal proximity,
# adequate thermal coverage, and a Sentinel-2 scene below 80% catalogue cloud.
eligible = data[
    data["all_three_found"].eq(1)
    & data["strict_temporal_match"].eq(1)
    & data["valid_fraction_pct"].ge(MIN_THERMAL_COVERAGE)
    & data["s2_cloudy_pixel_percentage"].le(MAX_S2_SCENE_CLOUD)
].copy()


score_groups = eligible.groupby(["city", "thermal_sensor"], observed=True)
coverage_rank = score_groups["valid_fraction_pct"].rank(pct=True, method="average")
cloud_rank = 1.0 - score_groups["s2_cloudy_pixel_percentage"].rank(
    pct=True, method="average"
)
s2_rank = 1.0 - score_groups["s2_time_difference_hours"].rank(
    pct=True, method="average"
)
s1_rank = 1.0 - score_groups["s1_time_difference_hours"].rank(
    pct=True, method="average"
)
era5_rank = 1.0 - score_groups["era5_time_difference_hours"].rank(
    pct=True, method="average"
)

eligible["pilot_quality_score"] = (
    0.35 * coverage_rank
    + 0.25 * cloud_rank
    + 0.20 * s2_rank
    + 0.15 * s1_rank
    + 0.05 * era5_rank
)


selected_parts = []
selection_audit = []

for city in EXPECTED_CITIES:
    for sensor in EXPECTED_SENSORS:
        group = eligible[
            eligible["city"].eq(city)
            & eligible["thermal_sensor"].eq(sensor)
        ].copy()

        for quarter in range(1, 5):
            candidates = group[group["quarter"].eq(quarter)].sort_values(
                ["pilot_quality_score", "valid_fraction_pct"],
                ascending=[False, False],
            )

            # Prefer different years to avoid selecting nearly identical dates.
            diverse = candidates.drop_duplicates("year", keep="first")
            chosen = diverse.head(SCENES_PER_QUARTER)

            if len(chosen) < SCENES_PER_QUARTER:
                remaining = candidates[~candidates["record_id"].isin(chosen["record_id"])]
                chosen = pd.concat(
                    [chosen, remaining.head(SCENES_PER_QUARTER - len(chosen))],
                    ignore_index=False,
                )

            selected_parts.append(chosen)
            selection_audit.append(
                {
                    "city": city,
                    "thermal_sensor": sensor,
                    "quarter": quarter,
                    "eligible_candidates": len(candidates),
                    "selected": len(chosen),
                    "status": (
                        "PASS"
                        if len(chosen) == SCENES_PER_QUARTER
                        else "INSUFFICIENT"
                    ),
                }
            )

pilot = pd.concat(selected_parts, ignore_index=True)
selection_audit_df = pd.DataFrame(selection_audit)

pilot = pilot.drop_duplicates("record_id").sort_values(
    ["city", "thermal_sensor", "quarter", "thermal_datetime_utc"]
)
pilot["pilot_id"] = [f"TFP{i:04d}" for i in range(1, len(pilot) + 1)]


expected_pilot_size = (
    len(EXPECTED_CITIES)
    * len(EXPECTED_SENSORS)
    * 4
    * SCENES_PER_QUARTER
)

if (selection_audit_df["status"] != "PASS").any():
    print("WARNING: At least one city-sensor-quarter group was insufficient.")

pilot.to_csv(OUTPUT_DIR / "01_balanced_pilot_manifest.csv", index=False)
selection_audit_df.to_csv(OUTPUT_DIR / "02_pilot_selection_audit.csv", index=False)


summary = (
    pilot.groupby(["city", "thermal_sensor"], observed=True)
    .agg(
        selected_scenes=("record_id", "count"),
        distinct_years=("year", "nunique"),
        median_thermal_coverage_pct=("valid_fraction_pct", "median"),
        median_s2_cloud_pct=("s2_cloudy_pixel_percentage", "median"),
        median_s2_offset_h=("s2_time_difference_hours", "median"),
        median_s1_offset_h=("s1_time_difference_hours", "median"),
    )
    .reset_index()
)
summary.to_csv(OUTPUT_DIR / "03_pilot_summary.csv", index=False)


# Compact high-information pilot-design figure.
sns.set_theme(style="whitegrid", context="notebook")
fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

count_matrix = pilot.pivot_table(
    index=["city", "thermal_sensor"],
    columns="quarter",
    values="record_id",
    aggfunc="count",
    fill_value=0,
)
count_matrix = count_matrix.reindex(columns=[1, 2, 3, 4], fill_value=0)
sns.heatmap(
    count_matrix,
    annot=True,
    fmt="g",
    cmap="crest",
    vmin=0,
    vmax=SCENES_PER_QUARTER,
    linewidths=0.5,
    cbar_kws={"label": "Selected scenes"},
    ax=axes[0],
)
axes[0].set_title("(a) Seasonal balance")
axes[0].set_xlabel("Quarter")
axes[0].set_ylabel("")

sns.scatterplot(
    data=pilot,
    x="s2_time_difference_hours",
    y="s1_time_difference_hours",
    hue="city",
    style="thermal_sensor",
    size="valid_fraction_pct",
    sizes=(35, 180),
    alpha=0.75,
    ax=axes[1],
)
axes[1].axvline(72, color="black", linestyle="--", linewidth=1)
axes[1].axhline(144, color="black", linestyle="--", linewidth=1)
axes[1].set_title("(b) Pilot temporal offsets")
axes[1].set_xlabel("Sentinel-2 offset (hours)")
axes[1].set_ylabel("Sentinel-1 offset (hours)")
axes[1].legend(frameon=False, fontsize=8, ncol=2)

score_matrix = pilot.pivot_table(
    index="city",
    columns="thermal_sensor",
    values="pilot_quality_score",
    aggfunc="median",
)
sns.heatmap(
    score_matrix,
    annot=True,
    fmt=".3f",
    cmap="mako",
    linewidths=0.7,
    cbar_kws={"label": "Median selection score"},
    ax=axes[2],
)
axes[2].set_title("(c) Pilot quality balance")
axes[2].set_xlabel("")
axes[2].set_ylabel("")

figure_path = OUTPUT_DIR / "thermofusion_stage07_pilot_design.png"
fig.savefig(figure_path, dpi=800, bbox_inches="tight", facecolor="white")
plt.show()


verdict = [
    "THERMOFUSION STAGE 07 PILOT SELECTION",
    f"Eligible scenes before balancing: {len(eligible)}",
    f"Selected pilot scenes: {len(pilot)}/{expected_pilot_size}",
    f"Unique pilot record IDs: {pilot['record_id'].nunique()}",
    f"Insufficient city-sensor-quarter groups: {(selection_audit_df['status'] != 'PASS').sum()}",
    "Pilot selection: PASS" if len(pilot) == expected_pilot_size else "Pilot selection: REVIEW REQUIRED",
]
(OUTPUT_DIR / "04_pilot_verdict.txt").write_text(
    "\n".join(verdict), encoding="utf-8"
)

zip_path = shutil.make_archive(
    "/content/drive/MyDrive/ThermoFusion_Stage07_Pilot",
    "zip",
    root_dir=OUTPUT_DIR,
)

print("\n" + "\n".join(verdict))
print("\nPilot summary:")
display(summary.round(2))
print(f"\nOutput folder: {OUTPUT_DIR}")
print(f"ZIP package: {zip_path}")
