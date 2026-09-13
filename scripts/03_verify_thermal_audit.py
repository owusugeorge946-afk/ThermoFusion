# =====================================================================
# ThermoFusion reproducibility code
# Source notebook cell(s): 6
# Audit status: KEEP
# Extracted without scientific-logic modification.
# Replace local Google Drive paths as described in README.md when needed.
# =====================================================================

# ThermoFusion Stage 03: verify and analyse all Stage 02 audit exports.
# Run this complete script in one Google Colab cell after Stage 02 finishes.

from google.colab import drive
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D


drive.mount("/content/drive")

INPUT_DIR = Path("/content/drive/MyDrive/ThermoFusion_Stage02_Audit")
OUTPUT_DIR = Path("/content/drive/MyDrive/ThermoFusion_Stage03_Verification")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_CITIES = ["Accra", "Lagos", "Abidjan", "Freetown"]
EXPECTED_SENSORS = ["Landsat", "ECOSTRESS"]
REQUIRED_COLUMNS = {
    "city",
    "sensor",
    "scene_id",
    "date",
    "year",
    "month",
    "valid_area_km2",
    "valid_fraction_pct",
}
THRESHOLDS = [0, 10, 25, 50, 75]


if not INPUT_DIR.exists():
    raise FileNotFoundError(
        f"Input folder not found: {INPUT_DIR}\n"
        "Confirm that all Earth Engine exports finished and check the Drive folder name."
    )

csv_files = sorted(INPUT_DIR.glob("*.csv"))
print(f"Found {len(csv_files)} CSV files in {INPUT_DIR}")

if len(csv_files) != 8:
    print("WARNING: Eight CSV files were expected.")


frames = []
file_checks = []

for path in csv_files:
    try:
        frame = pd.read_csv(path)
        missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
        status = "PASS" if not missing and len(frame) > 0 else "FAIL"

        file_checks.append(
            {
                "file": path.name,
                "rows": len(frame),
                "columns": len(frame.columns),
                "missing_columns": ", ".join(missing),
                "status": status,
            }
        )

        if not missing and len(frame) > 0:
            frame["source_file"] = path.name
            frames.append(frame)

    except Exception as error:
        file_checks.append(
            {
                "file": path.name,
                "rows": 0,
                "columns": 0,
                "missing_columns": str(error),
                "status": "FAIL",
            }
        )

file_check_df = pd.DataFrame(file_checks)
display(file_check_df)
file_check_df.to_csv(OUTPUT_DIR / "01_file_integrity_check.csv", index=False)

if not frames:
    raise RuntimeError("No valid audit tables were available for analysis.")

audit = pd.concat(frames, ignore_index=True)
audit["city"] = audit["city"].astype(str).str.strip()
audit["sensor"] = audit["sensor"].astype(str).str.strip()
audit["date"] = pd.to_datetime(audit["date"], errors="coerce")

for column in ["year", "month", "valid_area_km2", "valid_fraction_pct"]:
    audit[column] = pd.to_numeric(audit[column], errors="coerce")

audit["valid_fraction_pct"] = audit["valid_fraction_pct"].replace(
    [np.inf, -np.inf], np.nan
)


# Verify that every expected city-sensor combination is represented.
combination_rows = []
for city in EXPECTED_CITIES:
    for sensor in EXPECTED_SENSORS:
        subset = audit[(audit["city"] == city) & (audit["sensor"] == sensor)]
        combination_rows.append(
            {
                "city": city,
                "sensor": sensor,
                "rows": len(subset),
                "non_null_coverage": subset["valid_fraction_pct"].notna().sum(),
                "positive_coverage": subset["valid_fraction_pct"].gt(0).sum(),
                "status": "PASS" if len(subset) > 0 else "MISSING",
            }
        )

combination_df = pd.DataFrame(combination_rows)
display(combination_df)
combination_df.to_csv(OUTPUT_DIR / "02_city_sensor_completeness.csv", index=False)


# Detect impossible or suspicious values without silently changing them.
quality_flags = audit[
    audit["valid_fraction_pct"].isna()
    | audit["valid_fraction_pct"].lt(0)
    | audit["valid_fraction_pct"].gt(105)
    | audit["date"].isna()
].copy()
quality_flags.to_csv(OUTPUT_DIR / "03_suspicious_records.csv", index=False)


# Coverage distribution and threshold counts.
summary = (
    audit.groupby(["city", "sensor"], observed=True)
    .agg(
        candidate_scenes=("scene_id", "count"),
        valid_coverage_records=("valid_fraction_pct", "count"),
        zero_coverage_scenes=("valid_fraction_pct", lambda x: x.eq(0).sum()),
        median_coverage_pct=("valid_fraction_pct", "median"),
        mean_coverage_pct=("valid_fraction_pct", "mean"),
        p25_coverage_pct=("valid_fraction_pct", lambda x: x.quantile(0.25)),
        p75_coverage_pct=("valid_fraction_pct", lambda x: x.quantile(0.75)),
        maximum_coverage_pct=("valid_fraction_pct", "max"),
    )
    .reset_index()
)

for threshold in THRESHOLDS:
    counts = (
        audit.assign(eligible=audit["valid_fraction_pct"].ge(threshold))
        .groupby(["city", "sensor"], observed=True)["eligible"]
        .sum()
        .rename(f"scenes_ge_{threshold}pct")
        .reset_index()
    )
    summary = summary.merge(counts, on=["city", "sensor"], how="left")

summary.to_csv(OUTPUT_DIR / "04_coverage_summary.csv", index=False)
display(summary.round(2))


# Monthly availability at the preliminary 25% valid-coverage threshold.
eligible = audit[audit["valid_fraction_pct"].ge(25)].copy()
monthly = (
    eligible.groupby(["city", "sensor", "month"], observed=True)
    .size()
    .rename("eligible_scenes")
    .reset_index()
)
monthly.to_csv(OUTPUT_DIR / "05_monthly_eligible_scenes.csv", index=False)


# Advanced multipanel verification figure.
sns.set_theme(style="whitegrid", context="notebook")
fig = plt.figure(figsize=(17, 13), constrained_layout=True)
grid = fig.add_gridspec(2, 2)

# Panel a: eligible-scene matrix.
ax1 = fig.add_subplot(grid[0, 0])
matrix = eligible.groupby(["city", "sensor"]).size().unstack(fill_value=0)
matrix = matrix.reindex(index=EXPECTED_CITIES, columns=EXPECTED_SENSORS, fill_value=0)
sns.heatmap(
    matrix,
    annot=True,
    fmt="g",
    cmap="mako",
    linewidths=0.8,
    linecolor="white",
    cbar_kws={"label": "Scenes with at least 25% valid coverage"},
    ax=ax1,
)
ax1.set_xlabel("")
ax1.set_ylabel("")
ax1.set_title("(a) Usable thermal observations")

# Panel b: monthly city-sensor matrix.
ax2 = fig.add_subplot(grid[0, 1])
monthly_matrix = monthly.pivot_table(
    index=["city", "sensor"],
    columns="month",
    values="eligible_scenes",
    fill_value=0,
)
monthly_matrix = monthly_matrix.reindex(columns=range(1, 13), fill_value=0)
sns.heatmap(
    monthly_matrix,
    cmap="rocket_r",
    linewidths=0.35,
    cbar_kws={"label": "Eligible scenes"},
    ax=ax2,
)
ax2.set_xlabel("Month")
ax2.set_ylabel("")
ax2.set_title("(b) Seasonal observation availability")

# Panel c: coverage distributions.
ax3 = fig.add_subplot(grid[1, 0])
plot_data = audit.dropna(subset=["valid_fraction_pct"]).copy()
sns.violinplot(
    data=plot_data,
    x="city",
    y="valid_fraction_pct",
    hue="sensor",
    split=True,
    inner="quart",
    cut=0,
    density_norm="width",
    palette={"Landsat": "#2166ac", "ECOSTRESS": "#b2182b"},
    ax=ax3,
)
ax3.axhline(25, color="black", linestyle="--", linewidth=1.2, label="25% threshold")
ax3.set_xlabel("")
ax3.set_ylabel("Valid AOI coverage (%)")
ax3.set_title("(c) Scene-level valid-coverage distributions")
handles, labels = ax3.get_legend_handles_labels()
unique = dict(zip(labels, handles))
ax3.legend(unique.values(), unique.keys(), frameon=False, loc="upper right")

# Panel d: empirical coverage distributions. Draw manually for compatibility
# with Colab environments whose Seaborn ecdfplot does not support style=.
ax4 = fig.add_subplot(grid[1, 1])
sensor_colours = {"Landsat": "#2166ac", "ECOSTRESS": "#b2182b"}
city_styles = {
    "Accra": "-",
    "Lagos": "--",
    "Abidjan": "-.",
    "Freetown": ":",
}

for (city, sensor), group in plot_data.groupby(["city", "sensor"], observed=True):
    values = np.sort(group["valid_fraction_pct"].dropna().to_numpy())
    if len(values) == 0:
        continue
    cumulative = np.arange(1, len(values) + 1) / len(values)
    ax4.step(
        values,
        cumulative,
        where="post",
        color=sensor_colours.get(sensor, "#555555"),
        linestyle=city_styles.get(city, "-"),
        linewidth=1.8,
        alpha=0.9,
    )

ax4.axvline(25, color="black", linestyle="--", linewidth=1.2)
ax4.set_xlabel("Valid AOI coverage (%)")
ax4.set_ylabel("Cumulative proportion of scenes")
ax4.set_title("(d) Cross-city empirical coverage functions")
ax4.set_xlim(0, 100)
ax4.set_ylim(0, 1)

sensor_legend = [
    Line2D([0], [0], color=colour, linewidth=2.2, label=sensor)
    for sensor, colour in sensor_colours.items()
]
city_legend = [
    Line2D([0], [0], color="black", linestyle=style, linewidth=1.8, label=city)
    for city, style in city_styles.items()
]
legend1 = ax4.legend(
    handles=sensor_legend,
    title="Sensor",
    frameon=False,
    loc="lower right",
)
ax4.add_artist(legend1)
ax4.legend(
    handles=city_legend,
    title="City",
    frameon=False,
    loc="center right",
)

figure_path = OUTPUT_DIR / "thermofusion_stage03_verification_dashboard.png"
fig.savefig(figure_path, dpi=800, bbox_inches="tight", facecolor="white")
plt.show()


# Machine-readable verdict.
ecostress = audit[audit["sensor"] == "ECOSTRESS"]
ecostress_usable = ecostress[ecostress["valid_fraction_pct"].ge(25)]
missing_combinations = combination_df[combination_df["status"] != "PASS"]

verdict_lines = [
    "THERMOFUSION STAGE 03 VERIFICATION",
    f"CSV files found: {len(csv_files)} of 8 expected",
    f"Combined audit records: {len(audit)}",
    f"Suspicious records: {len(quality_flags)}",
    f"ECOSTRESS scenes with >=25% valid coverage: {len(ecostress_usable)}",
    f"Missing city-sensor combinations: {len(missing_combinations)}",
]

if len(csv_files) == 8 and len(missing_combinations) == 0:
    verdict_lines.append("File completeness verdict: PASS")
else:
    verdict_lines.append("File completeness verdict: REVIEW REQUIRED")

if len(ecostress_usable) > 0:
    verdict_lines.append("ECOSTRESS pixel-coverage verdict: USABLE SCENES PRESENT")
else:
    verdict_lines.append("ECOSTRESS pixel-coverage verdict: NO SCENES PASS 25% THRESHOLD")

verdict_text = "\n".join(verdict_lines)
(OUTPUT_DIR / "06_verification_verdict.txt").write_text(verdict_text, encoding="utf-8")
print("\n" + verdict_text)


# Package all verification outputs for convenient sharing.
zip_base = Path("/content/drive/MyDrive/ThermoFusion_Stage03_Verification")
zip_path = shutil.make_archive(str(zip_base), "zip", root_dir=OUTPUT_DIR)

print(f"\nVerification outputs: {OUTPUT_DIR}")
print(f"ZIP package: {zip_path}")
print("Upload the ZIP package for the next stage.")
