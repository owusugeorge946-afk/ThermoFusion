# =====================================================================
# ThermoFusion reproducibility code
# Source notebook cell(s): 13
# Audit status: KEEP
# Extracted without scientific-logic modification.
# Replace local Google Drive paths as described in README.md when needed.
# =====================================================================

# ThermoFusion Stage 06: verify Stage 05 multisensor temporal matches.
# Run in one Google Colab cell after all eight Stage 05 tasks complete.

from google.colab import drive
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


drive.mount("/content/drive")

MY_DRIVE = Path("/content/drive/MyDrive")
MATCH_DIR = MY_DRIVE / "ThermoFusion_Stage05_Matches"
MANIFEST_PATH = Path(
    "/content/drive/MyDrive/ThermoFusion_Stage04_Selection/"
    "02_selected_scene_manifest.csv"
)
OUTPUT_DIR = Path("/content/drive/MyDrive/ThermoFusion_Stage06_Match_Verification")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_FILES = 8
EXPECTED_RECORDS = 1998
S2_LIMIT_HOURS = 10 * 24
S1_LIMIT_HOURS = 12 * 24
ERA5_LIMIT_HOURS = 2


if not MANIFEST_PATH.exists():
    raise FileNotFoundError(f"Stage 04 manifest not found: {MANIFEST_PATH}")

expected_stems = [
    f"thermofusion_{city}_{sensor}_matches"
    for city in ["abidjan", "accra", "freetown", "lagos"]
    for sensor in ["ecostress", "landsat"]
]

# Earth Engine sometimes places exports in My Drive root or adds suffixes when
# a filename already exists. Find every expected export recursively and retain
# the newest copy of each exact city-sensor product.
files = []
discovery_rows = []
for stem in expected_stems:
    candidates = sorted(
        MY_DRIVE.rglob(f"{stem}*.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    selected_path = candidates[0] if candidates else None
    discovery_rows.append(
        {
            "expected_stem": stem,
            "copies_found": len(candidates),
            "selected_path": str(selected_path) if selected_path else "",
        }
    )
    if selected_path is not None:
        files.append(selected_path)

discovery_df = pd.DataFrame(discovery_rows)
display(discovery_df)
discovery_df.to_csv(OUTPUT_DIR / "00_stage05_file_discovery.csv", index=False)

print(f"Located {len(files)} of {EXPECTED_FILES} expected Stage 05 CSV files")
if len(files) != EXPECTED_FILES:
    missing_stems = discovery_df.loc[
        discovery_df["copies_found"].eq(0), "expected_stem"
    ].tolist()
    raise RuntimeError(
        "Could not locate all Stage 05 exports. Missing: "
        + ", ".join(missing_stems)
    )


required_columns = {
    "record_id",
    "city",
    "thermal_sensor",
    "thermal_scene_id",
    "thermal_datetime_utc",
    "valid_fraction_pct",
    "s2_found",
    "s2_scene_id",
    "s2_time_difference_hours",
    "s1_found",
    "s1_scene_id",
    "s1_time_difference_hours",
    "era5_found",
    "era5_scene_id",
    "era5_time_difference_hours",
}


frames = []
file_checks = []
for path in files:
    frame = pd.read_csv(path)
    missing = sorted(required_columns.difference(frame.columns))
    file_checks.append(
        {
            "file": path.name,
            "rows": len(frame),
            "columns": len(frame.columns),
            "missing_columns": ", ".join(missing),
            "status": "PASS" if not missing and len(frame) > 0 else "FAIL",
        }
    )
    if not missing and len(frame) > 0:
        frame["source_file"] = path.name
        frames.append(frame)

file_check_df = pd.DataFrame(file_checks)
display(file_check_df)
file_check_df.to_csv(OUTPUT_DIR / "01_stage05_file_integrity.csv", index=False)

if (file_check_df["status"] != "PASS").any():
    raise RuntimeError("At least one Stage 05 CSV failed schema validation.")

matches = pd.concat(frames, ignore_index=True)
manifest = pd.read_csv(MANIFEST_PATH)

for column in ["s2_found", "s1_found", "era5_found"]:
    matches[column] = pd.to_numeric(matches[column], errors="coerce").fillna(0).astype(int)

for column in [
    "valid_fraction_pct",
    "s2_time_difference_hours",
    "s1_time_difference_hours",
    "era5_time_difference_hours",
    "s2_cloudy_pixel_percentage",
]:
    if column in matches.columns:
        matches[column] = pd.to_numeric(matches[column], errors="coerce")

matches["thermal_datetime_utc"] = pd.to_datetime(
    matches["thermal_datetime_utc"], errors="coerce", utc=True
)


# Exact reconciliation with the Stage 04 manifest.
manifest_ids = set(manifest["record_id"].astype(str))
exported_ids = set(matches["record_id"].dropna().astype(str))
missing_ids = sorted(manifest_ids.difference(exported_ids))
unexpected_ids = sorted(exported_ids.difference(manifest_ids))
duplicate_mask = matches.duplicated("record_id", keep=False)
duplicate_records = matches.loc[duplicate_mask].sort_values("record_id")

pd.DataFrame({"missing_record_id": missing_ids}).to_csv(
    OUTPUT_DIR / "02_missing_record_ids.csv", index=False
)
pd.DataFrame({"unexpected_record_id": unexpected_ids}).to_csv(
    OUTPUT_DIR / "03_unexpected_record_ids.csv", index=False
)
duplicate_records.to_csv(OUTPUT_DIR / "04_duplicate_record_ids.csv", index=False)


# Flag temporal offsets outside the matching windows.
offset_flags = pd.Series(False, index=matches.index)
offset_flags |= matches["s2_found"].eq(1) & matches[
    "s2_time_difference_hours"
].gt(S2_LIMIT_HOURS + 0.001)
offset_flags |= matches["s1_found"].eq(1) & matches[
    "s1_time_difference_hours"
].gt(S1_LIMIT_HOURS + 0.001)
offset_flags |= matches["era5_found"].eq(1) & matches[
    "era5_time_difference_hours"
].gt(ERA5_LIMIT_HOURS + 0.001)
offset_flags |= matches["thermal_datetime_utc"].isna()
offset_anomalies = matches.loc[offset_flags].copy()
offset_anomalies.to_csv(OUTPUT_DIR / "05_offset_anomalies.csv", index=False)


matches["all_three_found"] = (
    matches[["s2_found", "s1_found", "era5_found"]].eq(1).all(axis=1)
)
matches["s2_within_3_days"] = matches["s2_found"].eq(1) & matches[
    "s2_time_difference_hours"
].le(72)
matches["s1_within_6_days"] = matches["s1_found"].eq(1) & matches[
    "s1_time_difference_hours"
].le(144)
matches["strict_temporal_match"] = (
    matches["s2_within_3_days"]
    & matches["s1_within_6_days"]
    & matches["era5_found"].eq(1)
)


summary_rows = []
for (city, sensor), group in matches.groupby(
    ["city", "thermal_sensor"], observed=True
):
    row = {
        "city": city,
        "thermal_sensor": sensor,
        "records": len(group),
        "s2_found_n": int(group["s2_found"].sum()),
        "s2_found_pct": 100 * group["s2_found"].mean(),
        "s1_found_n": int(group["s1_found"].sum()),
        "s1_found_pct": 100 * group["s1_found"].mean(),
        "era5_found_n": int(group["era5_found"].sum()),
        "era5_found_pct": 100 * group["era5_found"].mean(),
        "all_three_n": int(group["all_three_found"].sum()),
        "all_three_pct": 100 * group["all_three_found"].mean(),
        "strict_temporal_n": int(group["strict_temporal_match"].sum()),
        "strict_temporal_pct": 100 * group["strict_temporal_match"].mean(),
        "s2_offset_median_h": group.loc[
            group["s2_found"].eq(1), "s2_time_difference_hours"
        ].median(),
        "s2_offset_p90_h": group.loc[
            group["s2_found"].eq(1), "s2_time_difference_hours"
        ].quantile(0.90),
        "s1_offset_median_h": group.loc[
            group["s1_found"].eq(1), "s1_time_difference_hours"
        ].median(),
        "s1_offset_p90_h": group.loc[
            group["s1_found"].eq(1), "s1_time_difference_hours"
        ].quantile(0.90),
        "era5_offset_median_h": group.loc[
            group["era5_found"].eq(1), "era5_time_difference_hours"
        ].median(),
    }
    summary_rows.append(row)

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUTPUT_DIR / "06_matching_summary.csv", index=False)
matches.to_csv(OUTPUT_DIR / "07_verified_match_manifest.csv", index=False)
display(summary.round(2))


# Advanced multipanel diagnostic figure.
sns.set_theme(style="whitegrid", context="notebook")
fig, axes = plt.subplots(2, 2, figsize=(17, 13), constrained_layout=True)

rate_long = summary.melt(
    id_vars=["city", "thermal_sensor"],
    value_vars=["s2_found_pct", "s1_found_pct", "era5_found_pct", "all_three_pct"],
    var_name="match_type",
    value_name="match_rate_pct",
)
rate_long["row"] = rate_long["city"] + "–" + rate_long["thermal_sensor"]
rate_matrix = rate_long.pivot(
    index="row", columns="match_type", values="match_rate_pct"
)
rate_matrix = rate_matrix.reindex(
    columns=["s2_found_pct", "s1_found_pct", "era5_found_pct", "all_three_pct"]
)
rate_matrix.columns = ["Sentinel-2", "Sentinel-1", "ERA5-Land", "All three"]
sns.heatmap(
    rate_matrix,
    annot=True,
    fmt=".1f",
    vmin=0,
    vmax=100,
    cmap="viridis",
    linewidths=0.5,
    cbar_kws={"label": "Match rate (%)"},
    ax=axes[0, 0],
)
axes[0, 0].set_title("(a) Multisensor matching completeness")
axes[0, 0].set_xlabel("")
axes[0, 0].set_ylabel("")

offset_frames = []
for label, found, column in [
    ("Sentinel-2", "s2_found", "s2_time_difference_hours"),
    ("Sentinel-1", "s1_found", "s1_time_difference_hours"),
    ("ERA5-Land", "era5_found", "era5_time_difference_hours"),
]:
    part = matches.loc[matches[found].eq(1), ["city", "thermal_sensor", column]].copy()
    part = part.rename(columns={column: "offset_hours"})
    part["matched_source"] = label
    offset_frames.append(part)
offset_long = pd.concat(offset_frames, ignore_index=True)
sns.boxenplot(
    data=offset_long,
    x="matched_source",
    y="offset_hours",
    hue="thermal_sensor",
    palette={"Landsat": "#2166ac", "ECOSTRESS": "#b2182b"},
    showfliers=False,
    ax=axes[0, 1],
)
axes[0, 1].set_yscale("symlog", linthresh=1)
axes[0, 1].set_xlabel("")
axes[0, 1].set_ylabel("Absolute acquisition-time offset (hours)")
axes[0, 1].set_title("(b) Temporal-offset distributions")
axes[0, 1].legend(title="Thermal sensor", frameon=False)

strict_matrix = summary.pivot(
    index="city", columns="thermal_sensor", values="strict_temporal_pct"
)
sns.heatmap(
    strict_matrix,
    annot=True,
    fmt=".1f",
    vmin=0,
    vmax=100,
    cmap="magma",
    linewidths=0.8,
    cbar_kws={"label": "Strict temporal-match rate (%)"},
    ax=axes[1, 0],
)
axes[1, 0].set_title("(c) Stricter temporal-window sensitivity")
axes[1, 0].set_xlabel("")
axes[1, 0].set_ylabel("")

cloud_data = matches[
    matches["s2_found"].eq(1)
    & matches["s2_cloudy_pixel_percentage"].notna()
].copy()
sns.scatterplot(
    data=cloud_data,
    x="s2_time_difference_hours",
    y="s2_cloudy_pixel_percentage",
    hue="city",
    style="thermal_sensor",
    alpha=0.35,
    s=28,
    ax=axes[1, 1],
)
axes[1, 1].axvline(72, color="black", linestyle="--", linewidth=1.1)
axes[1, 1].set_xlabel("Sentinel-2 acquisition-time offset (hours)")
axes[1, 1].set_ylabel("Sentinel-2 scene cloudiness (%)")
axes[1, 1].set_title("(d) Temporal proximity–cloudiness trade-off")
axes[1, 1].legend(frameon=False, ncol=2, fontsize=8)

figure_path = OUTPUT_DIR / "thermofusion_stage06_match_diagnostics.png"
fig.savefig(figure_path, dpi=800, bbox_inches="tight", facecolor="white")
plt.show()


complete = (
    len(files) == EXPECTED_FILES
    and len(matches) == EXPECTED_RECORDS
    and not missing_ids
    and not unexpected_ids
    and duplicate_records.empty
    and offset_anomalies.empty
)

verdict_lines = [
    "THERMOFUSION STAGE 06 MATCH VERIFICATION",
    f"Files: {len(files)}/{EXPECTED_FILES}",
    f"Exported rows: {len(matches)}/{EXPECTED_RECORDS}",
    f"Missing manifest records: {len(missing_ids)}",
    f"Unexpected records: {len(unexpected_ids)}",
    f"Duplicate record IDs: {duplicate_records['record_id'].nunique()}",
    f"Temporal-offset anomalies: {len(offset_anomalies)}",
    f"All-three-source matches: {int(matches['all_three_found'].sum())}",
    f"Strict temporal matches: {int(matches['strict_temporal_match'].sum())}",
    "Overall verification: PASS" if complete else "Overall verification: REVIEW REQUIRED",
]
(OUTPUT_DIR / "08_match_verification_verdict.txt").write_text(
    "\n".join(verdict_lines), encoding="utf-8"
)

zip_path = shutil.make_archive(
    "/content/drive/MyDrive/ThermoFusion_Stage06_Match_Verification",
    "zip",
    root_dir=OUTPUT_DIR,
)

print("\n" + "\n".join(verdict_lines))
print(f"\nOutput folder: {OUTPUT_DIR}")
print(f"ZIP package: {zip_path}")
