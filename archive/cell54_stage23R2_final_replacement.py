# =====================================================================
# ThermoFusion reproducibility code
# Source notebook cell(s): 54
# Audit status: ARCHIVE
# Extracted without scientific-logic modification.
# Replace local Google Drive paths as described in README.md when needed.
# =====================================================================

# ==========================================================================================
# THERMOFUSION STAGE 23R2 — REPLACE THE FINAL SPATIAL-PREFLIGHT FAILURE
# Keeps the original experiment frozen; changes only the supplementary expanded-test manifest.
# ==========================================================================================

import os
import pandas as pd
from google.colab import drive
from IPython.display import display

drive.mount("/content/drive")

BASE = "/content/drive/MyDrive"

ORIGINAL_SPLIT = (
    f"{BASE}/ThermoFusion_Stage12_ModelReady_Pilot/"
    "01_scene_split_manifest.csv"
)

CURRENT_MANIFEST = (
    f"{BASE}/ThermoFusion_Stage23R_Replacements/"
    "01_revised_expanded_test_manifest.csv"
)

ELIGIBLE_CANDIDATES = (
    f"{BASE}/ThermoFusion_Stage22B_CorrectedExpandedTest/"
    "04_unique_eligible_candidates.csv"
)

PREFLIGHT = (
    f"{BASE}/ThermoFusion_Stage23R_Replacements/"
    "Stage23R_Export/01_spatial_preflight.csv"
)

OUT_DIR = f"{BASE}/ThermoFusion_Stage23R2_FinalReplacement"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_MANIFEST = f"{OUT_DIR}/01_final_expanded_test_manifest.csv"
OUT_AUDIT = f"{OUT_DIR}/02_final_replacement_audit.csv"

# ------------------------------------------------------------------
# Load files
# ------------------------------------------------------------------
original = pd.read_csv(ORIGINAL_SPLIT)
current = pd.read_csv(CURRENT_MANIFEST)
candidates = pd.read_csv(ELIGIBLE_CANDIDATES)
preflight = pd.read_csv(PREFLIGHT)

for df, name in [
    (original, "original split"),
    (current, "current manifest"),
    (candidates, "eligible candidates"),
    (preflight, "Stage 23R preflight"),
]:
    if "record_id" not in df.columns:
        raise KeyError(f"{name} is missing record_id")

# ------------------------------------------------------------------
# Identify the sole failed scene
# ------------------------------------------------------------------
status_col = next(
    (c for c in ["preflight_status", "status", "joint_valid_centre_found"]
     if c in preflight.columns),
    None
)

if status_col is None:
    raise KeyError(
        "Could not find a preflight-status column. "
        f"Available columns: {preflight.columns.tolist()}"
    )

if status_col == "joint_valid_centre_found":
    failed = preflight.loc[~preflight[status_col].astype(bool)].copy()
else:
    failed = preflight.loc[
        preflight[status_col].astype(str).str.upper().eq("FAIL")
    ].copy()

if len(failed) != 1:
    raise RuntimeError(
        f"Expected exactly one remaining failed scene, found {len(failed)}. "
        "Do not continue until this is resolved."
    )

failed_row = failed.iloc[0]
failed_id = failed_row["record_id"]

print("=" * 96)
print("THERMOFUSION STAGE 23R2 — FINAL REPLACEMENT")
print("=" * 96)
print(f"Failed record to replace: {failed_id}")

# Obtain city/sensor/quarter from the current manifest.
failed_current = current.loc[current["record_id"].eq(failed_id)].copy()
if len(failed_current) != 1:
    raise RuntimeError(
        "The failed record was not found exactly once in the current manifest."
    )

failed_current = failed_current.iloc[0]
city = failed_current["city"]
sensor = failed_current["thermal_sensor"]
quarter = failed_current["quarter"]

# ------------------------------------------------------------------
# Exclude all protected, used, and already failed records
# ------------------------------------------------------------------
original_ids = set(original["record_id"].astype(str))
current_ids = set(current["record_id"].astype(str))
failed_ids = set(failed["record_id"].astype(str))

# Keep 37 valid scenes and remove the one that failed spatial preflight.
kept = current.loc[~current["record_id"].astype(str).isin(failed_ids)].copy()

# Thermal-scene uniqueness protection.
thermal_key = "thermal_scene_id" if "thermal_scene_id" in candidates.columns else "record_id"
used_thermal = set()

if thermal_key in kept.columns:
    used_thermal.update(kept[thermal_key].dropna().astype(str))

# Standardize dates if present.
for df in [candidates, kept]:
    if "acquisition_date" in df.columns:
        df["acquisition_date"] = pd.to_datetime(df["acquisition_date"], utc=True)

# Candidate requirements: same city and sensor; still independent and high-coverage.
pool = candidates.copy()

pool = pool.loc[
    pool["city"].eq(city)
    & pool["thermal_sensor"].eq(sensor)
    & ~pool["record_id"].astype(str).isin(original_ids)
    & ~pool["record_id"].astype(str).isin(current_ids)
    & ~pool["record_id"].astype(str).isin(failed_ids)
].copy()

if "temporal_eligibility" in pool.columns:
    pool = pool.loc[
        pool["temporal_eligibility"].astype(str).str.upper().eq("PASS")
    ].copy()

if "valid_fraction_pct" in pool.columns:
    pool = pool.loc[pool["valid_fraction_pct"] >= 50.0].copy()

if "is_later_than_training" in pool.columns:
    pool = pool.loc[pool["is_later_than_training"].astype(bool)].copy()

if thermal_key in pool.columns and used_thermal:
    pool = pool.loc[~pool[thermal_key].astype(str).isin(used_thermal)].copy()

if pool.empty:
    raise RuntimeError(
        f"No eligible unused replacement remains for {city} × {sensor}. "
        "Do not lower the coverage threshold or alter the original model split."
    )

# Prefer same quarter; otherwise another quarter is acceptable and documented.
same_quarter = pool.loc[pool["quarter"].eq(quarter)].copy()

if not same_quarter.empty:
    replacement_pool = same_quarter
    replacement_source = "same_city_sensor_same_quarter"
else:
    replacement_pool = pool
    replacement_source = "same_city_sensor_other_quarter"

# Highest valid-pixel fraction first; newer acquisition breaks ties.
sort_cols = [c for c in ["valid_fraction_pct", "acquisition_date"] if c in replacement_pool.columns]
ascending = [False] * len(sort_cols)

replacement = replacement_pool.sort_values(
    sort_cols,
    ascending=ascending,
    kind="mergesort"
).iloc[0].copy()

# ------------------------------------------------------------------
# Build exact 38-scene final manifest
# ------------------------------------------------------------------
final_manifest = pd.concat(
    [kept, pd.DataFrame([replacement])],
    ignore_index=True
)

if len(final_manifest) != 38:
    raise RuntimeError(
        f"Final manifest has {len(final_manifest)} scenes; expected 38."
    )

if final_manifest["record_id"].nunique() != 38:
    raise RuntimeError("Duplicate record IDs detected in final manifest.")

if thermal_key in final_manifest.columns:
    n_unique_thermal = final_manifest[thermal_key].nunique()
    if n_unique_thermal != 38:
        raise RuntimeError(
            f"Thermal-scene uniqueness failed: {n_unique_thermal}/38 unique scenes."
        )

if "valid_fraction_pct" in final_manifest.columns:
    min_coverage = final_manifest["valid_fraction_pct"].min()
    if min_coverage < 50.0:
        raise RuntimeError(
            f"Coverage rule failed: minimum valid fraction is {min_coverage:.2f}%."
        )

final_manifest = final_manifest.sort_values(
    ["city", "thermal_sensor", "acquisition_date", "record_id"],
    kind="mergesort"
).reset_index(drop=True)

audit = pd.DataFrame([{
    "failed_record_id": failed_id,
    "failed_city": city,
    "failed_sensor": sensor,
    "failed_quarter": quarter,
    "replacement_record_id": replacement["record_id"],
    "replacement_quarter": replacement["quarter"],
    "replacement_valid_fraction_pct": replacement.get("valid_fraction_pct", pd.NA),
    "replacement_source": replacement_source,
}])

final_manifest.to_csv(OUT_MANIFEST, index=False)
audit.to_csv(OUT_AUDIT, index=False)

print()
print("FINAL REPLACEMENT SELECTED:")
display(audit)

print()
print(f"Final scenes: {len(final_manifest)}")
print(f"Unique record IDs: {final_manifest['record_id'].nunique()}")
print(f"Saved manifest: {OUT_MANIFEST}")
print(f"Saved audit: {OUT_AUDIT}")
print()
print("STAGE 23R2 VERDICT: PASS — rerun the Stage 23 export/preflight cell using OUT_MANIFEST.")
