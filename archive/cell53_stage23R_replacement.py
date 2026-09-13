# =====================================================================
# ThermoFusion reproducibility code
# Source notebook cell(s): 53
# Audit status: ARCHIVE
# Extracted without scientific-logic modification.
# Replace local Google Drive paths as described in README.md when needed.
# =====================================================================

# =============================================================================
# THERMOFUSION STAGE 23R — REPLACE SCENES THAT FAILED SPATIAL PREFLIGHT
# No original 64 scenes; no reuse of failed scenes; no lowered quality threshold.
# =============================================================================

from pathlib import Path
import pandas as pd

ROOT = Path("/content/drive/MyDrive")

ORIGINAL_SELECTION = (
    ROOT / "ThermoFusion_Stage22B_CorrectedExpandedTest"
    / "01_corrected_expanded_test_manifest.csv"
)

CANDIDATES = (
    ROOT / "ThermoFusion_Stage22B_CorrectedExpandedTest"
    / "04_unique_eligible_candidates.csv"
)

PREFLIGHT = (
    ROOT / "ThermoFusion_Stage23_ExpandedTemporalTest"
    / "01_spatial_preflight.csv"
)

OUT_DIR = ROOT / "ThermoFusion_Stage23R_Replacements"
OUT_DIR.mkdir(parents=True, exist_ok=True)

selected = pd.read_csv(ORIGINAL_SELECTION)
candidates = pd.read_csv(CANDIDATES)
preflight = pd.read_csv(PREFLIGHT)

failed_ids = set(
    preflight.loc[
        ~preflight["joint_valid_centre_found"],
        "record_id"
    ].astype(str)
)

passed = selected.loc[
    ~selected["record_id"].astype(str).isin(failed_ids)
].copy()

failed = selected.loc[
    selected["record_id"].astype(str).isin(failed_ids)
].copy()

print("Original selected scenes:", len(selected))
print("Spatially valid scenes retained:", len(passed))
print("Failed scenes requiring replacement:", len(failed))

display(
    failed[
        [
            "record_id", "city", "thermal_sensor",
            "quarter", "acquisition_date", "valid_fraction_pct"
        ]
    ]
)

# Exclude every original selected scene, including passed and failed records.
used_ids = set(selected["record_id"].astype(str))

# Candidate quality ranking: high valid fraction first; earliest later date second.
candidates["acquisition_date"] = pd.to_datetime(
    candidates["acquisition_date"],
    errors="coerce",
    utc=True
)

candidates["valid_fraction_pct"] = pd.to_numeric(
    candidates["valid_fraction_pct"],
    errors="coerce"
)

available = candidates.loc[
    ~candidates["record_id"].astype(str).isin(used_ids)
].copy()

available = available.sort_values(
    [
        "city", "thermal_sensor", "quarter",
        "valid_fraction_pct", "acquisition_date"
    ],
    ascending=[True, True, True, False, True]
)

replacement_rows = []
replacement_audit = []

for failed_row in failed.itertuples(index=False):
    pool = available.loc[
        (available["city"] == failed_row.city)
        & (available["thermal_sensor"] == failed_row.thermal_sensor)
        & (available["quarter"] == failed_row.quarter)
    ].copy()

    # If the same quarter has no candidate, retain city-sensor representation
    # by selecting from another unused quarter in that city-sensor group.
    replacement_source = "same_quarter"

    if pool.empty:
        pool = available.loc[
            (available["city"] == failed_row.city)
            & (available["thermal_sensor"] == failed_row.thermal_sensor)
        ].copy()

        replacement_source = "same_city_sensor_other_quarter"

    if pool.empty:
        replacement_audit.append({
            "failed_record_id": failed_row.record_id,
            "city": failed_row.city,
            "thermal_sensor": failed_row.thermal_sensor,
            "failed_quarter": failed_row.quarter,
            "replacement_found": False,
            "replacement_source": "NONE"
        })
        continue

    replacement = pool.iloc[0].copy()

    replacement["replaces_record_id"] = failed_row.record_id
    replacement["replacement_source"] = replacement_source

    replacement_rows.append(replacement)

    replacement_audit.append({
        "failed_record_id": failed_row.record_id,
        "city": failed_row.city,
        "thermal_sensor": failed_row.thermal_sensor,
        "failed_quarter": failed_row.quarter,
        "replacement_found": True,
        "replacement_record_id": replacement["record_id"],
        "replacement_quarter": replacement["quarter"],
        "replacement_source": replacement_source
    })

    # Prevent the same candidate from replacing more than one failed scene.
    available = available.loc[
        available["record_id"].astype(str) != str(replacement["record_id"])
    ].copy()

replacements = pd.DataFrame(replacement_rows)
audit = pd.DataFrame(replacement_audit)

if replacements.empty:
    raise RuntimeError(
        "No replacement candidates were found. Send me the replacement audit."
    )

revised = pd.concat(
    [passed, replacements],
    ignore_index=True
)

assert revised["record_id"].nunique() == len(revised), (
    "Duplicate records detected in revised selection."
)

assert not revised.duplicated(
    ["city", "thermal_sensor", "thermal_scene_id"]
).any(), (
    "Duplicate thermal scenes detected in revised selection."
)

revised_path = OUT_DIR / "01_revised_expanded_test_manifest.csv"
audit_path = OUT_DIR / "02_replacement_audit.csv"

revised.to_csv(revised_path, index=False)
audit.to_csv(audit_path, index=False)

print("\n" + "=" * 90)
print("STAGE 23R REPLACEMENT SELECTION COMPLETE")
print("=" * 90)
print("Revised scenes ready for spatial preflight:", len(revised))
print("Replacement candidates selected:", len(replacements))
print("Replacement audit:")
display(audit)

print("\nSaved files:")
print(revised_path)
print(audit_path)

print(
    "\nNEXT: Send me the replacement audit table. "
    "I will give you the updated preflight/export cell."
)
