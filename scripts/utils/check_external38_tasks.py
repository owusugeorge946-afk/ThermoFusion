# =====================================================================
# ThermoFusion reproducibility code
# Source notebook cell(s): 56
# Audit status: KEEP
# Extracted without scientific-logic modification.
# Replace local Google Drive paths as described in README.md when needed.
# =====================================================================

# ==========================================================================================
# THERMOFUSION STAGE 23R2 — EXPORT TASK STATUS CHECK
# ==========================================================================================

import pandas as pd
import ee
from google.colab import drive
from pathlib import Path
from IPython.display import display

drive.mount("/content/drive")

PROJECT_ID = "nana213"

TASK_REGISTER = Path(
    "/content/drive/MyDrive/ThermoFusion_Stage23R2_FinalReplacement/"
    "Stage23R2_Export/03_export_task_register.csv"
)

STATUS_CSV = Path(
    "/content/drive/MyDrive/ThermoFusion_Stage23R2_FinalReplacement/"
    "Stage23R2_Export/04_export_task_status.csv"
)

ee.Initialize(project=PROJECT_ID)

tasks = pd.read_csv(TASK_REGISTER)

task_ids = (
    tasks["predictor_task_id"].astype(str).tolist()
    + tasks["thermal_task_id"].astype(str).tolist()
)

status_records = []

# Earth Engine accepts task-status requests in manageable batches.
for start in range(0, len(task_ids), 50):
    batch_ids = task_ids[start:start + 50]
    status_records.extend(ee.data.getTaskStatus(batch_ids))

status_df = pd.DataFrame(status_records)

if "id" not in status_df.columns:
    raise RuntimeError("Earth Engine returned no task-status records.")

status_df = status_df.rename(columns={"id": "task_id"})

task_lookup = pd.concat([
    tasks[["export_id", "record_id", "city", "thermal_sensor", "predictor_task_id"]]
    .rename(columns={"predictor_task_id": "task_id"})
    .assign(export_type="predictors"),

    tasks[["export_id", "record_id", "city", "thermal_sensor", "thermal_task_id"]]
    .rename(columns={"thermal_task_id": "task_id"})
    .assign(export_type="thermal"),
], ignore_index=True)

report = task_lookup.merge(
    status_df,
    on="task_id",
    how="left",
)

report.to_csv(STATUS_CSV, index=False)

print("=" * 90)
print("THERMOFUSION STAGE 23R2 — EARTH ENGINE EXPORT STATUS")
print("=" * 90)

summary = (
    report.groupby("state", dropna=False)
    .size()
    .reset_index(name="tasks")
    .sort_values("state")
)

display(summary)

completed = int((report["state"] == "COMPLETED").sum())
failed = int((report["state"] == "FAILED").sum())
total = len(report)

print(f"Completed: {completed}/{total}")
print(f"Failed: {failed}/{total}")
print(f"Status register: {STATUS_CSV}")

if failed > 0:
    print("\nFAILED TASKS:")
    display(
        report.loc[
            report["state"].eq("FAILED"),
            ["export_id", "record_id", "export_type", "state", "error_message"]
        ]
    )
elif completed == total:
    print("\nFINAL STATUS: PASS — all 76 exports completed.")
else:
    print("\nExports are still running or queued. Run this cell again later.")
