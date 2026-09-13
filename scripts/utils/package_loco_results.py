# =====================================================================
# ThermoFusion reproducibility code
# Source notebook cell(s): 46
# Audit status: KEEP
# Extracted without scientific-logic modification.
# Replace local Google Drive paths as described in README.md when needed.
# =====================================================================

from pathlib import Path
import shutil

results_dir = Path(
    "/content/drive/MyDrive/"
    "ThermoFusion_Stage20_LeaveOneCityOut/"
    "combined_four_city_results"
)

archive_base = Path(
    "/content/drive/MyDrive/"
    "ThermoFusion_Stage21_FourCity_LOCO_Results"
)

archive_path = shutil.make_archive(
    str(archive_base),
    "zip",
    root_dir=str(results_dir)
)

print("Created:", archive_path)
print("\nIncluded files:")
for path in sorted(results_dir.iterdir()):
    print(" -", path.name)
