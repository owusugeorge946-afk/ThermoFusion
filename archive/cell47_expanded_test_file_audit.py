# =====================================================================
# ThermoFusion reproducibility code
# Source notebook cell(s): 47
# Audit status: ARCHIVE
# Extracted without scientific-logic modification.
# Replace local Google Drive paths as described in README.md when needed.
# =====================================================================

# ============================================================
# THERMOFUSION EXPANDED TEMPORAL TEST — FILE AUDIT
# ============================================================

from google.colab import drive
from pathlib import Path
import pandas as pd

drive.mount("/content/drive")

SEARCH_ROOT = Path("/content/drive/MyDrive")

KEYWORDS = [
    "thermofusion",
    "eligible",
    "inventory",
    "metadata",
    "scene",
    "checkpoint",
    "model",
    "normalization",
    "scaler",
    "tensor",
    "array",
    "stack",
    "mask"
]

VALID_EXTENSIONS = {
    ".csv", ".json", ".parquet",
    ".npy", ".npz", ".pt", ".pth",
    ".pkl", ".joblib", ".tif", ".tiff"
}

records = []

for path in SEARCH_ROOT.rglob("*"):
    if not path.is_file():
        continue

    path_lower = str(path).lower()

    if (
        path.suffix.lower() in VALID_EXTENSIONS
        and any(keyword in path_lower for keyword in KEYWORDS)
    ):
        records.append({
            "filename": path.name,
            "extension": path.suffix.lower(),
            "size_mb": round(path.stat().st_size / (1024**2), 3),
            "full_path": str(path)
        })

audit = pd.DataFrame(records)

if audit.empty:
    print("No candidate ThermoFusion files were found.")
else:
    audit = audit.sort_values(
        ["extension", "filename"]
    ).reset_index(drop=True)

    print("=" * 110)
    print("THERMOFUSION FILE AUDIT")
    print("=" * 110)
    print(f"Candidate files found: {len(audit)}")
    print()

    pd.set_option("display.max_rows", 500)
    pd.set_option("display.max_colwidth", 150)
    display(audit)

    output_path = (
        SEARCH_ROOT /
        "ThermoFusion_expanded_temporal_test_file_audit.csv"
    )
    audit.to_csv(output_path, index=False)

    print()
    print("Audit saved to:")
    print(output_path)
