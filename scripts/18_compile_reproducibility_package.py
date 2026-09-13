# =====================================================================
# ThermoFusion reproducibility code
# Source notebook cell(s): 33
# Audit status: KEEP
# Extracted without scientific-logic modification.
# Replace local Google Drive paths as described in README.md when needed.
# =====================================================================

# ThermoFusion Stage 18: compile the final pilot-analysis and reproducibility package.
# Run in one Google Colab cell after Stage 17 reports PASS. GPU is not required.

from google.colab import drive
from pathlib import Path
import hashlib
import json
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

drive.mount("/content/drive")
ROOT = Path("/content/drive/MyDrive")
STAGES = {
    11: ROOT / "ThermoFusion_Stage11_Pilot_Verification",
    12: ROOT / "ThermoFusion_Stage12_ModelReady_Pilot",
    13: ROOT / "ThermoFusion_Stage13_Pilot_Baseline",
    14: ROOT / "ThermoFusion_Stage14_Refined_Benchmark",
    15: ROOT / "ThermoFusion_Stage15_Calibration_Bootstrap",
    16: ROOT / "ThermoFusion_Stage16_Modality_Ablation",
    17: ROOT / "ThermoFusion_Stage17_Final_Uncertainty",
}
OUT = ROOT / "ThermoFusion_Stage18_Final_Pilot_Analysis"
FIGURE_DIR = OUT / "publication_figures"
TABLE_DIR = OUT / "publication_tables"
OUT.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)
# Make reruns deterministic without touching any upstream stage outputs.
for stale in list(FIGURE_DIR.glob("*.png")) + list(TABLE_DIR.glob("*.csv")):
    stale.unlink()

FILES = {
    "stage11_verdict": STAGES[11] / "03_pilot_verification_verdict.txt",
    "stage12_verdict": STAGES[12] / "07_model_ready_verdict.txt",
    "stage12_arrays": STAGES[12] / "04_model_ready_array_manifest.csv",
    "stage12_splits": STAGES[12] / "01_scene_split_manifest.csv",
    "stage13_verdict": STAGES[13] / "05_stage13_verdict.txt",
    "stage13_aggregate": STAGES[13] / "04_aggregate_metrics.csv",
    "stage14_verdict": STAGES[14] / "06_stage14_verdict.txt",
    "stage14_aggregate": STAGES[14] / "05_aggregate_model_comparison.csv",
    "stage15_verdict": STAGES[15] / "07_stage15_verdict.txt",
    "stage15_bootstrap": STAGES[15] / "06_paired_scene_bootstrap.csv",
    "stage16_verdict": STAGES[16] / "05_stage16_verdict.txt",
    "stage16_summary": STAGES[16] / "03_ablation_summary.csv",
    "stage16_bootstrap": STAGES[16] / "04_paired_ablation_bootstrap.csv",
    "stage17_verdict": STAGES[17] / "04_stage17_verdict.txt",
    "stage17_scene": STAGES[17] / "01_scene_accuracy_uncertainty.csv",
    "stage17_aggregate": STAGES[17] / "02_aggregate_accuracy_uncertainty.csv",
    "stage17_reliability": STAGES[17] / "03_uncertainty_reliability.csv",
}
for name, path in FILES.items():
    if not path.exists():
        raise FileNotFoundError(f"Missing required {name}: {path}")

required_passes = {
    "stage11_verdict": "Overall pilot verification: PASS",
    "stage12_verdict": "Overall model-ready verdict: PASS",
    "stage13_verdict": "Overall Stage 13 execution: PASS",
    "stage14_verdict": "Overall Stage 14 execution: PASS",
    "stage15_verdict": "Overall Stage 15 execution: PASS",
    "stage16_verdict": "Overall Stage 16 execution: PASS",
    "stage17_verdict": "Overall Stage 17 execution: PASS",
}
for key, phrase in required_passes.items():
    if phrase not in FILES[key].read_text(encoding="utf-8"):
        raise RuntimeError(f"Required PASS not found in {FILES[key]}")


def verdict_values(path):
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            values[key.strip()] = value.strip()
    return values


v11 = verdict_values(FILES["stage11_verdict"])
v12 = verdict_values(FILES["stage12_verdict"])
v13 = verdict_values(FILES["stage13_verdict"])
v14 = verdict_values(FILES["stage14_verdict"])
v15 = verdict_values(FILES["stage15_verdict"])
v16 = verdict_values(FILES["stage16_verdict"])
v17 = verdict_values(FILES["stage17_verdict"])

arrays = pd.read_csv(FILES["stage12_arrays"])
splits = pd.read_csv(FILES["stage12_splits"])
stage13 = pd.read_csv(FILES["stage13_aggregate"])
stage14 = pd.read_csv(FILES["stage14_aggregate"])
stage15_bootstrap = pd.read_csv(FILES["stage15_bootstrap"])
stage16 = pd.read_csv(FILES["stage16_summary"])
stage16_bootstrap = pd.read_csv(FILES["stage16_bootstrap"])
stage17_scene = pd.read_csv(FILES["stage17_scene"])
stage17 = pd.read_csv(FILES["stage17_aggregate"])
reliability = pd.read_csv(FILES["stage17_reliability"])


def exactly_one(frame, description):
    if len(frame) != 1:
        raise RuntimeError(
            f"Expected exactly one {description} row, found {len(frame)}."
        )
    return frame.iloc[0]

if len(arrays) != 64 or arrays.record_id.nunique() != 64:
    raise RuntimeError("Stage 12 does not contain 64 unique model-ready arrays.")
if splits.model_split.value_counts().to_dict() != {
    "train": 48, "validation": 8, "temporal_test": 8
}:
    raise RuntimeError("Final split is not 48/8/8.")

# Table 1: dataset and experimental design.
design = pd.DataFrame([
    ["Verified pilot bundles", 64, "Stage 11"],
    ["Model-ready arrays", 64, "Stage 12"],
    ["Training scenes", 48, "Scene-disjoint"],
    ["Validation scenes", 8, "One per city-sensor group"],
    ["Temporal-test scenes", 8, "Latest scene per city-sensor group"],
    ["Raster predictor channels", 16, "One inactive slope channel"],
    ["Context channels", 9, "Sensor, time and city"],
    ["Image dimensions", 256, "256 x 256 pixels"],
], columns=["item", "value", "note"])
design.to_csv(TABLE_DIR / "Table_1_dataset_and_design.csv", index=False)

# Table 2: exact headline performance across model-development stages.
baseline_test = exactly_one(
    stage13.query("model_split == 'temporal_test' and group == 'ALL'"),
    "Stage 13 all-scene temporal-test",
)
refined_test = exactly_one(stage14.query(
    "model_split == 'temporal_test' and model == 'refined_unet' and group == 'ALL'"
), "Stage 14 refined all-scene temporal-test")
final_test = exactly_one(
    stage17.query("model_split == 'temporal_test'"),
    "Stage 17 temporal-test aggregate",
)
performance = pd.DataFrame([
    ["Stage 13 baseline U-Net", baseline_test.mae_c, baseline_test.rmse_c,
     baseline_test.bias_c, baseline_test.r2],
    ["Stage 14 metadata-conditioned full model", refined_test.mae_c,
     refined_test.rmse_c, refined_test.bias_c, refined_test.r2],
    ["Stage 17 validation-selected no-terrain TTA model", final_test.mae_c,
     final_test.rmse_c, final_test.bias_c, final_test.r2],
], columns=["model", "mae_c", "rmse_c", "bias_c", "r2"])
performance.to_csv(TABLE_DIR / "Table_2_temporal_test_performance.csv", index=False)

# Table 3: full modality-ablation results with paired uncertainty.
ablation_test = stage16.query("model_split == 'temporal_test'").copy()
ablation_test = ablation_test.merge(stage16_bootstrap, on="variant", how="left")
ablation_test.to_csv(TABLE_DIR / "Table_3_modality_ablation.csv", index=False)

# Table 4: final accuracy and uncertainty by scene.
stage17_scene.to_csv(TABLE_DIR / "Table_4_scene_accuracy_uncertainty.csv", index=False)

# Table 5: calibration and uncertainty qualification.
stage15_rmse = exactly_one(
    stage15_bootstrap.query("metric == 'scene_macro_rmse_c'"),
    "Stage 15 scene-macro RMSE bootstrap",
)
uncertainty_summary = pd.DataFrame([
    ["Nominal interval coverage", 0.90, "Target"],
    ["Temporal-test interval coverage", final_test.interval_coverage,
     "Below nominal; intervals under-cover temporal shift"],
    ["Temporal-test uncertainty-error Spearman rho",
     final_test.uncertainty_error_spearman,
     "Positive ranking association"],
    ["Stage 15 calibrated-minus-climatology scene-macro RMSE",
     stage15_rmse.calibrated_minus_climatology,
     f"95% CI [{stage15_rmse.ci_lower:.4f}, {stage15_rmse.ci_upper:.4f}]"],
], columns=["quantity", "value", "interpretation"])
uncertainty_summary.to_csv(TABLE_DIR / "Table_5_calibration_and_uncertainty.csv", index=False)

# Final synthesis figure.
sns.set_theme(style="whitegrid", context="notebook")
fig, axes = plt.subplots(2, 2, figsize=(16, 13), constrained_layout=True)
sns.barplot(data=performance, x="model", y="rmse_c", ax=axes[0,0], color="#4c78a8")
axes[0,0].set(title="(a) Temporal-test model progression", xlabel="", ylabel="RMSE (°C)")
axes[0,0].tick_params(axis="x", rotation=18)

order = ablation_test.sort_values("scene_macro_rmse_c").variant
sns.barplot(data=ablation_test, x="variant", y="scene_macro_rmse_c",
            order=order, ax=axes[0,1], color="#72b7b2")
axes[0,1].set(title="(b) Temporal-test modality ablations",
              xlabel="", ylabel="Scene-macro RMSE (°C)")
axes[0,1].tick_params(axis="x", rotation=20)

test_scene = stage17_scene.query("model_split == 'temporal_test'")
sns.scatterplot(data=test_scene, x="rmse_c", y="interval_coverage",
                hue="thermal_sensor", style="city", s=90, ax=axes[1,0])
axes[1,0].axhline(0.90, color="black", linestyle="--", linewidth=1)
axes[1,0].set(title="(c) Scene accuracy and interval coverage",
              xlabel="Scene RMSE (°C)", ylabel="90% interval coverage")
axes[1,0].legend(frameon=False, fontsize=8)

test_reliability = reliability.query("model_split == 'temporal_test'")
axes[1,1].plot(test_reliability.mean_tta_std_c,
               test_reliability.mean_absolute_error_c, "o-", color="#b42c3b")
axes[1,1].set(title="(d) Uncertainty–error reliability",
              xlabel="Mean TTA standard deviation (°C)",
              ylabel="Mean absolute error (°C)")
figure_path = FIGURE_DIR / "Figure_Final_ThermoFusion_Synthesis.png"
fig.savefig(figure_path, dpi=800, bbox_inches="tight", facecolor="white")
plt.show()

# Preserve the most important verified source figures under publication names.
source_figures = {
    STAGES[14] / "thermofusion_stage14_refined_benchmark.png":
        FIGURE_DIR / "Figure_Model_Refinement.png",
    STAGES[15] / "thermofusion_stage15_calibration_bootstrap.png":
        FIGURE_DIR / "Figure_Calibration_Bootstrap.png",
    STAGES[16] / "thermofusion_stage16_modality_ablation.png":
        FIGURE_DIR / "Figure_Modality_Ablation.png",
    STAGES[17] / "thermofusion_stage17_final_performance.png":
        FIGURE_DIR / "Figure_Final_Performance_Uncertainty.png",
    STAGES[17] / "thermofusion_stage17_temporal_test_atlas.png":
        FIGURE_DIR / "Figure_Temporal_Test_Atlas.png",
}
for source, destination in source_figures.items():
    if not source.exists(): raise FileNotFoundError(f"Required figure missing: {source}")
    shutil.copy2(source, destination)

# Reproducibility manifest with checksums for every final input and output table.
def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


manifest_rows = []
for role, path in FILES.items():
    manifest_rows.append({"role": role, "path": str(path),
                          "size_bytes": path.stat().st_size, "sha256": sha256(path)})
for path in sorted(TABLE_DIR.glob("*.csv")):
    manifest_rows.append({"role": "final_table", "path": str(path),
                          "size_bytes": path.stat().st_size, "sha256": sha256(path)})
pd.DataFrame(manifest_rows).to_csv(OUT / "01_reproducibility_manifest.csv", index=False)

interval_gap = 0.90 - float(final_test.interval_coverage)
best_ablation = ablation_test.sort_values("scene_macro_rmse_c").iloc[0]
report = f"""# ThermoFusion final pilot analysis

## Study design

The verified pilot contains 64 city-sensor scenes: 48 training scenes, eight
validation scenes and eight chronologically later temporal-test scenes. Each
city-sensor combination contributes six training scenes, one validation scene
and one temporal-test scene. Model fitting, early stopping, architecture
selection and interval calibration did not use temporal-test data.

The model-ready inputs contain 16 raster channels. The slope channel had no
usable training pixels and was set to zero. The refined models added nine known
context channels representing thermal sensor, cyclical acquisition timing and
city identity.

## Model development

The initial U-Net obtained temporal-test MAE {baseline_test.mae_c:.4f} °C,
RMSE {baseline_test.rmse_c:.4f} °C and R² {baseline_test.r2:.4f}. Adding
city-sensor residual normalization and context reduced RMSE to
{refined_test.rmse_c:.4f} °C. Validation-controlled ablation selected the
without-terrain architecture for final uncertainty analysis.

The lowest temporal-test scene-macro RMSE occurred for `{best_ablation.variant}`
({best_ablation.scene_macro_rmse_c:.4f} °C), but this temporal-test result was
not used to select the final architecture.

## Final temporal-test performance

The validation-selected no-terrain model with eight test-time transformations
obtained MAE {final_test.mae_c:.4f} °C, RMSE {final_test.rmse_c:.4f} °C, bias
{final_test.bias_c:.4f} °C and R² {final_test.r2:.4f}. The positive bias and
the observed-versus-predicted plots show remaining compression at temperature
extremes, particularly for the hottest Landsat pixels.

## Baseline comparison

Stage 15 found a paired scene-macro RMSE difference of
{stage15_rmse.calibrated_minus_climatology:.4f} °C between the selected U-Net
and city-sensor climatology. Its 95% scene-bootstrap interval was
[{stage15_rmse.ci_lower:.4f}, {stage15_rmse.ci_upper:.4f}] °C. Because the
interval crosses zero, superiority over climatology was not established with
the eight-scene temporal test.

## Predictive uncertainty

The 90% validation-calibrated intervals covered
{final_test.interval_coverage:.2%} of sampled temporal-test pixels, a shortfall
of {interval_gap:.2%}. The uncertainty-error Spearman correlation was
{final_test.uncertainty_error_spearman:.4f}. TTA uncertainty therefore ranked
error-prone pixels moderately well, but the intervals were under-calibrated
under temporal shift and must not be described as achieving 90% test coverage.

## Main limitations

1. The pilot contains only eight independent temporal-test scenes, one per
   city-sensor group; pixel counts do not replace scene-level replication.
2. Modality-ablation bootstrap intervals overlap zero, so contributions are
   suggestive rather than conclusive.
3. The slope channel was unavailable, and terrain did not improve validation
   performance in this pilot.
4. Predictions regress toward the centre of the temperature distribution and
   underrepresent some extremes.
5. Validation-calibrated uncertainty intervals under-cover future scenes.

## Defensible conclusion

The pilot demonstrates that the metadata-conditioned residual U-Net learns
substantial within-scene spatial temperature structure and improves strongly
over the initial baseline. It does not yet establish consistent superiority
over city-sensor climatology or nominally calibrated predictive intervals.
Expansion to more independent scenes is required before operational mapping or
strong generalisation claims.
"""
(OUT / "02_final_methods_results_limitations.md").write_text(report, encoding="utf-8")

final_checks = {
    "all_required_stage_verdicts_pass": True,
    "unique_model_ready_scenes": arrays.record_id.nunique() == 64,
    "correct_split_counts": splits.model_split.value_counts().to_dict() == {
        "train": 48, "validation": 8, "temporal_test": 8},
    "six_publication_figures_present": len(list(FIGURE_DIR.glob("*.png"))) == 6,
    "five_publication_tables_present": len(list(TABLE_DIR.glob("*.csv"))) == 5,
    "final_metrics_finite": np.isfinite(performance.select_dtypes(include=np.number)).all().all(),
}
check_table = pd.DataFrame([{"check": key, "passed": value}
                            for key, value in final_checks.items()])
check_table.to_csv(OUT / "03_final_package_checks.csv", index=False)
overall_pass = all(final_checks.values())

verdict = [
    "THERMOFUSION STAGE 18 FINAL PILOT ANALYSIS",
    "Verified model-ready scenes: 64/64",
    "Training/validation/temporal-test scenes: 48/8/8",
    f"Final temporal-test RMSE (°C): {final_test.rmse_c:.4f}",
    f"Final temporal-test R²: {final_test.r2:.4f}",
    f"Observed coverage of nominal 90% intervals: {final_test.interval_coverage:.4f}",
    "Model superiority over city-sensor climatology: NOT ESTABLISHED",
    "Uncertainty intervals calibrated under temporal shift: NO",
    f"Publication tables: {len(list(TABLE_DIR.glob('*.csv')))}/5",
    f"Publication figures: {len(list(FIGURE_DIR.glob('*.png')))}/6",
    "Overall final package: PASS" if overall_pass else "Overall final package: REVIEW REQUIRED",
]
(OUT / "04_final_verdict.txt").write_text("\n".join(verdict), encoding="utf-8")
zip_path = shutil.make_archive(
    str(ROOT / "ThermoFusion_Stage18_Final_Pilot_Analysis"), "zip", root_dir=OUT
)
print("\n" + "\n".join(verdict))
print(f"\nOutput folder: {OUT}")
print(f"ZIP package: {zip_path}")
