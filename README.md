# ThermoFusion

Reproducibility package for the ThermoFusion manuscript:

**ThermoFusion: Metadata-conditioned multimodal Earth-observation learning for temporally transferable land-surface temperature reconstruction and uncertainty assessment**

## Purpose

This repository preserves the final scientific workflow used for:
- multi-city Landsat/ECOSTRESS thermal-scene auditing;
- Sentinel-1/Sentinel-2/terrain/context matching;
- leakage-safe scene splitting;
- balanced 64-scene development benchmark construction;
- metadata-conditioned residual U-Net training;
- strong climatological baselines;
- modality ablations;
- test-time augmentation and uncertainty analysis;
- leave-one-city-out geographical transfer;
- 38-scene and 52-scene frozen external temporal evaluations; and
- the integrated 90-scene later-date external evaluation.

## Important reproducibility rule

The scripts in `scripts/` were extracted from the authoritative Colab notebook without changing their scientific logic.  
They may still contain Google Colab and Google Drive path assumptions. Before public release, paths should be centralized in a configuration file, but **the model-selection, split, normalization, climatology, checkpoint, TTA, and uncertainty-calibration logic must not be changed**.

## Repository structure

```text
ThermoFusion/
├── README.md
├── LICENSE
├── requirements.txt
├── CITATION.cff
├── config/
├── scripts/
├── notebooks/
├── data/
├── results/
├── figures/
├── archive/
└── docs/
```

### `scripts/`
Final retained scientific workflow.

### `archive/`
Intermediate repair, diagnostic, pilot, and superseded figure-generation code retained only for provenance.

### `notebooks/`
The authoritative original Colab notebook.

### `data/`
Do not place large satellite rasters in GitHub. Store compact manifests, metadata tables, model-input contracts, normalization statistics, and small demonstration files only.

## Recommended execution order

1. `01_audit_city_assets_and_thermal_availability.py`
2. `02_audit_thermal_coverage.py`
3. `03_verify_thermal_audit.py`
4. `04_select_scenes_and_splits.py`
5. `05_match_multisensor_inputs.py`
6. `06_verify_multisensor_matches.py`
7. `07_build_balanced_pilot.py`
8. `08_export_alignment_test_chips.py`
9. `09_verify_alignment.py`
10. `10_export_balanced_dataset.py`
11. `11_verify_balanced_dataset.py`
12. `12_prepare_model_arrays.py`
13. `13_train_baseline_unet.py`
14. `14_train_thermofusion.py`
15. `15_calibration_and_bootstrap.py`
16. `16_modality_ablations.py`
17. `17_final_inference_and_uncertainty.py`
18. `18_compile_reproducibility_package.py`
19. `19_generate_manuscript_figures.py`
20. `20_loco_verify_dataset.py`
21. `21_loco_build_folds.py`
22. `22_run_loco_experiment.py`
23. `23_select_external38.py`
24. `24_export_external38.py`
25. `25_verify_external38.py`
26. `26_prepare_external38_arrays.py`
27. `27_infer_external38.py`
28. `28_evaluate_external38.py`
29. `29_preflight_external65.py`
30. `30_export_external52.py`
31. `31_verify_external52_exports.py`
32. `32_prepare_external52_arrays.py`
33. `33_infer_external52.py`
34. `34_evaluate_external52.py`
35. `35_integrate_external90.py`

Figure-generation utilities are under `scripts/figures/`.

## Data policy

Raw Landsat, ECOSTRESS, Sentinel-1, Sentinel-2, ERA5-Land, DEM, and large GeoTIFF/NumPy arrays should not be committed directly to GitHub.

Recommended public data products:
- scene manifests;
- train/validation/test assignments;
- external evaluation manifests;
- normalization statistics derived from training data;
- model-input band contract;
- baseline climatology tables;
- per-scene evaluation CSVs;
- compact example chips where licensing and size allow.

Large derived datasets should be deposited in Zenodo or another research-data repository and linked from this README.

## Provenance

The file `docs/cell_audit.csv` documents which of the 73 Colab cells were retained, merged, archived, or dropped.
## DOI

The archived ThermoFusion reproducibility package is available on Zenodo:

DOI: 10.5281/zenodo.22747479
## Authors

**George Owusu Amoah¹\, Rachel Olawoyin¹, Francis Quayson²˒³

¹ Department of Geography and Regional Planning, University of Cape Coast, Cape Coast, Ghana  
² Department of Land Surveying and Geospatial Science, The Hong Kong Polytechnic University, Hong Kong, China  
³ Research Institute for Land and Space, The Hong Kong Polytechnic University, Hong Kong, China  

\*Corresponding author: George Owusu Amoah  
Email: george.amoah003@stu.ucc.edu.gh
