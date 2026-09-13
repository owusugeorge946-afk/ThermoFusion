# ThermoFusion data directory

GitHub should contain only compact reproducibility data.

Recommended contents:
- `manifests/`: scene IDs, dates, city, sensor, split and quality metadata;
- model-input contract (16 raster channels + 9 contextual variables);
- training-only normalization statistics;
- frozen climatology baseline summaries;
- external38, external52 and integrated external90 evaluation tables;
- a few small demonstration chips if redistribution is permitted.

Do **not** commit:
- bulk Earth Engine exports;
- full satellite imagery;
- large GeoTIFF collections;
- large `.npy` / `.npz` tensors;
- model checkpoints larger than GitHub's practical file limits.

Use Zenodo or another research-data archive for large derived products.
