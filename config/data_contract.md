# ThermoFusion model-input contract

Raster predictors: 16
- Sentinel-2 reflectance/index channels: 10
- Sentinel-1 channels: 4
- terrain channels: 2

Context variables: 9
- thermal sensor indicator: 1
- day-of-year sine/cosine: 2
- UTC-hour sine/cosine: 2
- city one-hot indicators: 4

Total inputs: 25

The final selected manuscript model is the validation-selected no-terrain residual U-Net with 8-view test-time augmentation.
