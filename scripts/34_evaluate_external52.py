# =====================================================================
# ThermoFusion reproducibility code
# Source notebook cell(s): 70
# Audit status: KEEP
# Extracted without scientific-logic modification.
# Replace local Google Drive paths as described in README.md when needed.
# =====================================================================

# ======================================================================================
# THERMOFUSION — LARGE EXTERNAL TEMPORAL EVALUATION
# STAGE 07: FROZEN CLIMATOLOGY BASELINES + PAIRED SCENE BOOTSTRAP
#
# PRIMARY INFERENTIAL UNIT: independent thermal acquisition / scene
#
# NO model fitting
# NO normalization refitting
# NO climatology refitting
# NO calibration
# NO scene deletion based on performance
#
# Comparisons:
#   1. ThermoFusion vs global training-only climatology
#   2. ThermoFusion vs sensor training-only climatology
#   3. ThermoFusion vs city-sensor training-only climatology
#
# Inference:
#   - 50,000 ordinary paired scene bootstrap replicates
#   - 95% percentile CI
#   - bootstrap probability ThermoFusion has lower mean scene RMSE
#   - equal-observed-stratum sensitivity analysis to address imbalance
# ======================================================================================

from google.colab import drive
drive.mount("/content/drive")

from pathlib import Path
import json
import numpy as np
import pandas as pd

# ======================================================================================
# 1. PATHS
# ======================================================================================

ROOT = Path("/content/drive/MyDrive")

EXTERNAL_ROOT = (
    ROOT
    / "ThermoFusion_External_Evaluation"
)

S05 = (
    EXTERNAL_ROOT
    / "Stage05_ModelReady_External52"
)

S06 = (
    EXTERNAL_ROOT
    / "Stage06_FrozenInference_External52"
)

S14 = (
    ROOT
    / "ThermoFusion_Stage14_Refined_Benchmark"
)

ARRAY_MANIFEST_PATH = (
    S05
    / "01_external52_model_ready_manifest.csv"
)

STAGE05_VERDICT = (
    S05
    / "07_stage05_model_ready_verdict.txt"
)

MODEL_SCENE_PATH = (
    S06
    / "01_external52_scene_accuracy_uncertainty.csv"
)

STAGE06_VERDICT = (
    S06
    / "10_stage06_frozen_inference_verdict.txt"
)

TRAINING_STATS_PATH = (
    S14
    / "01_training_only_target_statistics.csv"
)

OUT = (
    EXTERNAL_ROOT
    / "Stage07_BaselineComparison_External52"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# ======================================================================================
# 2. FIXED SETTINGS
# ======================================================================================

SEED = 20260908
EXPECTED_SCENES = 52
BOOTSTRAPS = 50000
ALPHA = 0.05

CITIES = [
    "Abidjan",
    "Accra",
    "Freetown",
    "Lagos",
]

SENSORS = [
    "ECOSTRESS",
    "Landsat",
]


# ======================================================================================
# 3. REQUIRED INPUT CHECKS
# ======================================================================================

for path in [
    ARRAY_MANIFEST_PATH,
    STAGE05_VERDICT,
    MODEL_SCENE_PATH,
    STAGE06_VERDICT,
    TRAINING_STATS_PATH,
]:

    if not path.exists():

        raise FileNotFoundError(
            f"Required input not found:\n{path}"
        )


if (
    "Overall Stage 05 model-ready verdict: PASS"
    not in STAGE05_VERDICT.read_text(
        encoding="utf-8"
    )
):

    raise RuntimeError(
        "Stage 05 did not pass."
    )


if (
    "Overall Stage 06 frozen inference verdict: PASS"
    not in STAGE06_VERDICT.read_text(
        encoding="utf-8"
    )
):

    raise RuntimeError(
        "Stage 06 did not pass."
    )


# ======================================================================================
# 4. LOAD DATA
# ======================================================================================

arrays = pd.read_csv(
    ARRAY_MANIFEST_PATH
)

model = pd.read_csv(
    MODEL_SCENE_PATH
)

training_stats = pd.read_csv(
    TRAINING_STATS_PATH
)


if len(arrays) != EXPECTED_SCENES:

    raise RuntimeError(
        f"Expected 52 model-ready arrays, found {len(arrays)}."
    )


if len(model) != EXPECTED_SCENES:

    raise RuntimeError(
        f"Expected 52 Stage-06 scene results, found {len(model)}."
    )


if arrays["record_id"].duplicated().any():

    raise RuntimeError(
        "Duplicate record_id in Stage 05 manifest."
    )


if model["record_id"].duplicated().any():

    raise RuntimeError(
        "Duplicate record_id in Stage 06 results."
    )


# ======================================================================================
# 5. RECOVER FROZEN TRAINING-ONLY CLIMATOLOGIES
# ======================================================================================

# ----------------------------------------------------------------------
# GLOBAL
# ----------------------------------------------------------------------

global_rows = training_stats.loc[
    training_stats["level"].eq("GLOBAL")
].copy()


if len(global_rows) != 1:

    raise RuntimeError(
        f"Expected one GLOBAL training statistic, found {len(global_rows)}."
    )


global_mean = float(
    global_rows.iloc[0]["mean_c"]
)


# ----------------------------------------------------------------------
# SENSOR
# ----------------------------------------------------------------------

sensor_rows = training_stats.loc[
    training_stats["level"].eq("SENSOR")
].copy()


sensor_means = {
    str(row.group):
        float(row.mean_c)

    for row in sensor_rows.itertuples(
        index=False
    )
}


if set(sensor_means) != set(SENSORS):

    print(
        "Observed sensor climatologies:",
        sensor_means
    )

    raise RuntimeError(
        "Frozen SENSOR climatologies are incomplete."
    )


# ----------------------------------------------------------------------
# CITY × SENSOR
# ----------------------------------------------------------------------

group_rows = training_stats.loc[
    training_stats["level"].eq("GROUP")
].copy()


group_means = {}


for row in group_rows.itertuples(
    index=False
):

    parts = str(
        row.group
    ).split("|")

    if len(parts) != 2:

        raise RuntimeError(
            f"Malformed GROUP identifier: {row.group}"
        )

    city, sensor = parts

    group_means[
        (
            city,
            sensor
        )
    ] = float(
        row.mean_c
    )


if len(group_means) != 8:

    raise RuntimeError(
        f"Expected 8 city-sensor climatologies, "
        f"found {len(group_means)}."
    )


print("=" * 120)
print("THERMOFUSION STAGE 07 — FROZEN BASELINE COMPARISON")
print("=" * 120)

print(f"External scenes                    : {len(arrays)}")
print(f"Global climatology mean (°C)       : {global_mean:.4f}")
print(f"Sensor climatologies               : {len(sensor_means)}/2")
print(f"City-sensor climatologies          : {len(group_means)}/8")
print("Baseline refitting on external52   : NO")
print(f"Bootstrap replicates               : {BOOTSTRAPS:,}")


# ======================================================================================
# 6. METRICS
# ======================================================================================

def constant_metrics(
    observed,
    constant
):

    observed = np.asarray(
        observed,
        dtype="float64"
    )

    predicted = np.full(
        observed.shape,
        float(constant),
        dtype="float64"
    )

    error = (
        predicted
        - observed
    )

    denominator = np.sum(
        (
            observed
            - observed.mean()
        ) ** 2
    )

    r2 = (
        1.0
        - np.sum(error ** 2)
        / denominator

        if denominator > 0

        else np.nan
    )

    return {
        "mae_c":
            float(
                np.mean(
                    np.abs(error)
                )
            ),

        "rmse_c":
            float(
                np.sqrt(
                    np.mean(
                        error ** 2
                    )
                )
            ),

        "bias_c":
            float(
                np.mean(error)
            ),

        "r2":
            float(r2),
    }


# ======================================================================================
# 7. EVALUATE ALL THREE FROZEN BASELINES
# ======================================================================================

baseline_rows = []


for number, row in enumerate(
    arrays.sort_values(
        "export_id"
    ).itertuples(
        index=False
    ),
    start=1
):

    with np.load(
        row.array_path,
        allow_pickle=False
    ) as item:

        observed_grid = (
            item[
                "y_c"
            ]
            .astype(
                "float32"
            )
        )

        valid = (
            item[
                "valid_mask"
            ]
            .astype(bool)
        )


    if valid.sum() == 0:

        raise RuntimeError(
            f"{row.export_id}: no valid target pixels."
        )


    y = observed_grid[
        valid
    ]


    sensor_mean = sensor_means[
        row.thermal_sensor
    ]


    group_key = (
        row.city,
        row.thermal_sensor
    )


    if group_key not in group_means:

        raise RuntimeError(
            f"No frozen city-sensor climatology for "
            f"{group_key}."
        )


    city_sensor_mean = group_means[
        group_key
    ]


    baseline_definitions = [
        (
            "global_training_climatology",
            global_mean
        ),

        (
            "sensor_training_climatology",
            sensor_mean
        ),

        (
            "city_sensor_training_climatology",
            city_sensor_mean
        ),
    ]


    for baseline_name, climatology_mean in baseline_definitions:

        result = constant_metrics(
            y,
            climatology_mean
        )


        baseline_rows.append(
            {
                "export_id":
                    row.export_id,

                "external_id":
                    row.external_id,

                "record_id":
                    row.record_id,

                "city":
                    row.city,

                "thermal_sensor":
                    row.thermal_sensor,

                "baseline":
                    baseline_name,

                "climatology_mean_c":
                    climatology_mean,

                "valid_pixels":
                    int(
                        valid.sum()
                    ),

                **result,
            }
        )


    print(
        f"Baseline {number:02d}/52: "
        f"{row.export_id} | "
        f"{row.city} | "
        f"{row.thermal_sensor}"
    )


baselines = pd.DataFrame(
    baseline_rows
)


BASELINE_PATH = (
    OUT
    / "01_external52_scene_baseline_metrics.csv"
)


baselines.to_csv(
    BASELINE_PATH,
    index=False
)


# ======================================================================================
# 8. JOIN FROZEN THERMOFUSION RESULT
# ======================================================================================

model_keep = model[
    [
        "export_id",
        "external_id",
        "record_id",
        "city",
        "thermal_sensor",
        "mae_c",
        "rmse_c",
        "bias_c",
        "r2",
        "interval_coverage",
        "mean_interval_width_c",
        "median_tta_std_c",
        "target_valid_fraction",
        "s2_valid_fraction",
        "s1_valid_fraction",
    ]
].copy()


model_keep = model_keep.rename(
    columns={
        "mae_c":
            "model_mae_c",

        "rmse_c":
            "model_rmse_c",

        "bias_c":
            "model_bias_c",

        "r2":
            "model_r2",
    }
)


wide = baselines.pivot(
    index=[
        "export_id",
        "external_id",
        "record_id",
        "city",
        "thermal_sensor",
    ],
    columns="baseline",
    values=[
        "mae_c",
        "rmse_c",
        "bias_c",
    ]
)


wide.columns = [
    f"{baseline}_{metric}"

    for metric, baseline
    in wide.columns
]


wide = (
    wide
    .reset_index()
)


paired = model_keep.merge(
    wide,
    on=[
        "export_id",
        "external_id",
        "record_id",
        "city",
        "thermal_sensor",
    ],
    how="inner",
    validate="one_to_one"
)


if len(paired) != EXPECTED_SCENES:

    raise RuntimeError(
        f"Paired table contains {len(paired)} scenes instead of 52."
    )


# ======================================================================================
# 9. PAIRED DIFFERENCES
#
# delta_rmse = ThermoFusion - climatology
# negative values favour ThermoFusion.
#
# advantage = climatology - ThermoFusion
# positive values favour ThermoFusion.
# ======================================================================================

BASELINES = [
    "global_training_climatology",
    "sensor_training_climatology",
    "city_sensor_training_climatology",
]


for baseline_name in BASELINES:

    baseline_rmse_col = (
        f"{baseline_name}_rmse_c"
    )

    paired[
        f"delta_rmse_vs_{baseline_name}_c"
    ] = (
        paired[
            "model_rmse_c"
        ]
        -
        paired[
            baseline_rmse_col
        ]
    )


    paired[
        f"advantage_vs_{baseline_name}_c"
    ] = (
        paired[
            baseline_rmse_col
        ]
        -
        paired[
            "model_rmse_c"
        ]
    )


PAIRED_PATH = (
    OUT
    / "02_external52_paired_scene_comparison.csv"
)


paired.to_csv(
    PAIRED_PATH,
    index=False
)


# ======================================================================================
# 10. SCENE-MACRO MODEL AND BASELINE SUMMARY
# ======================================================================================

summary_rows = []


summary_rows.append(
    {
        "model":
            "ThermoFusion",

        "scenes":
            len(paired),

        "scene_macro_mae_c":
            paired[
                "model_mae_c"
            ].mean(),

        "scene_macro_rmse_c":
            paired[
                "model_rmse_c"
            ].mean(),

        "scene_macro_bias_c":
            paired[
                "model_bias_c"
            ].mean(),
    }
)


for baseline_name in BASELINES:

    summary_rows.append(
        {
            "model":
                baseline_name,

            "scenes":
                len(paired),

            "scene_macro_mae_c":
                baselines.loc[
                    baselines[
                        "baseline"
                    ].eq(
                        baseline_name
                    ),
                    "mae_c"
                ].mean(),

            "scene_macro_rmse_c":
                baselines.loc[
                    baselines[
                        "baseline"
                    ].eq(
                        baseline_name
                    ),
                    "rmse_c"
                ].mean(),

            "scene_macro_bias_c":
                baselines.loc[
                    baselines[
                        "baseline"
                    ].eq(
                        baseline_name
                    ),
                    "bias_c"
                ].mean(),
        }
    )


macro_summary = pd.DataFrame(
    summary_rows
)


SUMMARY_PATH = (
    OUT
    / "03_external52_model_baseline_summary.csv"
)


macro_summary.to_csv(
    SUMMARY_PATH,
    index=False
)


# ======================================================================================
# 11. ORDINARY 50,000-REPLICATE PAIRED SCENE BOOTSTRAP
#
# Resampling unit = scene/acquisition.
# ======================================================================================

rng = np.random.default_rng(
    SEED
)


bootstrap_rows = []


for baseline_name in BASELINES:

    delta = paired[
        f"delta_rmse_vs_{baseline_name}_c"
    ].to_numpy(
        dtype="float64"
    )


    n = len(
        delta
    )


    # Vectorized bootstrap in chunks to avoid unnecessary memory use.
    chunk_size = 5000

    draws = np.empty(
        BOOTSTRAPS,
        dtype="float64"
    )


    position = 0


    while position < BOOTSTRAPS:

        this_chunk = min(
            chunk_size,
            BOOTSTRAPS - position
        )


        indices = rng.integers(
            0,
            n,
            size=(
                this_chunk,
                n
            )
        )


        draws[
            position:
            position + this_chunk
        ] = (
            delta[
                indices
            ]
            .mean(
                axis=1
            )
        )


        position += (
            this_chunk
        )


    ci_lower = float(
        np.quantile(
            draws,
            ALPHA / 2
        )
    )


    ci_upper = float(
        np.quantile(
            draws,
            1 - ALPHA / 2
        )
    )


    probability_better = float(
        np.mean(
            draws < 0
        )
    )


    observed_delta = float(
        delta.mean()
    )


    observed_advantage = (
        -observed_delta
    )


    scene_wins = int(
        np.sum(
            delta < 0
        )
    )


    ties = int(
        np.sum(
            np.isclose(
                delta,
                0.0
            )
        )
    )


    bootstrap_rows.append(
        {
            "comparison":
                (
                    "ThermoFusion minus "
                    + baseline_name
                ),

            "baseline":
                baseline_name,

            "scenes":
                n,

            "mean_delta_rmse_c":
                observed_delta,

            "mean_advantage_c":
                observed_advantage,

            "ci_lower_95_delta_c":
                ci_lower,

            "ci_upper_95_delta_c":
                ci_upper,

            "probability_thermofusion_better":
                probability_better,

            "thermofusion_scene_wins":
                scene_wins,

            "thermofusion_scene_win_fraction":
                scene_wins / n,

            "ties":
                ties,

            "bootstrap_replicates":
                BOOTSTRAPS,

            "bootstrap_unit":
                "scene",
        }
    )


bootstrap = pd.DataFrame(
    bootstrap_rows
)


BOOTSTRAP_PATH = (
    OUT
    / "04_external52_paired_scene_bootstrap.csv"
)


bootstrap.to_csv(
    BOOTSTRAP_PATH,
    index=False
)


# ======================================================================================
# 12. CITY × SENSOR RESULTS
# ======================================================================================

subgroup_rows = []


for (
    city,
    sensor
), group in paired.groupby(
    [
        "city",
        "thermal_sensor"
    ],
    observed=True
):

    for baseline_name in BASELINES:

        baseline_rmse_col = (
            f"{baseline_name}_rmse_c"
        )


        delta_col = (
            f"delta_rmse_vs_{baseline_name}_c"
        )


        subgroup_rows.append(
            {
                "city":
                    city,

                "thermal_sensor":
                    sensor,

                "baseline":
                    baseline_name,

                "scenes":
                    len(group),

                "model_scene_macro_rmse_c":
                    float(
                        group[
                            "model_rmse_c"
                        ].mean()
                    ),

                "baseline_scene_macro_rmse_c":
                    float(
                        group[
                            baseline_rmse_col
                        ].mean()
                    ),

                "mean_delta_rmse_c":
                    float(
                        group[
                            delta_col
                        ].mean()
                    ),

                "mean_advantage_c":
                    float(
                        -group[
                            delta_col
                        ].mean()
                    ),

                "model_scene_win_fraction":
                    float(
                        (
                            group[
                                delta_col
                            ]
                            < 0
                        ).mean()
                    ),

                "model_scene_macro_mae_c":
                    float(
                        group[
                            "model_mae_c"
                        ].mean()
                    ),

                "model_scene_macro_bias_c":
                    float(
                        group[
                            "model_bias_c"
                        ].mean()
                    ),

                "mean_interval_coverage":
                    float(
                        group[
                            "interval_coverage"
                        ].mean()
                    ),
            }
        )


subgroups = pd.DataFrame(
    subgroup_rows
)


SUBGROUP_PATH = (
    OUT
    / "05_external52_city_sensor_baseline_comparison.csv"
)


subgroups.to_csv(
    SUBGROUP_PATH,
    index=False
)


# ======================================================================================
# 13. SENSOR-LEVEL COMPARISON
# ======================================================================================

sensor_result_rows = []


for sensor, group in paired.groupby(
    "thermal_sensor",
    observed=True
):

    for baseline_name in BASELINES:

        baseline_rmse_col = (
            f"{baseline_name}_rmse_c"
        )


        delta_col = (
            f"delta_rmse_vs_{baseline_name}_c"
        )


        sensor_result_rows.append(
            {
                "thermal_sensor":
                    sensor,

                "baseline":
                    baseline_name,

                "scenes":
                    len(group),

                "model_rmse_c":
                    float(
                        group[
                            "model_rmse_c"
                        ].mean()
                    ),

                "baseline_rmse_c":
                    float(
                        group[
                            baseline_rmse_col
                        ].mean()
                    ),

                "mean_delta_rmse_c":
                    float(
                        group[
                            delta_col
                        ].mean()
                    ),

                "mean_advantage_c":
                    float(
                        -group[
                            delta_col
                        ].mean()
                    ),

                "model_win_fraction":
                    float(
                        (
                            group[
                                delta_col
                            ]
                            < 0
                        ).mean()
                    ),
            }
        )


sensor_results = pd.DataFrame(
    sensor_result_rows
)


SENSOR_PATH = (
    OUT
    / "06_external52_sensor_baseline_comparison.csv"
)


sensor_results.to_csv(
    SENSOR_PATH,
    index=False
)


# ======================================================================================
# 14. EQUAL-OBSERVED-STRATUM SENSITIVITY
#
# IMPORTANT:
# External52 is highly imbalanced:
# Accra ECOSTRESS contributes 34/52 scenes.
#
# This sensitivity gives each observed city × sensor stratum equal weight.
# It does NOT invent absent strata.
# ======================================================================================

observed_strata = (
    paired[
        [
            "city",
            "thermal_sensor"
        ]
    ]
    .drop_duplicates()
    .sort_values(
        [
            "city",
            "thermal_sensor"
        ]
    )
)


strata = [
    (
        row.city,
        row.thermal_sensor
    )

    for row in observed_strata.itertuples(
        index=False
    )
]


equal_stratum_rows = []


for baseline_name in BASELINES:

    delta_col = (
        f"delta_rmse_vs_{baseline_name}_c"
    )


    stratum_effects = []


    for city, sensor in strata:

        values = paired.loc[
            (
                paired[
                    "city"
                ].eq(
                    city
                )
            )
            &
            (
                paired[
                    "thermal_sensor"
                ].eq(
                    sensor
                )
            ),
            delta_col
        ].to_numpy(
            dtype="float64"
        )


        stratum_effects.append(
            values.mean()
        )


    observed_equal_delta = float(
        np.mean(
            stratum_effects
        )
    )


    # ------------------------------------------------------------------
    # Stratified bootstrap:
    # resample scenes WITHIN each observed stratum,
    # then average the stratum means equally.
    # ------------------------------------------------------------------

    strat_rng = np.random.default_rng(
        SEED
        + sum(
            map(
                ord,
                baseline_name
            )
        )
    )


    draws = np.empty(
        BOOTSTRAPS,
        dtype="float64"
    )


    for b in range(
        BOOTSTRAPS
    ):

        boot_strata = []


        for city, sensor in strata:

            values = paired.loc[
                (
                    paired[
                        "city"
                    ].eq(
                        city
                    )
                )
                &
                (
                    paired[
                        "thermal_sensor"
                    ].eq(
                        sensor
                    )
                ),
                delta_col
            ].to_numpy(
                dtype="float64"
            )


            sampled = (
                strat_rng.choice(
                    values,
                    size=len(values),
                    replace=True
                )
            )


            boot_strata.append(
                sampled.mean()
            )


        draws[
            b
        ] = np.mean(
            boot_strata
        )


    equal_stratum_rows.append(
        {
            "baseline":
                baseline_name,

            "observed_strata":
                len(strata),

            "equal_stratum_mean_delta_rmse_c":
                observed_equal_delta,

            "equal_stratum_mean_advantage_c":
                -observed_equal_delta,

            "ci_lower_95_delta_c":
                float(
                    np.quantile(
                        draws,
                        0.025
                    )
                ),

            "ci_upper_95_delta_c":
                float(
                    np.quantile(
                        draws,
                        0.975
                    )
                ),

            "probability_thermofusion_better":
                float(
                    np.mean(
                        draws < 0
                    )
                ),

            "weighting":
                (
                    "equal weight across observed "
                    "city-sensor strata"
                ),

            "bootstrap_replicates":
                BOOTSTRAPS,
        }
    )


equal_stratum = pd.DataFrame(
    equal_stratum_rows
)


EQUAL_STRATUM_PATH = (
    OUT
    / "07_external52_equal_stratum_sensitivity.csv"
)


equal_stratum.to_csv(
    EQUAL_STRATUM_PATH,
    index=False
)


# ======================================================================================
# 15. SUPPORT / PERFORMANCE DIAGNOSTICS
#
# These are descriptive diagnostics only.
# They are NOT scene-exclusion criteria.
# ======================================================================================

diagnostic_rows = []


for variable in [
    "target_valid_fraction",
    "s2_valid_fraction",
    "s1_valid_fraction",
]:

    x = paired[
        variable
    ].to_numpy(
        dtype="float64"
    )


    y = paired[
        "model_rmse_c"
    ].to_numpy(
        dtype="float64"
    )


    if (
        np.nanstd(x)
        > 0
    ):

        correlation = float(
            pd.Series(x)
            .corr(
                pd.Series(y),
                method="spearman"
            )
        )

    else:

        correlation = np.nan


    diagnostic_rows.append(
        {
            "predictor":
                variable,

            "outcome":
                "model_rmse_c",

            "spearman_rho":
                correlation,

            "interpretation":
                (
                    "descriptive only; "
                    "not used for scene selection"
                ),
        }
    )


diagnostics = pd.DataFrame(
    diagnostic_rows
)


DIAGNOSTIC_PATH = (
    OUT
    / "08_external52_support_error_diagnostics.csv"
)


diagnostics.to_csv(
    DIAGNOSTIC_PATH,
    index=False
)


# ======================================================================================
# 16. IDENTIFY BEST FROZEN CLIMATOLOGY
# ======================================================================================

baseline_macro = (
    macro_summary.loc[
        macro_summary[
            "model"
        ].ne(
            "ThermoFusion"
        )
    ]
    .sort_values(
        "scene_macro_rmse_c"
    )
    .reset_index(
        drop=True
    )
)


best_baseline = (
    baseline_macro.iloc[0]
)


best_name = str(
    best_baseline[
        "model"
    ]
)


best_rmse = float(
    best_baseline[
        "scene_macro_rmse_c"
    ]
)


model_rmse = float(
    macro_summary.loc[
        macro_summary[
            "model"
        ].eq(
            "ThermoFusion"
        ),
        "scene_macro_rmse_c"
    ].iloc[0]
)


best_boot = (
    bootstrap.loc[
        bootstrap[
            "baseline"
        ].eq(
            best_name
        )
    ].iloc[0]
)


# ======================================================================================
# 17. FINAL INTEGRITY CHECKS
# ======================================================================================

required_numeric = [
    "model_rmse_c",
    "global_training_climatology_rmse_c",
    "sensor_training_climatology_rmse_c",
    "city_sensor_training_climatology_rmse_c",
]


finite_check = (
    np.isfinite(
        paired[
            required_numeric
        ].to_numpy()
    )
    .all()
)


bootstrap_check = (
    len(bootstrap) == 3
    and
    np.isfinite(
        bootstrap[
            [
                "mean_delta_rmse_c",
                "ci_lower_95_delta_c",
                "ci_upper_95_delta_c",
                "probability_thermofusion_better",
            ]
        ].to_numpy()
    ).all()
)


overall_pass = (
    len(paired)
    == 52

    and

    paired[
        "record_id"
    ].nunique()
    == 52

    and

    len(baselines)
    == 156

    and

    finite_check

    and

    bootstrap_check

    and

    len(equal_stratum)
    == 3
)


# ======================================================================================
# 18. SAVE CONFIGURATION
# ======================================================================================

configuration = {
    "seed":
        SEED,

    "external_scenes":
        52,

    "baselines": [
        "global_training_climatology",
        "sensor_training_climatology",
        "city_sensor_training_climatology",
    ],

    "baseline_source":
        (
            "ThermoFusion Stage 14 "
            "training-only target statistics"
        ),

    "bootstrap_replicates":
        BOOTSTRAPS,

    "confidence_level":
        0.95,

    "primary_bootstrap_unit":
        "scene",

    "primary_effect":
        (
            "mean scene RMSE difference: "
            "ThermoFusion minus frozen climatology"
        ),

    "negative_delta_favours":
        "ThermoFusion",

    "equal_stratum_sensitivity":
        True,

    "external_data_used_for_baseline_fitting":
        False,

    "external_data_used_for_model_fitting":
        False,

    "performance_based_scene_exclusion":
        False,
}


CONFIG_PATH = (
    OUT
    / "09_stage07_configuration.json"
)


with CONFIG_PATH.open(
    "w",
    encoding="utf-8"
) as handle:

    json.dump(
        configuration,
        handle,
        indent=2
    )


# ======================================================================================
# 19. VERDICT
# ======================================================================================

verdict = [

    "THERMOFUSION STAGE 07 — EXTERNAL52 FROZEN BASELINE COMPARISON",

    "Independent external scenes: 52/52",

    "Performance-based scene exclusions: 0",

    "Model fitting on external52: NO",

    "Climatology fitting on external52: NO",

    "Normalization refitting on external52: NO",

    f"Bootstrap replicates: {BOOTSTRAPS:,}",

    (
        f"ThermoFusion scene-macro RMSE (°C): "
        f"{model_rmse:.4f}"
    ),

    (
        f"Global climatology scene-macro RMSE (°C): "
        f"{macro_summary.loc[macro_summary.model.eq('global_training_climatology'), 'scene_macro_rmse_c'].iloc[0]:.4f}"
    ),

    (
        f"Sensor climatology scene-macro RMSE (°C): "
        f"{macro_summary.loc[macro_summary.model.eq('sensor_training_climatology'), 'scene_macro_rmse_c'].iloc[0]:.4f}"
    ),

    (
        f"City-sensor climatology scene-macro RMSE (°C): "
        f"{macro_summary.loc[macro_summary.model.eq('city_sensor_training_climatology'), 'scene_macro_rmse_c'].iloc[0]:.4f}"
    ),

    (
        f"Best frozen climatology baseline: "
        f"{best_name}"
    ),

    (
        f"Best baseline scene-macro RMSE (°C): "
        f"{best_rmse:.4f}"
    ),

    (
        f"ThermoFusion minus best-baseline RMSE (°C): "
        f"{best_boot.mean_delta_rmse_c:.4f}"
    ),

    (
        f"95% scene-bootstrap CI (°C): "
        f"[{best_boot.ci_lower_95_delta_c:.4f}, "
        f"{best_boot.ci_upper_95_delta_c:.4f}]"
    ),

    (
        f"Bootstrap probability ThermoFusion better: "
        f"{best_boot.probability_thermofusion_better:.4f}"
    ),

    (
        f"ThermoFusion scene wins vs best baseline: "
        f"{int(best_boot.thermofusion_scene_wins)}/52"
    ),

    (
        "Overall Stage 07 baseline-comparison verdict: PASS"
        if overall_pass
        else
        "Overall Stage 07 baseline-comparison verdict: REVIEW REQUIRED"
    ),
]


VERDICT_PATH = (
    OUT
    / "10_stage07_baseline_comparison_verdict.txt"
)


VERDICT_PATH.write_text(
    "\n".join(
        verdict
    ),
    encoding="utf-8"
)


# ======================================================================================
# 20. REPORT
# ======================================================================================

print(
    "\n"
    + "=" * 120
)

print(
    "\n".join(
        verdict
    )
)

print(
    "=" * 120
)


print(
    "\nMODEL / BASELINE SUMMARY"
)

print(
    "-" * 120
)

print(
    macro_summary.to_string(
        index=False
    )
)


print(
    "\nPAIRED 50,000-REPLICATE SCENE BOOTSTRAP"
)

print(
    "-" * 120
)

print(
    bootstrap.to_string(
        index=False
    )
)


print(
    "\nEQUAL-OBSERVED-STRATUM SENSITIVITY"
)

print(
    "-" * 120
)

print(
    equal_stratum.to_string(
        index=False
    )
)


print(
    "\nCITY × SENSOR RESULTS AGAINST CITY-SENSOR CLIMATOLOGY"
)

print(
    "-" * 120
)

print(
    subgroups.loc[
        subgroups[
            "baseline"
        ].eq(
            "city_sensor_training_climatology"
        )
    ].to_string(
        index=False
    )
)


print(
    "\nSUPPORT / ERROR DIAGNOSTICS"
)

print(
    "-" * 120
)

print(
    diagnostics.to_string(
        index=False
    )
)


print(
    "\nSaved:"
)

for path in [
    BASELINE_PATH,
    PAIRED_PATH,
    SUMMARY_PATH,
    BOOTSTRAP_PATH,
    SUBGROUP_PATH,
    SENSOR_PATH,
    EQUAL_STRATUM_PATH,
    DIAGNOSTIC_PATH,
    CONFIG_PATH,
    VERDICT_PATH,
]:

    print(
        path
    )


if overall_pass:

    print(
        "\nPASS — frozen baseline comparison "
        "completed for all 52 external scenes."
    )

    print(
        "NEXT STAGE: integrate the existing 38-scene "
        "stress test and the new 52-scene cohort, "
        "audit overlap, and produce the final expanded "
        "external temporal-evaluation evidence."
    )

else:

    print(
        "\nSTOP — Stage 07 requires review."
    )
