# =====================================================================
# ThermoFusion reproducibility code
# Source notebook cell(s): 71
# Audit status: KEEP
# Extracted without scientific-logic modification.
# Replace local Google Drive paths as described in README.md when needed.
# =====================================================================

# ======================================================================================
# THERMOFUSION — EXPANDED EXTERNAL TEMPORAL EVALUATION
# STAGE 08: INTEGRATE ORIGINAL38 + NEW52
#
# PURPOSE
#   1. Audit exact overlap between the original 38-scene stress test and new 52.
#   2. Recover/reconstruct scene-level evidence for the original 38.
#   3. Combine cohorts ONLY if overlap = 0.
#   4. Produce final 90-scene descriptive and inferential evidence.
#
# STRICT RULES
#   - NO model fitting
#   - NO normalization refitting
#   - NO climatology refitting
#   - NO conformal recalibration
#   - NO performance-based exclusions
#   - Scene/acquisition remains the inferential unit
#
# GPU NOT REQUIRED
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

# ----------------------------------------------------------------------
# ORIGINAL 38-SCENE COHORT
# ----------------------------------------------------------------------

OLD_MANIFEST = (
    ROOT
    / "ThermoFusion_Stage23R2_FinalReplacement"
    / "01_final_expanded_test_manifest.csv"
)

OLD_STAGE25 = (
    ROOT
    / "ThermoFusion_Stage25_ExpandedTest_ModelReady"
)

OLD_STAGE26 = (
    ROOT
    / "ThermoFusion_Stage26_ExpandedTemporalTest_Inference"
)

OLD_STAGE27 = (
    ROOT
    / "ThermoFusion_Stage27_ExpandedTemporalTest_Analysis"
)

# ----------------------------------------------------------------------
# NEW 52-SCENE COHORT
# ----------------------------------------------------------------------

NEW_STAGE02 = (
    ROOT
    / "ThermoFusion_External_Evaluation"
    / "Stage02_final_new_external_test_manifest.csv"
)

# fallback because Stage02 files may live directly under External_Evaluation
NEW_STAGE02_ALT = (
    ROOT
    / "ThermoFusion_External_Evaluation"
    / "Stage02_final_new_external_test_manifest.csv"
)

NEW_STAGE05 = (
    ROOT
    / "ThermoFusion_External_Evaluation"
    / "Stage05_ModelReady_External52"
)

NEW_STAGE06 = (
    ROOT
    / "ThermoFusion_External_Evaluation"
    / "Stage06_FrozenInference_External52"
)

NEW_STAGE07 = (
    ROOT
    / "ThermoFusion_External_Evaluation"
    / "Stage07_BaselineComparison_External52"
)

NEW_ARRAY_MANIFEST = (
    NEW_STAGE05
    / "01_external52_model_ready_manifest.csv"
)

NEW_MODEL_METRICS = (
    NEW_STAGE06
    / "01_external52_scene_accuracy_uncertainty.csv"
)

NEW_PAIRED = (
    NEW_STAGE07
    / "02_external52_paired_scene_comparison.csv"
)

NEW_STAGE07_VERDICT = (
    NEW_STAGE07
    / "10_stage07_baseline_comparison_verdict.txt"
)

# ----------------------------------------------------------------------
# ORIGINAL TRAINING-ONLY STATISTICS
# ----------------------------------------------------------------------

TRAINING_STATS = (
    ROOT
    / "ThermoFusion_Stage14_Refined_Benchmark"
    / "01_training_only_target_statistics.csv"
)

# ----------------------------------------------------------------------
# OUTPUT
# ----------------------------------------------------------------------

OUT = (
    ROOT
    / "ThermoFusion_External_Evaluation"
    / "Stage08_IntegratedExternal90"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# ======================================================================================
# 2. SETTINGS
# ======================================================================================

SEED = 20260908

EXPECTED_OLD = 38
EXPECTED_NEW = 52
EXPECTED_COMBINED = 90

BOOTSTRAPS = 50000
ALPHA = 0.05

BASELINES = [
    "global_training_climatology",
    "sensor_training_climatology",
    "city_sensor_training_climatology",
]


# ======================================================================================
# 3. BASIC INPUT CHECKS
# ======================================================================================

required = [
    OLD_MANIFEST,
    NEW_ARRAY_MANIFEST,
    NEW_MODEL_METRICS,
    NEW_PAIRED,
    NEW_STAGE07_VERDICT,
    TRAINING_STATS,
]

for path in required:

    if not path.exists():

        raise FileNotFoundError(
            f"Required input not found:\n{path}"
        )


if (
    "Overall Stage 07 baseline-comparison verdict: PASS"
    not in NEW_STAGE07_VERDICT.read_text(
        encoding="utf-8"
    )
):

    raise RuntimeError(
        "New 52-scene Stage 07 did not pass."
    )


print("=" * 120)
print("THERMOFUSION STAGE 08 — INTEGRATED EXTERNAL TEMPORAL EVALUATION")
print("=" * 120)


# ======================================================================================
# 4. LOAD ORIGINAL38 AND NEW52 MANIFESTS
# ======================================================================================

old_manifest = pd.read_csv(
    OLD_MANIFEST
)

new_manifest = pd.read_csv(
    NEW_ARRAY_MANIFEST
)


if len(old_manifest) != EXPECTED_OLD:

    raise RuntimeError(
        f"Expected 38 original external scenes, found {len(old_manifest)}."
    )


if len(new_manifest) != EXPECTED_NEW:

    raise RuntimeError(
        f"Expected 52 new external scenes, found {len(new_manifest)}."
    )


if "record_id" not in old_manifest.columns:

    raise RuntimeError(
        "Original38 manifest has no record_id column."
    )


if "record_id" not in new_manifest.columns:

    raise RuntimeError(
        "New52 manifest has no record_id column."
    )


if old_manifest["record_id"].duplicated().any():

    raise RuntimeError(
        "Duplicate record IDs inside original38."
    )


if new_manifest["record_id"].duplicated().any():

    raise RuntimeError(
        "Duplicate record IDs inside new52."
    )


# ======================================================================================
# 5. OVERLAP AUDIT — RECORD ID
# ======================================================================================

old_record_ids = set(
    old_manifest[
        "record_id"
    ].astype(str)
)

new_record_ids = set(
    new_manifest[
        "record_id"
    ].astype(str)
)


record_overlap = sorted(
    old_record_ids
    &
    new_record_ids
)


# ======================================================================================
# 6. OVERLAP AUDIT — THERMAL SCENE ID
# ======================================================================================

thermal_overlap = []


if (
    "thermal_scene_id"
    in old_manifest.columns
):

    old_scene_ids = set(
        old_manifest[
            "thermal_scene_id"
        ]
        .dropna()
        .astype(str)
    )

else:

    old_scene_ids = set()


# New Stage05 manifest may not carry thermal_scene_id.
# Recover from Stage02 if necessary.

candidate_stage02_paths = [
    NEW_STAGE02,
    NEW_STAGE02_ALT,
]


new_stage02 = None


for path in candidate_stage02_paths:

    if path.exists():

        frame = pd.read_csv(
            path
        )

        if (
            "record_id"
            in frame.columns
        ):

            new_stage02 = frame

            break


if (
    new_stage02 is not None
    and
    "thermal_scene_id"
    in new_stage02.columns
):

    new_scene_ids = set(
        new_stage02[
            "thermal_scene_id"
        ]
        .dropna()
        .astype(str)
    )

else:

    new_scene_ids = set()


if (
    old_scene_ids
    and new_scene_ids
):

    thermal_overlap = sorted(
        old_scene_ids
        &
        new_scene_ids
    )


# ======================================================================================
# 7. SAVE OVERLAP AUDIT
# ======================================================================================

overlap_rows = []


for value in record_overlap:

    overlap_rows.append(
        {
            "overlap_type":
                "record_id",

            "value":
                value,
        }
    )


for value in thermal_overlap:

    overlap_rows.append(
        {
            "overlap_type":
                "thermal_scene_id",

            "value":
                value,
        }
    )


overlap_table = pd.DataFrame(
    overlap_rows,
    columns=[
        "overlap_type",
        "value"
    ]
)


OVERLAP_PATH = (
    OUT
    / "01_original38_new52_overlap_audit.csv"
)


overlap_table.to_csv(
    OVERLAP_PATH,
    index=False
)


print("\nOVERLAP AUDIT")
print("-" * 120)

print(
    f"Original cohort scenes : {len(old_manifest)}"
)

print(
    f"New cohort scenes      : {len(new_manifest)}"
)

print(
    f"record_id overlap      : {len(record_overlap)}"
)

print(
    f"thermal_scene_id overlap: {len(thermal_overlap)}"
)


if record_overlap:

    print(
        "\nOVERLAPPING RECORD IDs:"
    )

    for value in record_overlap:

        print(
            value
        )


if thermal_overlap:

    print(
        "\nOVERLAPPING THERMAL SCENE IDs:"
    )

    for value in thermal_overlap:

        print(
            value
        )


if (
    len(record_overlap) > 0
    or
    len(thermal_overlap) > 0
):

    raise RuntimeError(
        "STOP — original38 and new52 are not independent. "
        "Resolve overlap before pooling."
    )


print(
    "\nPASS — no overlap detected between original38 and new52."
)


# ======================================================================================
# 8. LOAD FROZEN TRAINING-ONLY CLIMATOLOGIES
# ======================================================================================

training_stats = pd.read_csv(
    TRAINING_STATS
)


global_rows = training_stats.loc[
    training_stats[
        "level"
    ].eq(
        "GLOBAL"
    )
]


if len(global_rows) != 1:

    raise RuntimeError(
        "Expected exactly one GLOBAL training statistic."
    )


GLOBAL_MEAN = float(
    global_rows.iloc[0][
        "mean_c"
    ]
)


sensor_means = {

    str(row.group):
        float(
            row.mean_c
        )

    for row in training_stats.loc[
        training_stats[
            "level"
        ].eq(
            "SENSOR"
        )
    ].itertuples(
        index=False
    )
}


city_sensor_means = {}


for row in training_stats.loc[
    training_stats[
        "level"
    ].eq(
        "GROUP"
    )
].itertuples(
    index=False
):

    city, sensor = str(
        row.group
    ).split("|")

    city_sensor_means[
        (
            city,
            sensor
        )
    ] = float(
        row.mean_c
    )


if len(sensor_means) != 2:

    raise RuntimeError(
        "Expected 2 sensor climatologies."
    )


if len(city_sensor_means) != 8:

    raise RuntimeError(
        "Expected 8 city-sensor climatologies."
    )


# ======================================================================================
# 9. DISCOVER ORIGINAL38 MODEL-READY MANIFEST
#
# We need the original target arrays so baseline metrics can be reconstructed exactly.
# Instead of guessing one filename, search Stage25 recursively.
# ======================================================================================

def inspect_csvs(
    root,
    expected_rows=None
):

    results = []

    if not root.exists():

        return results


    for path in root.rglob(
        "*.csv"
    ):

        try:

            df = pd.read_csv(
                path
            )

        except Exception:

            continue


        result = {
            "path":
                path,

            "rows":
                len(df),

            "columns":
                list(
                    df.columns
                ),
        }


        if (
            expected_rows is None
            or
            len(df)
            == expected_rows
        ):

            results.append(
                result
            )


    return results


stage25_csvs = inspect_csvs(
    OLD_STAGE25,
    EXPECTED_OLD
)


array_candidates = []


for item in stage25_csvs:

    cols = set(
        item[
            "columns"
        ]
    )


    score = 0


    for column in [
        "record_id",
        "array_path",
        "city",
        "thermal_sensor",
    ]:

        if column in cols:

            score += 1


    if score >= 3:

        array_candidates.append(
            (
                score,
                item[
                    "path"
                ]
            )
        )


array_candidates = sorted(
    array_candidates,
    key=lambda x: (
        -x[0],
        str(
            x[1]
        )
    )
)


if not array_candidates:

    print(
        "\nStage25 CSV files discovered:"
    )

    for item in stage25_csvs:

        print(
            item[
                "path"
            ],
            item[
                "columns"
            ]
        )

    raise RuntimeError(
        "Could not automatically identify "
        "the original38 model-ready array manifest."
    )


OLD_ARRAY_MANIFEST_PATH = (
    array_candidates[0][1]
)


old_arrays = pd.read_csv(
    OLD_ARRAY_MANIFEST_PATH
)


print(
    f"\nOriginal38 model-ready manifest discovered:\n"
    f"{OLD_ARRAY_MANIFEST_PATH}"
)


if len(old_arrays) != EXPECTED_OLD:

    raise RuntimeError(
        "Original model-ready table is not 38 rows."
    )


if "array_path" not in old_arrays.columns:

    raise RuntimeError(
        "Original array manifest has no array_path."
    )


# ======================================================================================
# 10. DISCOVER ORIGINAL38 SCENE-LEVEL FROZEN MODEL METRICS
#
# Search Stage26 and Stage27 for a 38-row CSV containing scene RMSE.
# ======================================================================================

metric_roots = [
    OLD_STAGE26,
    OLD_STAGE27,
]


metric_candidates = []


for root in metric_roots:

    for item in inspect_csvs(
        root,
        EXPECTED_OLD
    ):

        cols = set(
            item[
                "columns"
        ]
    )


        score = 0


        # ID columns
        if "record_id" in cols:
            score += 3

        if "city" in cols:
            score += 1

        if "thermal_sensor" in cols:
            score += 1


        # likely metric columns
        if "rmse_c" in cols:
            score += 4

        if "mae_c" in cols:
            score += 2

        if "bias_c" in cols:
            score += 2

        if "interval_coverage" in cols:
            score += 2

        if "mean_interval_width_c" in cols:
            score += 1


        if score >= 5:

            metric_candidates.append(
                (
                    score,
                    item[
                        "path"
                    ],
                    item[
                        "columns"
                    ]
                )
            )


metric_candidates = sorted(
    metric_candidates,
    key=lambda x: (
        -x[0],
        str(
            x[1]
        )
    )
)


if not metric_candidates:

    print(
        "\nNo obvious original38 scene-metric CSV was found."
    )

    print(
        "CSV discovery results:"
    )


    for root in metric_roots:

        print(
            f"\nROOT: {root}"
        )

        for item in inspect_csvs(
            root,
            EXPECTED_OLD
        ):

            print(
                item[
                    "path"
                ],
                item[
                    "columns"
                ]
            )


    raise RuntimeError(
        "Could not automatically identify "
        "the original38 scene-level inference metrics."
    )


OLD_MODEL_METRICS_PATH = (
    metric_candidates[0][1]
)


old_model = pd.read_csv(
    OLD_MODEL_METRICS_PATH
)


print(
    f"\nOriginal38 scene metrics discovered:\n"
    f"{OLD_MODEL_METRICS_PATH}"
)


print(
    "Columns:",
    list(
        old_model.columns
    )
)


# ======================================================================================
# 11. STANDARDISE ORIGINAL38 IDENTIFIERS
# ======================================================================================

# Some previous stages may use expanded_id or export_id rather than external_id.
# record_id is authoritative whenever available.

if "record_id" not in old_arrays.columns:

    raise RuntimeError(
        "Original38 array manifest must contain record_id."
    )


if "record_id" not in old_model.columns:

    # attempt merge through a common scene identifier
    possible_keys = [
        "expanded_id",
        "export_id",
        "scene_id",
        "record_id",
    ]


    common = [
        c for c in possible_keys
        if (
            c in old_model.columns
            and
            c in old_arrays.columns
        )
    ]


    if not common:

        raise RuntimeError(
            "Cannot map original38 metrics to arrays: "
            "no common scene identifier."
        )


    key = common[0]


    old_model = old_model.merge(
        old_arrays[
            [
                key,
                "record_id"
            ]
        ],
        on=key,
        how="left",
        validate="one_to_one"
    )


if old_model[
    "record_id"
].isna().any():

    raise RuntimeError(
        "Some original38 metric rows could not be mapped "
        "to record_id."
    )


if old_model[
    "record_id"
].duplicated().any():

    raise RuntimeError(
        "Duplicate original38 model metric record IDs."
    )


# ======================================================================================
# 12. MAP CITY/SENSOR INTO ORIGINAL38 METRICS IF NEEDED
# ======================================================================================

meta_cols = [
    "record_id"
]


for c in [
    "city",
    "thermal_sensor",
    "thermal_datetime_utc",
]:

    if c in old_arrays.columns:

        meta_cols.append(
            c
        )


old_meta = (
    old_arrays[
        meta_cols
    ]
    .drop_duplicates(
        subset=[
            "record_id"
        ]
    )
)


for c in [
    "city",
    "thermal_sensor",
    "thermal_datetime_utc",
]:

    if c not in old_model.columns:

        if c in old_meta.columns:

            old_model = old_model.merge(
                old_meta[
                    [
                        "record_id",
                        c
                    ]
                ],
                on="record_id",
                how="left",
                validate="one_to_one"
            )


required_meta = [
    "city",
    "thermal_sensor"
]


if old_model[
    required_meta
].isna().any().any():

    raise RuntimeError(
        "Original38 city/sensor metadata incomplete."
    )


# ======================================================================================
# 13. STANDARDISE ORIGINAL MODEL METRIC COLUMN NAMES
# ======================================================================================

def first_existing(
    frame,
    candidates,
    required=True
):

    for c in candidates:

        if c in frame.columns:

            return c


    if required:

        raise RuntimeError(
            "None of these columns were found: "
            + ", ".join(
                candidates
            )
        )


    return None


old_rmse_col = first_existing(
    old_model,
    [
        "rmse_c",
        "model_rmse_c",
        "scene_rmse_c",
    ]
)


old_mae_col = first_existing(
    old_model,
    [
        "mae_c",
        "model_mae_c",
        "scene_mae_c",
    ]
)


old_bias_col = first_existing(
    old_model,
    [
        "bias_c",
        "model_bias_c",
        "scene_bias_c",
    ]
)


old_coverage_col = first_existing(
    old_model,
    [
        "interval_coverage",
        "coverage",
        "observed_coverage",
    ],
    required=False
)


old_width_col = first_existing(
    old_model,
    [
        "mean_interval_width_c",
        "interval_width_c",
    ],
    required=False
)


old_tta_col = first_existing(
    old_model,
    [
        "median_tta_std_c",
        "mean_tta_std_c",
        "tta_std_c",
    ],
    required=False
)


# ======================================================================================
# 14. ORIGINAL38 ARRAY LOOKUP
# ======================================================================================

old_array_lookup = (
    old_arrays
    .set_index(
        "record_id"
    )
)


missing_array_paths = []


for record_id in old_model[
    "record_id"
]:

    if record_id not in old_array_lookup.index:

        missing_array_paths.append(
            record_id
        )


if missing_array_paths:

    raise RuntimeError(
        f"{len(missing_array_paths)} original38 metric rows "
        f"lack model-ready arrays."
    )


# ======================================================================================
# 15. CONSTANT BASELINE METRICS
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
        float(
            constant
        ),
        dtype="float64"
    )


    error = (
        predicted
        - observed
    )


    return {
        "mae_c":
            float(
                np.mean(
                    np.abs(
                        error
                    )
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
                np.mean(
                    error
                )
            ),
    }


# ======================================================================================
# 16. RECONSTRUCT ORIGINAL38 BASELINES EXACTLY
# ======================================================================================

old_rows = []


for number, model_row in enumerate(
    old_model.sort_values(
        "record_id"
    ).itertuples(
        index=False
    ),
    start=1
):

    record_id = (
        model_row.record_id
    )


    array_row = (
        old_array_lookup.loc[
            record_id
        ]
    )


    array_path = Path(
        array_row[
            "array_path"
        ]
    )


    if not array_path.exists():

        raise FileNotFoundError(
            f"Original38 array missing:\n"
            f"{array_path}"
        )


    with np.load(
        array_path,
        allow_pickle=False
    ) as item:

        y = (
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
            .astype(
                bool
            )
        )


    observed = (
        y[
            valid
        ]
    )


    if observed.size == 0:

        raise RuntimeError(
            f"{record_id}: no valid target pixels."
        )


    city = str(
        model_row.city
    )

    sensor = str(
        model_row.thermal_sensor
    )


    if sensor not in sensor_means:

        raise RuntimeError(
            f"No sensor climatology for {sensor}."
        )


    if (
        city,
        sensor
    ) not in city_sensor_means:

        raise RuntimeError(
            f"No city-sensor climatology for "
            f"{city} {sensor}."
        )


    global_result = constant_metrics(
        observed,
        GLOBAL_MEAN
    )


    sensor_result = constant_metrics(
        observed,
        sensor_means[
            sensor
        ]
    )


    city_sensor_result = constant_metrics(
        observed,
        city_sensor_means[
            (
                city,
                sensor
            )
        ]
    )


    row_dict = {
        "cohort":
            "original38",

        "record_id":
            record_id,

        "city":
            city,

        "thermal_sensor":
            sensor,

        "model_mae_c":
            float(
                getattr(
                    model_row,
                    old_mae_col
                )
            ),

        "model_rmse_c":
            float(
                getattr(
                    model_row,
                    old_rmse_col
                )
            ),

        "model_bias_c":
            float(
                getattr(
                    model_row,
                    old_bias_col
                )
            ),

        "global_training_climatology_mae_c":
            global_result[
                "mae_c"
            ],

        "global_training_climatology_rmse_c":
            global_result[
                "rmse_c"
            ],

        "global_training_climatology_bias_c":
            global_result[
                "bias_c"
            ],

        "sensor_training_climatology_mae_c":
            sensor_result[
                "mae_c"
            ],

        "sensor_training_climatology_rmse_c":
            sensor_result[
                "rmse_c"
            ],

        "sensor_training_climatology_bias_c":
            sensor_result[
                "bias_c"
            ],

        "city_sensor_training_climatology_mae_c":
            city_sensor_result[
                "mae_c"
            ],

        "city_sensor_training_climatology_rmse_c":
            city_sensor_result[
                "rmse_c"
            ],

        "city_sensor_training_climatology_bias_c":
            city_sensor_result[
                "bias_c"
            ],
    }


    if old_coverage_col:

        row_dict[
            "interval_coverage"
        ] = float(
            getattr(
                model_row,
                old_coverage_col
            )
        )


    else:

        row_dict[
            "interval_coverage"
        ] = np.nan


    if old_width_col:

        row_dict[
            "mean_interval_width_c"
        ] = float(
            getattr(
                model_row,
                old_width_col
            )
        )


    else:

        row_dict[
            "mean_interval_width_c"
        ] = np.nan


    if old_tta_col:

        row_dict[
            "tta_uncertainty_c"
        ] = float(
            getattr(
                model_row,
                old_tta_col
            )
        )


    else:

        row_dict[
            "tta_uncertainty_c"
        ] = np.nan


    old_rows.append(
        row_dict
    )


    print(
        f"Original38 reconstruction {number:02d}/38: "
        f"{record_id} | "
        f"{city} | "
        f"{sensor}"
    )


old_paired = pd.DataFrame(
    old_rows
)


if len(old_paired) != 38:

    raise RuntimeError(
        "Original38 reconstructed table is not 38 rows."
    )


# ======================================================================================
# 17. CHECK ORIGINAL38 RECONSTRUCTION AGAINST VERIFIED STAGE27 AGGREGATES
#
# Expected approximately:
#   Model MAE  4.2731
#   Model RMSE 4.6225
#   Model bias -1.5341
#   City-sensor climatology RMSE 5.0291
# ======================================================================================

old_checks = {
    "model_mae":
        old_paired[
            "model_mae_c"
        ].mean(),

    "model_rmse":
        old_paired[
            "model_rmse_c"
        ].mean(),

    "model_bias":
        old_paired[
            "model_bias_c"
        ].mean(),

    "global_rmse":
        old_paired[
            "global_training_climatology_rmse_c"
        ].mean(),

    "sensor_rmse":
        old_paired[
            "sensor_training_climatology_rmse_c"
        ].mean(),

    "city_sensor_rmse":
        old_paired[
            "city_sensor_training_climatology_rmse_c"
        ].mean(),

    "coverage":
        old_paired[
            "interval_coverage"
        ].mean(),
}


print(
    "\nORIGINAL38 RECONSTRUCTION CHECK"
)

print(
    "-" * 120
)


for key, value in old_checks.items():

    print(
        f"{key:25s}: {value:.6f}"
    )


# Tight enough to verify we recovered the correct original table.

reference_checks = [
    abs(
        old_checks[
            "model_mae"
        ]
        - 4.273103
    ) < 0.01,

    abs(
        old_checks[
            "model_rmse"
        ]
        - 4.622534
    ) < 0.01,

    abs(
        old_checks[
            "model_bias"
        ]
        - (
            -1.534078
        )
    ) < 0.01,

    abs(
        old_checks[
            "city_sensor_rmse"
        ]
        - 5.029103
    ) < 0.01,
]


if not all(
    reference_checks
):

    raise RuntimeError(
        "Original38 reconstruction does not reproduce "
        "verified Stage27 aggregate results. "
        "Do not pool cohorts."
    )


print(
    "\nPASS — original38 reconstruction reproduces Stage27."
)


OLD_RECONSTRUCTED_PATH = (
    OUT
    / "02_original38_reconstructed_scene_evidence.csv"
)


old_paired.to_csv(
    OLD_RECONSTRUCTED_PATH,
    index=False
)


# ======================================================================================
# 18. LOAD NEW52 PAIRED TABLE
# ======================================================================================

new_paired_raw = pd.read_csv(
    NEW_PAIRED
)


if len(new_paired_raw) != 52:

    raise RuntimeError(
        "New52 paired table is not 52 rows."
    )


new_rows = []


for row in new_paired_raw.itertuples(
    index=False
):

    new_rows.append(
        {
            "cohort":
                "new52",

            "record_id":
                row.record_id,

            "city":
                row.city,

            "thermal_sensor":
                row.thermal_sensor,

            "model_mae_c":
                float(
                    row.model_mae_c
                ),

            "model_rmse_c":
                float(
                    row.model_rmse_c
                ),

            "model_bias_c":
                float(
                    row.model_bias_c
                ),

            "global_training_climatology_mae_c":
                float(
                    row.global_training_climatology_mae_c
                ),

            "global_training_climatology_rmse_c":
                float(
                    row.global_training_climatology_rmse_c
                ),

            "global_training_climatology_bias_c":
                float(
                    row.global_training_climatology_bias_c
                ),

            "sensor_training_climatology_mae_c":
                float(
                    row.sensor_training_climatology_mae_c
                ),

            "sensor_training_climatology_rmse_c":
                float(
                    row.sensor_training_climatology_rmse_c
                ),

            "sensor_training_climatology_bias_c":
                float(
                    row.sensor_training_climatology_bias_c
                ),

            "city_sensor_training_climatology_mae_c":
                float(
                    row.city_sensor_training_climatology_mae_c
                ),

            "city_sensor_training_climatology_rmse_c":
                float(
                    row.city_sensor_training_climatology_rmse_c
                ),

            "city_sensor_training_climatology_bias_c":
                float(
                    row.city_sensor_training_climatology_bias_c
                ),

            "interval_coverage":
                float(
                    row.interval_coverage
                ),

            "mean_interval_width_c":
                float(
                    row.mean_interval_width_c
                ),

            "tta_uncertainty_c":
                float(
                    row.median_tta_std_c
                ),
        }
    )


new_standardized = pd.DataFrame(
    new_rows
)


# ======================================================================================
# 19. COMBINE 38 + 52
# ======================================================================================

combined = pd.concat(
    [
        old_paired,
        new_standardized
    ],
    ignore_index=True
)


if len(combined) != EXPECTED_COMBINED:

    raise RuntimeError(
        f"Expected 90 combined scenes, found {len(combined)}."
    )


if combined[
    "record_id"
].duplicated().any():

    duplicates = combined.loc[
        combined[
            "record_id"
        ].duplicated(
            keep=False
        ),
        [
            "cohort",
            "record_id"
        ]
    ]

    print(
        duplicates
    )

    raise RuntimeError(
        "Duplicate record IDs after cohort combination."
    )


# ======================================================================================
# 20. CREATE PAIRED DIFFERENCES
# ======================================================================================

for baseline in BASELINES:

    baseline_rmse = (
        f"{baseline}_rmse_c"
    )


    combined[
        f"delta_rmse_vs_{baseline}_c"
    ] = (
        combined[
            "model_rmse_c"
        ]
        -
        combined[
            baseline_rmse
        ]
    )


    combined[
        f"advantage_vs_{baseline}_c"
    ] = (
        -combined[
            f"delta_rmse_vs_{baseline}_c"
        ]
    )


COMBINED_PATH = (
    OUT
    / "03_integrated90_scene_evidence.csv"
)


combined.to_csv(
    COMBINED_PATH,
    index=False
)


# ======================================================================================
# 21. COHORT-SPECIFIC SUMMARY
# ======================================================================================

cohort_rows = []


for cohort, group in combined.groupby(
    "cohort",
    observed=True
):

    cohort_rows.append(
        {
            "cohort":
                cohort,

            "scenes":
                len(group),

            "model_mae_c":
                group[
                    "model_mae_c"
                ].mean(),

            "model_rmse_c":
                group[
                    "model_rmse_c"
                ].mean(),

            "model_bias_c":
                group[
                    "model_bias_c"
                ].mean(),

            "global_climatology_rmse_c":
                group[
                    "global_training_climatology_rmse_c"
                ].mean(),

            "sensor_climatology_rmse_c":
                group[
                    "sensor_training_climatology_rmse_c"
                ].mean(),

            "city_sensor_climatology_rmse_c":
                group[
                    "city_sensor_training_climatology_rmse_c"
                ].mean(),

            "mean_interval_coverage":
                group[
                    "interval_coverage"
                ].mean(),
        }
    )


cohort_summary = pd.DataFrame(
    cohort_rows
)


COHORT_SUMMARY_PATH = (
    OUT
    / "04_cohort_specific_summary.csv"
)


cohort_summary.to_csv(
    COHORT_SUMMARY_PATH,
    index=False
)


# ======================================================================================
# 22. POOLED 90-SCENE SUMMARY
# ======================================================================================

pooled_summary = pd.DataFrame(
    [
        {
            "scenes":
                90,

            "model_scene_macro_mae_c":
                combined[
                    "model_mae_c"
                ].mean(),

            "model_scene_macro_rmse_c":
                combined[
                    "model_rmse_c"
                ].mean(),

            "model_scene_macro_bias_c":
                combined[
                    "model_bias_c"
                ].mean(),

            "global_climatology_rmse_c":
                combined[
                    "global_training_climatology_rmse_c"
                ].mean(),

            "sensor_climatology_rmse_c":
                combined[
                    "sensor_training_climatology_rmse_c"
                ].mean(),

            "city_sensor_climatology_rmse_c":
                combined[
                    "city_sensor_training_climatology_rmse_c"
                ].mean(),

            "mean_scene_interval_coverage":
                combined[
                    "interval_coverage"
                ].mean(),

            "median_scene_interval_coverage":
                combined[
                    "interval_coverage"
                ].median(),

            "mean_interval_width_c":
                combined[
                    "mean_interval_width_c"
                ].mean(),
        }
    ]
)


POOLED_SUMMARY_PATH = (
    OUT
    / "05_integrated90_summary.csv"
)


pooled_summary.to_csv(
    POOLED_SUMMARY_PATH,
    index=False
)


# ======================================================================================
# 23. 50,000-REPLICATE PAIRED SCENE BOOTSTRAP — POOLED 90
# ======================================================================================

rng = np.random.default_rng(
    SEED
)


bootstrap_rows = []


for baseline in BASELINES:

    delta = combined[
        f"delta_rmse_vs_{baseline}_c"
    ].to_numpy(
        dtype="float64"
    )


    n = len(
        delta
    )


    draws = np.empty(
        BOOTSTRAPS,
        dtype="float64"
    )


    chunk = 5000
    start = 0


    while start < BOOTSTRAPS:

        count = min(
            chunk,
            BOOTSTRAPS - start
        )


        indices = rng.integers(
            0,
            n,
            size=(
                count,
                n
            )
        )


        draws[
            start:
            start + count
        ] = (
            delta[
                indices
            ]
            .mean(
                axis=1
            )
        )


        start += count


    bootstrap_rows.append(
        {
            "baseline":
                baseline,

            "scenes":
                n,

            "mean_delta_rmse_c":
                float(
                    delta.mean()
                ),

            "mean_advantage_c":
                float(
                    -delta.mean()
                ),

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

            "thermofusion_scene_wins":
                int(
                    np.sum(
                        delta < 0
                    )
                ),

            "thermofusion_scene_win_fraction":
                float(
                    np.mean(
                        delta < 0
                    )
                ),

            "bootstrap_replicates":
                BOOTSTRAPS,

            "bootstrap_unit":
                "scene",
        }
    )


pooled_bootstrap = pd.DataFrame(
    bootstrap_rows
)


POOLED_BOOTSTRAP_PATH = (
    OUT
    / "06_integrated90_paired_scene_bootstrap.csv"
)


pooled_bootstrap.to_csv(
    POOLED_BOOTSTRAP_PATH,
    index=False
)


# ======================================================================================
# 24. COHORT-SPECIFIC BOOTSTRAP
# ======================================================================================

cohort_boot_rows = []


for cohort, cohort_group in combined.groupby(
    "cohort",
    observed=True
):

    for baseline in BASELINES:

        delta = cohort_group[
            f"delta_rmse_vs_{baseline}_c"
        ].to_numpy(
            dtype="float64"
        )


        local_rng = np.random.default_rng(
            SEED
            + sum(
                map(
                    ord,
                    cohort
                    + baseline
                )
            )
        )


        indices = local_rng.integers(
            0,
            len(delta),
            size=(
                BOOTSTRAPS,
                len(delta)
            )
        )


        draws = (
            delta[
                indices
            ]
            .mean(
                axis=1
            )
        )


        cohort_boot_rows.append(
            {
                "cohort":
                    cohort,

                "baseline":
                    baseline,

                "scenes":
                    len(
                        delta
                    ),

                "mean_delta_rmse_c":
                    float(
                        delta.mean()
                    ),

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

                "scene_win_fraction":
                    float(
                        np.mean(
                            delta < 0
                        )
                    ),
            }
        )


cohort_boot = pd.DataFrame(
    cohort_boot_rows
)


COHORT_BOOT_PATH = (
    OUT
    / "07_cohort_specific_bootstrap.csv"
)


cohort_boot.to_csv(
    COHORT_BOOT_PATH,
    index=False
)


# ======================================================================================
# 25. CITY × SENSOR SUMMARY — ALL 90
# ======================================================================================

subgroup_rows = []


for (
    city,
    sensor
), group in combined.groupby(
    [
        "city",
        "thermal_sensor"
    ],
    observed=True
):

    subgroup_rows.append(
        {
            "city":
                city,

            "thermal_sensor":
                sensor,

            "scenes":
                len(group),

            "model_rmse_c":
                group[
                    "model_rmse_c"
                ].mean(),

            "model_mae_c":
                group[
                    "model_mae_c"
                ].mean(),

            "model_bias_c":
                group[
                    "model_bias_c"
                ].mean(),

            "global_rmse_c":
                group[
                    "global_training_climatology_rmse_c"
                ].mean(),

            "sensor_rmse_c":
                group[
                    "sensor_training_climatology_rmse_c"
                ].mean(),

            "city_sensor_rmse_c":
                group[
                    "city_sensor_training_climatology_rmse_c"
                ].mean(),

            "delta_vs_global_c":
                group[
                    "delta_rmse_vs_global_training_climatology_c"
                ].mean(),

            "delta_vs_sensor_c":
                group[
                    "delta_rmse_vs_sensor_training_climatology_c"
                ].mean(),

            "delta_vs_city_sensor_c":
                group[
                    "delta_rmse_vs_city_sensor_training_climatology_c"
                ].mean(),

            "mean_interval_coverage":
                group[
                    "interval_coverage"
                ].mean(),

            "model_win_fraction_vs_city_sensor":
                (
                    group[
                        "delta_rmse_vs_city_sensor_training_climatology_c"
                    ]
                    < 0
                ).mean(),
        }
    )


subgroups = pd.DataFrame(
    subgroup_rows
)


SUBGROUP_PATH = (
    OUT
    / "08_integrated90_city_sensor_summary.csv"
)


subgroups.to_csv(
    SUBGROUP_PATH,
    index=False
)


# ======================================================================================
# 26. SENSOR SUMMARY
# ======================================================================================

sensor_rows = []


for sensor, group in combined.groupby(
    "thermal_sensor",
    observed=True
):

    sensor_rows.append(
        {
            "thermal_sensor":
                sensor,

            "scenes":
                len(group),

            "model_rmse_c":
                group[
                    "model_rmse_c"
                ].mean(),

            "city_sensor_climatology_rmse_c":
                group[
                    "city_sensor_training_climatology_rmse_c"
                ].mean(),

            "delta_vs_city_sensor_c":
                group[
                    "delta_rmse_vs_city_sensor_training_climatology_c"
                ].mean(),

            "mean_interval_coverage":
                group[
                    "interval_coverage"
                ].mean(),

            "scene_win_fraction_vs_city_sensor":
                (
                    group[
                        "delta_rmse_vs_city_sensor_training_climatology_c"
                    ]
                    < 0
                ).mean(),
        }
    )


sensor_summary = pd.DataFrame(
    sensor_rows
)


SENSOR_PATH = (
    OUT
    / "09_integrated90_sensor_summary.csv"
)


sensor_summary.to_csv(
    SENSOR_PATH,
    index=False
)


# ======================================================================================
# 27. EQUAL OBSERVED CITY-SENSOR STRATUM SENSITIVITY
# ======================================================================================

strata = list(
    combined[
        [
            "city",
            "thermal_sensor"
        ]
    ]
    .drop_duplicates()
    .itertuples(
        index=False,
        name=None
    )
)


equal_rows = []


for baseline in BASELINES:

    delta_col = (
        f"delta_rmse_vs_{baseline}_c"
    )


    observed_stratum_effects = []


    for city, sensor in strata:

        values = combined.loc[
            (
                combined[
                    "city"
                ].eq(
                    city
                )
            )
            &
            (
                combined[
                    "thermal_sensor"
                ].eq(
                    sensor
                )
            ),
            delta_col
        ].to_numpy(
            dtype="float64"
        )


        observed_stratum_effects.append(
            values.mean()
        )


    observed_effect = float(
        np.mean(
            observed_stratum_effects
        )
    )


    local_rng = np.random.default_rng(
        SEED
        + sum(
            map(
                ord,
                baseline
            )
        )
        + 900
    )


    draws = np.empty(
        BOOTSTRAPS,
        dtype="float64"
    )


    for b in range(
        BOOTSTRAPS
    ):

        stratum_means = []


        for city, sensor in strata:

            values = combined.loc[
                (
                    combined[
                        "city"
                    ].eq(
                        city
                    )
                )
                &
                (
                    combined[
                        "thermal_sensor"
                    ].eq(
                        sensor
                    )
                ),
                delta_col
            ].to_numpy(
                dtype="float64"
            )


            sampled = local_rng.choice(
                values,
                size=len(values),
                replace=True
            )


            stratum_means.append(
                sampled.mean()
            )


        draws[
            b
        ] = np.mean(
            stratum_means
        )


    equal_rows.append(
        {
            "baseline":
                baseline,

            "observed_city_sensor_strata":
                len(
                    strata
                ),

            "equal_stratum_mean_delta_rmse_c":
                observed_effect,

            "equal_stratum_advantage_c":
                -observed_effect,

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

            "bootstrap_replicates":
                BOOTSTRAPS,
        }
    )


equal_stratum = pd.DataFrame(
    equal_rows
)


EQUAL_PATH = (
    OUT
    / "10_integrated90_equal_stratum_sensitivity.csv"
)


equal_stratum.to_csv(
    EQUAL_PATH,
    index=False
)


# ======================================================================================
# 28. COHORT × CITY × SENSOR COMPOSITION
# ======================================================================================

composition = (
    combined
    .groupby(
        [
            "cohort",
            "city",
            "thermal_sensor"
        ],
        observed=True
    )
    .size()
    .rename(
        "scenes"
    )
    .reset_index()
)


COMPOSITION_PATH = (
    OUT
    / "11_integrated90_cohort_composition.csv"
)


composition.to_csv(
    COMPOSITION_PATH,
    index=False
)


# ======================================================================================
# 29. FINAL MANUSCRIPT TABLE
# ======================================================================================

manuscript_rows = []


for cohort_name, label in [
    (
        "original38",
        "Original supplementary external cohort"
    ),
    (
        "new52",
        "New independent external cohort"
    ),
]:

    group = combined.loc[
        combined[
            "cohort"
        ].eq(
            cohort_name
        )
    ]


    manuscript_rows.append(
        {
            "Evaluation":
                label,

            "Scenes":
                len(group),

            "ThermoFusion_MAE_C":
                group[
                    "model_mae_c"
                ].mean(),

            "ThermoFusion_RMSE_C":
                group[
                    "model_rmse_c"
                ].mean(),

            "Global_climatology_RMSE_C":
                group[
                    "global_training_climatology_rmse_c"
                ].mean(),

            "Sensor_climatology_RMSE_C":
                group[
                    "sensor_training_climatology_rmse_c"
                ].mean(),

            "City_sensor_climatology_RMSE_C":
                group[
                    "city_sensor_training_climatology_rmse_c"
                ].mean(),

            "Observed_90pct_coverage":
                group[
                    "interval_coverage"
                ].mean(),
        }
    )


manuscript_rows.append(
    {
        "Evaluation":
            "Combined independent external evaluation",

        "Scenes":
            90,

        "ThermoFusion_MAE_C":
            combined[
                "model_mae_c"
            ].mean(),

        "ThermoFusion_RMSE_C":
            combined[
                "model_rmse_c"
            ].mean(),

        "Global_climatology_RMSE_C":
            combined[
                "global_training_climatology_rmse_c"
            ].mean(),

        "Sensor_climatology_RMSE_C":
            combined[
                "sensor_training_climatology_rmse_c"
            ].mean(),

        "City_sensor_climatology_RMSE_C":
            combined[
                "city_sensor_training_climatology_rmse_c"
            ].mean(),

        "Observed_90pct_coverage":
            combined[
                "interval_coverage"
            ].mean(),
    }
)


manuscript_table = pd.DataFrame(
    manuscript_rows
)


MANUSCRIPT_TABLE_PATH = (
    OUT
    / "12_manuscript_ready_external_evaluation_table.csv"
)


manuscript_table.to_csv(
    MANUSCRIPT_TABLE_PATH,
    index=False
)


# ======================================================================================
# 30. IDENTIFY BEST POOLED CLIMATOLOGY
# ======================================================================================

pooled = pooled_summary.iloc[
    0
]


baseline_rmse_map = {
    "global_training_climatology":
        pooled[
            "global_climatology_rmse_c"
        ],

    "sensor_training_climatology":
        pooled[
            "sensor_climatology_rmse_c"
        ],

    "city_sensor_training_climatology":
        pooled[
            "city_sensor_climatology_rmse_c"
        ],
}


best_baseline = min(
    baseline_rmse_map,
    key=
        baseline_rmse_map.get
)


best_boot = pooled_bootstrap.loc[
    pooled_bootstrap[
        "baseline"
    ].eq(
        best_baseline
    )
].iloc[0]


# ======================================================================================
# 31. FINAL INTEGRITY
# ======================================================================================

finite_columns = [
    "model_mae_c",
    "model_rmse_c",
    "model_bias_c",
    "global_training_climatology_rmse_c",
    "sensor_training_climatology_rmse_c",
    "city_sensor_training_climatology_rmse_c",
]


overall_pass = (
    len(
        combined
    )
    == 90

    and

    combined[
        "record_id"
    ].nunique()
    == 90

    and

    len(
        record_overlap
    )
    == 0

    and

    len(
        thermal_overlap
    )
    == 0

    and

    np.isfinite(
        combined[
            finite_columns
        ].to_numpy()
    ).all()

    and

    len(
        pooled_bootstrap
    )
    == 3

    and

    all(
        reference_checks
    )
)


# ======================================================================================
# 32. SAVE CONFIGURATION
# ======================================================================================

configuration = {
    "original_external_scenes":
        38,

    "new_external_scenes":
        52,

    "combined_external_scenes":
        90,

    "record_id_overlap":
        len(
            record_overlap
        ),

    "thermal_scene_id_overlap":
        len(
            thermal_overlap
        ),

    "bootstrap_replicates":
        BOOTSTRAPS,

    "bootstrap_unit":
        "scene",

    "performance_based_exclusion":
        False,

    "model_refitting":
        False,

    "normalization_refitting":
        False,

    "climatology_refitting":
        False,

    "conformal_recalibration":
        False,

    "original38_model_metrics_source":
        str(
            OLD_MODEL_METRICS_PATH
        ),

    "original38_array_manifest_source":
        str(
            OLD_ARRAY_MANIFEST_PATH
        ),
}


CONFIG_PATH = (
    OUT
    / "13_stage08_configuration.json"
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
# 33. VERDICT
# ======================================================================================

verdict = [

    "THERMOFUSION STAGE 08 — INTEGRATED EXTERNAL TEMPORAL EVALUATION",

    "Original independent external cohort: 38/38",

    "New independent external cohort: 52/52",

    "Combined external scenes: 90",

    (
        f"record_id overlap between cohorts: "
        f"{len(record_overlap)}"
    ),

    (
        f"thermal_scene_id overlap between cohorts: "
        f"{len(thermal_overlap)}"
    ),

    "Performance-based exclusions: 0",

    "Model fitting on external cohorts: NO",

    "Normalization refitting: NO",

    "Climatology refitting: NO",

    "Conformal recalibration: NO",

    (
        f"Combined ThermoFusion scene-macro MAE (°C): "
        f"{pooled.model_scene_macro_mae_c:.4f}"
    ),

    (
        f"Combined ThermoFusion scene-macro RMSE (°C): "
        f"{pooled.model_scene_macro_rmse_c:.4f}"
    ),

    (
        f"Combined global climatology RMSE (°C): "
        f"{pooled.global_climatology_rmse_c:.4f}"
    ),

    (
        f"Combined sensor climatology RMSE (°C): "
        f"{pooled.sensor_climatology_rmse_c:.4f}"
    ),

    (
        f"Combined city-sensor climatology RMSE (°C): "
        f"{pooled.city_sensor_climatology_rmse_c:.4f}"
    ),

    (
        f"Combined observed 90% interval coverage: "
        f"{pooled.mean_scene_interval_coverage:.4f}"
    ),

    (
        f"Best pooled climatology baseline: "
        f"{best_baseline}"
    ),

    (
        f"ThermoFusion minus best baseline ΔRMSE (°C): "
        f"{best_boot.mean_delta_rmse_c:.4f}"
    ),

    (
        f"95% pooled scene-bootstrap CI (°C): "
        f"[{best_boot.ci_lower_95_delta_c:.4f}, "
        f"{best_boot.ci_upper_95_delta_c:.4f}]"
    ),

    (
        f"Bootstrap probability ThermoFusion better: "
        f"{best_boot.probability_thermofusion_better:.4f}"
    ),

    (
        f"ThermoFusion scene wins vs best baseline: "
        f"{int(best_boot.thermofusion_scene_wins)}/90"
    ),

    (
        "Overall Stage 08 integrated external evaluation: PASS"
        if overall_pass
        else
        "Overall Stage 08 integrated external evaluation: REVIEW REQUIRED"
    ),
]


VERDICT_PATH = (
    OUT
    / "14_stage08_integrated_external_verdict.txt"
)


VERDICT_PATH.write_text(
    "\n".join(
        verdict
    ),
    encoding="utf-8"
)


# ======================================================================================
# 34. REPORT
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
    "\nCOHORT-SPECIFIC RESULTS"
)

print(
    "-" * 120
)

print(
    cohort_summary.to_string(
        index=False
    )
)


print(
    "\nPOOLED 90-SCENE BOOTSTRAP"
)

print(
    "-" * 120
)

print(
    pooled_bootstrap.to_string(
        index=False
    )
)


print(
    "\nEQUAL CITY-SENSOR STRATUM SENSITIVITY"
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
    "\nCITY × SENSOR SUMMARY"
)

print(
    "-" * 120
)

print(
    subgroups.to_string(
        index=False
    )
)


print(
    "\nCOHORT COMPOSITION"
)

print(
    "-" * 120
)

print(
    composition.to_string(
        index=False
    )
)


print(
    "\nMANUSCRIPT-READY EXTERNAL TABLE"
)

print(
    "-" * 120
)

print(
    manuscript_table.to_string(
        index=False
    )
)


print(
    "\nSaved:"
)

for path in [
    OVERLAP_PATH,
    OLD_RECONSTRUCTED_PATH,
    COMBINED_PATH,
    COHORT_SUMMARY_PATH,
    POOLED_SUMMARY_PATH,
    POOLED_BOOTSTRAP_PATH,
    COHORT_BOOT_PATH,
    SUBGROUP_PATH,
    SENSOR_PATH,
    EQUAL_PATH,
    COMPOSITION_PATH,
    MANUSCRIPT_TABLE_PATH,
    CONFIG_PATH,
    VERDICT_PATH,
]:

    print(
        path
    )


if overall_pass:

    print(
        "\nPASS — Stage 08 integrated external evaluation is complete."
    )

    print(
        "The manuscript can now be updated using the final "
        "90-scene independent external evidence."
    )

else:

    print(
        "\nSTOP — Stage 08 requires review."
    )

    print(
        "Do not update the manuscript until the issue is resolved."
    )
