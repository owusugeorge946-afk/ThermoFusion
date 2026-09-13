# =====================================================================
# ThermoFusion reproducibility code
# Source notebook cell(s): 69
# Audit status: KEEP
# Extracted without scientific-logic modification.
# Replace local Google Drive paths as described in README.md when needed.
# =====================================================================

# ======================================================================================
# THERMOFUSION — LARGE EXTERNAL TEMPORAL EVALUATION
# STAGE 06: FROZEN FINAL-MODEL INFERENCE ON EXTERNAL52
#
# STRICT RULES:
#   - NO model fitting
#   - NO checkpoint selection
#   - NO normalization refitting
#   - NO climatology refitting
#   - NO conformal recalibration
#   - Exact Stage 17 final without-terrain model
#   - Exact eight-view dihedral TTA
#   - Exact frozen validation-only conformal multiplier
# ======================================================================================

from google.colab import drive
drive.mount("/content/drive")

from pathlib import Path
import json
import random
import shutil
import subprocess
import sys
import gc

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
except ImportError:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "torch"]
    )
    import torch
    import torch.nn as nn

try:
    from scipy.stats import spearmanr
except ImportError:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "scipy"]
    )
    from scipy.stats import spearmanr


# ======================================================================================
# 1. PATHS
# ======================================================================================

ROOT = Path("/content/drive/MyDrive")

# ----------------------------------------------------------------------
# NEW EXTERNAL52 MODEL-READY DATA
# ----------------------------------------------------------------------

S05 = (
    ROOT
    / "ThermoFusion_External_Evaluation"
    / "Stage05_ModelReady_External52"
)

ARRAY_MANIFEST_PATH = (
    S05
    / "01_external52_model_ready_manifest.csv"
)

STAGE05_VERDICT = (
    S05
    / "07_stage05_model_ready_verdict.txt"
)

# ----------------------------------------------------------------------
# ORIGINAL FROZEN THERMOFUSION COMPONENTS
# ----------------------------------------------------------------------

S14 = (
    ROOT
    / "ThermoFusion_Stage14_Refined_Benchmark"
)

S16 = (
    ROOT
    / "ThermoFusion_Stage16_Modality_Ablation"
)

S17 = (
    ROOT
    / "ThermoFusion_Stage17_Final_Uncertainty"
)

TRAINING_STATS_PATH = (
    S14
    / "01_training_only_target_statistics.csv"
)

STAGE16_VERDICT = (
    S16
    / "05_stage16_verdict.txt"
)

CHECKPOINT_PATH = (
    S16
    / "checkpoints"
    / "without_terrain_best.pt"
)

STAGE17_CONFIG_PATH = (
    S17
    / "05_configuration.json"
)

STAGE17_VERDICT = (
    S17
    / "04_stage17_verdict.txt"
)

# ----------------------------------------------------------------------
# NEW OUTPUT
# ----------------------------------------------------------------------

OUT = (
    ROOT
    / "ThermoFusion_External_Evaluation"
    / "Stage06_FrozenInference_External52"
)

PRED_DIR = (
    OUT
    / "predictions"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)

PRED_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ======================================================================================
# 2. CONSTANTS — EXACT FINAL CONFIGURATION
# ======================================================================================

SEED = 20260908

EXPECTED_SCENES = 52

BASE = 24

CITIES = [
    "Abidjan",
    "Accra",
    "Freetown",
    "Lagos"
]

SENSORS = [
    "ECOSTRESS",
    "Landsat"
]

EXPECTED_INPUT_CHANNELS = 25

EXPECTED_RASTER_CHANNELS = 16

TARGET_SIZE = 256

MAX_DIAGNOSTIC_PIXELS_PER_SCENE = 20000


# ======================================================================================
# 3. DETERMINISM
# ======================================================================================

def set_determinism(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_determinism(SEED)


# ======================================================================================
# 4. REQUIRED INPUT CHECKS
# ======================================================================================

required = [
    ARRAY_MANIFEST_PATH,
    STAGE05_VERDICT,
    TRAINING_STATS_PATH,
    STAGE16_VERDICT,
    CHECKPOINT_PATH,
    STAGE17_CONFIG_PATH,
    STAGE17_VERDICT,
]

for path in required:

    if not path.exists():

        raise FileNotFoundError(
            f"Required input not found:\n{path}"
        )


if (
    "Overall Stage 05 model-ready verdict: PASS"
    not in
    STAGE05_VERDICT.read_text(
        encoding="utf-8"
    )
):

    raise RuntimeError(
        "Stage 05 external52 model-ready "
        "dataset did not pass."
    )


if (
    "Overall Stage 16 execution: PASS"
    not in
    STAGE16_VERDICT.read_text(
        encoding="utf-8"
    )
):

    raise RuntimeError(
        "Original Stage 16 modality-ablation "
        "experiment did not pass."
    )


if (
    "Overall Stage 17 execution: PASS"
    not in
    STAGE17_VERDICT.read_text(
        encoding="utf-8"
    )
):

    raise RuntimeError(
        "Original Stage 17 final-model "
        "execution did not pass."
    )


# ======================================================================================
# 5. GPU
# ======================================================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

if device.type != "cuda":

    raise RuntimeError(
        "GPU required. In Colab select:\n"
        "Runtime > Change runtime type > T4 GPU."
    )

print(
    f"Compute device: {device}"
)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ======================================================================================
# 6. LOAD EXTERNAL52 MANIFEST
# ======================================================================================

scenes = pd.read_csv(
    ARRAY_MANIFEST_PATH
)

if len(scenes) != EXPECTED_SCENES:

    raise RuntimeError(
        f"Expected 52 model-ready external scenes, "
        f"found {len(scenes)}."
    )


if scenes[
    "record_id"
].duplicated().any():

    raise RuntimeError(
        "Duplicate record IDs in external52 manifest."
    )


if scenes[
    "export_id"
].duplicated().any():

    raise RuntimeError(
        "Duplicate export IDs in external52 manifest."
    )


scenes[
    "thermal_datetime_utc"
] = pd.to_datetime(
    scenes[
        "thermal_datetime_utc"
    ],
    utc=True,
    errors="raise"
)


if scenes[
    "thermal_datetime_utc"
].isna().any():

    raise RuntimeError(
        "At least one external scene "
        "has a missing timestamp."
    )


for path_string in scenes[
    "array_path"
]:

    if not Path(
        path_string
    ).exists():

        raise FileNotFoundError(
            f"Missing model-ready array:\n"
            f"{path_string}"
        )


print(
    "=" * 120
)

print(
    "THERMOFUSION STAGE 06 — "
    "FROZEN FINAL-MODEL INFERENCE ON EXTERNAL52"
)

print(
    "=" * 120
)

print(
    f"External scenes: {len(scenes)}"
)


# ======================================================================================
# 7. LOAD FROZEN CITY-SENSOR TRAINING STATISTICS
# ======================================================================================

stats_table = pd.read_csv(
    TRAINING_STATS_PATH
)

group_stats = {}


for row in stats_table.query(
    "level == 'GROUP'"
).itertuples(
    index=False
):

    city, sensor = (
        row.group.split("|")
    )

    group_stats[
        (
            city,
            sensor
        )
    ] = {
        "mean_c":
            float(
                row.mean_c
            ),

        "std_c":
            float(
                row.std_c
            ),
    }


if len(
    group_stats
) != 8:

    print(
        group_stats
    )

    raise RuntimeError(
        "Expected exactly eight frozen "
        "city-sensor training climatologies."
    )


expected_groups = {
    (
        city,
        sensor
    )
    for city in CITIES
    for sensor in SENSORS
}


if (
    set(
        group_stats.keys()
    )
    != expected_groups
):

    print(
        "Observed groups:",
        group_stats.keys()
    )

    raise RuntimeError(
        "Frozen city-sensor climatology "
        "groups do not match expected contract."
    )


print(
    "Frozen city-sensor climatologies: 8/8"
)

print(
    "Climatology refitting on external52: NO"
)


# ======================================================================================
# 8. LOAD FROZEN STAGE 17 CALIBRATION
# ======================================================================================

with STAGE17_CONFIG_PATH.open(
    "r",
    encoding="utf-8"
) as handle:

    stage17_config = json.load(
        handle
    )


selected_variant = (
    stage17_config[
        "selected_variant"
    ]
)

if selected_variant != "without_terrain":

    raise RuntimeError(
        f"Unexpected Stage 17 variant: "
        f"{selected_variant}"
    )


TTA_TRANSFORMS = int(
    stage17_config[
        "tta_transforms"
    ]
)

TARGET_COVERAGE = float(
    stage17_config[
        "target_coverage"
    ]
)

UNCERTAINTY_FLOOR_C = float(
    stage17_config[
        "uncertainty_floor_c"
    ]
)

CONFORMAL_Q = float(
    stage17_config[
        "conformal_multiplier"
    ]
)


if TTA_TRANSFORMS != 8:

    raise RuntimeError(
        f"Expected eight TTA transforms, "
        f"found {TTA_TRANSFORMS}."
    )


if (
    not np.isfinite(
        CONFORMAL_Q
    )
    or
    CONFORMAL_Q <= 0
):

    raise RuntimeError(
        "Frozen conformal multiplier "
        "is invalid."
    )


print(
    f"Selected frozen model: {selected_variant}"
)

print(
    f"TTA transforms: {TTA_TRANSFORMS}"
)

print(
    f"Nominal interval coverage: "
    f"{TARGET_COVERAGE:.0%}"
)

print(
    f"Frozen uncertainty floor: "
    f"{UNCERTAINTY_FLOOR_C:.2f} °C"
)

print(
    f"Frozen conformal multiplier: "
    f"{CONFORMAL_Q:.4f}"
)

print(
    "Conformal recalibration on external52: NO"
)


# ======================================================================================
# 9. CONTEXT VECTOR
#
# EXACT ORIGINAL ORDER:
#   0  ECOSTRESS indicator
#   1  DOY sin
#   2  DOY cos
#   3  UTC-hour sin
#   4  UTC-hour cos
#   5  Abidjan
#   6  Accra
#   7  Freetown
#   8  Lagos
# ======================================================================================

def context(row):

    stamp = row.thermal_datetime_utc

    doy_angle = (
        2
        * np.pi
        * (
            stamp.dayofyear
            - 1
        )
        / 365.25
    )

    decimal_hour = (
        stamp.hour
        + stamp.minute / 60.0
    )

    hour_angle = (
        2
        * np.pi
        * decimal_hour
        / 24.0
    )

    values = [

        float(
            row.thermal_sensor
            == "ECOSTRESS"
        ),

        np.sin(
            doy_angle
        ),

        np.cos(
            doy_angle
        ),

        np.sin(
            hour_angle
        ),

        np.cos(
            hour_angle
        ),
    ]


    values += [

        float(
            row.city == city
        )

        for city in CITIES
    ]


    values = np.asarray(
        values,
        dtype="float32"
    )


    if values.shape != (
        9,
    ):

        raise RuntimeError(
            "Context vector is not length 9."
        )


    return (
        values[
            :,
            None,
            None
        ]
    )


# ======================================================================================
# 10. LOAD ONE EXTERNAL SCENE
# ======================================================================================

def load_scene(row):

    with np.load(
        row.array_path,
        allow_pickle=False
    ) as item:

        x = (
            item[
                "x"
            ]
            .astype(
                "float32"
            )
        )

        observed = (
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


    if x.shape != (
        16,
        256,
        256
    ):

        raise RuntimeError(
            f"{row.export_id}: "
            f"unexpected raster shape "
            f"{x.shape}."
        )


    if observed.shape != (
        256,
        256
    ):

        raise RuntimeError(
            f"{row.export_id}: "
            "unexpected target shape."
        )


    if valid.shape != (
        256,
        256
    ):

        raise RuntimeError(
            f"{row.export_id}: "
            "unexpected mask shape."
        )


    if valid.sum() == 0:

        raise RuntimeError(
            f"{row.export_id}: "
            "no valid target pixels."
        )


    context_grid = np.broadcast_to(
        context(row),
        (
            9,
            256,
            256
        )
    ).copy()


    x = np.concatenate(
        [
            x,
            context_grid
        ],
        axis=0
    )


    # ------------------------------------------------------------------
    # EXACT FINAL WITHOUT-TERRAIN CONFIGURATION
    #
    # Raster indices:
    #   14 = elevation
    #   15 = slope
    # ------------------------------------------------------------------

    x[
        [
            14,
            15
        ]
    ] = 0.0


    if x.shape != (
        EXPECTED_INPUT_CHANNELS,
        TARGET_SIZE,
        TARGET_SIZE
    ):

        raise RuntimeError(
            f"{row.export_id}: "
            f"conditioned input shape "
            f"is {x.shape}, expected "
            f"(25, 256, 256)."
        )


    if not np.isfinite(
        x
    ).all():

        raise RuntimeError(
            f"{row.export_id}: "
            "input contains non-finite values."
        )


    return (
        x,
        observed,
        valid
    )


# ======================================================================================
# 11. EXACT FINAL U-NET ARCHITECTURE
# ======================================================================================

def block(
    inputs,
    outputs
):

    return nn.Sequential(

        nn.Conv2d(
            inputs,
            outputs,
            kernel_size=3,
            padding=1,
            bias=False
        ),

        nn.GroupNorm(
            min(
                8,
                outputs
            ),
            outputs
        ),

        nn.SiLU(
            inplace=True
        ),

        nn.Conv2d(
            outputs,
            outputs,
            kernel_size=3,
            padding=1,
            bias=False
        ),

        nn.GroupNorm(
            min(
                8,
                outputs
            ),
            outputs
        ),

        nn.SiLU(
            inplace=True
        ),
    )


class UNet(
    nn.Module
):

    def __init__(
        self
    ):

        super().__init__()

        b = BASE

        self.p = nn.MaxPool2d(
            2
        )

        self.e1 = block(
            25,
            b
        )

        self.e2 = block(
            b,
            2 * b
        )

        self.e3 = block(
            2 * b,
            4 * b
        )

        self.e4 = block(
            4 * b,
            8 * b
        )

        self.bridge = block(
            8 * b,
            16 * b
        )

        self.u4 = (
            nn.ConvTranspose2d(
                16 * b,
                8 * b,
                2,
                2
            )
        )

        self.d4 = block(
            16 * b,
            8 * b
        )

        self.u3 = (
            nn.ConvTranspose2d(
                8 * b,
                4 * b,
                2,
                2
            )
        )

        self.d3 = block(
            8 * b,
            4 * b
        )

        self.u2 = (
            nn.ConvTranspose2d(
                4 * b,
                2 * b,
                2,
                2
            )
        )

        self.d2 = block(
            4 * b,
            2 * b
        )

        self.u1 = (
            nn.ConvTranspose2d(
                2 * b,
                b,
                2,
                2
            )
        )

        self.d1 = block(
            2 * b,
            b
        )

        self.output = (
            nn.Conv2d(
                b,
                1,
                1
            )
        )


    def forward(
        self,
        x
    ):

        e1 = self.e1(
            x
        )

        e2 = self.e2(
            self.p(
                e1
            )
        )

        e3 = self.e3(
            self.p(
                e2
            )
        )

        e4 = self.e4(
            self.p(
                e3
            )
        )

        z = self.bridge(
            self.p(
                e4
            )
        )

        z = self.d4(
            torch.cat(
                [
                    self.u4(
                        z
                    ),
                    e4
                ],
                dim=1
            )
        )

        z = self.d3(
            torch.cat(
                [
                    self.u3(
                        z
                    ),
                    e3
                ],
                dim=1
            )
        )

        z = self.d2(
            torch.cat(
                [
                    self.u2(
                        z
                    ),
                    e2
                ],
                dim=1
            )
        )

        z = self.d1(
            torch.cat(
                [
                    self.u1(
                        z
                    ),
                    e1
                ],
                dim=1
            )
        )

        return self.output(
            z
        )


# ======================================================================================
# 12. LOAD ORIGINAL FROZEN CHECKPOINT
# ======================================================================================

model = UNet().to(
    device
)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
    weights_only=False
)


if checkpoint.get(
    "variant"
) != "without_terrain":

    raise RuntimeError(
        "Selected checkpoint is not "
        "the original without-terrain model."
    )


model.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)

model.eval()


print(
    "Frozen checkpoint loaded successfully."
)

print(
    "Model fitting on external52: NO"
)

print(
    "Model selection on external52: NO"
)


# ======================================================================================
# 13. EXACT EIGHT DIHEDRAL TRANSFORMS
# ======================================================================================

TRANSFORMS = [

    (
        k,
        flip
    )

    for k in range(
        4
    )

    for flip in [
        False,
        True
    ]
]


if len(
    TRANSFORMS
) != 8:

    raise RuntimeError(
        "Expected exactly eight "
        "TTA transforms."
    )


def transform_array(
    array,
    k,
    flip
):

    result = np.rot90(
        array,
        k,
        axes=(
            -2,
            -1
        )
    )

    if flip:

        result = (
            result[
                ...,
                :,
                ::-1
            ]
        )

    return np.ascontiguousarray(
        result
    )


def inverse_array(
    array,
    k,
    flip
):

    result = (
        array[
            ...,
            :,
            ::-1
        ]
        if flip
        else array
    )

    return np.ascontiguousarray(
        np.rot90(
            result,
            -k,
            axes=(
                -2,
                -1
            )
        )
    )


# ======================================================================================
# 14. FROZEN TTA PREDICTION
# ======================================================================================

def tta_predict(
    row
):

    x, observed, valid = (
        load_scene(
            row
        )
    )


    group_key = (
        row.city,
        row.thermal_sensor
    )


    if (
        group_key
        not in group_stats
    ):

        raise RuntimeError(
            f"No frozen climatology for "
            f"{group_key}."
        )


    gs = group_stats[
        group_key
    ]


    predictions = []


    with torch.no_grad():

        for (
            k,
            flip
        ) in TRANSFORMS:

            augmented = (
                transform_array(
                    x,
                    k,
                    flip
                )
            )


            tensor = (
                torch.from_numpy(
                    augmented[
                        None
                    ]
                )
                .to(
                    device
                )
            )


            residual = (
                model(
                    tensor
                )
                .detach()
                .cpu()
                .numpy()[
                    0,
                    0
                ]
            )


            residual = (
                inverse_array(
                    residual,
                    k,
                    flip
                )
            )


            reconstructed_c = (
                residual
                * gs[
                    "std_c"
                ]
                + gs[
                    "mean_c"
                ]
            )


            predictions.append(
                reconstructed_c
            )


            del tensor


    stack = np.stack(
        predictions
    ).astype(
        "float32"
    )


    prediction_mean = (
        stack.mean(
            axis=0
        )
        .astype(
            "float32"
        )
    )


    prediction_std = (
        stack.std(
            axis=0,
            ddof=1
        )
        .astype(
            "float32"
        )
    )


    return (
        observed,
        valid,
        prediction_mean,
        prediction_std
    )


# ======================================================================================
# 15. METRICS
# ======================================================================================

def metrics(
    observed,
    predicted
):

    observed = (
        observed
        .astype(
            "float64"
        )
    )

    predicted = (
        predicted
        .astype(
            "float64"
        )
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
        - np.sum(
            error ** 2
        )
        / denominator
        if denominator > 0
        else np.nan
    )


    return {

        "pixels":
            int(
                len(
                    observed
                )
            ),

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

        "r2":
            float(
                r2
            ),
    }


# ======================================================================================
# 16. RUN ALL 52 FROZEN INFERENCES
# ======================================================================================

scene_rows = []

diagnostic_frames = []


scenes = (
    scenes
    .sort_values(
        "export_id"
    )
    .reset_index(
        drop=True
    )
)


for number, row in enumerate(
    scenes.itertuples(
        index=False
    ),
    start=1
):

    (
        observed,
        valid,
        prediction_mean,
        prediction_std
    ) = tta_predict(
        row
    )


    # ------------------------------------------------------------------
    # Original frozen conformal interval
    # ------------------------------------------------------------------

    half_width = (
        CONFORMAL_Q
        * (
            prediction_std
            + UNCERTAINTY_FLOOR_C
        )
    )


    lower = (
        prediction_mean
        - half_width
    ).astype(
        "float32"
    )


    upper = (
        prediction_mean
        + half_width
    ).astype(
        "float32"
    )


    y = observed[
        valid
    ]


    prediction = (
        prediction_mean[
            valid
        ]
    )


    uncertainty = (
        prediction_std[
            valid
        ]
    )


    interval_width = (
        2.0
        * half_width[
            valid
        ]
    )


    covered = (
        (
            y
            >= lower[
                valid
            ]
        )
        &
        (
            y
            <= upper[
                valid
            ]
        )
    )


    absolute_error = (
        np.abs(
            prediction
            - y
        )
    )


    if (
        len(
            np.unique(
                uncertainty
            )
        )
        > 1
    ):

        correlation, p_value = (
            spearmanr(
                uncertainty,
                absolute_error
            )
        )

    else:

        correlation = np.nan
        p_value = np.nan


    scene_metric = metrics(
        y,
        prediction
    )


    scene_rows.append(
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

            "thermal_datetime_utc":
                row.thermal_datetime_utc,

            **scene_metric,

            "interval_coverage":
                float(
                    covered.mean()
                ),

            "mean_interval_width_c":
                float(
                    interval_width.mean()
                ),

            "median_interval_width_c":
                float(
                    np.median(
                        interval_width
                    )
                ),

            "mean_tta_std_c":
                float(
                    uncertainty.mean()
                ),

            "median_tta_std_c":
                float(
                    np.median(
                        uncertainty
                    )
                ),

            "uncertainty_error_spearman":
                float(
                    correlation
                )
                if np.isfinite(
                    correlation
                )
                else np.nan,

            "uncertainty_error_p":
                float(
                    p_value
                )
                if np.isfinite(
                    p_value
                )
                else np.nan,

            "target_valid_fraction":
                float(
                    valid.mean()
                ),

            "s2_valid_fraction":
                float(
                    row.s2_valid_fraction
                ),

            "s1_valid_fraction":
                float(
                    row.s1_valid_fraction
                ),
        }
    )


    # ------------------------------------------------------------------
    # Deterministic diagnostic pixel sample
    # ------------------------------------------------------------------

    sample_seed = (
        SEED
        + sum(
            map(
                ord,
                str(
                    row.export_id
                )
            )
        )
    )


    rng = np.random.default_rng(
        sample_seed
    )


    sample_size = min(
        MAX_DIAGNOSTIC_PIXELS_PER_SCENE,
        len(
            y
        )
    )


    choose = rng.choice(
        np.arange(
            len(
                y
            )
        ),
        size=
            sample_size,
        replace=False
    )


    diagnostic_frames.append(
        pd.DataFrame(
            {
                "export_id":
                    row.export_id,

                "record_id":
                    row.record_id,

                "city":
                    row.city,

                "thermal_sensor":
                    row.thermal_sensor,

                "observed_c":
                    y[
                        choose
                    ],

                "predicted_c":
                    prediction[
                        choose
                    ],

                "absolute_error_c":
                    absolute_error[
                        choose
                    ],

                "tta_std_c":
                    uncertainty[
                        choose
                    ],

                "covered":
                    covered[
                        choose
                    ],

                "interval_width_c":
                    interval_width[
                        choose
                    ],
            }
        )
    )


    # ------------------------------------------------------------------
    # Save full prediction
    # ------------------------------------------------------------------

    output_path = (
        PRED_DIR
        / (
            f"{row.export_id}"
            f"_frozen_final.npz"
        )
    )


    np.savez_compressed(
        output_path,

        observed_c=
            observed.astype(
                "float32"
            ),

        predicted_c=
            prediction_mean.astype(
                "float32"
            ),

        tta_std_c=
            prediction_std.astype(
                "float32"
            ),

        interval_lower_c=
            lower,

        interval_upper_c=
            upper,

        valid_mask=
            valid.astype(
                "uint8"
            ),

        export_id=
            np.asarray(
                row.export_id
            ),

        record_id=
            np.asarray(
                row.record_id
            ),

        city=
            np.asarray(
                row.city
            ),

        thermal_sensor=
            np.asarray(
                row.thermal_sensor
            ),
    )


    print(
        f"Inference {number:02d}/52: "
        f"{row.export_id} | "
        f"{row.city} | "
        f"{row.thermal_sensor} | "
        f"MAE={scene_metric['mae_c']:.3f} °C | "
        f"RMSE={scene_metric['rmse_c']:.3f} °C | "
        f"coverage={covered.mean():.3f}"
    )


    # ------------------------------------------------------------------
    # GPU housekeeping
    # ------------------------------------------------------------------

    if (
        number
        % 10
        == 0
    ):

        gc.collect()

        torch.cuda.empty_cache()


# ======================================================================================
# 17. SAVE SCENE-LEVEL RESULTS
# ======================================================================================

scene_metrics = pd.DataFrame(
    scene_rows
)


SCENE_METRICS_PATH = (
    OUT
    / "01_external52_scene_accuracy_uncertainty.csv"
)


scene_metrics.to_csv(
    SCENE_METRICS_PATH,
    index=False
)


# ======================================================================================
# 18. DIAGNOSTIC PIXEL TABLE
# ======================================================================================

diagnostic_pixels = pd.concat(
    diagnostic_frames,
    ignore_index=True
)


PIXEL_PATH = (
    OUT
    / "02_external52_sampled_pixel_diagnostics.csv"
)


diagnostic_pixels.to_csv(
    PIXEL_PATH,
    index=False
)


# ======================================================================================
# 19. SCENE-MACRO SUMMARY
#
# This is the principal external-evaluation summary.
# Every acquisition receives equal weight.
# ======================================================================================

scene_macro = pd.DataFrame(
    [
        {
            "scenes":
                len(
                    scene_metrics
                ),

            "scene_macro_mae_c":
                float(
                    scene_metrics[
                        "mae_c"
                    ].mean()
                ),

            "scene_macro_rmse_c":
                float(
                    scene_metrics[
                        "rmse_c"
                    ].mean()
                ),

            "scene_macro_bias_c":
                float(
                    scene_metrics[
                        "bias_c"
                    ].mean()
                ),

            "scene_macro_r2":
                float(
                    scene_metrics[
                        "r2"
                    ].mean()
                ),

            "mean_scene_interval_coverage":
                float(
                    scene_metrics[
                        "interval_coverage"
                    ].mean()
                ),

            "median_scene_interval_coverage":
                float(
                    scene_metrics[
                        "interval_coverage"
                    ].median()
                ),

            "mean_scene_interval_width_c":
                float(
                    scene_metrics[
                        "mean_interval_width_c"
                    ].mean()
                ),

            "mean_scene_tta_std_c":
                float(
                    scene_metrics[
                        "mean_tta_std_c"
                    ].mean()
                ),
        }
    ]
)


SCENE_MACRO_PATH = (
    OUT
    / "03_external52_scene_macro_summary.csv"
)


scene_macro.to_csv(
    SCENE_MACRO_PATH,
    index=False
)


# ======================================================================================
# 20. PIXEL-SAMPLED AGGREGATE
#
# Equal maximum contribution of 20,000 pixels per scene.
# This is diagnostic only, not the main inferential unit.
# ======================================================================================

sampled_metrics = metrics(
    diagnostic_pixels[
        "observed_c"
    ].to_numpy(),

    diagnostic_pixels[
        "predicted_c"
    ].to_numpy()
)


sampled_rho, sampled_p = (
    spearmanr(
        diagnostic_pixels[
            "tta_std_c"
        ],
        diagnostic_pixels[
            "absolute_error_c"
        ]
    )
)


sampled_summary = pd.DataFrame(
    [
        {
            **sampled_metrics,

            "interval_coverage":
                float(
                    diagnostic_pixels[
                        "covered"
                    ].mean()
                ),

            "mean_interval_width_c":
                float(
                    diagnostic_pixels[
                        "interval_width_c"
                    ].mean()
                ),

            "mean_tta_std_c":
                float(
                    diagnostic_pixels[
                        "tta_std_c"
                    ].mean()
                ),

            "uncertainty_error_spearman":
                float(
                    sampled_rho
                ),

            "uncertainty_error_p":
                float(
                    sampled_p
                ),

            "sampling_note":
                (
                    "Maximum 20,000 valid pixels "
                    "per scene; diagnostic only"
                ),
        }
    ]
)


SAMPLED_SUMMARY_PATH = (
    OUT
    / "04_external52_sampled_pixel_summary.csv"
)


sampled_summary.to_csv(
    SAMPLED_SUMMARY_PATH,
    index=False
)


# ======================================================================================
# 21. CITY × SENSOR SUMMARY
# ======================================================================================

subgroup_rows = []


for (
    city,
    sensor
), group in scene_metrics.groupby(
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
                len(
                    group
                ),

            "scene_macro_mae_c":
                float(
                    group[
                        "mae_c"
                    ].mean()
                ),

            "scene_macro_rmse_c":
                float(
                    group[
                        "rmse_c"
                    ].mean()
                ),

            "scene_macro_bias_c":
                float(
                    group[
                        "bias_c"
                    ].mean()
                ),

            "mean_interval_coverage":
                float(
                    group[
                        "interval_coverage"
                    ].mean()
                ),

            "mean_interval_width_c":
                float(
                    group[
                        "mean_interval_width_c"
                    ].mean()
                ),

            "median_tta_std_c":
                float(
                    group[
                        "median_tta_std_c"
                    ].median()
                ),
        }
    )


subgroups = pd.DataFrame(
    subgroup_rows
)


SUBGROUP_PATH = (
    OUT
    / "05_external52_city_sensor_summary.csv"
)


subgroups.to_csv(
    SUBGROUP_PATH,
    index=False
)


# ======================================================================================
# 22. SENSOR SUMMARY
# ======================================================================================

sensor_rows = []


for sensor, group in scene_metrics.groupby(
    "thermal_sensor",
    observed=True
):

    sensor_rows.append(
        {
            "thermal_sensor":
                sensor,

            "scenes":
                len(
                    group
                ),

            "scene_macro_mae_c":
                float(
                    group[
                        "mae_c"
                    ].mean()
                ),

            "scene_macro_rmse_c":
                float(
                    group[
                        "rmse_c"
                    ].mean()
                ),

            "scene_macro_bias_c":
                float(
                    group[
                        "bias_c"
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


sensor_summary = pd.DataFrame(
    sensor_rows
)


SENSOR_PATH = (
    OUT
    / "06_external52_sensor_summary.csv"
)


sensor_summary.to_csv(
    SENSOR_PATH,
    index=False
)


# ======================================================================================
# 23. UNCERTAINTY RELIABILITY DECILES
# ======================================================================================

diagnostic_pixels[
    "uncertainty_bin"
] = pd.qcut(
    diagnostic_pixels[
        "tta_std_c"
    ],
    10,
    duplicates="drop"
)


reliability = (
    diagnostic_pixels
    .groupby(
        "uncertainty_bin",
        observed=True
    )
    .agg(
        pixels=(
            "absolute_error_c",
            "size"
        ),

        mean_tta_std_c=(
            "tta_std_c",
            "mean"
        ),

        mean_absolute_error_c=(
            "absolute_error_c",
            "mean"
        ),

        coverage=(
            "covered",
            "mean"
        ),

        mean_interval_width_c=(
            "interval_width_c",
            "mean"
        ),
    )
    .reset_index()
)


reliability[
    "uncertainty_bin"
] = reliability[
    "uncertainty_bin"
].astype(
    str
)


RELIABILITY_PATH = (
    OUT
    / "07_external52_uncertainty_reliability.csv"
)


reliability.to_csv(
    RELIABILITY_PATH,
    index=False
)


# ======================================================================================
# 24. PREDICTION FILE INTEGRITY
# ======================================================================================

prediction_files = sorted(
    PRED_DIR.glob(
        "*_frozen_final.npz"
    )
)


integrity_rows = []


for row in scene_metrics.itertuples(
    index=False
):

    path = (
        PRED_DIR
        / (
            f"{row.export_id}"
            f"_frozen_final.npz"
        )
    )


    if not path.exists():

        integrity_rows.append(
            {
                "export_id":
                    row.export_id,

                "exists":
                    False,

                "shape_check":
                    False,

                "finite_check":
                    False,

                "mask_check":
                    False,

                "status":
                    "REVIEW",
            }
        )

        continue


    with np.load(
        path,
        allow_pickle=False
    ) as item:

        predicted = item[
            "predicted_c"
        ]

        tta_std = item[
            "tta_std_c"
        ]

        lower = item[
            "interval_lower_c"
        ]

        upper = item[
            "interval_upper_c"
        ]

        mask = item[
            "valid_mask"
        ]


        shape_check = (
            predicted.shape
            == (
                256,
                256
            )

            and

            tta_std.shape
            == (
                256,
                256
            )

            and

            lower.shape
            == (
                256,
                256
            )

            and

            upper.shape
            == (
                256,
                256
            )

            and

            mask.shape
            == (
                256,
                256
            )
        )


        finite_check = (
            np.isfinite(
                predicted
            ).all()

            and

            np.isfinite(
                tta_std
            ).all()

            and

            np.isfinite(
                lower
            ).all()

            and

            np.isfinite(
                upper
            ).all()
        )


        mask_check = (
            set(
                np.unique(
                    mask
                )
            )
            .issubset(
                {
                    0,
                    1
                }
            )

            and

            mask.sum()
            > 0
        )


        interval_order_check = (
            np.all(
                lower
                <= predicted
            )

            and

            np.all(
                predicted
                <= upper
            )
        )


        std_check = (
            np.all(
                tta_std
                >= 0
            )
        )


        checks = [
            shape_check,
            finite_check,
            mask_check,
            interval_order_check,
            std_check,
        ]


        integrity_rows.append(
            {
                "export_id":
                    row.export_id,

                "exists":
                    True,

                "shape_check":
                    shape_check,

                "finite_check":
                    finite_check,

                "mask_check":
                    mask_check,

                "interval_order_check":
                    interval_order_check,

                "std_nonnegative_check":
                    std_check,

                "status":
                    (
                        "PASS"
                        if all(
                            checks
                        )
                        else "REVIEW"
                    ),
            }
        )


integrity = pd.DataFrame(
    integrity_rows
)


INTEGRITY_PATH = (
    OUT
    / "08_external52_prediction_integrity.csv"
)


integrity.to_csv(
    INTEGRITY_PATH,
    index=False
)


# ======================================================================================
# 25. EXECUTION VERDICT
# ======================================================================================

macro = scene_macro.iloc[
    0
]


execution_pass = (

    len(
        scene_metrics
    )
    == 52

    and

    scene_metrics[
        "record_id"
    ]
    .nunique()
    == 52

    and

    len(
        prediction_files
    )
    == 52

    and

    integrity[
        "status"
    ]
    .eq(
        "PASS"
    )
    .all()

    and

    np.isfinite(
        scene_metrics[
            [
                "mae_c",
                "rmse_c",
                "bias_c",
                "interval_coverage",
                "mean_interval_width_c",
                "median_tta_std_c",
            ]
        ]
        .to_numpy()
    )
    .all()

    and

    scene_metrics[
        "interval_coverage"
    ]
    .between(
        0,
        1
    )
    .all()

    and

    CONFORMAL_Q
    > 0
)


# ======================================================================================
# 26. SAVE CONFIGURATION
# ======================================================================================

configuration = {

    "seed":
        SEED,

    "external_scenes":
        52,

    "selected_variant":
        "without_terrain",

    "checkpoint":
        str(
            CHECKPOINT_PATH
        ),

    "tta_transforms":
        8,

    "target_coverage":
        TARGET_COVERAGE,

    "uncertainty_floor_c":
        UNCERTAINTY_FLOOR_C,

    "conformal_multiplier":
        CONFORMAL_Q,

    "conformal_source":
        (
            "Original Stage 17 "
            "validation-only calibration"
        ),

    "normalization_refitting":
        False,

    "model_fitting":
        False,

    "model_selection":
        False,

    "climatology_refitting":
        False,

    "conformal_recalibration":
        False,

    "evaluation_role":
        (
            "independent frozen-model "
            "external temporal evaluation"
        ),
}


CONFIG_PATH = (
    OUT
    / "09_external52_frozen_configuration.json"
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
# 27. VERDICT
# ======================================================================================

verdict = [

    "THERMOFUSION STAGE 06 — FROZEN FINAL-MODEL EXTERNAL52",

    "Independent external scenes: 52/52",

    "Final configuration: metadata-conditioned residual U-Net without terrain",

    "Checkpoint source: original Stage 16 validation-selected checkpoint",

    "Predictor normalization refitting: NO",

    "Target normalization refitting: NO",

    "City-sensor climatology refitting: NO",

    "Model fitting on external52: NO",

    "Model selection on external52: NO",

    "Conformal recalibration on external52: NO",

    (
        f"Test-time transformations: "
        f"{len(TRANSFORMS)}"
    ),

    (
        f"Nominal prediction-interval coverage: "
        f"{TARGET_COVERAGE:.0%}"
    ),

    (
        f"Frozen validation-only conformal multiplier: "
        f"{CONFORMAL_Q:.4f}"
    ),

    (
        f"Scene-macro MAE (°C): "
        f"{macro.scene_macro_mae_c:.4f}"
    ),

    (
        f"Scene-macro RMSE (°C): "
        f"{macro.scene_macro_rmse_c:.4f}"
    ),

    (
        f"Scene-macro bias (°C): "
        f"{macro.scene_macro_bias_c:.4f}"
    ),

    (
        f"Mean scene interval coverage: "
        f"{macro.mean_scene_interval_coverage:.4f}"
    ),

    (
        f"Prediction files passing integrity checks: "
        f"{integrity['status'].eq('PASS').sum()}/52"
    ),

    (
        "Overall Stage 06 frozen inference verdict: PASS"
        if execution_pass
        else
        "Overall Stage 06 frozen inference verdict: REVIEW REQUIRED"
    ),
]


VERDICT_PATH = (
    OUT
    / "10_stage06_frozen_inference_verdict.txt"
)


VERDICT_PATH.write_text(
    "\n".join(
        verdict
    ),
    encoding="utf-8"
)


# ======================================================================================
# 28. FINAL REPORT
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
    "\nCITY × SENSOR RESULTS"
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
    "\nSENSOR RESULTS"
)

print(
    "-" * 120
)

print(
    sensor_summary.to_string(
        index=False
    )
)


print(
    "\nSaved:"
)

for path in [

    SCENE_METRICS_PATH,
    PIXEL_PATH,
    SCENE_MACRO_PATH,
    SAMPLED_SUMMARY_PATH,
    SUBGROUP_PATH,
    SENSOR_PATH,
    RELIABILITY_PATH,
    INTEGRITY_PATH,
    CONFIG_PATH,
    VERDICT_PATH,

]:

    print(
        path
    )


if execution_pass:

    print(
        "\nPASS — frozen inference completed "
        "for all 52 new external scenes."
    )

    print(
        "NEXT STAGE: compare the frozen model "
        "against the original global, sensor, "
        "and city-sensor climatology baselines "
        "and perform scene-level paired bootstrap inference."
    )

else:

    print(
        "\nSTOP — Stage 06 requires review."
    )

    print(
        "Do not perform baseline comparison "
        "or manuscript updating until resolved."
    )
