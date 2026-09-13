# =====================================================================
# ThermoFusion reproducibility code
# Source notebook cell(s): 68
# Audit status: KEEP
# Extracted without scientific-logic modification.
# Replace local Google Drive paths as described in README.md when needed.
# =====================================================================

# ======================================================================================
# THERMOFUSION — LARGE EXTERNAL TEMPORAL EVALUATION
# STAGE 05: BUILD MODEL-READY ARRAYS FOR 52 NEW EXTERNAL SCENES
#
# IMPORTANT:
#   - NO normalization refitting
#   - NO model fitting
#   - NO model selection
#   - Uses ONLY original Stage 12 training-derived normalization statistics
#   - Preserves exact 16-channel ThermoFusion raster input contract
# ======================================================================================

from google.colab import drive
drive.mount("/content/drive")

from pathlib import Path
import json
import subprocess
import sys

import numpy as np
import pandas as pd

try:
    import rasterio
    from rasterio.warp import reproject, Resampling
except ImportError:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "rasterio"]
    )
    import rasterio
    from rasterio.warp import reproject, Resampling


# ======================================================================================
# 1. PATHS
# ======================================================================================

ROOT = Path("/content/drive/MyDrive")

EXTERNAL_ROOT = (
    ROOT
    / "ThermoFusion_External_Evaluation"
)

EXPORT_DIR = (
    EXTERNAL_ROOT
    / "Stage04_Export"
)

FINAL52_PATH = (
    EXPORT_DIR
    / "02_final_external52_manifest.csv"
)

BATCH_VERIFY_PATHS = [
    EXPORT_DIR / "05_batch01_export_verification.csv",
    EXPORT_DIR / "05_batch02_export_verification.csv",
    EXPORT_DIR / "05_batch03_export_verification.csv",
]

# Authoritative original matching/timestamp inventory
VERIFIED_MATCH_PATH = (
    ROOT
    / "ThermoFusion_Stage06_Match_Verification"
    / "07_verified_match_manifest.csv"
)

# ----------------------------------------------------------------------
# FROZEN ORIGINAL STAGE 12 NORMALIZATION
# ----------------------------------------------------------------------

STAGE12_DIR = (
    ROOT
    / "ThermoFusion_Stage12_ModelReady_Pilot"
)

NORMALIZATION_PATH = (
    STAGE12_DIR
    / "02_training_normalization.csv"
)

TARGET_STATS_PATH = (
    STAGE12_DIR
    / "03_target_normalization.json"
)

STAGE12_VERDICT = (
    STAGE12_DIR
    / "07_model_ready_verdict.txt"
)

# ----------------------------------------------------------------------
# NEW OUTPUT
# ----------------------------------------------------------------------

OUTPUT_DIR = (
    EXTERNAL_ROOT
    / "Stage05_ModelReady_External52"
)

ARRAY_DIR = (
    OUTPUT_DIR
    / "arrays"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

ARRAY_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ======================================================================================
# 2. FIXED CONTRACT
# ======================================================================================

EXPECTED_SCENES = 52
TARGET_SIZE = 256
NODATA = -9999.0
NORMALIZATION_CLIP = 8.0

CONTINUOUS_INDICES = [
    0, 1, 2, 3, 4, 5,
    6, 7, 8,
    10, 11, 12,
    14, 15
]

VALIDITY_INDICES = [
    9, 13
]

S2_CONTINUOUS_INDICES = set(
    range(0, 9)
)

S1_CONTINUOUS_INDICES = {
    10, 11, 12
}

PREDICTOR_BANDS = [
    "s2_b2",
    "s2_b3",
    "s2_b4",
    "s2_b8",
    "s2_b11",
    "s2_b12",
    "ndvi",
    "ndbi",
    "ndmi",
    "s2_valid",
    "s1_vv_db",
    "s1_vh_db",
    "s1_vv_minus_vh_db",
    "s1_valid",
    "elevation_m",
    "slope_deg",
]


# ======================================================================================
# 3. CHECK REQUIRED INPUTS
# ======================================================================================

required_paths = [
    FINAL52_PATH,
    VERIFIED_MATCH_PATH,
    NORMALIZATION_PATH,
    TARGET_STATS_PATH,
    STAGE12_VERDICT,
    *BATCH_VERIFY_PATHS,
]

for path in required_paths:

    if not path.exists():

        raise FileNotFoundError(
            f"Required input not found:\n{path}"
        )


stage12_text = STAGE12_VERDICT.read_text(
    encoding="utf-8"
)

if (
    "Overall model-ready verdict: PASS"
    not in stage12_text
):

    raise RuntimeError(
        "Original Stage 12 model-ready dataset "
        "did not record an overall PASS."
    )


# ======================================================================================
# 4. LOAD FINAL 52-SCENE MANIFEST
# ======================================================================================

external = pd.read_csv(
    FINAL52_PATH
)

if len(external) != EXPECTED_SCENES:

    raise RuntimeError(
        f"Expected {EXPECTED_SCENES} scenes "
        f"in final external manifest, "
        f"found {len(external)}."
    )

if external["record_id"].duplicated().any():

    raise RuntimeError(
        "Duplicate record_id detected "
        "in final external manifest."
    )

if external["export_id"].duplicated().any():

    raise RuntimeError(
        "Duplicate export_id detected."
    )


# ======================================================================================
# 5. LOAD AND COMBINE ALL THREE EXPORT VERIFICATIONS
# ======================================================================================

verification_frames = []

for path in BATCH_VERIFY_PATHS:

    frame = pd.read_csv(path)

    verification_frames.append(
        frame
    )

verification = pd.concat(
    verification_frames,
    ignore_index=True
)

if len(verification) != EXPECTED_SCENES:

    raise RuntimeError(
        f"Expected 52 verification rows, "
        f"found {len(verification)}."
    )

if verification[
    "record_id"
].duplicated().any():

    raise RuntimeError(
        "Duplicate record IDs in "
        "combined export verification."
    )

if not verification[
    "verification_status"
].eq("PASS").all():

    bad = verification.loc[
        ~verification[
            "verification_status"
        ].eq("PASS"),
        [
            "export_id",
            "record_id",
            "verification_status"
        ]
    ]

    print(bad)

    raise RuntimeError(
        "At least one Stage 04 export "
        "did not pass physical verification."
    )


# ======================================================================================
# 6. AUTHORITATIVE THERMAL TIMESTAMPS
# ======================================================================================

matches = pd.read_csv(
    VERIFIED_MATCH_PATH
)

if "thermal_datetime_utc" not in matches.columns:

    raise RuntimeError(
        "thermal_datetime_utc is missing "
        "from Stage 06 verified manifest."
    )

matches[
    "thermal_datetime_utc"
] = pd.to_datetime(
    matches["thermal_datetime_utc"],
    utc=True,
    errors="coerce"
)

timestamp_lookup = (
    matches[
        [
            "record_id",
            "thermal_datetime_utc",
            "thermal_scene_id",
        ]
    ]
    .drop_duplicates(
        subset=["record_id"]
    )
)

if timestamp_lookup[
    "record_id"
].duplicated().any():

    raise RuntimeError(
        "Stage 06 contains ambiguous "
        "record_id timestamps."
    )


# ======================================================================================
# 7. BUILD AUTHORITATIVE EXTERNAL SCENE TABLE
# ======================================================================================

verification_keep = verification[
    [
        "export_id",
        "record_id",
        "predictor_path",
        "thermal_path",
        "predictor_copies",
        "thermal_copies",
        "s2_valid_fraction",
        "s1_valid_fraction",
        "thermal_valid_fraction",
        "verification_status",
    ]
].copy()

# Avoid duplicated export_id column if already in external table
scene_table = external.merge(
    verification_keep,
    on=[
        "export_id",
        "record_id"
    ],
    how="inner",
    validate="one_to_one",
    suffixes=(
        "",
        "_stage04"
    )
)

scene_table = scene_table.merge(
    timestamp_lookup,
    on="record_id",
    how="left",
    validate="one_to_one",
    suffixes=(
        "",
        "_stage06"
    )
)

if len(scene_table) != EXPECTED_SCENES:

    raise RuntimeError(
        f"Scene-table merge produced "
        f"{len(scene_table)} rows instead of 52."
    )

if scene_table[
    "thermal_datetime_utc"
].isna().any():

    missing = scene_table.loc[
        scene_table[
            "thermal_datetime_utc"
        ].isna(),
        [
            "export_id",
            "record_id"
        ]
    ]

    print(missing)

    raise RuntimeError(
        "At least one external scene "
        "has no authoritative thermal timestamp."
    )


# ======================================================================================
# 8. CHECK PHYSICAL PATHS
# ======================================================================================

for row in scene_table.itertuples(
    index=False
):

    predictor_path = Path(
        row.predictor_path
    )

    thermal_path = Path(
        row.thermal_path
    )

    if not predictor_path.exists():

        raise FileNotFoundError(
            f"Missing predictor:\n"
            f"{predictor_path}"
        )

    if not thermal_path.exists():

        raise FileNotFoundError(
            f"Missing thermal target:\n"
            f"{thermal_path}"
        )


# ======================================================================================
# 9. LOAD FROZEN ORIGINAL NORMALIZATION
# ======================================================================================

normalization = pd.read_csv(
    NORMALIZATION_PATH
)

required_norm_columns = {
    "band_index",
    "band_name",
    "training_pixels",
    "mean",
    "std",
    "active_channel",
}

missing_columns = (
    required_norm_columns
    - set(normalization.columns)
)

if missing_columns:

    raise RuntimeError(
        "Original normalization table "
        f"is missing columns: {missing_columns}"
    )


# ----------------------------------------------------------------------
# Make active_channel robust to CSV string/bool interpretation
# ----------------------------------------------------------------------

def parse_bool(value):

    if isinstance(
        value,
        (bool, np.bool_)
    ):
        return bool(value)

    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
        "y"
    }


normalization[
    "active_channel"
] = normalization[
    "active_channel"
].map(parse_bool)


# ----------------------------------------------------------------------
# Force exact expected continuous-band ordering
# ----------------------------------------------------------------------

normalization = (
    normalization
    .set_index("band_index")
    .reindex(
        CONTINUOUS_INDICES
    )
    .reset_index()
)

if normalization[
    "band_name"
].isna().any():

    raise RuntimeError(
        "Original Stage 12 normalization "
        "does not cover all required "
        "continuous channels."
    )


expected_names = [
    PREDICTOR_BANDS[index]
    for index
    in CONTINUOUS_INDICES
]

if (
    normalization[
        "band_name"
    ].tolist()
    != expected_names
):

    print(
        normalization[
            [
                "band_index",
                "band_name"
            ]
        ]
    )

    raise RuntimeError(
        "Stage 12 normalization band order "
        "does not match the frozen "
        "ThermoFusion contract."
    )


band_mean = (
    normalization[
        "mean"
    ]
    .astype(float)
    .to_numpy()
)

band_std = (
    normalization[
        "std"
    ]
    .astype(float)
    .to_numpy()
)

band_active = (
    normalization[
        "active_channel"
    ]
    .astype(bool)
    .to_numpy()
)


if not np.isfinite(
    band_mean
).all():

    raise RuntimeError(
        "Frozen predictor means "
        "contain non-finite values."
    )

if not np.isfinite(
    band_std
).all():

    raise RuntimeError(
        "Frozen predictor standard deviations "
        "contain non-finite values."
    )

if np.any(
    band_std <= 0
):

    raise RuntimeError(
        "Frozen predictor normalization "
        "contains non-positive SD."
    )


# ======================================================================================
# 10. LOAD FROZEN ORIGINAL TARGET NORMALIZATION
# ======================================================================================

with TARGET_STATS_PATH.open(
    "r",
    encoding="utf-8"
) as handle:

    target_stats = json.load(
        handle
    )

target_mean = float(
    target_stats["mean_c"]
)

target_std = float(
    target_stats["std_c"]
)

if (
    not np.isfinite(
        target_mean
    )
    or
    not np.isfinite(
        target_std
    )
    or
    target_std <= 0
):

    raise RuntimeError(
        "Frozen Stage 12 target "
        "normalization is invalid."
    )


print(
    "=" * 115
)

print(
    "THERMOFUSION STAGE 05 — "
    "FROZEN MODEL-READY EXTERNAL52"
)

print(
    "=" * 115
)

print(
    f"External scenes                  : "
    f"{len(scene_table)}"
)

print(
    f"Frozen continuous channels       : "
    f"{len(CONTINUOUS_INDICES)}"
)

print(
    f"Active training-derived channels : "
    f"{int(band_active.sum())}/"
    f"{len(CONTINUOUS_INDICES)}"
)

inactive_names = (
    normalization.loc[
        ~normalization[
            "active_channel"
        ],
        "band_name"
    ]
    .tolist()
)

print(
    "Inactive frozen channels         : "
    + (
        ", ".join(
            inactive_names
        )
        if inactive_names
        else "none"
    )
)

print(
    f"Frozen target mean (°C)           : "
    f"{target_mean:.4f}"
)

print(
    f"Frozen target SD (°C)             : "
    f"{target_std:.4f}"
)

print(
    "Normalization refitting          : NO"
)


# ======================================================================================
# 11. ALIGN ONE EXTERNAL SCENE
#
# EXACT STAGE-12 GEOMETRIC LOGIC:
#   thermal LST -> bilinear
#   thermal valid mask -> nearest
#   central 256 x 256 crop
# ======================================================================================

def aligned_scene(row):

    predictor_path = Path(
        row.predictor_path
    )

    thermal_path = Path(
        row.thermal_path
    )

    with rasterio.open(
        predictor_path
    ) as predictor, rasterio.open(
        thermal_path
    ) as thermal:

        x = (
            predictor
            .read()
            .astype("float32")
        )

        if x.shape[0] != 16:

            raise RuntimeError(
                f"{row.export_id}: "
                f"expected 16 predictor bands, "
                f"found {x.shape[0]}."
            )

        thermal_data = (
            thermal.read()
        )

        if thermal_data.shape[0] != 2:

            raise RuntimeError(
                f"{row.export_id}: "
                f"expected 2 thermal bands, "
                f"found {thermal_data.shape[0]}."
            )

        y_native = (
            thermal_data[0]
            .astype("float32")
        )

        native_mask = (
            (
                thermal_data[1]
                > 0.5
            )
            &
            np.isfinite(
                y_native
            )
            &
            (
                y_native
                != NODATA
            )
        ).astype(
            "uint8"
        )


        # ------------------------------------------------------------------
        # Reproject native thermal grid onto 10-m predictor grid
        # ------------------------------------------------------------------

        y = np.full(
            (
                predictor.height,
                predictor.width
            ),
            np.nan,
            dtype="float32"
        )

        target_mask = np.zeros(
            (
                predictor.height,
                predictor.width
            ),
            dtype="uint8"
        )


        reproject(
            source=y_native,

            destination=y,

            src_transform=
                thermal.transform,

            src_crs=
                thermal.crs,

            src_nodata=
                NODATA,

            dst_transform=
                predictor.transform,

            dst_crs=
                predictor.crs,

            dst_nodata=
                np.nan,

            resampling=
                Resampling.bilinear,
        )


        reproject(
            source=native_mask,

            destination=
                target_mask,

            src_transform=
                thermal.transform,

            src_crs=
                thermal.crs,

            src_nodata=
                0,

            dst_transform=
                predictor.transform,

            dst_crs=
                predictor.crs,

            dst_nodata=
                0,

            resampling=
                Resampling.nearest,
        )


    # ==================================================================================
    # CENTRAL 256 × 256 CROP
    # ==================================================================================

    original_height = (
        x.shape[1]
    )

    original_width = (
        x.shape[2]
    )

    if (
        original_height
        < TARGET_SIZE
        or
        original_width
        < TARGET_SIZE
    ):

        raise RuntimeError(
            f"{row.export_id}: predictor grid "
            f"is {original_height}x"
            f"{original_width}; "
            f"minimum is "
            f"{TARGET_SIZE}x"
            f"{TARGET_SIZE}."
        )


    row_offset = (
        original_height
        - TARGET_SIZE
    ) // 2

    col_offset = (
        original_width
        - TARGET_SIZE
    ) // 2


    rows = slice(
        row_offset,
        row_offset
        + TARGET_SIZE
    )

    cols = slice(
        col_offset,
        col_offset
        + TARGET_SIZE
    )


    x = x[
        :,
        rows,
        cols
    ]

    y = y[
        rows,
        cols
    ]

    target_mask = (
        target_mask[
            rows,
            cols
        ]
        > 0
    )


    # ==================================================================================
    # MODALITY-SPECIFIC VALIDITY
    # ==================================================================================

    s2_valid = (
        np.isfinite(
            x[9]
        )
        &
        (
            x[9]
            != NODATA
        )
        &
        (
            x[9]
            > 0.5
        )
    )


    s1_valid = (
        np.isfinite(
            x[13]
        )
        &
        (
            x[13]
            != NODATA
        )
        &
        (
            x[13]
            > 0.5
        )
    )


    target_valid = (
        target_mask
        &
        np.isfinite(y)
        &
        (
            y != NODATA
        )
        &
        (
            y > -30
        )
        &
        (
            y < 80
        )
    )


    # ==================================================================================
    # PER-BAND VALIDITY
    # ==================================================================================

    band_valid = {}

    for band_index in CONTINUOUS_INDICES:

        valid = (
            np.isfinite(
                x[band_index]
            )
            &
            (
                x[band_index]
                != NODATA
            )
        )


        if (
            band_index
            in S2_CONTINUOUS_INDICES
        ):

            valid &= s2_valid


        elif (
            band_index
            in S1_CONTINUOUS_INDICES
        ):

            valid &= s1_valid


        band_valid[
            band_index
        ] = valid


    return (
        x,
        y,
        target_valid,
        band_valid,
        s2_valid,
        s1_valid,
        original_height,
        original_width,
        row_offset,
        col_offset,
    )


# ======================================================================================
# 12. BUILD ALL 52 MODEL-READY ARRAYS
# ======================================================================================

array_rows = []

scene_table = (
    scene_table
    .sort_values(
        "export_id"
    )
    .reset_index(
        drop=True
    )
)


for number, row in enumerate(
    scene_table.itertuples(
        index=False
    ),
    start=1
):

    (
        x,
        y,
        target_valid,
        band_valid,
        s2_valid,
        s1_valid,
        original_height,
        original_width,
        row_offset,
        col_offset,
    ) = aligned_scene(
        row
    )


    if (
        target_valid.sum()
        == 0
    ):

        raise RuntimeError(
            f"{row.export_id}: "
            "no valid thermal target pixels "
            "after reprojection/cropping."
        )


    # ==================================================================================
    # APPLY FROZEN TRAINING-ONLY NORMALIZATION
    # ==================================================================================

    x_normalized = np.zeros_like(
        x,
        dtype="float32"
    )


    for position, band_index in enumerate(
        CONTINUOUS_INDICES
    ):

        # Original training data had no usable
        # pixels for inactive channels.
        # Preserve original zero-filled contract.
        if not band_active[
            position
        ]:

            x_normalized[
                band_index
            ] = 0.0

            continue


        standardized = (
            x[
                band_index
            ]
            - band_mean[
                position
            ]
        ) / band_std[
            position
        ]


        standardized = np.clip(
            standardized,
            -NORMALIZATION_CLIP,
            NORMALIZATION_CLIP
        )


        standardized[
            ~band_valid[
                band_index
            ]
        ] = 0.0


        x_normalized[
            band_index
        ] = (
            standardized
            .astype(
                "float32"
            )
        )


    # ----------------------------------------------------------------------
    # Validity channels are NEVER standardized
    # ----------------------------------------------------------------------

    x_normalized[9] = (
        s2_valid
        .astype(
            "float32"
        )
    )

    x_normalized[13] = (
        s1_valid
        .astype(
            "float32"
        )
    )


    # ==================================================================================
    # TARGET ARRAYS
    # ==================================================================================

    y_c = np.where(
        target_valid,
        y,
        0.0
    ).astype(
        "float32"
    )


    y_standardized = np.where(
        target_valid,
        (
            y
            - target_mean
        )
        / target_std,
        0.0
    ).astype(
        "float32"
    )


    valid_mask = (
        target_valid
        .astype(
            "uint8"
        )
    )


    # ==================================================================================
    # SAVE
    # ==================================================================================

    output_path = (
        ARRAY_DIR
        / (
            f"{row.export_id}"
            f"_model_ready.npz"
        )
    )


    np.savez_compressed(
        output_path,

        x=
            x_normalized,

        y_c=
            y_c,

        y_standardized=
            y_standardized,

        valid_mask=
            valid_mask,

        predictor_band_names=
            np.asarray(
                PREDICTOR_BANDS
            ),

        export_id=
            np.asarray(
                row.export_id
            ),

        external_id=
            np.asarray(
                row.external_id
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

        thermal_datetime_utc=
            np.asarray(
                str(
                    row.thermal_datetime_utc
                )
            ),

        evaluation_role=
            np.asarray(
                "external_frozen_test"
            ),
    )


    valid_values = (
        y[
            target_valid
        ]
    )


    array_rows.append(
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

            "evaluation_role":
                "external_frozen_test",

            "array_path":
                str(
                    output_path
                ),

            "array_size_mb":
                (
                    output_path
                    .stat()
                    .st_size
                    / 1e6
                ),

            "channels":
                x_normalized.shape[0],

            "height":
                x_normalized.shape[1],

            "width":
                x_normalized.shape[2],

            "target_valid_pixels":
                int(
                    target_valid.sum()
                ),

            "target_valid_fraction":
                float(
                    target_valid.mean()
                ),

            "s2_valid_fraction":
                float(
                    s2_valid.mean()
                ),

            "s1_valid_fraction":
                float(
                    s1_valid.mean()
                ),

            "target_min_c":
                float(
                    valid_values.min()
                ),

            "target_median_c":
                float(
                    np.median(
                        valid_values
                    )
                ),

            "target_max_c":
                float(
                    valid_values.max()
                ),

            "original_predictor_height":
                original_height,

            "original_predictor_width":
                original_width,

            "crop_row_offset":
                row_offset,

            "crop_column_offset":
                col_offset,

            "normalization_source":
                "original_stage12_training_only",

            "status":
                "PASS",
        }
    )


    print(
        f"Prepared {number:02d}/52: "
        f"{row.export_id} | "
        f"{row.city} | "
        f"{row.thermal_sensor} | "
        f"thermal="
        f"{target_valid.mean():.1%} | "
        f"S2="
        f"{s2_valid.mean():.1%} | "
        f"S1="
        f"{s1_valid.mean():.1%}"
    )


# ======================================================================================
# 13. ARRAY MANIFEST
# ======================================================================================

array_manifest = pd.DataFrame(
    array_rows
)

ARRAY_MANIFEST_PATH = (
    OUTPUT_DIR
    / "01_external52_model_ready_manifest.csv"
)

array_manifest.to_csv(
    ARRAY_MANIFEST_PATH,
    index=False
)


# ======================================================================================
# 14. SAVE AUTHORITATIVE SCENE METADATA
# ======================================================================================

SCENE_TABLE_PATH = (
    OUTPUT_DIR
    / "02_external52_scene_metadata.csv"
)

scene_table.to_csv(
    SCENE_TABLE_PATH,
    index=False
)


# ======================================================================================
# 15. SAVE FROZEN NORMALIZATION COPY
# ======================================================================================

FROZEN_NORM_PATH = (
    OUTPUT_DIR
    / "03_frozen_training_normalization.csv"
)

normalization.to_csv(
    FROZEN_NORM_PATH,
    index=False
)


with (
    OUTPUT_DIR
    / "04_frozen_target_normalization.json"
).open(
    "w",
    encoding="utf-8"
) as handle:

    json.dump(
        {
            "target_name":
                "land_surface_temperature_c",

            "mean_c":
                target_mean,

            "std_c":
                target_std,

            "statistics_source":
                (
                    "Original ThermoFusion "
                    "Stage 12 training scenes only; "
                    "not refitted on external52"
                ),
        },
        handle,
        indent=2
    )


# ======================================================================================
# 16. RELOAD AND VERIFY EVERY ARRAY
# ======================================================================================

verification_rows = []


for row in array_manifest.itertuples(
    index=False
):

    path = Path(
        row.array_path
    )


    with np.load(
        path,
        allow_pickle=False
    ) as item:

        x = item[
            "x"
        ]

        y_c = item[
            "y_c"
        ]

        y_standardized = item[
            "y_standardized"
        ]

        mask = item[
            "valid_mask"
        ]


        checks = {

            "shape_check":
                (
                    x.shape
                    == (
                        16,
                        TARGET_SIZE,
                        TARGET_SIZE
                    )
                    and
                    y_c.shape
                    == (
                        TARGET_SIZE,
                        TARGET_SIZE
                    )
                    and
                    y_standardized.shape
                    == (
                        TARGET_SIZE,
                        TARGET_SIZE
                    )
                    and
                    mask.shape
                    == (
                        TARGET_SIZE,
                        TARGET_SIZE
                    )
                ),

            "dtype_check":
                (
                    x.dtype
                    == np.float32
                    and
                    y_c.dtype
                    == np.float32
                    and
                    y_standardized.dtype
                    == np.float32
                    and
                    mask.dtype
                    == np.uint8
                ),

            "finite_check":
                (
                    np.isfinite(
                        x
                    ).all()
                    and
                    np.isfinite(
                        y_c
                    ).all()
                    and
                    np.isfinite(
                        y_standardized
                    ).all()
                ),

            "mask_check":
                (
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
                ),

            "identity_check":
                (
                    str(
                        item[
                            "export_id"
                        ]
                    )
                    ==
                    row.export_id
                ),

            "channel_check":
                (
                    len(
                        item[
                            "predictor_band_names"
                        ]
                    )
                    == 16
                ),

            "s2_mask_check":
                (
                    set(
                        np.unique(
                            x[9]
                        )
                    )
                    .issubset(
                        {
                            0.0,
                            1.0
                        }
                    )
                ),

            "s1_mask_check":
                (
                    set(
                        np.unique(
                            x[13]
                        )
                    )
                    .issubset(
                        {
                            0.0,
                            1.0
                        }
                    )
                ),

            # Original slope channel is inactive.
            "inactive_slope_zero_check":
                (
                    np.allclose(
                        x[15],
                        0.0
                    )
                    if (
                        "slope_deg"
                        in inactive_names
                    )
                    else True
                ),
        }


        verification_rows.append(
            {
                "export_id":
                    row.export_id,

                "record_id":
                    row.record_id,

                **checks,

                "status":
                    (
                        "PASS"
                        if all(
                            checks.values()
                        )
                        else "REVIEW"
                    )
            }
        )


array_verification = pd.DataFrame(
    verification_rows
)

VERIFY_PATH = (
    OUTPUT_DIR
    / "05_array_integrity_verification.csv"
)

array_verification.to_csv(
    VERIFY_PATH,
    index=False
)


# ======================================================================================
# 17. CITY × SENSOR SUMMARY
# ======================================================================================

group_summary = (
    array_manifest
    .groupby(
        [
            "city",
            "thermal_sensor"
        ],
        observed=True
    )
    .agg(
        scenes=(
            "record_id",
            "size"
        ),

        mean_target_valid_fraction=(
            "target_valid_fraction",
            "mean"
        ),

        mean_s2_valid_fraction=(
            "s2_valid_fraction",
            "mean"
        ),

        mean_s1_valid_fraction=(
            "s1_valid_fraction",
            "mean"
        ),

        mean_target_median_c=(
            "target_median_c",
            "mean"
        ),
    )
    .reset_index()
)

GROUP_PATH = (
    OUTPUT_DIR
    / "06_city_sensor_model_ready_summary.csv"
)

group_summary.to_csv(
    GROUP_PATH,
    index=False
)


# ======================================================================================
# 18. GLOBAL AUDIT
# ======================================================================================

npz_files = list(
    ARRAY_DIR.glob(
        "*_model_ready.npz"
    )
)

all_arrays_pass = (
    array_verification[
        "status"
    ]
    .eq("PASS")
    .all()
)


overall_pass = (
    len(
        array_manifest
    )
    == EXPECTED_SCENES

    and

    array_manifest[
        "record_id"
    ]
    .nunique()
    == EXPECTED_SCENES

    and

    array_manifest[
        "export_id"
    ]
    .nunique()
    == EXPECTED_SCENES

    and

    len(
        npz_files
    )
    == EXPECTED_SCENES

    and

    all_arrays_pass
)


# ======================================================================================
# 19. VERDICT
# ======================================================================================

verdict = [

    "THERMOFUSION STAGE 05 — MODEL-READY EXTERNAL52",

    (
        f"Verified Stage 04 external scenes: "
        f"{len(scene_table)}/52"
    ),

    (
        f"Model-ready arrays created: "
        f"{len(array_manifest)}/52"
    ),

    (
        "Arrays passing reload integrity checks: "
        f"{array_verification['status'].eq('PASS').sum()}/52"
    ),

    "Raster predictor channels: 16",

    (
        "Active continuous predictor channels: "
        f"{int(band_active.sum())}/"
        f"{len(CONTINUOUS_INDICES)}"
    ),

    (
        "Inactive continuous channels: "
        + (
            ", ".join(
                inactive_names
            )
            if inactive_names
            else "none"
        )
    ),

    "Array dimensions: 256 x 256",

    (
        "Frozen training-only target mean (°C): "
        f"{target_mean:.4f}"
    ),

    (
        "Frozen training-only target standard deviation (°C): "
        f"{target_std:.4f}"
    ),

    "Predictor normalization refitting: NO",

    "Target normalization refitting: NO",

    "Model fitting on external52: NO",

    "Model selection on external52: NO",

    "External52 role: independent frozen-model temporal evaluation",

    (
        "Overall Stage 05 model-ready verdict: PASS"
        if overall_pass
        else
        "Overall Stage 05 model-ready verdict: REVIEW REQUIRED"
    ),
]


VERDICT_PATH = (
    OUTPUT_DIR
    / "07_stage05_model_ready_verdict.txt"
)

VERDICT_PATH.write_text(
    "\n".join(
        verdict
    ),
    encoding="utf-8"
)


# ======================================================================================
# 20. FINAL REPORT
# ======================================================================================

print(
    "\n"
    + "=" * 115
)

print(
    "\n".join(
        verdict
    )
)

print(
    "=" * 115
)


print(
    "\nCITY × SENSOR COMPOSITION"
)

print(
    "-" * 115
)

print(
    group_summary.to_string(
        index=False
    )
)


print(
    "\nMODEL-READY VALIDITY SUMMARY"
)

print(
    "-" * 115
)

print(
    f"Mean target-valid fraction : "
    f"{array_manifest['target_valid_fraction'].mean():.3f}"
)

print(
    f"Mean S2-valid fraction     : "
    f"{array_manifest['s2_valid_fraction'].mean():.3f}"
)

print(
    f"Mean S1-valid fraction     : "
    f"{array_manifest['s1_valid_fraction'].mean():.3f}"
)


print(
    "\nSaved:"
)

print(
    ARRAY_MANIFEST_PATH
)

print(
    SCENE_TABLE_PATH
)

print(
    FROZEN_NORM_PATH
)

print(
    VERIFY_PATH
)

print(
    GROUP_PATH
)

print(
    VERDICT_PATH
)


if not overall_pass:

    print(
        "\nSTOP — do not perform frozen-model inference."
    )

    print(
        "At least one Stage 05 integrity requirement failed."
    )

else:

    print(
        "\nPASS — all 52 external scenes are model-ready."
    )

    print(
        "NEXT STAGE: frozen final ThermoFusion model inference "
        "with eight-view TTA and the original conformal calibration."
    )
