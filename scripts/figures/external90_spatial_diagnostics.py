# =====================================================================
# ThermoFusion reproducibility code
# Source notebook cell(s): 72
# Audit status: KEEP
# Extracted without scientific-logic modification.
# Replace local Google Drive paths as described in README.md when needed.
# =====================================================================

# ======================================================================================
# THERMOFUSION STAGE 09
# PUBLICATION-QUALITY SPATIAL DIAGNOSTICS FOR THE 90-SCENE EXTERNAL EVALUATION
#
# OUTPUT FIGURE:
#   (a) West African external-evaluation geography and scene counts
#   (b) Representative external scene — observed LST
#   (c) Representative external scene — ThermoFusion prediction
#   (d) Representative external scene — spatial residual
#   (e) Strongest-transfer external scene — spatial residual
#   (f) Most challenging external scene — spatial residual
#
# IMPORTANT
#   - Uses actual frozen-model prediction arrays.
#   - Does NOT refit the model.
#   - Does NOT recalibrate uncertainty.
#   - Does NOT remove difficult scenes.
#   - Best/worst scenes are used only as spatial diagnostics.
#
# GPU NOT REQUIRED.
# ======================================================================================

from google.colab import drive
drive.mount("/content/drive")

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

warnings.filterwarnings("ignore")

# ======================================================================================
# 1. PATHS
# ======================================================================================

ROOT = Path("/content/drive/MyDrive")

# Final integrated evidence
S08 = (
    ROOT
    / "ThermoFusion_External_Evaluation"
    / "Stage08_IntegratedExternal90"
)

EVIDENCE_PATH = (
    S08
    / "03_integrated90_scene_evidence.csv"
)

STAGE08_VERDICT = (
    S08
    / "14_stage08_integrated_external_verdict.txt"
)

# ----------------------------------------------------------------------
# NEW52
# ----------------------------------------------------------------------

NEW_S05 = (
    ROOT
    / "ThermoFusion_External_Evaluation"
    / "Stage05_ModelReady_External52"
)

NEW_ARRAY_MANIFEST = (
    NEW_S05
    / "01_external52_model_ready_manifest.csv"
)

NEW_S06 = (
    ROOT
    / "ThermoFusion_External_Evaluation"
    / "Stage06_FrozenInference_External52"
)

NEW_METRICS = (
    NEW_S06
    / "01_external52_scene_accuracy_uncertainty.csv"
)

NEW_PRED_ROOT = (
    NEW_S06
    / "predictions"
)

# ----------------------------------------------------------------------
# ORIGINAL38
# ----------------------------------------------------------------------

OLD_S25 = (
    ROOT
    / "ThermoFusion_Stage25_ExpandedTest_ModelReady"
)

OLD_ARRAY_MANIFEST = (
    OLD_S25
    / "01_expanded_test_array_manifest.csv"
)

OLD_S27 = (
    ROOT
    / "ThermoFusion_Stage27_ExpandedTemporalTest_Analysis"
)

OLD_METRICS = (
    OLD_S27
    / "05_scene_level_comparison_table.csv"
)

OLD_PRED_ROOT = (
    ROOT
    / "ThermoFusion_Stage26_ExpandedTemporalTest_Inference"
)

# ----------------------------------------------------------------------
# OUTPUT
# ----------------------------------------------------------------------

OUT = (
    ROOT
    / "ThermoFusion_External_Evaluation"
    / "Stage09_SpatialDiagnostics"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# ======================================================================================
# 2. REQUIRED INPUT CHECK
# ======================================================================================

required = [
    EVIDENCE_PATH,
    STAGE08_VERDICT,
    NEW_ARRAY_MANIFEST,
    NEW_METRICS,
    NEW_PRED_ROOT,
    OLD_ARRAY_MANIFEST,
    OLD_METRICS,
    OLD_PRED_ROOT,
]

for path in required:

    if not path.exists():

        raise FileNotFoundError(
            f"Required Stage 09 input not found:\n{path}"
        )


if (
    "Overall Stage 08 integrated external evaluation: PASS"
    not in STAGE08_VERDICT.read_text(
        encoding="utf-8"
    )
):

    raise RuntimeError(
        "Stage 08 did not pass."
    )


# ======================================================================================
# 3. LOAD FINAL 90-SCENE EVIDENCE
# ======================================================================================

evidence = pd.read_csv(
    EVIDENCE_PATH
)


if len(evidence) != 90:

    raise RuntimeError(
        f"Expected 90 external scenes, found {len(evidence)}."
    )


if evidence["record_id"].duplicated().any():

    raise RuntimeError(
        "Duplicate record IDs found in Stage 08 evidence."
    )


# ======================================================================================
# 4. OBJECTIVE SCENE SELECTION
# ======================================================================================

# Best pooled baseline from Stage 08 = SENSOR climatology.
DELTA_COL = (
    "delta_rmse_vs_sensor_training_climatology_c"
)


if DELTA_COL not in evidence.columns:

    raise RuntimeError(
        f"Required column missing: {DELTA_COL}"
    )


# ----------------------------------------------------------------------
# Representative scene:
# scene whose model RMSE lies closest to the median of all 90 scenes.
# ----------------------------------------------------------------------

median_rmse = float(
    evidence[
        "model_rmse_c"
    ].median()
)


evidence[
    "_distance_from_median_rmse"
] = np.abs(
    evidence[
        "model_rmse_c"
    ]
    - median_rmse
)


representative = (
    evidence
    .sort_values(
        [
            "_distance_from_median_rmse",
            "record_id"
        ]
    )
    .iloc[0]
)


# ----------------------------------------------------------------------
# Strongest transfer:
# most negative model - sensor-climatology RMSE difference.
# ----------------------------------------------------------------------

strongest = (
    evidence
    .sort_values(
        [
            DELTA_COL,
            "record_id"
        ],
        ascending=[
            True,
            True
        ]
    )
    .iloc[0]
)


# ----------------------------------------------------------------------
# Challenging transfer:
# largest positive model - sensor-climatology RMSE difference.
# ----------------------------------------------------------------------

challenging = (
    evidence
    .sort_values(
        [
            DELTA_COL,
            "record_id"
        ],
        ascending=[
            False,
            True
        ]
    )
    .iloc[0]
)


selection = pd.DataFrame(
    [
        {
            "diagnostic_role":
                "representative",

            "record_id":
                representative.record_id,

            "cohort":
                representative.cohort,

            "city":
                representative.city,

            "thermal_sensor":
                representative.thermal_sensor,

            "model_rmse_c":
                representative.model_rmse_c,

            "sensor_climatology_rmse_c":
                representative.sensor_training_climatology_rmse_c,

            "delta_rmse_vs_sensor_c":
                representative[
                    DELTA_COL
                ],

            "selection_rule":
                "model RMSE closest to median RMSE of all 90 external scenes",
        },

        {
            "diagnostic_role":
                "strongest_transfer",

            "record_id":
                strongest.record_id,

            "cohort":
                strongest.cohort,

            "city":
                strongest.city,

            "thermal_sensor":
                strongest.thermal_sensor,

            "model_rmse_c":
                strongest.model_rmse_c,

            "sensor_climatology_rmse_c":
                strongest.sensor_training_climatology_rmse_c,

            "delta_rmse_vs_sensor_c":
                strongest[
                    DELTA_COL
                ],

            "selection_rule":
                "minimum ThermoFusion-minus-sensor-climatology scene RMSE",
        },

        {
            "diagnostic_role":
                "challenging_transfer",

            "record_id":
                challenging.record_id,

            "cohort":
                challenging.cohort,

            "city":
                challenging.city,

            "thermal_sensor":
                challenging.thermal_sensor,

            "model_rmse_c":
                challenging.model_rmse_c,

            "sensor_climatology_rmse_c":
                challenging.sensor_training_climatology_rmse_c,

            "delta_rmse_vs_sensor_c":
                challenging[
                    DELTA_COL
                ],

            "selection_rule":
                "maximum ThermoFusion-minus-sensor-climatology scene RMSE",
        },
    ]
)


SELECTION_PATH = (
    OUT
    / "01_spatial_diagnostic_scene_selection.csv"
)


selection.to_csv(
    SELECTION_PATH,
    index=False
)


print("=" * 120)
print("THERMOFUSION STAGE 09 — SPATIAL DIAGNOSTIC SCENE SELECTION")
print("=" * 120)

print(
    selection.to_string(
        index=False
    )
)


# ======================================================================================
# 5. LOAD SOURCE MANIFESTS
# ======================================================================================

new_arrays = pd.read_csv(
    NEW_ARRAY_MANIFEST
)

old_arrays = pd.read_csv(
    OLD_ARRAY_MANIFEST
)

new_metrics = pd.read_csv(
    NEW_METRICS
)

old_metrics = pd.read_csv(
    OLD_METRICS
)


# ======================================================================================
# 6. BUILD RECORD-ID → ARRAY PATH LOOKUP
# ======================================================================================

array_lookup = {}


for row in old_arrays.itertuples(
    index=False
):

    array_lookup[
        str(
            row.record_id
        )
    ] = Path(
        row.array_path
    )


for row in new_arrays.itertuples(
    index=False
):

    array_lookup[
        str(
            row.record_id
        )
    ] = Path(
        row.array_path
    )


# ======================================================================================
# 7. RECORD-ID → EXPORT-ID LOOKUP
# ======================================================================================

export_lookup = {}


if (
    "export_id"
    in old_metrics.columns
):

    for row in old_metrics.itertuples(
        index=False
    ):

        export_lookup[
            str(
                row.record_id
            )
        ] = str(
            row.export_id
        )


if (
    "export_id"
    in new_metrics.columns
):

    for row in new_metrics.itertuples(
        index=False
    ):

        export_lookup[
            str(
                row.record_id
            )
        ] = str(
            row.export_id
        )


# ======================================================================================
# 8. INDEX ALL SAVED PREDICTION FILES
# ======================================================================================

prediction_files = []


for root in [
    OLD_PRED_ROOT,
    NEW_PRED_ROOT,
]:

    if root.exists():

        prediction_files.extend(
            list(
                root.rglob(
                    "*.npz"
                )
            )
        )


print(
    f"\nPrediction NPZ files indexed: "
    f"{len(prediction_files)}"
)


if len(prediction_files) == 0:

    raise RuntimeError(
        "No frozen prediction NPZ files were found."
    )


# ======================================================================================
# 9. PREDICTION FILE MATCHER
# ======================================================================================

def normalize_text(
    value
):

    return (
        str(
            value
        )
        .lower()
        .replace(
            " ",
            ""
        )
    )


def locate_prediction_file(
    record_id
):

    record_id = str(
        record_id
    )


    tokens = [
        record_id
    ]


    if record_id in export_lookup:

        tokens.append(
            export_lookup[
                record_id
            ]
        )


    normalized_tokens = [
        normalize_text(
            token
        )

        for token in tokens

        if str(
            token
        ).strip()
    ]


    matches = []


    for path in prediction_files:

        path_text = normalize_text(
            str(
                path
            )
        )


        score = 0


        for token in normalized_tokens:

            if token in path_text:

                score += len(
                    token
                )


        if score > 0:

            matches.append(
                (
                    score,
                    path
                )
            )


    if not matches:

        raise FileNotFoundError(
            f"No prediction file matched record_id={record_id}\n"
            f"Search tokens={tokens}"
        )


    matches = sorted(
        matches,
        key=lambda item:
            (
                -item[0],
                len(
                    str(
                        item[1]
                    )
                )
            )
    )


    return matches[
        0
    ][1]


# ======================================================================================
# 10. FLEXIBLE NPZ KEY DISCOVERY
# ======================================================================================

def first_array(
    item,
    candidates
):

    for key in candidates:

        if key in item.files:

            return (
                item[
                    key
                ],
                key
            )


    return (
        None,
        None
    )


# ======================================================================================
# 11. LOAD SPATIAL DATA FOR ONE SCENE
# ======================================================================================

def load_scene_maps(
    record_id
):

    record_id = str(
        record_id
    )


    # ------------------------------------------------------------------
    # Target/model-ready array
    # ------------------------------------------------------------------

    if record_id not in array_lookup:

        raise RuntimeError(
            f"No model-ready array found for {record_id}"
        )


    array_path = (
        array_lookup[
            record_id
        ]
    )


    if not array_path.exists():

        raise FileNotFoundError(
            f"Model-ready array missing:\n{array_path}"
        )


    with np.load(
        array_path,
        allow_pickle=False
    ) as target_item:

        target, target_key = first_array(
            target_item,
            [
                "y_c",
                "observed_c",
                "target_c",
                "target",
            ]
        )


        mask, mask_key = first_array(
            target_item,
            [
                "valid_mask",
                "mask",
                "target_valid_mask",
            ]
        )


    if target is None:

        raise RuntimeError(
            f"No target array found in {array_path}"
        )


    target = np.asarray(
        target
    ).squeeze()


    if mask is None:

        mask = np.isfinite(
            target
        )

    else:

        mask = (
            np.asarray(
                mask
            ).squeeze()
            > 0
        )


    # ------------------------------------------------------------------
    # Frozen prediction file
    # ------------------------------------------------------------------

    pred_path = locate_prediction_file(
        record_id
    )


    with np.load(
        pred_path,
        allow_pickle=False
    ) as pred_item:

        prediction, prediction_key = first_array(
            pred_item,
            [
                "prediction_c",
                "predicted_c",
                "mean_prediction_c",
                "prediction",
                "pred_c",
                "y_pred_c",
                "mean_c",
            ]
        )


        pred_mask, pred_mask_key = first_array(
            pred_item,
            [
                "valid_mask",
                "mask",
                "target_valid_mask",
            ]
        )


        uncertainty, uncertainty_key = first_array(
            pred_item,
            [
                "tta_std_c",
                "prediction_std_c",
                "uncertainty_c",
                "std_c",
                "tta_uncertainty_c",
            ]
        )


    if prediction is None:

        print(
            f"\nAvailable keys in prediction file "
            f"{pred_path.name}:"
        )

        with np.load(
            pred_path,
            allow_pickle=False
        ) as item:

            print(
                item.files
            )


        raise RuntimeError(
            f"Could not identify prediction grid for {record_id}"
        )


    prediction = np.asarray(
        prediction
    ).squeeze()


    # ------------------------------------------------------------------
    # Shape checks
    # ------------------------------------------------------------------

    if prediction.shape != target.shape:

        raise RuntimeError(
            f"Shape mismatch for {record_id}: "
            f"target={target.shape}, "
            f"prediction={prediction.shape}"
        )


    if pred_mask is not None:

        pred_mask = (
            np.asarray(
                pred_mask
            ).squeeze()
            > 0
        )

        if pred_mask.shape == mask.shape:

            mask = (
                mask
                &
                pred_mask
            )


    mask = (
        mask
        &
        np.isfinite(
            target
        )
        &
        np.isfinite(
            prediction
        )
    )


    observed = np.where(
        mask,
        target,
        np.nan
    )


    predicted = np.where(
        mask,
        prediction,
        np.nan
    )


    residual = np.where(
        mask,
        prediction
        - target,
        np.nan
    )


    absolute_error = np.where(
        mask,
        np.abs(
            prediction
            - target
        ),
        np.nan
    )


    if uncertainty is not None:

        uncertainty = np.asarray(
            uncertainty
        ).squeeze()


        if (
            uncertainty.shape
            == target.shape
        ):

            uncertainty = np.where(
                mask,
                uncertainty,
                np.nan
            )

        else:

            uncertainty = None


    return {
        "record_id":
            record_id,

        "array_path":
            array_path,

        "prediction_path":
            pred_path,

        "observed":
            observed,

        "predicted":
            predicted,

        "residual":
            residual,

        "absolute_error":
            absolute_error,

        "uncertainty":
            uncertainty,

        "valid_mask":
            mask,

        "prediction_key":
            prediction_key,

        "target_key":
            target_key,
    }


# ======================================================================================
# 12. LOAD THE THREE SELECTED SCENES
# ======================================================================================

rep_maps = load_scene_maps(
    representative.record_id
)

best_maps = load_scene_maps(
    strongest.record_id
)

challenge_maps = load_scene_maps(
    challenging.record_id
)


print(
    "\nSelected prediction files:"
)

print(
    "Representative:",
    rep_maps[
        "prediction_path"
    ]
)

print(
    "Strongest transfer:",
    best_maps[
        "prediction_path"
    ]
)

print(
    "Challenging transfer:",
    challenge_maps[
        "prediction_path"
    ]
)


# ======================================================================================
# 13. VERIFY SPATIAL METRICS AGAINST STAGE 08
# ======================================================================================

def spatial_rmse(
    maps
):

    residual = maps[
        "residual"
    ]


    values = residual[
        np.isfinite(
            residual
        )
    ]


    return float(
        np.sqrt(
            np.mean(
                values ** 2
            )
        )
    )


for label, row, maps in [
    (
        "representative",
        representative,
        rep_maps
    ),
    (
        "strongest_transfer",
        strongest,
        best_maps
    ),
    (
        "challenging_transfer",
        challenging,
        challenge_maps
    ),
]:

    computed = spatial_rmse(
        maps
    )


    stored = float(
        row.model_rmse_c
    )


    difference = abs(
        computed
        - stored
    )


    print(
        f"{label:22s} | "
        f"stored RMSE={stored:.4f} °C | "
        f"map RMSE={computed:.4f} °C | "
        f"difference={difference:.6f}"
    )


    if difference > 0.02:

        raise RuntimeError(
            f"Spatial reconstruction check failed for {label}."
        )


# ======================================================================================
# 14. STUDY AREA / CITY INFORMATION
# ======================================================================================

city_coords = {
    "Abidjan":
        (
            -4.01,
            5.36
        ),

    "Accra":
        (
            -0.19,
            5.60
        ),

    "Freetown":
        (
            -13.23,
            8.47
        ),

    "Lagos":
        (
            3.38,
            6.52
        ),
}


city_counts = (
    evidence
    .groupby(
        [
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


total_city_counts = (
    evidence
    .groupby(
        "city",
        observed=True
    )
    .size()
    .to_dict()
)


# ======================================================================================
# 15. OPTIONAL NATURAL-EARTH MAP BACKGROUND
#
# The script attempts to download a lightweight Natural Earth GeoJSON.
# If network access is unavailable, it still generates the city-location panel.
# ======================================================================================

world = None


try:

    import geopandas as gpd

    NATURAL_EARTH_URL = (
        "https://raw.githubusercontent.com/"
        "nvkelso/natural-earth-vector/master/"
        "geojson/ne_110m_admin_0_countries.geojson"
    )


    world = gpd.read_file(
        NATURAL_EARTH_URL
    )


    print(
        "\nNatural Earth background loaded."
    )


except Exception as exc:

    print(
        "\nNatural Earth background could not be loaded."
    )

    print(
        "The geographic panel will use city coordinates only."
    )

    print(
        "Reason:",
        exc
    )


# ======================================================================================
# 16. FIGURE PARAMETERS
# ======================================================================================

mpl.rcParams.update(
    {
        "font.family":
            "DejaVu Sans",

        "font.size":
            8.5,

        "axes.titlesize":
            9.5,

        "axes.labelsize":
            8.5,

        "xtick.labelsize":
            7.5,

        "ytick.labelsize":
            7.5,

        "legend.fontsize":
            7.5,

        "figure.dpi":
            150,

        "savefig.dpi":
            900,
    }
)


# ----------------------------------------------------------------------
# The model-ready grid is 256 × 256 pixels at ~10 m.
# Display as distance from chip centre in kilometres.
# ----------------------------------------------------------------------

GRID_SIZE = 256
PIXEL_SIZE_M = 10.0

HALF_WIDTH_KM = (
    GRID_SIZE
    * PIXEL_SIZE_M
    / 2000.0
)

SPATIAL_EXTENT = [
    -HALF_WIDTH_KM,
    HALF_WIDTH_KM,
    -HALF_WIDTH_KM,
    HALF_WIDTH_KM,
]


# ======================================================================================
# 17. COMMON LST RANGE FOR REPRESENTATIVE OBSERVED/PREDICTED PANELS
# ======================================================================================

rep_values = np.concatenate(
    [
        rep_maps[
            "observed"
        ][
            np.isfinite(
                rep_maps[
                    "observed"
                ]
            )
        ],

        rep_maps[
            "predicted"
        ][
            np.isfinite(
                rep_maps[
                    "predicted"
                ]
            )
        ],
    ]
)


lst_vmin = float(
    np.nanpercentile(
        rep_values,
        2
    )
)


lst_vmax = float(
    np.nanpercentile(
        rep_values,
        98
    )
)


# ======================================================================================
# 18. COMMON RESIDUAL RANGE FOR ALL THREE RESIDUAL MAPS
# ======================================================================================

all_residuals = np.concatenate(
    [
        rep_maps[
            "residual"
        ][
            np.isfinite(
                rep_maps[
                    "residual"
                ]
            )
        ],

        best_maps[
            "residual"
        ][
            np.isfinite(
                best_maps[
                    "residual"
                ]
            )
        ],

        challenge_maps[
            "residual"
        ][
            np.isfinite(
                challenge_maps[
                    "residual"
                ]
            )
        ],
    ]
)


res_limit = float(
    np.nanpercentile(
        np.abs(
            all_residuals
        ),
        98
    )
)


res_limit = max(
    res_limit,
    1.0
)


# ======================================================================================
# 19. HELPER FUNCTIONS
# ======================================================================================

def panel_label(
    ax,
    label
):

    ax.text(
        0.018,
        0.975,
        label,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=12,
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.18",
            facecolor="white",
            edgecolor="none",
            alpha=0.88,
        ),
        zorder=20,
    )


def raster_panel(
    ax,
    array,
    cmap,
    vmin,
    vmax,
    label,
    subtitle,
):

    image = ax.imshow(
        array,
        origin="upper",
        extent=SPATIAL_EXTENT,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )


    panel_label(
        ax,
        label
    )


    ax.set_title(
        subtitle,
        pad=5
    )


    ax.set_xlabel(
        "Easting offset from chip centre (km)"
    )


    ax.set_ylabel(
        "Northing offset from chip centre (km)"
    )


    ax.set_aspect(
        "equal"
    )


    return image


# ======================================================================================
# 20. CREATE FIGURE
# ======================================================================================

fig = plt.figure(
    figsize=(
        15.5,
        10.6
    )
)


gs = fig.add_gridspec(
    2,
    3,
    left=0.055,
    right=0.975,
    bottom=0.075,
    top=0.97,
    wspace=0.24,
    hspace=0.27,
)


# ======================================================================================
# PANEL A — EXTERNAL-EVALUATION GEOGRAPHY
# ======================================================================================

ax = fig.add_subplot(
    gs[
        0,
        0
    ]
)


if world is not None:

    subset = world.cx[
        -18:8,
        2:13
    ]


    subset.plot(
        ax=ax,
        facecolor="#F2F2F2",
        edgecolor="#777777",
        linewidth=0.55,
        zorder=1,
    )


sensor_text = {}


for city in city_coords:

    group = city_counts.loc[
        city_counts[
            "city"
        ].eq(
            city
        )
    ]


    eco = int(
        group.loc[
            group[
                "thermal_sensor"
            ].eq(
                "ECOSTRESS"
            ),
            "scenes"
        ].sum()
    )


    landsat = int(
        group.loc[
            group[
                "thermal_sensor"
            ].eq(
                "Landsat"
            ),
            "scenes"
        ].sum()
    )


    sensor_text[
        city
    ] = (
        f"E={eco}, L={landsat}"
    )


for city, (
    lon,
    lat
) in city_coords.items():

    count = int(
        total_city_counts.get(
            city,
            0
        )
    )


    bubble_size = (
        55
        + count
        * 10
    )


    ax.scatter(
        lon,
        lat,
        s=bubble_size,
        marker="o",
        facecolor="#2F6690",
        edgecolor="white",
        linewidth=1.1,
        alpha=0.92,
        zorder=5,
    )


    ax.text(
        lon,
        lat,
        str(
            count
        ),
        ha="center",
        va="center",
        fontsize=8,
        fontweight="bold",
        color="white",
        zorder=6,
    )


offsets = {
    "Freetown":
        (
            0.55,
            0.25
        ),

    "Abidjan":
        (
            0.55,
            -0.55
        ),

    "Accra":
        (
            0.55,
            0.28
        ),

    "Lagos":
        (
            -4.9,
            -0.65
        ),
}


for city, (
    lon,
    lat
) in city_coords.items():

    dx, dy = offsets[
        city
    ]


    ax.text(
        lon
        + dx,
        lat
        + dy,
        (
            f"{city}\n"
            f"{sensor_text[city]}"
        ),
        fontsize=7.5,
        ha="left",
        va="center",
        zorder=8,
    )


ax.set_xlim(
    -16,
    6
)


ax.set_ylim(
    3,
    10.8
)


ax.set_xlabel(
    "Longitude (°)"
)


ax.set_ylabel(
    "Latitude (°)"
)


ax.grid(
    True,
    linewidth=0.35,
    alpha=0.35,
)


panel_label(
    ax,
    "(a)"
)


ax.set_title(
    "Independent external temporal evaluation: 90 acquisitions",
    pad=5,
)


ax.text(
    0.02,
    0.035,
    "Bubble label = total scenes\nE = ECOSTRESS; L = Landsat",
    transform=ax.transAxes,
    fontsize=7,
    va="bottom",
    ha="left",
    bbox=dict(
        facecolor="white",
        alpha=0.82,
        edgecolor="#BBBBBB",
        linewidth=0.5,
    ),
)


# ======================================================================================
# PANEL B — REPRESENTATIVE OBSERVED
# ======================================================================================

ax_b = fig.add_subplot(
    gs[
        0,
        1
    ]
)


im_lst1 = raster_panel(
    ax_b,
    rep_maps[
        "observed"
    ],
    "inferno",
    lst_vmin,
    lst_vmax,
    "(b)",
    (
        f"Representative scene: observed LST\n"
        f"{representative.city} · "
        f"{representative.thermal_sensor}"
    ),
)


# ======================================================================================
# PANEL C — REPRESENTATIVE PREDICTION
# ======================================================================================

ax_c = fig.add_subplot(
    gs[
        0,
        2
    ]
)


im_lst2 = raster_panel(
    ax_c,
    rep_maps[
        "predicted"
    ],
    "inferno",
    lst_vmin,
    lst_vmax,
    "(c)",
    (
        f"ThermoFusion reconstruction\n"
        f"RMSE = {representative.model_rmse_c:.2f} °C"
    ),
)


# ======================================================================================
# PANEL D — REPRESENTATIVE RESIDUAL
# ======================================================================================

ax_d = fig.add_subplot(
    gs[
        1,
        0
    ]
)


im_res1 = raster_panel(
    ax_d,
    rep_maps[
        "residual"
    ],
    "RdBu_r",
    -res_limit,
    res_limit,
    "(d)",
    (
        f"Representative residual: predicted − observed\n"
        f"ΔRMSE vs sensor climatology = "
        f"{representative[DELTA_COL]:+.2f} °C"
    ),
)


# ======================================================================================
# PANEL E — STRONGEST TRANSFER
# ======================================================================================

ax_e = fig.add_subplot(
    gs[
        1,
        1
    ]
)


im_res2 = raster_panel(
    ax_e,
    best_maps[
        "residual"
    ],
    "RdBu_r",
    -res_limit,
    res_limit,
    "(e)",
    (
        f"Strongest transfer: {strongest.city} · "
        f"{strongest.thermal_sensor}\n"
        f"RMSE = {strongest.model_rmse_c:.2f} °C; "
        f"Δ = {strongest[DELTA_COL]:+.2f} °C"
    ),
)


# ======================================================================================
# PANEL F — CHALLENGING TRANSFER
# ======================================================================================

ax_f = fig.add_subplot(
    gs[
        1,
        2
    ]
)


im_res3 = raster_panel(
    ax_f,
    challenge_maps[
        "residual"
    ],
    "RdBu_r",
    -res_limit,
    res_limit,
    "(f)",
    (
        f"Challenging transfer: {challenging.city} · "
        f"{challenging.thermal_sensor}\n"
        f"RMSE = {challenging.model_rmse_c:.2f} °C; "
        f"Δ = {challenging[DELTA_COL]:+.2f} °C"
    ),
)


# ======================================================================================
# 21. SHARED COLORBARS
# ======================================================================================

# LST colorbar for panels B and C
lst_cbar = fig.colorbar(
    im_lst2,
    ax=[
        ax_b,
        ax_c
    ],
    orientation="horizontal",
    fraction=0.040,
    pad=0.075,
    aspect=45,
)


lst_cbar.set_label(
    "Land-surface temperature (°C)"
)


# Residual colorbar for panels D–F
res_cbar = fig.colorbar(
    im_res3,
    ax=[
        ax_d,
        ax_e,
        ax_f
    ],
    orientation="horizontal",
    fraction=0.040,
    pad=0.075,
    aspect=55,
)


res_cbar.set_label(
    "Prediction residual (°C): ThermoFusion − observed"
)


# ======================================================================================
# 22. SAVE PUBLICATION OUTPUTS
# ======================================================================================

PNG_PATH = (
    OUT
    / "ThermoFusion_External90_Spatial_Diagnostics.png"
)

PDF_PATH = (
    OUT
    / "ThermoFusion_External90_Spatial_Diagnostics.pdf"
)

SVG_PATH = (
    OUT
    / "ThermoFusion_External90_Spatial_Diagnostics.svg"
)


fig.savefig(
    PNG_PATH,
    dpi=900,
    bbox_inches="tight",
    facecolor="white",
)


fig.savefig(
    PDF_PATH,
    bbox_inches="tight",
    facecolor="white",
)


fig.savefig(
    SVG_PATH,
    bbox_inches="tight",
    facecolor="white",
)


plt.show()


# ======================================================================================
# 23. SAVE MAP METADATA / CAPTION SUPPORT
# ======================================================================================

metadata_rows = []


for role, row, maps in [
    (
        "representative",
        representative,
        rep_maps
    ),
    (
        "strongest_transfer",
        strongest,
        best_maps
    ),
    (
        "challenging_transfer",
        challenging,
        challenge_maps
    ),
]:

    metadata_rows.append(
        {
            "diagnostic_role":
                role,

            "record_id":
                row.record_id,

            "cohort":
                row.cohort,

            "city":
                row.city,

            "thermal_sensor":
                row.thermal_sensor,

            "model_rmse_c":
                float(
                    row.model_rmse_c
                ),

            "sensor_climatology_rmse_c":
                float(
                    row.sensor_training_climatology_rmse_c
                ),

            "delta_rmse_vs_sensor_c":
                float(
                    row[
                        DELTA_COL
                    ]
                ),

            "valid_pixels":
                int(
                    np.sum(
                        maps[
                            "valid_mask"
                        ]
                    )
                ),

            "prediction_file":
                str(
                    maps[
                        "prediction_path"
                    ]
                ),

            "array_file":
                str(
                    maps[
                        "array_path"
                    ]
                ),

            "prediction_npz_key":
                maps[
                    "prediction_key"
                ],

            "target_npz_key":
                maps[
                    "target_key"
                ],
        }
    )


metadata = pd.DataFrame(
    metadata_rows
)


METADATA_PATH = (
    OUT
    / "02_spatial_figure_scene_metadata.csv"
)


metadata.to_csv(
    METADATA_PATH,
    index=False
)


# ======================================================================================
# 24. DRAFT FIGURE CAPTION
# ======================================================================================

caption = f"""
Spatial diagnostics for the independent external temporal evaluation.
(a) Geographic distribution of the 90 independent later-date acquisitions across
Abidjan, Accra, Freetown and Lagos; bubble labels indicate scene counts and E/L
denote ECOSTRESS/Landsat contributions. (b) Observed land-surface temperature for
the representative external acquisition, selected objectively as the scene with
ThermoFusion RMSE closest to the median RMSE across all 90 external acquisitions.
(c) Corresponding frozen ThermoFusion reconstruction. (d) Spatial residual for
the representative acquisition, expressed as predicted minus observed LST.
(e) Residual field for the strongest-transfer acquisition relative to the frozen
sensor climatology (ΔRMSE = {strongest[DELTA_COL]:+.2f} °C).
(f) Residual field for the most challenging transfer case
(ΔRMSE = {challenging[DELTA_COL]:+.2f} °C).
The model, predictor normalization, climatological references and uncertainty
configuration remained frozen throughout external evaluation. Residual maps use
a common symmetric colour scale to permit direct spatial comparison.
""".strip()


CAPTION_PATH = (
    OUT
    / "03_spatial_figure_caption.txt"
)


CAPTION_PATH.write_text(
    caption,
    encoding="utf-8"
)


# ======================================================================================
# 25. STAGE 09 VERDICT
# ======================================================================================

outputs_exist = all(
    path.exists()

    for path in [
        PNG_PATH,
        PDF_PATH,
        SVG_PATH,
        SELECTION_PATH,
        METADATA_PATH,
        CAPTION_PATH,
    ]
)


verdict = [
    "THERMOFUSION STAGE 09 — EXTERNAL90 SPATIAL DIAGNOSTICS",

    "Integrated external scenes available: 90/90",

    "Scene selection based on predefined diagnostic rules: YES",

    "Performance-based scene exclusion: NO",

    "Model refitting: NO",

    "Normalization refitting: NO",

    "Climatology refitting: NO",

    (
        f"Representative scene: "
        f"{representative.record_id}"
    ),

    (
        f"Strongest-transfer scene: "
        f"{strongest.record_id}"
    ),

    (
        f"Challenging-transfer scene: "
        f"{challenging.record_id}"
    ),

    (
        f"Representative scene RMSE (°C): "
        f"{representative.model_rmse_c:.4f}"
    ),

    (
        f"Strongest-transfer ΔRMSE vs sensor climatology (°C): "
        f"{strongest[DELTA_COL]:.4f}"
    ),

    (
        f"Challenging-transfer ΔRMSE vs sensor climatology (°C): "
        f"{challenging[DELTA_COL]:.4f}"
    ),

    (
        "Overall Stage 09 spatial-figure verdict: PASS"
        if outputs_exist
        else
        "Overall Stage 09 spatial-figure verdict: REVIEW REQUIRED"
    ),
]


VERDICT_PATH = (
    OUT
    / "04_stage09_spatial_figure_verdict.txt"
)


VERDICT_PATH.write_text(
    "\n".join(
        verdict
    ),
    encoding="utf-8"
)


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
    "\nSaved:"
)

for path in [
    PNG_PATH,
    PDF_PATH,
    SVG_PATH,
    SELECTION_PATH,
    METADATA_PATH,
    CAPTION_PATH,
    VERDICT_PATH,
]:

    print(
        path
    )


print(
    "\nFigure caption:\n"
)

print(
    caption
)
