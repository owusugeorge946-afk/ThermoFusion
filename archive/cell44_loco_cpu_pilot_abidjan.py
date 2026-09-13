# =====================================================================
# ThermoFusion reproducibility code
# Source notebook cell(s): 44
# Audit status: ARCHIVE
# Extracted without scientific-logic modification.
# Replace local Google Drive paths as described in README.md when needed.
# =====================================================================

# ============================================================
# THERMOFUSION LEAVE-ONE-CITY-OUT: CPU PILOT
# Full architecture, held-out city = Abidjan
# Includes automatic checkpoint recovery
# ============================================================

from google.colab import drive
drive.mount("/content/drive")

from pathlib import Path
import json
import os
import random
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ------------------------------------------------------------
# 1. File locations
# ------------------------------------------------------------
DRIVE_ROOT = Path("/content/drive/MyDrive")

STAGE12 = (
    DRIVE_ROOT
    / "ThermoFusion_Stage12_ModelReady_Pilot"
)

STAGE20 = (
    DRIVE_ROOT
    / "ThermoFusion_Stage20_LeaveOneCityOut"
)

FOLD_NAME = "holdout_abidjan"
HELD_OUT_CITY = "Abidjan"

FOLD_MANIFEST = (
    STAGE20
    / "fold_manifests"
    / f"{FOLD_NAME}.csv"
)

OUTPUT_DIR = (
    STAGE20
    / "pilot_holdout_abidjan_cpu"
)

CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

BEST_CHECKPOINT = (
    CHECKPOINT_DIR
    / "holdout_abidjan_best_cpu.pt"
)

RECOVERY_CHECKPOINT = (
    CHECKPOINT_DIR
    / "holdout_abidjan_recovery_cpu.pt"
)

HISTORY_PATH = (
    OUTPUT_DIR
    / "02_training_history.csv"
)

# ------------------------------------------------------------
# 2. Full publication configuration
# ------------------------------------------------------------
SEED = 20260908

MAXIMUM_EPOCHS = 80
EARLY_STOPPING_PATIENCE = 15

BATCH_SIZE = 2
BASE_WIDTH = 24

LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-5
MINIMUM_LEARNING_RATE = 1e-6

HUBER_DELTA = 1.0
GRADIENT_CLIP = 5.0

NUM_WORKERS = 0
TERRAIN_CHANNELS = [14, 15]

# Set False only when you want to discard an existing
# recovery checkpoint and restart from epoch 1.
RESUME_IF_AVAILABLE = True

# ------------------------------------------------------------
# 3. CPU configuration
# ------------------------------------------------------------
device = torch.device("cpu")

available_cores = os.cpu_count() or 2
cpu_threads = max(1, available_cores - 1)

torch.set_num_threads(cpu_threads)

try:
    torch.set_num_interop_threads(
        min(4, cpu_threads)
    )
except RuntimeError:
    pass

print("=" * 90)
print("THERMOFUSION UNSEEN-CITY CPU PILOT")
print("=" * 90)
print(f"Device:              {device}")
print(f"Available CPU cores: {available_cores}")
print(f"PyTorch CPU threads: {cpu_threads}")
print(f"Held-out city:       {HELD_OUT_CITY}")
print(f"Maximum epochs:      {MAXIMUM_EPOCHS}")
print(f"Batch size:          {BATCH_SIZE}")
print(f"Base width:          {BASE_WIDTH}")

# ------------------------------------------------------------
# 4. Reproducibility
# ------------------------------------------------------------
def reset_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


reset_seed()

# ------------------------------------------------------------
# 5. Load fold manifest
# ------------------------------------------------------------
if not FOLD_MANIFEST.exists():
    raise FileNotFoundError(
        f"Fold manifest not found:\n{FOLD_MANIFEST}"
    )

scenes = pd.read_csv(FOLD_MANIFEST)

required_columns = [
    "fold",
    "held_out_city",
    "loco_role",
    "pilot_id",
    "record_id",
    "city",
    "thermal_sensor",
    "thermal_datetime_utc",
    "array_path",
]

missing_columns = [
    column
    for column in required_columns
    if column not in scenes.columns
]

if missing_columns:
    raise ValueError(
        f"Missing manifest columns: {missing_columns}"
    )

scenes["thermal_datetime_utc"] = pd.to_datetime(
    scenes["thermal_datetime_utc"],
    utc=True,
    errors="raise",
)

scenes["loco_role"] = (
    scenes["loco_role"]
    .astype(str)
    .str.strip()
    .str.lower()
)

train = scenes.query(
    "loco_role == 'train'"
).copy()

validation = scenes.query(
    "loco_role == 'validation'"
).copy()

test = scenes.query(
    "loco_role == 'test'"
).copy()

assert len(train) == 36
assert len(validation) == 6
assert len(test) == 16

assert HELD_OUT_CITY not in set(train["city"])
assert HELD_OUT_CITY not in set(validation["city"])
assert set(test["city"]) == {HELD_OUT_CITY}

train_ids = set(train["pilot_id"])
validation_ids = set(validation["pilot_id"])
test_ids = set(test["pilot_id"])

assert train_ids.isdisjoint(validation_ids)
assert train_ids.isdisjoint(test_ids)
assert validation_ids.isdisjoint(test_ids)

for path in scenes["array_path"]:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Array not found: {path}"
        )

print("\nFold composition:")
print(f"Training scenes:   {len(train)}")
print(f"Validation scenes: {len(validation)}")
print(f"Unseen test scenes:{len(test)}")
print(
    "Training cities:  "
    + ", ".join(sorted(train["city"].unique()))
)

# ------------------------------------------------------------
# 6. Training-only sensor statistics
# ------------------------------------------------------------
# City-specific statistics are prohibited because Abidjan
# is completely unseen during development.

sensor_arrays = {
    "Landsat": [],
    "ECOSTRESS": [],
}

for row in train.itertuples(index=False):

    with np.load(
        row.array_path,
        allow_pickle=False,
    ) as item:

        y_c = item["y_c"].astype("float32")
        valid_mask = (
            item["valid_mask"].astype(bool)
        )

    valid_values = y_c[valid_mask]

    if valid_values.size == 0:
        raise RuntimeError(
            f"No valid target pixels: {row.pilot_id}"
        )

    sensor_arrays[
        row.thermal_sensor
    ].append(valid_values)

sensor_stats = {}

for sensor, arrays in sensor_arrays.items():

    values = np.concatenate(arrays)

    mean_c = float(np.mean(values))
    std_c = float(np.std(values))

    if not np.isfinite(mean_c):
        raise RuntimeError(
            f"Non-finite mean for {sensor}"
        )

    if not np.isfinite(std_c) or std_c <= 0:
        raise RuntimeError(
            f"Invalid standard deviation for {sensor}"
        )

    sensor_stats[sensor] = {
        "mean_c": mean_c,
        "std_c": std_c,
        "valid_training_pixels": int(values.size),
    }

with open(
    OUTPUT_DIR
    / "01_training_only_sensor_statistics.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        sensor_stats,
        file,
        indent=2,
    )

print("\nTraining-only sensor statistics:")
display(pd.DataFrame(sensor_stats).T)

# ------------------------------------------------------------
# 7. Transferable metadata
# ------------------------------------------------------------
def transferable_context(row):

    timestamp = row.thermal_datetime_utc

    day_angle = (
        2
        * np.pi
        * (timestamp.dayofyear - 1)
        / 365.25
    )

    decimal_hour = (
        timestamp.hour
        + timestamp.minute / 60
        + timestamp.second / 3600
    )

    hour_angle = (
        2
        * np.pi
        * decimal_hour
        / 24
    )

    values = [
        float(
            row.thermal_sensor == "ECOSTRESS"
        ),
        np.sin(day_angle),
        np.cos(day_angle),
        np.sin(hour_angle),
        np.cos(hour_angle),
    ]

    # Original architecture included four city channels.
    # They remain present but are set to zero for all folds.
    values.extend([
        0.0,
        0.0,
        0.0,
        0.0,
    ])

    return np.asarray(
        values,
        dtype="float32",
    )[:, None, None]

# ------------------------------------------------------------
# 8. Dataset
# ------------------------------------------------------------
class LocoDataset(Dataset):

    def __init__(
        self,
        frame,
        augment=False,
    ):
        self.frame = frame.reset_index(drop=True)
        self.augment = augment

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):

        row = self.frame.iloc[index]

        with np.load(
            row.array_path,
            allow_pickle=False,
        ) as item:

            x = item["x"].astype("float32")

            y_c = item["y_c"].astype(
                "float32"
            )

            mask = item[
                "valid_mask"
            ].astype("float32")

        if x.shape != (16, 256, 256):
            raise RuntimeError(
                f"Unexpected input shape for "
                f"{row.pilot_id}: {x.shape}"
            )

        if y_c.shape != (256, 256):
            raise RuntimeError(
                f"Unexpected target shape for "
                f"{row.pilot_id}: {y_c.shape}"
            )

        if mask.shape != (256, 256):
            raise RuntimeError(
                f"Unexpected mask shape for "
                f"{row.pilot_id}: {mask.shape}"
            )

        # Match the final validation-selected
        # no-terrain configuration.
        x[TERRAIN_CHANNELS] = 0.0

        context = np.broadcast_to(
            transferable_context(row),
            (9, 256, 256),
        ).copy()

        x = np.concatenate(
            [x, context],
            axis=0,
        )

        stats = sensor_stats[
            row.thermal_sensor
        ]

        y = (
            (
                y_c - stats["mean_c"]
            )
            / stats["std_c"]
        )[None, :, :]

        mask = mask[None, :, :]

        y[:, mask[0] < 0.5] = 0.0

        if self.augment:

            if random.random() < 0.5:
                x = x[:, :, ::-1]
                y = y[:, :, ::-1]
                mask = mask[:, :, ::-1]

            if random.random() < 0.5:
                x = x[:, ::-1, :]
                y = y[:, ::-1, :]
                mask = mask[:, ::-1, :]

            rotation = random.randint(0, 3)

            if rotation:
                x = np.rot90(
                    x,
                    rotation,
                    axes=(1, 2),
                )

                y = np.rot90(
                    y,
                    rotation,
                    axes=(1, 2),
                )

                mask = np.rot90(
                    mask,
                    rotation,
                    axes=(1, 2),
                )

        return (
            torch.from_numpy(
                np.ascontiguousarray(x)
            ),
            torch.from_numpy(
                np.ascontiguousarray(y)
            ),
            torch.from_numpy(
                np.ascontiguousarray(mask)
            ),
            row.pilot_id,
        )

# ------------------------------------------------------------
# 9. Model architecture
# ------------------------------------------------------------
def convolution_block(
    input_channels,
    output_channels,
):

    groups = min(8, output_channels)

    return nn.Sequential(
        nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        ),
        nn.GroupNorm(
            groups,
            output_channels,
        ),
        nn.SiLU(inplace=True),
        nn.Conv2d(
            output_channels,
            output_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        ),
        nn.GroupNorm(
            groups,
            output_channels,
        ),
        nn.SiLU(inplace=True),
    )


class ThermoFusionUNet(nn.Module):

    def __init__(self):
        super().__init__()

        b = BASE_WIDTH

        self.pool = nn.MaxPool2d(2)

        self.e1 = convolution_block(25, b)
        self.e2 = convolution_block(b, 2 * b)
        self.e3 = convolution_block(
            2 * b,
            4 * b,
        )
        self.e4 = convolution_block(
            4 * b,
            8 * b,
        )

        self.bridge = convolution_block(
            8 * b,
            16 * b,
        )

        self.u4 = nn.ConvTranspose2d(
            16 * b,
            8 * b,
            kernel_size=2,
            stride=2,
        )
        self.d4 = convolution_block(
            16 * b,
            8 * b,
        )

        self.u3 = nn.ConvTranspose2d(
            8 * b,
            4 * b,
            kernel_size=2,
            stride=2,
        )
        self.d3 = convolution_block(
            8 * b,
            4 * b,
        )

        self.u2 = nn.ConvTranspose2d(
            4 * b,
            2 * b,
            kernel_size=2,
            stride=2,
        )
        self.d2 = convolution_block(
            4 * b,
            2 * b,
        )

        self.u1 = nn.ConvTranspose2d(
            2 * b,
            b,
            kernel_size=2,
            stride=2,
        )
        self.d1 = convolution_block(
            2 * b,
            b,
        )

        self.output = nn.Conv2d(
            b,
            1,
            kernel_size=1,
        )

    def forward(self, x):

        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))

        z = self.bridge(self.pool(e4))

        z = self.d4(
            torch.cat(
                [self.u4(z), e4],
                dim=1,
            )
        )

        z = self.d3(
            torch.cat(
                [self.u3(z), e3],
                dim=1,
            )
        )

        z = self.d2(
            torch.cat(
                [self.u2(z), e2],
                dim=1,
            )
        )

        z = self.d1(
            torch.cat(
                [self.u1(z), e1],
                dim=1,
            )
        )

        return self.output(z)

# ------------------------------------------------------------
# 10. Loss and metrics
# ------------------------------------------------------------
def masked_huber(
    prediction,
    target,
    mask,
):

    error = prediction - target
    absolute_error = torch.abs(error)

    loss = torch.where(
        absolute_error <= HUBER_DELTA,
        0.5 * error.square(),
        HUBER_DELTA * (
            absolute_error
            - 0.5 * HUBER_DELTA
        ),
    )

    return (
        (loss * mask).sum()
        / mask.sum().clamp_min(1)
    )


def regression_metrics(
    observed,
    predicted,
):

    observed = np.asarray(
        observed,
        dtype="float64",
    )

    predicted = np.asarray(
        predicted,
        dtype="float64",
    )

    error = predicted - observed

    denominator = np.sum(
        (
            observed
            - observed.mean()
        ) ** 2
    )

    return {
        "valid_pixels": int(observed.size),
        "mae_c": float(
            np.mean(np.abs(error))
        ),
        "rmse_c": float(
            np.sqrt(
                np.mean(error ** 2)
            )
        ),
        "bias_c": float(
            np.mean(error)
        ),
        "r2": (
            float(
                1
                - np.sum(error ** 2)
                / denominator
            )
            if denominator > 0
            else np.nan
        ),
    }

# ------------------------------------------------------------
# 11. Data loaders
# ------------------------------------------------------------
train_loader = DataLoader(
    LocoDataset(
        train,
        augment=True,
    ),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=False,
)

validation_loader = DataLoader(
    LocoDataset(
        validation,
        augment=False,
    ),
    batch_size=1,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=False,
)

test_loader = DataLoader(
    LocoDataset(
        test,
        augment=False,
    ),
    batch_size=1,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=False,
)

# ------------------------------------------------------------
# 12. Model and optimizer
# ------------------------------------------------------------
model = ThermoFusionUNet().to(device)

parameter_count = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

scheduler = (
    torch.optim.lr_scheduler
    .ReduceLROnPlateau(
        optimizer,
        factor=0.5,
        patience=5,
        min_lr=MINIMUM_LEARNING_RATE,
    )
)

print(
    f"\nTrainable parameters: "
    f"{parameter_count:,}"
)

# ------------------------------------------------------------
# 13. Validation loss
# ------------------------------------------------------------
def evaluate_loss(model, loader):

    model.eval()

    total_loss = 0.0
    total_pixels = 0.0

    with torch.no_grad():

        for x, y, mask, _ in loader:

            x = x.to(device)
            y = y.to(device)
            mask = mask.to(device)

            prediction = model(x)

            loss = masked_huber(
                prediction,
                y,
                mask,
            )

            valid_pixels = mask.sum().item()

            total_loss += (
                loss.item()
                * valid_pixels
            )

            total_pixels += valid_pixels

    return (
        total_loss
        / max(total_pixels, 1)
    )

# ------------------------------------------------------------
# 14. Resume or initialize training
# ------------------------------------------------------------
history = []

start_epoch = 1
best_epoch = 0
best_validation_loss = np.inf
stale_epochs = 0
elapsed_before_resume = 0.0

if (
    RESUME_IF_AVAILABLE
    and RECOVERY_CHECKPOINT.exists()
):

    print(
        "\nRecovery checkpoint found. "
        "Resuming training..."
    )

    recovery = torch.load(
        RECOVERY_CHECKPOINT,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        recovery["model_state_dict"]
    )

    optimizer.load_state_dict(
        recovery["optimizer_state_dict"]
    )

    scheduler.load_state_dict(
        recovery["scheduler_state_dict"]
    )

    start_epoch = (
        int(recovery["completed_epoch"])
        + 1
    )

    best_epoch = int(
        recovery["best_epoch"]
    )

    best_validation_loss = float(
        recovery["best_validation_loss"]
    )

    stale_epochs = int(
        recovery["stale_epochs"]
    )

    elapsed_before_resume = float(
        recovery.get(
            "elapsed_minutes",
            0.0,
        )
    )

    history = recovery.get(
        "history",
        [],
    )

    random.setstate(
        recovery["python_random_state"]
    )

    np.random.set_state(
        recovery["numpy_random_state"]
    )

    torch.set_rng_state(
        recovery["torch_random_state"]
    )

    print(
        f"Resuming from epoch {start_epoch}"
    )

    print(
        f"Current best epoch: {best_epoch}"
    )

    print(
        "Current best validation loss: "
        f"{best_validation_loss:.6f}"
    )

else:
    print("\nStarting from epoch 1.")

# ------------------------------------------------------------
# 15. CPU training
# ------------------------------------------------------------
training_started = time.time()

for epoch in range(
    start_epoch,
    MAXIMUM_EPOCHS + 1,
):

    epoch_started = time.time()

    model.train()

    total_training_loss = 0.0
    total_training_pixels = 0.0

    for batch_number, (
        x,
        y,
        mask,
        pilot_ids,
    ) in enumerate(
        train_loader,
        start=1,
    ):

        x = x.to(device)
        y = y.to(device)
        mask = mask.to(device)

        optimizer.zero_grad(
            set_to_none=True
        )

        prediction = model(x)

        loss = masked_huber(
            prediction,
            y,
            mask,
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=GRADIENT_CLIP,
        )

        optimizer.step()

        valid_pixels = mask.sum().item()

        total_training_loss += (
            loss.item()
            * valid_pixels
        )

        total_training_pixels += (
            valid_pixels
        )

        if (
            batch_number == 1
            or batch_number
            == len(train_loader)
            or batch_number % 5 == 0
        ):
            print(
                f"Epoch {epoch:02d} | "
                f"batch {batch_number:02d}/"
                f"{len(train_loader):02d} | "
                f"loss={loss.item():.6f}",
                flush=True,
            )

    training_loss = (
        total_training_loss
        / max(
            total_training_pixels,
            1,
        )
    )

    validation_loss = evaluate_loss(
        model,
        validation_loader,
    )

    scheduler.step(validation_loss)

    current_learning_rate = (
        optimizer.param_groups[0]["lr"]
    )

    improved = (
        validation_loss
        < best_validation_loss - 1e-6
    )

    if improved:

        best_validation_loss = (
            validation_loss
        )

        best_epoch = epoch
        stale_epochs = 0

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),
                "fold": FOLD_NAME,
                "held_out_city":
                    HELD_OUT_CITY,
                "epoch": epoch,
                "seed": SEED,
                "sensor_stats":
                    sensor_stats,
                "parameter_count":
                    parameter_count,
                "city_channels_used":
                    False,
                "terrain_channels_used":
                    False,
                "device": "cpu",
            },
            BEST_CHECKPOINT,
        )

    else:
        stale_epochs += 1

    epoch_minutes = (
        time.time() - epoch_started
    ) / 60

    total_elapsed_minutes = (
        elapsed_before_resume
        + (
            time.time()
            - training_started
        ) / 60
    )

    history.append({
        "epoch": epoch,
        "training_huber":
            training_loss,
        "validation_huber":
            validation_loss,
        "learning_rate":
            current_learning_rate,
        "improved": improved,
        "epoch_minutes":
            epoch_minutes,
        "cumulative_minutes":
            total_elapsed_minutes,
    })

    pd.DataFrame(history).to_csv(
        HISTORY_PATH,
        index=False,
    )

    # Save after every completed epoch.
    torch.save(
        {
            "completed_epoch": epoch,
            "model_state_dict":
                model.state_dict(),
            "optimizer_state_dict":
                optimizer.state_dict(),
            "scheduler_state_dict":
                scheduler.state_dict(),
            "best_epoch":
                best_epoch,
            "best_validation_loss":
                best_validation_loss,
            "stale_epochs":
                stale_epochs,
            "elapsed_minutes":
                total_elapsed_minutes,
            "history": history,
            "python_random_state":
                random.getstate(),
            "numpy_random_state":
                np.random.get_state(),
            "torch_random_state":
                torch.get_rng_state(),
        },
        RECOVERY_CHECKPOINT,
    )

    status = (
        "BEST"
        if improved
        else f"stale={stale_epochs}"
    )

    print(
        "\n"
        f"Epoch {epoch:02d}/"
        f"{MAXIMUM_EPOCHS} completed | "
        f"train={training_loss:.6f} | "
        f"validation={validation_loss:.6f} | "
        f"lr={current_learning_rate:.2e} | "
        f"{status} | "
        f"time={epoch_minutes:.2f} min\n",
        flush=True,
    )

    if (
        stale_epochs
        >= EARLY_STOPPING_PATIENCE
    ):
        print(
            "Early stopping triggered "
            f"at epoch {epoch}."
        )
        break

# ------------------------------------------------------------
# 16. Load the best model
# ------------------------------------------------------------
if not BEST_CHECKPOINT.exists():
    raise RuntimeError(
        "Best-model checkpoint was not created."
    )

best_checkpoint = torch.load(
    BEST_CHECKPOINT,
    map_location=device,
    weights_only=False,
)

model.load_state_dict(
    best_checkpoint[
        "model_state_dict"
    ]
)

model.eval()

# ------------------------------------------------------------
# 17. Evaluate each validation and test scene
# ------------------------------------------------------------
def evaluate_scenes(
    frame,
    loader,
    evaluation_split,
):

    result_rows = []

    with torch.no_grad():

        for x, _, mask, pilot_ids in loader:

            pilot_id = pilot_ids[0]

            row = frame.loc[
                frame["pilot_id"]
                == pilot_id
            ].iloc[0]

            with np.load(
                row.array_path,
                allow_pickle=False,
            ) as item:

                observed_c = item[
                    "y_c"
                ].astype("float32")

            prediction_z = (
                model(x.to(device))
                .cpu()
                .numpy()[0, 0]
            )

            statistics = sensor_stats[
                row.thermal_sensor
            ]

            predicted_c = (
                prediction_z
                * statistics["std_c"]
                + statistics["mean_c"]
            )

            valid = (
                mask.numpy()[0, 0]
                > 0.5
            )

            metrics = regression_metrics(
                observed_c[valid],
                predicted_c[valid],
            )

            result_rows.append({
                "fold": FOLD_NAME,
                "evaluation_split":
                    evaluation_split,
                "pilot_id": pilot_id,
                "record_id":
                    row.record_id,
                "city": row.city,
                "thermal_sensor":
                    row.thermal_sensor,
                "thermal_datetime_utc":
                    row.thermal_datetime_utc,
                **metrics,
            })

    return result_rows


scene_rows = []

scene_rows.extend(
    evaluate_scenes(
        validation,
        validation_loader,
        "validation",
    )
)

scene_rows.extend(
    evaluate_scenes(
        test,
        test_loader,
        "unseen_city_test",
    )
)

scene_metrics = pd.DataFrame(
    scene_rows
)

scene_metrics.to_csv(
    OUTPUT_DIR
    / "03_scene_metrics.csv",
    index=False,
)

# ------------------------------------------------------------
# 18. Summaries
# ------------------------------------------------------------
sensor_summary = (
    scene_metrics
    .groupby(
        [
            "evaluation_split",
            "thermal_sensor",
        ],
        observed=True,
    )
    .agg(
        scenes=(
            "pilot_id",
            "count",
        ),
        scene_macro_mae_c=(
            "mae_c",
            "mean",
        ),
        scene_macro_rmse_c=(
            "rmse_c",
            "mean",
        ),
        median_scene_rmse_c=(
            "rmse_c",
            "median",
        ),
        scene_macro_bias_c=(
            "bias_c",
            "mean",
        ),
        mean_scene_r2=(
            "r2",
            "mean",
        ),
    )
    .reset_index()
)

overall_summary = (
    scene_metrics
    .groupby(
        "evaluation_split",
        observed=True,
    )
    .agg(
        scenes=(
            "pilot_id",
            "count",
        ),
        scene_macro_mae_c=(
            "mae_c",
            "mean",
        ),
        scene_macro_rmse_c=(
            "rmse_c",
            "mean",
        ),
        median_scene_rmse_c=(
            "rmse_c",
            "median",
        ),
        scene_macro_bias_c=(
            "bias_c",
            "mean",
        ),
        mean_scene_r2=(
            "r2",
            "mean",
        ),
    )
    .reset_index()
)

sensor_summary.to_csv(
    OUTPUT_DIR
    / "04_sensor_summary.csv",
    index=False,
)

overall_summary.to_csv(
    OUTPUT_DIR
    / "05_overall_summary.csv",
    index=False,
)

print("\n" + "=" * 90)
print("OVERALL SCENE-MACRO RESULTS")
print("=" * 90)
display(overall_summary)

print("\nSensor-specific results:")
display(sensor_summary)

print("\nUnseen Abidjan scene results:")

display(
    scene_metrics.query(
        "evaluation_split "
        "== 'unseen_city_test'"
    )[
        [
            "pilot_id",
            "thermal_sensor",
            "mae_c",
            "rmse_c",
            "bias_c",
            "r2",
        ]
    ].sort_values(
        [
            "thermal_sensor",
            "pilot_id",
        ]
    )
)

# ------------------------------------------------------------
# 19. Final verification
# ------------------------------------------------------------
test_results = scene_metrics.query(
    "evaluation_split "
    "== 'unseen_city_test'"
)

execution_pass = (
    BEST_CHECKPOINT.exists()
    and best_epoch > 0
    and len(test_results) == 16
    and test_results[
        "pilot_id"
    ].nunique() == 16
    and np.isfinite(
        test_results[
            [
                "mae_c",
                "rmse_c",
                "bias_c",
                "r2",
            ]
        ].to_numpy()
    ).all()
)

total_minutes = (
    elapsed_before_resume
    + (
        time.time()
        - training_started
    ) / 60
)

verdict_lines = [
    "THERMOFUSION LEAVE-ONE-CITY-OUT CPU PILOT",
    f"Held-out city: {HELD_OUT_CITY}",
    f"Training scenes: {len(train)}",
    f"Validation scenes: {len(validation)}",
    f"Unseen-city test scenes: {len(test)}",
    f"Best epoch: {best_epoch}",
    (
        "Best validation Huber loss: "
        f"{best_validation_loss:.6f}"
    ),
    (
        "Total CPU time (minutes): "
        f"{total_minutes:.2f}"
    ),
    (
        "Trainable parameters: "
        f"{parameter_count}"
    ),
    "Held-out-city targets used for training: NO",
    "Held-out-city targets used for validation: NO",
    "City-identity metadata used: NO",
    "Residual statistics: training-only sensor-specific",
    "Terrain channels used: NO",
    (
        "Overall CPU pilot: PASS"
        if execution_pass
        else "Overall CPU pilot: REVIEW REQUIRED"
    ),
]

verdict_text = "\n".join(
    verdict_lines
)

(
    OUTPUT_DIR
    / "06_cpu_pilot_verdict.txt"
).write_text(
    verdict_text,
    encoding="utf-8",
)

configuration = {
    "fold": FOLD_NAME,
    "held_out_city":
        HELD_OUT_CITY,
    "device": "cpu",
    "cpu_threads": cpu_threads,
    "seed": SEED,
    "maximum_epochs":
        MAXIMUM_EPOCHS,
    "early_stopping_patience":
        EARLY_STOPPING_PATIENCE,
    "batch_size": BATCH_SIZE,
    "base_width": BASE_WIDTH,
    "learning_rate":
        LEARNING_RATE,
    "weight_decay":
        WEIGHT_DECAY,
    "huber_delta":
        HUBER_DELTA,
    "best_epoch": best_epoch,
    "total_minutes":
        total_minutes,
    "parameter_count":
        parameter_count,
    "sensor_stats":
        sensor_stats,
    "city_channels_used":
        False,
    "terrain_channels_used":
        False,
}

with open(
    OUTPUT_DIR
    / "07_cpu_configuration.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        configuration,
        file,
        indent=2,
    )

print("\n" + "=" * 90)
print(verdict_text)
print("=" * 90)

print(
    f"\nOutput folder:\n{OUTPUT_DIR}"
)

print(
    "\nRecovery checkpoint:\n"
    f"{RECOVERY_CHECKPOINT}"
)
