# =====================================================================
# ThermoFusion reproducibility code
# Source notebook cell(s): 45
# Audit status: KEEP
# Extracted without scientific-logic modification.
# Replace local Google Drive paths as described in README.md when needed.
# =====================================================================

# ============================================================
# THERMOFUSION STAGE 21
# Complete four-fold leave-one-city-out experiment on CPU
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
# 1. Configuration
# ------------------------------------------------------------
DRIVE_ROOT = Path("/content/drive/MyDrive")

STAGE20 = (
    DRIVE_ROOT
    / "ThermoFusion_Stage20_LeaveOneCityOut"
)

FOLD_DIR = STAGE20 / "fold_manifests"
COMBINED_DIR = STAGE20 / "combined_four_city_results"

COMBINED_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FOLDS = {
    "Abidjan": "holdout_abidjan",
    "Accra": "holdout_accra",
    "Freetown": "holdout_freetown",
    "Lagos": "holdout_lagos",
}

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
TERRAIN_CHANNELS = [14, 15]
NUM_WORKERS = 0
BOOTSTRAP_REPLICATES = 20000

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

print("=" * 100)
print("THERMOFUSION FOUR-CITY LEAVE-ONE-CITY-OUT")
print("=" * 100)
print(f"Device: {device}")
print(f"CPU threads: {cpu_threads}")
print(f"Folds: {list(FOLDS)}")
print(f"Maximum epochs per fold: {MAXIMUM_EPOCHS}")

# ------------------------------------------------------------
# 2. Reproducibility
# ------------------------------------------------------------
def reset_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


reset_seed()

# ------------------------------------------------------------
# 3. Model
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
            3,
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
            3,
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
        self.e3 = convolution_block(2 * b, 4 * b)
        self.e4 = convolution_block(4 * b, 8 * b)

        self.bridge = convolution_block(
            8 * b,
            16 * b,
        )

        self.u4 = nn.ConvTranspose2d(
            16 * b,
            8 * b,
            2,
            2,
        )
        self.d4 = convolution_block(
            16 * b,
            8 * b,
        )

        self.u3 = nn.ConvTranspose2d(
            8 * b,
            4 * b,
            2,
            2,
        )
        self.d3 = convolution_block(
            8 * b,
            4 * b,
        )

        self.u2 = nn.ConvTranspose2d(
            4 * b,
            2 * b,
            2,
            2,
        )
        self.d2 = convolution_block(
            4 * b,
            2 * b,
        )

        self.u1 = nn.ConvTranspose2d(
            2 * b,
            b,
            2,
            2,
        )
        self.d1 = convolution_block(
            2 * b,
            b,
        )

        self.output = nn.Conv2d(
            b,
            1,
            1,
        )

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))

        z = self.bridge(self.pool(e4))

        z = self.d4(
            torch.cat([self.u4(z), e4], 1)
        )
        z = self.d3(
            torch.cat([self.u3(z), e3], 1)
        )
        z = self.d2(
            torch.cat([self.u2(z), e2], 1)
        )
        z = self.d1(
            torch.cat([self.u1(z), e1], 1)
        )

        return self.output(z)

# ------------------------------------------------------------
# 4. Statistics, context and dataset
# ------------------------------------------------------------
def calculate_sensor_stats(training_frame):
    collected = {
        "Landsat": [],
        "ECOSTRESS": [],
    }

    for row in training_frame.itertuples(index=False):
        with np.load(
            row.array_path,
            allow_pickle=False,
        ) as item:
            target = item["y_c"].astype("float32")
            mask = item["valid_mask"].astype(bool)

        collected[row.thermal_sensor].append(
            target[mask]
        )

    statistics = {}

    for sensor, arrays in collected.items():
        values = np.concatenate(arrays)

        statistics[sensor] = {
            "mean_c": float(values.mean()),
            "std_c": float(values.std()),
            "valid_training_pixels": int(
                values.size
            ),
        }

        assert statistics[sensor]["std_c"] > 0

    return statistics


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
        2 * np.pi * decimal_hour / 24
    )

    values = [
        float(
            row.thermal_sensor == "ECOSTRESS"
        ),
        np.sin(day_angle),
        np.cos(day_angle),
        np.sin(hour_angle),
        np.cos(hour_angle),
        0.0,
        0.0,
        0.0,
        0.0,
    ]

    return np.asarray(
        values,
        dtype="float32",
    )[:, None, None]


class LocoDataset(Dataset):

    def __init__(
        self,
        frame,
        sensor_stats,
        augment=False,
    ):
        self.frame = frame.reset_index(drop=True)
        self.sensor_stats = sensor_stats
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
            target_c = item["y_c"].astype(
                "float32"
            )
            mask = item["valid_mask"].astype(
                "float32"
            )

        if x.shape != (16, 256, 256):
            raise RuntimeError(
                f"Unexpected input shape: {row.pilot_id}"
            )

        x[TERRAIN_CHANNELS] = 0.0

        context = np.broadcast_to(
            transferable_context(row),
            (9, 256, 256),
        ).copy()

        x = np.concatenate(
            [x, context],
            axis=0,
        )

        statistics = self.sensor_stats[
            row.thermal_sensor
        ]

        target_z = (
            (
                target_c
                - statistics["mean_c"]
            )
            / statistics["std_c"]
        )[None, :, :]

        mask = mask[None, :, :]
        target_z[:, mask[0] < 0.5] = 0.0

        if self.augment:
            if random.random() < 0.5:
                x = x[:, :, ::-1]
                target_z = target_z[:, :, ::-1]
                mask = mask[:, :, ::-1]

            if random.random() < 0.5:
                x = x[:, ::-1, :]
                target_z = target_z[:, ::-1, :]
                mask = mask[:, ::-1, :]

            rotation = random.randint(0, 3)

            if rotation:
                x = np.rot90(
                    x,
                    rotation,
                    axes=(1, 2),
                )
                target_z = np.rot90(
                    target_z,
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
                np.ascontiguousarray(target_z)
            ),
            torch.from_numpy(
                np.ascontiguousarray(mask)
            ),
            row.pilot_id,
        )

# ------------------------------------------------------------
# 5. Loss and metrics
# ------------------------------------------------------------
def masked_huber(
    prediction,
    target,
    mask,
):
    error = prediction - target
    absolute = torch.abs(error)

    loss = torch.where(
        absolute <= HUBER_DELTA,
        0.5 * error.square(),
        HUBER_DELTA * (
            absolute - 0.5 * HUBER_DELTA
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
        (observed - observed.mean()) ** 2
    )

    return {
        "valid_pixels": int(observed.size),
        "mae_c": float(
            np.mean(np.abs(error))
        ),
        "rmse_c": float(
            np.sqrt(np.mean(error ** 2))
        ),
        "bias_c": float(np.mean(error)),
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


def evaluate_loss(model, loader):
    model.eval()

    total = 0.0
    pixels = 0.0

    with torch.no_grad():
        for x, y, mask, _ in loader:
            x = x.to(device)
            y = y.to(device)
            mask = mask.to(device)

            loss = masked_huber(
                model(x),
                y,
                mask,
            )

            count = mask.sum().item()
            total += loss.item() * count
            pixels += count

    return total / max(pixels, 1)

# ------------------------------------------------------------
# 6. Evaluate model and sensor-climatology benchmark
# ------------------------------------------------------------
def evaluate_test_scenes(
    model,
    test_frame,
    test_loader,
    sensor_stats,
    fold_name,
    held_out_city,
):
    rows = []

    model.eval()

    with torch.no_grad():
        for x, _, mask, pilot_ids in test_loader:
            pilot_id = pilot_ids[0]

            row = test_frame.loc[
                test_frame["pilot_id"] == pilot_id
            ].iloc[0]

            with np.load(
                row.array_path,
                allow_pickle=False,
            ) as item:
                observed_c = item["y_c"].astype(
                    "float32"
                )

            valid = mask.numpy()[0, 0] > 0.5
            observed_valid = observed_c[valid]

            statistics = sensor_stats[
                row.thermal_sensor
            ]

            prediction_z = (
                model(x.to(device))
                .cpu()
                .numpy()[0, 0]
            )

            model_prediction = (
                prediction_z
                * statistics["std_c"]
                + statistics["mean_c"]
            )

            model_metrics = regression_metrics(
                observed_valid,
                model_prediction[valid],
            )

            climatology_prediction = np.full(
                observed_valid.shape,
                statistics["mean_c"],
                dtype="float32",
            )

            climatology_metrics = regression_metrics(
                observed_valid,
                climatology_prediction,
            )

            common = {
                "fold": fold_name,
                "held_out_city": held_out_city,
                "pilot_id": pilot_id,
                "record_id": row.record_id,
                "city": row.city,
                "thermal_sensor":
                    row.thermal_sensor,
                "thermal_datetime_utc":
                    row.thermal_datetime_utc,
            }

            rows.append({
                **common,
                "model": "ThermoFusion",
                **model_metrics,
            })

            rows.append({
                **common,
                "model":
                    "Training-only sensor climatology",
                **climatology_metrics,
            })

    return pd.DataFrame(rows)

# ------------------------------------------------------------
# 7. Run each geographical fold
# ------------------------------------------------------------
all_fold_results = []
fold_execution_rows = []

experiment_started = time.time()

for fold_index, (
    held_out_city,
    fold_name,
) in enumerate(FOLDS.items(), start=1):

    print("\n" + "=" * 100)
    print(
        f"FOLD {fold_index}/4: "
        f"{held_out_city.upper()} HELD OUT"
    )
    print("=" * 100)

    fold_seed = SEED
    reset_seed(fold_seed)

    manifest_path = (
        FOLD_DIR / f"{fold_name}.csv"
    )

    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    scenes = pd.read_csv(manifest_path)

    scenes["thermal_datetime_utc"] = pd.to_datetime(
        scenes["thermal_datetime_utc"],
        utc=True,
        errors="raise",
    )

    scenes["loco_role"] = (
        scenes["loco_role"]
        .astype(str)
        .str.lower()
        .str.strip()
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
    assert held_out_city not in set(train["city"])
    assert held_out_city not in set(
        validation["city"]
    )
    assert set(test["city"]) == {
        held_out_city
    }

    fold_output = (
        STAGE20
        / f"pilot_holdout_{held_out_city.lower()}_cpu"
    )

    checkpoint_dir = (
        fold_output / "checkpoints"
    )

    fold_output.mkdir(
        parents=True,
        exist_ok=True,
    )
    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_checkpoint = (
        checkpoint_dir
        / f"holdout_{held_out_city.lower()}_best_cpu.pt"
    )

    recovery_checkpoint = (
        checkpoint_dir
        / f"holdout_{held_out_city.lower()}_recovery_cpu.pt"
    )

    history_path = (
        fold_output
        / "02_training_history.csv"
    )

    sensor_stats = calculate_sensor_stats(
        train
    )

    with open(
        fold_output
        / "01_training_only_sensor_statistics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            sensor_stats,
            file,
            indent=2,
        )

    train_loader = DataLoader(
        LocoDataset(
            train,
            sensor_stats,
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
            sensor_stats,
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
            sensor_stats,
            augment=False,
        ),
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

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

    history = []
    start_epoch = 1
    best_epoch = 0
    best_validation_loss = np.inf
    stale_epochs = 0
    elapsed_before_resume = 0.0

    # Use an existing best checkpoint where training
    # has already completed, including Abidjan.
    skip_training = False

    if best_checkpoint.exists():
        saved_best = torch.load(
            best_checkpoint,
            map_location=device,
            weights_only=False,
        )

        existing_epoch = int(
            saved_best.get("epoch", 0)
        )

        if recovery_checkpoint.exists():
            recovery = torch.load(
                recovery_checkpoint,
                map_location=device,
                weights_only=False,
            )

            completed_epoch = int(
                recovery.get(
                    "completed_epoch",
                    0,
                )
            )

            stale_epochs = int(
                recovery.get(
                    "stale_epochs",
                    0,
                )
            )

            if (
                completed_epoch
                >= MAXIMUM_EPOCHS
                or stale_epochs
                >= EARLY_STOPPING_PATIENCE
            ):
                skip_training = True
                best_epoch = existing_epoch

                best_validation_loss = float(
                    recovery.get(
                        "best_validation_loss",
                        np.nan,
                    )
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

                print(
                    "Completed checkpoint found. "
                    "Training will be skipped."
                )

    if not skip_training:

        if recovery_checkpoint.exists():
            recovery = torch.load(
                recovery_checkpoint,
                map_location=device,
                weights_only=False,
            )

            model.load_state_dict(
                recovery["model_state_dict"]
            )

            optimizer.load_state_dict(
                recovery[
                    "optimizer_state_dict"
                ]
            )

            scheduler.load_state_dict(
                recovery[
                    "scheduler_state_dict"
                ]
            )

            start_epoch = (
                int(
                    recovery[
                        "completed_epoch"
                    ]
                )
                + 1
            )

            best_epoch = int(
                recovery["best_epoch"]
            )

            best_validation_loss = float(
                recovery[
                    "best_validation_loss"
                ]
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
                recovery[
                    "python_random_state"
                ]
            )
            np.random.set_state(
                recovery[
                    "numpy_random_state"
                ]
            )
            torch.set_rng_state(
                recovery[
                    "torch_random_state"
                ]
            )

            print(
                f"Resuming at epoch {start_epoch}."
            )

        fold_started = time.time()

        for epoch in range(
            start_epoch,
            MAXIMUM_EPOCHS + 1,
        ):
            epoch_started = time.time()
            model.train()

            running_loss = 0.0
            running_pixels = 0.0

            for batch_number, (
                x,
                y,
                mask,
                _,
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
                    GRADIENT_CLIP,
                )

                optimizer.step()

                pixels = mask.sum().item()
                running_loss += (
                    loss.item() * pixels
                )
                running_pixels += pixels

                if (
                    batch_number == 1
                    or batch_number
                    == len(train_loader)
                    or batch_number % 6 == 0
                ):
                    print(
                        f"{held_out_city} | "
                        f"epoch {epoch:02d} | "
                        f"batch {batch_number:02d}/"
                        f"{len(train_loader):02d} | "
                        f"loss={loss.item():.5f}",
                        flush=True,
                    )

            training_loss = (
                running_loss
                / max(running_pixels, 1)
            )

            validation_loss = evaluate_loss(
                model,
                validation_loader,
            )

            scheduler.step(validation_loss)

            current_lr = (
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
                        "fold": fold_name,
                        "held_out_city":
                            held_out_city,
                        "epoch": epoch,
                        "seed": fold_seed,
                        "sensor_stats":
                            sensor_stats,
                        "parameter_count":
                            parameter_count,
                        "city_channels_used":
                            False,
                        "terrain_channels_used":
                            False,
                    },
                    best_checkpoint,
                )

            else:
                stale_epochs += 1

            epoch_minutes = (
                time.time() - epoch_started
            ) / 60

            total_fold_minutes = (
                elapsed_before_resume
                + (
                    time.time() - fold_started
                ) / 60
            )

            history.append({
                "fold": fold_name,
                "held_out_city":
                    held_out_city,
                "epoch": epoch,
                "training_huber":
                    training_loss,
                "validation_huber":
                    validation_loss,
                "learning_rate":
                    current_lr,
                "improved": improved,
                "epoch_minutes":
                    epoch_minutes,
                "cumulative_minutes":
                    total_fold_minutes,
            })

            pd.DataFrame(history).to_csv(
                history_path,
                index=False,
            )

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
                        total_fold_minutes,
                    "history": history,
                    "python_random_state":
                        random.getstate(),
                    "numpy_random_state":
                        np.random.get_state(),
                    "torch_random_state":
                        torch.get_rng_state(),
                },
                recovery_checkpoint,
            )

            state = (
                "BEST"
                if improved
                else f"stale={stale_epochs}"
            )

            print(
                f"{held_out_city} | "
                f"epoch {epoch:02d} completed | "
                f"train={training_loss:.5f} | "
                f"validation={validation_loss:.5f} | "
                f"{state} | "
                f"{epoch_minutes:.2f} min\n",
                flush=True,
            )

            if (
                stale_epochs
                >= EARLY_STOPPING_PATIENCE
            ):
                print(
                    f"{held_out_city}: "
                    f"early stopping at epoch {epoch}."
                )
                break

    if not best_checkpoint.exists():
        raise RuntimeError(
            f"No best checkpoint for {held_out_city}"
        )

    saved_best = torch.load(
        best_checkpoint,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        saved_best["model_state_dict"]
    )

    model.eval()

    fold_results = evaluate_test_scenes(
        model=model,
        test_frame=test,
        test_loader=test_loader,
        sensor_stats=sensor_stats,
        fold_name=fold_name,
        held_out_city=held_out_city,
    )

    fold_results.to_csv(
        fold_output
        / "03_scene_metrics_with_baseline.csv",
        index=False,
    )

    all_fold_results.append(fold_results)

    model_summary = (
        fold_results
        .groupby(
            ["model", "thermal_sensor"],
            observed=True,
        )
        .agg(
            scenes=("pilot_id", "count"),
            scene_macro_mae_c=(
                "mae_c",
                "mean",
            ),
            scene_macro_rmse_c=(
                "rmse_c",
                "mean",
            ),
            scene_macro_bias_c=(
                "bias_c",
                "mean",
            ),
        )
        .reset_index()
    )

    model_summary.to_csv(
        fold_output
        / "04_model_and_baseline_summary.csv",
        index=False,
    )

    print(
        f"\n{held_out_city} test summary:"
    )
    display(model_summary)

    fold_execution_rows.append({
        "fold": fold_name,
        "held_out_city":
            held_out_city,
        "best_epoch": int(
            saved_best.get(
                "epoch",
                best_epoch,
            )
        ),
        "best_validation_huber":
            best_validation_loss,
        "train_scenes": len(train),
        "validation_scenes":
            len(validation),
        "test_scenes": len(test),
        "status": "PASS",
    })

    del model
    del optimizer
    del scheduler

# ------------------------------------------------------------
# 8. Combine the four unseen-city folds
# ------------------------------------------------------------
combined = pd.concat(
    all_fold_results,
    ignore_index=True,
)

expected_rows = (
    4       # held-out cities
    * 16    # scenes per city
    * 2     # model and climatology
)

assert len(combined) == expected_rows
assert combined["pilot_id"].nunique() == 64

for model_name in combined["model"].unique():
    model_records = combined.query(
        "model == @model_name"
    )

    assert len(model_records) == 64
    assert model_records["pilot_id"].nunique() == 64

combined.to_csv(
    COMBINED_DIR
    / "01_all_unseen_city_scene_metrics.csv",
    index=False,
)

# ------------------------------------------------------------
# 9. City, sensor and overall summaries
# ------------------------------------------------------------
city_summary = (
    combined
    .groupby(
        ["model", "held_out_city"],
        observed=True,
    )
    .agg(
        scenes=("pilot_id", "count"),
        scene_macro_mae_c=("mae_c", "mean"),
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
    )
    .reset_index()
)

sensor_summary = (
    combined
    .groupby(
        ["model", "thermal_sensor"],
        observed=True,
    )
    .agg(
        scenes=("pilot_id", "count"),
        scene_macro_mae_c=("mae_c", "mean"),
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
    )
    .reset_index()
)

overall_summary = (
    combined
    .groupby("model", observed=True)
    .agg(
        scenes=("pilot_id", "count"),
        scene_macro_mae_c=("mae_c", "mean"),
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
    )
    .reset_index()
)

city_summary.to_csv(
    COMBINED_DIR
    / "02_city_summary.csv",
    index=False,
)

sensor_summary.to_csv(
    COMBINED_DIR
    / "03_sensor_summary.csv",
    index=False,
)

overall_summary.to_csv(
    COMBINED_DIR
    / "04_overall_summary.csv",
    index=False,
)

# ------------------------------------------------------------
# 10. Paired scene-bootstrap inference
# ------------------------------------------------------------
wide = combined.pivot(
    index="pilot_id",
    columns="model",
    values="rmse_c",
)

wide["delta_rmse_c"] = (
    wide["ThermoFusion"]
    - wide[
        "Training-only sensor climatology"
    ]
)

observed_delta = float(
    wide["delta_rmse_c"].mean()
)

rng = np.random.default_rng(SEED)

delta_values = wide[
    "delta_rmse_c"
].to_numpy()

bootstrap_deltas = np.empty(
    BOOTSTRAP_REPLICATES,
    dtype="float64",
)

for replicate in range(
    BOOTSTRAP_REPLICATES
):
    sample = rng.choice(
        delta_values,
        size=len(delta_values),
        replace=True,
    )

    bootstrap_deltas[replicate] = (
        sample.mean()
    )

bootstrap_result = pd.DataFrame([{
    "comparison":
        "ThermoFusion minus sensor climatology",
    "scenes": len(delta_values),
    "mean_delta_rmse_c": observed_delta,
    "ci_lower_95_c": float(
        np.quantile(
            bootstrap_deltas,
            0.025,
        )
    ),
    "ci_upper_95_c": float(
        np.quantile(
            bootstrap_deltas,
            0.975,
        )
    ),
    "probability_thermofusion_better":
        float(
            np.mean(
                bootstrap_deltas < 0
            )
        ),
    "bootstrap_replicates":
        BOOTSTRAP_REPLICATES,
}])

bootstrap_result.to_csv(
    COMBINED_DIR
    / "05_paired_scene_bootstrap.csv",
    index=False,
)

pd.DataFrame({
    "bootstrap_delta_rmse_c":
        bootstrap_deltas
}).to_csv(
    COMBINED_DIR
    / "06_bootstrap_distribution.csv",
    index=False,
)

execution_summary = pd.DataFrame(
    fold_execution_rows
)

execution_summary.to_csv(
    COMBINED_DIR
    / "07_fold_execution_summary.csv",
    index=False,
)

# ------------------------------------------------------------
# 11. Final verdict
# ------------------------------------------------------------
execution_pass = (
    len(combined) == 128
    and combined["held_out_city"].nunique() == 4
    and combined["pilot_id"].nunique() == 64
    and np.isfinite(
        combined[
            [
                "mae_c",
                "rmse_c",
                "bias_c",
            ]
        ].to_numpy()
    ).all()
)

elapsed_minutes = (
    time.time() - experiment_started
) / 60

verdict = [
    "THERMOFUSION FOUR-CITY LEAVE-ONE-CITY-OUT",
    "Geographical folds completed: 4/4",
    "Unique unseen-city test scenes: 64/64",
    "Models evaluated: ThermoFusion and training-only sensor climatology",
    "Held-out-city observations used for training: NO",
    "Held-out-city observations used for validation: NO",
    "City-identity metadata used: NO",
    "Residual statistics: training-only sensor-specific",
    "Terrain channels used: NO",
    f"Bootstrap replicates: {BOOTSTRAP_REPLICATES}",
    f"Current-session elapsed minutes: {elapsed_minutes:.2f}",
    (
        "Overall Stage 21 execution: PASS"
        if execution_pass
        else
        "Overall Stage 21 execution: REVIEW REQUIRED"
    ),
]

verdict_text = "\n".join(verdict)

(
    COMBINED_DIR
    / "08_stage21_verdict.txt"
).write_text(
    verdict_text,
    encoding="utf-8",
)

print("\n" + "=" * 100)
print("FOUR-CITY OVERALL SUMMARY")
print("=" * 100)
display(overall_summary)

print("\nResults by held-out city:")
display(city_summary)

print("\nResults by sensor:")
display(sensor_summary)

print("\nPaired bootstrap comparison:")
display(bootstrap_result)

print("\nFold execution:")
display(execution_summary)

print("\n" + verdict_text)
print(f"\nResults saved to:\n{COMBINED_DIR}")
