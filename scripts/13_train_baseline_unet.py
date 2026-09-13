# =====================================================================
# ThermoFusion reproducibility code
# Source notebook cell(s): 27
# Audit status: KEEP
# Extracted without scientific-logic modification.
# Replace local Google Drive paths as described in README.md when needed.
# =====================================================================

# ThermoFusion Stage 13: train and evaluate a leakage-safe pilot U-Net baseline.
# Run in one Google Colab cell after Stage 12 reports an overall PASS.

from google.colab import drive
from pathlib import Path
import json
import random
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "torch"])
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader


drive.mount("/content/drive")

MY_DRIVE = Path("/content/drive/MyDrive")
INPUT_DIR = MY_DRIVE / "ThermoFusion_Stage12_ModelReady_Pilot"
ARRAY_DIR = INPUT_DIR / "arrays"
MANIFEST_PATH = INPUT_DIR / "04_model_ready_array_manifest.csv"
SPLIT_PATH = INPUT_DIR / "01_scene_split_manifest.csv"
STAGE12_VERDICT = INPUT_DIR / "07_model_ready_verdict.txt"
TARGET_STATS_PATH = INPUT_DIR / "03_target_normalization.json"
OUTPUT_DIR = MY_DRIVE / "ThermoFusion_Stage13_Pilot_Baseline"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 20260908
EXPECTED_SCENES = 64
EXPECTED_SPLITS = {"train": 48, "validation": 8, "temporal_test": 8}
EPOCHS = 60
PATIENCE = 12
BATCH_SIZE = 2
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-5
BASE_FILTERS = 24
NUM_WORKERS = 2
MIN_VALID_PIXELS_PER_BATCH = 1


def set_determinism(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_determinism(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Compute device: {device}")

for required in [ARRAY_DIR, MANIFEST_PATH, SPLIT_PATH, STAGE12_VERDICT, TARGET_STATS_PATH]:
    if not required.exists():
        raise FileNotFoundError(f"Required Stage 12 input not found: {required}")
if "Overall model-ready verdict: PASS" not in STAGE12_VERDICT.read_text(encoding="utf-8"):
    raise RuntimeError("Stage 12 did not record an overall PASS.")

array_manifest = pd.read_csv(MANIFEST_PATH)
split_manifest = pd.read_csv(SPLIT_PATH)
with TARGET_STATS_PATH.open("r", encoding="utf-8") as handle:
    target_stats = json.load(handle)
target_mean = float(target_stats["mean_c"])
target_std = float(target_stats["std_c"])

if len(array_manifest) != EXPECTED_SCENES or array_manifest["record_id"].nunique() != EXPECTED_SCENES:
    raise RuntimeError("Stage 12 array manifest is not a unique 64-scene table.")
if array_manifest["model_split"].value_counts().to_dict() != EXPECTED_SPLITS:
    raise RuntimeError("Stage 12 split counts are not 48 train, 8 validation and 8 temporal test.")
if array_manifest["pilot_id"].duplicated().any():
    raise RuntimeError("Duplicate pilot IDs were found in the Stage 12 array manifest.")

# Recover authoritative city/sensor metadata while retaining the saved array paths.
metadata_columns = [
    "pilot_id", "record_id", "city", "thermal_sensor",
    "thermal_datetime_utc", "model_split",
]
metadata = split_manifest[metadata_columns].copy()
scenes = array_manifest.drop(
    columns=[column for column in ["city", "thermal_sensor", "thermal_datetime_utc"]
             if column in array_manifest.columns]
).merge(
    metadata,
    on=["pilot_id", "record_id", "model_split"],
    how="left",
    validate="one_to_one",
)
if scenes[["city", "thermal_sensor", "thermal_datetime_utc"]].isna().any().any():
    raise RuntimeError("Scene metadata could not be joined cleanly.")


class SceneDataset(Dataset):
    def __init__(self, frame, augment=False):
        self.frame = frame.reset_index(drop=True)
        self.augment = augment

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        path = Path(row.array_path)
        if not path.exists():
            raise FileNotFoundError(f"Model-ready array not found: {path}")
        with np.load(path, allow_pickle=False) as item:
            x = item["x"].astype("float32")
            y = item["y_standardized"].astype("float32")[None, :, :]
            mask = item["valid_mask"].astype("float32")[None, :, :]
        if x.shape != (16, 256, 256) or y.shape != (1, 256, 256) or mask.shape != (1, 256, 256):
            raise RuntimeError(f"Unexpected array shape for {row.pilot_id}.")
        if mask.sum() == 0:
            raise RuntimeError(f"No valid target pixels for {row.pilot_id}.")
        if self.augment:
            if random.random() < 0.5:
                x, y, mask = x[:, :, ::-1], y[:, :, ::-1], mask[:, :, ::-1]
            if random.random() < 0.5:
                x, y, mask = x[:, ::-1, :], y[:, ::-1, :], mask[:, ::-1, :]
            rotations = random.randint(0, 3)
            if rotations:
                x = np.rot90(x, rotations, axes=(1, 2))
                y = np.rot90(y, rotations, axes=(1, 2))
                mask = np.rot90(mask, rotations, axes=(1, 2))
        return (
            torch.from_numpy(np.ascontiguousarray(x)),
            torch.from_numpy(np.ascontiguousarray(y)),
            torch.from_numpy(np.ascontiguousarray(mask)),
            row.pilot_id,
        )


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, in_channels=16, base=24):
        super().__init__()
        self.enc1 = conv_block(in_channels, base)
        self.enc2 = conv_block(base, base * 2)
        self.enc3 = conv_block(base * 2, base * 4)
        self.enc4 = conv_block(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.bridge = conv_block(base * 8, base * 16)
        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.dec4 = conv_block(base * 16, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = conv_block(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = conv_block(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = conv_block(base * 2, base)
        self.output = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        bridge = self.bridge(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(bridge), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.output(d1)


def masked_mse(prediction, target, mask):
    denominator = mask.sum()
    if denominator < MIN_VALID_PIXELS_PER_BATCH:
        raise RuntimeError("A batch contained no valid target pixels.")
    return (((prediction - target) ** 2) * mask).sum() / denominator


train_frame = scenes.loc[scenes.model_split.eq("train")].copy()
validation_frame = scenes.loc[scenes.model_split.eq("validation")].copy()
test_frame = scenes.loc[scenes.model_split.eq("temporal_test")].copy()

train_loader = DataLoader(
    SceneDataset(train_frame, augment=True), batch_size=BATCH_SIZE,
    shuffle=True, num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available(),
)
validation_loader = DataLoader(
    SceneDataset(validation_frame), batch_size=1, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available(),
)
test_loader = DataLoader(
    SceneDataset(test_frame), batch_size=1, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available(),
)

model = UNet(base=BASE_FILTERS).to(device)
parameter_count = sum(parameter.numel() for parameter in model.parameters())
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=4, min_lr=1e-6,
)
amp_enabled = device.type == "cuda"
scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
checkpoint_path = OUTPUT_DIR / "01_best_validation_model.pt"


def validation_loss(loader):
    model.eval()
    weighted_loss = 0.0
    valid_pixels = 0.0
    with torch.no_grad():
        for x, y, mask, _ in loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            prediction = model(x)
            squared_error = ((prediction - y) ** 2 * mask).sum().item()
            weighted_loss += squared_error
            valid_pixels += mask.sum().item()
    return weighted_loss / max(valid_pixels, 1.0)


history = []
best_loss = np.inf
best_epoch = 0
stale_epochs = 0
started = time.time()

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_squared_error = 0.0
    train_valid_pixels = 0.0
    for x, y, mask, _ in train_loader:
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            prediction = model(x)
            loss = masked_mse(prediction, y, mask)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()
        pixels = mask.sum().item()
        train_squared_error += loss.item() * pixels
        train_valid_pixels += pixels

    train_loss = train_squared_error / max(train_valid_pixels, 1.0)
    val_loss = validation_loss(validation_loader)
    scheduler.step(val_loss)
    learning_rate = optimizer.param_groups[0]["lr"]
    history.append({
        "epoch": epoch, "train_mse_standardized": train_loss,
        "validation_mse_standardized": val_loss, "learning_rate": learning_rate,
    })

    improved = val_loss < best_loss - 1e-6
    if improved:
        best_loss = val_loss
        best_epoch = epoch
        stale_epochs = 0
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "validation_mse_standardized": val_loss,
                "target_mean_c": target_mean,
                "target_std_c": target_std,
                "input_channels": 16,
                "base_filters": BASE_FILTERS,
                "seed": SEED,
            },
            checkpoint_path,
        )
    else:
        stale_epochs += 1

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | train={train_loss:.5f} | "
        f"validation={val_loss:.5f} | lr={learning_rate:.2e}"
        + (" | BEST" if improved else "")
    )
    if stale_epochs >= PATIENCE:
        print(f"Early stopping after {epoch} epochs; best epoch was {best_epoch}.")
        break

history_df = pd.DataFrame(history)
history_df.to_csv(OUTPUT_DIR / "02_training_history.csv", index=False)

checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


def regression_metrics(observed, predicted):
    observed = np.asarray(observed, dtype="float64")
    predicted = np.asarray(predicted, dtype="float64")
    residual = predicted - observed
    mae = np.mean(np.abs(residual))
    rmse = np.sqrt(np.mean(residual ** 2))
    bias = np.mean(residual)
    denominator = np.sum((observed - observed.mean()) ** 2)
    r2 = 1.0 - np.sum(residual ** 2) / denominator if denominator > 0 else np.nan
    return {"pixels": len(observed), "mae_c": mae, "rmse_c": rmse, "bias_c": bias, "r2": r2}


def evaluate(loader, split_name):
    scene_rows = []
    pixel_groups = []
    prediction_dir = OUTPUT_DIR / f"predictions_{split_name}"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for x, y, mask, pilot_ids in loader:
            pilot_id = pilot_ids[0]
            prediction_z = model(x.to(device)).cpu().numpy()[0, 0]
            observed_z = y.numpy()[0, 0]
            valid = mask.numpy()[0, 0] > 0.5
            predicted_c = prediction_z * target_std + target_mean
            observed_c = observed_z * target_std + target_mean
            metrics = regression_metrics(observed_c[valid], predicted_c[valid])
            metadata_row = scenes.loc[scenes.pilot_id.eq(pilot_id)].iloc[0]
            scene_rows.append({
                "pilot_id": pilot_id, "record_id": metadata_row.record_id,
                "city": metadata_row.city, "thermal_sensor": metadata_row.thermal_sensor,
                "thermal_datetime_utc": metadata_row.thermal_datetime_utc,
                "model_split": split_name, **metrics,
            })
            pixel_groups.append(pd.DataFrame({
                "pilot_id": pilot_id,
                "city": metadata_row.city,
                "thermal_sensor": metadata_row.thermal_sensor,
                "observed_c": observed_c[valid].astype("float32"),
                "predicted_c": predicted_c[valid].astype("float32"),
            }))
            np.savez_compressed(
                prediction_dir / f"{pilot_id}_prediction.npz",
                predicted_c=predicted_c.astype("float32"),
                observed_c=np.where(valid, observed_c, 0).astype("float32"),
                valid_mask=valid.astype("uint8"),
            )
    return pd.DataFrame(scene_rows), pd.concat(pixel_groups, ignore_index=True)


validation_scenes, validation_pixels = evaluate(validation_loader, "validation")
test_scenes, test_pixels = evaluate(test_loader, "temporal_test")
scene_metrics = pd.concat([validation_scenes, test_scenes], ignore_index=True)
scene_metrics.to_csv(OUTPUT_DIR / "03_scene_metrics.csv", index=False)

summary_rows = []
for split_name, pixels in [("validation", validation_pixels), ("temporal_test", test_pixels)]:
    summary_rows.append({"model_split": split_name, "group": "ALL", **regression_metrics(pixels.observed_c, pixels.predicted_c)})
    for (city, sensor), group in pixels.groupby(["city", "thermal_sensor"], observed=True):
        summary_rows.append({
            "model_split": split_name, "group": f"{city}-{sensor}",
            **regression_metrics(group.observed_c, group.predicted_c),
        })
summary_metrics = pd.DataFrame(summary_rows)
summary_metrics.to_csv(OUTPUT_DIR / "04_aggregate_metrics.csv", index=False)

# Diagnostic figure based only on validation and held-out temporal-test predictions.
sns.set_theme(style="whitegrid", context="notebook")
fig, axes = plt.subplots(1, 3, figsize=(20, 6), constrained_layout=True)
axes[0].plot(history_df.epoch, history_df.train_mse_standardized, label="Train")
axes[0].plot(history_df.epoch, history_df.validation_mse_standardized, label="Validation")
axes[0].axvline(best_epoch, color="black", linestyle="--", linewidth=1, label=f"Best: {best_epoch}")
axes[0].set_title("(a) Learning curves")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Masked MSE (standardized)")
axes[0].legend(frameon=False)

plot_pixels = pd.concat([
    validation_pixels.assign(model_split="validation"),
    test_pixels.assign(model_split="temporal_test"),
], ignore_index=True)
if len(plot_pixels) > 100000:
    plot_pixels = plot_pixels.sample(100000, random_state=SEED)
sns.scatterplot(
    data=plot_pixels, x="observed_c", y="predicted_c", hue="model_split",
    alpha=0.15, s=10, linewidth=0, ax=axes[1],
)
limits = [
    min(plot_pixels.observed_c.min(), plot_pixels.predicted_c.min()),
    max(plot_pixels.observed_c.max(), plot_pixels.predicted_c.max()),
]
axes[1].plot(limits, limits, "k--", linewidth=1)
axes[1].set_xlim(limits)
axes[1].set_ylim(limits)
axes[1].set_title("(b) Observed versus predicted")
axes[1].set_xlabel("Observed LST (°C)")
axes[1].set_ylabel("Predicted LST (°C)")
axes[1].legend(frameon=False)

sns.boxplot(
    data=scene_metrics, x="thermal_sensor", y="rmse_c", hue="model_split", ax=axes[2],
)
axes[2].set_title("(c) Scene-level RMSE")
axes[2].set_xlabel("")
axes[2].set_ylabel("RMSE (°C)")
axes[2].legend(frameon=False)

figure_path = OUTPUT_DIR / "thermofusion_stage13_baseline_diagnostics.png"
fig.savefig(figure_path, dpi=800, bbox_inches="tight", facecolor="white")
plt.show()

validation_all = summary_metrics.query("model_split == 'validation' and group == 'ALL'").iloc[0]
test_all = summary_metrics.query("model_split == 'temporal_test' and group == 'ALL'").iloc[0]
integrity_pass = (
    checkpoint_path.exists()
    and len(validation_scenes) == 8
    and len(test_scenes) == 8
    and validation_scenes.pilot_id.nunique() == 8
    and test_scenes.pilot_id.nunique() == 8
    and set(validation_scenes.pilot_id).isdisjoint(set(test_scenes.pilot_id))
    and np.isfinite(summary_metrics[["mae_c", "rmse_c", "bias_c", "r2"]].to_numpy()).all()
)

verdict = [
    "THERMOFUSION STAGE 13 PILOT BASELINE",
    f"Compute device: {device}",
    f"Model parameters: {parameter_count:,}",
    f"Training scenes: {len(train_frame)}/48",
    f"Validation scenes: {len(validation_scenes)}/8",
    f"Independent temporal-test scenes: {len(test_scenes)}/8",
    f"Best validation epoch: {best_epoch}",
    f"Validation MAE (°C): {validation_all.mae_c:.4f}",
    f"Validation RMSE (°C): {validation_all.rmse_c:.4f}",
    f"Validation R²: {validation_all.r2:.4f}",
    f"Temporal-test MAE (°C): {test_all.mae_c:.4f}",
    f"Temporal-test RMSE (°C): {test_all.rmse_c:.4f}",
    f"Temporal-test bias (°C): {test_all.bias_c:.4f}",
    f"Temporal-test R²: {test_all.r2:.4f}",
    "Temporal-test data used for model selection: NO",
    "Overall Stage 13 execution: PASS" if integrity_pass else "Overall Stage 13 execution: REVIEW REQUIRED",
]
(OUTPUT_DIR / "05_stage13_verdict.txt").write_text("\n".join(verdict), encoding="utf-8")

run_config = {
    "seed": SEED, "epochs_requested": EPOCHS, "epochs_completed": int(history_df.epoch.max()),
    "early_stopping_patience": PATIENCE, "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
    "base_filters": BASE_FILTERS, "model_parameters": parameter_count,
    "device": str(device), "target_mean_c": target_mean, "target_std_c": target_std,
    "elapsed_minutes": (time.time() - started) / 60,
}
with (OUTPUT_DIR / "06_run_configuration.json").open("w", encoding="utf-8") as handle:
    json.dump(run_config, handle, indent=2)

zip_path = shutil.make_archive(
    "/content/drive/MyDrive/ThermoFusion_Stage13_Pilot_Baseline",
    "zip",
    root_dir=OUTPUT_DIR,
)
print("\n" + "\n".join(verdict))
print(f"\nOutput folder: {OUTPUT_DIR}")
print(f"ZIP package: {zip_path}")
