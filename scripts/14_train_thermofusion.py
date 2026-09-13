# =====================================================================
# ThermoFusion reproducibility code
# Source notebook cell(s): 28
# Audit status: KEEP
# Extracted without scientific-logic modification.
# Replace local Google Drive paths as described in README.md when needed.
# =====================================================================

# ThermoFusion Stage 14: metadata-conditioned residual U-Net and fair baselines.
# Run in one Google Colab cell after Stage 13 reports PASS.

from google.colab import drive
from pathlib import Path
import json, random, shutil, subprocess, sys, time
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
ROOT = Path("/content/drive/MyDrive")
S12 = ROOT / "ThermoFusion_Stage12_ModelReady_Pilot"
S13 = ROOT / "ThermoFusion_Stage13_Pilot_Baseline"
OUT = ROOT / "ThermoFusion_Stage14_Refined_Benchmark"
PRED_DIR = OUT / "predictions"
OUT.mkdir(parents=True, exist_ok=True)
PRED_DIR.mkdir(parents=True, exist_ok=True)

SEED, EPOCHS, PATIENCE, BATCH = 20260908, 80, 15, 2
BASE, LR, WEIGHT_DECAY, DELTA, WORKERS = 24, 2e-4, 1e-5, 1.0, 2
CITIES = ["Abidjan", "Accra", "Freetown", "Lagos"]
SENSORS = ["ECOSTRESS", "Landsat"]

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Compute device: {device}")

required = [
    S12 / "04_model_ready_array_manifest.csv",
    S12 / "01_scene_split_manifest.csv",
    S12 / "07_model_ready_verdict.txt",
    S13 / "05_stage13_verdict.txt",
]
for path in required:
    if not path.exists():
        raise FileNotFoundError(f"Required input not found: {path}")
if "Overall model-ready verdict: PASS" not in required[2].read_text(encoding="utf-8"):
    raise RuntimeError("Stage 12 did not pass.")
if "Overall Stage 13 execution: PASS" not in required[3].read_text(encoding="utf-8"):
    raise RuntimeError("Stage 13 did not pass.")

arrays = pd.read_csv(required[0])
meta = pd.read_csv(required[1])[[
    "pilot_id", "record_id", "city", "thermal_sensor",
    "thermal_datetime_utc", "model_split",
]]
arrays = arrays.drop(columns=[c for c in ["city", "thermal_sensor", "thermal_datetime_utc"] if c in arrays])
scenes = arrays.merge(meta, on=["pilot_id", "record_id", "model_split"], validate="one_to_one")
scenes["thermal_datetime_utc"] = pd.to_datetime(scenes.thermal_datetime_utc, utc=True, errors="raise")
if len(scenes) != 64 or scenes.record_id.nunique() != 64:
    raise RuntimeError("Stage 12 does not contain 64 unique scenes.")
if scenes.model_split.value_counts().to_dict() != {"train": 48, "validation": 8, "temporal_test": 8}:
    raise RuntimeError("Expected a 48/8/8 scene split.")


def load_target(row):
    with np.load(row.array_path, allow_pickle=False) as item:
        y = item["y_c"].astype("float32")
        mask = item["valid_mask"].astype(bool)
    if y.shape != (256, 256) or mask.shape != (256, 256) or mask.sum() == 0:
        raise RuntimeError(f"Invalid target for {row.pilot_id}.")
    return y, mask


# Calculate all climatologies and residual scales from training pixels only.
accumulators = {}
for row in scenes.query("model_split == 'train'").itertuples(index=False):
    y, mask = load_target(row)
    values = y[mask].astype("float64")
    keys = [("GLOBAL", "ALL"), ("SENSOR", row.thermal_sensor),
            ("GROUP", row.city, row.thermal_sensor)]
    for key in keys:
        acc = accumulators.setdefault(key, [0, 0.0, 0.0])
        acc[0] += values.size; acc[1] += values.sum(); acc[2] += np.square(values).sum()


def finish_stat(acc):
    count, total, total_sq = acc
    mean = total / count
    variance = max(total_sq / count - mean * mean, 1e-6)
    return {"pixels": int(count), "mean_c": float(mean), "std_c": float(np.sqrt(variance))}


stats = {key: finish_stat(value) for key, value in accumulators.items()}
group_stats = {(c, s): stats[("GROUP", c, s)] for c in CITIES for s in SENSORS}
pd.DataFrame([
    {"level": key[0], "group": "|".join(key[1:]), **value}
    for key, value in stats.items()
]).to_csv(OUT / "01_training_only_target_statistics.csv", index=False)


def context(row):
    stamp = row.thermal_datetime_utc
    doy = 2 * np.pi * (stamp.dayofyear - 1) / 365.25
    hour = 2 * np.pi * (stamp.hour + stamp.minute / 60) / 24
    values = [float(row.thermal_sensor == "ECOSTRESS"),
              np.sin(doy), np.cos(doy), np.sin(hour), np.cos(hour)]
    values += [float(row.city == city) for city in CITIES]
    return np.asarray(values, dtype="float32")[:, None, None]


class ThermoDataset(Dataset):
    def __init__(self, frame, augment=False):
        self.frame = frame.reset_index(drop=True); self.augment = augment

    def __len__(self): return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with np.load(row.array_path, allow_pickle=False) as item:
            x = item["x"].astype("float32")
            y = item["y_c"].astype("float32")
            mask = item["valid_mask"].astype("float32")
        x = np.concatenate([x, np.broadcast_to(context(row), (9, 256, 256)).copy()])
        gs = group_stats[(row.city, row.thermal_sensor)]
        y = ((y - gs["mean_c"]) / gs["std_c"])[None]
        y[:, mask < 0.5] = 0; mask = mask[None]
        if self.augment:
            if random.random() < 0.5:
                x, y, mask = x[:, :, ::-1], y[:, :, ::-1], mask[:, :, ::-1]
            if random.random() < 0.5:
                x, y, mask = x[:, ::-1, :], y[:, ::-1, :], mask[:, ::-1, :]
            k = random.randint(0, 3)
            if k:
                x = np.rot90(x, k, (1, 2)); y = np.rot90(y, k, (1, 2)); mask = np.rot90(mask, k, (1, 2))
        return (torch.from_numpy(np.ascontiguousarray(x)),
                torch.from_numpy(np.ascontiguousarray(y)),
                torch.from_numpy(np.ascontiguousarray(mask)), row.pilot_id)


def conv_block(inputs, outputs):
    return nn.Sequential(
        nn.Conv2d(inputs, outputs, 3, padding=1, bias=False),
        nn.GroupNorm(min(8, outputs), outputs), nn.SiLU(inplace=True),
        nn.Conv2d(outputs, outputs, 3, padding=1, bias=False),
        nn.GroupNorm(min(8, outputs), outputs), nn.SiLU(inplace=True),
    )


class ResidualUNet(nn.Module):
    def __init__(self):
        super().__init__(); b = BASE; self.pool = nn.MaxPool2d(2)
        self.e1 = conv_block(25, b); self.e2 = conv_block(b, 2*b)
        self.e3 = conv_block(2*b, 4*b); self.e4 = conv_block(4*b, 8*b)
        self.bridge = conv_block(8*b, 16*b)
        self.u4 = nn.ConvTranspose2d(16*b, 8*b, 2, 2); self.d4 = conv_block(16*b, 8*b)
        self.u3 = nn.ConvTranspose2d(8*b, 4*b, 2, 2); self.d3 = conv_block(8*b, 4*b)
        self.u2 = nn.ConvTranspose2d(4*b, 2*b, 2, 2); self.d2 = conv_block(4*b, 2*b)
        self.u1 = nn.ConvTranspose2d(2*b, b, 2, 2); self.d1 = conv_block(2*b, b)
        self.output = nn.Conv2d(b, 1, 1)

    def forward(self, x):
        e1 = self.e1(x); e2 = self.e2(self.pool(e1)); e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3)); z = self.bridge(self.pool(e4))
        z = self.d4(torch.cat([self.u4(z), e4], 1)); z = self.d3(torch.cat([self.u3(z), e3], 1))
        z = self.d2(torch.cat([self.u2(z), e2], 1)); z = self.d1(torch.cat([self.u1(z), e1], 1))
        return self.output(z)


def masked_huber(pred, target, mask):
    error = pred - target; absolute = error.abs()
    loss = torch.where(absolute <= DELTA, 0.5 * error.square(), DELTA * (absolute - 0.5 * DELTA))
    return (loss * mask).sum() / mask.sum().clamp_min(1)


train = scenes.query("model_split == 'train'").copy()
validation = scenes.query("model_split == 'validation'").copy()
test = scenes.query("model_split == 'temporal_test'").copy()
loader_args = dict(num_workers=WORKERS, pin_memory=torch.cuda.is_available())
train_loader = DataLoader(ThermoDataset(train, True), batch_size=BATCH, shuffle=True, **loader_args)
val_loader = DataLoader(ThermoDataset(validation), batch_size=1, shuffle=False, **loader_args)
test_loader = DataLoader(ThermoDataset(test), batch_size=1, shuffle=False, **loader_args)

model = ResidualUNet().to(device)
parameters = sum(p.numel() for p in model.parameters())
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5, min_lr=1e-6)
amp = device.type == "cuda"; scaler = torch.amp.GradScaler("cuda", enabled=amp)
checkpoint_path = OUT / "02_best_refined_model.pt"


def validation_loss():
    model.eval(); total = pixels = 0.0
    with torch.no_grad():
        for x, y, mask, _ in val_loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            count = mask.sum().item(); total += masked_huber(model(x), y, mask).item() * count; pixels += count
    return total / max(pixels, 1)


history = []; best = np.inf; best_epoch = stale = 0; started = time.time()
for epoch in range(1, EPOCHS + 1):
    model.train(); total = pixels = 0.0
    for x, y, mask, _ in train_loader:
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp):
            prediction = model(x); loss = masked_huber(prediction, y, mask)
        scaler.scale(loss).backward(); scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer); scaler.update()
        count = mask.sum().item(); total += loss.item() * count; pixels += count
    train_loss = total / max(pixels, 1); val_loss = validation_loss(); scheduler.step(val_loss)
    lr = optimizer.param_groups[0]["lr"]; improved = val_loss < best - 1e-6
    if improved:
        best, best_epoch, stale = val_loss, epoch, 0
        torch.save({"model_state_dict": model.state_dict(), "epoch": epoch,
                    "input_channels": 25, "group_stats": group_stats, "seed": SEED}, checkpoint_path)
    else: stale += 1
    history.append({"epoch": epoch, "train_huber": train_loss,
                    "validation_huber": val_loss, "learning_rate": lr})
    print(f"Epoch {epoch:02d}/{EPOCHS} | train={train_loss:.5f} | validation={val_loss:.5f} | lr={lr:.2e}" +
          (" | BEST" if improved else ""))
    if stale >= PATIENCE:
        print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}."); break

history = pd.DataFrame(history); history.to_csv(OUT / "03_training_history.csv", index=False)
model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=False)["model_state_dict"])
model.eval()


def metric(y, prediction):
    y = np.asarray(y, dtype="float64"); prediction = np.asarray(prediction, dtype="float64")
    residual = prediction - y; denominator = np.sum((y - y.mean()) ** 2)
    return {"pixels": len(y), "mae_c": np.mean(np.abs(residual)),
            "rmse_c": np.sqrt(np.mean(residual**2)), "bias_c": np.mean(residual),
            "r2": 1 - np.sum(residual**2) / denominator if denominator > 0 else np.nan}


global_mean = stats[("GLOBAL", "ALL")]["mean_c"]
sensor_mean = {s: stats[("SENSOR", s)]["mean_c"] for s in SENSORS}
scene_rows, pixel_frames = [], []


def evaluate(frame, loader, split):
    with torch.no_grad():
        for x, _, _, pilot_ids in loader:
            pid = pilot_ids[0]; row = frame.loc[frame.pilot_id.eq(pid)].iloc[0]
            y, valid = load_target(row); gs = group_stats[(row.city, row.thermal_sensor)]
            refined = model(x.to(device)).cpu().numpy()[0, 0] * gs["std_c"] + gs["mean_c"]
            predictions = {
                "global_climatology": np.full_like(y, global_mean),
                "sensor_climatology": np.full_like(y, sensor_mean[row.thermal_sensor]),
                "city_sensor_climatology": np.full_like(y, gs["mean_c"]),
                "refined_unet": refined,
            }
            for name, pred in predictions.items():
                scene_rows.append({"model_split": split, "model": name, "pilot_id": pid,
                                   "city": row.city, "thermal_sensor": row.thermal_sensor,
                                   **metric(y[valid], pred[valid])})
                pixel_frames.append(pd.DataFrame({"model_split": split, "model": name,
                    "pilot_id": pid, "city": row.city, "thermal_sensor": row.thermal_sensor,
                    "observed_c": y[valid], "predicted_c": pred[valid]}))
            np.savez_compressed(PRED_DIR / f"{pid}_prediction.npz",
                                predicted_c=refined.astype("float32"), observed_c=y,
                                valid_mask=valid.astype("uint8"))


evaluate(validation, val_loader, "validation"); evaluate(test, test_loader, "temporal_test")
scene_metrics = pd.DataFrame(scene_rows)
scene_metrics.to_csv(OUT / "04_scene_model_comparison.csv", index=False)
pixels = pd.concat(pixel_frames, ignore_index=True)
aggregate_rows = []
for (split, name), group in pixels.groupby(["model_split", "model"], observed=True):
    aggregate_rows.append({"model_split": split, "model": name, "group": "ALL",
                           **metric(group.observed_c, group.predicted_c)})
    for (city, sensor), sub in group.groupby(["city", "thermal_sensor"], observed=True):
        aggregate_rows.append({"model_split": split, "model": name,
                               "group": f"{city}-{sensor}", **metric(sub.observed_c, sub.predicted_c)})
aggregate = pd.DataFrame(aggregate_rows)
aggregate.to_csv(OUT / "05_aggregate_model_comparison.csv", index=False)

overall = aggregate.query("group == 'ALL'")
test_all = overall.query("model_split == 'temporal_test'").set_index("model")
refined = test_all.loc["refined_unet"]
baseline_name = test_all.drop(index="refined_unet").rmse_c.idxmin()
baseline = test_all.loc[baseline_name]
improvement = 100 * (baseline.rmse_c - refined.rmse_c) / baseline.rmse_c
val_refined = overall.query("model_split == 'validation' and model == 'refined_unet'").iloc[0]

sns.set_theme(style="whitegrid", context="notebook")
fig, axes = plt.subplots(1, 3, figsize=(20, 6), constrained_layout=True)
axes[0].plot(history.epoch, history.train_huber, label="Train")
axes[0].plot(history.epoch, history.validation_huber, label="Validation")
axes[0].axvline(best_epoch, color="black", ls="--", lw=1, label=f"Best: {best_epoch}")
axes[0].set(title="(a) Refined learning curves", xlabel="Epoch", ylabel="Masked Huber loss")
axes[0].legend(frameon=False)
sns.barplot(data=overall, x="model", y="rmse_c", hue="model_split", ax=axes[1])
axes[1].set(title="(b) Fair baseline comparison", xlabel="", ylabel="RMSE (°C)")
axes[1].tick_params(axis="x", rotation=25); axes[1].legend(frameon=False)
plot_data = pixels.query("model == 'refined_unet'")
if len(plot_data) > 100000: plot_data = plot_data.sample(100000, random_state=SEED)
sns.scatterplot(data=plot_data, x="observed_c", y="predicted_c", hue="model_split",
                alpha=0.15, s=10, linewidth=0, ax=axes[2])
limits = [min(plot_data.observed_c.min(), plot_data.predicted_c.min()),
          max(plot_data.observed_c.max(), plot_data.predicted_c.max())]
axes[2].plot(limits, limits, "k--", lw=1)
axes[2].set(xlim=limits, ylim=limits, title="(c) Refined observed versus predicted",
            xlabel="Observed LST (°C)", ylabel="Predicted LST (°C)")
axes[2].legend(frameon=False)
fig.savefig(OUT / "thermofusion_stage14_refined_benchmark.png",
            dpi=800, bbox_inches="tight", facecolor="white")
plt.show()

execution_pass = (checkpoint_path.exists() and len(scene_metrics) == 64
                  and scene_metrics.pilot_id.nunique() == 16
                  and np.isfinite(aggregate[["mae_c", "rmse_c", "bias_c", "r2"]]).all().all())
verdict = [
    "THERMOFUSION STAGE 14 REFINED BENCHMARK",
    f"Training scenes: {len(train)}/48",
    f"Validation scenes: {len(validation)}/8",
    f"Independent temporal-test scenes: {len(test)}/8",
    "Input channels: 25 (16 raster + 9 known context channels)",
    f"Best epoch: {best_epoch}",
    f"Validation refined MAE (°C): {val_refined.mae_c:.4f}",
    f"Validation refined RMSE (°C): {val_refined.rmse_c:.4f}",
    f"Validation refined R²: {val_refined.r2:.4f}",
    f"Temporal-test refined MAE (°C): {refined.mae_c:.4f}",
    f"Temporal-test refined RMSE (°C): {refined.rmse_c:.4f}",
    f"Temporal-test refined bias (°C): {refined.bias_c:.4f}",
    f"Temporal-test refined R²: {refined.r2:.4f}",
    f"Best temporal-test climatology: {baseline_name}",
    f"Best climatology RMSE (°C): {baseline.rmse_c:.4f}",
    f"Refined-model RMSE improvement over best climatology (%): {improvement:.2f}",
    "Temporal-test data used for training or model selection: NO",
    "Overall Stage 14 execution: PASS" if execution_pass else "Overall Stage 14 execution: REVIEW REQUIRED",
]
(OUT / "06_stage14_verdict.txt").write_text("\n".join(verdict), encoding="utf-8")
with (OUT / "07_configuration.json").open("w", encoding="utf-8") as handle:
    json.dump({"seed": SEED, "epochs_completed": int(history.epoch.max()),
               "best_epoch": best_epoch, "parameters": parameters,
               "device": str(device), "elapsed_minutes": (time.time()-started)/60}, handle, indent=2)
zip_path = shutil.make_archive(str(ROOT / "ThermoFusion_Stage14_Refined_Benchmark"),
                               "zip", root_dir=OUT)
print("\n" + "\n".join(verdict))
print(f"\nOutput folder: {OUT}")
print(f"ZIP package: {zip_path}")
