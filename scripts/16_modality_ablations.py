# =====================================================================
# ThermoFusion reproducibility code
# Source notebook cell(s): 30
# Audit status: KEEP
# Extracted without scientific-logic modification.
# Replace local Google Drive paths as described in README.md when needed.
# =====================================================================

# ThermoFusion Stage 16: independently retrained modality ablations.
# Run in one Google Colab cell with a T4 GPU after Stage 15 reports PASS.

from google.colab import drive
from pathlib import Path
import json, random, shutil, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

drive.mount("/content/drive")
ROOT = Path("/content/drive/MyDrive")
S12 = ROOT / "ThermoFusion_Stage12_ModelReady_Pilot"
S14 = ROOT / "ThermoFusion_Stage14_Refined_Benchmark"
S15 = ROOT / "ThermoFusion_Stage15_Calibration_Bootstrap"
OUT = ROOT / "ThermoFusion_Stage16_Modality_Ablation"
CHECKPOINT_DIR = OUT / "checkpoints"
OUT.mkdir(parents=True, exist_ok=True); CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

SEED, EPOCHS, PATIENCE, BATCH = 20260908, 60, 12, 2
BASE, LR, WEIGHT_DECAY, DELTA, WORKERS = 24, 2e-4, 1e-5, 1.0, 2
CITIES = ["Abidjan", "Accra", "Freetown", "Lagos"]
SENSORS = ["ECOSTRESS", "Landsat"]
VARIANTS = {
    "full_model": [],
    "without_sentinel2": list(range(0, 10)),
    "without_sentinel1": list(range(10, 14)),
    "without_terrain": [14, 15],
    "without_context": list(range(16, 25)),
}


def reset_seed():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


reset_seed()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type != "cuda":
    raise RuntimeError("Stage 16 requires a GPU runtime. Select Runtime > Change runtime type > T4 GPU.")
print(f"Compute device: {device}")

paths = {
    "arrays": S12 / "04_model_ready_array_manifest.csv",
    "splits": S12 / "01_scene_split_manifest.csv",
    "s12_verdict": S12 / "07_model_ready_verdict.txt",
    "s14_verdict": S14 / "06_stage14_verdict.txt",
    "s14_checkpoint": S14 / "02_best_refined_model.pt",
    "training_stats": S14 / "01_training_only_target_statistics.csv",
    "s15_verdict": S15 / "07_stage15_verdict.txt",
}
for path in paths.values():
    if not path.exists(): raise FileNotFoundError(f"Required input not found: {path}")
if "Overall model-ready verdict: PASS" not in paths["s12_verdict"].read_text():
    raise RuntimeError("Stage 12 did not pass.")
if "Overall Stage 14 execution: PASS" not in paths["s14_verdict"].read_text():
    raise RuntimeError("Stage 14 did not pass.")
if "Overall Stage 15 execution: PASS" not in paths["s15_verdict"].read_text():
    raise RuntimeError("Stage 15 did not pass.")

arrays = pd.read_csv(paths["arrays"])
meta = pd.read_csv(paths["splits"])[[
    "pilot_id", "record_id", "city", "thermal_sensor", "thermal_datetime_utc", "model_split"
]]
arrays = arrays.drop(columns=[c for c in ["city", "thermal_sensor", "thermal_datetime_utc"] if c in arrays])
scenes = arrays.merge(meta, on=["pilot_id", "record_id", "model_split"], validate="one_to_one")
scenes["thermal_datetime_utc"] = pd.to_datetime(scenes.thermal_datetime_utc, utc=True, errors="raise")
if len(scenes) != 64 or scenes.model_split.value_counts().to_dict() != {
    "train": 48, "validation": 8, "temporal_test": 8
}:
    raise RuntimeError("Stage 12 split is not the expected 48/8/8 dataset.")

stats_table = pd.read_csv(paths["training_stats"])
group_stats = {}
for row in stats_table.query("level == 'GROUP'").itertuples(index=False):
    city, sensor = row.group.split("|")
    group_stats[(city, sensor)] = {"mean_c": float(row.mean_c), "std_c": float(row.std_c)}
if len(group_stats) != 8: raise RuntimeError("Eight city-sensor training statistics are required.")


def context(row):
    stamp = row.thermal_datetime_utc
    doy = 2*np.pi*(stamp.dayofyear-1)/365.25
    hour = 2*np.pi*(stamp.hour + stamp.minute/60)/24
    values = [float(row.thermal_sensor == "ECOSTRESS"),
              np.sin(doy), np.cos(doy), np.sin(hour), np.cos(hour)]
    values += [float(row.city == city) for city in CITIES]
    return np.asarray(values, dtype="float32")[:, None, None]


class ThermoDataset(Dataset):
    def __init__(self, frame, variant, augment=False):
        self.frame = frame.reset_index(drop=True); self.variant = variant; self.augment = augment
    def __len__(self): return len(self.frame)
    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with np.load(row.array_path, allow_pickle=False) as item:
            x = item["x"].astype("float32")
            y_c = item["y_c"].astype("float32")
            mask = item["valid_mask"].astype("float32")
        x = np.concatenate([x, np.broadcast_to(context(row), (9,256,256)).copy()])
        x[VARIANTS[self.variant]] = 0.0
        gs = group_stats[(row.city, row.thermal_sensor)]
        y = ((y_c-gs["mean_c"])/gs["std_c"])[None]; y[:, mask < 0.5] = 0; mask = mask[None]
        if self.augment:
            if random.random() < .5: x,y,mask = x[:,:,::-1],y[:,:,::-1],mask[:,:,::-1]
            if random.random() < .5: x,y,mask = x[:,::-1,:],y[:,::-1,:],mask[:,::-1,:]
            k = random.randint(0,3)
            if k: x,y,mask = np.rot90(x,k,(1,2)),np.rot90(y,k,(1,2)),np.rot90(mask,k,(1,2))
        return (torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(y)),
                torch.from_numpy(np.ascontiguousarray(mask)), row.pilot_id)


def block(inputs, outputs):
    return nn.Sequential(nn.Conv2d(inputs,outputs,3,padding=1,bias=False),
        nn.GroupNorm(min(8,outputs),outputs),nn.SiLU(inplace=True),
        nn.Conv2d(outputs,outputs,3,padding=1,bias=False),
        nn.GroupNorm(min(8,outputs),outputs),nn.SiLU(inplace=True))


class UNet(nn.Module):
    def __init__(self):
        super().__init__(); b=BASE; self.p=nn.MaxPool2d(2)
        self.e1=block(25,b); self.e2=block(b,2*b); self.e3=block(2*b,4*b); self.e4=block(4*b,8*b)
        self.bridge=block(8*b,16*b); self.u4=nn.ConvTranspose2d(16*b,8*b,2,2); self.d4=block(16*b,8*b)
        self.u3=nn.ConvTranspose2d(8*b,4*b,2,2); self.d3=block(8*b,4*b)
        self.u2=nn.ConvTranspose2d(4*b,2*b,2,2); self.d2=block(4*b,2*b)
        self.u1=nn.ConvTranspose2d(2*b,b,2,2); self.d1=block(2*b,b); self.output=nn.Conv2d(b,1,1)
    def forward(self,x):
        e1=self.e1(x); e2=self.e2(self.p(e1)); e3=self.e3(self.p(e2)); e4=self.e4(self.p(e3))
        z=self.bridge(self.p(e4)); z=self.d4(torch.cat([self.u4(z),e4],1)); z=self.d3(torch.cat([self.u3(z),e3],1))
        z=self.d2(torch.cat([self.u2(z),e2],1)); return self.output(self.d1(torch.cat([self.u1(z),e1],1)))


def loss_fn(pred,target,mask):
    error=pred-target; absolute=error.abs()
    loss=torch.where(absolute<=DELTA,.5*error.square(),DELTA*(absolute-.5*DELTA))
    return (loss*mask).sum()/mask.sum().clamp_min(1)


def regression_metrics(observed,predicted):
    observed=np.asarray(observed,dtype="float64"); predicted=np.asarray(predicted,dtype="float64")
    error=predicted-observed; denominator=np.sum((observed-observed.mean())**2)
    return {"pixels":len(observed),"mae_c":np.mean(np.abs(error)),"rmse_c":np.sqrt(np.mean(error**2)),
            "bias_c":np.mean(error),"r2":1-np.sum(error**2)/denominator if denominator>0 else np.nan}


train=scenes.query("model_split=='train'").copy(); validation=scenes.query("model_split=='validation'").copy()
test=scenes.query("model_split=='temporal_test'").copy(); loader_kwargs=dict(num_workers=WORKERS,pin_memory=True)


def loaders(variant):
    return (
        DataLoader(ThermoDataset(train,variant,True),batch_size=BATCH,shuffle=True,**loader_kwargs),
        DataLoader(ThermoDataset(validation,variant),batch_size=1,shuffle=False,**loader_kwargs),
        DataLoader(ThermoDataset(test,variant),batch_size=1,shuffle=False,**loader_kwargs),
    )


def evaluate_loss(model,loader):
    model.eval(); total=pixels=0.0
    with torch.no_grad():
        for x,y,mask,_ in loader:
            x,y,mask=x.to(device),y.to(device),mask.to(device); count=mask.sum().item()
            total+=loss_fn(model(x),y,mask).item()*count; pixels+=count
    return total/max(pixels,1)


def train_variant(variant):
    reset_seed(); train_loader,val_loader,test_loader=loaders(variant); model=UNet().to(device)
    checkpoint=CHECKPOINT_DIR/f"{variant}_best.pt"
    optimizer=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=.5,patience=5,min_lr=1e-6)
    scaler=torch.amp.GradScaler("cuda",enabled=True); best=np.inf; best_epoch=stale=0; history=[]
    for epoch in range(1,EPOCHS+1):
        model.train(); total=pixels=0.0
        for x,y,mask,_ in train_loader:
            x,y,mask=x.to(device),y.to(device),mask.to(device); optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                prediction=model(x); loss=loss_fn(prediction,y,mask)
            scaler.scale(loss).backward(); scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(),5)
            scaler.step(optimizer); scaler.update(); count=mask.sum().item(); total+=loss.item()*count; pixels+=count
        train_loss=total/max(pixels,1); val_loss=evaluate_loss(model,val_loader); scheduler.step(val_loss)
        improved=val_loss<best-1e-6
        if improved:
            best,best_epoch,stale=val_loss,epoch,0
            torch.save({"model_state_dict":model.state_dict(),"variant":variant,"epoch":epoch,"seed":SEED},checkpoint)
        else: stale+=1
        history.append({"variant":variant,"epoch":epoch,"train_huber":train_loss,"validation_huber":val_loss})
        print(f"{variant} | epoch {epoch:02d}/{EPOCHS} | train={train_loss:.5f} | val={val_loss:.5f}"+(" | BEST" if improved else ""))
        if stale>=PATIENCE: break
    model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=False)["model_state_dict"]); model.eval()
    return model,val_loader,test_loader,pd.DataFrame(history),best_epoch


def load_full_model():
    model=UNet().to(device)
    saved=torch.load(paths["s14_checkpoint"],map_location=device,weights_only=False)
    model.load_state_dict(saved["model_state_dict"]); model.eval()
    _,val_loader,test_loader=loaders("full_model")
    return model,val_loader,test_loader,pd.DataFrame(),int(saved["epoch"])


scene_rows=[]; histories=[]; best_epochs={}; started=time.time()
for variant in VARIANTS:
    print(f"\n{'='*80}\nVARIANT: {variant}\n{'='*80}")
    if variant=="full_model": model,val_loader,test_loader,history,best_epoch=load_full_model()
    else: model,val_loader,test_loader,history,best_epoch=train_variant(variant)
    if not history.empty: histories.append(history)
    best_epochs[variant]=best_epoch
    for split,frame,loader in [("validation",validation,val_loader),("temporal_test",test,test_loader)]:
        with torch.no_grad():
            for x,_,mask,pids in loader:
                pid=pids[0]; row=frame.loc[frame.pilot_id.eq(pid)].iloc[0]
                with np.load(row.array_path,allow_pickle=False) as item: observed=item["y_c"].astype("float32")
                valid=mask.numpy()[0,0]>.5; gs=group_stats[(row.city,row.thermal_sensor)]
                predicted=model(x.to(device)).cpu().numpy()[0,0]*gs["std_c"]+gs["mean_c"]
                scene_rows.append({"variant":variant,"model_split":split,"pilot_id":pid,"city":row.city,
                                   "thermal_sensor":row.thermal_sensor,**regression_metrics(observed[valid],predicted[valid])})
    del model; torch.cuda.empty_cache()

scene_metrics=pd.DataFrame(scene_rows); scene_metrics.to_csv(OUT/"01_scene_ablation_metrics.csv",index=False)
if histories: pd.concat(histories,ignore_index=True).to_csv(OUT/"02_ablation_training_histories.csv",index=False)

summary=(scene_metrics.groupby(["variant","model_split"],observed=True)
    .agg(scene_macro_mae_c=("mae_c","mean"),scene_macro_rmse_c=("rmse_c","mean"),
         median_scene_rmse_c=("rmse_c","median"),mean_scene_r2=("r2","mean"))
    .reset_index())
full_test=summary.query("variant=='full_model' and model_split=='temporal_test'").iloc[0]
full_by_split=(summary.query("variant=='full_model'")
    .set_index("model_split")["scene_macro_rmse_c"].to_dict())
summary["full_model_rmse_same_split_c"]=summary.model_split.map(full_by_split)
summary["delta_rmse_vs_full_c"]=(summary.scene_macro_rmse_c-summary.full_model_rmse_same_split_c)
summary["relative_rmse_change_pct"]=100*summary.delta_rmse_vs_full_c/summary.full_model_rmse_same_split_c
summary.to_csv(OUT/"03_ablation_summary.csv",index=False)

# Paired test-scene bootstrap for each ablation minus the full model.
test_wide=scene_metrics.query("model_split=='temporal_test'").pivot(index="pilot_id",columns="variant",values="rmse_c")
rng=np.random.default_rng(SEED); bootstrap_rows=[]; draws=10000; ids=test_wide.index.to_numpy()
for variant in VARIANTS:
    if variant=="full_model": continue
    delta=test_wide[variant]-test_wide["full_model"]; samples=np.empty(draws)
    for n in range(draws): samples[n]=delta.loc[rng.choice(ids,size=len(ids),replace=True)].mean()
    bootstrap_rows.append({"variant":variant,"mean_delta_rmse_c":delta.mean(),
        "ci_lower":np.quantile(samples,.025),"ci_upper":np.quantile(samples,.975),
        "probability_ablation_worse":np.mean(samples>0),"bootstrap_replicates":draws})
bootstrap=pd.DataFrame(bootstrap_rows); bootstrap.to_csv(OUT/"04_paired_ablation_bootstrap.csv",index=False)

sns.set_theme(style="whitegrid",context="notebook"); fig,axes=plt.subplots(1,3,figsize=(20,6),constrained_layout=True)
sns.barplot(data=summary,x="variant",y="scene_macro_rmse_c",hue="model_split",ax=axes[0])
axes[0].set(title="(a) Independently retrained ablations",xlabel="",ylabel="Scene-macro RMSE (°C)")
axes[0].tick_params(axis="x",rotation=24); axes[0].legend(frameon=False)
test_plot=scene_metrics.query("model_split=='temporal_test'")
sns.boxplot(data=test_plot,x="variant",y="rmse_c",ax=axes[1]); sns.stripplot(data=test_plot,x="variant",y="rmse_c",color="black",size=4,ax=axes[1])
axes[1].set(title="(b) Temporal-test scene dispersion",xlabel="",ylabel="Scene RMSE (°C)"); axes[1].tick_params(axis="x",rotation=24)
plot_boot=bootstrap.sort_values("mean_delta_rmse_c"); y=np.arange(len(plot_boot))
axes[2].errorbar(plot_boot.mean_delta_rmse_c,y,xerr=[plot_boot.mean_delta_rmse_c-plot_boot.ci_lower,
    plot_boot.ci_upper-plot_boot.mean_delta_rmse_c],fmt="o",capsize=4,color="#b42c3b")
axes[2].axvline(0,color="black",ls="--",lw=1); axes[2].set_yticks(y,plot_boot.variant)
axes[2].set(title="(c) Paired bootstrap effects",xlabel="Ablation minus full-model RMSE (°C)",ylabel="")
fig.savefig(OUT/"thermofusion_stage16_modality_ablation.png",dpi=800,bbox_inches="tight",facecolor="white"); plt.show()

test_summary=summary.query("model_split=='temporal_test'").sort_values("scene_macro_rmse_c")
best_variant=test_summary.iloc[0]
execution_pass=(len(scene_metrics)==80 and scene_metrics.pilot_id.nunique()==16 and len(summary)==10
                and len(bootstrap)==4 and np.isfinite(scene_metrics[["mae_c","rmse_c","bias_c","r2"]]).all().all())
verdict=["THERMOFUSION STAGE 16 MODALITY ABLATION",
    "Ablation method: independent retraining with identical split and seed",
    f"Variants completed: {len(VARIANTS)}/5",f"Validation scenes per variant: 8/8",f"Temporal-test scenes per variant: 8/8",
    f"Best temporal-test variant: {best_variant.variant}",f"Best temporal-test scene-macro RMSE (°C): {best_variant.scene_macro_rmse_c:.4f}",
    f"Full-model temporal-test scene-macro RMSE (°C): {full_test.scene_macro_rmse_c:.4f}",
    "Temporal-test data used for training or model selection: NO",
    "Overall Stage 16 execution: PASS" if execution_pass else "Overall Stage 16 execution: REVIEW REQUIRED"]
(OUT/"05_stage16_verdict.txt").write_text("\n".join(verdict),encoding="utf-8")
with (OUT/"06_configuration.json").open("w") as handle:
    json.dump({"seed":SEED,"variants":VARIANTS,"best_epochs":best_epochs,"elapsed_minutes":(time.time()-started)/60},handle,indent=2)
zip_path=shutil.make_archive(str(ROOT/"ThermoFusion_Stage16_Modality_Ablation"),"zip",root_dir=OUT)
print("\n"+"\n".join(verdict)); print(f"\nOutput folder: {OUT}"); print(f"ZIP package: {zip_path}")
