# =====================================================================
# ThermoFusion reproducibility code
# Source notebook cell(s): 31
# Audit status: KEEP
# Extracted without scientific-logic modification.
# Replace local Google Drive paths as described in README.md when needed.
# =====================================================================

# ThermoFusion Stage 17: final validation-selected model and uncertainty analysis.
# Run in one Google Colab cell with a T4 GPU after Stage 16 reports PASS.

from google.colab import drive
from pathlib import Path
import json, shutil, subprocess, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn

try:
    from scipy.stats import spearmanr
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "scipy"])
    from scipy.stats import spearmanr

drive.mount("/content/drive")
ROOT = Path("/content/drive/MyDrive")
S12 = ROOT / "ThermoFusion_Stage12_ModelReady_Pilot"
S14 = ROOT / "ThermoFusion_Stage14_Refined_Benchmark"
S16 = ROOT / "ThermoFusion_Stage16_Modality_Ablation"
OUT = ROOT / "ThermoFusion_Stage17_Final_Uncertainty"
PRED_DIR = OUT / "temporal_test_predictions"
OUT.mkdir(parents=True, exist_ok=True); PRED_DIR.mkdir(parents=True, exist_ok=True)

SEED = 20260908
BASE = 24
TARGET_COVERAGE = 0.90
UNCERTAINTY_FLOOR_C = 0.50
CALIBRATION_PIXELS_PER_SCENE = 20000
CITIES = ["Abidjan", "Accra", "Freetown", "Lagos"]
SENSORS = ["ECOSTRESS", "Landsat"]

paths = {
    "arrays": S12 / "04_model_ready_array_manifest.csv",
    "splits": S12 / "01_scene_split_manifest.csv",
    "s12_verdict": S12 / "07_model_ready_verdict.txt",
    "training_stats": S14 / "01_training_only_target_statistics.csv",
    "s16_verdict": S16 / "05_stage16_verdict.txt",
    "checkpoint": S16 / "checkpoints" / "without_terrain_best.pt",
    "ablation": S16 / "03_ablation_summary.csv",
}
for path in paths.values():
    if not path.exists(): raise FileNotFoundError(f"Required input not found: {path}")
if "Overall model-ready verdict: PASS" not in paths["s12_verdict"].read_text():
    raise RuntimeError("Stage 12 did not pass.")
if "Overall Stage 16 execution: PASS" not in paths["s16_verdict"].read_text():
    raise RuntimeError("Stage 16 did not pass.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type != "cuda":
    raise RuntimeError("Select Runtime > Change runtime type > T4 GPU.")
print(f"Compute device: {device}")

arrays = pd.read_csv(paths["arrays"])
meta = pd.read_csv(paths["splits"])[[
    "pilot_id", "record_id", "city", "thermal_sensor", "thermal_datetime_utc", "model_split"
]]
arrays = arrays.drop(columns=[c for c in ["city", "thermal_sensor", "thermal_datetime_utc"] if c in arrays])
scenes = arrays.merge(meta, on=["pilot_id", "record_id", "model_split"], validate="one_to_one")
scenes["thermal_datetime_utc"] = pd.to_datetime(scenes.thermal_datetime_utc, utc=True, errors="raise")
validation = scenes.query("model_split == 'validation'").copy()
test = scenes.query("model_split == 'temporal_test'").copy()
if len(scenes) != 64 or len(validation) != 8 or len(test) != 8:
    raise RuntimeError("Expected 64 scenes with 8 validation and 8 temporal-test scenes.")

stats_table = pd.read_csv(paths["training_stats"])
group_stats = {}
for row in stats_table.query("level == 'GROUP'").itertuples(index=False):
    city, sensor = row.group.split("|")
    group_stats[(city, sensor)] = {"mean_c": float(row.mean_c), "std_c": float(row.std_c)}
if len(group_stats) != 8: raise RuntimeError("Eight city-sensor training statistics are required.")


def context(row):
    stamp = row.thermal_datetime_utc
    doy = 2*np.pi*(stamp.dayofyear-1)/365.25
    hour = 2*np.pi*(stamp.hour+stamp.minute/60)/24
    values = [float(row.thermal_sensor == "ECOSTRESS"),
              np.sin(doy), np.cos(doy), np.sin(hour), np.cos(hour)]
    values += [float(row.city == city) for city in CITIES]
    return np.asarray(values, dtype="float32")[:, None, None]


def load_scene(row):
    with np.load(row.array_path, allow_pickle=False) as item:
        x = item["x"].astype("float32")
        observed = item["y_c"].astype("float32")
        valid = item["valid_mask"].astype(bool)
    x = np.concatenate([x, np.broadcast_to(context(row), (9,256,256)).copy()])
    x[[14,15]] = 0.0
    if x.shape != (25,256,256) or observed.shape != (256,256) or valid.sum() == 0:
        raise RuntimeError(f"Invalid scene array for {row.pilot_id}.")
    return x, observed, valid


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


model = UNet().to(device)
checkpoint = torch.load(paths["checkpoint"], map_location=device, weights_only=False)
if checkpoint.get("variant") != "without_terrain":
    raise RuntimeError("The selected checkpoint is not the without-terrain model.")
model.load_state_dict(checkpoint["model_state_dict"]); model.eval()

# Eight dihedral transforms. Every output is inverse-transformed to the original grid.
TRANSFORMS = [(k, flip) for k in range(4) for flip in [False, True]]


def transform_array(array, k, flip):
    result = np.rot90(array, k, axes=(-2,-1))
    if flip: result = result[..., :, ::-1]
    return np.ascontiguousarray(result)


def inverse_array(array, k, flip):
    result = array[..., :, ::-1] if flip else array
    return np.ascontiguousarray(np.rot90(result, -k, axes=(-2,-1)))


def tta_predict(row):
    x, observed, valid = load_scene(row)
    predictions = []
    gs = group_stats[(row.city,row.thermal_sensor)]
    with torch.no_grad():
        for k, flip in TRANSFORMS:
            augmented = transform_array(x,k,flip)
            tensor = torch.from_numpy(augmented[None]).to(device)
            residual = model(tensor).cpu().numpy()[0,0]
            residual = inverse_array(residual,k,flip)
            predictions.append(residual*gs["std_c"]+gs["mean_c"])
    stack = np.stack(predictions).astype("float32")
    return observed, valid, stack.mean(axis=0), stack.std(axis=0,ddof=1)


cache = {}
for split_name, frame in [("validation",validation),("temporal_test",test)]:
    for number,row in enumerate(frame.itertuples(index=False),start=1):
        cache[row.pilot_id] = tta_predict(row)
        print(f"TTA {split_name} {number:02d}/8: {row.pilot_id}")

# Equal scene sampling prevents high-coverage scenes dominating conformal calibration.
scores=[]; calibration_rng=np.random.default_rng(SEED)
for row in validation.itertuples(index=False):
    observed,valid,mean,std=cache[row.pilot_id]
    indices=np.flatnonzero(valid); size=min(CALIBRATION_PIXELS_PER_SCENE,len(indices))
    chosen=calibration_rng.choice(indices,size=size,replace=False)
    scale=std.ravel()[chosen]+UNCERTAINTY_FLOOR_C
    scores.append(np.abs(observed.ravel()[chosen]-mean.ravel()[chosen])/scale)
scores=np.concatenate(scores)
n=len(scores); rank=min(int(np.ceil((n+1)*TARGET_COVERAGE)),n)
conformal_q=float(np.partition(scores,rank-1)[rank-1])
print(f"Validation-only conformal multiplier: {conformal_q:.4f}")


def metrics(observed,predicted):
    error=predicted.astype("float64")-observed.astype("float64")
    denominator=np.sum((observed-observed.mean())**2)
    return {"pixels":len(observed),"mae_c":np.mean(np.abs(error)),"rmse_c":np.sqrt(np.mean(error**2)),
            "bias_c":np.mean(error),"r2":1-np.sum(error**2)/denominator if denominator>0 else np.nan}


scene_rows=[]; sampled_rows=[]
for split_name,frame in [("validation",validation),("temporal_test",test)]:
    for row in frame.itertuples(index=False):
        observed,valid,mean,std=cache[row.pilot_id]
        half_width=conformal_q*(std+UNCERTAINTY_FLOOR_C)
        lower=mean-half_width; upper=mean+half_width
        y=observed[valid]; prediction=mean[valid]; uncertainty=std[valid]; width=2*half_width[valid]
        covered=(y>=lower[valid])&(y<=upper[valid]); absolute_error=np.abs(prediction-y)
        correlation,p_value=spearmanr(uncertainty,absolute_error)
        scene_rows.append({"model_split":split_name,"pilot_id":row.pilot_id,"city":row.city,
            "thermal_sensor":row.thermal_sensor,**metrics(y,prediction),"interval_coverage":covered.mean(),
            "mean_interval_width_c":width.mean(),"median_tta_std_c":np.median(uncertainty),
            "uncertainty_error_spearman":correlation,"uncertainty_error_p":p_value})
        choose=np.random.default_rng(SEED+sum(map(ord,row.pilot_id))).choice(
            np.arange(len(y)),size=min(20000,len(y)),replace=False)
        sampled_rows.append(pd.DataFrame({"model_split":split_name,"pilot_id":row.pilot_id,
            "city":row.city,"thermal_sensor":row.thermal_sensor,"observed_c":y[choose],
            "predicted_c":prediction[choose],"absolute_error_c":absolute_error[choose],
            "tta_std_c":uncertainty[choose],"covered":covered[choose]}))
        if split_name=="temporal_test":
            np.savez_compressed(PRED_DIR/f"{row.pilot_id}_final.npz",observed_c=observed,
                predicted_c=mean,tta_std_c=std,interval_lower_c=lower.astype("float32"),
                interval_upper_c=upper.astype("float32"),valid_mask=valid.astype("uint8"))

scene_metrics=pd.DataFrame(scene_rows); scene_metrics.to_csv(OUT/"01_scene_accuracy_uncertainty.csv",index=False)
pixels=pd.concat(sampled_rows,ignore_index=True)
aggregate_rows=[]
for split_name,group in pixels.groupby("model_split",observed=True):
    correlation,p_value=spearmanr(group.tta_std_c,group.absolute_error_c)
    aggregate_rows.append({"model_split":split_name,**metrics(group.observed_c.to_numpy(),group.predicted_c.to_numpy()),
        "interval_coverage":group.covered.mean(),"mean_tta_std_c":group.tta_std_c.mean(),
        "uncertainty_error_spearman":correlation,"uncertainty_error_p":p_value})
aggregate=pd.DataFrame(aggregate_rows); aggregate.to_csv(OUT/"02_aggregate_accuracy_uncertainty.csv",index=False)

# Reliability bins are computed separately for validation and temporal test.
reliability=[]
for split_name,group in pixels.groupby("model_split",observed=True):
    bins=pd.qcut(group.tta_std_c,10,duplicates="drop")
    table=group.assign(uncertainty_bin=bins).groupby("uncertainty_bin",observed=True).agg(
        pixels=("absolute_error_c","size"),mean_tta_std_c=("tta_std_c","mean"),
        mean_absolute_error_c=("absolute_error_c","mean"),coverage=("covered","mean")).reset_index()
    table["uncertainty_bin"]=table.uncertainty_bin.astype(str); table["model_split"]=split_name
    reliability.append(table)
reliability=pd.concat(reliability,ignore_index=True); reliability.to_csv(OUT/"03_uncertainty_reliability.csv",index=False)

# Figure 1: final quantitative performance.
sns.set_theme(style="whitegrid",context="notebook")
fig,axes=plt.subplots(1,3,figsize=(20,6),constrained_layout=True)
test_pixels=pixels.query("model_split=='temporal_test'")
plot=test_pixels.sample(min(100000,len(test_pixels)),random_state=SEED)
sns.scatterplot(data=plot,x="observed_c",y="predicted_c",hue="thermal_sensor",alpha=.18,s=10,linewidth=0,ax=axes[0])
limits=[min(plot.observed_c.min(),plot.predicted_c.min()),max(plot.observed_c.max(),plot.predicted_c.max())]
axes[0].plot(limits,limits,"k--",lw=1); axes[0].set(xlim=limits,ylim=limits,title="(a) Final temporal-test predictions",
    xlabel="Observed LST (°C)",ylabel="Predicted LST (°C)"); axes[0].legend(frameon=False)
sns.boxplot(data=scene_metrics.query("model_split=='temporal_test'"),x="thermal_sensor",y="rmse_c",ax=axes[1])
sns.stripplot(data=scene_metrics.query("model_split=='temporal_test'"),x="thermal_sensor",y="rmse_c",color="black",size=5,ax=axes[1])
axes[1].set(title="(b) Scene-level accuracy",xlabel="",ylabel="RMSE (°C)")
test_rel=reliability.query("model_split=='temporal_test'")
axes[2].plot(test_rel.mean_tta_std_c,test_rel.mean_absolute_error_c,"o-",color="#b42c3b")
axes[2].set(title="(c) Uncertainty–error reliability",xlabel="Mean TTA standard deviation (°C)",ylabel="Mean absolute error (°C)")
fig.savefig(OUT/"thermofusion_stage17_final_performance.png",dpi=800,bbox_inches="tight",facecolor="white"); plt.show()

# Figure 2: one row per locked temporal-test scene.
test_order=test.sort_values(["city","thermal_sensor"]).reset_index(drop=True)
fig,axes=plt.subplots(8,4,figsize=(14,27),constrained_layout=True)
all_observed=np.concatenate([cache[row.pilot_id][0][cache[row.pilot_id][1]] for row in test_order.itertuples()])
vmin,vmax=np.percentile(all_observed,[2,98])
for i,row in enumerate(test_order.itertuples(index=False)):
    observed,valid,mean,std=cache[row.pilot_id]; error=np.abs(mean-observed)
    panels=[np.where(valid,observed,np.nan),np.where(valid,mean,np.nan),np.where(valid,error,np.nan),np.where(valid,std,np.nan)]
    titles=["Observed LST","Predicted LST","Absolute error","TTA uncertainty"]
    for j,(panel,title) in enumerate(zip(panels,titles)):
        kwargs={"cmap":"inferno"} if j<2 else {"cmap":"magma"}
        if j<2: kwargs.update(vmin=vmin,vmax=vmax)
        axes[i,j].imshow(panel,**kwargs); axes[i,j].set_xticks([]); axes[i,j].set_yticks([])
        if i==0: axes[i,j].set_title(title)
    axes[i,0].set_ylabel(f"{row.city}\n{row.thermal_sensor}\n{row.pilot_id}")
fig.savefig(OUT/"thermofusion_stage17_temporal_test_atlas.png",dpi=800,bbox_inches="tight",facecolor="white"); plt.show()

test_result=aggregate.query("model_split=='temporal_test'").iloc[0]
validation_result=aggregate.query("model_split=='validation'").iloc[0]
execution_pass=(len(scene_metrics)==16 and len(list(PRED_DIR.glob("*.npz")))==8
    and np.isfinite(scene_metrics.select_dtypes(include=np.number)).all().all()
    and 0<=test_result.interval_coverage<=1 and conformal_q>0)
verdict=["THERMOFUSION STAGE 17 FINAL MODEL AND UNCERTAINTY",
    "Final configuration: metadata-conditioned residual U-Net without terrain",
    "Architecture selection source: validation performance in Stage 16",
    f"Test-time transformations: {len(TRANSFORMS)}",f"Nominal prediction-interval coverage: {TARGET_COVERAGE:.0%}",
    f"Validation-only conformal multiplier: {conformal_q:.4f}",
    f"Temporal-test MAE (°C): {test_result.mae_c:.4f}",f"Temporal-test RMSE (°C): {test_result.rmse_c:.4f}",
    f"Temporal-test bias (°C): {test_result.bias_c:.4f}",f"Temporal-test R²: {test_result.r2:.4f}",
    f"Temporal-test interval coverage: {test_result.interval_coverage:.4f}",
    f"Temporal-test uncertainty-error Spearman rho: {test_result.uncertainty_error_spearman:.4f}",
    "Temporal-test data used for architecture selection or interval calibration: NO",
    "Overall Stage 17 execution: PASS" if execution_pass else "Overall Stage 17 execution: REVIEW REQUIRED"]
(OUT/"04_stage17_verdict.txt").write_text("\n".join(verdict),encoding="utf-8")
with (OUT/"05_configuration.json").open("w") as handle:
    json.dump({"seed":SEED,"selected_variant":"without_terrain","tta_transforms":len(TRANSFORMS),
        "target_coverage":TARGET_COVERAGE,"uncertainty_floor_c":UNCERTAINTY_FLOOR_C,
        "calibration_pixels_per_scene":CALIBRATION_PIXELS_PER_SCENE,"conformal_multiplier":conformal_q},handle,indent=2)
zip_path=shutil.make_archive(str(ROOT/"ThermoFusion_Stage17_Final_Uncertainty"),"zip",root_dir=OUT)
print("\n"+"\n".join(verdict)); print(f"\nOutput folder: {OUT}"); print(f"ZIP package: {zip_path}")
