# =====================================================================
# ThermoFusion reproducibility code
# Source notebook cell(s): 34
# Audit status: ARCHIVE
# Extracted without scientific-logic modification.
# Replace local Google Drive paths as described in README.md when needed.
# =====================================================================

# ThermoFusion Stage 19: publication-grade figures and tables from verified outputs.
# Run in one Google Colab cell after Stage 17. GPU is not required.

from google.colab import drive
from pathlib import Path
import json, shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from openpyxl.styles import Font, PatternFill

drive.mount('/content/drive')
ROOT = Path('/content/drive/MyDrive')
S12 = ROOT/'ThermoFusion_Stage12_ModelReady_Pilot'
S13 = ROOT/'ThermoFusion_Stage13_Pilot_Baseline'
S14 = ROOT/'ThermoFusion_Stage14_Refined_Benchmark'
S15 = ROOT/'ThermoFusion_Stage15_Calibration_Bootstrap'
S16 = ROOT/'ThermoFusion_Stage16_Modality_Ablation'
S17 = ROOT/'ThermoFusion_Stage17_Final_Uncertainty'
OUT = ROOT/'ThermoFusion_Stage19_Publication_Graphics'
PNG, VEC, TAB = OUT/'figures_1200dpi', OUT/'figures_vector', OUT/'tables'
for folder in (PNG,VEC,TAB):
    folder.mkdir(parents=True,exist_ok=True)
    for old in folder.glob('*'): old.unlink()

DPI=900; SEED=20260908
COL={'Accra':'#3B6FB6','Lagos':'#E07A3F','Abidjan':'#3A9D6F','Freetown':'#B64E5A'}
mpl.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':11,
 'axes.labelsize':9.5,'xtick.labelsize':8.5,'ytick.labelsize':8.5,
 'legend.fontsize':8,'axes.linewidth':.8,'savefig.dpi':DPI,'pdf.fonttype':42,
 'ps.fonttype':42,'svg.fonttype':'none'})
sns.set_theme(style='ticks',context='paper')

def need(path):
    if not path.exists(): raise FileNotFoundError(f'Required verified output missing: {path}')
    return path
def read(path,columns):
    frame=pd.read_csv(need(path)); missing=set(columns)-set(frame.columns)
    if missing: raise RuntimeError(f'{path.name} missing columns: {sorted(missing)}')
    return frame
def save(fig,stem,vector=True):
    fig.savefig(PNG/f'{stem}.png',dpi=DPI,bbox_inches='tight',facecolor='white')
    fig.savefig(VEC/f'{stem}.pdf',bbox_inches='tight',facecolor='white')
    if vector: fig.savefig(VEC/f'{stem}.svg',bbox_inches='tight',facecolor='white')
    plt.show(); plt.close(fig)
def one(frame,label):
    if len(frame)!=1: raise RuntimeError(f'Expected one {label} row; found {len(frame)}')
    return frame.iloc[0]
def panel(ax,label): ax.text(-.09,1.04,label,transform=ax.transAxes,fontweight='bold',va='bottom')

splits=read(S12/'01_scene_split_manifest.csv',['pilot_id','city','thermal_sensor','model_split'])
a13=read(S13/'04_aggregate_metrics.csv',['model_split','group','mae_c','rmse_c','bias_c','r2'])
a14=read(S14/'05_aggregate_model_comparison.csv',['model_split','model','group','mae_c','rmse_c','bias_c','r2'])
s15=read(S15/'04_scene_evaluation.csv',['pilot_id','model_split','method','rmse_c','mae_c'])
b15=read(S15/'06_paired_scene_bootstrap.csv',['metric','calibrated_minus_climatology','ci_lower','ci_upper'])
s16=read(S16/'01_scene_ablation_metrics.csv',['pilot_id','model_split','variant','rmse_c'])
a16=read(S16/'03_ablation_summary.csv',['variant','model_split','scene_macro_rmse_c','delta_rmse_vs_full_c'])
b16=read(S16/'04_paired_ablation_bootstrap.csv',['variant','mean_delta_rmse_c','ci_lower','ci_upper','probability_ablation_worse'])
s17=read(S17/'01_scene_accuracy_uncertainty.csv',['pilot_id','model_split','city','thermal_sensor','rmse_c','mae_c','bias_c','r2','interval_coverage','mean_interval_width_c','median_tta_std_c'])
a17=read(S17/'02_aggregate_accuracy_uncertainty.csv',['model_split','mae_c','rmse_c','bias_c','r2','interval_coverage','mean_tta_std_c','uncertainty_error_spearman'])
rel=read(S17/'03_uncertainty_reliability.csv',['model_split','mean_tta_std_c','mean_absolute_error_c','coverage'])
pred_dir=need(S17/'temporal_test_predictions')
if splits.model_split.value_counts().to_dict()!={'train':48,'validation':8,'temporal_test':8}:
    raise RuntimeError('Scene split is not the verified 48/8/8 design.')

# Exact tables used by figures.
tables={
 'Table_01_scene_design':splits,
 'Table_02_stage13_aggregate':a13,
 'Table_03_stage14_comparison':a14,
 'Table_04_calibration_scene_metrics':s15,
 'Table_05_ablation_scene_metrics':s16,
 'Table_06_ablation_bootstrap':b16,
 'Table_07_final_scene_accuracy_uncertainty':s17,
 'Table_08_uncertainty_reliability':rel,
}
for name,frame in tables.items(): frame.to_csv(TAB/f'{name}.csv',index=False)
with pd.ExcelWriter(OUT/'ThermoFusion_Publication_Tables.xlsx',engine='openpyxl') as writer:
    for name,frame in tables.items():
        frame.to_excel(writer,sheet_name=name.replace('Table_','T')[:31],index=False)
        ws=writer.book[name.replace('Table_','T')[:31]]; ws.freeze_panes='A2'; ws.auto_filter.ref=ws.dimensions
        for c in ws[1]:
            c.font=Font(name='Calibri',size=11,bold=True,color='FFFFFF')
            c.fill=PatternFill('solid',fgColor='17365D')
        for cells in ws.columns:
            ws.column_dimensions[cells[0].column_letter].width=min(42,max(len(str(c.value or '')) for c in cells)+2)

# Figure 1: experimental design and leakage-control topology.
fig=plt.figure(figsize=(15,8)); gs=fig.add_gridspec(2,3,height_ratios=[1,1.15],wspace=.28,hspace=.35)
ax=fig.add_subplot(gs[0,:]); ax.axis('off')
nodes=[(.04,'64 verified\ncity–sensor scenes','#24476B'),(.25,'48 training\nmodel fitting','#2F7D69'),(.47,'8 validation\nearly stopping + selection','#C9822B'),(.70,'Frozen model\n8-view TTA','#684A8A'),(.90,'8 temporal tests\nfinal inference only','#A33F48')]
for x,t,c in nodes:
    ax.text(x,.5,t,ha='center',va='center',color='white',fontweight='bold',transform=ax.transAxes,
            bbox=dict(boxstyle='round,pad=.65',fc=c,ec='white',lw=1.5))
for (x1,_,_),(x2,_,_) in zip(nodes[:-1],nodes[1:]):
    ax.annotate('',xy=(x2-.075,.5),xytext=(x1+.075,.5),xycoords=ax.transAxes,
                arrowprops=dict(arrowstyle='-|>',lw=1.5,color='#555'))
ax.text(.5,.08,'Temporal-test scenes never enter fitting, early stopping, architecture selection, or interval calibration',ha='center',transform=ax.transAxes,fontweight='bold')
count=(splits.groupby(['city','thermal_sensor','model_split']).size().unstack(fill_value=0)
       .reindex(columns=['train','validation','temporal_test']))
ax1=fig.add_subplot(gs[1,0]); sns.heatmap(count,annot=True,fmt='g',cmap='Blues',cbar=False,linewidths=.8,ax=ax1)
ax1.set(xlabel='Scene role',ylabel='City–sensor stratum',title='(a) Balanced scene-disjoint design')
ax2=fig.add_subplot(gs[1,1]); sensor=splits.groupby(['city','thermal_sensor']).size().unstack()
sensor.plot(kind='bar',stacked=True,color=['#6A4C93','#1982C4'],ax=ax2,width=.72)
ax2.set(title='(b) City and thermal-sensor representation',xlabel='',ylabel='Scenes'); ax2.legend(frameon=False)
ax3=fig.add_subplot(gs[1,2]); vals=np.array([[16,9,25],[48,8,8],[8,8,8]])
sns.heatmap(vals,annot=True,fmt='g',cmap='mako',cbar=False,linewidths=.8,ax=ax3,
            xticklabels=['Raster','Context','Total/stratum'],yticklabels=['Input channels','Train/val/test','City/sensor/TTA'])
ax3.set(title='(c) Analysis dimensionality',xlabel='',ylabel='')
save(fig,'Figure_01_Experimental_Design_and_Leakage_Control')

# Figure 2: performance evolution with four complementary metrics.
base=one(a13.query("model_split=='temporal_test' and group=='ALL'"),'Stage 13 test')
refn=one(a14.query("model_split=='temporal_test' and model=='refined_unet' and group=='ALL'"),'Stage 14 test')
final=one(a17.query("model_split=='temporal_test'"),'Stage 17 test')
perf=pd.DataFrame([
 ['Baseline U-Net',base.mae_c,base.rmse_c,base.bias_c,base.r2],
 ['Metadata-conditioned U-Net',refn.mae_c,refn.rmse_c,refn.bias_c,refn.r2],
 ['Validation-selected no-terrain + TTA',final.mae_c,final.rmse_c,final.bias_c,final.r2]],
 columns=['model','MAE','RMSE','Bias','R2'])
perf.to_csv(TAB/'Table_09_model_progression.csv',index=False)
tables['Table_09_model_progression']=perf
# Rewrite after Table 9 exists so the workbook contains every publication table.
with pd.ExcelWriter(OUT/'ThermoFusion_Publication_Tables.xlsx',engine='openpyxl') as writer:
    for name,frame in tables.items():
        sheet=name.replace('Table_','T')[:31]; frame.to_excel(writer,sheet_name=sheet,index=False)
        ws=writer.book[sheet]; ws.freeze_panes='A2'; ws.auto_filter.ref=ws.dimensions
        for c in ws[1]:
            c.font=Font(name='Calibri',size=11,bold=True,color='FFFFFF'); c.fill=PatternFill('solid',fgColor='17365D')
        for cells in ws.columns:
            ws.column_dimensions[cells[0].column_letter].width=min(42,max(len(str(c.value or '')) for c in cells)+2)
fig,axs=plt.subplots(2,2,figsize=(13,9),constrained_layout=True)
for ax,metric,lab,col in zip(axs.flat,['RMSE','MAE','Bias','R2'],['(a)','(b)','(c)','(d)'],['#355C7D','#2A9D8F','#E76F51','#6A4C93']):
    y=np.arange(3); ax.hlines(y,0,perf[metric],color='#D6DCE4',lw=7); ax.plot(perf[metric],y,'o',ms=9,color=col)
    ax.set_yticks(y,perf.model); ax.invert_yaxis(); ax.set_xlabel(metric+(' (°C)' if metric!='R2' else ''))
    ax.set_title(f'{lab} Temporal-test {metric}'); ax.grid(axis='x',alpha=.25)
    for yy,v in zip(y,perf[metric]): ax.text(v,yy,f'  {v:.3f}',va='center',fontweight='bold')
save(fig,'Figure_02_Model_Progression_Multimetric')

# Figure 3: ablation effect sizes, scene heterogeneity and validation/test ranks.
wide=s16.query("model_split=='temporal_test'").pivot(index='pilot_id',columns='variant',values='rmse_c')
deltas=wide.subtract(wide.full_model,axis=0).drop(columns='full_model')
forest=b16.sort_values('mean_delta_rmse_c')
fig=plt.figure(figsize=(16,6)); gs=fig.add_gridspec(1,3,width_ratios=[1.05,1.5,1])
ax=fig.add_subplot(gs[0,0]); y=np.arange(len(forest))
ax.errorbar(forest.mean_delta_rmse_c,y,xerr=[forest.mean_delta_rmse_c-forest.ci_lower,forest.ci_upper-forest.mean_delta_rmse_c],fmt='o',capsize=4,color='#A33F48')
ax.axvline(0,color='k',ls='--'); ax.set_yticks(y,forest.variant.str.replace('_',' ')); ax.set(xlabel='Ablation − full model RMSE (°C)',title='(a) Paired bootstrap effects')
ax=fig.add_subplot(gs[0,1]); sns.heatmap(deltas.T,cmap='vlag',center=0,annot=True,fmt='.2f',linewidths=.5,cbar_kws={'label':'Δ scene RMSE (°C)'},ax=ax)
ax.set(title='(b) Scene-specific modality dependence',xlabel='Temporal-test scene',ylabel='')
ax=fig.add_subplot(gs[0,2]); rank=a16.pivot(index='variant',columns='model_split',values='scene_macro_rmse_c')
ax.scatter(rank.validation,rank.temporal_test,s=75,c='#355C7D')
for name,row in rank.iterrows(): ax.annotate(name.replace('_',' '),(row.validation,row.temporal_test),xytext=(4,3),textcoords='offset points',fontsize=7)
lo=min(rank.min())-.05; hi=max(rank.max())+.05; ax.plot([lo,hi],[lo,hi],'k--',lw=1); ax.set(xlim=(lo,hi),ylim=(lo,hi),xlabel='Validation scene-macro RMSE (°C)',ylabel='Temporal-test scene-macro RMSE (°C)',title='(c) Rank transfer across splits')
save(fig,'Figure_03_Modality_Ablation_and_Transfer')

# Load exact final rasters once for density, extremes, reliability and atlas.
meta=s17.query("model_split=='temporal_test'").set_index('pilot_id'); stacks=[]; rasters={}
rng=np.random.default_rng(SEED)
for pid,row in meta.iterrows():
    path=need(pred_dir/f'{pid}_final.npz')
    with np.load(path) as z:
        required={'observed_c','predicted_c','tta_std_c','interval_lower_c','interval_upper_c','valid_mask'}
        if not required.issubset(z.files): raise RuntimeError(f'{path.name} lacks {sorted(required-set(z.files))}')
        d={k:z[k] for k in required}; rasters[pid]=d
        valid=d['valid_mask'].astype(bool)&np.isfinite(d['observed_c'])&np.isfinite(d['predicted_c'])
        idx=np.flatnonzero(valid); idx=rng.choice(idx,min(25000,len(idx)),replace=False)
        stacks.append(pd.DataFrame({'pilot_id':pid,'city':row.city,'thermal_sensor':row.thermal_sensor,
          'observed_c':d['observed_c'].ravel()[idx],'predicted_c':d['predicted_c'].ravel()[idx],
          'uncertainty_c':d['tta_std_c'].ravel()[idx]}))
pix=pd.concat(stacks,ignore_index=True); pix['error_c']=pix.predicted_c-pix.observed_c; pix['abs_error_c']=pix.error_c.abs()

# Figure 4: sensor-specific density, conditional bias and extremes response.
fig=plt.figure(figsize=(15,10)); gs=fig.add_gridspec(2,3)
for j,sensor in enumerate(['Landsat','ECOSTRESS']):
    g=pix.query('thermal_sensor==@sensor'); ax=fig.add_subplot(gs[0,j]); h=ax.hexbin(g.observed_c,g.predicted_c,gridsize=70,bins='log',mincnt=1,cmap='magma',rasterized=True)
    lo=min(g.observed_c.min(),g.predicted_c.min()); hi=max(g.observed_c.max(),g.predicted_c.max()); ax.plot([lo,hi],[lo,hi],'w--',lw=1.2)
    ax.set(xlabel='Observed LST (°C)',ylabel='Predicted LST (°C)',title=f"({'ab'[j]}) {sensor}: density and 1:1 agreement"); fig.colorbar(h,ax=ax,label='log pixel density')
ax=fig.add_subplot(gs[0,2]); bins=pd.qcut(pix.observed_c,12,duplicates='drop'); q=pix.assign(bin=bins).groupby(['thermal_sensor','bin'],observed=True).agg(obs=('observed_c','mean'),bias=('error_c','mean'),mae=('abs_error_c','mean')).reset_index()
for sensor,g in q.groupby('thermal_sensor'): ax.plot(g.obs,g.bias,'o-',label=sensor)
ax.axhline(0,color='k',ls='--'); ax.set(title='(c) Conditional bias across thermal range',xlabel='Observed LST-bin mean (°C)',ylabel='Prediction bias (°C)'); ax.legend(frameon=False)
ax=fig.add_subplot(gs[1,:2]); sns.violinplot(data=pix.sample(min(100000,len(pix)),random_state=SEED),x='thermal_sensor',y='error_c',hue='city',palette=COL,cut=0,inner='quart',ax=ax)
ax.axhline(0,color='k',ls='--'); ax.set(title='(d) Cross-city residual distributions',xlabel='',ylabel='Prediction error (°C)'); ax.legend(frameon=False,ncol=4)
ax=fig.add_subplot(gs[1,2]); for_plot=q.copy()
for sensor,g in for_plot.groupby('thermal_sensor'): ax.plot(g.obs,g.mae,'o-',label=sensor)
ax.set(title='(e) Error amplification at thermal extremes',xlabel='Observed LST-bin mean (°C)',ylabel='MAE (°C)'); ax.legend(frameon=False)
save(fig,'Figure_04_Thermal_Response_and_Extreme_Behaviour')

# Figure 5: uncertainty discrimination, calibration and selective prediction.
fig,axs=plt.subplots(2,2,figsize=(13,10),constrained_layout=True); testrel=rel.query("model_split=='temporal_test'").sort_values('mean_tta_std_c')
axs[0,0].plot(testrel.mean_tta_std_c,testrel.mean_absolute_error_c,'o-',color='#A33F48'); axs[0,0].set(title='(a) Uncertainty–error reliability',xlabel='Mean TTA uncertainty (°C)',ylabel='MAE (°C)')
sp=s17.query("model_split=='temporal_test'").sort_values('interval_coverage'); axs[0,1].barh(sp.pilot_id,sp.interval_coverage,color=[COL[x] for x in sp.city]); axs[0,1].axvline(.9,color='k',ls='--'); axs[0,1].set(xlim=(0,1.02),title='(b) Scene-wise interval coverage',xlabel='Coverage',ylabel='')
axs[1,0].scatter(sp.mean_interval_width_c,sp.interval_coverage,c=[COL[x] for x in sp.city],s=70); axs[1,0].axhline(.9,color='k',ls='--'); axs[1,0].set(title='(c) Sharpness–coverage trade-off',xlabel='Mean interval width (°C)',ylabel='Coverage')
ordered=pix.sort_values('uncertainty_c'); fractions=np.linspace(.1,1,19); risks=[ordered.iloc[:max(1,int(len(ordered)*f))].abs_error_c.mean() for f in fractions]
axs[1,1].plot(fractions,risks,'o-',color='#355C7D'); axs[1,1].set(title='(d) Selective-prediction risk–coverage curve',xlabel='Fraction of least-uncertain pixels retained',ylabel='MAE (°C)')
save(fig,'Figure_05_Uncertainty_Calibration_and_Selectivity')

# Figure 6: spatially explicit photogrammetric-style diagnostic atlas.
pids=list(meta.sort_values(['city','thermal_sensor']).index); fig,axs=plt.subplots(len(pids),4,figsize=(8,16),constrained_layout=True)
obs_all=np.concatenate([rasters[p]['observed_c'][rasters[p]['valid_mask'].astype(bool)] for p in pids]); vmin,vmax=np.nanpercentile(obs_all,[2,98])
for i,pid in enumerate(pids):
    d=rasters[pid]; valid=d['valid_mask'].astype(bool); obs=np.where(valid,d['observed_c'],np.nan); pred=np.where(valid,d['predicted_c'],np.nan); err=np.where(valid,np.abs(pred-obs),np.nan); unc=np.where(valid,d['tta_std_c'],np.nan)
    arrays=[obs,pred,err,unc]; cmaps=['inferno','inferno','magma','viridis']; ranges=[(vmin,vmax),(vmin,vmax),(0,np.nanpercentile(err,98)),(0,np.nanpercentile(unc,98))]
    for j,(arr,cmap,rr) in enumerate(zip(arrays,cmaps,ranges)):
        im=axs[i,j].imshow(arr,cmap=cmap,vmin=rr[0],vmax=rr[1]); axs[i,j].set_xticks([]); axs[i,j].set_yticks([])
        if i==0: axs[i,j].set_title(['Observed LST','Predicted LST','Absolute error','TTA uncertainty'][j],fontweight='bold')
        if j==0: axs[i,j].set_ylabel(f"{meta.loc[pid,'city']}\n{meta.loc[pid,'thermal_sensor']}\n{pid}",rotation=0,ha='right',va='center')
        if j>=2: fig.colorbar(im,ax=axs[i,j],fraction=.035,pad=.015)
save(fig,'Figure_06_Spatial_Prediction_Error_Uncertainty_Atlas',vector=False)

checks={'figures_png':len(list(PNG.glob('*.png')))==6,'figure_pdfs':len(list(VEC.glob('*.pdf')))==6,
 'vector_svgs':len(list(VEC.glob('*.svg')))==5,'csv_tables':len(list(TAB.glob('*.csv')))==9,
 'test_scenes':len(meta)==8,'finite_final_metrics':np.isfinite(a17.select_dtypes('number')).all().all()}
pd.DataFrame([{'check':k,'passed':v} for k,v in checks.items()]).to_csv(OUT/'publication_output_checks.csv',index=False)
verdict=['THERMOFUSION STAGE 19 PUBLICATION GRAPHICS',f'900-dpi figures: {len(list(PNG.glob("*.png")))}/6',
 f'PDF figures: {len(list(VEC.glob("*.pdf")))}/6',f'Editable SVG charts: {len(list(VEC.glob("*.svg")))}/5',
 f'Publication tables: {len(list(TAB.glob("*.csv")))}/9','Verified analysis values changed: NO',
 'Overall publication-graphics verdict: PASS' if all(checks.values()) else 'Overall publication-graphics verdict: REVIEW REQUIRED']
(OUT/'publication_graphics_verdict.txt').write_text('\n'.join(verdict),encoding='utf-8')
with (OUT/'publication_graphics_configuration.json').open('w') as f: json.dump({'dpi':DPI,'seed':SEED,'source_stages':[12,13,14,15,16,17]},f,indent=2)
zip_path=shutil.make_archive(str(ROOT/'ThermoFusion_Stage19_Publication_Graphics'),'zip',root_dir=OUT)
print('\n'+'\n'.join(verdict)); print(f'\nOutput folder: {OUT}\nZIP package: {zip_path}')
