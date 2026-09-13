# =====================================================================
# ThermoFusion reproducibility code
# Source notebook cell(s): 38
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
sites=pd.DataFrame([
 ['Abidjan',5.3600,-4.0083,"Côte d'Ivoire",16,8,8],['Accra',5.6037,-0.1870,'Ghana',16,8,8],
 ['Freetown',8.4657,-13.2317,'Sierra Leone',16,8,8],['Lagos',6.5244,3.3792,'Nigeria',16,8,8]],
 columns=['city','latitude','longitude','country','scenes','landsat_scenes','ecostress_scenes'])
tables={
 'Table_00_study_sites':sites,
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

# Figure 0: geographic study-area context and sampling design.
try:
    import geopandas as gpd
except ImportError:
    import subprocess,sys
    subprocess.check_call([sys.executable,'-m','pip','install','-q','geopandas','pyogrio'])
    import geopandas as gpd
world_url='https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip'
world=gpd.read_file(world_url); africa=world.loc[world.CONTINENT.eq('Africa')]
fig=plt.figure(figsize=(15,6.5)); gs=fig.add_gridspec(1,3,width_ratios=[.8,1.65,1],wspace=.16)
ax=fig.add_subplot(gs[0,0]); africa.plot(ax=ax,color='#E8ECEF',edgecolor='#7B8790',lw=.35); ax.set(xlim=(-20,55),ylim=(-36,38),title='(a) African context'); ax.axis('off'); ax.add_patch(mpl.patches.Rectangle((-16,4),22,8,fill=False,ec='#A33F48',lw=1.4))
ax=fig.add_subplot(gs[0,1]); africa.plot(ax=ax,color='#F0F2F3',edgecolor='#79848C',lw=.45)
focus=world.loc[world.ADMIN.isin(["Côte d'Ivoire",'Ghana','Sierra Leone','Nigeria'])]; focus.plot(ax=ax,color='#D9E7F2',edgecolor='#24476B',lw=.8)
for _,r in sites.iterrows(): ax.scatter(r.longitude,r.latitude,s=80,c=COL[r.city],edgecolor='white',lw=.8,zorder=4); ax.annotate(r.city,(r.longitude,r.latitude),xytext=(5,5),textcoords='offset points',fontweight='bold',fontsize=8)
ax.set(xlim=(-16,7),ylim=(3.5,11.5),xlabel='Longitude',ylabel='Latitude',title='(b) Four West African coastal-city study sites'); ax.grid(ls=':',lw=.5,alpha=.55)
ax=fig.add_subplot(gs[0,2]); ax.axis('off'); ax.set_title('(c) Balanced observation design')
for i,r in sites.iterrows():
    y=.84-i*.2; ax.add_patch(mpl.patches.Circle((.1,y),.035,transform=ax.transAxes,fc=COL[r.city],ec='white')); ax.text(.17,y,r.city,transform=ax.transAxes,va='center',fontweight='bold'); ax.text(.17,y-.055,f"{r.country}  |  {r.latitude:.2f}°N, {abs(r.longitude):.2f}°{'W' if r.longitude<0 else 'E'}",transform=ax.transAxes,va='center',fontsize=7,color='#444'); ax.text(.72,y,'8 Landsat\n8 ECOSTRESS',transform=ax.transAxes,ha='center',va='center',fontsize=7,bbox=dict(boxstyle='round',fc='#EEF3F7',ec='#9BAAB5'))
ax.text(.5,.04,'64 scenes  •  4 cities  •  2 thermal sensors\n48 training  •  8 validation  •  8 temporal test',transform=ax.transAxes,ha='center',fontweight='bold',bbox=dict(boxstyle='round,pad=.5',fc='#17365D',ec='none'),color='white')
save(fig,'Figure_00_Study_Area_and_Observation_Design')

# Figure 1: multimodal architecture, information flow and leakage firewall.
'''
fig=plt.figure(figsize=(16,10)); gs=fig.add_gridspec(2,3,height_ratios=[1.18,1],width_ratios=[1.25,1,1],wspace=.32,hspace=.32)
ax=fig.add_subplot(gs[0,:2]); ax.set_xlim(0,10); ax.set_ylim(0,7); ax.axis('off')
streams=[(5.8,'Sentinel-2 optical\n10 reflectance/index channels','#2878B5'),(4.35,'Sentinel-1 SAR\nVV, VH and polarization contrast','#694C9A'),(2.9,'Terrain\nelevation, slope and aspect','#4B8B3B'),(1.45,'Known context\ncity, sensor and acquisition cycle','#D48627')]
for y,t,c in streams:
    ax.add_patch(mpl.patches.FancyBboxPatch((.15,y-.43),2.55,.86,boxstyle='round,pad=.08',fc=c,ec='white',lw=1.2)); ax.text(1.43,y,t,ha='center',va='center',color='white',fontweight='bold',fontsize=8)
    ax.annotate('',xy=(3.25,3.65),xytext=(2.72,y),arrowprops=dict(arrowstyle='-|>',color=c,lw=1.5))
blocks=[(3.25,2.65,1.45,2.0,'Modality-aware\nmasking + scaling','#355C7D'),(5.15,2.35,1.7,2.6,'Metadata-conditioned\nresidual U-Net\nencoder ↔ decoder','#235347'),(7.3,2.65,1.25,2.0,'Residual LST\nreconstruction','#9E3B4A'),(8.9,2.65,.95,2.0,'8-view TTA\nmean + spread','#533A71')]
for x,y,w,h,t,c in blocks:
    ax.add_patch(mpl.patches.FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.10',fc=c,ec='white',lw=1.3)); ax.text(x+w/2,y+h/2,t,ha='center',va='center',color='white',fontweight='bold',fontsize=8)
for x1,x2 in [(4.7,5.15),(6.85,7.3),(8.55,8.9)]: ax.annotate('',xy=(x2,3.65),xytext=(x1,3.65),arrowprops=dict(arrowstyle='-|>',lw=1.6,color='#333'))
ax.annotate('city–sensor climatology',xy=(7.9,2.62),xytext=(6.0,.65),ha='center',fontsize=8,arrowprops=dict(arrowstyle='-|>',lw=1.2,color='#9E3B4A'))
ax.text(.1,6.7,'(a) Multimodal reconstruction architecture',fontweight='bold',fontsize=11)
ax=fig.add_subplot(gs[0,2]); ax.axis('equal'); sizes=[48,8,8]; colors=['#2F806C','#D48627','#A33F48']; wedges,_=ax.pie(sizes,radius=1,colors=colors,startangle=90,wedgeprops=dict(width=.28,edgecolor='white'))
ax.pie([6,1,1]*8,radius=.68,colors=[colors[0],colors[1],colors[2]]*8,startangle=90,wedgeprops=dict(width=.18,edgecolor='white',linewidth=.7))
ax.text(0,0,'64 scenes\n8 strata',ha='center',va='center',fontweight='bold'); ax.set_title('(b) Nested split structure'); ax.legend(wedges,['Training (48)','Validation (8)','Temporal test (8)'],loc='lower center',bbox_to_anchor=(.5,-.18),frameon=False,ncol=1)
count=(splits.groupby(['city','thermal_sensor','model_split']).size().unstack(fill_value=0).reindex(columns=['train','validation','temporal_test']))
ax=fig.add_subplot(gs[1,0]); sns.heatmap(count,annot=True,fmt='g',cmap='Blues',cbar=False,linewidths=.8,ax=ax); ax.set(xlabel='Scene role',ylabel='City–sensor stratum',title='(c) Stratum-level balance')
ax=fig.add_subplot(gs[1,1]); firewall=np.array([[1,1,1,0],[1,1,1,0],[0,0,0,1]])
sns.heatmap(firewall,cmap=mpl.colors.ListedColormap(['#F3D7D9','#2F806C']),cbar=False,annot=np.where(firewall,'USED','BLOCKED'),fmt='',linewidths=1.2,ax=ax,xticklabels=['Fit','Early stop','Architecture','Final report'],yticklabels=['Train','Validation','Temporal test']); ax.set(title='(d) Leakage-control firewall',xlabel='',ylabel='')
ax=fig.add_subplot(gs[1,2]); channels=pd.Series({'Sentinel-2 optical':10,'Sentinel-1 SAR':3,'Terrain':3,'Known context':9}); ax.barh(channels.index,channels.values,color=['#2878B5','#694C9A','#4B8B3B','#D48627']); ax.invert_yaxis(); ax.set(title='(e) Information composition',xlabel='Input channels',ylabel='');
for i,v in enumerate(channels): ax.text(v+.15,i,str(v),va='center',fontweight='bold')
fig.text(.5,.015,'Temporal-test observations are isolated from fitting, early stopping, architecture selection and conformal calibration.',ha='center',fontweight='bold',color='#8E2934')
save(fig,'Figure_01_Multimodal_Architecture_and_Leakage_Firewall')
'''

def cube(ax,x,y,w,h,d,color,label='',sub='',layers=1):
    edge='#17324D'; top=mpl.colors.to_rgba(color,.72); side=mpl.colors.to_rgba(color,.52)
    for k in range(layers-1,-1,-1):
        xx=x+k*d*.22; yy=y+k*d*.11
        ax.add_patch(mpl.patches.Polygon([(xx,yy+h),(xx+d,yy+h+d),(xx+w+d,yy+h+d),(xx+w,yy+h)],fc=top,ec=edge,lw=.6))
        ax.add_patch(mpl.patches.Polygon([(xx+w,yy),(xx+w+d,yy+d),(xx+w+d,yy+h+d),(xx+w,yy+h)],fc=side,ec=edge,lw=.6))
        ax.add_patch(mpl.patches.Rectangle((xx,yy),w,h,fc=color,ec=edge,lw=.7))
    ax.text(x+w/2,y-.12,label,ha='center',va='top',fontweight='bold',fontsize=6.8)
    ax.text(x+w/2,y-.39,sub,ha='center',va='top',fontsize=5.9,color='#333')
def link(ax,x1,x2,y,color='#21618C'):
    ax.annotate('',xy=(x2,y),xytext=(x1,y),arrowprops=dict(arrowstyle='-|>',lw=1.1,color=color))
def dashed_frame(ax,title):
    ax.add_patch(mpl.patches.FancyBboxPatch((.008,.025),.984,.94,transform=ax.transAxes,boxstyle='round,pad=.01,rounding_size=.025',fc='none',ec='#23465F',lw=1.05,ls=(0,(3,2))))
    ax.text(.025,.94,title,transform=ax.transAxes,va='top',fontweight='bold',fontsize=10)

fig=plt.figure(figsize=(17,11)); gs=fig.add_gridspec(3,1,height_ratios=[1.45,.82,.88],hspace=.12)
ax=fig.add_subplot(gs[0]); ax.set_xlim(0,18); ax.set_ylim(0,6.6); ax.axis('off'); dashed_frame(ax,'(a) ThermoFusion metadata-conditioned residual U-Net')
cube(ax,.3,2.0,.7,2.7,.17,'#79C7DD','INPUT','25 × 256 × 256',5)
xs=[1.85,3.35,4.75,6.05,7.7,9.7,11.2,12.6,14.0]; hs=[2.65,2.2,1.8,1.45,1.12,1.45,1.8,2.2,2.65]; ch=[24,48,96,192,384,192,96,48,24]; names=['E1','E2','E3','E4','BRIDGE','D4','D3','D2','D1']; colors=['#E65A2F']*4+['#8A68B3']+['#60B29C']*4; scales=[256,128,64,32,16,32,64,128,256]
for x,h,c,n,q,s in zip(xs,hs,colors,names,ch,scales): cube(ax,x,2.03,.57,h,.14,c,n,f'{q} × {s} × {s}',3)
link(ax,1.22,1.85,3.3)
for a,b,h1,h2 in zip(xs[:-1],xs[1:],hs[:-1],hs[1:]): link(ax,a+.75,b,2.03+min(h1,h2)/2)
cube(ax,15.45,2.13,.52,2.45,.12,'#F3D37A','RESIDUAL','1 × 256 × 256'); link(ax,14.75,15.45,3.28)
for ei,di,level in [(1.85,14.0,5.0),(3.35,12.6,5.4),(4.75,11.2,5.7),(6.05,9.7,5.9)]:
    ax.annotate('',xy=(di+.3,2.03+hs[xs.index(di)]),xytext=(ei+.3,2.03+hs[xs.index(ei)]),arrowprops=dict(arrowstyle='-|>',lw=1,color='#4C6FB1',connectionstyle='arc3,rad=-.08'))
ax.text(8.4,6.02,'skip concatenations',ha='center',color='#4C6FB1',fontsize=7,fontweight='bold')
ax.text(16.45,3.3,'+',fontsize=17,fontweight='bold',ha='center'); ax.text(16.45,2.45,'city–sensor\nclimatology',ha='center',fontsize=6.2); link(ax,16.75,17.25,3.3,'#9A3F49'); cube(ax,17.25,2.35,.24,1.9,.07,'#F6E7B0','LST','°C')

ax=fig.add_subplot(gs[1]); ax.set_xlim(0,18); ax.set_ylim(0,4.7); ax.axis('off'); dashed_frame(ax,'(b) Multimodal tensor construction and feature conditioning')
for x,t,c,l,n in [(.45,'Sentinel-2','#2B83BA',4,'10 bands'),(2.15,'Sentinel-1','#7651A8',3,'3 bands'),(3.85,'Terrain','#4C9B45',3,'3 bands'),(5.55,'Context','#D88B2D',4,'9 bands')]:
    cube(ax,x,1.25,.72,2.0,.13,c,t,n, l); link(ax,x+.95,7.75,2.25)
ax.add_patch(mpl.patches.Ellipse((9.0,2.25),2.5,2.65,fc='#EFF3F7',ec='#23465F')); ax.text(9,2.5,'CHANNEL FUSION',ha='center',fontweight='bold',fontsize=8); ax.text(9,1.85,'masking + training-only scaling',ha='center',fontsize=6.2)
link(ax,10.28,11.75,2.25); cube(ax,11.75,1.1,1.05,2.3,.18,'#59B4C6','CONDITIONED TENSOR','25 × 256 × 256',5)
ax.annotate('terrain gated to zero in the validation-selected final model',xy=(4.2,1.18),xytext=(7.0,.42),ha='center',fontsize=6.7,color='#9A3F49',arrowprops=dict(arrowstyle='-|>',color='#9A3F49'))
ax.text(14.0,2.7,'Conv 3×3 → GroupNorm → SiLU',fontweight='bold',fontsize=7.2); ax.text(14.0,2.12,'two convolutions per block',fontsize=6.3); ax.text(14.0,1.55,'MaxPool ↓2  |  TransposedConv ↑2',fontsize=6.3)

ax=fig.add_subplot(gs[2]); ax.set_xlim(0,18); ax.set_ylim(0,5.0); ax.axis('off'); dashed_frame(ax,'(c) Eight-view inference, residual reconstruction and predictive uncertainty')
cube(ax,.45,1.2,.75,2.25,.14,'#E65A2F','FROZEN U-NET','no-terrain configuration',3); link(ax,1.45,2.75,2.35)
for x,lab in [(2.75,'0°'),(3.75,'90°'),(4.75,'180°'),(5.75,'270°')]: cube(ax,x,1.65,.34,1.25,.08,'#68A9C9',lab,'original + flip',2)
link(ax,6.55,7.65,2.35); ax.add_patch(mpl.patches.Ellipse((8.75,2.35),2.05,2.25,fc='#F1EDF7',ec='#533A71')); ax.text(8.75,2.35,'inverse transform\n+ ensemble',ha='center',va='center',fontweight='bold',fontsize=7.5)
link(ax,9.8,10.8,2.35); cube(ax,10.8,1.2,.68,2.25,.11,'#57AE95','MEAN LST','pixel estimate',2); cube(ax,12.1,1.2,.68,2.25,.11,'#8A68B3','TTA σ','dispersion',2)
link(ax,13.0,14.1,2.35); ax.add_patch(mpl.patches.Ellipse((15.1,2.35),1.9,2.15,fc='#FCE6D2',ec='#A85E2B')); ax.text(15.1,2.53,'q = 4.6028',ha='center',fontweight='bold'); ax.text(15.1,1.98,'validation-only\nconformal scaling',ha='center',fontsize=6.2)
link(ax,16.07,16.85,2.35); cube(ax,16.85,1.3,.28,2.0,.07,'#F3D37A','INTERVAL','lower–upper')
ax.text(11.2,.35,'Nominal 90% interval → observed temporal-test coverage 75.42%; uncertainty ranks error but under-covers temporal shift',ha='center',fontsize=7,color='#9A3F49',fontweight='bold')
save(fig,'Figure_01_3D_ThermoFusion_Architecture')

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
# Evidence dashboard: skill, generalisation, baseline comparison and scene transfer.
'''LEGACY_FIGURE_02_DISABLED
fig=plt.figure(figsize=(16,10)); gs=fig.add_gridspec(2,3,wspace=.32,hspace=.32)
ax=fig.add_subplot(gs[0,0]); norm=perf.set_index('model').copy(); norm['MAE skill']=1-norm.MAE/norm.MAE.iloc[0]; norm['RMSE skill']=1-norm.RMSE/norm.RMSE.iloc[0]; norm['R²']=norm.R2.clip(0,1); norm['Bias control']=1-norm.Bias.abs()/norm.Bias.abs().max(); radar=norm[['MAE skill','RMSE skill','R²','Bias control']].clip(0,1)
angles=np.linspace(0,2*np.pi,len(radar.columns),endpoint=False); angles=np.r_[angles,angles[0]]
ax.remove(); ax=fig.add_subplot(gs[0,0],projection='polar')
for (name,row),c in zip(radar.iterrows(),['#6C7A89','#2A9D8F','#7B4F9E']): vals=np.r_[row.values,row.values[0]]; ax.plot(angles,vals,lw=2,label=name,color=c); ax.fill(angles,vals,color=c,alpha=.08)
ax.set_xticks(angles[:-1],radar.columns); ax.set_ylim(0,1); ax.set_yticks([.25,.5,.75,1]); ax.set_title('(a) Normalized multimetric skill',pad=18); ax.legend(frameon=False,bbox_to_anchor=(.5,-.18),loc='upper center',fontsize=7)
ax=fig.add_subplot(gs[0,1]); all14=a14.query("group=='ALL' and model in ['refined_unet','city_sensor_climatology','global_climatology','sensor_climatology']"); pivot=all14.pivot(index='model',columns='model_split',values='rmse_c').dropna()
for name,row in pivot.iterrows(): ax.plot([0,1],[row.validation,row.temporal_test],'-o',lw=1.7,label=f"{name.replace('_',' ')} ({row.temporal_test:.2f}°C)")
ax.set_xticks([0,1],['Validation','Temporal test']); ax.set(ylabel='RMSE (°C)',title='(b) Generalisation trajectories'); ax.legend(frameon=False,fontsize=7)
ax=fig.add_subplot(gs[0,2]); heat=perf.set_index('model')[['MAE','RMSE','Bias','R2']].copy(); heat[['MAE','RMSE','Bias']]=heat[['MAE','RMSE','Bias']].apply(lambda x:(x-x.min())/(x.max()-x.min()+1e-12)); heat['R2']=1-(heat.R2-heat.R2.min())/(heat.R2.max()-heat.R2.min()+1e-12); sns.heatmap(heat,annot=perf.set_index('model')[['MAE','RMSE','Bias','R2']],fmt='.3f',cmap='rocket_r',linewidths=.8,cbar_kws={'label':'Relative error burden'},ax=ax); ax.set(title='(c) Joint metric evidence matrix',xlabel='',ylabel='')
pair=s15.query("model_split=='temporal_test' and method in ['selected_calibrated_unet','city_sensor_climatology']").pivot(index='pilot_id',columns='method',values='rmse_c').dropna()
ax=fig.add_subplot(gs[1,0]);
city_lookup=splits.set_index('pilot_id').city.to_dict()
for pid,row in pair.iterrows(): ax.plot([0,1],row.values,'-o',alpha=.8,lw=1.2,color=COL.get(city_lookup.get(pid),'#557A95'))
ax.set_xticks([0,1],['Selected U-Net','City–sensor\nclimatology']); ax.set(ylabel='Scene RMSE (°C)',title='(d) Paired temporal-test scenes')
ax=fig.add_subplot(gs[1,1]); ci=one(b15.query("metric=='scene_macro_rmse_c'"),'Stage 15 RMSE bootstrap'); ax.errorbar(ci.calibrated_minus_climatology,0,xerr=[[ci.calibrated_minus_climatology-ci.ci_lower],[ci.ci_upper-ci.calibrated_minus_climatology]],fmt='o',capsize=6,color='#A33F48',lw=2); ax.axvline(0,color='k',ls='--'); ax.set_yticks([]); ax.set(xlabel='U-Net − climatology RMSE (°C)',title='(e) Paired scene-bootstrap inference'); ax.text(ci.calibrated_minus_climatology,.12,f'95% CI [{ci.ci_lower:.2f}, {ci.ci_upper:.2f}]',ha='center',fontsize=8)
ax=fig.add_subplot(gs[1,2]); scene=s17.query("model_split=='temporal_test'"); sc=ax.scatter(scene.rmse_c,scene.bias_c.abs(),c=scene.interval_coverage,s=80+180*scene.median_tta_std_c/scene.median_tta_std_c.max(),cmap='viridis',vmin=0,vmax=1,edgecolor='white');
for _,r in scene.iterrows(): ax.annotate(r.pilot_id,(r.rmse_c,abs(r.bias_c)),xytext=(3,3),textcoords='offset points',fontsize=6)
ax.set(xlabel='Scene RMSE (°C)',ylabel='Absolute scene bias (°C)',title='(f) Error–bias–coverage landscape'); fig.colorbar(sc,ax=ax,label='90% interval coverage')
save(fig,'Figure_02_Model_Evidence_and_Generalisation_Dashboard')
'''

# Figure 2 replacement: generalisation and paired statistical evidence.
# Raw units are retained; no arbitrary composite score is used.
all14=a14.query("group=='ALL' and model in ['refined_unet','city_sensor_climatology','global_climatology','sensor_climatology']")
pivot=all14.pivot(index='model',columns='model_split',values='rmse_c').dropna(subset=['validation','temporal_test'])
pair=s15.query("model_split=='temporal_test' and method in ['selected_calibrated_unet','city_sensor_climatology']").pivot(index='pilot_id',columns='method',values='rmse_c').dropna()
pair=pair.rename(columns={'selected_calibrated_unet':'model_rmse_c','city_sensor_climatology':'climatology_rmse_c'})
pair=pair.join(splits.set_index('pilot_id')[['city','thermal_sensor']]); pair['delta_rmse_c']=pair.model_rmse_c-pair.climatology_rmse_c
pair.reset_index().to_csv(TAB/'Table_10_paired_model_climatology_evidence.csv',index=False); tables['Table_10_paired_evidence']=pair.reset_index()
rng_boot=np.random.default_rng(SEED); delta=pair.delta_rmse_c.to_numpy(); boot=rng_boot.choice(delta,size=(50000,len(delta)),replace=True).mean(axis=1)
ci_emp=np.quantile(boot,[.025,.975]); p_better=float(np.mean(boot<0))

fig=plt.figure(figsize=(16.5,9.4)); gs=fig.add_gridspec(2,3,wspace=.34,hspace=.38)
ax=fig.add_subplot(gs[0,0]); model_colors={'refined_unet':'#2A9D8F','city_sensor_climatology':'#3B6FB6','global_climatology':'#E07A3F','sensor_climatology':'#8C5A9E'}
for name,row in pivot.sort_values('validation',ascending=False).iterrows():
    c=model_colors[name]; ax.annotate('',xy=(row.temporal_test,row.validation),xytext=(row.validation,row.validation),arrowprops=dict(arrowstyle='-|>',lw=1.6,color=c)); ax.scatter(row.validation,row.validation,s=48,facecolor='white',edgecolor=c,lw=1.4); ax.scatter(row.temporal_test,row.validation,s=58,color=c,zorder=3); ax.text(row.temporal_test+.06,row.validation,name.replace('_',' '),va='center',fontsize=7)
lo=min(pivot.min())-.25; hi=max(pivot.max())+.35; ax.plot([lo,hi],[lo,hi],ls='--',lw=.8,color='#777'); ax.set(xlim=(lo,hi),ylim=(lo,hi),xlabel='Temporal-test RMSE (°C)',ylabel='Validation RMSE (°C)',title='(a) Validation-to-temporal transfer'); ax.text(.03,.04,'Arrow: validation → temporal test',transform=ax.transAxes,fontsize=7,color='#555')
ax=fig.add_subplot(gs[0,1])
for pid,row in pair.sort_values('model_rmse_c').iterrows():
    c=COL[row.city]; ax.plot([0,1],[row.model_rmse_c,row.climatology_rmse_c],color=c,lw=1.35,alpha=.85); ax.scatter([0,1],[row.model_rmse_c,row.climatology_rmse_c],s=34,color=c,edgecolor='white',lw=.45,zorder=3); ax.text(-.035,row.model_rmse_c,pid,ha='right',va='center',fontsize=5.8)
ax.set_xticks([0,1],['Selected U-Net','City–sensor\nclimatology']); ax.set_xlim(-.28,1.18); ax.set(ylabel='Scene RMSE (°C)',title='(b) Paired scene response'); ax.legend(handles=[mpl.lines.Line2D([],[],marker='o',ls='',color=c,label=k) for k,c in COL.items()],frameon=False,ncol=2,loc='upper center',fontsize=6.5)
ax=fig.add_subplot(gs[0,2]); sns.kdeplot(x=boot,fill=True,color='#8E2934',alpha=.24,lw=1.5,ax=ax); ax.axvline(0,color='#222',ls='--',lw=1); ax.axvspan(ci_emp[0],ci_emp[1],color='#8E2934',alpha=.10); ax.axvline(boot.mean(),color='#8E2934',lw=2); ax.set(xlabel='ΔRMSE = U-Net − climatology (°C)',ylabel='Bootstrap density',title='(c) Paired scene-bootstrap distribution'); ax.text(.04,.94,f'mean = {boot.mean():.2f} °C\n95% CI [{ci_emp[0]:.2f}, {ci_emp[1]:.2f}]\nP(Δ < 0) = {p_better:.2f}',transform=ax.transAxes,va='top',fontsize=7.5,bbox=dict(boxstyle='round',fc='white',ec='#B7B7B7'))
ax=fig.add_subplot(gs[1,0]); metrics=['MAE','RMSE','Bias','R2']; y=np.arange(len(metrics)); basevals=perf.iloc[0][metrics].astype(float); colors=['#2A9D8F','#6B4C9A']
for j,(label,row) in enumerate(perf.iloc[1:].set_index('model').iterrows()):
    vals=[100*(basevals[m]-float(row[m]))/abs(basevals[m]) if m!='R2' else 100*(float(row[m])-basevals[m])/max(1e-9,1-basevals[m]) for m in metrics]; yy=y+(j-.5)*.16
    ax.scatter(vals,yy,s=55,color=colors[j],label=label,zorder=3)
    for v,y0 in zip(vals,yy): ax.plot([0,v],[y0,y0],color=colors[j],lw=1.3)
ax.axvline(0,color='#333',lw=.8); ax.set_yticks(y,metrics); ax.invert_yaxis(); ax.set(xlabel='Improvement relative to Stage 13 baseline (%)',title='(d) Metric-wise improvement'); ax.legend(frameon=False,fontsize=6.5)
ax=fig.add_subplot(gs[1,1]); scene=s17.query("model_split=='temporal_test'").copy(); mat=scene.pivot(index='city',columns='thermal_sensor',values='rmse_c').reindex(index=['Abidjan','Accra','Freetown','Lagos'],columns=['Landsat','ECOSTRESS']); ann=mat.copy().astype(object)
for c in mat.index:
    for sensor in mat.columns:
        r=scene.query('city==@c and thermal_sensor==@sensor').iloc[0]; ann.loc[c,sensor]=f"{r.rmse_c:.2f}\nR² {r.r2:.2f}"
sns.heatmap(mat,annot=ann,fmt='',cmap='mako_r',linewidths=1,cbar_kws={'label':'Scene RMSE (°C)'},ax=ax); ax.set(title='(e) City–sensor generalisation matrix',xlabel='',ylabel='')
ax=fig.add_subplot(gs[1,2]); grid=np.linspace(0,max(pair.model_rmse_c.max(),pair.climatology_rmse_c.max())+.2,200)
for col,label,c in [('model_rmse_c','Selected U-Net','#2A9D8F'),('climatology_rmse_c','City–sensor climatology','#3B6FB6')]:
    vals=np.sort(pair[col]); ax.step(grid,np.searchsorted(vals,grid,side='right')/len(vals),where='post',lw=2,color=c,label=label)
ax.set(xlabel='Scene RMSE threshold (°C)',ylabel='Fraction of scenes ≤ threshold',title='(f) Empirical scene-error dominance',xlim=(0,grid.max()),ylim=(0,1.03)); ax.legend(frameon=False,loc='lower right')
fig.suptitle('Temporal generalisation, paired effects, and subgroup evidence',fontsize=13,fontweight='bold',y=1.01)
save(fig,'Figure_02_Generalisation_and_Paired_Evidence')

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

# Figure 5 replacement: uncertainty audit under temporal shift. TTA spread is
# treated as a ranking signal; conformal interval coverage is assessed separately.
pix['uncertainty_decile']=pd.qcut(pix.uncertainty_c,10,labels=False,duplicates='drop')+1
udec=pix.groupby(['thermal_sensor','uncertainty_decile']).agg(n_pixels=('abs_error_c','size'),mean_uncertainty_c=('uncertainty_c','mean'),mae_c=('abs_error_c','mean'),p90_abs_error_c=('abs_error_c',lambda x:np.quantile(x,.9))).reset_index()
udec.to_csv(TAB/'Table_11_uncertainty_decile_diagnostics.csv',index=False); tables['Table_11_uncertainty_deciles']=udec
fractions=np.linspace(.05,1,20); risk_rows=[]
for sensor,g in [('All',pix),*list(pix.groupby('thermal_sensor'))]:
    ordered=g.sort_values('uncertainty_c')
    for f in fractions:
        keep=ordered.iloc[:max(1,int(len(ordered)*f))]
        risk_rows.append([sensor,f,len(keep),keep.abs_error_c.mean(),np.quantile(keep.abs_error_c,.9)])
risk=pd.DataFrame(risk_rows,columns=['thermal_sensor','retained_fraction','n_pixels','mae_c','p90_abs_error_c'])
risk.to_csv(TAB/'Table_12_selective_prediction_risk.csv',index=False); tables['Table_12_selective_risk']=risk

fig=plt.figure(figsize=(16.5,9.6)); gs=fig.add_gridspec(2,3,wspace=.34,hspace=.38)
ax=fig.add_subplot(gs[0,0]); h=ax.hexbin(pix.uncertainty_c,pix.abs_error_c,gridsize=65,bins='log',mincnt=1,cmap='magma',rasterized=True); qpix=pix.groupby('uncertainty_decile').agg(u=('uncertainty_c','mean'),mae=('abs_error_c','mean'),p90=('abs_error_c',lambda x:np.quantile(x,.9))).reset_index(); ax.plot(qpix.u,qpix.mae,'o-',color='#35B779',lw=1.7,label='decile MAE'); ax.plot(qpix.u,qpix.p90,'s--',color='#FDE725',lw=1.4,label='decile 90th error'); ax.set(xlabel='TTA standard deviation (°C)',ylabel='Absolute error (°C)',title='(a) Pixel-level uncertainty–error density'); ax.legend(frameon=False); fig.colorbar(h,ax=ax,label='log pixel density')
ax=fig.add_subplot(gs[0,1])
for sensor,g in udec.groupby('thermal_sensor'):
    c={'Landsat':'#E07A3F','ECOSTRESS':'#3B6FB6'}[sensor]; ax.plot(g.uncertainty_decile,g.mae_c,'o-',lw=1.8,label=sensor,color=c); ax.fill_between(g.uncertainty_decile,g.mae_c,g.p90_abs_error_c,color=c,alpha=.10)
ax.set(xticks=range(1,11),xlabel='TTA uncertainty decile (low → high)',ylabel='Absolute error (°C)',title='(b) Sensor-stratified error escalation'); ax.legend(frameon=False)
ax=fig.add_subplot(gs[0,2]); sp=s17.query("model_split=='temporal_test'").sort_values('mean_interval_width_c')
for _,r in sp.iterrows():
    ax.plot([0,r.mean_interval_width_c],[r.interval_coverage,r.interval_coverage],color=COL[r.city],lw=1.2,alpha=.75); ax.scatter(r.mean_interval_width_c,r.interval_coverage,s=65,marker='s' if r.thermal_sensor=='Landsat' else 'o',color=COL[r.city],edgecolor='white',zorder=3); ax.annotate(r.pilot_id,(r.mean_interval_width_c,r.interval_coverage),xytext=(3,3),textcoords='offset points',fontsize=5.7)
ax.axhline(.9,color='#222',ls='--',lw=1); ax.set(xlabel='Mean 90% interval width (°C)',ylabel='Observed coverage',ylim=(0,1.04),title='(c) Scene-level sharpness–coverage')
ax=fig.add_subplot(gs[1,0]); covmat=sp.pivot(index='city',columns='thermal_sensor',values='interval_coverage').reindex(index=['Abidjan','Accra','Freetown','Lagos'],columns=['Landsat','ECOSTRESS']); sns.heatmap(covmat,annot=True,fmt='.2f',vmin=0,vmax=1,cmap='RdYlGn',linewidths=1,cbar_kws={'label':'Observed 90% coverage'},ax=ax); ax.set(title='(d) Coverage under city–sensor shift',xlabel='',ylabel='')
ax=fig.add_subplot(gs[1,1]); rc={'All':'#222','Landsat':'#E07A3F','ECOSTRESS':'#3B6FB6'}
for sensor,g in risk.groupby('thermal_sensor'): ax.plot(g.retained_fraction,g.mae_c,'o-',ms=3,lw=1.8,color=rc[sensor],label=sensor)
ax.set(xlabel='Fraction of least-uncertain pixels retained',ylabel='Selective MAE (°C)',title='(e) Risk–coverage under abstention'); ax.legend(frameon=False)
ax=fig.add_subplot(gs[1,2]); loq,hiq=pix.uncertainty_c.quantile([.2,.8]); groups=[('Lowest 20% uncertainty',pix.loc[pix.uncertainty_c<=loq,'abs_error_c'],'#2A9D8F'),('Highest 20% uncertainty',pix.loc[pix.uncertainty_c>=hiq,'abs_error_c'],'#A33F48')]; grid=np.linspace(0,np.quantile(pix.abs_error_c,.995),250)
for label,vals,c in groups:
    vals=np.sort(vals); ax.step(grid,np.searchsorted(vals,grid,side='right')/len(vals),where='post',lw=2,color=c,label=label)
ax.set(xlabel='Absolute prediction error (°C)',ylabel='Empirical cumulative probability',title='(f) Error separation by uncertainty'); ax.legend(frameon=False,loc='lower right')
fig.suptitle('Predictive-uncertainty reliability and interval validity under temporal shift',fontsize=13,fontweight='bold',y=1.01)
save(fig,'Figure_05_Uncertainty_Audit_Under_Temporal_Shift')

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

# Final workbook rewrite after the two advanced figure-source tables are created.
with pd.ExcelWriter(OUT/'ThermoFusion_Publication_Tables.xlsx',engine='openpyxl') as writer:
    for name,frame in tables.items():
        sheet=name.replace('Table_','T')[:31]; frame.to_excel(writer,sheet_name=sheet,index=False)
        ws=writer.book[sheet]; ws.freeze_panes='A2'; ws.auto_filter.ref=ws.dimensions
        for c in ws[1]: c.font=Font(name='Calibri',size=11,bold=True,color='FFFFFF'); c.fill=PatternFill('solid',fgColor='17365D')
        for cells in ws.columns: ws.column_dimensions[cells[0].column_letter].width=min(42,max(len(str(c.value or '')) for c in cells)+2)

checks={'figures_png':len(list(PNG.glob('*.png')))==7,'figure_pdfs':len(list(VEC.glob('*.pdf')))==7,
 'vector_svgs':len(list(VEC.glob('*.svg')))==6,'csv_tables':len(list(TAB.glob('*.csv')))==13,
 'test_scenes':len(meta)==8,'finite_final_metrics':np.isfinite(a17.select_dtypes('number')).all().all()}
pd.DataFrame([{'check':k,'passed':v} for k,v in checks.items()]).to_csv(OUT/'publication_output_checks.csv',index=False)
verdict=['THERMOFUSION STAGE 19 PUBLICATION GRAPHICS',f'900-dpi figures: {len(list(PNG.glob("*.png")))}/7',
 f'PDF figures: {len(list(VEC.glob("*.pdf")))}/7',f'Editable SVG charts: {len(list(VEC.glob("*.svg")))}/6',
 f'Publication tables: {len(list(TAB.glob("*.csv")))}/13','Verified analysis values changed: NO',
 'Overall publication-graphics verdict: PASS' if all(checks.values()) else 'Overall publication-graphics verdict: REVIEW REQUIRED']
(OUT/'publication_graphics_verdict.txt').write_text('\n'.join(verdict),encoding='utf-8')
with (OUT/'publication_graphics_configuration.json').open('w') as f: json.dump({'dpi':DPI,'seed':SEED,'source_stages':[12,13,14,15,16,17]},f,indent=2)
zip_path=shutil.make_archive(str(ROOT/'ThermoFusion_Stage19_Publication_Graphics'),'zip',root_dir=OUT)
print('\n'+'\n'.join(verdict)); print(f'\nOutput folder: {OUT}\nZIP package: {zip_path}')
