"""G: reuse D's shared bootstrap predictions; distinguish uncertainty scopes."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE=Path(__file__).resolve().parent;OUT=BASE/'工作单配图'
p=pd.read_csv(OUT/'Figure_D_bootstrap_predictions.csv')
stats=p.groupby(['model','sample_id']).agg(mean_probability=('probability','mean'),
    prediction_sd=('probability',lambda x: x.std(ddof=0)),
    min_probability=('probability','min'),max_probability=('probability','max')).reset_index()
stats['class_switch']= (stats.min_probability<.5)&(stats.max_probability>=.5)
stats.to_csv(OUT/'Figure_G_per_record_stability.csv',index=False)
wide=stats.pivot(index='sample_id',columns='model',values='prediction_sd')
summary=[]
for model in ['Bagging','Random forest']:
    summary.append({'model':model,'records_with_lower_sd_than_single_tree':int((wide[model]<wide['Single tree']).sum()),
        'n_validation_records':len(wide),'fraction_with_lower_sd':float((wide[model]<wide['Single tree']).mean())})
pd.DataFrame(summary).to_csv(OUT/'Figure_G_paired_sd_summary.csv',index=False)
BLUE,ORANGE,TEAL,GRAY='#28658A','#C46B35','#459C93','#596774'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
    'axes.spines.right':False,'axes.linewidth':.7,'pdf.fonttype':42,'ps.fonttype':42})
fig=plt.figure(figsize=(11,7.7))
gs=fig.add_gridspec(2,2,left=.08,right=.97,top=.79,bottom=.13,hspace=.55,height_ratios=[1.25,1])
ax=fig.add_subplot(gs[0,0]);bx=fig.add_subplot(gs[0,1]);cx=fig.add_subplot(gs[1,:]);cx.axis('off')
fig.text(.04,.955,'Prediction stability and uncertainty answer different questions',fontsize=15,weight='bold',color='#25313B')
fig.text(.04,.89,'WDBC  |  30 shared bootstrap training sets  |  Fixed validation records (n = 114) and model seeds',fontsize=9,color=GRAY)
for name,color in [('Single tree',BLUE),('Bagging',ORANGE),('Random forest',TEAL)]:
    sd=np.sort(stats.loc[stats.model==name,'prediction_sd'])
    ax.step(sd,np.arange(1,len(sd)+1)/len(sd),where='post',color=color,lw=1.5,label=name)
ax.set(xlabel='Prediction SD across 30 refits',ylabel='Cumulative fraction of validation records',ylim=(0,1.03))
ax.set_title('Distribution of record-level variability',loc='left',fontsize=10,weight='bold',pad=16)
ax.legend(frameon=False,fontsize=8,loc='lower right')
lim=float(wide.to_numpy().max())*1.05
for name,color,marker in [('Bagging',ORANGE,'s'),('Random forest',TEAL,'o')]:
    bx.scatter(wide['Single tree'],wide[name],color=color,marker=marker,s=20,alpha=.6,edgecolors='none',label=name)
bx.plot([0,lim],[0,lim],ls='--',color='#8B979F',lw=.9)
bx.set(xlim=(-.005,lim),ylim=(-.005,lim),xlabel='Single-tree prediction SD',ylabel='Ensemble prediction SD')
bx.set_title('Paired comparison on the same records',loc='left',fontsize=10,weight='bold',pad=16)
bx.legend(frameon=False,fontsize=8,loc='upper left')
for a,letter in [(ax,'a'),(bx,'b')]:
    a.text(-.14,1.10,letter,transform=a.transAxes,fontsize=13,weight='bold')
    a.grid(axis='y',color='#E2E7EB',lw=.6);a.set_axisbelow(True)
cx.text(-.02,1.12,'c',transform=cx.transAxes,fontsize=13,weight='bold')
cx.text(.02,1.10,'What was resampled — and what the result can support',fontsize=10,weight='bold')
table=cx.table(cellText=[
    ['Training stability','Training rows; refit models','Prediction SD / switching','Measured here'],
    ['Metric uncertainty','Evaluation rows; model fixed','Conditional metric interval','Not computed here'],
    ['External uncertainty','New people / centers / times','Transport to a new setting','Not established']],
    colLabels=['Question','What changes?','Interpretation','Evidence status'],
    colWidths=[.18,.30,.30,.22],cellLoc='left',colLoc='left',bbox=[0,0,1,.92])
table.auto_set_font_size(False);table.set_fontsize(8.1)
for (r,c),cell in table.get_celld().items():
    cell.set_edgecolor('white');cell.set_linewidth(2);cell.PAD=.09
    if r==0:cell.set_facecolor('#EAF1F5');cell.set_text_props(weight='bold',color=BLUE)
    else:cell.set_facecolor('#F4F6F8' if r%2 else 'white')
fig.text(.04,.07,'Each dot in panel b is one validation record; points below the diagonal have lower SD for the ensemble. Overlapping points may hide records.',fontsize=8,color=GRAY)
fig.text(.04,.027,'Model settings match D. These refit distributions are not individual clinical-risk confidence intervals or evidence of external validity.',fontsize=8,color=GRAY)
for ext in ['png','pdf','svg']:fig.savefig(OUT/f'Figure_G_stability_uncertainty.{ext}',dpi=600,bbox_inches='tight',pad_inches=.13)
plt.close(fig)
print(pd.DataFrame(summary).to_string(index=False))
print(stats.groupby('model').agg(mean_sd=('prediction_sd','mean'),median_sd=('prediction_sd','median'),switch_rate=('class_switch','mean')).to_string())
