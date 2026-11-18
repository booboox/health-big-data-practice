"""E3: prespecified learning-rate and stage comparisons on fixed WDBC split."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import log_loss,roc_auc_score
import sklearn

BASE=Path(__file__).resolve().parent;OUT=BASE/'工作单配图'
DATA=(Path(__file__).resolve().parents[2] / '10. Ensemble' / 'data' / 'student')
t=pd.read_csv(DATA/'train.csv');v=pd.read_csv(DATA/'validation.csv')
features=[c for c in t if c not in ['sample_id','malignant']]
rows=[]
for rate in [.03,.1,.3]:
    m=GradientBoostingClassifier(loss='log_loss',learning_rate=rate,n_estimators=500,max_depth=3,random_state=1010)
    m.fit(t[features],t.malignant)
    for stage,(pt,pv) in enumerate(zip(m.staged_predict_proba(t[features]),m.staged_predict_proba(v[features])),1):
        rows.append({'learning_rate':rate,'stage':stage,'train_log_loss':log_loss(t.malignant,pt),
            'validation_log_loss':log_loss(v.malignant,pv),'train_roc_auc':roc_auc_score(t.malignant,pt[:,1]),
            'validation_roc_auc':roc_auc_score(v.malignant,pv[:,1])})
    print('Finished learning rate',rate,flush=True)
df=pd.DataFrame(rows);df.to_csv(OUT/'Figure_E3_learning_curves.csv',index=False)
summary=[]
for rate in [.03,.1,.3]:
    sub=df[df.learning_rate==rate];best=sub.loc[sub.validation_log_loss.idxmin()];last=sub.iloc[-1]
    summary.append({'learning_rate':rate,'best_stage_by_validation_log_loss':int(best.stage),
        'best_validation_log_loss':best.validation_log_loss,'validation_auc_at_best_stage':best.validation_roc_auc,
        'validation_log_loss_at_500':last.validation_log_loss,'validation_auc_at_500':last.validation_roc_auc,
        'train_log_loss_at_500':last.train_log_loss})
pd.DataFrame(summary).to_csv(OUT/'Figure_E3_best_stages.csv',index=False)
(OUT/'Figure_E3_metadata.json').write_text(json.dumps({'sklearn':sklearn.__version__,'seed':1010,
    'learning_rates':[.03,.1,.3],'n_estimators':500,'max_depth':3,'loss':'log_loss',
    'selection':'first minimum validation log-loss within each learning rate',
    'automatic_early_stopping':False,'test_used':False},indent=2),encoding='utf-8')
BLUE,ORANGE,GRAY='#28658A','#C46B35','#596774'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
    'axes.spines.right':False,'axes.linewidth':.7,'pdf.fonttype':42,'ps.fonttype':42})
fig,axes=plt.subplots(2,3,figsize=(11,7.1))
fig.subplots_adjust(left=.08,right=.98,top=.80,bottom=.21,wspace=.34,hspace=.38)
fig.text(.04,.96,'Boosting can overfit probabilities before ranking degrades',fontsize=15,weight='bold',color='#25313B')
fig.text(.04,.90,'WDBC  |  Training n = 341; validation n = 114  |  Tree depth = 3  |  500 stages per learning rate',fontsize=9,color=GRAY)
for col,rate in enumerate([.03,.1,.3]):
    sub=df[df.learning_rate==rate];best=sub.loc[sub.validation_log_loss.idxmin()]
    for row,metric in enumerate(['log_loss','roc_auc']):
        ax=axes[row,col]
        ax.plot(sub.stage,sub['train_'+metric],color=BLUE,lw=1.5,label='Training')
        ax.plot(sub.stage,sub['validation_'+metric],color=ORANGE,lw=1.5,label='Validation')
        ax.axvline(best.stage,color=GRAY,ls=(0,(3,3)),lw=.8)
        ax.set(xlim=(1,500),xticks=[1,100,250,500],ylabel='Log-loss' if row==0 else 'ROC-AUC')
        if row==0:
            ax.set_ylim(0,max(df.train_log_loss.max(),df.validation_log_loss.max())*1.06)
            ax.set_title(f'Learning rate = {rate:g}',loc='left',fontsize=11,weight='bold',pad=16)
            ax.scatter(best.stage,best.validation_log_loss,s=26,color=ORANGE,edgecolor='white',lw=.5,zorder=4)
            ax.text(.96,.93,f'Best stage: {int(best.stage)}\nLoss: {best.validation_log_loss:.4f}',transform=ax.transAxes,ha='right',va='top',fontsize=8,color=ORANGE)
        else:
            ax.set_ylim(.90,1.004);ax.set_xlabel('Boosting stage')
        ax.text(-.21,1.09,'abcdef'[row*3+col],transform=ax.transAxes,fontsize=13,weight='bold')
        ax.grid(axis='y',color='#E2E7EB',lw=.6);ax.set_axisbelow(True)
handles,labels=axes[0,0].get_legend_handles_labels()
fig.legend(handles,labels,loc='lower left',bbox_to_anchor=(.07,.11),ncol=2,frameon=False)
fig.text(.04,.085,'Dashed lines: stage with the lowest validation log-loss for each learning rate. Stages were inspected after fitting.',fontsize=8,color=GRAY)
fig.text(.04,.045,'Validation minima are selection results, not unbiased performance estimates. No test data used; no uncertainty bands estimated.',fontsize=8,color=GRAY)
for ext in ['png','pdf','svg']:fig.savefig(OUT/f'Figure_E3_boosting_learning_curves.{ext}',dpi=600,bbox_inches='tight',pad_inches=.13)
plt.close(fig)
print(pd.DataFrame(summary).to_string(index=False))
