"""F: preset model comparison with shared folds and fold-local scaling."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score,balanced_accuracy_score,accuracy_score,confusion_matrix,roc_curve
import sklearn

BASE=Path(__file__).resolve().parent;OUT=BASE/'工作单配图'
DATA=(Path(__file__).resolve().parents[2] / '10. Ensemble' / 'data' / 'student')
t=pd.read_csv(DATA/'train.csv');v=pd.read_csv(DATA/'validation.csv');dev=pd.concat([t,v],ignore_index=True)
f=[c for c in t if c not in ['sample_id','malignant']]
models={'Logistic regression':make_pipeline(StandardScaler(),LogisticRegression(max_iter=5000,random_state=1010)),
    'RBF-SVM':make_pipeline(StandardScaler(),SVC(C=1,kernel='rbf',random_state=1010)),
    'Random forest':RandomForestClassifier(n_estimators=100,max_features='sqrt',random_state=1010,n_jobs=1),
    'Gradient boosting':GradientBoostingClassifier(n_estimators=100,learning_rate=.1,max_depth=3,random_state=1010)}
def score(m,x):return m.decision_function(x) if hasattr(m,'decision_function') else m.predict_proba(x)[:,1]
folds=list(StratifiedKFold(n_splits=5,shuffle=True,random_state=1010).split(dev[f],dev.malignant))
pd.DataFrame([{'sample_id':dev.iloc[i].sample_id,'fold':k} for k,(_,va) in enumerate(folds,1) for i in va]).to_csv(OUT/'Figure_F_fold_assignments.csv',index=False)
rows=[];cvrows=[];predrows=[];rocdata={}
for name,template in models.items():
    m=clone(template).fit(t[f],t.malignant);s=score(m,v[f]);pred=m.predict(v[f]);tn,fp,fn,tp=confusion_matrix(v.malignant,pred,labels=[0,1]).ravel()
    rows.append({'model':name,'validation_roc_auc':roc_auc_score(v.malignant,s),
        'validation_balanced_accuracy':balanced_accuracy_score(v.malignant,pred),
        'validation_accuracy':accuracy_score(v.malignant,pred),'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)})
    predrows.extend({'model':name,'sample_id':sid,'malignant':int(y),'prediction':int(p),'score':float(q)} for sid,y,p,q in zip(v.sample_id,v.malignant,pred,s))
    rocdata[name]=roc_curve(v.malignant,s)
    for k,(tr,va) in enumerate(folds,1):
        cm=clone(template).fit(dev.iloc[tr][f],dev.iloc[tr].malignant)
        cvrows.append({'model':name,'fold':k,'roc_auc':roc_auc_score(dev.iloc[va].malignant,score(cm,dev.iloc[va][f])),
            'balanced_accuracy':balanced_accuracy_score(dev.iloc[va].malignant,cm.predict(dev.iloc[va][f]))})
    print('Finished:',name,flush=True)
cv=pd.DataFrame(cvrows);cv.to_csv(OUT/'Figure_F_cv_folds.csv',index=False)
summary=cv.groupby('model',sort=False).agg(cv_auc_mean=('roc_auc','mean'),cv_auc_sd=('roc_auc',lambda s:s.std(ddof=1)),
    cv_ba_mean=('balanced_accuracy','mean'),cv_ba_sd=('balanced_accuracy',lambda s:s.std(ddof=1))).reset_index()
table=pd.DataFrame(rows).merge(summary,on='model');table.to_csv(OUT/'Figure_F_model_comparison.csv',index=False)
pd.DataFrame(predrows).to_csv(OUT/'Figure_F_validation_predictions.csv',index=False)
(OUT/'Figure_F_metadata.json').write_text(json.dumps({'sklearn':sklearn.__version__,'seed':1010,'train_n':341,'validation_n':114,
    'development_cv_n':455,'cv':'shared StratifiedKFold(5, shuffle=True, random_state=1010)',
    'scaling':'StandardScaler inside LR/SVM pipelines; fit only on each training fold',
    'parameters':{name:str(m) for name,m in models.items()},'test_used':False,
    'cv_caveat':'Preset comparison after earlier exploratory inspection of the same development data; not an independent external estimate.',
    'sd_ddof':1},indent=2),encoding='utf-8')

colors=['#28658A','#8A78A7','#459C93','#C46B35'];short=['LR','RBF-SVM','RF','GB']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,'pdf.fonttype':42,'ps.fonttype':42})
fig,axes=plt.subplots(1,3,figsize=(11.5,4.9),gridspec_kw={'width_ratios':[1,1,1.1]})
fig.subplots_adjust(left=.065,right=.98,top=.73,bottom=.29,wspace=.40)
fig.text(.04,.95,'Tree ensembles do not outperform every simpler baseline',fontsize=15,weight='bold',color='#25313B')
fig.text(.04,.86,'WDBC  |  Preset models; shared split and five CV folds  |  Scaling fitted within training data only',fontsize=9,color='#596774')
ax,bx,cx=axes
x=np.arange(4)
ax.scatter(x,table.validation_roc_auc,marker='o',s=35,c=colors,label='ROC-AUC',zorder=3)
ax.scatter(x,table.validation_balanced_accuracy,marker='s',s=30,facecolors='white',edgecolors=colors,label='Balanced accuracy',zorder=3)
ax.set(xticks=x,xticklabels=short,ylim=(.93,1.008),ylabel='Validation score',xlim=(-.5,3.5))
ax.set_title('Fixed validation set (n = 114)',loc='left',fontsize=10,weight='bold',pad=16)
ax.legend(frameon=False,fontsize=7.5,loc='lower left')
for j,name in enumerate(models):
    vals=cv.loc[cv.model==name,'roc_auc'].to_numpy()
    bx.scatter(j+np.linspace(-.11,.11,5),vals,color=colors[j],s=17,alpha=.65,zorder=3)
    bx.errorbar(j,table.iloc[j].cv_auc_mean,yerr=table.iloc[j].cv_auc_sd,fmt='_',color='#25313B',ms=15,capsize=4,lw=1.3,zorder=4)
bx.set(xticks=x,xticklabels=short,ylim=(.95,1.015),ylabel='Development CV ROC-AUC',xlim=(-.5,3.5))
bx.set_title('Shared five-fold CV (n = 455)',loc='left',fontsize=10,weight='bold',pad=16)
for name,color in zip(models,colors):
    fpr,tpr,_=rocdata[name];cx.plot(fpr,tpr,color=color,lw=1.4,label=name)
cx.plot([0,1],[0,1],ls='--',color='#B0BAC2',lw=.8)
cx.set(xlim=(-.01,1.01),ylim=(-.01,1.03),xlabel='False-positive rate',ylabel='True-positive rate')
cx.set_title('Validation ROC curves',loc='left',fontsize=10,weight='bold',pad=16)
cx.legend(frameon=False,fontsize=7,loc='lower right')
for a,letter in zip(axes,'abc'):
    a.text(-.20,1.11,letter,transform=a.transAxes,fontsize=13,weight='bold')
    a.grid(axis='y',color='#E2E7EB',lw=.6);a.set_axisbelow(True)
fig.text(.04,.17,'CV dots: individual folds. Black marks: mean ± fold SD (not confidence intervals). SVM AUC uses decision scores.',fontsize=8,color='#596774')
fig.text(.04,.11,'LR = logistic regression; RF = random forest; GB = gradient boosting. Hyperparameters were not selected from the E3 minima.',fontsize=8,color='#596774')
fig.text(.04,.045,'Both evaluations reuse the development data. They are not independent evidence; no locked-test or external data were evaluated.',fontsize=8,color='#596774')
for ext in ['png','pdf','svg']:fig.savefig(OUT/f'Figure_F_fair_model_comparison.{ext}',dpi=600,bbox_inches='tight',pad_inches=.13)
plt.close(fig)
print(table.to_string(index=False))
