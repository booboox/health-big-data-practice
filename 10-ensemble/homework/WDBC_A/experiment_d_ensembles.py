"""Worksheet D: fixed-split comparison, shared-bootstrap stability, tree counts."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
import json, sklearn

BASE=Path(__file__).resolve().parent; OUT=BASE/'工作单配图'
DATA=Path(r'C:/Users/10840/Downloads/9. Ensemble/10. Ensemble/data/student')
t=pd.read_csv(DATA/'train.csv'); v=pd.read_csv(DATA/'validation.csv')
features=[c for c in t if c not in ['sample_id','malignant']]
X,y,Xv,yv=t[features],t.malignant,v[features],v.malignant
SEED=1010
def models(n=100):
    return {'Single tree':DecisionTreeClassifier(max_depth=4,random_state=SEED),
        'Bagging':BaggingClassifier(estimator=DecisionTreeClassifier(random_state=SEED),n_estimators=n,max_features=1.0,random_state=SEED,n_jobs=1),
        'Random forest':RandomForestClassifier(n_estimators=n,max_features='sqrt',random_state=SEED,n_jobs=1)}
rows=[]; preds=[]
for name,m in models().items():
    m.fit(X,y); p=m.predict_proba(Xv)[:,1]; pred=m.predict(Xv)
    rows.append({'model':name,'accuracy':accuracy_score(yv,pred),'balanced_accuracy':balanced_accuracy_score(yv,pred),'roc_auc':roc_auc_score(yv,p)})
    preds.extend({'model':name,'sample_id':sid,'malignant':int(yy),'probability':float(pp)} for sid,yy,pp in zip(v.sample_id,yv,p))
perf=pd.DataFrame(rows);perf.to_csv(OUT/'Figure_D_model_comparison.csv',index=False)
pd.DataFrame(preds).to_csv(OUT/'Figure_D_validation_predictions.csv',index=False)
print('Fixed-split comparison complete.',flush=True)
rng=np.random.default_rng(SEED)
indices=[rng.choice(len(y),len(y),replace=True) for _ in range(30)]
stable=[]; individual=[]
for name in models():
    probabilities=[]
    for i,idx in enumerate(indices):
        m=models()[name];m.fit(X.iloc[idx],y.iloc[idx]);p=m.predict_proba(Xv)[:,1]
        probabilities.append(p)
        individual.extend({'model':name,'replicate':i+1,'sample_id':sid,'probability':float(pp)} for sid,pp in zip(v.sample_id,p))
    p=np.array(probabilities); share=(p>=.5).mean(axis=0)
    stable.append({'model':name,'mean_prediction_sd':p.std(axis=0,ddof=0).mean(),
        'class_flip_rate':((share>0)&(share<1)).mean()})
    print('Shared-bootstrap stability complete:',name,flush=True)
st=pd.DataFrame(stable);st.to_csv(OUT/'Figure_D_prediction_stability.csv',index=False)
pd.DataFrame(individual).to_csv(OUT/'Figure_D_bootstrap_predictions.csv',index=False)
curves=[]
for n in [1,10,50,100,300]:
    for name,m in models(n).items():
        if name=='Single tree':continue
        m.fit(X,y)
        curves.append({'model':name,'n_trees':n,'roc_auc':roc_auc_score(yv,m.predict_proba(Xv)[:,1]),'balanced_accuracy':balanced_accuracy_score(yv,m.predict(Xv))})
curve=pd.DataFrame(curves);curve.to_csv(OUT/'Figure_D_tree_count_curve.csv',index=False)
(OUT/'Figure_D_metadata.json').write_text(json.dumps({'seed':SEED,'sklearn':sklearn.__version__,
    'train_n':len(t),'validation_n':len(v),'bootstrap_refits':30,'single_tree_max_depth':4,
    'bagging_base_max_depth':None,'rf_max_depth':None,'n_estimators':100,
    'probability_sd_ddof':0,'flip_threshold':'>=0.5','test_set_used':False},indent=2),encoding='utf-8')

BLUE,ORANGE,TEAL='#28658A','#C46B35','#459C93'
colors=[BLUE,ORANGE,TEAL]; short=['Single\ntree','Bagging','Random\nforest']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,'pdf.fonttype':42,'ps.fonttype':42})
def save(fig,name):
    for ext in ['png','pdf','svg']:fig.savefig(OUT/f'{name}.{ext}',dpi=600,bbox_inches='tight',pad_inches=.13)
    plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(10.5,4.6))
fig.subplots_adjust(left=.07,right=.98,bottom=.28,top=.72,wspace=.40)
fig.text(.04,.95,'Tree ensembles improve discrimination and prediction stability',fontsize=14,weight='bold',color='#25313B')
fig.text(.04,.86,'WDBC  |  Training n = 341; validation n = 114  |  100 trees per ensemble  |  30 shared bootstrap refits',fontsize=9,color='#596774')
for i,(ax,values,title,ylabel) in enumerate(zip(axes,[perf.roc_auc,st.mean_prediction_sd,st.class_flip_rate],
    ['Validation discrimination','Prediction variability','Class switching'],['ROC-AUC','Mean prediction SD','Fraction of validation records'])):
    ax.bar(range(3),values,color=colors,width=.58)
    ax.set(xticks=range(3),xticklabels=short,ylabel=ylabel)
    ax.set_title(title,loc='left',fontsize=10,weight='bold',pad=15)
    ax.text(-.21,1.12,'abc'[i],transform=ax.transAxes,fontsize=13,weight='bold')
    top=1.12 if i==0 else max(values)*1.28
    ax.set_ylim(0,top)
    for x,val in enumerate(values):
        label=f'{val:.4f}' if i<2 else f'{val:.1%}'
        ax.text(x,val+top*.025,label,ha='center',fontsize=9)
    ax.grid(axis='y',color='#E2E7EB',lw=.6);ax.set_axisbelow(True)
fig.text(.04,.16,'SD: variability of predicted malignant probability across refits, averaged over fixed validation records.',fontsize=8,color='#596774')
fig.text(.04,.10,'Class switching: records predicted on both sides of 0.5 across 30 refits. It depends on the number of refits.',fontsize=8,color='#596774')
fig.text(.04,.04,'Single tree: max_depth = 4 (course baseline). Ensembles: unrestricted trees. Point estimates; no confidence intervals estimated.',fontsize=8,color='#596774')
save(fig,'Figure_D1_performance_stability')

fig,axes=plt.subplots(1,2,figsize=(9,4.2))
fig.subplots_adjust(left=.09,right=.97,bottom=.24,top=.73,wspace=.30)
fig.text(.04,.95,'More trees bring diminishing returns',fontsize=15,weight='bold',color='#25313B')
fig.text(.04,.85,'WDBC fixed validation set (n = 114)  |  Tree counts: 1, 10, 50, 100, 300',fontsize=9,color='#596774')
for ax,metric,title,letter in zip(axes,['roc_auc','balanced_accuracy'],['ROC-AUC','Balanced accuracy'],'ab'):
    for name,color,marker in [('Bagging',ORANGE,'s'),('Random forest',TEAL,'o')]:
        sub=curve[curve.model==name]
        ax.plot(sub.n_trees,sub[metric],color=color,marker=marker,ms=4,lw=1.5,label=name)
    ax.set(xscale='log',xticks=[1,10,50,100,300],xticklabels=['1','10','50','100','300'],xlabel='Number of trees (log scale)',ylabel=title,ylim=(.85,1.015))
    ax.text(-.15,1.12,letter,transform=ax.transAxes,fontsize=13,weight='bold')
    ax.grid(axis='y',color='#E2E7EB',lw=.6)
axes[0].legend(frameon=False,fontsize=8,loc='lower right')
fig.text(.04,.10,'Shared split and random seed (1010). Lines connect tested settings; small validation changes may reflect few records.',fontsize=8,color='#596774')
fig.text(.04,.035,'A plateau describes this experiment; adding trees does not guarantee monotonic improvement on a finite validation set.',fontsize=8,color='#596774')
save(fig,'Figure_D2_tree_count')
print(perf.to_string(index=False));print(st.to_string(index=False));print(curve.to_string(index=False))
