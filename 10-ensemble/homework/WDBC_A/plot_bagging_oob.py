"""Worksheet C3-C4: bagging workflow and an actual WDBC OOB check."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
import sklearn

BASE=Path(__file__).resolve().parent
OUT=BASE/'工作单配图'
DATA=Path(r'C:/Users/10840/Downloads/9. Ensemble/10. Ensemble/data/student')
train=pd.read_csv(DATA/'train.csv'); valid=pd.read_csv(DATA/'validation.csv')
features=[c for c in train if c not in ['sample_id','malignant']]
model=BaggingClassifier(estimator=DecisionTreeClassifier(random_state=1010), n_estimators=100,
    max_features=1.0, bootstrap=True, oob_score=True, random_state=1010, n_jobs=1)
model.fit(train[features],train.malignant)
counts=np.zeros(len(train),dtype=int)
for idx in model.estimators_samples_:
    included=np.zeros(len(train),dtype=bool); included[idx]=True
    counts+=~included
assert (counts>0).all()
rows=[]
for label,y,prob in [('OOB',train.malignant,model.oob_decision_function_[:,1]),
                     ('Validation',valid.malignant,model.predict_proba(valid[features])[:,1])]:
    pred=(prob>.5).astype(int)  # sklearn argmax resolves an exact tie to class 0
    rows.append({'evaluation':label,'n':len(y),'accuracy':accuracy_score(y,pred),
        'balanced_accuracy':balanced_accuracy_score(y,pred),'roc_auc':roc_auc_score(y,prob)})
result=pd.DataFrame(rows)
result.to_csv(OUT/'Figure_C34_oob_validation_results.csv',index=False)
pd.DataFrame({'sample_id':train.sample_id,'malignant':train.malignant,
    'oob_tree_count':counts,'oob_probability':model.oob_decision_function_[:,1]}).to_csv(OUT/'Figure_C34_oob_predictions.csv',index=False)
(OUT/'Figure_C34_metadata.json').write_text(json.dumps({'sklearn':sklearn.__version__,'seed':1010,
    'n_estimators':100,'base_tree_max_depth':None,'max_features':1.0,
    'oob_tree_count_min':int(counts.min()),'oob_tree_count_mean':float(counts.mean()),
    'oob_tree_count_max':int(counts.max()),'test_set_used':False},indent=2),encoding='utf-8')

BLUE,ORANGE,INK,GRAY='#28658A','#C46B35','#25313B','#596774'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'ps.fonttype':42})
fig,axes=plt.subplots(1,2,figsize=(11,5.5))
fig.subplots_adjust(left=.045,right=.97,top=.76,bottom=.27,wspace=.20)
fig.text(.04,.95,'Bagging trains in parallel; OOB predicts with unseen trees',fontsize=15,weight='bold',color=INK)
fig.text(.04,.88,'WDBC  |  341 training records  |  100 full-feature bootstrap trees  |  Random seed = 1010',fontsize=9,color=GRAY)
for a in axes: a.set(xlim=(0,1),ylim=(0,1)); a.axis('off')
def box(a,x,y,w,h,text,color=BLUE,fill='#EDF4F7',fs=9):
    a.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle='round,pad=0.012,rounding_size=0.025',fc=fill,ec=color,lw=1))
    a.text(x,y,text,ha='center',va='center',color=color,fontsize=fs)
def arrow(a,p,q,color='#9CAEB9'):
    a.annotate('',xy=q,xytext=p,arrowprops={'arrowstyle':'->','color':color,'lw':1})
ax,bx=axes
ax.set_title('Independent fitting, then aggregation',loc='left',fontsize=11,weight='bold',pad=18)
ax.text(-.04,1.12,'a',transform=ax.transAxes,fontsize=13,weight='bold')
box(ax,.5,.91,.57,.13,'Training data (n = 341)')
for x,k in zip([.15,.5,.85],['1','2','B']):
    box(ax,x,.65,.26,.15,f'Bootstrap {k}\n341 draws',fs=8)
    arrow(ax,(.5,.83),(x,.74))
    box(ax,x,.35,.22,.13,f'Tree {k}')
    arrow(ax,(x,.56),(x,.43))
    arrow(ax,(x,.27),(.5,.16))
ax.text(.5,.49,'Fit separately; no preceding model required',ha='center',fontsize=8,color=GRAY)
box(ax,.5,.09,.70,.12,'Mean predicted probability → class label',fs=8.5)

bx.set_title('OOB evaluation for one training record',loc='left',fontsize=11,weight='bold',pad=18)
bx.text(-.04,1.12,'b',transform=bx.transAxes,fontsize=13,weight='bold')
box(bx,.5,.91,.60,.13,'Record i: known training label')
box(bx,.23,.65,.37,.17,'Included in a tree’s\nbootstrap sample',GRAY,'#F1F3F5',8)
box(bx,.77,.65,.37,.17,'Absent from a tree’s\nbootstrap sample',BLUE,'#EDF4F7',8)
arrow(bx,(.5,.83),(.23,.75));arrow(bx,(.5,.83),(.77,.75))
bx.text(.23,.45,'Exclude this tree\nfrom record i’s\nOOB prediction',ha='center',va='center',fontsize=9,color=GRAY)
box(bx,.77,.42,.37,.15,'Average only these\ntrees’ probabilities',fs=8)
arrow(bx,(.77,.55),(.77,.51))
box(bx,.5,.11,.85,.15,'Repeat for every training record;\ncompare OOB predictions with known labels',ORANGE,'#FCF1E9',8.5)
arrow(bx,(.77,.33),(.5,.20))
fig.text(.04,.175,f'OOB accuracy: {result.iloc[0].accuracy:.2%}     Validation accuracy: {result.iloc[1].accuracy:.2%}',fontsize=10,weight='bold',color=INK)
fig.text(.04,.105,f'Every training record received OOB predictions: {counts.min()}–{counts.max()} eligible trees per record (mean {counts.mean():.1f}).',fontsize=9,color=GRAY)
fig.text(.04,.045,'OOB uses training records with record-specific held-out trees. Validation uses separate records and all trees. Test data remain untouched.',fontsize=8,color=GRAY)
for ext in ['png','pdf','svg']:
    fig.savefig(OUT/f'Figure_C34_bagging_oob.{ext}',dpi=600,bbox_inches='tight',pad_inches=.13)
plt.close(fig)
print(result.to_string(index=False));print('OOB trees per record:',counts.min(),counts.mean(),counts.max())
