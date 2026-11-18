"""E1-E2: explicit binary discrete AdaBoost and sklearn gradient boosting."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
import sklearn

BASE=Path(__file__).resolve().parent;OUT=BASE/'工作单配图'
DATA=Path(r'C:/Users/10840/Downloads/9. Ensemble/10. Ensemble/data/student')
t=pd.read_csv(DATA/'train.csv').sort_values('sample_id').reset_index(drop=True)
features=[c for c in t if c not in ['sample_id','malignant']]
X=t[features]; y=t.malignant.to_numpy(); signed=2*y-1
weights=np.ones(len(t))/len(t);records=[];rounds=[];weight_columns=[]
for r in range(1,9):
    stump=DecisionTreeClassifier(max_depth=1,random_state=1010)
    stump.fit(X,y,sample_weight=weights)
    pred=stump.predict(X);wrong=pred!=y
    err=float(weights[wrong].sum())
    if not 0<err<.5:raise RuntimeError(f'Invalid weak-learner error: {err}')
    alpha=.5*np.log((1-err)/err)
    raw=weights*np.exp(-alpha*signed*(2*pred-1))
    new=raw/raw.sum()
    weight_columns.append(weights.copy())
    rounds.append({'round':r,'weighted_error':err,'alpha':alpha,'n_wrong':int(wrong.sum()),
        'mean_ratio_correct':float(np.mean(new[~wrong]/weights[~wrong])),
        'mean_ratio_wrong':float(np.mean(new[wrong]/weights[wrong])),
        'root_feature':features[stump.tree_.feature[0]],'root_threshold':stump.tree_.threshold[0]})
    records.extend({'round':r,'sample_id':sid,'malignant':int(yy),'prediction':int(pr),
        'wrong':bool(wr),'weight_before':float(w),'weight_after':float(nw)}
        for sid,yy,pr,wr,w,nw in zip(t.sample_id,y,pred,wrong,weights,new))
    weights=new
ad=pd.DataFrame(rounds);ad.to_csv(OUT/'Figure_E1_adaboost_rounds.csv',index=False)
pd.DataFrame(records).to_csv(OUT/'Figure_E1_adaboost_weights.csv',index=False)

BLUE,ORANGE,GRAY='#28658A','#C46B35','#596774'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
    'axes.spines.right':False,'axes.linewidth':.7,'pdf.fonttype':42,'ps.fonttype':42})
def save(fig,name):
    for ext in ['png','pdf','svg']:fig.savefig(OUT/f'{name}.{ext}',dpi=600,bbox_inches='tight',pad_inches=.13)
    plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(10,4.8),gridspec_kw={'width_ratios':[1,1.1]})
fig.subplots_adjust(left=.085,right=.96,top=.73,bottom=.27,wspace=.45)
fig.text(.04,.95,'AdaBoost reweights records after each weak learner',fontsize=15,weight='bold',color='#25313B')
fig.text(.04,.86,'WDBC training set (n = 341)  |  Eight sequential decision stumps  |  Explicit binary discrete AdaBoost',fontsize=9,color=GRAY)
ax,bx=axes
matrix=np.array(weight_columns).T*len(t)
im=ax.imshow(matrix,aspect='auto',cmap='YlOrBr',norm=LogNorm(vmin=min(.1,matrix.min()),vmax=max(10,matrix.max())),interpolation='nearest',extent=[.5,8.5,len(t)+.5,.5])
ax.set(xticks=range(1,9),xlabel='Boosting round (before fitting)',ylabel='Training record (ordered by sample ID)')
fig.colorbar(im,ax=ax,fraction=.045,pad=.03,label='Weight / initial uniform weight')
ax.set_title('The training distribution evolves',loc='left',fontsize=10,weight='bold',pad=15)
ax.text(-.19,1.12,'a',transform=ax.transAxes,fontsize=13,weight='bold')
bx.plot(ad['round'],ad.mean_ratio_wrong,color=ORANGE,marker='s',ms=4,label='Misclassified this round')
bx.plot(ad['round'],ad.mean_ratio_correct,color=BLUE,marker='o',ms=4,label='Correct this round')
bx.axhline(1,color='#84909A',ls='--',lw=.8)
bx.set(xlabel='Boosting round',ylabel='Normalized weight after / before',xticks=range(1,9),yscale='log')
bx.set_title('Errors receive more weight next round',loc='left',fontsize=10,weight='bold',pad=15)
bx.text(-.19,1.12,'b',transform=bx.transAxes,fontsize=13,weight='bold')
bx.legend(frameon=False,fontsize=8,loc='upper right')
bx.grid(axis='y',color='#E2E7EB',lw=.6)
fig.text(.04,.16,r'$w_i^{(t+1)}\propto w_i^{(t)}\exp[-\alpha_t y_i h_t(x_i)]$, with signed labels and predictions in {−1, +1}.',fontsize=9)
fig.text(.04,.10,'Panel b shows the within-group mean ratio; all records in each group have the same multiplier in a given round.',fontsize=8,color=GRAY)
fig.text(.04,.04,'Each stump depends on the preceding weights. The dashed line marks no weight change. No validation or test records used.',fontsize=8,color=GRAY)
save(fig,'Figure_E1_adaboost_weight_updates')

gb=GradientBoostingClassifier(loss='log_loss',n_estimators=20,learning_rate=.1,max_depth=3,random_state=1010).fit(X,y)
p=list(gb.staged_predict_proba(X))[4][:,1]
residual=y-p;correct=(p>=.5)==y
gd=pd.DataFrame({'sample_id':t.sample_id,'malignant':y,'probability_after_5':p,'negative_gradient':residual,'correct_at_0_5':correct})
gd.to_csv(OUT/'Figure_E2_negative_gradients.csv',index=False)
stats=gd.groupby('correct_at_0_5').agg(n=('sample_id','size'),mean_abs_gradient=('negative_gradient',lambda s: np.abs(s).mean()))
fig,axes=plt.subplots(1,2,figsize=(10,4.8),gridspec_kw={'width_ratios':[1.15,1]})
fig.subplots_adjust(left=.09,right=.97,top=.73,bottom=.26,wspace=.36)
fig.text(.04,.95,'Gradient boosting updates all nonzero loss gradients',fontsize=15,weight='bold',color='#25313B')
fig.text(.04,.86,'WDBC training set  |  State after five gradient-boosting stages  |  Binary log-loss; negative gradient = y − p',fontsize=9,color=GRAY)
ax,bx=axes
grid=np.linspace(0,1,200)
ax.plot(grid,1-grid,color=ORANGE,lw=1,alpha=.55)
ax.plot(grid,-grid,color=BLUE,lw=1,alpha=.55)
for val,color,label in [(0,BLUE,'Benign (y = 0)'),(1,ORANGE,'Malignant (y = 1)')]:
    idx=y==val
    ax.scatter(p[idx],residual[idx],color=color,s=16,alpha=.45,edgecolors='none',label=label)
ax.axvline(.5,color='#8A959D',ls='--',lw=.8);ax.axhline(0,color='#8A959D',lw=.8)
ax.set(xlim=(0,1),ylim=(-1.03,1.03),xlabel='Current malignant probability, p',ylabel='Negative gradient, y − p')
ax.set_title('Direction depends on probability and label',loc='left',fontsize=10,weight='bold',pad=15)
ax.text(-.17,1.12,'a',transform=ax.transAxes,fontsize=13,weight='bold')
ax.legend(frameon=False,fontsize=8,loc='lower left')
vals=[stats.loc[True,'mean_abs_gradient'],stats.loc[False,'mean_abs_gradient']]
bx.bar([0,1],vals,color=[BLUE,ORANGE],width=.55)
bx.set(xticks=[0,1],xticklabels=[f'Correct\n(n = {stats.loc[True,"n"]})',f'Incorrect\n(n = {stats.loc[False,"n"]})'],ylabel='Mean absolute negative gradient',ylim=(0,max(vals)*1.25))
for i,val in enumerate(vals):bx.text(i,val+.015,f'{val:.3f}',ha='center',fontsize=9)
bx.set_title('Correct predictions still contribute',loc='left',fontsize=10,weight='bold',pad=15)
bx.text(-.17,1.12,'b',transform=bx.transAxes,fontsize=13,weight='bold')
bx.grid(axis='y',color='#E2E7EB',lw=.6);bx.set_axisbelow(True)
fig.text(.04,.15,'The next tree fits the negative gradients evaluated at the current model; leaf updates then optimize the loss.',fontsize=8,color=GRAY)
fig.text(.04,.09,'Positive y − p requests an increase in the raw score; negative y − p requests a decrease.',fontsize=8,color=GRAY)
fig.text(.04,.035,'These are training diagnostics, not estimates of validation performance. All data and plotted quantities are saved for reproduction.',fontsize=8,color=GRAY)
save(fig,'Figure_E2_gradient_boosting_residuals')
(OUT/'Figure_E12_metadata.json').write_text(json.dumps({'sklearn':sklearn.__version__,
    'seed':1010,'adaboost':'Explicit binary discrete AdaBoost; alpha=0.5*log((1-error)/error); full normalized exponential weight update',
    'adaboost_rounds':8,'adaboost_base_depth':1,'sort':'sample_id ascending',
    'gradient_boosting':{'n_estimators':20,'learning_rate':.1,'max_depth':3,'diagnostic_stage':5},
    'validation_used':False,'test_used':False},indent=2),encoding='utf-8')
print(ad.to_string(index=False));print(stats.to_string())
