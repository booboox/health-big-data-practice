"""Worksheet D3: model-specific feature importance and training correlations."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
import sklearn

BASE=Path(__file__).resolve().parent; OUT=BASE/'工作单配图'
DATA=Path(r'C:/Users/10840/Downloads/9. Ensemble/10. Ensemble/data/student')
t=pd.read_csv(DATA/'train.csv');v=pd.read_csv(DATA/'validation.csv')
features=[c for c in t if c not in ['sample_id','malignant']]
model=RandomForestClassifier(n_estimators=100,max_features='sqrt',random_state=1010,n_jobs=1).fit(t[features],t.malignant)
baseline=roc_auc_score(v.malignant,model.predict_proba(v[features])[:,1])
perm=permutation_importance(model,v[features],v.malignant,scoring='roc_auc',n_repeats=50,random_state=1010,n_jobs=1)
result=pd.DataFrame({'feature':features,'impurity_importance':model.feature_importances_,
    'permutation_mean':perm.importances_mean,'permutation_sd':perm.importances_std})
result['impurity_rank']=result.impurity_importance.rank(ascending=False,method='min').astype(int)
result['permutation_rank']=result.permutation_mean.rank(ascending=False,method='min').astype(int)
result.sort_values('impurity_importance',ascending=False).to_csv(OUT/'Figure_D3_feature_importance.csv',index=False)
pd.DataFrame(perm.importances,index=features).rename_axis('feature').to_csv(OUT/'Figure_D3_permutation_repeats.csv')
corr=t[features].corr();corr.to_csv(OUT/'Figure_D3_training_correlations.csv')
pairs=[]
for i,f in enumerate(features):
    for g in features[i+1:]:
        if abs(corr.loc[f,g])>.9:pairs.append({'feature_1':f,'feature_2':g,'pearson_r':corr.loc[f,g]})
pd.DataFrame(pairs).sort_values('pearson_r',ascending=False).to_csv(OUT/'Figure_D3_high_correlations.csv',index=False)

BLUE,ORANGE,GRAY='#28658A','#C46B35','#596774'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
    'axes.spines.right':False,'axes.linewidth':.7,'pdf.fonttype':42,'ps.fonttype':42})
fig,axes=plt.subplots(1,2,figsize=(11,5.2))
fig.subplots_adjust(left=.19,right=.97,top=.75,bottom=.27,wspace=.88)
fig.text(.04,.955,'Feature importance depends on how it is measured',fontsize=15,weight='bold',color='#25313B')
fig.text(.04,.875,f'WDBC  |  Same 100-tree random forest  |  Validation ROC-AUC = {baseline:.4f}  |  Top 10 features per method',fontsize=9,color=GRAY)
for ax,col,title,color,letter in zip(axes,['impurity_importance','permutation_mean'],
    ['Training impurity reduction','Validation permutation importance'],[BLUE,ORANGE],'ab'):
    sub=result.nlargest(10,col).iloc[::-1]
    positions=np.arange(len(sub))
    if col=='impurity_importance':
        ax.barh(positions,sub[col],height=.60,color=color)
        ax.set_xlabel('Normalized mean impurity decrease')
    else:
        ax.errorbar(sub[col],positions,xerr=sub.permutation_sd,fmt='o',ms=4,color=color,ecolor='#D6A17F',capsize=2,lw=1)
        ax.axvline(0,color='#87929B',lw=.8,ls='--')
        ax.set_xlabel('Decrease in validation ROC-AUC')
    ax.set(yticks=positions,yticklabels=sub.feature)
    ax.set_title(title,loc='left',fontsize=10,weight='bold',pad=16)
    ax.text(-.52,1.11,letter,transform=ax.transAxes,fontsize=13,weight='bold')
    ax.grid(axis='x',color='#E2E7EB',lw=.6);ax.set_axisbelow(True)
fig.text(.04,.16,'Permutation points: mean decrease over 50 shuffles; whiskers: ±1 SD across shuffles (not confidence intervals).',fontsize=8,color=GRAY)
fig.text(.04,.10,'The two panels use different units and may list different features. Correlated features can substitute for one another.',fontsize=8,color=GRAY)
fig.text(.04,.04,'Importances describe this fitted model and evaluation set; they do not identify causal effects.',fontsize=8,color=GRAY)
def save(fig,name):
    for ext in ['png','pdf','svg']:fig.savefig(OUT/f'{name}.{ext}',dpi=600,bbox_inches='tight',pad_inches=.13)
    plt.close(fig)
save(fig,'Figure_D3_importance_comparison')

# Full matrix in the supplied semantic order; small grouped matrix for a readable worksheet insert.
fig,ax=plt.subplots(figsize=(9.7,8.8))
fig.subplots_adjust(left=.24,right=.88,bottom=.28,top=.83)
im=ax.imshow(corr,vmin=-1,vmax=1,cmap='RdBu_r',interpolation='nearest')
ax.set(xticks=range(30),yticks=range(30),xticklabels=features,yticklabels=features)
ax.tick_params(axis='both',labelsize=6.4,length=0)
plt.setp(ax.get_xticklabels(),rotation=65,ha='right',rotation_mode='anchor')
for x in [9.5,19.5]:ax.axhline(x,color='white',lw=1.5);ax.axvline(x,color='white',lw=1.5)
cax=fig.add_axes([.90,.30,.018,.48]);fig.colorbar(im,cax=cax,label='Pearson r')
fig.text(.04,.96,'Correlated predictors carry overlapping information',fontsize=15,weight='bold',color='#25313B')
fig.text(.04,.90,f'WDBC training set only (n = 341)  |  All 30 predictors  |  {len(pairs)} feature pairs with |r| > 0.9',fontsize=9,color=GRAY)
fig.text(.04,.045,'White lines separate mean, standard-error and worst-value feature groups. Correlation does not establish causation.',fontsize=8,color=GRAY)
save(fig,'Figure_D3_full_correlation')
group=['mean radius','mean perimeter','mean area','worst radius','worst perimeter','worst area']
fig,ax=plt.subplots(figsize=(7.5,6.1))
fig.subplots_adjust(left=.24,right=.86,bottom=.26,top=.76)
mat=corr.loc[group,group]
im=ax.imshow(mat,vmin=-1,vmax=1,cmap='RdBu_r')
ax.set(xticks=range(6),yticks=range(6),xticklabels=group,yticklabels=group)
ax.tick_params(length=0,labelsize=8)
plt.setp(ax.get_xticklabels(),rotation=45,ha='right',rotation_mode='anchor')
for i in range(6):
    for j in range(6):ax.text(j,i,f'{mat.iloc[i,j]:.2f}',ha='center',va='center',fontsize=9,color='white' if abs(mat.iloc[i,j])>.6 else '#25313B')
fig.colorbar(im,ax=ax,fraction=.045,pad=.045,label='Pearson r')
fig.text(.04,.95,'Size-related features are strongly correlated',fontsize=14,weight='bold',color='#25313B')
fig.text(.04,.86,'WDBC training set (n = 341)  |  Prespecified radius, perimeter and area feature group',fontsize=8.5,color=GRAY)
fig.text(.04,.04,'Correlated predictors may share split importance and substitute for one another during single-feature permutation.',fontsize=8,color=GRAY)
save(fig,'Figure_D3_size_correlation')
(OUT/'Figure_D3_metadata.json').write_text(json.dumps({'sklearn':sklearn.__version__,'seed':1010,
    'n_estimators':100,'max_features':'sqrt','permutation_repeats':50,'scoring':'roc_auc',
    'permutation_sd_ddof':0,'baseline_validation_auc':baseline,'correlation_data':'training only',
    'high_correlation_pair_count':len(pairs),'test_set_used':False},indent=2),encoding='utf-8')
print('Top impurity:');print(result.nlargest(6,'impurity_importance').to_string(index=False))
print('Top permutation:');print(result.nlargest(6,'permutation_mean').to_string(index=False))
print('Size features:');print(result[result.feature.isin(group)].to_string(index=False))
print('Pairs with |r|>0.9:',len(pairs))
