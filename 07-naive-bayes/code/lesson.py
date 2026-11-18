"""课程实验代码，按单元格顺序提取自 notebook/pbl-machine-learning-basics-naive-bayes-lab.ipynb。
来源：用户提供的原始课程资料包；并非新增完成的个人作业。
建议优先在原 Notebook 中逐格运行；本脚本仅做语法核对，未完整重跑实验。
"""
from pathlib import Path as _CoursePath
import os as _course_os
import sys as _course_sys
_course_root = _CoursePath(__file__).resolve().parents[1]
_course_os.chdir(_course_root / 'notebook')
_course_sys.path.insert(0, str(_course_root / "scripts"))

# %% 原 Notebook 单元格 2
from pathlib import Path
import sys, json, itertools
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import confusion_matrix, log_loss, mean_squared_error, roc_auc_score, precision_recall_curve, average_precision_score, auc, brier_score_loss
root = Path.cwd().resolve()
if root.name == 'notebook': root = root.parent
assert (root/'assets').exists(), '请从本课目录或notebook目录运行'
assets = root/'assets/pbl-machine-learning-basics-naive-bayes'
df = pd.read_csv(assets/'acute-inflammations.csv')
print('Python:', sys.executable)
print(df.shape)
display(df.head())

# %% 原 Notebook 单元格 4
assert df.shape == (120,8)
assert df.isna().sum().sum() == 0
display(df.describe(include='all'))
print('全行重复余量:', df.duplicated().sum(), '重复组总行:', df.duplicated(keep=False).sum())
print('原始输入组合:', df.iloc[:,:6].drop_duplicates().shape[0])
X = np.column_stack([(df.iloc[:,0]>=38).astype(int), *(df[c].eq('yes').astype(int) for c in df.columns[1:6])])
y = df['bladder_inflammation'].eq('yes').to_numpy(dtype=int)
print('二值组合:', np.unique(X,axis=0).shape[0], '类别计数:', np.bincount(y))

# %% 原 Notebook 单元格 6
dev, te = train_test_split(np.arange(len(y)), test_size=.2, random_state=7, stratify=y)
tr, va = train_test_split(dev, test_size=.25, random_state=7, stratify=y[dev])
assert not (set(tr)&set(va) or set(tr)&set(te) or set(va)&set(te))
for name, ix in [('训练',tr),('验证',va),('测试',te)]: print(name, len(ix), np.bincount(y[ix]))
saved=pd.read_csv(assets/'split-indices.csv')
for label,ix in [('tr',tr),('va',va),('te',te)]: assert saved.loc[saved.split==label,'source_row'].tolist()==(ix+1).tolist()

# %% 原 Notebook 单元格 8
counts=np.array([X[tr][y[tr]==c].sum(axis=0) for c in [0,1]])
n=np.bincount(y[tr]); theta=(counts+1)/(n[:,None]+2); priors=n/n.sum()
model=BernoulliNB(alpha=1).fit(X[tr],y[tr])
assert np.allclose(theta,np.exp(model.feature_log_prob_))
q=np.array([0,0,0,1,1,0])
factors=np.where(q,theta,1-theta)
scores=priors*factors.prod(axis=1); post=scores/scores.sum()
print('counts',counts,'theta',theta,'priors',priors,sep='\n')
print('因子',factors,'类别分数',scores,'后验',post,sep='\n')
assert np.allclose(post,model.predict_proba([q])[0])
saved_counts=pd.read_csv(assets/'packet-b1.csv'); assert np.array_equal(saved_counts.iloc[:,1:].to_numpy(),counts[[1,0]].T)

# %% 原 Notebook 单元格 11
toy_y=np.array([1,0,1,0]); pa=np.array([.8,.2,.4,.1]); pb=np.array([.99,.01,.01,.01])
for name,p in [('甲',pa),('乙',pb)]:
    per=-(toy_y*np.log(p)+(1-toy_y)*np.log(1-p))
    print(name,'逐例损失',per,'平均',per.mean(),'accuracy',((p>=.5)==toy_y).mean())
    assert np.isclose(per.mean(),log_loss(toy_y,p))
print('MSE',mean_squared_error([37,39],[38,37]),'℃²')
assert mean_squared_error([37,39],[38,37])==2.5

# %% 原 Notebook 单元格 13
p=model.predict_proba(X[va])[:,1]
rows=[]
for threshold in [.1,.5,.9]:
    tn,fp,fn,tp=confusion_matrix(y[va],p>=threshold,labels=[0,1]).ravel()
    rows.append([threshold,tn,fp,fn,tp,(tp+tn)/len(va),tp/(tp+fn),tn/(tn+fp),tp/(tp+fp),2*tp/(2*tp+fp+fn)])
metric_table=pd.DataFrame(rows,columns=['threshold','TN','FP','FN','TP','accuracy','sensitivity','specificity','precision','F1'])
display(metric_table)
assert np.array_equal(metric_table.iloc[:3,1:5].to_numpy(),pd.read_csv(assets/'packet-c1.csv').iloc[:3,1:].to_numpy())
print('先验基线precision分母为0：未定义；recall=0。')

# %% 原 Notebook 单元格 14
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
fpr,tpr,_=roc_curve(y[va],p); precision,recall,_=precision_recall_curve(y[va],p)
fig,ax=plt.subplots(1,2,figsize=(9,3))
ax[0].plot(fpr,tpr);ax[0].set(xlabel='False positive rate',ylabel='Sensitivity',title='Validation ROC')
ax[1].plot(recall,precision);ax[1].set(xlabel='Recall',ylabel='Precision',title='Validation PR');plt.show()
for name,prob in [('original',p),('0.25+0.50p',.25+.5*p)]:
    pp,rr,_=precision_recall_curve(y[va],prob)
    print(name, 'ROC',roc_auc_score(y[va],prob),'PR trapezoid',auc(rr,pp),'AP',average_precision_score(y[va],prob),'logloss',log_loss(y[va],prob))
# 非完美排序支持例，说明AP与梯形PR面积一般不等
pp,rr,_=precision_recall_curve([1,0,1,0],[.9,.8,.7,.1])
print('Optional illustration: AP',average_precision_score([1,0,1,0],[.9,.8,.7,.1]),'PR trapezoid',auc(rr,pp))

# %% 原 Notebook 单元格 15
cal=[]
for lo,hi in [(0,.1),(.1,.5),(.5,.9),(.9,1)]:
    mask=(p>=lo)&((p<hi) if hi<1 else (p<=hi))
    if mask.sum(): cal.append([lo,hi,int(mask.sum()),int(y[va][mask].sum()),p[mask].mean(),y[va][mask].mean()])
cal=pd.DataFrame(cal,columns=['lower','upper','n','positive','mean_p','observed_rate']);display(cal)
saved_cal=pd.read_csv(assets/'packet-c3.csv')
assert np.allclose(cal.iloc[:,2:].to_numpy(),saved_cal.iloc[:,1:].to_numpy(),atol=5e-5)
print('Brier',brier_score_loss(y[va],p),'（不是纯校准误差）')

# %% 原 Notebook 单元格 18
joint=[]
for c in [1,0]:
    for a,b in itertools.product([0,1],repeat=2):
        joint.append([c,a,b,int(((y[tr]==c)&(X[tr,3]==a)&(X[tr,4]==b)).sum())])
display(pd.DataFrame(joint,columns=['class','urine','pain','n']))
assert np.array_equal(np.array(joint),pd.read_csv(assets/'packet-d1.csv').values)
grid=np.array(list(itertools.product([0,1],repeat=6)))
dup=BernoulliNB(alpha=1).fit(np.column_stack([X[tr],X[tr,1]]),y[tr])
pg=model.predict_proba(grid)[:,1];pdg=dup.predict_proba(np.column_stack([grid,grid[:,1]]))[:,1]
print('类别改变',((pg>=.5)!=(pdg>=.5)).sum(),'最大概率差',abs(pg-pdg).max())
assert np.allclose(np.column_stack([grid,pg,pdg]),pd.read_csv(assets/'redundancy-all64.csv').to_numpy())

# %% 原 Notebook 单元格 19
testp=model.predict_proba(X[te])[:,1]
print('固定测试',confusion_matrix(y[te],testp>=.5,labels=[0,1]),'log loss',log_loss(y[te],testp))
train_patterns=set(map(tuple,X[tr]));raw=df.iloc[:,:6].astype(str).to_numpy();raw_train=set(map(tuple,raw[tr]))
for name,ix in [('validation',va),('test',te)]:
    print(name,'binary patterns',len(set(map(tuple,X[ix]))),'seen',sum(tuple(v) in train_patterns for v in X[ix]),'raw seen',sum(tuple(v) in raw_train for v in raw[ix]))
assert np.array_equal(confusion_matrix(y[te],testp>=.5),[[12,0],[0,12]])
TP,FN,FP,TN=95,5,495,9405
print('假设低患病率precision',TP/(TP+FP),'accuracy',(TP+TN)/10000,'all negative accuracy',.99)

# %% 原 Notebook 单元格 21
two=BernoulliNB(alpha=1).fit(X[tr][:,[3,4]],y[tr])
classroom=pd.read_csv(assets/'packet-b3.csv')
classroom['posterior']=two.predict_proba(classroom.iloc[:,:2].to_numpy())[:,1]
classroom['prediction']=(classroom['posterior']>=.5).astype(int);display(classroom)
cm=confusion_matrix(y[va],two.predict(X[va][:,[3,4]]),labels=[0,1]); print(cm)
weighted=np.zeros((2,2),dtype=int)
for row in classroom.itertuples(index=False,name=None): weighted[int(row[2]),int(row[5])]+=int(row[3])
assert np.array_equal(cm,weighted)
tn,fp,fn,tp=cm.ravel();print('sensitivity',tp/(tp+fn),'specificity',tn/(tn+fp))
print('核心复现检查全部通过。')
