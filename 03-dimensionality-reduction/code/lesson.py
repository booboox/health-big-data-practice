"""课程实验代码，按单元格顺序提取自 notebook/pbl-high-dimension-reduction-classroom.ipynb。
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
"""Single-cohort experiments; all fit operations explicitly respect patient splits.

No synthetic expression, no feature order chosen by labels. Reproduce with run_all().
"""
import os
for _k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:
 os.environ[_k]='1'
from pathlib import Path
import json,re,hashlib,warnings,time
import numpy as np,pandas as pd
from scipy.spatial.distance import pdist,squareform,cdist
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from threadpoolctl import threadpool_limits
threadpool_limits(1)
plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'font.size':10})
ROOT=Path.cwd() if (Path.cwd()/'data').exists() else Path.cwd().parent
OUT=ROOT/'assets'/'pbl-high-dimension-reduction';OUT.mkdir(parents=True,exist_ok=True)
DATA=ROOT/'data';SEED=20260921
_cache=ROOT/'review'/'numba-cache';_cache.mkdir(parents=True,exist_ok=True)
os.environ['NUMBA_CACHE_DIR']=str(_cache)
COUNTS=[20,100,500,2000,5000]

import sys
sys.path.insert(0,str(ROOT/"scripts"))
from IPython.display import display, Image

# %% 原 Notebook 单元格 3
def dump(name,obj):
 (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x)),encoding='utf-8')

def csv(name,rows):
 df=rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows)
 df.to_csv(OUT/name,index=False,encoding='utf-8-sig');return df

def plot_save(name):
 plt.tight_layout();plt.savefig(OUT/name,dpi=160);plt.close()

def load():
 a=np.load(DATA/'cohort120.npz',allow_pickle=True)
 return {k:a[k] for k in a.files}

def dev_data(a):
 d=a['development'];return a['X'][d],a['y'][d],a['ids'][d],a['genes']
a=load()
X,y,ids,genes=dev_data(a)
assert a["X"].shape==(120,23146)
assert len(X)==80 and len(a["test"])==40
print("Shape",a["X"].shape,"seen risk counts",np.bincount(y),"unseen risk counts",np.bincount(a["y"][a["test"]]))
print(pd.read_csv(DATA/"patient-manifest.csv").head())

# %% 原 Notebook 单元格 5
order=np.random.default_rng(SEED+100).permutation(X.shape[1])
fig,ax=plt.subplots(figsize=(9,4))
for k in COUNTS+[X.shape[1]]:
 z=StandardScaler().fit_transform(X[:,order[:k]])
 dd=squareform(pdist(z));np.fill_diagonal(dd,np.inf);near=dd.min(1);np.fill_diagonal(dd,0);far=dd.max(1)
 print(k,"mean",pdist(z).mean(),"near",near.mean(),"far",far.mean(),"contrast",np.mean((far-near)/near))
 ax.hist(pdist(z)/pdist(z).mean(),bins=30,density=True,histtype="step",label=str(k))
ax.set(xlabel="距离 / 同维数平均距离",ylabel="密度",title="同一80人；第1种排列；20至23146标准化基因")
ax.legend(title="基因数")
plot_save("classroom-distances.png")
display(Image(filename=str(OUT/"classroom-distances.png")))

# %% 原 Notebook 单元格 7
def fixed_demo(a,quick=False):
 """Fixed learner and fixed patients. Gene ordering never uses labels.
 Unseen outcomes are descriptive observations and never control model settings.
 """
 tr=a['development'];te=a['test'];X=a['X'];y=a['y'];p=X.shape[1]
 rows=[];predictions=[];fits=[]
 for rep in range(1 if quick else 20):
  order=np.random.default_rng(SEED+100+rep).permutation(p)
  for k in COUNTS+[p]:
   idx=order[:k];sc=StandardScaler().fit(X[tr][:,idx]);zt=sc.transform(X[tr][:,idx])
   model=LogisticRegression(C=1,solver='liblinear',dual=k>len(tr),max_iter=10000,random_state=SEED).fit(zt,y[tr])
   fits.append(dict(repeat=rep+1,p=k,fit_n=int(sc.n_samples_seen_),fit_ids=';'.join(a['ids'][tr]),gene_order_sha256=hashlib.sha256(idx.tobytes()).hexdigest(),mean_sha256=hashlib.sha256(sc.mean_.tobytes()).hexdigest()))
   for group,ix in [('seen',tr),('unseen',te)]:
    pred=model.predict(sc.transform(X[ix][:,idx]));correct=pred==y[ix]
    r=dict(repeat=rep+1,p=k,group=group,n=len(ix),correct=int(correct.sum()),proportion=float(correct.mean()))
    for lab,name in [(1,'high'),(0,'nonhigh')]:
     mask=y[ix]==lab;r[name+'_n']=int(mask.sum());r[name+'_correct']=int(correct[mask].sum());r[name+'_proportion']=float(correct[mask].mean())
    rows.append(r)
    predictions.extend(dict(repeat=rep+1,p=k,group=group,patient=a['ids'][i],target=int(y[i]),prediction=int(pr)) for i,pr in zip(ix,pred))
  print('Fixed gene ordering',rep+1,'complete',flush=True)
 df=pd.DataFrame(rows)
 if quick:return df
 csv('seen-unseen-results.csv',df);csv('seen-unseen-predictions.csv',predictions);csv('fit-audit.csv',fits)
 summary=df.groupby(['p','group']).agg(proportion=('proportion','mean'),minimum=('proportion','min'),maximum=('proportion','max')).reset_index()
 csv('seen-unseen-summary.csv',summary)
 fig,ax=plt.subplots(figsize=(9,4))
 for group,label in [('seen','见过的80人'),('unseen','未见过的40人')]:
  g=summary[summary.group==group];ax.plot(g.p,g.proportion,'o-',label=label);ax.fill_between(g.p,g.minimum,g.maximum,alpha=.15)
 ax.set(xscale='log',xlabel='真实基因数',ylabel='预测正确比例',ylim=(0,1.05),title=f'固定80/40人；20种嵌套基因排列；p=20至{p}；固定分类器')
 ax.legend();plot_save('seen-unseen.png')
 return df
first=fixed_demo(a,quick=True)
print(first.to_string(index=False))

# %% 原 Notebook 单元格 9
from umap import UMAP
sc=StandardScaler().fit(X);z=sc.transform(X)
pc=PCA(n_components=2,svd_solver="full").fit(z);xy=pc.transform(z)
um=UMAP(n_neighbors=15,min_dist=.1,random_state=SEED,n_jobs=1).fit(z)
print("mean",z.mean(),"median gene sd",np.median(z.std(0)),"PCA ratios",pc.explained_variance_ratio_)
fig,axs=plt.subplots(1,2,figsize=(10,4))
for ax,label,coords in [(axs[0],"PCA",xy),(axs[1],"UMAP",um.embedding_)]:
 ax.scatter(coords[:,0],coords[:,1],c=y,cmap="coolwarm");ax.set(xlabel=label+" 1",ylabel=label+" 2",title=label)
fig.suptitle("同一80人；全部23146基因；蓝非高危，红高危")
plot_save("classroom-pca-umap.png")
display(Image(filename=str(OUT/"classroom-pca-umap.png")))
