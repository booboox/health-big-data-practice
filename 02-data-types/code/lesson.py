"""课程实验代码，按单元格顺序提取自 notebook/pbl-understanding-data-lab.ipynb。
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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
ASSET_DIR = Path('../assets/pbl-understanding-data')
heart = pd.read_csv(ASSET_DIR / 'heart-disease-four-site.csv')
heart['disease_binary'] = (heart['num'] > 0).astype(int)
cancer_bunch = load_breast_cancer(as_frame=True)
cancer_x = cancer_bunch.data.copy()
print(heart.shape, cancer_x.shape)

# %% 原 Notebook 单元格 4
dictionary = pd.read_csv(ASSET_DIR / 'packet-b-data-dictionary.csv')
scale_cases = pd.read_csv(ASSET_DIR / 'packet-b-measurement-scale-cases.csv')
structures = pd.read_csv(ASSET_DIR / 'packet-b-structure-types.csv')
display(dictionary)
display(scale_cases)
display(structures)
assert dictionary.loc[dictionary.variable.eq('cp'), 'measurement_scale'].iloc[0] == 'nominal multiclass'
assert set(scale_cases.measurement_scale) >= {'nominal','ordinal','interval','ratio'}

# %% 原 Notebook 单元格 7
numeric = [c for c in heart.columns if c not in ['record_id','site','disease_binary']]
missing_overall = heart[numeric].isna().sum().sort_values(ascending=False)
missing_by_site = heart.groupby('site')[numeric].apply(lambda x: x.isna().mean()*100)
display(missing_overall.to_frame('missing_n'))
display(missing_by_site)
print('duplicate feature rows', heart.drop(columns=['record_id','disease_binary']).duplicated().sum())
print('recorded zeros', {c:int((heart[c]==0).sum()) for c in ['trestbps','chol','thalach']})
display(pd.read_csv(ASSET_DIR / 'packet-b-quality-evidence-boundaries.csv'))
print(pd.crosstab(heart.site, heart.num, normalize='index').round(3))

# %% 原 Notebook 单元格 10
cols = ['age','trestbps','chol','thalach','oldpeak']
x = heart[cols].fillna(heart[cols].median())
raw = x.to_numpy()
z = StandardScaler().fit_transform(raw)
d_raw = pairwise_distances(raw, metric='euclidean')
d_z_e = pairwise_distances(z, metric='euclidean')
d_z_m = pairwise_distances(z, metric='manhattan')
d_z_p3 = pairwise_distances(z, metric='minkowski', p=3)
VI = np.linalg.pinv(np.cov(z, rowvar=False))
d_z_mahal = pairwise_distances(z, metric='mahalanobis', VI=VI)
anchor = 0
for name, d in [('raw Euclidean', d_raw), ('standardized Euclidean', d_z_e), ('standardized Manhattan', d_z_m), ('Minkowski p3', d_z_p3), ('Mahalanobis', d_z_mahal)]:
    order = np.argsort(d[anchor])[1:7]
    print(name, heart.loc[order, 'record_id'].tolist())
# Mahalanobis additionally depends on a stable covariance estimate.
print(pd.read_csv(ASSET_DIR / 'packet-c-metric-comparison.csv'))
# cosine and correlation respond differently to scaling and translation
v = z[0]
for label, w in [('scaled', 2*v), ('shifted', v+2), ('reversed', v[::-1])]:
    cosine = np.dot(v,w)/(np.linalg.norm(v)*np.linalg.norm(w))
    correlation = np.corrcoef(v,w)[0,1]
    print(label, 'cosine', round(cosine,3), 'correlation', round(correlation,3))

# %% 原 Notebook 单元格 11
mi_frame = heart[['age','chol','cp','exang','disease_binary']].copy()
mi_frame['age_bin'] = pd.qcut(mi_frame.age, 5, duplicates='drop').cat.codes
mi_frame['chol_bin'] = pd.qcut(mi_frame.chol, 5, duplicates='drop').cat.codes
mi_x = mi_frame[['age_bin','chol_bin','cp','exang']].fillna(-1).astype(int)
mi = mutual_info_classif(mi_x, mi_frame.disease_binary, discrete_features=True, random_state=42)
pd.Series(mi, index=mi_x.columns, name='mutual_information')

# %% 原 Notebook 单元格 14
binary = pd.read_csv(ASSET_DIR / 'packet-c-binary-similarity.csv')
mixed = pd.read_csv(ASSET_DIR / 'packet-c-mixed-similarity.csv')
display(binary)
display(mixed)
print('SMC counts joint zeros; Jaccard ignores them. Mixed distance excludes missing dimensions and rescales continuous variables.')

# %% 原 Notebook 单元格 17
sampling = pd.read_csv(ASSET_DIR / 'packet-c-sampling-replicates.csv')
display(sampling.groupby('method')[['max_site_proportion_error','disease_prevalence_abs_error']].agg(['mean','std','max']))
aggregation = heart.groupby('site').agg(n=('record_id','size'), age_mean=('age','mean'), age_sd=('age','std'), prevalence=('disease_binary','mean'))
display(aggregation)
print('Patient rows:', len(heart), 'Aggregated site rows:', len(aggregation))

# %% 原 Notebook 单元格 20
for col in ['chol','oldpeak','trestbps']:
    s = heart[col].dropna()
    print(col, 'raw skew', round(s.skew(),3), 'log1p skew', round(np.log1p(s-s.clip(upper=0).min()).skew(),3))
print(pd.cut(heart.age.dropna(), 5).value_counts().sort_index())
print(pd.qcut(heart.age.dropna(), 5, duplicates='drop').value_counts().sort_index())

# %% 原 Notebook 单元格 21
frame = heart[['age','trestbps','chol','thalach','oldpeak','disease_binary']]
train, test = train_test_split(frame, test_size=.30, random_state=42, stratify=frame.disease_binary)
train_x, test_x = train.drop(columns='disease_binary'), test.drop(columns='disease_binary')
train_median = train_x.median()
scaler = StandardScaler().fit(train_x.fillna(train_median))
test_z = scaler.transform(test_x.fillna(train_median))
print('Train-fitted test means', pd.Series(test_z.mean(axis=0), index=train_x.columns).round(3))
print('The test set did not determine medians or scaling parameters.')

# %% 原 Notebook 单元格 24
z_bc = StandardScaler().fit_transform(cancer_x)
pca = PCA().fit(z_bc)
cum = np.cumsum(pca.explained_variance_ratio_)
print('components for 90%', np.argmax(cum>=.90)+1)
print('components for 95%', np.argmax(cum>=.95)+1)
plt.plot(np.arange(1,31), cum, marker='o'); plt.axhline(.9, ls='--'); plt.xlabel('components'); plt.ylabel('cumulative variance'); plt.show()
for d in [2,5,10,20,30]:
    distances = pairwise_distances(z_bc[:150,:d])
    vals = distances[np.triu_indices_from(distances,k=1)]
    print(d, 'CV of distances', round(vals.std()/vals.mean(),3))

# %% 原 Notebook 单元格 25
loadings = pd.DataFrame(PCA(n_components=3).fit(z_bc).components_.T, index=cancer_x.columns, columns=['PC1','PC2','PC3'])
for pc in loadings:
    print(pc, loadings[pc].abs().nlargest(6).index.tolist())

# %% 原 Notebook 单元格 28
workflow = pd.read_csv(ASSET_DIR / 'packet-d-workflow-decision.csv')
boundaries = pd.read_csv(ASSET_DIR / 'packet-d-boundary-checklist.csv')
display(workflow)
display(boundaries)
assert set(workflow.inquiry_questions.str.cat(sep=' ').split()) >= {f'Q{i}' for i in range(1,11)}
