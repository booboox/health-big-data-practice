"""课程实验代码，按单元格顺序提取自 notebook/pbl-svm-margin-lab.ipynb。
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
import importlib.util
import numpy as np, pandas as pd
from IPython.display import Image, display
ROOT=Path.cwd().parent if Path.cwd().name=='notebook' else Path.cwd()
spec=importlib.util.spec_from_file_location('lesson9',ROOT/'scripts'/'pbl-svm-margin-build.py')
lesson=importlib.util.module_from_spec(spec); spec.loader.exec_module(lesson)
# analyze() 加载本地数据、训练内缩放、固定拆分、重训所有候选、生成 B–D 表图。
a=lesson.analyze()
print(a['summary'])

# %% 原 Notebook 单元格 4
display(pd.DataFrame(a['audit'])[['feature','min','max','train_sd']])
points=a['toy']
rows=[]
for name,w,b in [('a',np.array([1.,0.]),0.),('b',np.array([1.,0.]),-.8),('5a',np.array([5.,0.]),0.)]:
    distance=np.abs(points@w+b)/np.linalg.norm(w)
    rows.append({'边界':name,'最小距离':distance.min(),'最近样本':[['N1','N2','N3','P1','P2','P3'][i] for i in np.flatnonzero(np.isclose(distance,distance.min()))]})
display(pd.DataFrame(rows))

# %% 原 Notebook 单元格 5
display(Image(filename=str(lesson.AS/'figure-s1.png')))

# %% 原 Notebook 单元格 7
# 表 B1 的三行 + 学习单 Q2③ 的核对（移除 P2）
display(pd.DataFrame(a['deletions']))

# %% 原 Notebook 单元格 9
display(a['sim'][a['sim'].kernel=='linear'][['C','errors','violations','sv','width','validation_correct']])
m=np.array([1.4,.4,-.2,-1.])
display(pd.DataFrame({'带标签得分':m,'合页损失':np.maximum(0,1-m),'判对':m>0,'间隔违例':m<1}))

# %% 原 Notebook 单元格 10
display(Image(filename=str(lesson.AS/'figure-b1.png')))

# %% 原 Notebook 单元格 12
display(a['sim'][(a['sim'].C==1)&(a['sim'].gamma==1)][['kernel','errors','sv','validation_correct']])
x=np.array([1.,2.]); z=np.array([2.,0.])
phi=lambda t: np.array([t[0]**2,np.sqrt(2)*t[0]*t[1],t[1]**2])
print('K=',(x@z)**2,' phi(x)·phi(z)=',phi(x)@phi(z))
print('RBF 相似度（距离1）：gamma=0.1 ->',np.exp(-.1),'；gamma=10 ->',np.exp(-10))

# %% 原 Notebook 单元格 13
display(Image(filename=str(lesson.AS/'figure-c1.png')))

# %% 原 Notebook 单元格 15
display(pd.DataFrame(a['best'])[['kernel','C','gamma','sv','train_ba','validation_ba','validation_tp','validation_tn']])
display(a['contrast'][['setting','width','sv','train_ba','validation_ba','validation_tn']])
display(a['grid'][a['grid'].kernel=='rbf'][['C','gamma','scaled','train_ba','validation_ba','sv']])

# %% 原 Notebook 单元格 16
display(Image(filename=str(lesson.AS/'figure-c2.png')))

# %% 原 Notebook 单元格 18
display(a['test'])
display(a['stress'])
pd.testing.assert_frame_equal(pd.read_csv(lesson.AS/'packet-d-test.csv'),a['test'])

# %% 原 Notebook 单元格 19
display(Image(filename=str(lesson.AS/'figure-d1.png')))

# %% 原 Notebook 单元格 22
import inspect
print(inspect.getsource(lesson.analyze))
