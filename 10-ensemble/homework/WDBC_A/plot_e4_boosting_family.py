"""E4: source-based conceptual comparison; no performance benchmark implied."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT=Path(__file__).resolve().parent/'工作单配图'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'ps.fonttype':42})
BLUE,ORANGE,TEAL,INK,GRAY='#28658A','#C46B35','#459C93','#25313B','#596774'
fig,ax=plt.subplots(figsize=(11,6.3))
fig.subplots_adjust(left=.025,right=.975,bottom=.20,top=.83)
ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
fig.text(.04,.955,'One boosting principle, different implementations',fontsize=15,weight='bold',color=INK)
fig.text(.04,.89,'Conceptual comparison of tree boosters  |  Literature-based summary, not a WDBC benchmark',fontsize=9,color=GRAY)
def box(x,y,w,h,color,fill):
    ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle='round,pad=0.008,rounding_size=0.018',fc=fill,ec=color,lw=1))
box(.5,.88,.88,.20,BLUE,'#EDF4F7')
ax.text(.5,.92,'Gradient-boosted decision trees (GBDT)',ha='center',fontsize=12,weight='bold',color=BLUE)
ax.text(.5,.84,r'$F_t(x)=F_{t-1}(x)+\eta f_t(x)$',ha='center',fontsize=14,color=INK)
cards=[(.17,BLUE,'scikit-learn [1]','GradientBoostingClassifier',[
    'Fits trees to negative gradients',
    'Controls: learning rate, stages,',
    'tree depth and subsampling',
    '',
    'HistGradientBoosting is a separate',
    'histogram-based implementation',
    'with native missing-value support']),
(.5,ORANGE,'XGBoost [2]','Regularized tree boosting',[
    'Uses gradient and Hessian terms',
    'Penalizes tree / leaf complexity',
    'Optimizes split search and',
    'sparse-data processing',
    '',
    'Efficient tree construction does',
    'not remove stage dependence']),
(.83,TEAL,'LightGBM [3, 4]','Efficient tree boosting',[
    'Histogram-based split search',
    'Leaf-wise (best-first) tree growth',
    'GOSS: gradient-based sampling',
    'EFB: bundles exclusive features',
    '',
    'GOSS is optional, not the default',
    'sampling strategy'])]
for x,color,title,subtitle,lines in cards:
    ax.annotate('',xy=(x,.66),xytext=(.5,.77),arrowprops={'arrowstyle':'->','color':'#A3B2BC','lw':1})
    box(x,.34,.305,.62,color,'white')
    ax.text(x,.59,title,ha='center',fontsize=12,weight='bold',color=color)
    ax.text(x,.52,subtitle,ha='center',fontsize=9,color=GRAY)
    for j,line in enumerate(lines):
        ax.text(x-.133,.43-j*.052,line,ha='left',fontsize=8.4,color=INK)
fig.text(.04,.165,'Boosting stages depend on previous stages. Split search or work inside a stage may still be parallelized.',fontsize=9,weight='bold',color=INK)
fig.text(.04,.11,'[1] sklearn GBDT / histogram GBDT docs    [2] XGBoost model tutorial    [3] LightGBM features / parameters',fontsize=8,color=GRAY)
fig.text(.04,.065,'[4] Ke et al., NeurIPS 2017. Full links and worksheet answer are provided in the accompanying Chinese caption file.',fontsize=8,color=GRAY)
fig.text(.04,.025,'Features are representative, not exclusive to one library. Accuracy and speed depend on data, settings and hardware.',fontsize=8,color=GRAY)
for ext in ['png','pdf','svg']:fig.savefig(OUT/f'Figure_E4_boosting_family.{ext}',dpi=600,bbox_inches='tight',pad_inches=.13)
plt.close(fig)
print('Saved Figure_E4_boosting_family (PNG, PDF, SVG).')
