"""Worksheet C1-C2: explicit sampling example and Monte Carlo simulation."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parent / '工作单配图'
BLUE, ORANGE, GRAY = '#28658A', '#C46B35', '#596774'
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':9,
    'axes.spines.top':False, 'axes.spines.right':False, 'pdf.fonttype':42, 'ps.fonttype':42})
rng = np.random.default_rng(1010)
records = []
for n in [5, 341]:
    for i in range(1000):
        draw = rng.integers(0, n, size=n)
        unique = np.unique(draw).size / n
        records.append({'n':n, 'replicate':i+1, 'unique_fraction':unique, 'oob_fraction':1-unique})
df = pd.DataFrame(records)
df.to_csv(OUT / 'Figure_C12_bootstrap_simulation_data.csv', index=False)
summary = df.groupby('n').agg(unique_mean=('unique_fraction','mean'), unique_sd=('unique_fraction','std'), oob_mean=('oob_fraction','mean'))
summary['theoretical_unique'] = [1-(1-1/n)**n for n in summary.index]
summary.to_csv(OUT / 'Figure_C12_bootstrap_summary.csv')

fig = plt.figure(figsize=(10.5, 5.4), facecolor='white')
fig.text(.04,.955,'Bootstrap sampling creates duplicates and omissions',fontsize=15,weight='bold',color='#25313B')
fig.text(.04,.89,'Sampling with replacement  |  Sample size is preserved; the number of distinct records varies',fontsize=9,color=GRAY)
gs=fig.add_gridspec(1,2,left=.06,right=.97,bottom=.27,top=.76,width_ratios=[1.2,1],wspace=.30)
ax=fig.add_subplot(gs[0,0]); bx=fig.add_subplot(gs[0,1])
ax.set(xlim=(-.85,4.85),ylim=(-1.05,3.6)); ax.axis('off')
ax.set_title('One resample from five records',loc='left',fontsize=11,weight='bold',pad=18)
ax.text(-.05,1.12,'a',transform=ax.transAxes,fontsize=13,weight='bold')
original=[1,2,3,4,5]; draw=[2,2,5,1,2]
def box(x,y,label,color,fill):
    ax.add_patch(FancyBboxPatch((x-.29,y-.25),.58,.50,boxstyle='round,pad=0.02,rounding_size=0.06',facecolor=fill,edgecolor=color,lw=1))
    ax.text(x,y,str(label),ha='center',va='center',color=color,fontsize=12,weight='bold')
ax.text(-.65,3.25,'Original records',fontsize=9,color=GRAY)
for x,k in enumerate(original): box(x,2.7,k,ORANGE if k in [3,4] else BLUE,'#FCF1E9' if k in [3,4] else '#EDF4F7')
for x,k in enumerate(draw):
    ax.annotate('',xy=(x,1.05),xytext=(k-1,2.4),arrowprops={'arrowstyle':'->','color':'#A8B9C4','lw':.8},zorder=0)
    box(x,.75,k,BLUE,'#EDF4F7')
ax.text(-.65,.12,'Resample: [2, 2, 5, 1, 2]',fontsize=10,weight='bold',color=BLUE)
ax.text(-.65,-.36,'Distinct records: {1, 2, 5}  (3/5 = 60%)',fontsize=9)
ax.text(-.65,-.83,'Out-of-bag records: {3, 4}  (2/5 = 40%)',fontsize=9,color=ORANGE)

bins=np.arange(-.01,1.031,.04)
for n,color in [(5,ORANGE),(341,BLUE)]:
    values=df.loc[df.n==n,'unique_fraction']
    bx.hist(values,bins=bins,weights=np.ones(len(values))/len(values),color=color,alpha=.5,label=f'n = {n}',edgecolor='white',lw=.4)
    theory=1-(1-1/n)**n
    bx.axvline(theory,color=color,lw=1.3,ls='--')
bx.set(xlim=(.15,1.035),xlabel='Distinct-record fraction',ylabel='Fraction of 1,000 resamples')
bx.set_title('Repeated bootstrap sampling',loc='left',fontsize=11,weight='bold',pad=18)
bx.text(-.16,1.12,'b',transform=bx.transAxes,fontsize=13,weight='bold')
bx.grid(axis='y',color='#E2E7EB',lw=.6); bx.set_axisbelow(True)
bx.legend(frameon=False,loc='upper left',fontsize=8)
fig.text(.04,.175,r'Exact expectation: $1-(1-1/n)^n$. Dashed lines mark finite-sample expectations.',fontsize=9)
fig.text(.04,.105,'n = 5: expected 67.23% distinct / 32.77% OOB.   n = 341: expected 63.27% distinct / 36.73% OOB.',fontsize=9)
fig.text(.04,.045,'Illustrative index simulation; 1,000 resamples per sample size; seed = 1010. OOB fraction = 1 minus distinct-record fraction.',fontsize=8,color=GRAY)
for ext in ['png','pdf','svg']:
    fig.savefig(OUT/f'Figure_C12_bootstrap_sampling.{ext}',dpi=600,bbox_inches='tight',pad_inches=.13)
plt.close(fig)
print(summary.to_string())
