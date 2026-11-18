"""Worksheet B: visualize the supplied illustrative voting example."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

OUT = Path(__file__).resolve().parent / '工作单配图'
OUT.mkdir(exist_ok=True)
BLUE, ORANGE, INK = '#28658A', '#C46B35', '#25313B'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': '#73808A', 'axes.linewidth': .7,
    'text.color': INK, 'axes.labelcolor': INK, 'xtick.color': INK,
    'ytick.color': INK, 'pdf.fonttype': 42, 'ps.fonttype': 42})

names = ['A', 'B', 'C', 'D']
truth = np.array([1, 0, 1, 0])
probs = np.array([[.80, .70, .45], [.20, .60, .65], [.35, .75, .80], [.15, .25, .70]])
classes = (probs >= .5).astype(int)
hard = (classes.sum(axis=1) >= 2).astype(int)
mean = probs.mean(axis=1)
soft = (mean >= .5).astype(int)
pd.DataFrame({'sample': names, 'truth': truth,
    'tree_1_probability': probs[:, 0], 'tree_2_probability': probs[:, 1],
    'tree_3_probability': probs[:, 2], 'hard_vote': hard,
    'mean_probability': mean, 'soft_vote': soft}).to_csv(OUT / 'Figure_B1_voting_data.csv', index=False)

fig = plt.figure(figsize=(10, 4.8), facecolor='white')
gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1], left=.09, right=.97, bottom=.27, top=.75, wspace=.23)
ax, bx = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
fig.text(.045, .96, 'Hard and soft voting can disagree', fontsize=15, weight='bold')
fig.text(.045, .885, 'Worksheet example  |  Four illustrative samples and three classifiers  |  Positive-class threshold = 0.5', fontsize=9, color='#596774')

for a in [ax, bx]:
    a.axhspan(.58, 1.42, color=ORANGE, alpha=.07, lw=0)
    a.set_ylim(3.55, -.55)
ax.axvline(.5, color='#7A858E', ls=(0, (3, 3)), lw=1)
colors = ['#85AFC5', '#4D8AAD', '#1C4D69']
markers = ['o', '^', 's']
for j in range(3):
    ax.scatter(probs[:, j], np.arange(4) + (j - 1) * .15, color=colors[j], marker=markers[j], s=35, zorder=3)
for i, p in enumerate(mean):
    ax.scatter(p, i, marker='D', s=54, color=ORANGE, edgecolor='white', lw=.8, zorder=5)
    ax.text(p, i + .37, f'{p:.4f}', ha='center', fontsize=8, color=ORANGE)
ax.set(xlim=(0, 1), xticks=np.arange(0, 1.01, .2),
    yticks=range(4), yticklabels=[f'{s}   (y = {t})' for s, t in zip(names, truth)],
    xlabel='Predicted probability of class 1')
ax.set_title('Individual probabilities and their mean', loc='left', fontsize=11, weight='bold', pad=18)
ax.text(-.13, 1.10, 'a', transform=ax.transAxes, fontsize=13, weight='bold')
ax.grid(axis='x', color='#E2E7EB', lw=.6)
ax.set_axisbelow(True)

bx.set_xlim(-.65, 2.65)
bx.axis('off')
bx.set_title('Class predictions', loc='left', fontsize=11, weight='bold', pad=18)
bx.text(-.06, 1.10, 'b', transform=bx.transAxes, fontsize=13, weight='bold')
for j, title in enumerate(['True label', 'Hard vote', 'Soft vote']):
    bx.text(j, -.67, title, ha='center', va='bottom', fontsize=9)
for i in range(4):
    for j, val in enumerate([truth[i], hard[i], soft[i]]):
        wrong = j > 0 and val != truth[i]
        color = ORANGE if wrong else BLUE
        bx.add_patch(Rectangle((j - .30, i - .25), .60, .50,
            facecolor=color, alpha=.13 if wrong else .07, edgecolor='none'))
        bx.text(j, i, str(val), ha='center', va='center', fontsize=12,
            weight='bold' if j else 'normal', color=color)
    if i == 1:
        bx.text(1, i + .39, 'incorrect', ha='center', fontsize=7.5, color=ORANGE)
        bx.text(2, i + .39, 'correct', ha='center', fontsize=7.5, color=BLUE)

handles = [Line2D([], [], marker=markers[j], color=colors[j], ls='', label=f'Tree {j+1}', markersize=5) for j in range(3)]
handles += [Line2D([], [], marker='D', color=ORANGE, ls='', label='Mean probability', markersize=5)]
fig.legend(handles=handles, loc='lower left', bbox_to_anchor=(.08, .13), ncol=4, frameon=False,
    handletextpad=.35, columnspacing=1.4)
fig.text(.045, .095, 'Sample B: two classifiers vote for class 1, but the mean probability is 0.4833, yielding class 0.', fontsize=9, weight='bold')
fig.text(.045, .04, 'Hard voting uses majority class labels. Soft voting averages probabilities. This illustrative example does not establish superiority.', fontsize=8, color='#596774')
for ext in ['png', 'pdf', 'svg']:
    fig.savefig(OUT / f'Figure_B1_hard_soft_voting.{ext}', dpi=600, bbox_inches='tight', pad_inches=.13)
plt.close(fig)
print('Saved Figure_B1_hard_soft_voting (PNG, PDF, SVG) and source data.')
