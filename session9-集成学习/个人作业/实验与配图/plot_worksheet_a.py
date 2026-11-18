"""Generate reproducible worksheet figures; no test labels are loaded."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, MaxNLocator
from sklearn.tree import DecisionTreeClassifier

OUT = Path(__file__).resolve().parent
FIGURES = OUT / '工作单配图'
FIGURES.mkdir(exist_ok=True)
DATA = (Path(__file__).resolve().parents[2] / '10. Ensemble' / 'data' / 'student')
BLUE, ORANGE, INK = '#28658A', '#C46B35', '#25313B'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
    'axes.labelsize': 10, 'axes.titlesize': 11, 'axes.titleweight': 'bold',
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': '#73808A', 'axes.linewidth': .7,
    'text.color': INK, 'axes.labelcolor': INK, 'xtick.color': INK,
    'ytick.color': INK, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'savefig.facecolor': 'white'})

def save(fig, name):
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(FIGURES / f'{name}.{ext}', dpi=600, bbox_inches='tight', pad_inches=.13)
    plt.close(fig)

def depth_figure():
    df = pd.read_csv(OUT / 'tree_depth_results.csv')
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.65))
    fig.subplots_adjust(left=.07, right=.985, bottom=.25, top=.77, wspace=.38)
    fig.suptitle('Tree complexity and validation performance', x=.07, y=.98, ha='left', fontsize=14, weight='bold')
    fig.text(.07, .87, 'WDBC  |  Training n = 341; validation n = 114  |  Fixed split and random seed (1010)', fontsize=9, color='#596774')
    for ax, metric, title, letter in zip(axes, ['accuracy', 'balanced_accuracy', 'roc_auc'],
                                        ['Accuracy', 'Balanced accuracy', 'ROC-AUC'], 'abc'):
        ax.axvspan(1.86, 2.14, color=BLUE, alpha=.07, lw=0)
        for split, color, marker, label in [('train', BLUE, 'o', 'Training'), ('validation', ORANGE, 's', 'Validation')]:
            ax.plot(range(5), df[f'{split}_{metric}'], marker=marker, color=color, lw=1.6, ms=4.5, label=label)
        ax.set(xticks=range(5), xticklabels=['1', '2', '3', '4', 'None'], xlabel='Maximum tree depth', ylim=(.895, 1.009), yticks=[.90, .925, .95, .975, 1.0])
        ax.set_title(title, loc='left', pad=13)
        ax.text(-.19, 1.12, letter, transform=ax.transAxes, weight='bold', fontsize=13)
        ax.grid(axis='y', color='#DCE2E7', lw=.6)
        ax.set_axisbelow(True)
        if metric != 'roc_auc': ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=1))
        best = float(df.loc[2, f'validation_{metric}'])
        label = f'{best:.1%}' if metric != 'roc_auc' else f'{best:.4f}'
        ax.annotate(label, xy=(2, best), xytext=(0, -19), textcoords='offset points', ha='center', fontsize=8, color=ORANGE)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower left', bbox_to_anchor=(.065, .045), ncol=2, frameon=False)
    fig.text(.07, .015, 'Shading marks depth 3. None = unrestricted (observed depth 7). Lines connect tested settings; no uncertainty intervals estimated.', fontsize=8, color='#596774')
    save(fig, 'Figure_A3_tree_depth')

def bootstrap_figure():
    df = pd.read_csv(DATA / 'train.csv')
    features = [c for c in df if c not in ('sample_id', 'malignant')]
    rng = np.random.default_rng(1010)
    rows = []
    for i in range(100):
        idx = rng.choice(len(df), len(df), replace=True)
        tree = DecisionTreeClassifier(max_depth=2, random_state=1010).fit(df.iloc[idx][features], df.iloc[idx]['malignant'])
        rows.append({'replicate': i + 1, 'root_feature': features[tree.tree_.feature[0]],
                     'root_threshold': tree.tree_.threshold[0], 'unique_samples': len(np.unique(idx))})
    results = pd.DataFrame(rows)
    results.to_csv(OUT / 'bootstrap_root_results.csv', index=False, encoding='utf-8-sig')
    summary = results.groupby('root_feature').agg(count=('replicate', 'size'),
        threshold_min=('root_threshold', 'min'), threshold_median=('root_threshold', 'median'),
        threshold_max=('root_threshold', 'max')).sort_values('count', ascending=False)
    summary.to_csv(OUT / 'bootstrap_root_summary.csv', encoding='utf-8-sig')
    n = len(summary)
    fig = plt.figure(figsize=(10.5, max(4.2, 2 + n * .46)))
    grid = fig.add_gridspec(1, 2, left=.23, right=.97, top=.75, bottom=.24, width_ratios=[1, 1.25], wspace=.2)
    ax = fig.add_subplot(grid[0, 0])
    y = np.arange(n)
    ax.barh(y, summary['count'], color=BLUE, height=.55)
    ax.set(yticks=y, yticklabels=summary.index, xlabel='Bootstrap replicates (%)', xlim=(0, max(summary['count']) * 1.2))
    ax.invert_yaxis()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(axis='x', color='#DCE2E7', lw=.6)
    ax.set_axisbelow(True)
    for yi, count in zip(y, summary['count']): ax.text(count + .8, yi, str(count), va='center', fontsize=9)
    ax.set_title('Root feature selection', loc='left', pad=22)
    ax.text(-.65, 1.16, 'a', transform=ax.transAxes, fontsize=13, weight='bold')
    bx = fig.add_subplot(grid[0, 1])
    bx.set_xlim(0, 1)
    bx.set_ylim(ax.get_ylim())
    bx.axis('off')
    bx.set_title('Root threshold variation', loc='left', pad=22)
    bx.text(-.06, 1.16, 'b', transform=bx.transAxes, fontsize=13, weight='bold')
    for xpos, label in zip([.08, .42, .77], ['Minimum', 'Median', 'Maximum']):
        bx.text(xpos, 1.015, label, transform=bx.transAxes, fontsize=8, color='#596774', ha='center')
    for yi, (_, row) in zip(y, summary.iterrows()):
        for xpos, key in zip([.08, .42, .77], ['threshold_min', 'threshold_median', 'threshold_max']):
            bx.text(xpos, yi, f'{row[key]:.4g}', va='center', ha='center', fontsize=9)
    fig.suptitle('Training resampling changes the root split', x=.04, y=.97, ha='left', fontsize=14, weight='bold')
    fig.text(.04, .85, 'WDBC  |  100 bootstrap samples of 341 training rows  |  Trees restricted to depth 2', fontsize=9, color='#596774')
    fig.text(.04, .10, 'Each resample draws 341 rows with replacement. Model seed is fixed (1010); only training rows change.', fontsize=8, color='#596774')
    fig.text(.04, .055, 'Counts equal percentages (100 replicates). Thresholds retain each feature’s native units and are not comparable across features.', fontsize=8, color='#596774')
    save(fig, 'Figure_A4_bootstrap_roots')
    print(summary.to_string())

if __name__ == '__main__':
    depth_figure()
    bootstrap_figure()
