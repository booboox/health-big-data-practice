"""Verify ensemble variance using correlated synthetic prediction errors."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = Path(__file__).resolve().parent / '工作单配图'
rng = np.random.default_rng(1010)
N = 30000
B = np.array([1, 2, 5, 10, 20, 50, 100, 300])
rhos = [0, .2, .5, .9, 1]
colors = ['#28658A', '#55A3A0', '#8A85AC', '#C46B35', '#6C747B']
rows = []
# For each trial, errors = sqrt(rho)*shared + sqrt(1-rho)*independent.
# Each individual error has mean zero and variance one.
for rho in rhos:
    shared = rng.normal(size=N)
    independent_sum = np.zeros(N)
    prev = 0
    for size in B:
        independent_sum += rng.normal(size=(N, int(size - prev))).sum(axis=1)
        avg = np.sqrt(rho) * shared + np.sqrt(1-rho) * independent_sum / size
        rows.append({'rho': rho, 'n_models': int(size), 'theoretical_variance': rho + (1-rho)/size,
                     'simulated_variance': avg.var(ddof=1), 'mean_error': avg.mean(),
                     'trials': N})
        prev = size
df = pd.DataFrame(rows)
df.to_csv(OUT / 'Figure_B2_variance_simulation_data.csv', index=False)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False,
    'axes.spines.right': False, 'axes.linewidth': .7, 'pdf.fonttype': 42, 'ps.fonttype': 42})
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), gridspec_kw={'width_ratios': [1.3, 1]})
fig.subplots_adjust(left=.075, right=.97, bottom=.29, top=.73, wspace=.30)
fig.text(.04, .95, 'More models reduce variance; shared errors set the limit', fontsize=14, weight='bold', color='#25313B')
fig.text(.04, .86, r'Synthetic errors  |  Individual variance $\sigma^2=1$  |  30,000 trials per setting  |  Seed = 1010', fontsize=9, color='#596774')
fine = np.geomspace(1, 300, 400)
ax, bx = axes
for rho, color in zip(rhos, colors):
    subset = df[df.rho == rho]
    ax.plot(fine, rho+(1-rho)/fine, color=color, lw=1.5, label=fr'$\rho={rho:g}$')
    ax.scatter(subset.n_models, subset.simulated_variance, color=color, s=19, edgecolor='white', linewidth=.4, zorder=3)
ax.set(xscale='log', xlim=(.9, 650), ylim=(-.025, 1.06), xlabel='Number of models, B (log scale)', ylabel='Variance of mean error', xticks=[1, 10, 100, 300], xticklabels=['1','10','100','300'])
ax.set_title('Theory and simulation', loc='left', fontsize=11, weight='bold', pad=14)
ax.text(-.13, 1.10, 'a', transform=ax.transAxes, fontsize=13, weight='bold')
for rho, color in zip(rhos, colors):
    ax.text(340, rho + (1-rho)/300, fr'$\rho={rho:g}$', color=color, fontsize=8, va='center')
at100 = df[df.n_models == 100]
x = np.arange(len(rhos))
bx.bar(x, at100.theoretical_variance, color=colors, width=.58, alpha=.8)
bx.scatter(x, at100.simulated_variance, color='#25313B', s=20, zorder=4)
for i, v in enumerate(at100.theoretical_variance): bx.text(i, v+.035, f'{v:.3f}', ha='center', fontsize=8)
bx.set(xticks=x, xticklabels=[str(r) for r in rhos], xlabel=r'Error correlation, $\rho$', ylabel='Variance of mean error', ylim=(0, 1.14))
bx.set_title('At 100 models', loc='left', fontsize=11, weight='bold', pad=14)
bx.text(-.16, 1.10, 'b', transform=bx.transAxes, fontsize=13, weight='bold')
for a in axes:
    a.grid(axis='y', color='#E2E7EB', linewidth=.6)
    a.set_axisbelow(True)
fig.legend(handles=[Line2D([], [], color='#596774', lw=1.5, label='Theory'),
                    Line2D([], [], color='#25313B', marker='o', ls='', markersize=4, label='Monte Carlo simulation')],
           loc='lower left', bbox_to_anchor=(.065, .12), ncol=2, frameon=False)
fig.text(.04, .078, r'Theory: $\mathrm{Var}(\bar{e})=\rho+(1-\rho)/B$; as $B\to\infty$, the variance approaches $\rho$.', fontsize=9)
fig.text(.04, .025, 'Constructed equal-variance, equicorrelated errors; this is a mechanism simulation, not a WDBC performance comparison.', fontsize=8, color='#596774')
for ext in ['png', 'pdf', 'svg']:
    fig.savefig(OUT / f'Figure_B2_ensemble_variance.{ext}', dpi=600, bbox_inches='tight', pad_inches=.13)
plt.close(fig)
print(at100[['rho','theoretical_variance','simulated_variance']].to_string(index=False))
