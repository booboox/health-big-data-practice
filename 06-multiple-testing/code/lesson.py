"""课程实验代码，按单元格顺序提取自 notebooks/multiple-test-lab.ipynb。
来源：用户提供的原始课程资料包；并非新增完成的个人作业。
建议优先在原 Notebook 中逐格运行；本脚本仅做语法核对，未完整重跑实验。
"""
from pathlib import Path as _CoursePath
import os as _course_os
import sys as _course_sys
_course_root = _CoursePath(__file__).resolve().parents[1]
_course_os.chdir(_course_root / 'notebooks')
_course_sys.path.insert(0, str(_course_root / "scripts"))

# %% 原 Notebook 单元格 2
from pathlib import Path
import os, json, platform, hashlib
import numpy as np
import pandas as pd
import scipy
from scipy import stats
import matplotlib.pyplot as plt
from IPython.display import display

# 从项目内部任意工作目录启动，均定位到同一课程和根目录 .tmp。
REPO = next((p for p in [Path.cwd(), *Path.cwd().parents]
             if (p / 'class2026/materials/6. multiple-test/data/leukemia.csv').exists()), None)
if REPO is None:
    raise FileNotFoundError('请从本项目目录内启动 Notebook 内核')
ROOT = REPO / 'class2026/materials/6. multiple-test'
RUN_ID = os.environ.get('MULTIPLE_TEST_RUN_ID', 'session-01')
if not RUN_ID or Path(RUN_ID).name != RUN_ID or RUN_ID in {'.', '..'} or '/' in RUN_ID or '\\' in RUN_ID:
    raise ValueError('RUN_ID 必须是一个文件夹名')
OUT = REPO / '.tmp/multiple-test/student-results' / RUN_ID
OUT.mkdir(parents=True, exist_ok=True)
versions = {'python': platform.python_version(), 'numpy': np.__version__,
            'pandas': pd.__version__, 'scipy': scipy.__version__}
(OUT / 'versions.json').write_text(json.dumps(versions, indent=2), encoding='utf-8')
print('版本:', versions)
print('结果目录:', OUT)

# %% 原 Notebook 单元格 4
"""Transparent numerical support for the multiple-testing worksheet."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

META = ["sample_id", "original_split", "diagnosis"]
METHODS = ["raw", "bonferroni", "bh"]


def load_data():
    df = pd.read_csv(ROOT / "data/leukemia.csv")
    x = df.drop(columns=META)
    if df.shape != (72, 7132) or not np.isfinite(x.to_numpy()).all():
        raise ValueError("Unexpected shape or nonfinite expression values")
    if not df.sample_id.is_unique or not x.columns.is_unique:
        raise ValueError("Duplicate identifiers")
    return df


def adjust(p):
    """All input p-values constitute ONE family; do not prefilter on p."""
    p = np.asarray(p, dtype=float)
    if p.ndim != 1 or not len(p) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Expected a nonempty, finite p-value vector in [0, 1]")
    m = len(p)
    order = np.argsort(p, kind="stable")
    ranks = np.arange(1, m + 1)
    sorted_adjusted = np.minimum.accumulate((p[order] * m / ranks)[::-1])[::-1]
    bh = np.empty(m)
    bh[order] = np.minimum(sorted_adjusted, 1)
    return pd.DataFrame({"raw": p, "bonferroni": np.minimum(m * p, 1), "bh": bh})


def welch_table(df, probes=None):
    """Two-sided Welch tests, AML minus ALL; pointwise 95% CI."""
    if probes is None:
        probes = list(df.drop(columns=META).columns)
    if not len(probes):
        raise ValueError("No probes to test")
    a = df.loc[df.diagnosis.eq("AML"), probes].to_numpy(float)
    b = df.loc[df.diagnosis.eq("ALL"), probes].to_numpy(float)
    if min(len(a), len(b)) < 2:
        raise ValueError("Each group needs at least two observations")
    test = stats.ttest_ind(a, b, axis=0, equal_var=False)
    ci = test.confidence_interval(confidence_level=0.95)
    delta = a.mean(axis=0) - b.mean(axis=0)
    pooled = np.sqrt(((len(a)-1)*a.var(axis=0, ddof=1) + (len(b)-1)*b.var(axis=0, ddof=1)) / (len(a)+len(b)-2))
    if (pooled == 0).any():
        raise ValueError("Zero pooled SD: investigate instead of silently dropping a test")
    result = pd.DataFrame({"probe": probes, "n_AML": len(a), "n_ALL": len(b),
                           "mean_AML": a.mean(axis=0), "mean_ALL": b.mean(axis=0),
                           "delta_AML_minus_ALL": delta, "ci95_low": ci.low,
                           "ci95_high": ci.high, "cohen_d": delta/pooled})
    return pd.concat([result, adjust(test.pvalue)], axis=1)


def discovery():
    df = load_data()
    return welch_table(df.loc[df.original_split.eq("train")])


def counts(table, alpha=0.05):
    return {method: int((table[method] <= alpha).sum()) for method in METHODS}


def freeze_candidates(table, path, limit=10):
    chosen = table.loc[table.bh <= .05].sort_values(["raw", "probe"]).head(limit)
    locked = chosen[["probe", "delta_AML_minus_ALL"]].rename(columns={"delta_AML_minus_ALL": "discovery_delta"})
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents accidental replacement after viewing replication.
    with path.open("x", encoding="utf-8", newline="") as f:
        locked.to_csv(f, index=False)
    return locked


def replicate(candidate_path):
    locked = pd.read_csv(candidate_path)
    if locked.empty:
        return pd.DataFrame(columns=["probe", "replicated"])
    if not locked.probe.is_unique or len(locked) > 10:
        raise ValueError("Expected at most ten unique, previously fixed candidates")
    df = load_data()
    out = welch_table(df.loc[df.original_split.eq("test")], list(locked.probe))
    out = out.merge(locked, on="probe", validate="one_to_one")
    out["same_direction"] = np.sign(out.delta_AML_minus_ALL) == np.sign(out.discovery_delta)
    out["replicated"] = (out.bonferroni <= .05) & out.same_direction
    return out


def simulate(repeats=1000, seed=42, delta=0.0, signals=0, m=1000, n_all=27, n_aml=11):
    """Independent Gaussian teaching data, NOT measured Golub expression.

    Reusing the seed across delta settings yields paired noise realizations.
    All-null power is undefined (NaN), never reported as zero.
    """
    if repeats < 2 or not 0 <= signals <= m or (signals > 0 and delta == 0):
        raise ValueError("Invalid simulation settings")
    rng = np.random.default_rng(seed)
    truth = np.arange(m) < signals
    rows = []
    for run in range(repeats):
        a = rng.normal(size=(n_aml, m))
        b = rng.normal(size=(n_all, m))
        a[:, truth] += delta
        # Simulation guarantees normal, equal-variance populations: pooled t
        # has an exact null distribution, including the extreme tails tested
        # by Bonferroni. Real expression data instead use approximate Welch.
        p = stats.ttest_ind(a, b, axis=0, equal_var=True).pvalue
        adjusted = adjust(p)
        for method in METHODS:
            rejected = adjusted[method].to_numpy() <= .05
            r = int(rejected.sum())
            v = int((rejected & ~truth).sum())
            s = int((rejected & truth).sum())
            rows.append({"run": run, "method": method, "R": r, "V": v,
                         "S": s, "FDP": v/max(r, 1), "any_false": int(v > 0),
                         "power": s/signals if signals else np.nan})
    return pd.DataFrame(rows)


def summarize_simulation(runs):
    rows = []
    for method, d in runs.groupby("method", sort=False):
        rows.append({"method": method, "repeats": len(d), "mean_R": d.R.mean(),
                     "mean_V": d.V.mean(), "FWER_estimate": d.any_false.mean(),
                     "FWER_MCSE": d.any_false.std(ddof=1)/np.sqrt(len(d)),
                     "FDR_estimate": d.FDP.mean(),
                     "FDR_MCSE": d.FDP.std(ddof=1)/np.sqrt(len(d)),
                     "mean_power": d.power.mean()})
    return pd.DataFrame(rows)

# %% 原 Notebook 单元格 6
null_runs = simulate(repeats=1000, seed=42, delta=0, signals=0)
null_runs.to_csv(OUT / 'simulation-all-null.csv', index=False)
print('第一次运行（run=0）：')
display(null_runs.loc[null_runs.run.eq(0)])

# %% 原 Notebook 单元格 7
def variation(runs):
    return runs.groupby('method', sort=False).R.agg(
        mean_R='mean', sd_R='std',
        p05_R=lambda x: x.quantile(.05), p95_R=lambda x: x.quantile(.95))
null_summary = summarize_simulation(null_runs)
display(variation(null_runs))
display(null_summary)
fig, axes = plt.subplots(1, 3, figsize=(11, 3), layout='constrained')
for ax, method in zip(axes, METHODS):
    vals = null_runs.loc[null_runs.method.eq(method), 'R']
    ax.hist(vals, bins=np.arange(vals.min()-.5, vals.max()+1.5), color='#287f8e')
    ax.set(title=method, xlabel='Discoveries per repetition', ylabel='Repetitions')
fig.savefig(OUT / 'all-null-variation.png', dpi=150)
plt.show()

# %% 原 Notebook 单元格 9
signal_summaries = []
for label, delta in [('signal_1.0', 1.0), ('signal_0.5', .5)]:
    runs = simulate(repeats=1000, seed=42, delta=delta, signals=100)
    runs.to_csv(OUT / f'simulation-{label}.csv', index=False)
    summary = summarize_simulation(runs)
    summary.insert(0, 'scenario', label)
    signal_summaries.append(summary)
signal_summary = pd.concat(signal_summaries, ignore_index=True)
signal_summary.to_csv(OUT / 'signal-summary.csv', index=False)
display(signal_summary)

# %% 原 Notebook 单元格 11
data = load_data()
expected_hash = '915bce00295ac470f6bd970a4032730a170a080642a5da53ab0bcb0545156f19'
actual_hash = hashlib.sha256((ROOT / 'data/leukemia.csv').read_bytes()).hexdigest()
assert actual_hash == expected_hash, '数据文件与课程快照不同'
print('CSV shape:', data.shape)
display(pd.crosstab(data.original_split, data.diagnosis))
print('探针数:', len(data.columns.difference(META)))

# %% 原 Notebook 单元格 13
table = discovery()
np.testing.assert_allclose(table.bh, stats.false_discovery_control(table.raw, method='bh'))
table.to_csv(OUT / 'discovery.csv', index=False)
print('发现数:', counts(table))
print('家族大小:', len(table), 'Bonferroni raw p 阈值:', .05/len(table))
fig, ax = plt.subplots(figsize=(6, 3), layout='constrained')
ax.hist(table.raw, bins=np.linspace(0, 1, 31), color='#287f8e')
ax.set(xlabel='Raw p-value', ylabel='Probe count', title='Discovery family: all probes')
fig.savefig(OUT / 'discovery-p-values.png', dpi=150)
plt.show()

# %% 原 Notebook 单元格 15
top20 = table.sort_values(['raw', 'probe']).head(20).copy()
wrong = adjust(top20.raw)
top20['WRONG_bh_after_selecting_top20'] = wrong.bh.to_numpy()
top20['WRONG_bonf_after_selecting_top20'] = wrong.bonferroni.to_numpy()
top20.to_csv(OUT / 'selection-bias.csv', index=False)
display(top20[['probe', 'raw', 'bonferroni', 'bh',
               'WRONG_bh_after_selecting_top20', 'WRONG_bonf_after_selecting_top20']])

# %% 原 Notebook 单元格 17
chosen = table.loc[table.bh.le(.05)].sort_values(['raw', 'probe']).head(10)
display(chosen[['probe', 'n_AML', 'n_ALL', 'delta_AML_minus_ALL',
                'ci95_low', 'ci95_high', 'cohen_d', 'raw', 'bonferroni', 'bh']])
locked_path = OUT / 'locked-candidates.csv'
if locked_path.exists():
    locked = pd.read_csv(locked_path)
    if locked.probe.tolist() != chosen.probe.tolist():
        raise ValueError('已有锁定名单不同；不得按验证结果覆盖名单')
    np.testing.assert_allclose(locked.discovery_delta, chosen.delta_AML_minus_ALL)
    print('保留原锁定名单:', locked_path)
else:
    locked = freeze_candidates(table, locked_path)
    print('已锁定名单:', locked_path)
display(locked)

# %% 原 Notebook 单元格 19
rep = replicate(locked_path)
rep.to_csv(OUT / 'replication.csv', index=False)
if rep.empty:
    print('无候选，未进行验证检验。')
else:
    display(rep[['probe', 'n_AML', 'n_ALL', 'discovery_delta',
                 'delta_AML_minus_ALL', 'ci95_low', 'ci95_high', 'raw',
                 'bonferroni', 'same_direction', 'replicated']])
    print('锁定候选数:', len(rep), '满足预定验证判据:', int(rep.replicated.sum()))
