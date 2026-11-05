"""修正后的基线：真持续性 Q(t) -> Q(t+h)，以及月份气候均值"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'results/phase2_v5'
OUT.mkdir(parents=True, exist_ok=True)

TEST_START = pd.Timestamp('2004-01-01')


def metrics(y, p):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return {'RMSE（cfs）': float(np.sqrt(np.mean((y - p) ** 2))),
            'MAE（cfs）': float(np.mean(np.abs(y - p))),
            'R²': 1 - ss_res / ss_tot,
            'NSE': 1 - ss_res / ss_tot,
            '样本数': int(len(y))}


def main():
    df = pd.read_csv(ROOT / '01047000_fixed.csv', parse_dates=['Date']).set_index('Date')
    q = df['Discharge']
    train_q = q[q.index < TEST_START]
    rows = []

    for h in range(1, 8):
        # 真持续性：用起报日当天的实测径流
        issue = q[(q.index >= TEST_START) & (q.index + pd.Timedelta(days=h) <= q.index.max())]
        y = q.reindex(issue.index + pd.Timedelta(days=h)).to_numpy(float)
        p = issue.to_numpy(float)

        # 月份气候均值：按目标月份取训练期均值
        tgt_month = (issue.index + pd.Timedelta(days=h)).month
        clim = train_q.groupby(train_q.index.month).mean()
        p2 = np.array([clim[m] for m in tgt_month], dtype=float)

        for name, pred in [('真持续性', p), ('月份气候均值', p2)]:
            m = metrics(y, pred)
            rows.append({'基线': name, '预见期（天）': h, **m})

    out = pd.DataFrame(rows)
    out.to_csv(OUT / '基线指标.csv', index=False, encoding='utf-8-sig')

    print('=' * 74)
    print('修正后的基线（真持续性使用 Q(t)）')
    print('=' * 74)
    print(out.pivot_table(index='预见期（天）', columns='基线',
                          values='NSE').round(4).to_string())
    print()
    print(out.pivot_table(index='预见期（天）', columns='基线',
                          values='RMSE（cfs）').round(1).to_string())
    print()
    print(f'已保存: {OUT / "基线指标.csv"}')


if __name__ == '__main__':
    main()
