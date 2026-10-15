"""Baseline vs. existing-model comparison (all recomputed from artifacts)."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase2_v3_data as d3
import phase2_v3_baselines as b3

ROOT = Path(__file__).resolve().parent.parent


def build_comparison():
    train, test, cols, meta = d3.build_dataset()
    base, _ = b3.evaluate_baselines(train, test)
    old = pd.read_csv(ROOT / 'results/phase2_v2/预测精度指标.csv')
    rows = []
    for name in ['持续性', '月份气候均值']:
        s = base[base['基线'] == name]
        for _, r in s.iterrows():
            rows.append({'模型': name, '策略': '基线', '预见期（天）': r['预见期（天）'],
                         'RMSE（cfs）': r['RMSE（cfs）'], 'MAE（cfs）': r['MAE（cfs）'],
                         'NSE': r['NSE']})
    for _, r in old.iterrows():
        rows.append({'模型': r['模型'], '策略': r['策略'],
                     '预见期（天）': r['预见期（天）'], 'RMSE（cfs）': r['RMSE（cfs）'],
                     'MAE（cfs）': r['MAE（cfs）'], 'NSE': r['NSE']})
    frame = pd.DataFrame(rows)
    frame['优于月份气候均值'] = frame['RMSE（cfs）'] < frame.groupby('预见期（天）')['RMSE（cfs）'].transform(
        lambda s: s[frame.loc[s.index, '模型'] == '月份气候均值'].iloc[0])
    out = ROOT / 'results/phase2_v3/基线对比.csv'
    frame.to_csv(out, index=False, encoding='utf-8-sig')
    return frame, out


if __name__ == '__main__':
    frame, out = build_comparison()
    pivot = frame.pivot_table(index=['模型', '策略'], columns='预见期（天）', values='NSE')
    print(pivot.round(3).to_string())
    print('\nsaved:', out)
