"""CV-only comparison of v2 vs v3 selections (no test-set peeking)."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'results/phase2_v3'


def load():
    rows = []
    old = pd.read_csv(ROOT / 'results/phase2_v2/最佳超参数.csv')
    for _, r in old.iterrows():
        rows.append({'模型': r['模型'], '策略': r['策略'], '选参预见期': r['选参预见期'],
                     '轮次': 'v2', '交叉验证RMSE（cfs）': r['交叉验证RMSE']})
    ann = pd.read_csv(OUT / 'ANN选参与诊断.csv')
    for _, r in ann.iterrows():
        rows.append({'模型': 'ANN', '策略': r['策略'], '选参预见期': r['选参预见期'],
                     '轮次': 'v3', '交叉验证RMSE（cfs）': r['交叉验证RMSE（cfs）']})
    rf = pd.read_csv(OUT / 'RF选参.csv')
    for _, r in rf.iterrows():
        rows.append({'模型': 'RF', '策略': r['策略'], '选参预见期': r['选参预见期'],
                     '轮次': 'v3', '交叉验证RMSE（cfs）': r['交叉验证RMSE（cfs）']})
    return pd.DataFrame(rows)


if __name__ == '__main__':
    frame = load()
    pivot = frame.pivot_table(index=['模型', '策略', '选参预见期'], columns='轮次',
                              values='交叉验证RMSE（cfs）')
    pivot['变化量'] = pivot['v3'] - pivot['v2']
    pivot['是否改善'] = pivot['变化量'] < 0
    print(pivot.round(2).to_string())
    print('\n改善任务数:', int(pivot['是否改善'].sum()), '/', len(pivot))
    frame.to_csv(OUT / '选参验证误差对比_v2_v3.csv', index=False, encoding='utf-8-sig')
