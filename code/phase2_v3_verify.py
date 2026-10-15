"""Recompute every reported v3 test metric from the saved daily predictions."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase2_v3_baselines as b3

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'results/phase2_v3'


def main():
    m = pd.read_csv(OUT / '测试精度指标.csv')
    p = pd.read_csv(OUT / '逐日预测结果.csv')
    assert len(m) == 28, f'指标行数异常: {len(m)}'
    assert len(p) == 28 * 359, f'预测行数异常: {len(p)}'
    for _, row in m.iterrows():
        s = p[(p['模型'] == row['模型']) & (p['策略'] == row['策略']) &
              (p['预见期（天）'] == row['预见期（天）'])]
        assert len(s) == 359, (row['模型'], row['策略'], row['预见期（天）'], len(s))
        y, pred = s['实测径流（cfs）'], s['预测径流（cfs）']
        assert np.isclose(np.sqrt(mean_squared_error(y, pred)), row['RMSE（cfs）'])
        assert np.isclose(mean_absolute_error(y, pred), row['MAE（cfs）'])
        assert np.isclose(r2_score(y, pred), row['R²'])
        assert np.isclose(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum(),
                          row['NSE'])
    print('28 项测试指标与 10052 行逐日预测全部一致')

    v2 = pd.read_csv(ROOT / 'results/phase2_v2/预测精度指标.csv')
    merged = m.merge(v2, on=['模型', '策略', '预见期（天）'],
                     suffixes=('_v3', '_v2'))
    merged['RMSE变化'] = merged['RMSE（cfs）_v3'] - merged['RMSE（cfs）_v2']
    merged['NSE变化'] = merged['NSE_v3'] - merged['NSE_v2']
    merged.to_csv(OUT / '测试指标_v3_v2对比.csv', index=False, encoding='utf-8-sig')

    base = pd.read_csv(OUT / '基线指标.csv')
    clim = base[base['基线'] == '月份气候均值'].set_index('预见期（天）')['RMSE（cfs）']
    merged['月份气候均值RMSE'] = merged['预见期（天）'].map(clim)
    merged['优于气候均值'] = merged['RMSE（cfs）_v3'] < merged['月份气候均值RMSE']
    better = merged.groupby('预见期（天）')['优于气候均值'].sum()
    print('\n各预见期中优于月份气候均值的模型数（共4个）:')
    print(better.to_string())
    print('\nRMSE改善任务数:', int((merged['RMSE变化'] < 0).sum()), '/', len(merged))
    print('RMSE退步任务数:', int((merged['RMSE变化'] > 0).sum()))
    print(merged[merged['RMSE变化'] > 0][['模型', '策略', '预见期（天）', 'RMSE变化']]
          .round(2).to_string(index=False))
    merged.to_csv(OUT / '测试指标_含基线对比.csv', index=False, encoding='utf-8-sig')


if __name__ == '__main__':
    main()
