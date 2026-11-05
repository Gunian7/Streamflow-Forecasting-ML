"""构建修正后的建模数据集

修正内容：
1. 气象变量按正确的日历日对齐（原 CSV 的气象列整体比 Daymet 晚 1 天）
2. 恢复真实的积雪水当量 SWE（原 CSV 全为 0）
3. 保留 USGS 实测径流（已与官方 API 核对一致）

输出：01047000_fixed.csv（列结构与原 CSV 一致，可直接替换）
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DIAG = ROOT / 'results/diagnostics'


def load_daymet(path):
    dm = pd.read_csv(path, skiprows=6)
    dm.columns = ['year', 'yday', 'Dayl', 'Prcp', 'Srad', 'Swe', 'Tmax', 'Tmin', 'Vp']
    dm = dm.dropna(subset=['year', 'yday'])
    dm = dm.astype({'year': int, 'yday': int})
    dm['Date'] = (pd.to_datetime(dm['year'].astype(str) + '-01-01')
                  + pd.to_timedelta(dm['yday'] - 1, unit='D'))
    return dm.set_index('Date').drop(columns=['year', 'yday'])


def main():
    dm = load_daymet(DIAG / 'daymet_full.txt')
    old = pd.read_csv(ROOT / '01047000.csv', parse_dates=['Date']).set_index('Date')

    print('=' * 72)
    print('DAMET 原始数据')
    print('=' * 72)
    print(f'日期范围: {dm.index.min().date()} ~ {dm.index.max().date()}  行数: {len(dm)}')
    full_range = pd.date_range(dm.index.min(), dm.index.max(), freq='D')
    missing = full_range.difference(dm.index)
    print(f'缺失日期数: {len(missing)}')
    if len(missing):
        print('  缺失样例:', [d.date().isoformat() for d in missing[:5]])

    print()
    print('变量统计:')
    print(dm.describe().round(2).to_string())

    # 合并：以径流日期为准
    out = old[['Discharge']].copy()
    out = out.join(dm, how='left')

    print()
    print('=' * 72)
    print('合并结果')
    print('=' * 72)
    print(f'总行数: {len(out)}')
    na = out.isna().sum()
    print('缺失值统计:')
    print(na.to_string())

    # 若开头缺气象（1999只有部分），用前值/插值补
    out = out.ffill().bfill()

    print()
    print('=' * 72)
    print('修正前后对比（2000-01-01 ~ 2000-01-05）')
    print('=' * 72)
    comp = pd.concat([
        old[['Prcp', 'Tmax', 'Swe']].add_prefix('旧_'),
        out[['Prcp', 'Tmax', 'Swe']].add_prefix('新_'),
    ], axis=1).loc['2000-01-01':'2000-01-05']
    print(comp.round(2).to_string())

    print()
    print('=' * 72)
    print('物理合理性校验')
    print('=' * 72)
    print(f'积雪非零天数: 旧 {int((old["Swe"] > 0).sum())} / 新 {int((out["Swe"] > 0).sum())}')
    print(f'积雪最大值:   旧 {old["Swe"].max():.1f} / 新 {out["Swe"].max():.1f} kg/m2')
    win = out[out.index.month.isin([12, 1, 2])]
    print(f'冬季积雪均值: 新 {win["Swe"].mean():.1f} kg/m2')

    print()
    print('径流与降水的最佳滞后（应为 1 天左右）:')
    for lag in range(0, 4):
        r = out['Discharge'].corr(out['Prcp'].shift(lag))
        print(f'  lag{lag}: r = {r:.4f}')

    print()
    print('积雪消融与径流的关系:')
    melt = (out['Swe'].diff() < 0).astype(int)
    print(f'  融雪日数: {int(melt.sum())}')
    print(f'  融雪日平均径流: {out.loc[melt==1, "Discharge"].mean():.1f} cfs')
    print(f'  非融雪日平均径流: {out.loc[melt==0, "Discharge"].mean():.1f} cfs')

    # 保存
    out.index.name = 'Date'
    save = out.reset_index()
    save = save[['Date', 'Discharge', 'Dayl', 'Prcp', 'Srad', 'Swe', 'Tmax', 'Tmin', 'Vp']]
    path = ROOT / '01047000_fixed.csv'
    save.to_csv(path, index=False, float_format='%.2f')
    print()
    print(f'✓ 已保存: {path}')
    print(f'  行数 {len(save)}, 列数 {len(save.columns)}')
    print(f'  日期 {save["Date"].min().date()} ~ {save["Date"].max().date()}')


if __name__ == '__main__':
    main()
