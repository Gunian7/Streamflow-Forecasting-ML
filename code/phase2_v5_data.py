"""Phase-2 v5 数据层：修正对齐后的气象 + 当天径流 + 积雪特征

相对 v3 的修正：
1. 气象变量按 Daymet 正确日历对齐（原 CSV 错位 1 天）
2. 恢复真实 SWE，并派生融雪量、正积温
3. 把当天径流 Q(t) 纳入特征（起报时刻已知，不构成泄露）
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

import step2_feature_engineering as fe
from phase2_v3_config import PROTOCOL as BASE_PROTOCOL

ROOT = Path(__file__).resolve().parent.parent
MAX_HORIZON = 7
TEST_START = pd.Timestamp('2004-01-01')

# 仍排除：日期辅助列、季节整数列；Swe / Discharge 已解禁
STILL_EXCLUDED = ['Date', 'Year', 'Month', 'Day', 'DayOfWeek', 'DayOfYear', 'Season']

SNOW_FEATURES = ['Swe', 'Swe_melt1', 'Swe_melt_3', 'Swe_melt_7',
                 'Swe_roll_mean_7', 'PDD', 'PDD_3', 'Snow_cover']


def _add_snow_features(df):
    """积雪与融雪特征（只用当天及更早的信息）"""
    df = df.copy()
    df['Swe_melt1'] = (df['Swe'].shift(1) - df['Swe']).clip(lower=0)
    df['Swe_melt_3'] = df['Swe_melt1'].rolling(3).sum()
    df['Swe_melt_7'] = df['Swe_melt1'].rolling(7).sum()
    df['Swe_roll_mean_7'] = df['Swe'].rolling(7).mean()
    df['PDD'] = df['Tmax'].clip(lower=0)
    df['PDD_3'] = df['PDD'].rolling(3).sum()
    df['Snow_cover'] = (df['Swe'] > 0).astype(int)
    return df


def build_dataset_v5():
    """返回 (train, test, feature_cols, meta)"""
    clean = fe.build_features()
    clean = _add_snow_features(clean)

    cols = [c for c in clean.columns
            if c not in STILL_EXCLUDED and not c.startswith('Target_')]

    # v5 明确纳入当天径流
    if 'Discharge' not in cols:
        raise ValueError('当天径流未进入特征')
    if 'Swe' not in cols:
        raise ValueError('SWE 未进入特征')

    # 去掉零方差列
    nunique = clean[cols].nunique()
    zero_var = [c for c in cols if nunique[c] <= 1]
    cols = [c for c in cols if c not in zero_var]

    # 丢弃因滚动窗口产生的 NaN 行（保留前 7 天以外的部分）
    clean = clean.dropna(subset=cols + [f'Target_lead{h}' for h in range(1, 8)])

    train = clean[clean['Date'] + pd.Timedelta(days=MAX_HORIZON) < TEST_START].copy()
    test = clean[clean['Date'] >= TEST_START].copy()

    if train.empty or test.empty:
        raise ValueError('训练或测试集为空')

    # 边界校验：训练样本的任一目标日期不得进入测试段
    for h in range(1, MAX_HORIZON + 1):
        if (train['Date'] + pd.Timedelta(days=h)).max() >= TEST_START:
            raise ValueError(f'训练第{h}天目标日期越过测试起点')

    diffs = clean['Date'].diff().dropna().dt.days
    if not (diffs == 1).all():
        raise ValueError('数据存在非逐日间隔，shift 语义不成立')

    meta = {
        '数据文件': str(fe.DATA_PATH),
        '训练起': str(train['Date'].min().date()),
        '训练止': str(train['Date'].max().date()),
        '训练样本': int(len(train)),
        '测试起': str(test['Date'].min().date()),
        '测试止': str(test['Date'].max().date()),
        '测试样本': int(len(test)),
        '特征数量': len(cols),
        '含当天径流': True,
        '含积雪特征': True,
        '剔除零方差列': zero_var,
    }
    return train, test, cols, meta


def target_matrix(frame, log=True):
    """目标矩阵；log=True 时返回 log1p 变换"""
    Y = frame[[f'Target_lead{h}' for h in range(1, 8)]].to_numpy(float)
    return np.log1p(Y) if log else Y


def inverse_target(pred, log=True):
    pred = np.asarray(pred, dtype=float)
    return np.expm1(pred) if log else pred


if __name__ == '__main__':
    train, test, cols, meta = build_dataset_v5()
    print(json.dumps({k: v for k, v in meta.items() if k != '特征'}, ensure_ascii=False, indent=2))
    print('\n特征列表:')
    for c in cols:
        print(' ', c)
