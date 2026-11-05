"""Phase-two v3 data: strict date/label boundaries for multi-step runoff."""
import numpy as np
import pandas as pd
import step2_feature_engineering as fe
from phase2_v3_config import PROTOCOL

MAX_HORIZON = PROTOCOL['max_horizon']


def build_dataset():
    """Return (train, test, feature_cols, meta) with leakage-safe boundaries."""
    clean = fe.build_features()
    cols = [c for c in clean.columns
            if c not in PROTOCOL['features_excluded'] and not c.startswith('Target_')]
    if 'Season' in cols or 'Discharge' in cols:
        raise ValueError('候选特征包含被排除的 Season 整数列或当天径流')
    test_start = pd.Timestamp(PROTOCOL['test_start'])
    # Training labels must not reach into the test year: the h-th target of a
    # training row must be earlier than the first test issue date.
    train = clean[clean['Date'] + pd.Timedelta(days=MAX_HORIZON) < test_start].copy()
    test = clean[clean['Date'] >= test_start].copy()
    if train.empty or test.empty:
        raise ValueError('训练或测试集为空')
    if not (train['Date'].is_unique and test['Date'].is_unique):
        raise ValueError('存在重复日期，不能安全按日期对齐')
    if train['Date'].max() >= test['Date'].min():
        raise ValueError('训练日期越界进入测试段')
    gap_days = int((test['Date'].min() - train['Date'].max()).days)
    if gap_days < 1 or gap_days > MAX_HORIZON + 1:
        raise ValueError(f'训练与测试间隔异常: {gap_days} 天')
    # Real invariant: no training target date may reach into the test segment.
    for h in range(1, MAX_HORIZON + 1):
        target_dates = train['Date'] + pd.Timedelta(days=h)
        if target_dates.max() >= test_start:
            raise ValueError(f'训练样本第{h}天目标日期进入测试段')
    diffs = clean['Date'].diff().dropna().dt.days
    if not (diffs == 1).all():
        raise ValueError('原始数据存在非逐日间隔，shift(k) 不等于 k 天')
    targets = [f'Target_lead{h}' for h in range(1, MAX_HORIZON + 1)]
    meta = {
        '训练起': str(train['Date'].min().date()),
        '训练止': str(train['Date'].max().date()),
        '训练样本': len(train),
        '测试起': str(test['Date'].min().date()),
        '测试止': str(test['Date'].max().date()),
        '测试样本': len(test),
        '特征数量': len(cols),
        '特征': cols,
        '最后已知径流': 'Discharge_lag1',
    }
    return train, test, cols, meta


def matrices(train, test, cols):
    X = train[cols].to_numpy(dtype=float)
    Xt = test[cols].to_numpy(dtype=float)
    Y = train[[f'Target_lead{h}' for h in range(1, MAX_HORIZON + 1)]].to_numpy(dtype=float)
    Yt = test[[f'Target_lead{h}' for h in range(1, MAX_HORIZON + 1)]].to_numpy(dtype=float)
    return X, Xt, Y, Yt


if __name__ == '__main__':
    train, test, cols, meta = build_dataset()
    print({k: v for k, v in meta.items() if k != '特征'})
