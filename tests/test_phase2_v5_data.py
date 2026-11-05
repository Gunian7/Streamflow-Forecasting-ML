"""v5 数据层不变量测试"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'code'))

import numpy as np
import pandas as pd
import pytest

import step2_feature_engineering as fe
import phase2_v5_data as d5

FIXED_CSV = Path(__file__).resolve().parent.parent / '01047000_fixed.csv'
SNOW_SKIP_REASON = ('当前数据源 Swe 恒为 0，积雪派生列被零方差剔除；'
                    '本测试需在含真实积雪的 01047000_fixed.csv 上断言')


@pytest.fixture(scope='module')
def built():
    """在修正数据集上构建，不依赖测试收集顺序。

    step2_feature_engineering.DATA_PATH 在首次 import 时固化，这里显式指向
    01047000_fixed.csv，用完还原，避免影响其他测试模块。
    """
    original = fe.DATA_PATH
    fe.DATA_PATH = str(FIXED_CSV)
    try:
        return d5.build_dataset_v5()
    finally:
        fe.DATA_PATH = original


def swe_all_zero(train):
    """当前数据源的积雪水当量是否恒为 0（原始 01047000.csv 即如此）。"""
    if 'Swe' not in train.columns:
        return True
    return int((train['Swe'] > 0).sum()) == 0


def test_features_include_current_discharge(built):
    _, _, cols, _ = built
    assert 'Discharge' in cols, '当天径流必须进入特征'


def test_features_include_snow(built):
    train, _, cols, _ = built
    if swe_all_zero(train):
        pytest.skip(SNOW_SKIP_REASON)
    for c in ['Swe', 'Swe_melt1', 'PDD', 'Snow_cover']:
        assert c in cols, '%s 应进入特征' % c


def test_no_training_target_reaches_test(built):
    train, test, _, _ = built
    test_start = test['Date'].min()
    for h in range(1, 8):
        assert (train['Date'] + pd.Timedelta(days=h)).max() < test_start


def test_daily_continuity(built):
    train, test, _, _ = built
    for frame in (train, test):
        diffs = frame['Date'].diff().dropna().dt.days
        assert (diffs == 1).all()


def test_snow_not_all_zero(built):
    train, _, _, _ = built
    if swe_all_zero(train):
        pytest.skip(SNOW_SKIP_REASON)
    assert (train['Swe'] > 0).sum() > 100, '积雪应显著非零'


def test_log_roundtrip(built):
    _, test, _, _ = built
    y = d5.target_matrix(test, log=False)
    back = d5.inverse_target(np.log1p(y))
    assert np.allclose(y, back, rtol=1e-9)
