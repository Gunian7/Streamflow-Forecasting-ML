"""v5 数据层不变量测试"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'code'))

import numpy as np
import pandas as pd
import pytest

import os
os.environ.setdefault('RUNOFF_DATA_PATH',
                      str(Path(__file__).resolve().parent.parent / '01047000_fixed.csv'))

import phase2_v5_data as d5


@pytest.fixture(scope='module')
def built():
    return d5.build_dataset_v5()


def test_features_include_current_discharge(built):
    _, _, cols, _ = built
    assert 'Discharge' in cols, '当天径流必须进入特征'


def test_features_include_snow(built):
    _, _, cols, _ = built
    for c in ['Swe', 'Swe_melt1', 'PDD']:
        assert c in cols


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
    assert (train['Swe'] > 0).sum() > 100, '积雪应显著非零'


def test_log_roundtrip(built):
    _, test, _, _ = built
    y = d5.target_matrix(test, log=False)
    back = d5.inverse_target(np.log1p(y))
    assert np.allclose(y, back, rtol=1e-9)
