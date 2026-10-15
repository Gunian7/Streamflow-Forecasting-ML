import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
import phase2_v3_data as d3
import phase2_v3_config as cfg


def synthetic():
    dates = pd.date_range('2003-11-01', periods=20, freq='D')
    df = pd.DataFrame({'Date': dates, 'Discharge': np.arange(20, dtype=float) * 10})
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['DayOfYear'] = df['Date'].dt.dayofyear
    for c in ['Season_spring', 'Season_summer', 'Season_autumn', 'Season_winter']:
        df[c] = 0
    for h in range(1, cfg.PROTOCOL['max_horizon'] + 1):
        df[f'Target_lead{h}'] = df['Discharge'].shift(-h)
    return df


class DataBoundaryTests(unittest.TestCase):
    def test_tampered_target_rejected(self):
        df = synthetic()
        df.loc[5, 'Target_lead3'] = 999.0
        with self.assertRaises(AssertionError):
            _ = _check_frames(df)

    def test_missing_date_rejected(self):
        df = synthetic()
        broken = pd.concat([df.iloc[:7], df.iloc[8:]]).reset_index(drop=True)
        diffs = broken['Date'].diff().dropna().dt.days
        self.assertFalse((diffs == 1).all())

    def test_targets_equal_shifted_flow(self):
        df = synthetic()
        for h in range(1, 8):
            expect = df['Discharge'].shift(-h)
            pd.testing.assert_series_equal(df[f'Target_lead{h}'], expect,
                                           check_names=False)

    def test_real_dataset_boundaries(self):
        train, test, cols, meta = d3.build_dataset()
        self.assertNotIn('Discharge', cols)
        self.assertNotIn('Season', cols)
        self.assertEqual(meta['训练样本'], 1447)
        self.assertEqual(meta['测试样本'], 359)
        self.assertEqual(meta['测试起'], '2004-01-01')
        self.assertEqual(meta['训练止'], '2003-12-24')
        self.assertEqual(meta['训练样本'], 1447)
        X, Xt, Y, Yt = d3.matrices(train, test, cols)
        self.assertEqual(X.shape[0], len(train))
        self.assertEqual(Y.shape, (len(train), 7))
        self.assertEqual(Xt.shape[0], len(test))
        self.assertEqual(Yt.shape, (len(test), 7))


def _check_frames(df):
    for h in range(1, cfg.PROTOCOL['max_horizon'] + 1):
        expect = df['Discharge'].shift(-h)
        if not np.allclose(df[f'Target_lead{h}'].fillna(-1), expect.fillna(-1)):
            raise AssertionError(f'Target_lead{h} 与 shift(-{h}) 不一致')
    return True


if __name__ == '__main__':
    unittest.main()
