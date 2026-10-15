import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
import phase2_v3_baselines as b3
import phase2_v3_data as d3


class BaselineTests(unittest.TestCase):
    def test_persistence_repeats_last_value(self):
        out = b3.persistence(np.array([[10.], [20.], [30.]]))
        self.assertEqual(out.shape, (3, 7))
        for row, value in zip(out, [10., 20., 30.]):
            self.assertTrue((row == value).all())

    def test_persistence_rejects_other_horizon(self):
        with self.assertRaises(ValueError):
            b3.persistence(np.array([[1.]]), horizon=3)

    def test_metrics_match_manual_computation(self):
        a = np.array([1., 2., 3., 4.])
        p = np.array([1., 2., 3., 5.])
        got = b3.metrics(a, p)
        self.assertAlmostEqual(got['RMSE（cfs）'], 0.5)
        self.assertAlmostEqual(got['MAE（cfs）'], 0.25)
        nse = 1 - 1.0 / ((a - a.mean()) ** 2).sum()
        self.assertAlmostEqual(got['NSE'], nse)

    def test_climatology_uses_train_only_and_month_of_target(self):
        dates = pd.to_datetime(['2003-12-30', '2003-12-31'])
        train = pd.DataFrame({'Date': pd.to_datetime(['2003-01-05', '2003-02-06']),
                              'Discharge': [100., 200.]})
        out = b3.climatology_month_mean(dates, train)
        # 2003-12-30 + 1 day = 12-31 (no training month) -> overall mean 150
        self.assertEqual(out[0, 0], 150.0)
        # 2003-12-31 + 1 day = 2004-01-01 (month 1, trained) -> 100.0
        self.assertEqual(out[1, 0], 100.0)

    def test_real_baselines_are_computed(self):
        train, test, cols, meta = d3.build_dataset()
        frame, preds = b3.evaluate_baselines(train, test)
        self.assertEqual(len(frame), 14)
        self.assertEqual(set(frame['基线']), {'持续性', '月份气候均值'})
        for name in ['持续性', '月份气候均值']:
            self.assertEqual(sorted(frame[frame['基线'] == name]['预见期（天）']),
                             list(range(1, 8)))
        for name, matrix in preds.items():
            self.assertEqual(matrix.shape, (len(test), 7))


if __name__ == '__main__':
    unittest.main()
