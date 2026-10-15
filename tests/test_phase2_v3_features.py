import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
import phase2_v3_features as f3
import phase2_v3_data as d3


def flow(n):
    return pd.DataFrame({'Date': pd.date_range('2000-01-01', periods=n, freq='D'),
                         'Prcp': np.arange(n, dtype=float),
                         'Discharge': np.arange(n, dtype=float) * 2.0,
                         'Tmax': np.arange(n, dtype=float) + 5,
                         'Tmin': np.arange(n, dtype=float)})


class FeatureGroupTests(unittest.TestCase):
    def test_groups_are_cumulative_and_exclude_forbidden(self):
        train, test, cols, meta = d3.build_dataset()
        g = f3.feature_groups(cols)
        a, b, c = g['A_历史径流与季节'], g['B_加气象观测'], g['C_加窗口统计']
        self.assertTrue(set(a) <= set(b))
        self.assertTrue(set(b) <= set(c))
        for name, members in g.items():
            self.assertNotIn('Discharge', members, name)
            self.assertNotIn('Season', members, name)
            self.assertNotIn('Swe', members, name)
            self.assertFalse([m for m in members if m.startswith('Target_')])

    def test_group_sizes(self):
        train, test, cols, meta = d3.build_dataset()
        g = f3.feature_groups(cols)
        self.assertEqual(len(g['A_历史径流与季节']), 12)
        self.assertEqual(len(g['B_加气象观测']), 38)
        self.assertEqual(len(g['C_加窗口统计']), 62)

    def test_long_window_first_valid_index(self):
        df = flow(40)
        out = f3.add_long_windows(df)
        # shift(1) then rolling(14): first valid at original index 14
        self.assertTrue(out['Prcp_roll_sum_14'].iloc[:14].isna().all())
        self.assertFalse(np.isnan(out['Prcp_roll_sum_14'].iloc[14]))
        # shift(1).rolling(14) at index 14 uses Prcp[0..13] = 0+1+...+13
        expect = float(np.arange(0, 14).sum())
        self.assertAlmostEqual(out['Prcp_roll_sum_14'].iloc[14], expect)

    def test_long_window_excludes_today(self):
        df = flow(20)
        out = f3.add_long_windows(df)
        self.assertAlmostEqual(out['Prcp_roll_sum_14'].iloc[15],
                               float(np.arange(1, 15).sum()))

    def test_group_d_extends_group_c(self):
        train, test, cols, meta = d3.build_dataset()
        c = set(f3.feature_groups(cols)['C_加窗口统计'])
        d = set(f3.group_d_columns(cols))
        self.assertTrue(c <= d)
        self.assertEqual(len(d - c), 5)

    def test_common_dates_intersection(self):
        f1 = pd.DataFrame({'Date': pd.to_datetime(['2000-01-01', '2000-01-02'])})
        f2 = pd.DataFrame({'Date': pd.to_datetime(['2000-01-02', '2000-01-03'])})
        got = f3.common_dates([f1, f2])
        self.assertEqual(list(got), [pd.Timestamp('2000-01-02')])


if __name__ == '__main__':
    unittest.main()
