import unittest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import step2_feature_engineering as fe

class FeatureCSVTests(unittest.TestCase):
    def test_appended_columns_preserve_dates_values_and_training_scalers(self):
        original = pd.DataFrame({'Date': ['2000-01-01', '2000-01-02', '2004-01-01'], 'Discharge': [0., 10., 20.]})
        full = original.copy()
        full['Date'] = pd.to_datetime(full['Date'])
        full['Season'] = 0
        full['Discharge_lag1'] = [np.nan, 0., 10.]
        mm = MinMaxScaler().fit([[0.], [10.]])
        std = StandardScaler().fit([[0.], [10.]])
        bundle = {'feature_cols': ['Discharge'], 'scaler_mm_X': mm, 'scaler_std_X': std}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'values.csv'
            fe.export_feature_values(original, full, bundle, path)
            out = pd.read_csv(path)
        pd.testing.assert_frame_equal(out[original.columns], original)
        self.assertEqual(list(out.columns[:2]), list(original.columns))
        self.assertEqual(out.loc[2, 'Discharge_minmax'], 2.)
        self.assertEqual(out.loc[2, 'Discharge_zscore'], 3.)
        self.assertTrue(pd.isna(out.loc[0, 'Discharge_lag1']))
        self.assertEqual(out.loc[2, 'Season'], 0)
        np.testing.assert_equal(mm.data_max_, [10.])

    def test_seasons_are_one_hot(self):
        _, full = fe.build_features(return_full=True)
        cols = ['Season_spring', 'Season_summer', 'Season_autumn', 'Season_winter']
        self.assertNotIn('Season', full.columns)
        self.assertTrue((full[cols].sum(axis=1) == 1).all())
        for col, months in zip(cols, [[3,4,5], [6,7,8], [9,10,11], [12,1,2]]):
            self.assertTrue((full[col] == full['Month'].isin(months).astype(int)).all())

    def test_selection_excludes_current_discharge(self):
        self.assertEqual(fe.selection_feature_indices(['Discharge', 'Prcp', 'Discharge_lag1']), [1, 2])

if __name__ == '__main__':
    unittest.main()
