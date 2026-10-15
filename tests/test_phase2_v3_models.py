import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
import phase2_v3_models as m3


def toy(n=60, d=5):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, d))
    Y = np.column_stack([X[:, 0] * (h + 1) + rng.normal(scale=.1, size=n)
                         for h in range(1, 8)])
    return X, Y


class ModelTests(unittest.TestCase):
    def test_ann_predicts_native_multioutput(self):
        X, Y = toy()
        model = m3.make_ann()
        model.fit(X[:40], Y[:40])
        out = m3.predict_matrix(model, X[40:])
        self.assertEqual(out.shape, (20, 7))

    def test_rf_predicts_native_multioutput(self):
        X, Y = toy()
        model = m3.make_rf(n_estimators=20)
        model.fit(X[:40], Y[:40])
        self.assertEqual(m3.predict_matrix(model, X[40:]).shape, (20, 7))

    def test_ann_grid_has_no_learning_rate_search(self):
        grid = m3.ann_param_grid()
        keys = set(grid)
        self.assertEqual(keys, {'regressor__model__hidden_layer_sizes',
                                'regressor__model__alpha'})
        self.assertNotIn('regressor__model__learning_rate_init', keys)
        self.assertEqual(grid['regressor__model__alpha'], [1.0, 10.0, 100.0])
        self.assertEqual(grid['regressor__model__hidden_layer_sizes'],
                         [(8,), (16,), (32,)])

    def test_rf_grid_leaf_extends_v2(self):
        self.assertEqual(m3.rf_param_grid()['min_samples_leaf'], [5, 10, 20])
        self.assertEqual(m3.rf_param_grid('refine'),
                         {'max_features': [0.5, 1.0], 'max_samples': [0.7, None]})

    def test_predict_matrix_rejects_single_output(self):
        class Single:
            def predict(self, X):
                return np.zeros(len(X))
        with self.assertRaises(ValueError):
            m3.predict_matrix(Single(), np.zeros((3, 2)))

    def test_diagnostics_flags_iteration_limit(self):
        class Fake:
            n_iter_, max_iter, loss_ = 5000, 5000, 0.5

        class Steps(dict):
            pass

        class Reg:
            named_steps = Steps(model=Fake())

        class Est:
            regressor_ = Reg()

        class FakeSearch:
            best_estimator_ = Est()
            best_score_ = -123.0

        got = m3.ann_diagnostics(FakeSearch)
        self.assertTrue(got['是否触及迭代上限'])
        self.assertEqual(got['实际迭代次数'], 5000)
        self.assertAlmostEqual(got['交叉验证RMSE（cfs）'], 123.0)


if __name__ == '__main__':
    unittest.main()
