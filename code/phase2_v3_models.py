"""Phase-two v3 ANN / RF estimators. Native multi-output only."""
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_ann():
    """Scaled input, scaled target, native multi-output MLP."""
    pipe = Pipeline([('scale', StandardScaler()),
                     ('model', MLPRegressor(solver='lbfgs', max_iter=5000,
                                            max_fun=100000, tol=1e-5,
                                            early_stopping=False,
                                            random_state=42))])
    return TransformedTargetRegressor(regressor=pipe,
                                      transformer=StandardScaler())


def ann_param_grid():
    # L-BFGS has no learning_rate_init search; regularisation is widened
    # instead of growing the network.
    return {'regressor__model__hidden_layer_sizes': [(8,), (16,), (32,)],
            'regressor__model__alpha': [1.0, 10.0, 100.0]}


def make_rf(n_estimators=300):
    return RandomForestRegressor(n_estimators=n_estimators, n_jobs=1,
                                 random_state=42)


def rf_param_grid(stage='coarse'):
    if stage == 'coarse':
        return {'max_depth': [8, 16, None], 'min_samples_leaf': [5, 10, 20]}
    return {'max_features': [0.5, 1.0], 'max_samples': [0.7, None]}


def ann_diagnostics(search):
    """Actual optimisation behaviour of the refit best estimator."""
    model = search.best_estimator_.regressor_.named_steps['model']
    hit_limit = bool(model.n_iter_ >= model.max_iter)
    return {'实际迭代次数': int(model.n_iter_),
            '最大迭代保护值': int(model.max_iter),
            '是否触及迭代上限': hit_limit,
            '训练损失（缩放尺度）': float(model.loss_),
            '交叉验证RMSE（cfs）': float(-search.best_score_)}


def predict_matrix(estimator, X):
    """Return (n, 7); a single-target estimator yields (n,) and is invalid."""
    out = estimator.predict(X)
    out = np.asarray(out)
    if out.ndim != 2 or out.shape[1] != 7:
        raise ValueError(f'期望原生多输出 (n,7)，实际 {out.shape}')
    return out
