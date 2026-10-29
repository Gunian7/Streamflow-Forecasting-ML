"""Six-model direct Stacking with a GradientBoosting meta-learner."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from enhanced_features import get_enhanced_dataset
from phase3_lstm import PseudoLSTM
import phase2_v5_data as d5

OUT = ROOT / "results" / "phase3_enhanced"
OUT.mkdir(parents=True, exist_ok=True)
FAMILIES = ["ANN", "RF", "PseudoLSTM", "PyTorch-LSTM", "XGBoost", "LightGBM"]
HORIZONS = range(1, 8)
SEED = 42


def metrics(y, prediction):
    y = np.asarray(y, dtype=float).ravel()
    prediction = np.asarray(prediction, dtype=float).ravel()
    if len(y) != len(prediction):
        raise ValueError("y and prediction must have the same length")
    ss_res = float(np.sum((y - prediction) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return {
        "RMSE": float(np.sqrt(np.mean((y - prediction) ** 2))),
        "MAE": float(np.mean(np.abs(y - prediction))),
        "R²": float(1.0 - ss_res / ss_tot),
    }


def select_feature_matrix(frame, feature_cols, expected_dim=None):
    """Select an exact ordered feature contract and validate model input width."""
    missing = [name for name in feature_cols if name not in frame.columns]
    if missing:
        raise ValueError(f"missing required features: {missing}")
    matrix = frame.loc[:, feature_cols].to_numpy(dtype=float)
    if expected_dim is not None and matrix.shape[1] != expected_dim:
        raise ValueError(
            f"feature dimension mismatch: model expects {expected_dim}, got {matrix.shape[1]}"
        )
    if not np.isfinite(matrix).all():
        raise ValueError("feature matrix contains non-finite values")
    return matrix


def align_by_date(old_frame, enhanced_frame):
    """Return old/enhanced frames on their common dates, in chronological order."""
    old = old_frame.copy()
    enhanced = enhanced_frame.copy()
    old["Date"] = pd.to_datetime(old["Date"])
    enhanced["Date"] = pd.to_datetime(enhanced["Date"])
    common = pd.Index(old["Date"]).intersection(pd.Index(enhanced["Date"])).sort_values()
    if common.empty:
        raise ValueError("old and enhanced datasets have no common dates")
    old = old.set_index("Date").loc[common].reset_index()
    enhanced = enhanced.set_index("Date").loc[common].reset_index()
    return old, enhanced


def _load_models():
    # Legacy PseudoLSTM artifacts were pickled from __main__ in an earlier run.
    sys.modules["__main__"].PseudoLSTM = PseudoLSTM
    models = {name: {} for name in FAMILIES}
    for horizon in HORIZONS:
        models["ANN"][horizon] = joblib.load(ROOT / f"results/phase2_v5/ANN_{horizon}.joblib")
        models["RF"][horizon] = joblib.load(ROOT / f"results/phase2_v5/RF_{horizon}.joblib")
        models["PseudoLSTM"][horizon] = joblib.load(
            ROOT / f"results/phase3_lstm/lstm_direct_{horizon}.joblib"
        )
        for family in ["PyTorch-LSTM", "XGBoost", "LightGBM"]:
            filename = {
                "PyTorch-LSTM": "pytorch_lstm",
                "XGBoost": "xgboost",
                "LightGBM": "lightgbm",
            }[family]
            path = OUT / f"{filename}_direct_{horizon}.joblib"
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing {family} model for day {horizon}: {path}. "
                    "Run code/train_pytorch_lstm.py and code/train_boosting.py first."
                )
            models[family][horizon] = joblib.load(path)
    return models


def _predict(model, X, family=None):
    prediction = np.asarray(model.predict(X), dtype=float)
    if prediction.ndim == 2 and prediction.shape[1] == 1:
        prediction = prediction[:, 0]
    if prediction.ndim != 1:
        raise ValueError(f"base model must produce one value per row, got {prediction.shape}")
    # Phase-2 ANN/RF artifacts were fitted on log1p targets.
    if family in {"ANN", "RF"}:
        prediction = np.expm1(np.clip(prediction, -20, 20))
    return np.clip(prediction, 0, None)


def _window_map(frame, feature_cols, sequence_length=7):
    """Create rolling LSTM windows keyed by endpoint date."""
    dates = pd.to_datetime(frame["Date"]).to_numpy()
    values = select_feature_matrix(frame, feature_cols)
    windows = {}
    for end in range(sequence_length - 1, len(frame)):
        if dates[end] - dates[end - sequence_length + 1] != np.timedelta64(sequence_length - 1, "D"):
            continue
        windows[pd.Timestamp(dates[end])] = values[end - sequence_length + 1:end + 1]
    return windows


def _load_day_models(horizon):
    """Load fold-training templates with established search parameters."""
    from sklearn.base import clone as sklearn_clone
    import phase2_v5_experiment as experiment

    ann_search = joblib.load(ROOT / f"results/phase2_v5/ANN_{horizon}.joblib")
    rf_search = joblib.load(ROOT / f"results/phase2_v5/RF_{horizon}.joblib")
    return {
        "ANN": sklearn_clone(ann_search.best_estimator_),
        "RF": sklearn_clone(rf_search.best_estimator_),
        "PseudoLSTM": PseudoLSTM(hidden_layer_sizes=(64, 32), alpha=0.001,
                                 max_iter=500, random_state=SEED + horizon),
        "XGBoost": __import__("train_boosting")._xgb_model(),
        "LightGBM": __import__("train_boosting")._lgb_model(),
    }


def _fit_oof_base(family, model, X, y, train_idx, val_idx, horizon):
    if family == "XGBoost":
        from train_boosting import _split_validation
        X_fit, X_val, y_fit, y_val = _split_validation(X[train_idx], y[train_idx], 0.15)
        model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
    elif family == "LightGBM":
        from train_boosting import _fit_lightgbm, _split_validation
        X_fit, X_val, y_fit, y_val = _split_validation(X[train_idx], y[train_idx], 0.15)
        _fit_lightgbm(model, X_fit, y_fit, X_val, y_val)
    else:
        model.fit(X[train_idx], y[train_idx])
    return np.clip(_predict(model, X[val_idx], family=family), 0, None)


def _pytorch_oof_predict(train_frame, feature_cols, y, horizon, splits=3):
    """Fit fold-local PyTorch models so meta-training predictions are out-of-fold."""
    import torch
    from pytorch_lstm import LSTMRegressor, LSTMTrainer
    from train_pytorch_lstm import make_windows, split_validation

    dates = pd.to_datetime(train_frame["Date"])
    windows, labels, endpoint_dates = make_windows(
        train_frame,
        feature_cols,
        [f"Target_lead{horizon}"],
        sequence_length=7,
    )
    label_by_date = {pd.Timestamp(date): float(label[0]) for date, label in zip(endpoint_dates, labels)}
    output = {}
    tscv = TimeSeriesSplit(n_splits=splits, gap=7)
    endpoint_positions = np.arange(len(windows))
    for train_idx, val_idx in tscv.split(endpoint_positions):
        Xtr, Xval = windows[train_idx], windows[val_idx]
        ytr, yval = labels[train_idx, 0], labels[val_idx, 0]
        Xfit, Xvalid, yfit, yvalid = split_validation(Xtr, ytr, validation_fraction=0.2)
        model = LSTMRegressor(input_dim=len(feature_cols), hidden_dim=64, num_layers=2,
                              output_dim=1, dropout=0.2, bidirectional=False)
        torch.manual_seed(SEED + horizon + int(train_idx[-1]))
        trainer = LSTMTrainer(model, device="cuda" if torch.cuda.is_available() else "cpu")
        trainer.fit(Xfit, yfit, Xvalid, yvalid, epochs=60, batch_size=64,
                    early_stopping_patience=10, verbose=False)
        pred = np.asarray(trainer.predict(Xval)).reshape(-1)
        for date, value in zip(endpoint_dates[val_idx], pred):
            output[pd.Timestamp(date)] = float(value)
    return output, label_by_date


def _create_oof_meta(train_old, train_enhanced, cols_old, cols_enhanced, horizon):
    from sklearn.model_selection import TimeSeriesSplit

    y = train_enhanced[f"Target_lead{horizon}"].to_numpy(dtype=float)
    X_old = select_feature_matrix(train_old, cols_old, expected_dim=71)
    X_enh = select_feature_matrix(train_enhanced, cols_enhanced, expected_dim=len(cols_enhanced))
    oof = np.full((len(y), len(FAMILIES)), np.nan, dtype=float)
    splitter = TimeSeriesSplit(n_splits=3, gap=7)

    for train_idx, val_idx in splitter.split(X_enh):
        templates = _load_day_models(horizon)
        for col, family in enumerate(["ANN", "RF", "PseudoLSTM"]):
            oof[val_idx, col] = _fit_oof_base(
                family, templates[family], X_old, y, train_idx, val_idx, horizon
            )
        for col, family in [(4, "XGBoost"), (5, "LightGBM")]:
            oof[val_idx, col] = _fit_oof_base(
                family, templates[family], X_enh, y, train_idx, val_idx, horizon
            )

    pytorch_oof, _ = _pytorch_oof_predict(train_enhanced, cols_enhanced, y, horizon)
    for row, date in enumerate(pd.to_datetime(train_enhanced["Date"])):
        if date in pytorch_oof:
            oof[row, 3] = pytorch_oof[date]

    valid = np.isfinite(oof).all(axis=1)
    return oof[valid], y[valid], pd.to_datetime(train_enhanced.loc[valid, "Date"]).reset_index(drop=True)


def stacking_6models():
    train_enhanced, test_enhanced, cols_enhanced, meta = get_enhanced_dataset()
    train_old, test_old, cols_old, _ = d5.build_dataset_v5()
    train_old, train_enhanced = align_by_date(train_old, train_enhanced)
    test_old, test_enhanced = align_by_date(test_old, test_enhanced)

    models = _load_models()
    X_train_old = select_feature_matrix(train_old, cols_old, expected_dim=71)
    X_test_old = select_feature_matrix(test_old, cols_old, expected_dim=71)
    X_train_enh = select_feature_matrix(train_enhanced, cols_enhanced)
    X_test_enh = select_feature_matrix(test_enhanced, cols_enhanced)
    train_windows = _window_map(train_enhanced, cols_enhanced)
    test_windows = _window_map(test_enhanced, cols_enhanced)

    rows, detail_rows = [], []
    for horizon in HORIZONS:
        meta_train, y_train, train_dates = _create_oof_meta(
            train_old, train_enhanced, cols_old, cols_enhanced, horizon
        )
        candidate_dates = sorted(test_windows)
        # Use only test endpoint dates for which all six models have matching inputs.
        test_rows = test_enhanced.set_index("Date").loc[candidate_dates]
        old_rows = test_old.set_index("Date").loc[candidate_dates]
        x_old = select_feature_matrix(old_rows, cols_old, expected_dim=71)
        x_enh = select_feature_matrix(test_rows, cols_enhanced)
        x_seq = np.stack([test_windows[date] for date in candidate_dates])
        meta_test = np.column_stack([
            _predict(models["ANN"][horizon], x_old, "ANN"),
            _predict(models["RF"][horizon], x_old, "RF"),
            _predict(models["PseudoLSTM"][horizon], x_old, "PseudoLSTM"),
            _predict(models["PyTorch-LSTM"][horizon], x_seq, "PyTorch-LSTM"),
            _predict(models["XGBoost"][horizon], x_enh, "XGBoost"),
            _predict(models["LightGBM"][horizon], x_enh, "LightGBM"),
        ])
        meta_model = GradientBoostingRegressor(
            n_estimators=100, max_depth=2, learning_rate=0.03,
            min_samples_leaf=5, loss="squared_error", random_state=SEED,
        )
        meta_model.fit(meta_train, y_train)
        prediction = np.clip(meta_model.predict(meta_test), 0, None)
        y_test = test_rows[f"Target_lead{horizon}"].to_numpy(dtype=float)
        score = metrics(y_test, prediction)
        rows.append({
            "集成方法": "6模型-Stacking(GBR)",
            "策略": "direct",
            "预见期（天）": horizon,
            **score,
            "训练元特征样本": len(y_train),
            "测试样本": len(y_test),
            "基学习器": "+".join(FAMILIES),
        })
        detail_rows.extend(
            {
                "策略": "direct",
                "方法": "6模型-Stacking(GBR)",
                "预见期（天）": horizon,
                "日期": date,
                "实测": float(observed),
                "预测": float(pred),
            }
            for date, observed, pred in zip(candidate_dates, y_test, prediction)
        )
        joblib.dump(meta_model, OUT / f"stacking6_meta_{horizon}.joblib", compress=3)
        print(f"  第{horizon}天: R²={score['R²']:.4f}, RMSE={score['RMSE']:.1f}", flush=True)

    results = pd.DataFrame(rows)
    details = pd.DataFrame(detail_rows)
    results.to_csv(OUT / "enhanced_ensemble_metrics.csv", index=False, encoding="utf-8-sig")
    details.to_csv(OUT / "enhanced_ensemble_predictions.csv", index=False, encoding="utf-8-sig")
    return results


def main():
    started = time.time()
    print("=" * 74)
    print("6模型增强集成（Gradient Boosting 元学习器）")
    results = stacking_6models()
    print("\n增强集成结果:")
    print(results[["预见期（天）", "R²", "RMSE", "MAE"]].to_string(index=False))

    old_path = ROOT / "results" / "phase3_ensemble" / "ensemble_metrics.csv"
    old = pd.read_csv(old_path, encoding="utf-8-sig")
    old_day1 = old[(old["策略"] == "direct") & (old["方法"] == "Stacking") &
                   (old["预见期（天）"] == 1)].iloc[0]
    new_day1 = results[results["预见期（天）"] == 1].iloc[0]
    print("\n与原 Stacking 对比（第1天）:")
    print(f"  原 Stacking (3模型): R²={old_day1['R²']:.4f}, RMSE={old_day1['RMSE']:.1f}")
    print(f"  增强 Stacking (6模型): R²={new_day1['R²']:.4f}, RMSE={new_day1['RMSE']:.1f}")
    print(f"  提升: ΔR²={new_day1['R²'] - old_day1['R²']:+.4f}")
    print(f"  目标 R²≥0.80: {'达成' if new_day1['R²'] >= 0.80 else '未达成'}")
    print(f"总耗时: {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
