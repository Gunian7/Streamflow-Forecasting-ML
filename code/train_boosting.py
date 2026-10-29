"""Train XGBoost and LightGBM direct forecasting models with early stopping."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from enhanced_features import get_enhanced_dataset

OUT = ROOT / "results" / "phase3_enhanced"
OUT.mkdir(parents=True, exist_ok=True)
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


def _split_validation(X, y, validation_fraction=0.2):
    n_val = max(1, int(len(X) * validation_fraction))
    if n_val >= len(X):
        raise ValueError("validation split leaves no training samples")
    return X[:-n_val], X[-n_val:], y[:-n_val], y[-n_val:]


def _save_feature_metadata(feature_cols, meta):
    payload = {"feature_columns": list(feature_cols), "data_meta": meta}
    (OUT / "boosting_features.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _xgb_model():
    return xgb.XGBRegressor(
        n_estimators=800,
        max_depth=5,
        learning_rate=0.03,
        min_child_weight=3,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=SEED,
        n_jobs=-1,
        tree_method="hist",
        early_stopping_rounds=40,
    )


def _lgb_model():
    return lgb.LGBMRegressor(
        n_estimators=800,
        max_depth=5,
        num_leaves=31,
        learning_rate=0.03,
        min_child_samples=15,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="regression",
        random_state=SEED,
        n_jobs=-1,
        verbosity=-1,
    )


def _fit_lightgbm(model, X_train, y_train, X_val, y_val):
    model.fit(
        X_train,
        y_train,
        eval_X=X_val,
        eval_y=y_val,
        callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)],
    )
    return model


def train_family(family, train, test, feature_cols):
    X = train[feature_cols].to_numpy(dtype=float)
    Xt = test[feature_cols].to_numpy(dtype=float)
    X_fit, X_val, _, _ = _split_validation(X, train["Target_lead1"].to_numpy(float))
    n_val = len(X_val)
    rows = []
    details = []

    for horizon in HORIZONS:
        y = train[f"Target_lead{horizon}"].to_numpy(dtype=float)
        y_test = test[f"Target_lead{horizon}"].to_numpy(dtype=float)
        X_fit, X_val, y_fit, y_val = _split_validation(X, y)
        model = _xgb_model() if family == "XGBoost" else _lgb_model()
        if family == "XGBoost":
            model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
        else:
            _fit_lightgbm(model, X_fit, y_fit, X_val, y_val)

        prediction = np.clip(np.asarray(model.predict(Xt), dtype=float), 0, None)
        score = metrics(y_test, prediction)
        best_iteration = getattr(model, "best_iteration", None)
        if best_iteration is None:
            best_iteration = getattr(model, "best_iteration_", None)
        row = {
            "模型": family,
            "策略": "直接",
            "预见期（天）": horizon,
            **score,
            "best_iteration": best_iteration,
            "训练样本": len(X_fit),
            "验证样本": n_val,
            "特征数": len(feature_cols),
        }
        rows.append(row)
        details.extend(
            {
                "模型": family,
                "预见期（天）": horizon,
                "样本序号": i,
                "日期": date,
                "实测": float(observed),
                "预测": float(pred),
            }
            for i, (date, observed, pred) in enumerate(
                zip(test["Date"], y_test, prediction)
            )
        )
        joblib.dump(model, OUT / f"{family.lower()}_direct_{horizon}.joblib", compress=3)
        print(
            f"  {family} 第{horizon}天: R²={score['R²']:.4f}, "
            f"RMSE={score['RMSE']:.1f}, best_iteration={best_iteration}",
            flush=True,
        )

    return pd.DataFrame(rows), pd.DataFrame(details)


def main():
    started = time.time()
    train, test, feature_cols, meta = get_enhanced_dataset()
    print("=" * 74)
    print("训练梯度提升树模型（XGBoost & LightGBM）")
    print(f"训练集: {len(train)}, 测试集: {len(test)}, 特征数: {len(feature_cols)}")
    _save_feature_metadata(feature_cols, meta)

    xgb_metrics, xgb_details = train_family("XGBoost", train, test, feature_cols)
    lgb_metrics, lgb_details = train_family("LightGBM", train, test, feature_cols)
    metrics_frame = pd.concat([xgb_metrics, lgb_metrics], ignore_index=True)
    details_frame = pd.concat([xgb_details, lgb_details], ignore_index=True)
    metrics_frame.to_csv(OUT / "boosting_metrics.csv", index=False, encoding="utf-8-sig")
    details_frame.to_csv(OUT / "boosting_predictions.csv", index=False, encoding="utf-8-sig")

    print("\n梯度提升树结果汇总:")
    print(metrics_frame.pivot_table(index="预见期（天）", columns="模型", values="R²").round(4))
    print(f"总耗时: {time.time() - started:.1f}s")
    return metrics_frame


if __name__ == "__main__":
    main()
