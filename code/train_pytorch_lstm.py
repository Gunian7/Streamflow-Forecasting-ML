"""Train real PyTorch LSTMs and compare them with the PseudoLSTM baseline."""
from __future__ import annotations

import copy
import json
import os
import random
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from enhanced_features import get_enhanced_dataset
from phase3_lstm import PseudoLSTM
from pytorch_lstm import LSTMRegressor, LSTMTrainer

OUT = ROOT / "results" / "phase3_enhanced"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42
HORIZONS = range(1, 8)


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(4, os.cpu_count() or 1))


def metrics(y, prediction):
    y = np.asarray(y, dtype=float).ravel()
    prediction = np.asarray(prediction, dtype=float).ravel()
    ss_res = float(np.sum((y - prediction) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return {
        "RMSE": float(np.sqrt(np.mean((y - prediction) ** 2))),
        "MAE": float(np.mean(np.abs(y - prediction))),
        "R²": float(1.0 - ss_res / ss_tot),
    }


def make_windows(frame, feature_cols, target_cols, sequence_length):
    """Create causal windows ending at each row in a contiguous daily frame."""
    values = frame[feature_cols].to_numpy(dtype=np.float32)
    targets = frame[target_cols].to_numpy(dtype=np.float32)
    dates = pd.to_datetime(frame["Date"]).to_numpy()
    if len(frame) < sequence_length:
        raise ValueError("Not enough rows to create a sequence window")
    expected = np.timedelta64(1, "D")
    windows, labels, kept_dates = [], [], []
    for end in range(sequence_length - 1, len(frame)):
        if dates[end] - dates[end - sequence_length + 1] != np.timedelta64(sequence_length - 1, "D"):
            continue
        windows.append(values[end - sequence_length + 1:end + 1])
        labels.append(targets[end])
        kept_dates.append(dates[end])
    if not windows:
        raise ValueError("No contiguous daily sequence windows were created")
    return np.stack(windows), np.stack(labels), np.asarray(kept_dates)


def split_validation(X, y, validation_fraction=0.2):
    n_val = max(1, int(len(X) * validation_fraction))
    if n_val >= len(X):
        raise ValueError("Validation split leaves no training samples")
    return X[:-n_val], X[-n_val:], y[:-n_val], y[-n_val:]


def build_data(sequence_length):
    train, test, feature_cols, meta = get_enhanced_dataset()
    target_cols = [f"Target_lead{h}" for h in HORIZONS]
    X_train, Y_train, train_dates = make_windows(train, feature_cols, target_cols, sequence_length)
    X_test, Y_test, test_dates = make_windows(test, feature_cols, target_cols, sequence_length)
    return X_train, Y_train, X_test, Y_test, feature_cols, meta, train_dates, test_dates


def save_trainer(trainer, path, config):
    """Save a portable checkpoint plus the complete fitted trainer for reuse."""
    payload = {
        "config": config,
        "model_state_dict": trainer.model.state_dict(),
        "scaler_X": trainer.scaler_X,
        "scaler_y": trainer.scaler_y,
        "history": trainer.history,
    }
    torch.save(payload, path.with_suffix(".pt"))
    joblib.dump(trainer, path.with_suffix(".joblib"), compress=3)


def train_one(X_train, y_train, X_test, model_config, epochs, batch_size,
              early_stopping_patience, seed=SEED):
    set_seed(seed)
    X_tr, X_val, y_tr, y_val = split_validation(X_train, y_train)
    model = LSTMRegressor(**model_config)
    trainer = LSTMTrainer(model, device="cuda" if torch.cuda.is_available() else "cpu",
                          lr=1e-3, scheduler_patience=5)
    trainer.fit(X_tr, y_tr, X_val, y_val, epochs=epochs, batch_size=batch_size,
                early_stopping_patience=early_stopping_patience, verbose=False)
    prediction = trainer.predict(X_test)
    return trainer, prediction


def train_direct(X_train, Y_train, X_test, Y_test, feature_count, args):
    results, details = [], []
    print("\n真实 PyTorch LSTM 直接策略训练:", flush=True)
    for horizon in HORIZONS:
        config = {
            "input_dim": feature_count,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "output_dim": 1,
            "dropout": args.dropout,
            "bidirectional": args.bidirectional,
        }
        trainer, prediction = train_one(
            X_train, Y_train[:, horizon - 1], X_test, config,
            args.epochs, args.batch_size, args.early_stopping_patience,
            seed=SEED + horizon,
        )
        prediction = np.clip(prediction, 0, None)
        observed = Y_test[:, horizon - 1]
        score = metrics(observed, prediction)
        results.append({"策略": "PyTorch-LSTM-直接", "预见期（天）": horizon, **score})
        details.extend({"策略": "PyTorch-LSTM-直接", "预见期（天）": horizon,
                        "样本序号": i, "实测": float(obs), "预测": float(pred)}
                       for i, (obs, pred) in enumerate(zip(observed, prediction)))
        save_trainer(trainer, OUT / f"pytorch_lstm_direct_{horizon}", config)
        print(f"  第{horizon}天: R²={score['R²']:.4f}, RMSE={score['RMSE']:.1f}", flush=True)
    return pd.DataFrame(results), pd.DataFrame(details)


def train_multi(X_train, Y_train, X_test, Y_test, feature_count, args):
    print("\n真实 PyTorch LSTM 多输出策略训练:", flush=True)
    config = {
        "input_dim": feature_count,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "output_dim": len(HORIZONS),
        "dropout": args.dropout,
        "bidirectional": args.bidirectional,
    }
    trainer, prediction = train_one(
        X_train, Y_train, X_test, config, args.epochs, args.batch_size,
        args.early_stopping_patience, seed=SEED,
    )
    prediction = np.clip(prediction, 0, None)
    results, details = [], []
    for horizon in HORIZONS:
        observed = Y_test[:, horizon - 1]
        pred_h = prediction[:, horizon - 1]
        score = metrics(observed, pred_h)
        results.append({"策略": "PyTorch-LSTM-多输出", "预见期（天）": horizon, **score})
        details.extend({"策略": "PyTorch-LSTM-多输出", "预见期（天）": horizon,
                        "样本序号": i, "实测": float(obs), "预测": float(pred)}
                       for i, (obs, pred) in enumerate(zip(observed, pred_h)))
        print(f"  第{horizon}天: R²={score['R²']:.4f}, RMSE={score['RMSE']:.1f}", flush=True)
    save_trainer(trainer, OUT / "pytorch_lstm_multi", config)
    return pd.DataFrame(results), pd.DataFrame(details)


def pseudo_baseline(X_train, Y_train, X_test, Y_test):
    """Evaluate the existing PseudoLSTM on the same enhanced window endpoint rows."""
    results = []
    for horizon in HORIZONS:
        model = PseudoLSTM(hidden_layer_sizes=(64, 32), alpha=0.001,
                           max_iter=500, random_state=SEED + horizon)
        flat_train = X_train.reshape(len(X_train), -1)
        flat_test = X_test.reshape(len(X_test), -1)
        model.fit(flat_train, Y_train[:, horizon - 1])
        pred = np.clip(model.predict(flat_test), 0, None)
        results.append({"策略": "PseudoLSTM-增强窗口", "预见期（天）": horizon,
                        **metrics(Y_test[:, horizon - 1], pred)})
    return pd.DataFrame(results)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-length", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--early-stopping-patience", type=int, default=15)
    parser.add_argument("--skip-baseline", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    start = time.time()
    set_seed()
    X_train, Y_train, X_test, Y_test, feature_cols, meta, train_dates, test_dates = build_data(args.sequence_length)
    print("=" * 74)
    print("增强版：真实 PyTorch LSTM 训练")
    print(f"训练窗口: {X_train.shape}, 测试窗口: {X_test.shape}, 特征数: {len(feature_cols)}")
    print(f"设备: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    direct, direct_details = train_direct(X_train, Y_train, X_test, Y_test,
                                          len(feature_cols), args)
    multi, multi_details = train_multi(X_train, Y_train, X_test, Y_test,
                                       len(feature_cols), args)
    all_results = pd.concat([direct, multi], ignore_index=True)
    all_details = pd.concat([direct_details, multi_details], ignore_index=True)
    all_results.to_csv(OUT / "pytorch_lstm_metrics.csv", index=False, encoding="utf-8-sig")
    all_details.to_csv(OUT / "pytorch_lstm_predictions.csv", index=False, encoding="utf-8-sig")

    comparison = all_results.copy()
    if not args.skip_baseline:
        baseline = pseudo_baseline(X_train, Y_train, X_test, Y_test)
        baseline.to_csv(OUT / "pseudolstm_enhanced_metrics.csv", index=False, encoding="utf-8-sig")
        comparison = pd.concat([baseline, comparison], ignore_index=True)
    comparison.to_csv(OUT / "pytorch_lstm_comparison.csv", index=False, encoding="utf-8-sig")

    print("\nPyTorch LSTM R² 汇总:")
    print(all_results.pivot(index="预见期（天）", columns="策略", values="R²").round(4).to_string())
    day1 = all_results[(all_results["策略"] == "PyTorch-LSTM-直接") &
                       (all_results["预见期（天）"] == 1)].iloc[0]
    old = pd.read_csv(ROOT / "results" / "phase3_lstm" / "lstm_metrics.csv")
    old_day1 = old[(old["策略"] == "直接") & (old["预见期（天）"] == 1)].iloc[0]
    print("\n与原版 PseudoLSTM 对比（第1天）:")
    print(f"  PseudoLSTM:   R²={old_day1['R²']:.4f}, RMSE={old_day1['RMSE']:.1f}")
    print(f"  PyTorch LSTM: R²={day1['R²']:.4f}, RMSE={day1['RMSE']:.1f}")
    print(f"  提升: ΔR²={day1['R²'] - old_day1['R²']:+.4f}")
    print(f"\n目标检查: 第1天 R² {'已达成' if day1['R²'] >= 0.79 else '未达成'} (阈值 0.79)")
    metadata = {"sequence_length": args.sequence_length, "feature_count": len(feature_cols),
                "train_windows": len(X_train), "test_windows": len(X_test),
                "feature_columns": feature_cols, "data_meta": meta,
                "train_first_window_date": str(train_dates[0]),
                "test_first_window_date": str(test_dates[0]), "args": vars(args)}
    (OUT / "pytorch_lstm_run.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n总耗时: {time.time() - start:.1f}s")
    print(f"结果保存至: {OUT}")
    return all_results


if __name__ == "__main__":
    main()
