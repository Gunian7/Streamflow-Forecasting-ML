"""第三阶段统一编排脚本（子任务 1-7 + 4 张中文图表 + 数字自洽校验）

顺序执行：
  stage1  PyTorch-LSTM 网格搜索（TimeSeriesSplit，仅训练段）
  stage2  最优参数重训（直接 1-7 天 + 多输出）
  stage3  XGBoost / LightGBM 直接策略基学习器
  stage4  PseudoLSTM（sklearn 版）直接 + 多输出基学习器
  stage5  集成：直接策略 6 模型简单平均 / Stacking；多输出策略 3 模型简单平均 / Stacking
  stage6  4 张中文图表
  stage7  数字自洽校验

用法：python code/phase3_run_all.py [all|stage1|...|stage7]
"""
from __future__ import annotations

import argparse
import ast
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent
OUT = ROOT / "results" / "phase3_enhanced"
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

# 数据源统一为按 Daymet v4 官方日值对齐的修正数据集 01047000_fixed.csv。
# step2_feature_engineering 在 import 时读取 RUNOFF_DATA_PATH，必须先于任何数据模块设定；
# enhanced_features 用 setdefault，因此这里的显式赋值优先生效。
_DATA = ROOT / "01047000_fixed.csv"
assert _DATA.exists(), f"原始数据缺失: {_DATA}"
os.environ["RUNOFF_DATA_PATH"] = str(_DATA)
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

warnings.filterwarnings("ignore")
HORIZONS = list(range(1, 8))
GLOBAL_START = time.time()
DEADLINE_S = 3300


def metrics(y, p):
    y = np.asarray(y, float).ravel()
    p = np.asarray(p, float).ravel()
    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return {
        "RMSE": float(np.sqrt(np.mean((y - p) ** 2))),
        "MAE": float(np.mean(np.abs(y - p))),
        "R²": float(1.0 - ss_res / ss_tot),
    }


def guard(label):
    used = time.time() - GLOBAL_START
    if used > DEADLINE_S:
        raise SystemExit(f"[超时保护] 已用 {used/60:.1f} 分钟，停止于 {label}")
    print(f"[时长] {label}: 累计 {used/60:.1f} 分钟", flush=True)


def run_script(name, timeout=2400):
    t0 = time.time()
    print(f"\n>>> 执行 {name}", flush=True)
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run(
        [sys.executable, str(CODE / name)], cwd=str(ROOT), capture_output=True,
        text=True, encoding="utf-8", errors="replace", timeout=timeout, env=env,
    )
    dt = time.time() - t0
    if proc.returncode != 0:
        print(proc.stdout[-4000:])
        print(proc.stderr[-4000:])
        raise RuntimeError(f"{name} 退出码 {proc.returncode}（{dt:.0f}s）")
    tail = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()][-6:]
    print("\n".join(tail))
    print(f"<<< {name} 完成，耗时 {dt:.1f}s", flush=True)
    return dt


def load_frames():
    """两套特征按 Date 取交集，保证所有基学习器在同一训练段/预测集上。"""
    import enhanced_features as ef
    import phase2_v5_data as d5

    train_b, test_b, cols_b, _ = d5.build_dataset_v5()
    train_e, test_e, cols_e, _ = ef.get_enhanced_dataset()

    tr_b = train_b[train_b["Date"].isin(set(train_e["Date"]))].reset_index(drop=True)
    te_b = test_b[test_b["Date"].isin(set(test_e["Date"]))].reset_index(drop=True)
    assert len(tr_b) == len(train_e), "训练段日期对齐失败"
    assert len(te_b) == len(test_e), "测试段日期对齐失败"
    assert (tr_b["Date"].to_numpy() == train_e["Date"].to_numpy()).all(), "训练段日期顺序不一致"
    assert (te_b["Date"].to_numpy() == test_e["Date"].to_numpy()).all(), "测试段日期顺序不一致"
    print(f"[数据] 训练 {len(tr_b)} 行 / 测试 {len(te_b)} 行；基础特征 {len(cols_b)} 维，增强特征 {len(cols_e)} 维", flush=True)
    return {
        "train_base": tr_b, "test_base": te_b, "cols_base": list(cols_b),
        "train_enh": train_e, "test_enh": test_e, "cols_enh": list(cols_e),
    }


def _register_main_alias():
    """phase3_lstm.py 以 __main__ 身份执行，其 pickle 模块路径为 __main__.PseudoLSTM。"""
    import phase3_lstm

    sys.modules["__main__"].PseudoLSTM = phase3_lstm.PseudoLSTM
    return phase3_lstm


def _clone_best(sk_model):
    from sklearn.base import clone

    est = sk_model.best_estimator_ if hasattr(sk_model, "best_estimator_") else sk_model
    return clone(est)


def _refit_booster(model):
    p = dict(model.get_params())
    p.pop("early_stopping_rounds", None)
    p.pop("callbacks", None)
    best_it = getattr(model, "best_iteration", None)
    if best_it is None:
        best_it = getattr(model, "best_iteration_", None)
    if best_it:
        p["n_estimators"] = int(best_it) + 1
    return type(model)(**p)


def _lstm_params(grid, task):
    row = grid[grid["task"] == task].iloc[0]
    return {
        "hidden_dim": int(row["best_hidden_dim"]),
        "num_layers": int(row["best_num_layers"]),
        "lr": float(row["best_lr"]),
        "dropout": float(row["best_dropout"]),
    }


def _fit_lstm(X_fit, y_fit, params, output_dim, device):
    import torch

    from pytorch_lstm import LSTMRegressor, LSTMTrainer

    torch.manual_seed(42)
    model = LSTMRegressor(
        input_dim=X_fit.shape[1], hidden_dim=params["hidden_dim"],
        num_layers=params["num_layers"], output_dim=output_dim, dropout=params["dropout"],
    )
    trainer = LSTMTrainer(model, device=device, lr=params["lr"])
    n_val = max(32, int(len(X_fit) * 0.1))
    trainer.fit(X_fit[:-n_val], y_fit[:-n_val], X_fit[-n_val:], y_fit[-n_val:],
                epochs=120, batch_size=32, early_stopping_patience=15, verbose=False)
    return trainer


# --------------------------------------------------------------------------- #
# stage1-4：调用既有脚本
# --------------------------------------------------------------------------- #
def stage1():
    return run_script("pytorch_lstm_grid_search.py", timeout=2400)


def stage2():
    return run_script("retrain_pytorch_lstm_final.py", timeout=1800)


def stage3():
    return run_script("train_boosting.py", timeout=1800)


def stage4():
    return run_script("phase3_lstm.py", timeout=1800)


# --------------------------------------------------------------------------- #
# stage5：集成学习
# --------------------------------------------------------------------------- #
def _fit_ridge_meta(meta, y):
    """标准化 + 时序交叉验证选 alpha 的 Ridge 元学习器（Ridge 为正则线性回归）。"""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(meta)
    Z = scaler.transform(meta)
    best_alpha, best_err = 1.0, float("inf")
    for alpha in [1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0, 10000.0]:
        errs = []
        for tr_i, va_i in TimeSeriesSplit(n_splits=3).split(Z):
            if len(tr_i) < 30:
                continue
            r = Ridge(alpha=alpha).fit(Z[tr_i], np.asarray(y)[tr_i])
            errs.append(float(np.mean((np.asarray(y)[va_i] - r.predict(Z[va_i])) ** 2)))
        if errs and np.mean(errs) < best_err:
            best_alpha, best_err = alpha, float(np.mean(errs))
    model = Ridge(alpha=best_alpha).fit(Z, y)
    return model, scaler, best_alpha


def stage5():
    import torch
    from sklearn.linear_model import Ridge

    device = "cuda" if torch.cuda.is_available() else "cpu"
    frames = load_frames()
    tr_b, te_b, cols_b = frames["train_base"], frames["test_base"], frames["cols_base"]
    tr_e, te_e, cols_e = frames["train_enh"], frames["test_enh"], frames["cols_enh"]

    Xb_train = tr_b[cols_b].to_numpy(float)
    Xb_test = te_b[cols_b].to_numpy(float)
    Xe_train = tr_e[cols_e].to_numpy(float)
    Xe_test = te_e[cols_e].to_numpy(float)
    dates = te_e["Date"].dt.strftime("%Y-%m-%d").to_numpy()
    report_dates = pd.to_datetime(te_e["Date"]).reset_index(drop=True)
    Y_te = {h: te_e[f"Target_lead{h}"].to_numpy(float) for h in HORIZONS}
    Y_tr = {h: tr_e[f"Target_lead{h}"].to_numpy(float) for h in HORIZONS}

    grid = pd.read_csv(OUT / "pytorch_lstm_grid_summary.csv")
    lstm_dir_params = {h: _lstm_params(grid, f"direct_day{h}") for h in HORIZONS}
    lstm_mul_params = _lstm_params(grid, "multi_output")

    pseudo_grid = pd.read_csv(ROOT / "results/phase3_lstm/lstm_grid_summary.csv")
    pseudo_params = {
        str(r["task"]): {
            "hidden_layer_sizes": ast.literal_eval(str(r["best_hidden_layers"])),
            "alpha": float(r["best_alpha"]),
        }
        for _, r in pseudo_grid.iterrows()
    }
    phase3_lstm = _register_main_alias()

    preds = []

    def add(strategy, method, h, obs, pred):
        pred = np.clip(np.asarray(pred, float).ravel(), 0, None)
        preds.append(pd.DataFrame({
            "策略": strategy, "方法": method, "预见期（天）": h,
            # 起报日=特征所在日；目标日=起报日+预见期，指标与曲线均以目标日为准
            "起报日": dates,
            "目标日": (report_dates + pd.Timedelta(days=int(h))).dt.strftime("%Y-%m-%d").to_numpy(),
            "实测": np.asarray(obs, float).ravel(), "预测": pred,
        }))

    names_direct = ["ANN", "RF", "XGBoost", "LightGBM", "PyTorch-LSTM", "PseudoLSTM"]
    base_direct = {n: {} for n in names_direct}
    for h in HORIZONS:
        for name in ["ANN", "RF"]:
            # 兼容性修复：phase2_v5 的 ANN/RF 以 log1p 目标训练，predict 输出为对数空间，需 expm1 还原
            m = joblib.load(ROOT / "results/phase2_v5" / f"{name}_{h}.joblib")
            p_te = np.expm1(np.asarray(m.predict(Xb_test), float))
            base_direct[name][h] = p_te
            add("直接", name, h, Y_te[h], p_te)
        for name, fname in [("XGBoost", "xgboost"), ("LightGBM", "lightgbm")]:
            m = joblib.load(OUT / f"{fname}_direct_{h}.joblib")
            p_te = m.predict(Xe_test)
            base_direct[name][h] = p_te
            add("直接", name, h, Y_te[h], p_te)
        tr_lstm = joblib.load(OUT / f"pytorch_lstm_direct_{h}.joblib")
        p_te = tr_lstm.predict(Xe_test)
        base_direct["PyTorch-LSTM"][h] = p_te
        add("直接", "PyTorch-LSTM", h, Y_te[h], p_te)
        m = joblib.load(ROOT / "results/phase3_lstm" / f"lstm_direct_{h}.joblib")
        p_te = m.predict(Xb_test)
        base_direct["PseudoLSTM"][h] = p_te
        add("直接", "PseudoLSTM", h, Y_te[h], p_te)
    print("[预测] 直接策略 6 个基学习器测试集预测完成", flush=True)

    multi_names = ["ANN", "RF", "PyTorch-LSTM"]
    multi_test = {}
    for name in multi_names:
        if name == "ANN":
            p_te = np.expm1(np.asarray(joblib.load(ROOT / "results/phase2_v5/ANN_multi.joblib").predict(Xb_test), float))
        elif name == "RF":
            p_te = np.expm1(np.asarray(joblib.load(ROOT / "results/phase2_v5/RF_multi.joblib").predict(Xb_test), float))
        else:
            p_te = joblib.load(OUT / "pytorch_lstm_multi.joblib").predict(Xe_test)
        multi_test[name] = np.asarray(p_te, float).reshape(len(te_e), 7)
        for h in HORIZONS:
            add("多输出", name, h, Y_te[h], multi_test[name][:, h - 1])
    print("[预测] 多输出策略 3 个基学习器测试集预测完成", flush=True)

    # 简单平均：与 Stacking 共用同一批基学习器与同一预测集
    for h in HORIZONS:
        mat = np.column_stack([np.asarray(base_direct[n][h], float).ravel() for n in names_direct])
        add("直接", f"{len(names_direct)}模型简单平均", h, Y_te[h], mat.mean(axis=1))
    for h in HORIZONS:
        mat = np.column_stack([multi_test[n][:, h - 1] for n in multi_names])
        add("多输出", f"{len(multi_names)}模型简单平均", h, Y_te[h], mat.mean(axis=1))

    # ---------------- Stacking（时序留出块生成元特征 + Ridge 元学习器） ----------------
    n_tr = len(tr_e)
    n_hold = int(round(n_tr * 0.2))
    hold_start = n_tr - n_hold
    print(f"[Stacking] 训练段 {n_tr} 行；元特征留出块=最后 {n_hold} 行（索引 {hold_start} 起）", flush=True)

    def split_for(h):
        fit_end = hold_start - h
        assert fit_end > 200, "留出块过大"
        return slice(0, fit_end), slice(hold_start, n_tr)

    meta_direct = {h: np.zeros((n_hold, len(names_direct))) for h in HORIZONS}
    for h in HORIZONS:
        sl_fit, sl_ho = split_for(h)
        y_fit = Y_tr[h][sl_fit]
        for k, name in enumerate(names_direct):
            t0 = time.time()
            if name in ["ANN", "RF"]:
                m = _clone_best(joblib.load(ROOT / "results/phase2_v5" / f"{name}_{h}.joblib")).fit(
                    Xb_train[sl_fit], np.log1p(y_fit))
                meta_direct[h][:, k] = np.expm1(np.asarray(m.predict(Xb_train[sl_ho]), float)).ravel()
            elif name in ["XGBoost", "LightGBM"]:
                fname = "xgboost" if name == "XGBoost" else "lightgbm"
                m = _refit_booster(joblib.load(OUT / f"{fname}_direct_{h}.joblib")).fit(
                    Xe_train[sl_fit], y_fit)
                meta_direct[h][:, k] = m.predict(Xe_train[sl_ho])
            elif name == "PyTorch-LSTM":
                m = _fit_lstm(Xe_train[sl_fit], y_fit, lstm_dir_params[h], 1, device)
                meta_direct[h][:, k] = m.predict(Xe_train[sl_ho])
            else:
                # phase3_lstm 只对第 1 天与多输出做网格搜索，其余预见期沿用第 1 天最优参数
                p_params = pseudo_params.get(f"direct_day{h}", pseudo_params["direct_day1"])
                m = phase3_lstm.PseudoLSTM(**p_params, random_state=42).fit(
                    Xb_train[sl_fit], y_fit)
                meta_direct[h][:, k] = m.predict(Xb_train[sl_ho])
            print(f"  留出块重训 {name} 第{h}天 -> {time.time()-t0:.1f}s", flush=True)

    sl_fit, sl_ho = split_for(7)
    Y_fit_multi = np.column_stack([Y_tr[h][sl_fit] for h in HORIZONS])
    meta_multi = {h: np.zeros((n_hold, len(multi_names))) for h in HORIZONS}
    for k, name in enumerate(multi_names):
        t0 = time.time()
        if name in ["ANN", "RF"]:
            m = _clone_best(joblib.load(ROOT / "results/phase2_v5" / f"{name}_multi.joblib")).fit(
                Xb_train[sl_fit], np.log1p(Y_fit_multi))
            p_ho = np.expm1(np.asarray(m.predict(Xb_train[sl_ho]), float))
        else:
            m = _fit_lstm(Xe_train[sl_fit], Y_fit_multi, lstm_mul_params, 7, device)
            p_ho = m.predict(Xe_train[sl_ho])
        p_ho = np.asarray(p_ho, float).reshape(n_hold, 7)
        for h in HORIZONS:
            meta_multi[h][:, k] = p_ho[:, h - 1]
        print(f"  留出块重训 {name} 多输出 -> {time.time()-t0:.1f}s", flush=True)

    y_hold = {h: Y_tr[h][sl_ho] for h in HORIZONS}
    stack_direct_weights = {}
    stack_direct_alpha = {}
    for h in HORIZONS:
        r, scaler, alpha = _fit_ridge_meta(meta_direct[h], y_hold[h])
        test_mat = np.column_stack([np.asarray(base_direct[n][h], float).ravel() for n in names_direct])
        add("直接", f"{len(names_direct)}模型Stacking", h, Y_te[h],
            r.predict(scaler.transform(test_mat)))
        stack_direct_weights[h] = r.coef_.tolist()
        stack_direct_alpha[h] = alpha
        print(f"  直接 Stacking 第{h}天: alpha={alpha}", flush=True)

    multi_weights = {}
    multi_alpha = {}
    for h in HORIZONS:
        r, scaler, alpha = _fit_ridge_meta(meta_multi[h], y_hold[h])
        test_mat = np.column_stack([multi_test[n][:, h - 1] for n in multi_names])
        add("多输出", f"{len(multi_names)}模型Stacking", h, Y_te[h],
            r.predict(scaler.transform(test_mat)))
        multi_weights[h] = r.coef_.tolist()
        multi_alpha[h] = alpha
        print(f"  多输出 Stacking 第{h}天: alpha={alpha}", flush=True)

    # ---------------- 落盘 ----------------
    all_pred = pd.concat(preds, ignore_index=True)
    all_pred.to_csv(OUT / "ensemble_predictions.csv", index=False, encoding="utf-8-sig")

    def table(strategy, method):
        sub = all_pred[(all_pred["策略"] == strategy) & (all_pred["方法"] == method)]
        rows = []
        for h in HORIZONS:
            block = sub[sub["预见期（天）"] == h]
            m = metrics(block["实测"].to_numpy(), block["预测"].to_numpy())
            rows.append({"策略": strategy, "方法": method, "预见期（天）": h,
                         "RMSE": m["RMSE"], "MAE": m["MAE"], "R²": m["R²"]})
        return pd.DataFrame(rows)

    files = {
        "6模型简单平均_直接.csv": table("直接", "6模型简单平均"),
        "6模型Stacking_直接.csv": table("直接", "6模型Stacking"),
        "3模型简单平均_多输出.csv": table("多输出", "3模型简单平均"),
        "3模型Stacking_多输出.csv": table("多输出", "3模型Stacking"),
    }
    for fname, df in files.items():
        df.to_csv(OUT / fname, index=False, encoding="utf-8-sig")
        print(f"\n{fname}\n{df.round(4).to_string(index=False)}", flush=True)

    pd.DataFrame([
        {"策略": "直接", "预见期（天）": h, "ridge_alpha": stack_direct_alpha[h],
         **{f"w_{n}": stack_direct_weights[h][i] for i, n in enumerate(names_direct)}}
        for h in HORIZONS
    ]).to_csv(OUT / "stacking_直接_权重.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([
        {"策略": "多输出", "预见期（天）": h, "ridge_alpha": multi_alpha[h],
         **{f"w_{n}": multi_weights[h][i] for i, n in enumerate(multi_names)}}
        for h in HORIZONS
    ]).to_csv(OUT / "stacking_多输出_权重.csv", index=False, encoding="utf-8-sig")
    print(f"\n[保存] ensemble_predictions.csv 共 {len(all_pred)} 行", flush=True)

    base_dim, enh_dim = len(cols_b), len(cols_e)
    pd.DataFrame([
        {"基学习器": n, "策略": "直接", "特征维数": base_dim if n in ("ANN", "RF", "PseudoLSTM") else enh_dim}
        for n in names_direct
    ] + [{"基学习器": n, "策略": "多输出", "特征维数": base_dim if n in ("ANN", "RF") else enh_dim}
         for n in multi_names]).to_csv(OUT / "ensemble_members.csv", index=False, encoding="utf-8-sig")
    return all_pred


# --------------------------------------------------------------------------- #
# 指标汇总（供绘图与校验）
# --------------------------------------------------------------------------- #
def metrics_table(pred_df):
    rows = []
    for (strategy, method, h), g in pred_df.groupby(["策略", "方法", "预见期（天）"], sort=False):
        m = metrics(g["实测"].to_numpy(), g["预测"].to_numpy())
        rows.append({"策略": strategy, "方法": method, "预见期（天）": int(h), **m})
    return pd.DataFrame(rows)


def _setup_font():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    installed = {f.name for f in font_manager.fontManager.ttflist}
    chosen = next((c for c in ["SimSun", "Microsoft YaHei", "SimHei"] if c in installed), "DejaVu Sans")
    plt.rcParams["font.sans-serif"] = [chosen, "Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.unicode_minus"] = False
    # 实测 matplotlib 3.11.2：SimSun 在 legend（offsetbox）文本中丢字，标题与坐标轴正常。
    # 按“宋体优先、不可用回退微软雅黑”的规则，仅图例回退到 Microsoft YaHei。
    legend_family = "Microsoft YaHei" if (chosen == "SimSun" and "Microsoft YaHei" in installed) else chosen
    print(f"[字体] 正文 {chosen} / 图例 {legend_family}", flush=True)
    return plt, chosen, legend_family


def stage6():
    plt, font, legend_family = _setup_font()
    pred = pd.read_csv(OUT / "ensemble_predictions.csv", encoding="utf-8-sig")
    mt = metrics_table(pred)
    grid = pd.read_csv(OUT / "pytorch_lstm_grid_summary.csv", encoding="utf-8-sig")

    def series(strategy, method, col="R²"):
        s = mt[(mt["策略"] == strategy) & (mt["方法"] == method)].sort_values("预见期（天）")
        return s["预见期（天）"].to_numpy(), s[col].to_numpy()

    # ---- 图1：精度随预见期衰减（直接与多输出并列） ----
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), sharey=True)
    direct_methods = [("ANN", "RF", "XGBoost", "LightGBM", "PyTorch-LSTM", "PseudoLSTM",
                       "6模型简单平均", "6模型Stacking")]
    styles = ["o-", "s-", "^-", "v-", "D-", "P-", "*-", "X-"]
    for (ax, strategy, methods, title) in [
        (axes[0], "直接", direct_methods[0], "（a）直接策略（每个预见期独立建模）"),
        (axes[1], "多输出", ["ANN", "RF", "PyTorch-LSTM", "3模型简单平均", "3模型Stacking"],
         "（b）多输出策略（单模型一次输出 7 天）"),
    ]:
        for i, name in enumerate(methods):
            x, y = series(strategy, name)
            lw = 2.6 if "平均" in name or "Stacking" in name else 1.5
            ax.plot(x, y, styles[i % len(styles)], lw=lw, label=name, alpha=0.9)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("预见期（天）", fontsize=12)
        ax.set_xticks(HORIZONS)
        ax.grid(alpha=0.3, linestyle="--")
        ax.axhline(0, color="grey", lw=0.8, ls=":")
        ax.legend(ncol=2, loc="lower left", framealpha=0.85, prop={"family": legend_family, "size": 8})
    axes[0].set_ylabel("决定系数 R²", fontsize=12)
    fig.suptitle("图1  测试集（2004 年）精度随预见期衰减 — 直接策略与多输出策略并列", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    p1 = FIG / "fig1_精度随预见期衰减.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)

    # ---- 图2：策略对比柱状图（第 1 天与第 7 天） ----
    combo = [
        ("直接", "PyTorch-LSTM", "直接·PyTorch-LSTM"),
        ("直接", "6模型简单平均", "直接·简单平均"),
        ("直接", "6模型Stacking", "直接·Stacking"),
        ("多输出", "PyTorch-LSTM", "多输出·PyTorch-LSTM"),
        ("多输出", "3模型简单平均", "多输出·简单平均"),
        ("多输出", "3模型Stacking", "多输出·Stacking"),
    ]
    colors = ["#7f9fbf", "#4f81bd", "#1f4e79", "#e0a06a", "#c55a11", "#843c0c"]
    fig, ax = plt.subplots(figsize=(11.5, 5.4))
    width = 0.12
    for gi, h in enumerate([1, 7]):
        base = gi * 1.0
        for i, (strategy, method, label) in enumerate(combo):
            row = mt[(mt["策略"] == strategy) & (mt["方法"] == method) & (mt["预见期（天）"] == h)]
            val = float(row["R²"].iloc[0])
            xpos = base + (i - 2.5) * width
            ax.bar(xpos, val, width=width * 0.95, color=colors[i], label=label if gi == 0 else None,
                   edgecolor="black", linewidth=0.4)
            ax.text(xpos, val + 0.012, f"{val:.3f}", ha="center", va="bottom", fontsize=7.5, rotation=90)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["第 1 天预见期", "第 7 天预见期"], fontsize=12)
    ax.set_ylabel("决定系数 R²", fontsize=12)
    ax.axhline(0, color="black", lw=0.8)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    # 柱顶旋转数值标签需要留白，y 轴上下各留 12%
    _vals = [float(mt[(mt["策略"] == s) & (mt["方法"] == m) & (mt["预见期（天）"] == h)]["R²"].iloc[0])
             for h in (1, 7) for s, m, _ in combo]
    ax.set_ylim(min(0.0, min(_vals)) * 1.12, max(0.0, max(_vals)) * 1.12)
    ax.legend(ncol=3, loc="upper right", framealpha=0.9, prop={"family": legend_family, "size": 9})
    ax.set_title("图2  各策略在第 1 天与第 7 天的测试集精度对比", fontsize=14, fontweight="bold", pad=26)
    fig.tight_layout()
    p2 = FIG / "fig2_策略对比柱状图.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)

    # ---- 图3：2004 年测试段水文过程线（实测 vs 最优模型第 1 天预测） ----
    h1 = mt[mt["预见期（天）"] == 1].sort_values("R²", ascending=False)
    best = h1.iloc[0]
    date_col = "目标日" if "目标日" in pred.columns else "日期"
    best_pred = pred[(pred["策略"] == best["策略"]) & (pred["方法"] == best["方法"])
                     & (pred["预见期（天）"] == 1)].sort_values(date_col)
    fig, ax = plt.subplots(figsize=(13, 5.2))
    x = pd.to_datetime(best_pred[date_col])
    ax.plot(x, best_pred["实测"], color="#1f4e79", lw=2.0, label="实测径流")
    ax.plot(x, best_pred["预测"], color="#c55a11", lw=1.5, ls="-", alpha=0.9,
            label=f"最优模型预测（{best['策略']}·{best['方法']}）")
    ax.set_xlabel("目标日期（起报日 + 预见期）", fontsize=12)
    ax.set_ylabel("日径流量（cfs）", fontsize=12)
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend(loc="upper left", prop={"family": legend_family, "size": 11})
    ax.set_title(
        f"图3  2004 年测试段水文过程线（目标日期）：实测与最优模型第 1 天预测（R²={best['R²']:.4f}，RMSE={best['RMSE']:.1f} cfs）",
        fontsize=13.5, fontweight="bold")
    fig.autofmt_xdate(rotation=20)
    fig.tight_layout()
    p3 = FIG / "fig3_测试段水文过程线.png"
    fig.savefig(p3, dpi=150)
    plt.close(fig)

    # ---- 图4：网格搜索交叉验证误差对比 ----
    labels = [("直接第%d天" % i) if t.startswith("direct_day") else "多输出"
              for i, t in enumerate(grid["task"], start=1)]
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    colors4 = ["#4f81bd"] * 7 + ["#c55a11"]
    bars = ax.bar(range(len(grid)), grid["cv_rmse"], color=colors4, edgecolor="black", linewidth=0.5)
    for b, v in zip(bars, grid["cv_rmse"]):
        ax.text(b.get_x() + b.get_width() / 2, v + 8, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(len(grid)))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("交叉验证 RMSE（cfs）", fontsize=12)
    ax.set_xlabel("任务", fontsize=12)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_title("图4  PyTorch-LSTM 网格搜索（TimeSeriesSplit 3 折）最优组合的交叉验证误差对比",
                 fontsize=13.5, fontweight="bold")
    fig.tight_layout()
    p4 = FIG / "fig4_网格搜索交叉验证误差.png"
    fig.savefig(p4, dpi=150)
    plt.close(fig)

    out = []
    for p in [p1, p2, p3, p4]:
        out.append((str(p.resolve()), p.stat().st_size))
        print(f"[图] {p}  {p.stat().st_size} 字节", flush=True)
    return out


# --------------------------------------------------------------------------- #
# stage7：数字自洽校验
# --------------------------------------------------------------------------- #
def stage7():
    pred = pd.read_csv(OUT / "ensemble_predictions.csv", encoding="utf-8-sig")
    checks = []
    files = {
        "6模型简单平均_直接.csv": ("直接", "6模型简单平均"),
        "6模型Stacking_直接.csv": ("直接", "6模型Stacking"),
        "3模型简单平均_多输出.csv": ("多输出", "3模型简单平均"),
        "3模型Stacking_多输出.csv": ("多输出", "3模型Stacking"),
    }
    for fname, (strategy, method) in files.items():
        df = pd.read_csv(OUT / fname, encoding="utf-8-sig")
        for h in [1, 7]:
            csv_r2 = float(df[df["预见期（天）"] == h]["R²"].iloc[0])
            g = pred[(pred["策略"] == strategy) & (pred["方法"] == method) & (pred["预见期（天）"] == h)]
            rec = metrics(g["实测"].to_numpy(), g["预测"].to_numpy())["R²"]
            checks.append({
                "文件": fname, "策略": strategy, "方法": method, "预见期": h,
                "CSV_R²": csv_r2, "重算_R²": rec, "差值": abs(csv_r2 - rec),
                "行数": len(df), "通过": abs(csv_r2 - rec) < 1e-6,
            })
    lstm_metrics = pd.read_csv(OUT / "pytorch_lstm_metrics.csv", encoding="utf-8-sig")
    for strat_row, strategy in [("PyTorch-LSTM-直接", "直接"), ("PyTorch-LSTM-多输出", "多输出")]:
        csv_r2 = float(lstm_metrics[(lstm_metrics["策略"] == strat_row) & (lstm_metrics["预见期（天）"] == 1)]["R²"].iloc[0])
        g = pred[(pred["策略"] == strategy) & (pred["方法"] == "PyTorch-LSTM") & (pred["预见期（天）"] == 1)]
        rec = metrics(g["实测"].to_numpy(), g["预测"].to_numpy())["R²"]
        checks.append({
            "文件": "pytorch_lstm_metrics.csv", "策略": strategy, "方法": "PyTorch-LSTM", "预见期": 1,
            "CSV_R²": csv_r2, "重算_R²": rec, "差值": abs(csv_r2 - rec),
            "行数": len(lstm_metrics), "通过": abs(csv_r2 - rec) < 1e-6,
        })
    df = pd.DataFrame(checks)
    df.to_csv(OUT / "自洽性校验.csv", index=False, encoding="utf-8-sig")

    g = pd.read_csv(OUT / "pytorch_lstm_grid_summary.csv", encoding="utf-8-sig")
    print("=" * 78)
    print("验收自检")
    print("=" * 78)
    print(f"1) grid_summary 行数={len(g)}，列={list(g.columns)}")
    print(f"2) pytorch_lstm_metrics 行数={len(lstm_metrics)}，策略分布="
          f"{lstm_metrics['策略'].value_counts().to_dict()}")
    for fname in files:
        d = pd.read_csv(OUT / fname, encoding="utf-8-sig")
        print(f"3) {fname}: 行数={len(d)}，列={list(d.columns)}")
    print("5) 数字自洽（CSV 与 预测文件重算）:")
    print(df[["文件", "策略", "方法", "预见期", "CSV_R²", "重算_R²", "差值", "通过"]].to_string(index=False))
    mt = metrics_table(pred)
    best1 = mt[mt["预见期（天）"] == 1].sort_values("R²", ascending=False).iloc[0]
    print(f"6) 第1天最优 R² = {best1['R²']:.6f}（{best1['策略']}·{best1['方法']}），阈值 0.80 -> "
          f"{'达标' if best1['R²'] >= 0.80 else '未达标'}")
    figs = sorted(FIG.glob("*.png"))
    print(f"4) figures 数量={len(figs)} -> {[p.name for p in figs]}")
    return df


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", nargs="?", default="all",
                    choices=["all"] + [f"stage{i}" for i in range(1, 8)])
    args = ap.parse_args()
    order = ["stage1", "stage2", "stage3", "stage4", "stage5", "stage6", "stage7"]
    targets = order if args.stage == "all" else [args.stage]
    timings = {}
    for name in targets:
        guard(name)
        t0 = time.time()
        globals()[name]()
        timings[name] = time.time() - t0
        print(f"\n[{name}] 耗时 {timings[name]:.1f}s", flush=True)
    print("\n" + "=" * 78)
    print("各阶段耗时：")
    for k, v in timings.items():
        print(f"  {k}: {v:.1f}s")
    print(f"总耗时: {time.time() - GLOBAL_START:.1f}s")


if __name__ == "__main__":
    main()
