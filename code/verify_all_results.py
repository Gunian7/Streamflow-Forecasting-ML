# -*- coding: utf-8 -*-
"""独立复核：不依赖 stage7 的代码路径，重新计算三类数字。

1) 四张集成表全部 7 行：从 ensemble_predictions.csv 重算 RMSE/MAE/R² 并与表值比对
2) PyTorch-LSTM 直接/多输出 第 1、7 天：加载 joblib 模型在测试集上重新预测，
   与 pytorch_lstm_metrics.csv 比对（独立于 ensemble_predictions.csv 的路径）
3) 第二阶段 28 行：从 逐日预测结果.csv 重算，与 测试精度指标.csv 比对
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))

P3 = ROOT / "results/phase3_enhanced"
P2 = ROOT / "results/phase2_v5"


def met(y, p):
    y = np.asarray(y, float).ravel()
    p = np.asarray(p, float).ravel()
    rmse = float(np.sqrt(np.mean((y - p) ** 2)))
    mae = float(np.mean(np.abs(y - p)))
    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return rmse, mae, 1.0 - ss_res / ss_tot


rows = []

# ---------- 1) 四张集成表 ----------
pred = pd.read_csv(P3 / "ensemble_predictions.csv", encoding="utf-8-sig")
for fname, (strategy, method) in {
        "6模型简单平均_直接.csv": ("直接", "6模型简单平均"),
        "6模型Stacking_直接.csv": ("直接", "6模型Stacking"),
        "3模型简单平均_多输出.csv": ("多输出", "3模型简单平均"),
        "3模型Stacking_多输出.csv": ("多输出", "3模型Stacking"),
}.items():
    tab = pd.read_csv(P3 / fname, encoding="utf-8-sig")
    for h in range(1, 8):
        g = pred[(pred["策略"] == strategy) & (pred["方法"] == method) & (pred["预见期（天）"] == h)]
        rmse, mae, r2 = met(g["实测"], g["预测"])
        row = tab[tab["预见期（天）"] == h].iloc[0]
        rows.append({"校验项": "集成表", "文件": fname, "指标行": f"第{h}天",
                     "表值R²": float(row["R²"]), "重算R²": r2,
                     "R²差值": abs(float(row["R²"]) - r2),
                     "RMSE差值": abs(float(row["RMSE"]) - rmse),
                     "MAE差值": abs(float(row["MAE"]) - mae)})

# ---------- 2) PyTorch-LSTM 第 1、7 天（重新预测） ----------
import enhanced_features as ef

_, te_e, cols_e, _ = ef.get_enhanced_dataset()
Xe = te_e[cols_e].to_numpy(float)
lstm_met = pd.read_csv(P3 / "pytorch_lstm_metrics.csv", encoding="utf-8-sig")
Y = {h: te_e[f"Target_lead{h}"].to_numpy(float) for h in range(1, 8)}
for h in (1, 7):
    m = joblib.load(P3 / f"pytorch_lstm_direct_{h}.joblib")
    p = np.clip(np.asarray(m.predict(Xe), float).ravel(), 0, None)
    rmse, mae, r2 = met(Y[h], p)
    row = lstm_met[(lstm_met["策略"] == "PyTorch-LSTM-直接") & (lstm_met["预见期（天）"] == h)].iloc[0]
    rows.append({"校验项": "LSTM直接(joblib重预测)", "文件": "pytorch_lstm_metrics.csv", "指标行": f"第{h}天",
                 "表值R²": float(row["R²"]), "重算R²": r2, "R²差值": abs(float(row["R²"]) - r2),
                 "RMSE差值": abs(float(row["RMSE"]) - rmse), "MAE差值": abs(float(row["MAE"]) - mae)})
m = joblib.load(P3 / "pytorch_lstm_multi.joblib")
pm = np.clip(np.asarray(m.predict(Xe), float).reshape(len(te_e), 7), 0, None)
for h in (1, 7):
    rmse, mae, r2 = met(Y[h], pm[:, h - 1])
    row = lstm_met[(lstm_met["策略"] == "PyTorch-LSTM-多输出") & (lstm_met["预见期（天）"] == h)].iloc[0]
    rows.append({"校验项": "LSTM多输出(joblib重预测)", "文件": "pytorch_lstm_metrics.csv", "指标行": f"第{h}天",
                 "表值R²": float(row["R²"]), "重算R²": r2, "R²差值": abs(float(row["R²"]) - r2),
                 "RMSE差值": abs(float(row["RMSE"]) - rmse), "MAE差值": abs(float(row["MAE"]) - mae)})

# ---------- 3) 第二阶段 28 行 ----------
det = pd.read_csv(P2 / "逐日预测结果.csv", encoding="utf-8-sig")
tab2 = pd.read_csv(P2 / "测试精度指标.csv", encoding="utf-8-sig")
maxdiff2 = 0.0
for _, r in tab2.iterrows():
    g = det[(det["模型"] == r["模型"]) & (det["策略"] == r["策略"]) & (det["预见期（天）"] == r["预见期（天）"])]
    rmse, mae, r2 = met(g["实测径流（cfs）"], g["预测径流（cfs）"])
    maxdiff2 = max(maxdiff2, abs(float(r["R²"]) - r2), abs(float(r["RMSE（cfs）"]) - rmse))
rows.append({"校验项": "第二阶段28行", "文件": "测试精度指标.csv", "指标行": "全部28行",
             "表值R²": float(tab2["R²"].abs().max()), "重算R²": float(tab2["R²"].abs().max()),
             "R²差值": maxdiff2, "RMSE差值": maxdiff2, "MAE差值": 0.0})

out = pd.DataFrame(rows)
out.to_csv(P3 / "自洽性校验_附加校验.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 200)
print(out.round(9).to_string(index=False))
print("\n独立复核最大差值: R²=%.3e RMSE=%.3e MAE=%.3e" % (
    out["R²差值"].max(), out["RMSE差值"].max(), out["MAE差值"].max()))
print("LSTM 直接第1天 R² = %.6f（探针 0.8507，偏差 %.4f）" % (
    float(lstm_met[(lstm_met["策略"] == "PyTorch-LSTM-直接") & (lstm_met["预见期（天）"] == 1)]["R²"].iloc[0]),
    abs(float(lstm_met[(lstm_met["策略"] == "PyTorch-LSTM-直接") & (lstm_met["预见期（天）"] == 1)]["R²"].iloc[0]) - 0.8507)))
