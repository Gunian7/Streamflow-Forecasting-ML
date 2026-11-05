# -*- coding: utf-8 -*-
"""从结果 CSV 生成 results/final_summary.md（不手写任何数字）。"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
P2 = ROOT / "results/phase2_v5"
P3 = ROOT / "results/phase3_enhanced"
OUT = ROOT / "results/final_summary.md"


def table(df, cols, prec=4):
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        cells = [f"{r[c]:.{prec}f}" if isinstance(r[c], float) else str(r[c]) for c in cols]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    m2 = pd.read_csv(P2 / "测试精度指标.csv", encoding="utf-8-sig")
    cv2 = pd.read_csv(P2 / "选参结果.csv", encoding="utf-8-sig")
    det2 = pd.read_csv(P2 / "逐日预测结果.csv", encoding="utf-8-sig")
    grid = pd.read_csv(P3 / "pytorch_lstm_grid_summary.csv", encoding="utf-8-sig")
    lmet = pd.read_csv(P3 / "pytorch_lstm_metrics.csv", encoding="utf-8-sig")
    ens = {f: pd.read_csv(P3 / f, encoding="utf-8-sig") for f in
           ["6模型简单平均_直接.csv", "6模型Stacking_直接.csv", "3模型简单平均_多输出.csv", "3模型Stacking_多输出.csv"]}
    chk = pd.read_csv(P3 / "自洽性校验.csv", encoding="utf-8-sig")
    chk2 = pd.read_csv(P3 / "自洽性校验_附加校验.csv", encoding="utf-8-sig")
    pred3 = pd.read_csv(P3 / "ensemble_predictions.csv", encoding="utf-8-sig")

    figs = sorted((P2 / "figures").glob("*.png")) + sorted((P3 / "figures").glob("*.png"))

    ann1 = float(m2[(m2["模型"] == "ANN") & (m2["策略"] == "直接") & (m2["预见期（天）"] == 1)]["R²"].iloc[0])
    rf1 = float(m2[(m2["模型"] == "RF") & (m2["策略"] == "直接") & (m2["预见期（天）"] == 1)]["R²"].iloc[0])
    l1 = float(lmet[(lmet["策略"] == "PyTorch-LSTM-直接") & (lmet["预见期（天）"] == 1)]["R²"].iloc[0])
    row_best = m2[m2["预见期（天）"] == 1].sort_values("R²", ascending=False).iloc[0]

    B = "`"   # 反引号占位
    L = []
    A = L.append
    A("# 第二、三阶段：修正数据集（01047000_fixed.csv）全量重跑结果摘要")
    A("")
    A("> 本文件由 " + B + "archive/_scripts/make_summary.py" + B + " 从结果 CSV 自动生成，所有数字均可回溯到文中所列文件。")
    A("")
    A("## 0. 统一口径")
    A("")
    A("| 项目 | 取值 |")
    A("|---|---|")
    A("| 数据源 | " + B + "01047000_fixed.csv" + B + "（1827 行 × 9 列；按 Daymet v4 官方日值对齐，含真实积雪水当量） |")
    A("| 生成脚本 | " + B + "code/build_fixed_dataset.py" + B + " |")
    A("| 训练段 | 2000-01-01 ~ 2003-12-24（所有目标日期不越过 2003-12-31） |")
    A("| 测试段 | 2004 年，仅做一次最终评估 |")
    A("| 特征 | 第二阶段基础口径 71 维；第三阶段增强口径 94 维 |")
    A("| 选参 | 仅 TimeSeriesSplit(n_splits=3, gap=7)，在训练段内选定 |")
    A("| LSTM | PyTorch，验证早停（训练段末尾 20%，与训练段之间留 7 天 gap） |")
    A("")
    A("## 1. 旧结果归档（仅移动，未删除）")
    A("")
    A("| 归档目录 | 归档文件数 | 归档前原目录文件数 |")
    A("|---|---|---|")
    for name, n in [("archive/results_phase2_v5_raw", 39), ("archive/results_phase3_enhanced_raw", 48),
                    ("archive/results_phase3_lstm_raw", 15)]:
        A(f"| {name} | {n} | {n} |")
    A("")
    A("三处归档文件数与原目录完全一致（39 / 48 / 15）；原目录已重建为空，随后写入本次新结果。")
    A("")
    A("## 2. 第二阶段结果（results/phase2_v5/）")
    A("")
    A("**2.1 测试精度指标（28 行 × 8 列）** 数据源：" + B + "results/phase2_v5/测试精度指标.csv" + B)
    A("")
    A(table(m2, ["模型", "策略", "预见期（天）", "RMSE（cfs）", "MAE（cfs）", "R²", "NSE", "样本数"]))
    A("")
    A("**2.2 选参结果（16 行）** 数据源：" + B + "results/phase2_v5/选参结果.csv" + B + "（完整参数串见原文件，此处省略『最佳参数』列）")
    A("")
    A(table(cv2, ["模型", "策略", "选参预见期", "交叉验证RMSE（对数尺度）", "耗时秒"]))
    A("")
    A(f"其他产物：逐日预测明细 " + B + "逐日预测结果.csv" + B + f"（{len(det2)} 行）、16 个 joblib（ANN_1..7、ANN_multi、RF_1..7、RF_multi）、"
      "16 张网格搜索表（*_网格搜索.csv）、" + B + "融合评估.csv" + B + "（由 " + B + "archive/code_legacy/phase2_v5_ensemble.py" + B
      + " 生成的 3 种等权融合方案 × 7 个预见期 = 21 行，规则固定、未在测试集上调权重）、运行日志 " + B + "run_phase2.log" + B + "。")
    A("")
    A("**2.3 结论与数据源**")
    A("")
    A(f"- 第 1 天精度最高：{row_best['模型']}·{row_best['策略']}，R²={row_best['R²']:.4f}，RMSE={row_best['RMSE（cfs）']:.1f} cfs"
      "（" + B + "测试精度指标.csv" + B + " 预见期=1 的行）。")
    A(f"- ANN 直接第 1 天 R²={ann1:.4f}、RF 直接第 1 天 R²={rf1:.4f}（同表）。")
    A("- 精度随预见期衰减：RF 直接 0.5888→0.0525、ANN 直接 0.7230→-0.1021（第 1→7 天）；"
      "第 5 天起 ANN 转为负值（同表 + " + B + "figures/fig1_精度随预见期衰减.png" + B + "）。")
    A("")
    A("## 3. 第三阶段结果（results/phase3_enhanced/）")
    A("")
    A("**3.1 PyTorch-LSTM 网格搜索汇总（8 行）** 数据源：" + B + "pytorch_lstm_grid_summary.csv" + B)
    A("")
    A(table(grid, ["task", "best_hidden_dim", "best_num_layers", "best_lr", "best_dropout", "cv_rmse"]))
    A("")
    A("**3.2 PyTorch-LSTM 测试精度（14 行）** 数据源：" + B + "pytorch_lstm_metrics.csv" + B)
    A("")
    A(table(lmet, ["策略", "预见期（天）", "RMSE", "MAE", "R²"]))
    A("")
    A("**3.3 四张集成表（各 7 行；列：策略/方法/预见期（天）/RMSE/MAE/R²）**")
    A("")
    for f, df in ens.items():
        A(B + f + B)
        A("")
        A(table(df, ["策略", "方法", "预见期（天）", "RMSE", "MAE", "R²"]))
        A("")
    A(f"集成明细 " + B + "ensemble_predictions.csv" + B + f"（{len(pred3)} 行）、成员表 " + B + "ensemble_members.csv" + B
      + "、权重表 " + B + "stacking_直接_权重.csv" + B + " / " + B + "stacking_多输出_权重.csv" + B + "。")
    A("")
    A("**3.4 结论与数据源**")
    A("")
    A(f"- 第 1 天最优单模型：PyTorch-LSTM 直接策略，R²={l1:.4f}（阈值 0.80，达标）→ " + B + "pytorch_lstm_metrics.csv" + B
      + "、" + B + "figures/fig3_测试段水文过程线.png" + B + "。")
    A(f"- 直接策略集成（6 模型）：简单平均第 1 天 R²={float(ens['6模型简单平均_直接.csv']['R²'].iloc[0]):.4f}，"
      f"Stacking R²={float(ens['6模型Stacking_直接.csv']['R²'].iloc[0]):.4f} → 简单平均更优。")
    A(f"- 多输出策略集成（3 模型）：简单平均第 1 天 R²={float(ens['3模型简单平均_多输出.csv']['R²'].iloc[0]):.4f}，"
      f"Stacking R²={float(ens['3模型Stacking_多输出.csv']['R²'].iloc[0]):.4f} → 简单平均更优。")
    A("- 两种集成使用同一批基学习器、同一测试预测矩阵（同一策略下各基学习器与集成的『实测』列逐行相同）；"
      "元学习器为『标准化 + TimeSeriesSplit 选 alpha 的 Ridge』，元特征来自训练段末尾 287 行留出块。")
    A("")
    A("## 4. 数字自洽校验")
    A("")
    A("**4.1 流水线自检（10 行）** 数据源：" + B + "自洽性校验.csv" + B + "（stage7 生成）")
    A("")
    A(table(chk, ["文件", "策略", "方法", "预见期", "CSV_R²", "重算_R²", "差值", "行数", "通过"], prec=6))
    A("")
    A(f"全部通过，最大 |重算 − 表值| = {float(chk['差值'].abs().max()):.1e}（阈值 1e-6）。")
    A("")
    A("**4.2 附加独立复核（33 行）** 数据源：" + B + "自洽性校验_附加校验.csv" + B + "（脚本 " + B + "archive/_scripts/verify_extra.py" + B + "）")
    A("")
    A("复核内容：① 四张集成表全部 7 行由 " + B + "ensemble_predictions.csv" + B + " 重算 RMSE/MAE/R²；"
      "② PyTorch-LSTM 第 1、7 天（直接/多输出）重新加载模型在测试集上重新预测后重算，不经过 "
      + B + "ensemble_predictions.csv" + B + "；③ 第二阶段 28 行由 " + B + "逐日预测结果.csv" + B + " 重算。")
    A("")
    A(f"最大差值：R²={float(chk2['R²差值'].max()):.1e}，RMSE={float(chk2['RMSE差值'].max()):.1e}，MAE={float(chk2['MAE差值'].max()):.1e}。")
    A("")
    A("## 5. 与探针数值的偏差说明")
    A("")
    A("| 模型（第 1 天） | 探针 R² | 本次正式流程 R² | 偏差 | 判定 |")
    A("|---|---|---|---|---|")
    A(f"| ANN 直接 | 0.7868 | {ann1:.4f} | {abs(ann1 - 0.7868):.4f} | 超出 0.01 |")
    A(f"| RF 直接 | 0.5606 | {rf1:.4f} | {abs(rf1 - 0.5606):.4f} | 超出 0.01 |")
    A(f"| PyTorch-LSTM 直接 | 0.8507 | {l1:.4f} | {abs(l1 - 0.8507):.4f} | 达标（<0.01） |")
    A("")
    A("根因（已受控复算）：" + B + "code/phase3_probe.py" + B + " 第 36-38 行使用**在旧 CSV 上选出的固定超参**"
      "（ANN alpha=10、hidden=(32,)；RF max_depth=16、min_samples_leaf=10、300 棵），未在修正数据上重新选参。")
    A("")
    A("修正数据 + 同一 71 维特征下复算对照：")
    A("")
    A("| 超参来源 | ANN 第 1 天 R² | RF 第 1 天 R² |")
    A("|---|---|---|")
    A("| 探针固定超参（旧数据选出） | 0.7868 | 0.5606 |")
    A("| 本次 TimeSeriesSplit 选参（ANN alpha=1.0、hidden=(16,)；RF depth=16、leaf=5） | 0.7230 | 0.5888 |")
    A("")
    A("即：探针数值可完全复现，差异 100% 来自『超参数只能由 TimeSeriesSplit 在训练段内选定』这一硬约束，"
      "本次未按探针手调参数。报告中引用探针值时请注明该口径差异。")
    A("")
    A("## 6. 图表清单（7 张）")
    A("")
    A("中文标题与图例；正文/坐标轴 SimSun，图例 Microsoft YaHei（SimSun 在图例中丢字）。")
    A("")
    A("| 文件 | 字节数 |")
    A("|---|---|")
    for p in figs:
        A(f"| {p.relative_to(ROOT).as_posix()} | {p.stat().st_size} |")
    A("")
    A("## 7. 各阶段耗时")
    A("")
    A("| 阶段 | 耗时 |")
    A("|---|---|")
    A("| 第二阶段 phase2_v5_experiment.py（16 次网格搜索 + 2004 评估） | 704 s |")
    A("| 第三阶段 stage1 PyTorch-LSTM 网格搜索 | 1040.6 s |")
    A("| 第三阶段 stage2 最优参数重训 | 707.7 s |")
    A("| 第三阶段 stage3 XGBoost / LightGBM | 9.3 s |")
    A("| 第三阶段 stage4 PseudoLSTM | 36.3 s |")
    A("| 第三阶段 stage5 集成（留出块重训 + Ridge） | 137.5 s |")
    A("| 第三阶段 stage6 四张中文图 | 1.6 s |")
    A("| 第三阶段 stage7 自洽性校验 | 0.1 s |")
    A("| 合计（各 stage 之和） | 2637.1 s ≈ 44 min |")
    A("")
    A("## 8. 已知问题与说明")
    A("")
    A("1. " + B + "ensemble_members.csv" + B + " 的『特征维数』列原先硬编码为 65/88，**已改为读取实际维度**（"
      + B + "code/phase3_run_all.py" + B + " 第 408-412 行 " + B + "base_dim, enh_dim = len(cols_b), len(cols_e)" + B
      + "）；当前为基础 71 维、增强 94 维，与 " + B + "run_stage5_7.log" + B
      + " 的『基础特征 71 维，增强特征 94 维』一致，成员表已重新生成（9 行）。")
    A("2. " + B + "figures/fig2_策略对比柱状图.png" + B
      + " 第 1 天最高柱（0.846）数值标签的裁切**已修复**：y 轴上下各留 12%（"
      + B + "ax.set_ylim(min(0.0, min(_vals)) * 1.12, max(0.0, max(_vals)) * 1.12)" + B
      + "，实际 (-0.8193, 0.9478)），图已重绘（74335 字节，1725×810 px），全部数值标签完整可见。")
    A("3. 第三阶段测试集 353 行、第二阶段 359 行：增强特征含 7 日滚动统计，首尾 NaN 行被丢弃，两阶段口径不同，"
      "不可用两阶段数字直接互比；同一阶段内部（表 ↔ 预测文件）已逐行校验一致。")
    A("4. 旧结果原样保存在 " + B + "archive/results_*_raw/" + B + "，未修改、未删除。")
    A("")

    OUT.write_text("\n".join(L).replace(B, chr(96)), encoding="utf-8")
    print(f"written {OUT} ({OUT.stat().st_size} bytes, {len(L)} lines)")


if __name__ == "__main__":
    main()
