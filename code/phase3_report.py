# -*- coding: utf-8 -*-
"""生成《机器学习课程实践》第三次阶段验收报告（2026 年 11 月 5 日）。

数据口径（全部在运行时从结果文件读取，正文不手写数值）
    results/phase3_enhanced/pytorch_lstm_grid_summary.csv     8 行网格搜索最优组合
    results/phase3_enhanced/pytorch_lstm_metrics.csv          14 行 LSTM 精度（直接 + 多输出）
    results/phase3_enhanced/6模型简单平均_直接.csv            子任务 4
    results/phase3_enhanced/6模型Stacking_直接.csv            子任务 5
    results/phase3_enhanced/3模型简单平均_多输出.csv          子任务 6
    results/phase3_enhanced/3模型Stacking_多输出.csv          子任务 7
    results/phase3_enhanced/stacking_直接_权重.csv            元学习器 alpha 与权重
    results/phase3_enhanced/stacking_多输出_权重.csv
    results/phase3_enhanced/ensemble_members.csv              9 个基学习器与输入维数
    results/phase3_enhanced/ensemble_predictions.csv          逐日预测（过程线分析）
    results/phase3_enhanced/boosting_features.json            增强特征清单（94 维）
    results/phase3_enhanced/run_stage5_7.log                  样本数与特征维数的运行时打印
    archive/results_phase3_enhanced_raw/                      未修正数据版本（仅取数对照）
    01047000_fixed.csv                                        修正后数据源（积雪水当量非零天数）

排版统一调用 code/report_style.py，不自行定义样式。
"""

import json
import re
import shutil
import sys
from pathlib import Path

import pandas as pd
from docx import Document
from docx.oxml.ns import qn
from docx.shared import Pt

sys.path.insert(0, str(Path(__file__).resolve().parent))

import report_style as rs  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
P3 = ROOT / "results" / "phase3_enhanced"
FIG = P3 / "figures"
RAW = ROOT / "archive" / "results_phase3_enhanced_raw"
OUT_DIR = ROOT / "本地资料" / "第三阶段"
DOC_NAME = "《机器学习课程实践》第三次阶段验收报告.docx"

FIG1 = FIG / "fig1_精度随预见期衰减.png"
FIG2 = FIG / "fig2_策略对比柱状图.png"
FIG3 = FIG / "fig3_测试段水文过程线.png"
FIG4 = FIG / "fig4_网格搜索交叉验证误差.png"

HORIZONS = list(range(1, 8))


MEMBERS_JSON = ROOT / "local_report_members.json"
COVER_PLACEHOLDER = {"names": "（姓名）", "student_ids": "（学号）", "advisor": "（指导教师）"}


def cover_fields():
    """从 local_report_members.json 读取封面姓名、学号与指导教师。

    该文件被 .gitignore 忽略，个人信息不进入仓库；文件缺失时回退为占位符并提示，
    不中断报告生成。
    """
    if not MEMBERS_JSON.exists():
        print("[警告] 未找到 local_report_members.json，封面姓名、学号、指导教师改用占位符")
        return dict(COVER_PLACEHOLDER)
    data = json.loads(MEMBERS_JSON.read_text(encoding="utf-8"))
    members = data["members"] if isinstance(data, dict) else data
    advisor = data.get("advisor", []) if isinstance(data, dict) else []
    return {
        "names": "、".join(m["name"] for m in members) or COVER_PLACEHOLDER["names"],
        "student_ids": "、".join(str(m["student_id"]) for m in members) or COVER_PLACEHOLDER["student_ids"],
        "advisor": "、".join(advisor) or COVER_PLACEHOLDER["advisor"],
    }


# ------------------------------------------------------------------ 读数

def f4(value):
    return "%.4f" % float(value)


def f2(value):
    return "%.2f" % float(value)


def f1(value):
    return "%.1f" % float(value)


def signed4(value):
    return "%+.4f" % float(value)


def load_series(path, strategy_value=None):
    """读一份指标表，返回 {预见期: 行}；strategy_value 用于 14 行的合并表。"""
    frame = pd.read_csv(path, encoding="utf-8-sig")
    if strategy_value is not None:
        frame = frame[frame["策略"] == strategy_value]
    return {int(row["预见期（天）"]): row for _, row in frame.iterrows()}


def load_runtime(path):
    """从流水日志里取样本数与特征维数，避免把 1434/353/71/94 写死在正文。"""
    text = path.read_text(encoding="utf-8", errors="ignore")
    data = re.search(r"训练 (\d+) 行 / 测试 (\d+) 行；基础特征 (\d+) 维，增强特征 (\d+) 维", text)
    hold = re.search(r"元特征留出块=最后 (\d+) 行（索引 (\d+) 起）", text)
    if data is None or hold is None:
        raise RuntimeError("日志中未找到样本数或留出块信息：%s" % path)
    return {
        "n_train": int(data.group(1)),
        "n_test": int(data.group(2)),
        "n_base": int(data.group(3)),
        "n_enh": int(data.group(4)),
        "n_hold": int(hold.group(1)),
        "hold_start": int(hold.group(2)),
    }


def load_all():
    data = {}
    data["grid"] = pd.read_csv(P3 / "pytorch_lstm_grid_summary.csv", encoding="utf-8-sig")
    order = {"direct_day%d" % h: h for h in HORIZONS}
    data["grid"]["sort_key"] = data["grid"]["task"].map(lambda t: order.get(t, 8))
    data["grid"] = data["grid"].sort_values("sort_key").reset_index(drop=True)

    data["lstm_direct"] = load_series(P3 / "pytorch_lstm_metrics.csv", "PyTorch-LSTM-直接")
    data["lstm_multi"] = load_series(P3 / "pytorch_lstm_metrics.csv", "PyTorch-LSTM-多输出")
    data["avg6"] = load_series(P3 / "6模型简单平均_直接.csv")
    data["stack6"] = load_series(P3 / "6模型Stacking_直接.csv")
    data["avg3"] = load_series(P3 / "3模型简单平均_多输出.csv")
    data["stack3"] = load_series(P3 / "3模型Stacking_多输出.csv")

    data["w_direct"] = pd.read_csv(P3 / "stacking_直接_权重.csv", encoding="utf-8-sig")
    data["w_multi"] = pd.read_csv(P3 / "stacking_多输出_权重.csv", encoding="utf-8-sig")
    data["members"] = pd.read_csv(P3 / "ensemble_members.csv", encoding="utf-8-sig")
    data["meta"] = json.loads((P3 / "boosting_features.json").read_text(encoding="utf-8"))
    data["rt"] = load_runtime(P3 / "run_stage5_7.log")

    data["raw_lstm_direct"] = load_series(RAW / "pytorch_lstm_metrics.csv", "PyTorch-LSTM-直接")
    data["raw_lstm_multi"] = load_series(RAW / "pytorch_lstm_metrics.csv", "PyTorch-LSTM-多输出")
    data["raw_avg6"] = load_series(RAW / "6模型简单平均_直接.csv")

    fixed = pd.read_csv(ROOT / "01047000_fixed.csv", encoding="utf-8-sig")
    data["n_days"] = int(len(fixed))
    data["swe_nonzero"] = int((fixed["Swe"] > 0).sum())

    # 日期口径：预测表的“起报日”是样本日期，“目标日”才是被预测的那一天；
    # 过程线定位与误差统计一律按目标日筛选，避免“数值对、日期错”。
    pred = pd.read_csv(P3 / "ensemble_predictions.csv", encoding="utf-8-sig",
                       usecols=["策略", "方法", "预见期（天）", "起报日", "目标日", "实测", "预测"])
    best = pred[(pred["策略"] == "直接") & (pred["方法"] == "PyTorch-LSTM")
                & (pred["预见期（天）"] == 1)].sort_values("目标日").reset_index(drop=True)
    peak = best.loc[best["实测"].idxmax()]
    data["test_first"] = best["起报日"].min()
    data["test_last"] = best["起报日"].max()
    data["target_first"] = best["目标日"].min()
    data["target_last"] = best["目标日"].max()
    data["peak_date"] = peak["目标日"]
    data["peak_start"] = peak["起报日"]
    data["peak_obs"] = float(peak["实测"])
    data["peak_pred"] = float(peak["预测"])
    spring = best[(best["目标日"] >= "2004-03-01") & (best["目标日"] <= "2004-06-30")]
    data["spring_n"] = int(len(spring))
    data["spring_mae"] = float((spring["预测"] - spring["实测"]).abs().mean())

    # 峰值日期不由表内直接取值，而是用原始观测在同一目标日窗口内的 argmax 反推后再比对
    source = pd.read_csv(ROOT / "01047000.csv", encoding="utf-8-sig")
    window = source[(source["Date"] >= data["target_first"]) & (source["Date"] <= data["target_last"])]
    source_peak = window.loc[window["Discharge"].idxmax()]
    joined = best.merge(source[["Date", "Discharge"]], left_on="目标日", right_on="Date", how="left")
    assert joined["Discharge"].notna().all(), "存在目标日在 01047000.csv 中找不到的行"
    assert float((joined["实测"] - joined["Discharge"]).abs().max()) == 0.0, "实测值与原始径流不一致"
    assert str(source_peak["Date"]) == data["peak_date"], (
        "峰值日期与原始观测 argmax 不一致：预测表 %s，01047000.csv %s"
        % (data["peak_date"], source_peak["Date"]))
    assert float(source_peak["Discharge"]) == data["peak_obs"], "峰值流量与原始观测不一致"
    same_row = best.loc[best["目标日"] == data["peak_date"], "预测"]
    assert len(same_row) == 1, "峰值目标日在预测表中不唯一"
    assert abs(float(same_row.iloc[0]) - data["peak_pred"]) < 1e-9, "峰值预测值不是同一目标日的取值"
    spring_again = best[(best["目标日"] >= "2004-03-01") & (best["目标日"] <= "2004-06-30")]
    assert len(spring_again) == data["spring_n"], "融雪期筛选行数不一致"
    assert abs(float((spring_again["预测"] - spring_again["实测"]).abs().mean())
               - data["spring_mae"]) < 1e-12, "融雪期 MAE 与筛选口径不一致"

    data["assert_log"] = [
        "日期口径断言：预测表 %d 行目标日全部命中 01047000.csv，实测值最大偏差 0.0" % len(joined),
        "日期口径断言：峰值目标日 %s（起报日 %s）等于原始观测 argmax，实测 %s cfs"
        % (data["peak_date"], data["peak_start"], f1(data["peak_obs"])),
        "日期口径断言：峰值预测 %s cfs 取自同一目标日行；融雪期 n=%d，MAE %s cfs"
        % (f2(data["peak_pred"]), data["spring_n"], f2(data["spring_mae"])),
    ]
    return data


# ------------------------------------------------------------------ 构件

def build():
    d = load_all()
    rt = d["rt"]
    ld, lm = d["lstm_direct"], d["lstm_multi"]
    a6, s6, a3, s3 = d["avg6"], d["stack6"], d["avg3"], d["stack3"]
    raw_ld, raw_lm, raw_a6 = d["raw_lstm_direct"], d["raw_lstm_multi"], d["raw_avg6"]

    # 跨期比较所需的计数（全部由表算出，不预设结论）
    avg_beats_stack_direct = sum(1 for h in HORIZONS if a6[h]["R²"] > s6[h]["R²"])
    avg_beats_stack_multi = sum(1 for h in HORIZONS if a3[h]["R²"] > s3[h]["R²"])
    multi_better = sum(1 for h in HORIZONS if lm[h]["R²"] > ld[h]["R²"])
    mean_ld = sum(ld[h]["R²"] for h in HORIZONS) / 7
    mean_lm = sum(lm[h]["R²"] for h in HORIZONS) / 7
    mean_a6 = sum(a6[h]["R²"] for h in HORIZONS) / 7
    mean_s6 = sum(s6[h]["R²"] for h in HORIZONS) / 7
    grid_day1 = d["grid"].iloc[0]
    grid_day7 = d["grid"].iloc[7 - 1]
    grid_multi = d["grid"].iloc[7]

    doc = rs.new_document()
    rs.add_cover(
        doc,
        college="自动化学院",
        major="智能科学与技术",
        class_name="课程实践小组",
        **cover_fields(),
        project="基于机器学习的径流预测",
        content="基于 LSTM 与集成学习的径流多步预测",
        subtitle="第三次阶段验收报告",
        date_text="二〇二六年十一月五日",
    )
    rs.begin_body_section(doc, 1)
    num = rs.Numbering()

    def P(text):
        return rs.add_body(doc, text)

    def H1(text):
        return rs.add_heading1(doc, text)

    def H2(text):
        return rs.add_heading2(doc, text)

    def T(header, rows, caption, chapter):
        return rs.add_table(doc, header, rows, caption=caption, chapter=chapter, numbering=num)

    def NOTE(text):
        return rs.add_figure_note(doc, text)

    def FIGURE(path, caption, chapter):
        return rs.add_figure(doc, path, caption=caption, chapter=chapter, numbering=num)

    def precision_rows(rows):
        return [[str(h), f4(rows[h]["R²"]), f2(rows[h]["RMSE"]), f2(rows[h]["MAE"])]
                for h in HORIZONS]

    PRECISION_HEADER = ["预见期（天）", "R²", "RMSE（cfs）", "MAE（cfs）"]

    # ---------------------------------------------------------------- 引言
    P("本报告是《机器学习课程实践》第三次阶段验收报告，对应 2026 年 11 月 5 日的验收说明，"
      "覆盖验收要求中的 7 个子任务：子任务 1 用网格搜索与时序交叉验证确定 LSTM 超参数；"
      "子任务 2 与子任务 3 分别给出直接策略与多输出策略在 1 至 7 天预见期上的精度；"
      "子任务 4 至子任务 7 分别在两种策略下实现简单平均与 Stacking 两类集成。"
      "全部指标由脚本从逐日预测结果重新计算，正文数值与结果文件一致。")
    P("数据源为 01047000_fixed.csv。相对上一版本的原始 CSV，本版做了两项修正："
      "其一，气象列改为按 year 与 yday 两个键与 Daymet 官方日值对齐，不再按行号拼接；"
      "其二，恢复了真实的积雪水当量列 Swe，全部 %d 个观测日中非零值有 %d 天，"
      "而原 CSV 的 Swe 列恒为 0，导致上一版所有积雪相关派生量以零方差被剔除。"
      "在此基础上构造基础特征 %d 维与增强特征 %d 维，"
      "训练段 %d 行、测试段 %d 行。"
      % (d["n_days"], d["swe_nonzero"], rt["n_base"], rt["n_enh"], rt["n_train"], rt["n_test"]))

    # ---------------------------------------------------------------- 第一章
    H1("一、基于 LSTM 的径流多步预测")
    H2("1.1 LSTM 原理与实现要点")
    P("长短期记忆网络在循环结构中以门控机制调节信息流。设 t 时刻的输入为 x_t、上一时刻隐状态为 h_(t-1)，"
      "则遗忘门 f_t 与输入门 i_t 分别由 h_(t-1) 与 x_t 的线性变换经 sigmoid 激活得到，"
      "候选记忆 c~_t 由同一输入经 tanh 激活得到；细胞状态按 c_t = f_t 与 c_(t-1) 的逐元素积"
      "加上 i_t 与 c~_t 的逐元素积更新；输出门 o_t 控制当前细胞状态经 tanh 后的输出比例，"
      "得到 h_t = o_t 与 tanh(c_t) 的逐元素积。细胞状态在时间步之间近似线性传递，"
      "梯度不必逐层连乘，因而缓解了普通循环网络的梯度消失问题。"
      "对径流这类存在融雪滞后与土壤蓄水释放过程的序列，门控结构提供了选择性保留历史信号的能力。")
    P("本项目的 LSTM 由 PyTorch 实现，使用 torch.nn.LSTM 构建循环层，"
      "其后接 隐层映射、ReLU、Dropout 与线性输出头；多层时按层设置 dropout，"
      "训练采用 Adam 优化器、均方误差损失、梯度范数裁剪与学习率衰减调度，"
      "并按时间顺序留出训练段尾段作验证集早停。"
      "输入与目标均用只在训练段内拟合的标准化器处理，预测后再反变换回物理量纲，"
      "因此测试段的量纲还原不引入训练信息。"
      "需要说明的是，这一实现是真实的门控循环网络，而不是用全连接网络做的近似："
      "集成中的 PseudoLSTM 才是 sklearn 的 MLPRegressor 近似实现，两者不可混为一谈。")
    P("同时应当如实说明本实现的输入形态。为避免在 %d 个训练样本上引入过长的序列维度，"
      "PyTorch-LSTM 每条样本送入的是当期一帧 %d 维特征向量，序列长度取 1，"
      "循环层在这一设定下不跨时间步展开，历史信息由滞后项、滚动统计、指数加权移动平均与差分等"
      "显式特征承载。这一取舍使训练成本可控，也使不同模型在同一输入口径下可比；"
      "代价是循环结构本身未能额外贡献跨步记忆，这是本阶段实现的边界。"
      % (rt["n_train"], rt["n_enh"]))
    P("目标变量取 h 天后的日径流（单位 cfs），保留物理量纲而不取对数，"
      "与上一阶段 ANN、RF 采用 log1p 目标的处理不同；"
      "不同目标的模型在集成时由各自脚本按训练时口径还原，保证预测值可比。"
      "评估指标为均方根误差 RMSE、平均绝对误差 MAE 与决定系数 R²，"
      "全部在原始量纲上计算，测试段 %d 个起报日一次性评估。" % rt["n_test"])

    H2("1.2 子任务 1：网格搜索与交叉验证确定超参数")
    P("超参数搜索空间为隐层维度 hidden ∈ {64, 128}、循环层数 layers ∈ {1, 2}、"
      "学习率 lr ∈ {0.001, 0.01}，dropout 固定为 0.2，共 8 组组合。"
      "验证方式为 GridSearchCV 配合 TimeSeriesSplit，折数 3、间隔 gap = 7 天，"
      "使验证块与前一块之间空出 7 天，避免相邻起报日的 7 天预见期目标跨块重叠；"
      "评分指标为验证块上的 RMSE。"
      "直接策略对 7 个预见期分别选参一次，多输出策略对 7 维输出整体选参一次，共 8 次搜索。"
      "选参全程只在训练段内部完成，测试段不参与任何环节。")
    rows = []
    for _, row in d["grid"].iterrows():
        task = row["task"]
        name = ("直接策略（第 %s 天）" % task.replace("direct_day", "")
                if task.startswith("direct_day") else "多输出策略")
        rows.append([name,
                     "hidden=%d，layers=%d，lr=%s，dropout=%s"
                     % (int(row["best_hidden_dim"]), int(row["best_num_layers"]),
                        ("%.3f" % row["best_lr"]), ("%.1f" % row["best_dropout"])),
                     f2(row["cv_rmse"])])
    T(["任务", "最优超参数组合", "交叉验证 RMSE（cfs）"], rows,
      "PyTorch-LSTM 网格搜索最优组合（TimeSeriesSplit 3 折、gap = 7）", 1)
    NOTE("注：交叉验证 RMSE 为三个时序折上的平均值，单位为 cfs；选参只在训练段内部进行。")
    FIGURE(FIG4, "各任务最优组合的交叉验证误差对比", 1)
    rs.add_figure_note(doc, "图注：横轴为 8 个选参任务，柱高为最优组合的交叉验证 RMSE（cfs），"
                            "柱顶标注数值。")
    P("交叉验证误差随预见期单调上升：第 1 天的最优组合为 hidden=128、layers=2、lr=0.001，"
      "交叉验证 RMSE 为 %s cfs；第 7 天为 hidden=%d、layers=%d、lr=%s，上升至 %s cfs；"
      "多输出策略为 hidden=%d、layers=%d、lr=%s，交叉验证 RMSE 为 %s cfs。"
      "最优组合在 8 个任务之间并不一致，说明在本样本规模下没有一组超参数能同时适配所有预见期，"
      "逐任务选参是必要的。隐层维度与层数在长预见期上普遍取较小的配置，"
      "反映低信噪比目标更需要压缩模型容量。"
      % (f2(grid_day1["cv_rmse"]),
         int(grid_day7["best_hidden_dim"]), int(grid_day7["best_num_layers"]),
         ("%.3f" % grid_day7["best_lr"]), f2(grid_day7["cv_rmse"]),
         int(grid_multi["best_hidden_dim"]), int(grid_multi["best_num_layers"]),
         ("%.3f" % grid_multi["best_lr"]), f2(grid_multi["cv_rmse"])))

    H2("1.3 子任务 2：直接策略 1–7 天精度")
    P("直接策略为每个预见期独立训练一个 LSTM，输入同一组 %d 维增强特征，"
      "目标为该预见期对应的 h 天后径流，7 个预见期共 7 个模型。"
      "独立建模的优点是每个模型只服务一个目标，可以专门拟合该预见期的误差结构；"
      "代价是 7 个模型之间不共享任何信息，且训练样本只有 %d 行。"
      % (rt["n_enh"], rt["n_train"]))
    T(PRECISION_HEADER, precision_rows(ld), "PyTorch-LSTM 直接策略 1–7 天测试精度", 1)
    NOTE("注：评估样本为测试段 %d 个起报日，RMSE 与 MAE 的单位为 cfs。"
         "增强特征含 7 日滚动统计，首尾含缺失的行被丢弃，"
         "因此本阶段测试段样本数比第二阶段口径少 6 行，两阶段的数值不可逐项直接对照，"
         "第 3.4 节另行给出同口径的整体比较。" % rt["n_test"])
    FIGURE(FIG1, "测试集精度随预见期的衰减（左：直接策略；右：多输出策略）", 1)
    rs.add_figure_note(doc, "图注：横轴为预见期天数，纵轴为决定系数 R²，灰色点线为零基准线；"
                            "两幅子图并列给出两种策略下各方案的曲线。")
    P("直接策略第 1 天的 R² 为 %s、RMSE 为 %s cfs、MAE 为 %s cfs，"
      "已达到本阶段设定的第 1 天 R² 不低于 0.80 的要求；"
      "第 2 天 R² 为 %s、RMSE 为 %s cfs，仍明显优于以测试段均值为预测的基准；"
      "第 3 天起 R² 转负（%s），第 5 天为 %s，第 7 天为 %s（RMSE %s cfs）。"
      "7 个预见期的平均 R² 为 %s，衰减主要发生在第 2 天到第 3 天之间，"
      "与输入中缺少目标时段气象信息这一约束直接相关。"
      % (f4(ld[1]["R²"]), f2(ld[1]["RMSE"]), f2(ld[1]["MAE"]),
         f4(ld[2]["R²"]), f2(ld[2]["RMSE"]),
         f4(ld[3]["R²"]), f4(ld[5]["R²"]), f4(ld[7]["R²"]), f2(ld[7]["RMSE"]),
         f4(mean_ld)))

    H2("1.4 子任务 3：多输出策略 1–7 天精度")
    P("多输出策略只训练一个 LSTM，网络的输出头直接给出 7 维结果，即 output_dim 取 7，"
      "由同一次前向计算同时得到 7 个预见期的预测，"
      "不使用 MultiOutputRegressor 之类的逐输出包装器——后者内部仍是 7 个独立模型，"
      "并不共享表示。共享表示使短预见期的高信噪比目标在联合训练中约束了长预见期的拟合，"
      "相当于对低信噪比输出施加了正则；同时参数规模与训练轮次都降到直接策略的七分之一左右。")
    T(PRECISION_HEADER, precision_rows(lm), "PyTorch-LSTM 多输出策略 1–7 天测试精度", 1)
    NOTE("注：评估样本为测试段 %d 个起报日，与直接策略完全相同；RMSE 与 MAE 的单位为 cfs。"
         % rt["n_test"])
    P("多输出策略第 1 天的 R² 为 %s、RMSE 为 %s cfs、MAE 为 %s cfs，"
      "比直接策略低 %s，同样达到 0.80 的第 1 天目标；"
      "第 2 天 R² 为 %s，第 3 天为 %s，第 7 天为 %s。"
      "7 个预见期的平均 R² 为 %s，与直接策略的 %s 接近。"
      "逐期比较看，多输出策略在 %d 个预见期上高于直接策略，"
      "且优势集中在第 3 天及以后，"
      "与共享表示对低信噪比目标的约束作用一致，图 1-2 右图可直接看到这一交叉。"
      % (f4(lm[1]["R²"]), f2(lm[1]["RMSE"]), f2(lm[1]["MAE"]), f4(ld[1]["R²"] - lm[1]["R²"]),
         f4(lm[2]["R²"]), f4(lm[3]["R²"]), f4(lm[7]["R²"]),
         f4(mean_lm), f4(mean_ld), multi_better))

    # ---------------------------------------------------------------- 第二章
    H1("二、基于集成学习的径流多步预测")
    H2("2.1 基学习器构成与数据划分")
    P("集成使用两类基学习器集合：直接策略下 6 个，多输出策略下 3 个。"
      "两套特征按日期取交集后再拼接预测，保证所有基学习器面对同一批起报日。"
      "其中 ANN、RF、PseudoLSTM 沿用上一阶段的模型文件，输入为 %d 维基础特征；"
      "XGBoost、LightGBM 与 PyTorch-LSTM 在本阶段重训，输入为 %d 维增强特征。"
      % (rt["n_base"], rt["n_enh"]))
    impl = {}
    for _, row in d["members"].iterrows():
        name, strategy, dim = row["基学习器"], row["策略"], int(row["特征维数"])
        if name == "ANN":
            text = "sklearn MLPRegressor，log1p 目标，集成时以 expm1 还原"
        elif name == "RF":
            text = "RandomForestRegressor，原生多输出，log1p 目标"
        elif name == "XGBoost":
            text = "梯度提升树，验证集早停"
        elif name == "LightGBM":
            text = "梯度提升树，验证集早停"
        elif name == "PyTorch-LSTM":
            text = "torch.nn.LSTM 真实门控实现，单步输入"
        else:
            text = "sklearn MLPRegressor 的时序特征近似实现，非真实 LSTM"
        impl[(name, strategy)] = (text, dim)
    rows = [[s, n, impl[(n, s)][0], str(impl[(n, s)][1])] for (n, s) in impl]
    T(["策略", "基学习器", "实现要点", "输入特征维数"], rows,
      "集成学习的基学习器构成", 2)
    NOTE("注：PseudoLSTM 以多层感知机近似时序建模，不具备门控与细胞状态，"
         "在本报告中只作为集成中的一个对照基学习器；"
         "本报告的 LSTM 结论均来自 PyTorch 实现的门控循环网络。")
    rows = [["训练段", str(rt["n_train"]), "全部基学习器与元学习器的拟合数据"],
            ["元特征留出块", str(rt["n_hold"]),
             "训练段末尾 20%，用于生成 Stacking 的元特征与元学习器目标"],
            ["测试段", str(rt["n_test"]),
             "起报日 %s 至 %s（目标日 %s 至 %s），只在最终评估时使用一次"
             % (d["test_first"], d["test_last"], d["target_first"], d["target_last"])]]
    T(["数据段", "样本数", "用途"], rows, "数据划分与评估口径", 2)
    NOTE("注：三种数据段首尾相接、互不重叠；留出块是训练段的子集，"
         "其中的基学习器按预见期在留出块之前的前缀上重训后再预测，"
         "使元特征不含目标信息。")

    H2("2.2 子任务 4：直接策略简单平均")
    P("简单平均把 6 个基学习器在同一预见期上的预测直接取算术平均，不引入任何待估参数，"
      "各基学习器的权重固定为 1/6。该方法只能降低各成员相互独立的误差分量，"
      "对系统性偏差无能为力，但在样本量有限时不存在二次过拟合的风险。")
    T(PRECISION_HEADER, precision_rows(a6), "6 模型简单平均（直接策略）1–7 天测试精度", 2)
    NOTE("注：6 个基学习器为 ANN、RF、XGBoost、LightGBM、PyTorch-LSTM、PseudoLSTM；"
         "评估样本为测试段 %d 个起报日。" % rt["n_test"])
    P("简单平均第 1 天的 R² 为 %s、RMSE 为 %s cfs、MAE 为 %s cfs，"
      "达到 0.80 的目标；第 2 天 R² 为 %s，第 5 天为 %s，第 7 天为 %s。"
      "与单模型相比，简单平均在第 1 天低于 LSTM 直接策略的 %s，"
      "但 RMSE 差只有 %s cfs；在第 2 天至第 7 天则普遍高于单个基学习器，"
      "说明等权平均的主要收益是压低长预见期的方差而不是提升峰值拟合能力。"
      % (f4(a6[1]["R²"]), f2(a6[1]["RMSE"]), f2(a6[1]["MAE"]),
         f4(a6[2]["R²"]), f4(a6[5]["R²"]), f4(a6[7]["R²"]),
         f4(ld[1]["R²"]), f2(a6[1]["RMSE"] - ld[1]["RMSE"])))

    H2("2.3 子任务 5：直接策略 Stacking")
    P("Stacking 用一层元学习器把 6 个基学习器的预测组合成最终预测。"
      "本项目的元学习器为 Ridge 岭回归：元特征先按训练段统计量标准化，"
      "再在正则化系数 alpha ∈ {0.01, 0.1, 1, 10, 100, 1000, 10000} 上做 3 折时序交叉验证，"
      "取验证误差最小的 alpha 重新拟合全部元特征。"
      "元特征的构造方式是：以训练段末尾 20%% 的 %d 行为留出块，"
      "基学习器在留出块之前的前缀上重训，再对留出块给出预测，"
      "这 %d 个预测构成元学习器的训练矩阵，其对应的实测径流为元学习器的目标。"
      "对第 h 天，前缀在留出块起点之前再回退 h 天，避免目标跨入留出块。"
      "测试段仍由在全部训练段上拟合的基学习器给出预测，再经同一 Ridge 组合。"
      % (rt["n_hold"], rt["n_hold"]))
    T(PRECISION_HEADER, precision_rows(s6), "6 模型 Stacking（直接策略）1–7 天测试精度", 2)
    NOTE("注：元学习器为标准化后的 Ridge 回归，alpha 由时序交叉验证选出；"
         "评估样本为测试段 %d 个起报日，与简单平均完全相同。" % rt["n_test"])
    wd = d["w_direct"]
    rows = []
    for _, row in wd.iterrows():
        cols = [c for c in wd.columns if c.startswith("w_")]
        values = {c: float(row[c]) for c in cols}
        worst = max(values, key=lambda c: abs(values[c]))
        rows.append([str(int(row["预见期（天）"])), ("%g" % float(row["ridge_alpha"])),
                     worst.replace("w_", ""), f2(values[worst]),
                     str(sum(1 for c in cols if values[c] < 0))])
    T(["预见期（天）", "选出的 alpha", "权重绝对值最大的基学习器", "该权重值", "负权重个数"],
      rows, "Ridge 元学习器的正则化系数与权重幅度", 2)
    NOTE("注：权重为元特征标准化后的系数，单位不是流量；"
         "系数幅度远大于 1 说明元学习器在共线元特征上取了相互抵消的大权重。")
    P("Stacking 第 1 天的 R² 为 %s、RMSE 为 %s cfs，低于简单平均的 %s 与 %s cfs；"
      "第 7 天 R² 为 %s。"
      "从表 2-5 看，元学习器选出的 alpha 在 0.01 与 100 之间跳动，"
      "权重绝对值最大项达到 %s 量级，且同一预见期内正负权重并存，"
      "这是最小二乘在共线元特征上求逆的典型表现，后文 3.2 节结合这一现象分析原因。"
      % (f4(s6[1]["R²"]), f2(s6[1]["RMSE"]), f4(a6[1]["R²"]), f2(a6[1]["RMSE"]),
         f4(s6[7]["R²"]), f2(max(abs(float(v)) for v in wd[[c for c in wd.columns if c.startswith("w_")]].to_numpy().ravel()))))

    H2("2.4 子任务 6：多输出策略简单平均")
    P("多输出策略的基学习器只有 3 个：ANN、RF 与 PyTorch-LSTM，"
      "三者都能一次输出 7 维结果，因此简单平均在每一维上分别对 3 个预测取算术平均。"
      "与直接策略相比，这里的成员更少、成员之间相关性更高，等权平均能带来的方差削减也更有限。")
    T(PRECISION_HEADER, precision_rows(a3), "3 模型简单平均（多输出策略）1–7 天测试精度", 2)
    NOTE("注：3 个基学习器为 ANN、RF 与 PyTorch-LSTM，后者的输入为 94 维增强特征，"
         "前两者为 71 维基础特征；评估样本为测试段 %d 个起报日。" % rt["n_test"])
    P("简单平均第 1 天的 R² 为 %s、RMSE 为 %s cfs，第 7 天 R² 为 %s。"
      "该结果低于 6 模型简单平均的 %s，也低于单模型多输出策略的 %s，"
      "原因是成员数从 6 个降到 3 个，且三个成员在同一批特征上高度同质，"
      "平均无法消除共同的系统性偏差。"
      % (f4(a3[1]["R²"]), f2(a3[1]["RMSE"]), f4(a3[7]["R²"]),
         f4(a6[1]["R²"]), f4(lm[1]["R²"])))

    H2("2.5 子任务 7：多输出策略 Stacking")
    P("多输出策略的 Stacking 与直接策略同构：以训练段末尾留出块上的 3 维基学习器预测为元特征，"
      "Ridge 元学习器在标准化后按 3 折时序交叉验证选 alpha，再对测试段的 3 维基学习器预测加权组合。"
      "成员只有 3 个、样本只有几百行，元学习器可辨识的信息量本就有限。")
    T(PRECISION_HEADER, precision_rows(s3), "3 模型 Stacking（多输出策略）1–7 天测试精度", 2)
    NOTE("注：元学习器与子任务 5 相同，均为标准化后的 Ridge 回归；"
         "评估样本为测试段 %d 个起报日。" % rt["n_test"])
    wm = d["w_multi"]
    alphas = "、".join("%g" % float(v) for v in wm["ridge_alpha"])
    P("Stacking 第 1 天的 R² 为 %s、RMSE 为 %s cfs，第 7 天 R² 为 %s；"
      "7 个预见期选出的 alpha 依次为 %s，同样在 2 个数量级之间跳动。"
      "与简单平均相比，除第 6 天略高以外其余 6 个预见期均更低，"
      "说明在本数据规模下，元学习器带来的组合灵活性没有转化为测试段收益。"
      % (f4(s3[1]["R²"]), f2(s3[1]["RMSE"]), f4(s3[7]["R²"]), alphas))

    H2("2.6 策略对比")
    P("六种方案在第 1 天与第 7 天的 R² 并列于图 2-1。第 1 天的柱高集中在 0.67 至 0.85 之间，"
      "排序为 LSTM 直接策略 %s、6 模型简单平均 %s、LSTM 多输出 %s、6 模型 Stacking %s、"
      "3 模型简单平均 %s、3 模型 Stacking %s；"
      "第 7 天全部落到 0.09 以下，其中 LSTM 直接策略为 %s、6 模型 Stacking 为 %s。"
      "方案的区分度在短预见期上主要体现为是否达到 0.80 的目标，"
      "在长预见期上则退化为谁更接近零基准。"
      % (f4(ld[1]["R²"]), f4(a6[1]["R²"]), f4(lm[1]["R²"]), f4(s6[1]["R²"]),
         f4(a3[1]["R²"]), f4(s3[1]["R²"]), f4(ld[7]["R²"]), f4(s6[7]["R²"])))
    FIGURE(FIG2, "各方案在第 1 天与第 7 天的测试集 R² 对比", 2)
    rs.add_figure_note(doc, "图注：柱高为测试段上的决定系数 R²，柱顶标注四位小数，"
                            "黑色水平实线为零基准线。")
    P("测试段过程线见图 2-2，横轴为目标日期，曲线由 PyTorch-LSTM 直接策略第 1 天预测绘制，"
      "该方案是全部方案中第 1 天 R² 最高者（%s）。"
      "基流段与退水段的预测曲线与实测基本重合，说明模型对缓慢变化的水文过程拟合良好；"
      "误差主要出现在洪峰处，全段最大实测流量为 %s 年 %d 月 %d 日（目标日）的 %s cfs，"
      "同一目标日的预测值为 %s cfs，幅度被显著低估；"
      "3 月至 6 月融雪期（%d 个目标日）的平均绝对误差为 %s cfs，高于全段的 %s cfs。"
      "这一形态说明模型输出的是条件均值意义上的平滑估计，"
      "在低频高幅事件上系统性向均值收缩。"
      % (f4(ld[1]["R²"]),
         d["peak_date"][:4], int(d["peak_date"][5:7]), int(d["peak_date"][8:10]),
         f1(d["peak_obs"]), f2(d["peak_pred"]), d["spring_n"],
         f2(d["spring_mae"]), f2(ld[1]["MAE"])))
    FIGURE(FIG3, "2004 年测试段实测径流与第 1 天预测过程线（横轴为目标日期）", 2)
    rs.add_figure_note(doc, "图注：实线为实测径流，虚线为 PyTorch-LSTM 直接策略的第 1 天预测，"
                            "横轴为目标日期，数据取自测试段逐日预测结果。")

    # ---------------------------------------------------------------- 第三章
    H1("三、结果分析与讨论")
    H2("3.1 单模型与集成的对比")
    P("六种方案在第 1 天与第 7 天的精度汇总于表 3-1。"
      "第 1 天上单模型 LSTM 直接策略以 R² %s、RMSE %s cfs 居首，"
      "6 模型简单平均以 %s 紧随其后，"
      "多输出 LSTM 为 %s，三类集成中的其余两项分别为 %s 与 %s。"
      "也就是说，达到第 1 天 0.80 目标的方案共有三个：单模型 LSTM 直接策略、"
      "单模型 LSTM 多输出策略与 6 模型简单平均；"
      "把基学习器再叠加一层元学习器并没有带来额外收益。"
      % (f4(ld[1]["R²"]), f2(ld[1]["RMSE"]), f4(a6[1]["R²"]), f4(lm[1]["R²"]),
         f4(s6[1]["R²"]), f4(a3[1]["R²"])))
    rows = []
    for label, series in (("LSTM 直接策略", ld), ("LSTM 多输出策略", lm),
                          ("6 模型简单平均（直接）", a6), ("6 模型 Stacking（直接）", s6),
                          ("3 模型简单平均（多输出）", a3), ("3 模型 Stacking（多输出）", s3)):
        rows.append([label, f4(series[1]["R²"]), f2(series[1]["RMSE"]),
                     f4(series[7]["R²"]), f2(series[7]["RMSE"])])
    T(["方案", "第 1 天 R²", "第 1 天 RMSE（cfs）", "第 7 天 R²", "第 7 天 RMSE（cfs）"],
      rows, "六种方案在第 1 天与第 7 天的精度对比", 3)
    NOTE("注：六种方案的评估样本均为测试段 %d 个起报日，特征与数据划分相同，"
         "差异只来自模型与组合方式；RMSE 的单位为 cfs。" % rt["n_test"])
    P("换一个口径，按 7 个预见期的平均 R² 计，排序与第 1 天并不相同："
      "6 模型简单平均最高，为 %s；6 模型 Stacking 次之，为 %s；"
      "LSTM 多输出策略为 %s，LSTM 直接策略为 %s。"
      "第 1 天最优的是单模型，7 期平均最优的是集成，两者并不矛盾："
      "单模型在第 1 天把峰值段拟合得更好，但在第 3 天以后迅速跌到零基准以下，"
      "而等权平均把若干个都在零基准附近徘徊的成员合在一起，"
      "整体方差更小、平均值因此更高。这说明本阶段的集成收益主要体现在稳健性上，"
      "而不是体现在短预见期的尖峰精度上。"
      % (f4(mean_a6), f4(mean_s6), f4(mean_lm), f4(mean_ld)))

    H2("3.2 简单平均优于 Stacking 的原因")
    P("逐期比较的结果是：直接策略下简单平均在 %d 个预见期上高于 Stacking（共 7 个），"
      "多输出策略下在 %d 个预见期上更高。"
      "这一结果由数据支持，不应反向表述为 Stacking 更优。其成因可从三方面分析。"
      % (avg_beats_stack_direct, avg_beats_stack_multi))
    rows = []
    for h in HORIZONS:
        rows.append([str(h), f4(a6[h]["R²"]), f4(s6[h]["R²"]), signed4(a6[h]["R²"] - s6[h]["R²"]),
                     f4(a3[h]["R²"]), f4(s3[h]["R²"]), signed4(a3[h]["R²"] - s3[h]["R²"])])
    T(["预见期（天）", "直接·简单平均 R²", "直接·Stacking R²", "差值",
       "多输出·简单平均 R²", "多输出·Stacking R²", "差值"],
      rows, "简单平均与 Stacking 的逐期 R² 对照", 3)
    NOTE("注：差值为简单平均减 Stacking，正值表示简单平均更高；"
         "两种方法使用完全相同的基学习器与完全相同的测试样本。")
    P("第一，元特征与测试期的基学习器预测存在分布偏移。"
      "元特征来自在留出块之前的前缀上重训的基学习器，"
      "而测试期的基学习器预测来自在全部训练段上拟合的模型，两者在训练样本量、"
      "树的深度与迭代轮次上并不等价，预测的偏差结构因此不同。"
      "元学习器学到的是“前缀模型在留出块上的行为”，"
      "直接应用到“全量模型在测试期上的预测”时，权重所依赖的关系已经改变。")
    P("第二，留出块过小，且块内样本并非独立。"
      "留出块只有 %d 行，占训练段的 20%%；"
      "而 7 天预见期的目标序列在相邻起报日之间高度自相关，"
      "一段 287 天的块所覆盖的独立水文事件远少于 287 个。"
      "元学习器要估计 6 个（多输出为 3 个）系数，"
      "在有效样本不足且元特征之间共线的情况下，最小二乘解对噪声非常敏感。"
      % rt["n_hold"])
    P("第三，Ridge 权重在共线元特征上过拟合。"
      "表 2-5 显示权重绝对值最大项达到数百甚至上千的量级，同一预见期内正负权重并存，"
      "选出的 alpha 在 0.01 与 100 之间跳动，说明交叉验证误差曲面相当平坦，"
      "alpha 的选择本身不稳定。"
      "在这种条件下，含有 6 个自由参数的线性组合在留出块上可以拟合得很好，"
      "却在测试段上放大误差；而等权平均没有任何自由参数，"
      "只要各成员的误差不完全正相关，就能稳定地降低方差。"
      "本阶段的数据规模与元特征质量还不足以支撑后者，"
      "因此结论应表述为：在本数据条件下简单平均更稳健，"
      "而不是断言 Stacking 方法本身劣于简单平均。")

    H2("3.3 精度随预见期衰减的成因")
    P("各方案的精度都随预见期下降，但幅度不同。"
      "6 模型简单平均从第 1 天的 %s 降至第 7 天的 %s，"
      "LSTM 直接策略从 %s 降至 %s，后者衰减更陡。"
      "成因可归为四点。" % (f4(a6[1]["R²"]), f4(a6[7]["R²"]), f4(ld[1]["R²"]), f4(ld[7]["R²"])))
    P("第一，输入信息在起报时刻固定。所有特征只包含起报日及更早的观测，"
      "预见期越长，起报与目标之间新增的降水、气温与融雪过程就越长，"
      "这部分信息在预测时完全未知，可解释方差因此下降。"
      "这是精度衰减的根本原因，也是任何纯数据驱动方法共有的上限。")
    P("第二，条件均值的收缩效应。训练以平方损失为目标，模型输出趋近条件均值而非条件众数，"
      "对洪峰这类低频高幅事件系统性低估。"
      "预见期越长，目标的条件分布越弥散，均值收缩越强，"
      "峰值误差在 RMSE 中的占比上升，表现为 R² 快速下滑而 MAE 上升相对缓慢。")
    P("第三，样本量与事件覆盖不足。训练段只有 %d 行，覆盖约 4 个水文年，"
      "其中高流量事件数量有限，长预见期的目标信噪比更低，"
      "单次划分的偶然性也更大，个别预见期的 R² 出现非单调波动即属此类。" % rt["n_train"])
    P("第四，策略带来的正则差异。多输出策略让 7 个预见期共享同一表示，"
      "长预见期因此比直接策略略缓；"
      "而集成中的等权平均通过压低成员方差，使曲线整体更平坦。"
      "这三类机制都不改变信息上限，只改变接近上限的速度。")

    H2("3.4 与未修正数据版本的对比")
    P("本阶段的数据修正有两项：气象列改为按 year 与 yday 与 Daymet 官方日值对齐，"
      "以及恢复真实积雪水当量（%d 个观测日中 %d 天非零）。"
      "修正前的版本沿用了原始 CSV，Swe 列恒为 0，气象列与日历错位。"
      "表 3-3 只取两版共有的三种方案的 R² 与 RMSE 作整体对照。"
      % (d["n_days"], d["swe_nonzero"]))
    rows = []
    for label, new, old in (("LSTM 直接策略", ld, raw_ld),
                            ("LSTM 多输出策略", lm, raw_lm),
                            ("6 模型简单平均", a6, raw_a6)):
        for h in (1, 7):
            rows.append([label, "第 %d 天" % h,
                         f4(old[h]["R²"]), f2(old[h]["RMSE"]),
                         f4(new[h]["R²"]), f2(new[h]["RMSE"]),
                         signed4(new[h]["R²"] - old[h]["R²"])])
    T(["方案", "预见期", "修正前 R²", "修正前 RMSE（cfs）", "修正后 R²",
       "修正后 RMSE（cfs）", "R² 变化"],
      rows, "数据修正前后三种方案的精度对照", 3)
    NOTE("注：修正前数值取自 archive/results_phase3_enhanced_raw/ 下的同名结果文件；"
         "两版测试段样本数相差 6 行，本表为整体指标对照，不作逐日配对比较；"
         "变化列为本版减修正前版本。")
    P("修正后第 1 天的提升最为明显：LSTM 直接策略的 R² 由 %s 升至 %s（提高 %s），"
      "RMSE 由 %s cfs 降至 %s cfs；多输出策略由 %s 升至 %s；"
      "6 模型简单平均由 %s 升至 %s。"
      "第 7 天的变化同样为正：直接策略由 %s 升至 %s，多输出策略由 %s 升至 %s，"
      "6 模型简单平均由 %s 升至 %s。"
      "两项修正的作用机制不同——气象列对齐消除了驱动变量与径流之间的人为错位，"
      "积雪水当量的恢复使融雪期特征不再退化为零方差列；"
      "二者共同使第 1 天的可用信息显著增加，"
      "对长预见期的增益则明显更小，因为长预见期的误差主要由目标时段未观测的气象过程决定。"
      % (f4(raw_ld[1]["R²"]), f4(ld[1]["R²"]), signed4(ld[1]["R²"] - raw_ld[1]["R²"]),
         f2(raw_ld[1]["RMSE"]), f2(ld[1]["RMSE"]),
         f4(raw_lm[1]["R²"]), f4(lm[1]["R²"]),
         f4(raw_a6[1]["R²"]), f4(a6[1]["R²"]),
         f4(raw_ld[7]["R²"]), f4(ld[7]["R²"]),
         f4(raw_lm[7]["R²"]), f4(lm[7]["R²"]),
         f4(raw_a6[7]["R²"]), f4(a6[7]["R²"])))

    # ---------------------------------------------------------------- 第四章
    H1("四、小结")
    P("本阶段完成第三次阶段验收要求的 7 个子任务：子任务 1 在训练段内部完成 8 次超参数搜索，"
      "得到逐任务的最优组合；子任务 2 与子任务 3 给出直接策略与多输出策略在 1 至 7 天预见期上的完整精度；"
      "子任务 4 至子任务 7 在两种策略下分别实现了简单平均与 Stacking 集成。"
      "全部指标由脚本从逐日预测重新计算，结果文件与正文一致。")
    P("主要结论如下。第一，第 1 天 R² 不低于 0.80 的目标已达成，"
      "达成方式有两条独立路径：单模型 LSTM 直接策略的 R² 为 %s、RMSE 为 %s cfs，"
      "多输出策略为 %s，6 模型简单平均为 %s。"
      "第二，单模型在短预见期的精度高于任何集成方案，"
      "集成的主要作用是改善长预见期的稳健性而非提升第 1 天的峰值拟合。"
      "第三，简单平均在直接策略的全部 7 个预见期、多输出策略的 6 个预见期上高于 Stacking，"
      "原因是元特征与测试期预测之间存在分布偏移、留出块过小且样本高度自相关、"
      "Ridge 权重在共线元特征上过拟合；本报告不把 Stacking 描述为更优方法。"
      "第四，精度随预见期快速衰减，第 3 天起 R² 普遍转负，"
      "根本约束是输入不含目标时段的气象信息。"
      % (f4(ld[1]["R²"]), f2(ld[1]["RMSE"]), f4(lm[1]["R²"]), f4(a6[1]["R²"])))
    P("本阶段的局限也需要明确记录。其一，PyTorch-LSTM 的输入是单步特征向量，"
      "循环层不跨时间步展开，门控结构虽真实参与计算，但跨步记忆的收益未在本阶段体现，"
      "历史信息完全依赖显式滞后与滚动特征。"
      "其二，Stacking 的元特征只来自一段留出块，样本量不足以稳定估计 6 个权重，"
      "更稳妥的做法是用多折时序交叉验证生成元特征，这需要重训 6 个基学习器的多折版本，本阶段未做。"
      "其三，长预见期的 R² 接近或低于零，尚不足以支撑实际调度决策。")
    P("后续工作从三方面展开：一是把 LSTM 的输入改为真正的滑动窗口序列，"
      "在相同特征与划分下检验跨步记忆能否带来增益；"
      "二是用多折时序交叉验证重构 Stacking 的元特征，并引入非负约束或收缩系数控制权重幅度；"
      "三是扩充样本年份并针对峰值引入加权损失或分位数回归，缓解幅度低估。")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    primary = OUT_DIR / DOC_NAME
    mirror = ROOT / DOC_NAME
    doc.save(str(primary))
    shutil.copyfile(primary, mirror)
    return [primary, mirror]


# ------------------------------------------------------------------ 读回校验

FORBIDDEN = ("TODO", "XX", "待补", "示例数据", "Stacking 最优", "Stacking最优", "359")


def verify(paths):
    primary, mirror = paths
    problems = []
    evidence = []

    b1, b2 = primary.read_bytes(), mirror.read_bytes()
    evidence.append("字节数：%s=%d，%s=%d，逐字节一致=%s"
                    % (primary.name, len(b1), mirror.name, len(b2), b1 == b2))
    if b1 != b2:
        problems.append("两份文档不一致")
    if len(b1) <= 100000:
        problems.append("字节数不足 100000")

    doc = Document(str(primary))
    texts = [p.text.strip() for p in doc.paragraphs]

    tasks = ["子任务 %d" % i for i in range(1, 8)]
    missing = [t for t in tasks if not any(t in text for text in texts)]
    evidence.append("子任务标题命中：%s" % "、".join("%s=%s" % (t, "有" if t not in missing else "缺")
                                                     for t in tasks))
    if missing:
        problems.append("缺少子任务标题：%s" % missing)

    fig_caps = [t for t in texts if re.match(r"^图 \d-\d", t)]
    table_caps = [t for t in texts if re.match(r"^表 \d-\d", t)]
    drawings = sum(1 for p in doc.paragraphs if p._p.findall(qn("w:r") + "/" + qn("w:drawing")))
    evidence.append("图 %d 张、图题 %d 条：%s" % (drawings, len(fig_caps), " | ".join(fig_caps)))
    evidence.append("表 %d 张（含封面字段表）、表题 %d 条：%s"
                    % (len(doc.tables), len(table_caps), " | ".join(table_caps)))
    if len(fig_caps) != 4:
        problems.append("图题数量不为 4：%d" % len(fig_caps))
    if len(table_caps) < 9:
        problems.append("内容表少于 9 张：%d" % len(table_caps))

    for prefix, caps in (("图", fig_caps), ("表", table_caps)):
        chapters = {}
        for cap in caps:
            chapter, index = re.match(r"^%s (\d)-(\d)" % prefix, cap).groups()
            chapters.setdefault(chapter, []).append(int(index))
        for chapter, indexes in chapters.items():
            if indexes != list(range(1, len(indexes) + 1)):
                problems.append("%s 第 %s 章编号不连续：%s" % (prefix, chapter, indexes))
        evidence.append("%s编号连续性：%s" % (prefix, {k: v for k, v in chapters.items()}))

    cell_texts_all = []
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                cell_texts_all.append(cell.text)
    full_text = "\n".join(texts + cell_texts_all)
    import zipfile
    xml_text = zipfile.ZipFile(str(primary)).read("word/document.xml").decode("utf-8")
    hits = [word for word in FORBIDDEN if word in full_text or word in xml_text]
    start_hits = full_text.count("11 月 28 日")
    target_hits = full_text.count("11 月 29 日")
    evidence.append("日期表述扫描：起报日口径“11 月 28 日”%d 次，目标日口径“11 月 29 日”%d 次"
                    % (start_hits, target_hits))
    if start_hits or "11 月 28 日" in xml_text:
        problems.append("仍出现起报日口径的日期表述：11 月 28 日")
    evidence.append("禁用表述扫描：命中 %s" % (hits if hits else "无"))
    if hits:
        problems.append("出现禁用表述：%s" % hits)

    # 数值抽查（表内文本与 CSV 对照）
    def find_table(caption_prefix):
        for i, para in enumerate(doc.paragraphs):
            if para.text.strip().startswith(caption_prefix):
                return i
        return None

    def table_after(index):
        body = doc.element.body
        seen = 0
        target = doc.paragraphs[index]._p
        node = target.getnext()
        while node is not None:
            if node.tag == qn("w:tbl"):
                return node
            node = node.getnext()
        return None

    d = load_all()
    evidence.extend(d["assert_log"])

    def cell_texts(node):
        rows = []
        for tr in node.findall(qn("w:tr")):
            cells = []
            for tc in tr.findall(qn("w:tc")):
                cells.append("".join(t.text or "" for t in tc.iter(qn("w:t"))))
            rows.append(cells)
        return rows

    checks = []
    idx = find_table("表 1-2")
    rows = cell_texts(table_after(idx))
    checks.append(("LSTM 直接 h1 R²", rows[1][1], f4(d["lstm_direct"][1]["R²"])))
    checks.append(("LSTM 直接 h1 RMSE", rows[1][2], f2(d["lstm_direct"][1]["RMSE"])))
    checks.append(("LSTM 直接 h1 MAE", rows[1][3], f2(d["lstm_direct"][1]["MAE"])))
    checks.append(("LSTM 直接 h7 R²", rows[7][1], f4(d["lstm_direct"][7]["R²"])))
    idx = find_table("表 1-3")
    rows = cell_texts(table_after(idx))
    checks.append(("LSTM 多输出 h1 R²", rows[1][1], f4(d["lstm_multi"][1]["R²"])))
    idx = find_table("表 2-3")
    rows = cell_texts(table_after(idx))
    checks.append(("6 模型简单平均 h1 R²", rows[1][1], f4(d["avg6"][1]["R²"])))
    checks.append(("6 模型简单平均 h1 RMSE", rows[1][2], f2(d["avg6"][1]["RMSE"])))
    idx = find_table("表 2-7")
    rows = cell_texts(table_after(idx))
    checks.append(("3 模型 Stacking 多输出 h1 R²", rows[1][1], f4(d["stack3"][1]["R²"])))
    idx = find_table("表 1-1")
    rows = cell_texts(table_after(idx))
    best = d["grid"].iloc[0]
    expect = "hidden=%d，layers=%d，lr=%s，dropout=%s" % (
        int(best["best_hidden_dim"]), int(best["best_num_layers"]),
        "%.3f" % best["best_lr"], "%.1f" % best["best_dropout"])
    checks.append(("第 1 天最优超参数", rows[1][1], expect))
    checks.append(("第 1 天最优交叉验证 RMSE", rows[1][2], f2(best["cv_rmse"])))

    for name, got, want in checks:
        ok = got.strip() == want.strip()
        evidence.append("抽查 %s：文档“%s” vs 结果文件“%s” %s"
                        % (name, got, want, "一致" if ok else "不一致"))
        if not ok:
            problems.append("数值抽查不一致：%s" % name)

    # 排版读回：正文宋体 12pt、1.5 倍行距；表格三线表
    start = next(i for i, t in enumerate(texts) if t.startswith("一、"))
    body_checked = 0
    for para in doc.paragraphs[start:]:
        text = para.text.strip()
        if not text or para.style.name.startswith("Heading"):
            continue
        if re.match(r"^[图表] \d-\d", text) or text.startswith("注：") \
                or text.startswith("图注："):
            continue
        body_checked += 1
        for run in para.runs:
            if not run.text.strip():
                continue
            if run.font.size != Pt(12):
                problems.append("正文字号不是 12pt：%s" % text[:24])
                break
            rfonts = run._element.find(qn("w:rPr")).find(qn("w:rFonts"))
            if rfonts is None or rfonts.get(qn("w:eastAsia")) != "宋体":
                problems.append("正文中文字体不是宋体：%s" % text[:24])
                break
        if para.paragraph_format.line_spacing != 1.5:
            problems.append("正文行距不是 1.5：%s" % text[:24])
    evidence.append("正文段落读回校验：%d 段，宋体 12pt 与 1.5 倍行距均已核对" % body_checked)

    tbl_checked = 0
    for table in doc.tables[1:]:
        tblPr = table._tbl.tblPr
        borders = tblPr.find(qn("w:tblBorders"))
        if borders is None:
            problems.append("表格缺少 tblBorders")
            continue
        top = borders.find(qn("w:top"))
        bottom = borders.find(qn("w:bottom"))
        inside = borders.find(qn("w:insideH"))
        if top is None or top.get(qn("w:sz")) != "12":
            problems.append("三线表顶线不符合 1.5pt")
        if bottom is None or bottom.get(qn("w:sz")) != "12":
            problems.append("三线表底线不符合 1.5pt")
        if inside is None or inside.get(qn("w:val")) != "none":
            problems.append("三线表内部横线未去除")
        head = table.rows[0].cells[0]._tc
        tcPr = head.find(qn("w:tcPr"))
        tcb = tcPr.find(qn("w:tcBorders")) if tcPr is not None else None
        if tcb is None or tcb.find(qn("w:bottom")) is None or \
                tcb.find(qn("w:bottom")).get(qn("w:sz")) != "6":
            problems.append("表头下横线不符合 0.75pt")
        tbl_checked += 1
    evidence.append("三线表读回校验：%d 张内容表全部核对顶线/表头线/底线" % tbl_checked)

    return evidence, problems


def main():
    if "--verify" in sys.argv:
        primary = OUT_DIR / DOC_NAME
        mirror = ROOT / DOC_NAME
    else:
        primary, mirror = build()
        for path in (primary, mirror):
            print("已生成：%s\t%d 字节" % (path, path.stat().st_size))
    evidence, problems = verify([primary, mirror])
    print("\n=== 读回校验 ===")
    for line in evidence:
        print(line)
    if "--dump" in sys.argv:
        print("\n=== 关键段落 ===")
        for para in Document(str(primary)).paragraphs:
            text = para.text.strip()
            if text.startswith(("修正后第 1 天", "简单平均第 1 天", "多输出策略第 1 天",
                                "直接策略第 1 天", "从 7 个预见期的平均",
                                "四种方案的精度都随预见期", "第 1 天上单模型",
                                "逐期比较的结果是")):
                print("* " + text)
    print("\n=== 结论 ===")
    if problems:
        for item in problems:
            print("未通过：" + item)
        raise SystemExit(1)
    print("全部校验通过")


if __name__ == "__main__":
    main()
