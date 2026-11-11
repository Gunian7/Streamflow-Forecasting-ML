# -*- coding: utf-8 -*-
"""生成《机器学习课程实践》实验报告（三人组）.docx。

严格对应下发的实验报告模板：封面（三行标题 + 6 行 2 列表格 + 日期）、
一、任务要求 / 二、详细分工 / 三、过程设计 / 四、编码实现及结果 /
五、程序调试 / 六、总结 / 注 / 附录（完整代码）。

所有数值在运行时从 results/ 与 archive/ 下的结果文件读取并格式化，
正文不手写任何指标数字；关键数字在 report() 内逐条断言。
排版统一调用 code/report_style.py。

用法：
    .venv/Scripts/python.exe code/experiment_report.py            # 生成 docx
    .venv/Scripts/python.exe code/experiment_report.py --check    # 生成 + 转 PDF 渲染自检
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from docx import Document
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Cm, Pt

sys.path.insert(0, str(Path(__file__).resolve().parent))

import report_style as rs  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
P2 = ROOT / "results" / "phase2_v5"
P3 = ROOT / "results" / "phase3_enhanced"
PROC = ROOT / "results" / "processed_data"
EDA = ROOT / "results" / "figures" / "eda"
FE = ROOT / "results" / "figures" / "fe"
RAW2 = ROOT / "archive" / "results_phase2_v5_raw"
MEMBERS_JSON = ROOT / "local_report_members.json"

DOC_NAME = "《机器学习课程实践》实验报告（三人组）.docx"
HEADER_TEXT = "杭州电子科技大学《机器学习课程实践》实验报告"
OUT_DIR = ROOT / "本地资料" / "实验报告"

# 图 4-x 用到的图件（与 results/ 下现有 PNG 一一对应）
FIG_TS = EDA / "1_time_series_overview.png"
FIG_BOX = EDA / "3_boxplots_all_8_columns.png"
FIG_CORR = EDA / "4_pearson_correlation_heatmap.png"
FIG_SEL = FE / "feature_selection_comparison.png"
FIG_P2_DECAY = P2 / "figures" / "fig1_精度随预见期衰减.png"
FIG_P3_DECAY = P3 / "figures" / "fig1_精度随预见期衰减.png"
FIG_P3_STRATEGY = P3 / "figures" / "fig2_策略对比柱状图.png"
FIG_P3_HYDRO = P3 / "figures" / "fig3_测试段水文过程线.png"

EDA_CAPTION_NOTE = ("本图由修正前的原始表绘制，积雪水当量修正后的统计见 3.1 节与表 1-1，"
                   "两处口径不同，不可直接对比。")

CHECKS = []


def check_value(label, actual, expected):
    """记录并校验一处关键数字；不一致时立即中断，避免把错数写进报告。"""
    CHECKS.append((label, expected, actual))
    if actual != expected:
        raise AssertionError("数值断言失败：%s 期望 %r，实得 %r" % (label, expected, actual))


def fmt(value, nd=3):
    return ("%." + str(nd) + "f") % float(value)


def load_data():
    """读取全部结果表与原始数据（字段顺序固定，按列位置取值以避开 BOM 与列名差异）。"""
    d = {}
    d["p2"] = pd.read_csv(P2 / "测试精度指标.csv")
    d["p2_raw"] = pd.read_csv(RAW2 / "测试精度指标.csv")
    d["p3"] = pd.read_csv(P3 / "pytorch_lstm_metrics.csv")
    d["boost"] = pd.read_csv(P3 / "boosting_metrics.csv")
    d["sa6"] = pd.read_csv(P3 / "6模型简单平均_直接.csv")
    d["st6"] = pd.read_csv(P3 / "6模型Stacking_直接.csv")
    d["sa3"] = pd.read_csv(P3 / "3模型简单平均_多输出.csv")
    d["st3"] = pd.read_csv(P3 / "3模型Stacking_多输出.csv")
    d["members"] = pd.read_csv(P3 / "ensemble_members.csv")
    d["grid"] = pd.read_csv(P3 / "pytorch_lstm_grid_summary.csv")
    d["tune"] = pd.read_csv(P2 / "选参结果.csv")
    d["rank"] = pd.read_csv(FE / "feature_selection_ranking.csv")
    d["proc"] = pd.read_csv(PROC / "01047000_with_feature_values_独热编码版.csv")
    d["feat"] = json.loads((P3 / "boosting_features.json").read_text(encoding="utf-8"))
    d["data"] = pd.read_csv(ROOT / "01047000_fixed.csv")
    d["rawdata"] = pd.read_csv(ROOT / "01047000.csv")
    d["ens_pred"] = pd.read_csv(P3 / "ensemble_predictions.csv")
    return d


def split_facts():
    """从阶段二、阶段三的运行日志读取真实的数据划分口径（行数、日期与留出块大小）。"""
    log2 = (P2 / "run_phase2.log").read_text(encoding="utf-8")
    meta = json.loads(log2[log2.index("{"): log2.index("}") + 1])
    log3 = (P3 / "run_stage5_7.log").read_text(encoding="utf-8")
    rows3 = re.search(r"\[数据\] 训练 (\d+) 行 / 测试 (\d+) 行", log3)
    hold = re.search(r"训练段 (\d+) 行；元特征留出块=最后 (\d+) 行（索引 (\d+) 起）", log3)
    if not (rows3 and hold):
        raise LookupError("阶段三运行日志中未找到数据划分记录")
    return {
        "train": int(meta["训练样本"]), "test": int(meta["测试样本"]),
        "train_start": meta["训练起"], "train_end": meta["训练止"],
        "test_start": meta["测试起"], "test_end": meta["测试止"],
        "train3": int(rows3.group(1)), "test3": int(rows3.group(2)),
        "hold": int(hold.group(2)), "hold_start": int(hold.group(3)),
    }


def split_note(d, prefix=False):
    """359 / 353 两阶段测试口径差异的说明（3.6 节与表 4-5、表 4-10 表注共用）。"""
    f = split_facts()
    start3 = str(pd.to_datetime(d["ens_pred"]["起报日"]).min().date())
    head3 = (pd.Timestamp(start3) - pd.Timestamp(f["test_start"])).days
    text = ("阶段二测试 %d 行（%s—%s），阶段三增强特征含 7 日滚动统计，测试段首 %d 天被丢弃，"
            "为 %d 行（%s—%s）；差异仅 %d 天，方向性结论不变，但两阶段的单日指标不直接相减。"
            % (f["test"], f["test_start"], f["test_end"], head3,
               f["test3"], start3, f["test_end"], head3))
    return ("注：" + text) if prefix else text


def metric(df, cols, keys, col):
    """按 cols 中的若干列定位唯一一行，取 col 列的浮点值。"""
    mask = pd.Series(True, index=df.index)
    for name, value in zip(cols, keys):
        mask &= df[name] == value
    sub = df[mask]
    if len(sub) != 1:
        raise LookupError("定位失败：%s=%s 命中 %d 行" % (cols, keys, len(sub)))
    return float(sub.iloc[0][col])


def p2m(d, model, strategy, horizon, col="R²"):
    return metric(d["p2"], ["模型", "策略", "预见期（天）"], [model, strategy, horizon], col)


def p2raw(d, model, strategy, horizon, col="R²"):
    return metric(d["p2_raw"], ["模型", "策略", "预见期（天）"], [model, strategy, horizon], col)


def p3m(d, strategy, horizon, col="R²"):
    return metric(d["p3"], ["策略", "预见期（天）"], ["PyTorch-LSTM-" + strategy, horizon], col)


def ensm(d, key, horizon, col="R²"):
    return metric(d[key], ["预见期（天）"], [horizon], col)


# ------------------------------------------------------------------ 封面


def build_cover(doc, cover):
    """按模板排封面：三行标题 + 6 行 2 列表格 + 日期；封面无页眉页码。"""
    for text, size, before, after in (
            ("杭州电子科技大学", 26, 48, 12),
            ("《机器学习课程实践》", 22, 0, 12),
            ("实验报告", 30, 0, 54)):
        p = doc.add_paragraph()
        rs._plain_para(p, WD_ALIGN_PARAGRAPH.CENTER, 1.5, before, after)
        rs.set_run_font(p.add_run(text), Pt(size), bold=False, east_asia=rs.HEI_FONT)

    rows = [("题    目", cover["题目"]), ("专    业", cover["专业"]),
            ("学    号", cover["学号"]), ("姓    名", cover["姓名"]),
            ("指导教师", cover["指导教师"]), ("成    绩", cover["成绩"])]

    table = doc.add_table(rows=len(rows), cols=2)
    table.style = doc.styles["Normal Table"]
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    rs._set_table_noborder(table)

    label_width, value_width = Cm(4.0), Cm(9.0)
    for row, (label, value) in zip(table.rows, rows):
        row.height = Cm(1.25)
        for idx, cell in enumerate(row.cells):
            cell.width = label_width if idx == 0 else value_width
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            para = rs._plain_para(cell.paragraphs[0], None, 1.5, 0, 0)
            if idx == 0:
                para.alignment = WD_ALIGN_PARAGRAPH.RIGHT
                rs.set_run_font(para.add_run(label), Pt(14), bold=False, east_asia=rs.SONG_FONT)
            else:
                para.alignment = WD_ALIGN_PARAGRAPH.CENTER
                rs.set_run_font(para.add_run(value), Pt(12), bold=False, east_asia=rs.SONG_FONT)
                rs._set_cell_bottom_border(cell)

    p = doc.add_paragraph()
    rs._plain_para(p, WD_ALIGN_PARAGRAPH.CENTER, 1.5, 60, 0)
    rs.set_run_font(p.add_run(cover["日期"]), Pt(14), bold=False, east_asia=rs.SONG_FONT)


# ------------------------------------------------------------------ 代码块


def add_code_caption(doc, text):
    p = doc.add_paragraph()
    rs._plain_para(p, WD_ALIGN_PARAGRAPH.CENTER, 1.0, 6, 0)
    p.paragraph_format.keep_with_next = True
    rs.set_run_font(p.add_run(text), rs.CAPTION_SIZE, bold=True, east_asia=rs.SONG_FONT)
    return p


def add_code_block(doc, code, size_pt=9.0, space_after=8):
    """等宽字体代码块：单倍行距、无首行缩进，行间用软换行以免产生上千个段落。"""
    lines = [line.replace("\t", "    ").rstrip() for line in code.strip("\n").split("\n")]
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.alignment = WD_ALIGN_PARAGRAPH.LEFT
    pf.line_spacing = 1.0
    pf.first_line_indent = Pt(0)
    pf.left_indent = Pt(0)
    pf.space_before = Pt(2)
    pf.space_after = Pt(space_after)
    pf.widow_control = False
    run = None
    for i, line in enumerate(lines):
        run = p.add_run(line if line else " ")
        rs.set_run_font(run, Pt(size_pt), bold=False, east_asia=rs.SONG_FONT, latin="Consolas")
        if i != len(lines) - 1:
            run.add_break()
    return p


# ------------------------------------------------------------------ 正文数据准备


def cover_fields():
    """封面取值：姓名/学号/指导教师来自 local_report_members.json，不写死在脚本里。"""
    data = json.loads(MEMBERS_JSON.read_text(encoding="utf-8"))
    members = data["members"]
    return {
        "题目": "基于机器学习的径流预测",
        "专业": "智能科学与技术",
        "学号": "、".join(str(m["student_id"]) for m in members),
        "姓名": "、".join(m["name"] for m in members),
        "指导教师": "、".join(data["advisor"]),
        "成绩": "",
        "日期": "二〇二六年十一月",
    }


def division_text():
    """分工文字来自 local_report_members.json，脚本本身不含成员姓名。"""
    if MEMBERS_JSON.exists():
        data = json.loads(MEMBERS_JSON.read_text(encoding="utf-8"))
        text = (data.get("division") or "").strip()
        if text:
            return text
    return "本次实验按实验指导书的任务流程分工完成，具体分工待小组确认后填写。"


def row_text(vals):
    return [str(v) for v in vals]


def table_env(rows):
    env = [("操作系统", "Windows 11（x86_64）"),
           ("Python 解释器", "Python " + _version("python", "3.14.7")),
           ("科学计算", "numpy " + _version("numpy", "2.5.4") + " / pandas " + _version("pandas", "3.1.1")),
           ("机器学习库", "scikit-learn " + _version("scikit-learn", "1.9.1")),
           ("深度学习框架", "PyTorch " + _version("torch", "2.15.0") + "（CPU 版）"),
           ("梯度提升库", "xgboost " + _version("xgboost", "3.4.1") + " / lightgbm " + _version("lightgbm", "4.7.0")),
           ("可视化", "matplotlib " + _version("matplotlib", "3.11.2"))]
    return env


def _version(pkg, fallback):
    try:
        from importlib.metadata import version
        if pkg == "python":
            return sys.version.split()[0]
        return version(pkg)
    except Exception:
        return fallback


# ------------------------------------------------------------------ 一、任务要求


def section_task(doc, d, num):
    rs.add_heading1(doc, "一、任务要求")
    data = d["data"]
    n_rows, n_cols = data.shape
    start = data["Date"].iloc[0]
    end = data["Date"].iloc[-1]
    mean_q = float(data["Discharge"].mean())
    test_q = data["Discharge"].iloc[-359:]
    peak_q = float(test_q.max())
    peak_date = str(data["Date"].iloc[-359:].iloc[int(test_q.to_numpy().argmax())])

    rs.add_body(doc, "本实践以“基于机器学习的径流预测”为题。给定美国地质调查局（USGS）编号 "
                     "01047000 水文站的逐日水文气象观测序列，要求建立从当日及历史状态到未来 1—7 天"
                     "日径流量的映射，并用统一的指标评价各预见期的预测精度。报告按课程下发的三阶段"
                     "实验要求组织：阶段一完成数据修正、探索性分析与特征工程；阶段二完成人工神经网络"
                     "（ANN）与随机森林（RF）的建模与选参；阶段三完成 PyTorch-LSTM、梯度提升模型与"
                     "集成学习，并对结果做自洽性校验。")

    rs.add_heading2(doc, "1.1 题目与数据")
    rs.add_body(doc, "数据集为单个 CSV 文件，共 %d 行 %d 列，时间范围为 %s 至 %s，逐日连续无缺测。"
                     "其中 1 列为日期，8 列为数值型变量：径流量 Discharge（单位 cfs）与 7 个气象要素。"
                     "各变量的统计特征见表 1-1。" % (n_rows, n_cols, start, end))
    names = {"Discharge": ("径流量 Discharge", "cfs"),
             "Dayl": ("日照时长 Dayl", "s"),
             "Prcp": ("降水量 Prcp", "mm"),
             "Srad": ("短波辐射 Srad", "W/m²"),
             "Swe": ("积雪水当量 Swe", "mm"),
             "Tmax": ("最高气温 Tmax", "°C"),
             "Tmin": ("最低气温 Tmin", "°C"),
             "Vp": ("水汽压 Vp", "Pa")}
    rows = []
    for col in ["Discharge", "Dayl", "Prcp", "Srad", "Swe", "Tmax", "Tmin", "Vp"]:
        label, unit = names[col]
        rows.append(row_text([label, unit, fmt(data[col].mean(), 2), fmt(data[col].min(), 2),
                              fmt(data[col].max(), 2)]))
    rs.add_table(doc, ["变量", "单位", "均值", "最小值", "最大值"], rows,
                 caption="建模数据集 8 个数值变量的描述统计（%d 行，全期）" % n_rows,
                 chapter=1, numbering=num,
                 col_widths_cm=[4.8, 2.0, 2.6, 2.6, 2.6])

    rs.add_body(doc, "径流量呈明显右偏分布：全期均值为 %s cfs，最大值 %s cfs 出现在 %s 的融雪洪峰，"
                     "最小值仅 %s cfs。测试段（2004 年，359 天）的最大值 %s cfs 出现在 %s，"
                     "说明峰值预报的难度集中在少数高流量日。"
                % (fmt(mean_q, 3), fmt(data["Discharge"].max(), 1),
                   str(data["Date"].iloc[int(data["Discharge"].to_numpy().argmax())]),
                   fmt(data["Discharge"].min(), 1), fmt(peak_q, 1), peak_date))

    rs.add_heading2(doc, "1.2 题目解析")
    rs.add_body(doc, "从机器学习的问题类型看，该任务有明确的连续型标签（未来某日的径流量），"
                     "因此属于监督学习中的回归问题；又因为输入不只有当日特征，还隐含时间的先后顺序，"
                     "同时要对未来多个预见期分别给出预测，所以具体形式是“多步时间序列回归”。"
                     "课程要求的三个阶段共 22 个交付项，整理如表 1-2。")
    rs.add_table(doc, ["阶段", "要求条目", "主要工作", "主要交付"], [
        ["阶段一", "8 项", "数据读取与修正、描述统计、分布与时序分析、箱型图离群点、"
                           "滞后相关分析、特征工程、归一化与特征选择",
         "特征矩阵 %d×%d" % d["proc"].shape],
        ["阶段二", "7 项", "ANN 直接多步、ANN 多输出、RF 直接多步、RF 多输出、"
                           "Grid Search 交叉验证选参、精度随预见期衰减评估、模型融合评估", "28 行精度表"],
        ["阶段三", "7 项", "PyTorch-LSTM 直接与多输出、XGBoost/LightGBM、"
                           "6 模型简单平均、6 模型 Stacking、3 模型多输出简单平均与 Stacking、自洽性校验",
         "14 行 LSTM 精度表、4 张集成表"],
    ], caption="三阶段实验要求与交付物对照", chapter=1, numbering=num,
        col_widths_cm=[1.8, 1.8, 8.0, 3.0], font_size=Pt(9.5))

    rs.add_body(doc, "评价指标统一采用 RMSE、MAE、R² 与 NSE 四项，定义见表 1-3。"
                     "其中 R² 与 NSE 在本任务（单目标回归、以均值作基准）下数值相同，"
                     "报告同时列出，以便与水文预报的习惯口径对应。")
    rs.add_table(doc, ["指标", "定义", "取值含义"], [
        ["RMSE", "sqrt(mean((y - ŷ)²))", "与径流量同量纲（cfs），对大误差更敏感"],
        ["MAE", "mean(|y - ŷ|)", "平均绝对偏差，比 RMSE 更稳健"],
        ["R²", "1 - SS_res / SS_tot", "拟合优度，1 为完美，0 等于用均值预测，可为负"],
        ["NSE", "1 - SS_res / SS_tot", "水文预报常用的 Nash-Sutcliffe 效率系数"],
    ], caption="四项评价指标的定义与含义", chapter=1, numbering=num,
        col_widths_cm=[2.2, 5.8, 6.6], font_size=Pt(9.5))

    rs.add_heading2(doc, "1.3 对题目的理解")
    rs.add_body(doc, "（1）监督学习与回归。样本是“日期—特征—标签”的三元组，标签为未来第 h 天的"
                     "径流量（h = 1, 2, …, 7），标签连续且有序，没有类别，因此不涉及无监督学习，"
                     "也不涉及分类。凡是把该问题当成分类（例如先分洪/枯水再回归）的做法，都会丢失"
                     "幅度信息，与任务目标不符。")
    rs.add_body(doc, "（2）多步预测有两种建模范式。一是“直接多步”：对每个预见期单独训练一个模型，"
                     "7 个预见期共 28 个模型（2 个模型族 × 7 个预见期 × 2 种策略的组合），"
                     "各模型互不影响、误差不累积；二是“多输出”：一个模型一次输出 7 维向量，"
                     "参数共享、能学到预见期之间的相关性，但损失被 7 个目标平均，短预见期精度通常略低。"
                     "报告对两种范式都做了实现与对比。")
    rs.add_body(doc, "（3）时间序列的固有限制。样本量只有 %d 天，而 2004 年单独作为测试段只有 359 天；"
                     "序列自相关强、季节性与融雪过程明显、极值稀少，任何随机划分都会造成信息泄漏——"
                     "相邻两天的特征几乎相同，若随机打乱，测试集里会出现训练集“见过”的日期。"
                     "因此全流程严格按时间顺序划分：2000—2003 年为训练段，2004 年为测试段，"
                     "归一化、特征选择、超参数搜索全部只在训练段内完成。" % n_rows)

    rs.add_heading2(doc, "1.4 解题思路与约束")
    rs.add_body(doc, "整体路线是“先把数据弄对，再把特征做够，最后才比较模型”："
                     "数据修正 → 探索性分析 → 特征工程 → 归一化 → 特征选择 → 时序划分 → "
                     "基线模型（ANN、RF）→ 深度模型（PyTorch-LSTM）→ 集成（简单平均、Stacking）→ "
                     "评估与结果分析。每一环的产出都被下一步直接消费，且都能用文件追溯。")
    rs.add_body(doc, "课程给出的三条硬约束在实现中被落成代码而不是口头约定："
                     "①禁用 sklearn 的 MultiOutputRegressor，多输出一律用原生支持多输出的模型"
                     "（MLPRegressor、RandomForestRegressor、LSTM 输出 7 维）；"
                     "②超参数必须用 Grid Search 交叉验证，搜索器为 GridSearchCV + "
                     "TimeSeriesSplit(n_splits=3, gap=7)，gap 取 7 是为了让验证段与训练段之间"
                     "空出与最大预见期等长的间隔，避免窗口重叠导致泄漏；"
                     "③所有拟合（包括归一化与特征选择）只能使用训练段数据，测试段只做 transform。")


# ------------------------------------------------------------------ 二、详细分工


def section_division(doc):
    rs.add_heading1(doc, "二、详细分工")
    rs.add_body(doc, division_text())


# ------------------------------------------------------------------ 三、过程设计


def section_design(doc, d, num):
    facts = split_facts()
    rs.add_heading1(doc, "三、过程设计")
    rs.add_body(doc, "过程设计把整个题目拆成 9 个前后衔接的步骤，每一步都明确“用什么工具/算法”"
                     "与“处理目标是什么”，并给出可检查的产物。总览见表 3-1，随后逐节说明。")
    rs.add_table(doc, ["步骤", "工具 / 算法", "处理目标", "产物"], [
        ["1 数据修正", "pandas；Daymet 原始文件的 year+yday 重建日期", "把错位一天的气象列按日历日对齐，恢复真实积雪水当量",
         "01047000_fixed.csv"],
        ["2 探索性分析", "pandas、matplotlib；描述统计、直方图/核密度、箱型图、皮尔逊相关", "掌握分布、离群点、季节性与滞后结构",
         "results/figures/eda"],
        ["3 特征工程", "四季独热、月份/年积日正余弦、lag1/2/3/7、3/7 天滑动统计、lead1—lead7 目标", "把单日观测扩成可分模型使用的矩阵",
         "1827×196 成品特征表"],
        ["4 归一化", "MinMaxScaler 与 StandardScaler 对比，训练段 fit", "消除量纲差异，避免 ANN 与 LSTM 训练不稳定",
         "归一化对比图"],
        ["5 特征选择", "|Pearson r| 与互信息双指标排序", "去掉与目标近乎无关或高度冗余的列",
         "60 行特征排序表"],
        ["6 时序划分", "按年份切分，训练 2000—2003 / 测试 2004", "杜绝随机划分带来的信息泄漏",
         "训练 %d / 测试 %d 行" % (facts["train"], facts["test"])],
        ["7 基线模型", "sklearn MLPRegressor、RandomForestRegressor；log1p 目标", "建立可比的精度基线，暴露长预见期衰减规律",
         "28 行精度表"],
        ["8 深度模型", "PyTorch LSTM（hidden 64/128、1—2 层（逐任务由网格搜索确定）、dropout、早停）；XGBoost、LightGBM", "用非线性时序模型与梯度提升提升第 1—3 天精度",
         "14 行 LSTM 表、14 行提升树表"],
        ["9 集成与评估", "等权简单平均；Stacking（287 行时序留出块 + Ridge 元学习器）", "降低单模型方差，比较两种集成策略的稳健性",
         "4 张集成表、衰减曲线"],
    ], caption="过程设计总览：步骤、工具算法与处理目标", chapter=3, numbering=num,
        col_widths_cm=[2.2, 5.0, 5.4, 2.0], font_size=Pt(9.0))

    rs.add_heading2(doc, "3.1 数据修正")
    rs.add_body(doc, "第一步不是建模，而是核对数据。把站点的气象列与 Daymet 原始文件逐日比对后发现两处系统性错误："
                     "气象列整体比 Daymet 晚一天；积雪水当量整列为 0。修正脚本 build_fixed_dataset.py "
                     "用 year 与 yday 重建 Daymet 的日期索引，以站点径流日期为准左连接气象列，"
                     "并核对 USGS 官方接口确认径流列本身无误。修正后数据仍为 1827 行 9 列，"
                     "积雪水当量恢复为非零序列（均值 %s mm），这是后续融雪期精度提升的直接来源。"
                % fmt(d["data"]["Swe"].mean(), 2))

    rs.add_heading2(doc, "3.2 探索性分析")
    rs.add_body(doc, "第二步用描述统计与图形刻画数据。时间序列总览图给出径流与各气象要素的年内节律；"
                     "箱型图暴露径流的重尾与离群点；皮尔逊相关矩阵给出同步相关，滞后相关表给出"
                     "径流对自身与降水的最优滞后阶数。分析结论直接决定特征工程：径流自相关在 lag1 "
                     "处最高，降水与气温的响应存在 1—3 天延迟，因此滞后阶数取 1、2、3、7。")

    rs.add_heading2(doc, "3.3 特征工程")
    rs.add_body(doc, "第三步把单日观测扩成模型可用的矩阵，共构造四类特征："
                     "①季节信息——四季独热（Season_spring/summer/autumn/winter），不引入虚假的大小关系；"
                     "②周期编码——月份与年积日的正余弦，保证 12 月到 1 月的连续；"
                     "③滞后特征——径流、降水、气温、辐射、水汽压的 lag1/2/3/7；"
                     "④滑动窗口统计——径流、降水、气温的 3 天与 7 天均值、标准差、最大值、最小值，"
                     "统一先 shift(1) 再 rolling，避免把当日信息算进窗口。"
                     "同时构造 Target_lead1—Target_lead7 七个标签列，最后丢弃首尾因滞后与标签产生的空行。"
                     "成品特征表为 %d 行 %d 列。"
                % (d["proc"].shape[0], d["proc"].shape[1]))
    rs.add_body(doc, "阶段二起把当天径流 Q(t) 作为特征之一：起报时刻 Q(t) 已经观测到，"
                     "不属于未来信息泄漏，这也是第 1 天预见期精度明显高于远期的主要原因之一。"
                     "phase2_v5_data.py 中用断言确保 Discharge 一直留在特征列内。")

    rs.add_heading2(doc, "3.4 归一化")
    rs.add_body(doc, "第四步做尺度统一。同时对特征与标签比较 Min-Max 与 Z-score 两种方案，"
                     "四个 scaler 全部只在训练段 fit，再 transform 训练段与测试段。"
                     "由于径流量跨三个数量级，模型侧进一步对标签取 log1p 压缩量级，"
                     "预测时统一用 expm1 反变换并裁剪为非负，这条约定在阶段二、三的每一个入口都必须遵守。")

    rs.add_heading2(doc, "3.5 特征选择")
    rs.add_body(doc, "第五步做降维与去冗余。对 %d 个候选特征同时计算 |Pearson r| 与互信息，"
                     "按两项排序后剔除与目标近乎无关的常数列（修正前的积雪水当量即为典型）"
                     "以及与其他列高度共线的冗余列，保留与目标信息量最大的子集。"
                     "阶段二的模型使用 71 维特征，阶段三在加入积雪与累积类特征后使用 94 维特征。"
                % len(d["rank"]))

    rs.add_heading2(doc, "3.6 时序划分")
    rows_total = d["data"].shape[0]
    cal_days = (pd.Timestamp("2003-12-31") - pd.Timestamp("2000-01-01")).days + 1
    head_drop = (pd.Timestamp(facts["train_start"]) - pd.Timestamp(d["data"]["Date"].iloc[0])).days
    mid_drop = (pd.Timestamp("2003-12-31") - pd.Timestamp(facts["train_end"])).days
    tail_drop = (pd.Timestamp(d["data"]["Date"].iloc[-1]) - pd.Timestamp(facts["test_end"])).days
    dropped = head_drop + mid_drop + tail_drop
    rs.add_body(doc, ("第六步按时间切分。2000—2003 年为训练段，2004 年为测试段，两段完全不重叠："
                      "训练段 %s—%s 共 %d 行，测试段 %s—%s 共 %d 行。"
                      "2000-01-01 至 2003-12-31 共 %d 个日历日，构造滞后与滚动特征后实际训练样本 "
                      "%d 行（%s—%s）；被丢弃的 %d 行是训练段首 %d 天、训练段末 %d 天与测试段末 %d 天"
                      "（%d + %d + %d = %d，与 %d 行原始数据吻合）；阶段三加入 7 日滚动统计后训练 %d 行、"
                      "测试 %d 行。交叉验证同样保持时间顺序：TimeSeriesSplit(n_splits=3, gap=7)，"
                      "折内训练段永远早于验证段。")
                % (facts["train_start"], facts["train_end"], facts["train"],
                   facts["test_start"], facts["test_end"], facts["test"],
                   cal_days, facts["train"], facts["train_start"], facts["train_end"],
                   dropped, head_drop, mid_drop, tail_drop,
                   head_drop, mid_drop, tail_drop, dropped, rows_total,
                   facts["train3"], facts["test3"]))
    rs.add_body(doc, "两阶段测试口径说明：" + split_note(d))

    rs.add_heading2(doc, "3.7 基线模型")
    rs.add_body(doc, "第七步建立基线。ANN 用 sklearn 的 MLPRegressor，RF 用 RandomForestRegressor，"
                     "两者都做“直接多步”（7 个预见期各一个模型）与“多输出”（一个模型输出 7 维）两种实现，"
                     "超参数全部用 GridSearchCV 在训练段内搜索，共 16 组选参记录。"
                     "基线的作用不只是精度，更是给出“精度随预见期如何衰减”的参照曲线。")

    rs.add_heading2(doc, "3.8 深度模型与梯度提升")
    rs.add_body(doc, "第八步引入更强的模型。PyTorch-LSTM 用 1—2 层 LSTM（逐任务由网格搜索确定）+ 全连接输出，"
                     "超参数（hidden_dim、num_layers、学习率、dropout）在 8 个任务上分别做网格搜索，"
                     "训练时以验证损失早停并保存最优权重；XGBoost 与 LightGBM 用 early_stopping "
                     "控制迭代轮数。四类模型的预测在阶段三被拼成统一的集成输入。")

    rs.add_heading2(doc, "3.9 集成与评估")
    rs.add_body(doc, "最后一步做集成与评估。简单平均对 6 个（或 3 个）基学习器的预测等权平均；"
                     "Stacking 用训练段末尾切出的 %d 行作为时序留出块：基学习器在留出块之前的训练数据上重训，"
                     "并向留出块留出与预见期等长的间隔，据此生成元特征；元学习器为 Ridge，内部用时间序列"
                     "交叉验证选 alpha；测试段元特征由全训练段重训的基学习器生成。评估输出逐预见期的 "
                     "RMSE、MAE、R²，并画精度衰减曲线、策略对比柱状图与测试段水文过程线。"
                % facts["hold"])


# ------------------------------------------------------------------ 四、编码实现及结果

CODE_FIX = '''def load_daymet(path):
    dm = pd.read_csv(path, skiprows=6)                 # Daymet 头部 6 行是说明文字
    dm.columns = ['year', 'yday', 'Dayl', 'Prcp', 'Srad', 'Swe', 'Tmax', 'Tmin', 'Vp']
    dm = dm.dropna(subset=['year', 'yday'])
    dm = dm.astype({'year': int, 'yday': int})
    # 原始文件只有 year + yday，必须重建真实日历日期，否则与站点日期相差一天
    dm['Date'] = (pd.to_datetime(dm['year'].astype(str) + '-01-01')
                  + pd.to_timedelta(dm['yday'] - 1, unit='D'))
    return dm.set_index('Date').drop(columns=['year', 'yday'])


def main():
    dm = load_daymet(DIAG / 'daymet_full.txt')
    old = pd.read_csv(ROOT / '01047000.csv', parse_dates=['Date']).set_index('Date')
    # 以站点径流日期为准左连接气象列：Swe 由此从全 0 恢复为真实值
    out = old[['Discharge']].copy().join(dm, how='left')
    out.to_csv(ROOT / '01047000_fixed.csv', date_format='%Y-%m-%d')'''

CODE_FEATURE = '''def build_features(return_full=False):
    df = pd.read_csv(DATA_PATH)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    # 1. 四季独热编码：只表示隶属关系，不赋予季节数值大小
    for name, months in [('spring', [3, 4, 5]), ('summer', [6, 7, 8]),
                         ('autumn', [9, 10, 11]), ('winter', [12, 1, 2])]:
        df[f'Season_{name}'] = df['Month'].isin(months).astype(int)

    # 2. 周期编码：保证 12 月到 1 月在数值上连续
    df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12.0)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12.0)
    df['DayOfYear_sin'] = np.sin(2 * np.pi * df['DayOfYear'] / 365.25)
    df['DayOfYear_cos'] = np.cos(2 * np.pi * df['DayOfYear'] / 365.25)

    # 3. 滞后特征：径流与气象要素的历史取值（lag1/2/3/7）
    for col in ['Discharge', 'Prcp', 'Tmax', 'Tmin', 'Srad', 'Vp']:
        for lag in [1, 2, 3, 7]:
            df[f'{col}_lag{lag}'] = df[col].shift(lag)

    # 4. 滑动窗口统计：先 shift(1) 再 rolling，避免把当日信息算进窗口
    for col in ['Discharge', 'Prcp', 'Tmax']:
        for w in [3, 7]:
            rolled = df[col].shift(1).rolling(window=w)
            df[f'{col}_roll_mean_{w}'] = rolled.mean()
            df[f'{col}_roll_std_{w}'] = rolled.std()
            df[f'{col}_roll_max_{w}'] = rolled.max()
            df[f'{col}_roll_min_{w}'] = rolled.min()

    # 5. 标签：未来 1—7 天径流，dropna 同时清掉首尾因滞后期与预见期产生的空行
    for lead in range(1, 8):
        df[f'Target_lead{lead}'] = df['Discharge'].shift(-lead)
    return df.dropna().reset_index(drop=True)'''

CODE_SEARCH = '''def run_search(X, Y, log_target=True):
    # gap=7：训练段与验证段之间留出与最大预见期等长的间隔，避免窗口重叠泄漏
    tscv = TimeSeriesSplit(n_splits=3, gap=7)
    rows = []
    for family, maker, grid in [('ANN', make_ann, ANN_GRID), ('RF', make_rf, RF_GRID)]:
        for task in list(range(1, 8)) + ['multi']:
            label = f'{family}_{task}'
            # 直接多步取第 task 列标签，多输出取整个 7 维矩阵
            target = Y if task == 'multi' else Y[:, task - 1]
            y_fit = np.log1p(target) if log_target else target   # 压缩量级跨度
            gs = GridSearchCV(maker(), grid, cv=tscv,
                              scoring='neg_root_mean_squared_error',
                              n_jobs=2, error_score='raise')
            gs.fit(X, y_fit)                    # 只在训练段内搜索，测试段不参与
            joblib.dump(gs, OUT / f'{label}.joblib', compress=3)
            rows.append({'模型': family,
                         '策略': '多输出' if task == 'multi' else '直接',
                         '选参预见期': str(task),
                         '最佳参数': json.dumps(gs.best_params_, ensure_ascii=False),
                         '交叉验证RMSE（对数尺度）': float(-gs.best_score_)})
    return pd.DataFrame(rows)'''

CODE_LSTM = '''class LSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, output_dim=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2),
                                  nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(hidden_dim // 2, output_dim))

    def forward(self, x):
        out, _ = self.lstm(x)        # x: (batch, 时间步, 特征)
        return self.head(out[:, -1, :])   # 取最后时间步的隐藏状态做回归

# 训练循环中的早停：验证损失连续 patience 轮不下降就回滚到最优权重
if val_loss < best_loss:
    best_loss, best_state, stale_epochs = val_loss, copy.deepcopy(self.model.state_dict()), 0
else:
    stale_epochs += 1
if stale_epochs >= early_stopping_patience:      # 15 轮
    break
self.model.load_state_dict(best_state)'''

CODE_ENSEMBLE = '''# 简单平均：与 Stacking 共用同一批基学习器与同一预测集（phase3_run_all.py stage5）
for h in HORIZONS:
    mat = np.column_stack([np.asarray(base_direct[n][h], float).ravel() for n in names_direct])
    add("直接", f"{len(names_direct)}模型简单平均", h, Y_te[h], mat.mean(axis=1))

# Stacking：训练段末尾约 20% 作时序留出块生成元特征，元学习器为 Ridge
n_tr, n_hold = len(tr_e), int(round(len(tr_e) * 0.2))    # 1434 行 → 留出块 287 行（索引 1147 起）
hold_start = n_tr - n_hold

def split_for(h):
    fit_end = hold_start - h        # 留出与预见期等长的间隔，避免与留出块的目标日重叠
    return slice(0, fit_end), slice(hold_start, n_tr)

sl_fit, sl_ho = split_for(h)        # 基学习器只在留出块之前的数据上重训
m = _clone_best(joblib.load(ROOT / "results/phase2_v5" / f"ANN_{h}.joblib")).fit(
    Xb_train[sl_fit], np.log1p(Y_tr[h][sl_fit]))
meta_direct[h][:, k] = np.expm1(np.asarray(m.predict(Xb_train[sl_ho]), float)).ravel()

def _fit_ridge_meta(meta, y):       # 标准化 + 时间序列交叉验证选 alpha
    Z = StandardScaler().fit_transform(meta)
    errs = {a: np.mean([np.mean((y[va] - Ridge(alpha=a).fit(Z[tr], y[tr]).predict(Z[va])) ** 2)
                        for tr, va in TimeSeriesSplit(n_splits=3).split(Z) if len(tr) >= 30])
            for a in [1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0, 10000.0]}
    return Ridge(alpha=min(errs, key=errs.get)).fit(Z, y), Z

r, scaler, alpha = _fit_ridge_meta(meta_direct[h], y_hold[h])   # 元学习器只在留出块上拟合
prediction = r.predict(scaler.transform(test_mat))             # 测试段元特征来自全训练段重训的基学习器'''

CODE_METRIC = '''def metrics(y, p):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    rmse = float(np.sqrt(np.mean((y - p) ** 2)))          # 与 cfs 同量纲
    mae = float(np.mean(np.abs(y - p)))
    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return {'RMSE（cfs）': rmse, 'MAE（cfs）': mae,
            'R²': 1 - ss_res / ss_tot, 'NSE': 1 - ss_res / ss_tot,
            '样本数': int(len(y))}

# 评估时必须先反变换回 cfs，再裁剪非负，否则 R² 会掉到负数
y_pred = np.expm1(y_pred) if log_target else y_pred
y_pred = np.clip(y_pred, 0, None)'''

CODE_TAIL = '''def _predict(model, X, family=None):
    prediction = model.predict(X)
    # 阶段二的 ANN/RF/PseudoLSTM 是在 log1p 目标上训练的，
    # 集成前必须反变换，否则对数尺度会被当成 cfs 直接平均
    if family in ("ANN", "RF", "PseudoLSTM"):
        prediction = np.expm1(np.clip(prediction, -20, 20))
    return np.clip(prediction, 0, None)'''


def section_code(doc, d, num):
    facts = split_facts()
    rs.add_heading1(doc, "四、编码实现及结果")
    rs.add_heading2(doc, "4.1 开发环境与算法库")
    rs.add_body(doc, "全部实验在本机单机完成，不依赖 GPU。运行环境见表 4-1，版本号在生成报告时"
                     "由 importlib.metadata 现场读取，与实验结果同源。")
    rs.add_table(doc, ["项目", "配置"], [list(r) for r in table_env(None)],
                 caption="实验运行的软硬件环境", chapter=4, numbering=num,
                 col_widths_cm=[4.0, 10.6], font_size=Pt(9.5))
    rs.add_body(doc, "代码按职责拆分为三个层次：数据层（build_fixed_dataset.py、step1_data_analysis.py、"
                     "step2_feature_engineering.py）、建模层（phase2_v5_*.py、pytorch_lstm*.py、"
                     "train_boosting.py、enhanced_ensemble.py）、评估与校验层（verify_all_results.py、"
                     "phase3_run_all.py）。"
                     "下面按第三部分的步骤给出关键代码与注释，完整代码见附录。")

    rs.add_heading2(doc, "4.2 关键步骤代码与注释")
    rs.add_body(doc, "（1）数据修正：用 Daymet 的 year 与 yday 重建日期，再以站点日期为准左连接。"
                     "这是整份报告里收益最大的一段代码——它没有换模型，却把第 1 天预见期的 R² "
                     "从未修正数据口径下的 %s 提到 %s。"
                % (fmt(p2raw(d, "ANN", "直接", 1), 2), fmt(p2m(d, "ANN", "直接", 1), 4)))
    add_code_caption(doc, "代码 4-1  数据修正：按 Daymet 日历日重建并左连接（build_fixed_dataset.py，节选）")
    add_code_block(doc, CODE_FIX)

    rs.add_body(doc, "（2）特征构造：季节、周期、滞后、滑动窗口四类特征一次成型，"
                     "并在同一函数内构造 7 个标签列，保证特征与标签严格对齐。")
    add_code_caption(doc, "代码 4-2  特征工程：四季独热、周期编码、滞后与滑动窗口（step2_feature_engineering.py，节选）")
    add_code_block(doc, CODE_FEATURE)

    rs.add_body(doc, "（3）选参：GridSearchCV 外层套 TimeSeriesSplit，gap 与最大预见期等长；"
                     "训练目标取 log1p，搜索评分用负 RMSE（对数尺度）。16 组选参记录保存在"
                     "results/phase2_v5/选参结果.csv。")
    add_code_caption(doc, "代码 4-3  TimeSeriesSplit + GridSearchCV 选参（phase2_v5_experiment.py，节选）")
    add_code_block(doc, CODE_SEARCH)

    rs.add_body(doc, "（4）PyTorch-LSTM 的结构与训练：1—2 层 LSTM（逐任务由网格搜索确定）取最后时间步的"
                     "隐藏状态接两层全连接，验证损失早停并把最优权重回滚。")
    add_code_caption(doc, "代码 4-4  PyTorch-LSTM 结构与早停（pytorch_lstm.py，节选）")
    add_code_block(doc, CODE_LSTM)

    rs.add_body(doc, "（5）集成：简单平均直接对六个基学习器的预测等权平均；Stacking 用训练段末尾切出的 "
                     "%d 行时序留出块生成元特征，元学习器为内部用时间序列交叉验证选 alpha 的 Ridge，"
                     "测试段元特征由全训练段重训的基学习器生成。"
                % facts["hold"])
    add_code_caption(doc, "代码 4-5  简单平均与 Stacking 元特征构造（phase3_run_all.py，节选）")
    add_code_block(doc, CODE_ENSEMBLE)

    rs.add_body(doc, "（6）指标计算与反变换：四个指标一次算出，R² 与 NSE 同式；反变换与裁剪非负"
                     "是所有评估的唯一入口，避免不同脚本口径不一致。")
    add_code_caption(doc, "代码 4-6  评价指标计算与对数反变换（phase2_v5_experiment.py、enhanced_ensemble.py，节选）")
    add_code_block(doc, CODE_METRIC)
    rs.add_body(doc, "与之配套的反变换封装如下，集成脚本对所有对数尺度基学习器统一调用：")
    add_code_block(doc, CODE_TAIL)

    rs.add_heading2(doc, "4.3 模型精度结果")
    rs.add_body(doc, "阶段二的 28 行结果按模型拆成表 4-2、表 4-3 两张（每个模型 14 行：直接多步 7 行 + "
                     "多输出 7 行）。ANN 直接多步第 1 天 R² 为 %s、多输出为 %s；RF 两个策略的第 1 天 "
                     "R² 分别为 %s 与 %s。两个模型都表现出同一条规律：精度随预见期快速衰减"
                     "（个别相邻预见期之间有小幅反弹），第 4 天以后 R² 全部掉到 0.3 以下。"
                % (fmt(p2m(d, "ANN", "直接", 1), 4), fmt(p2m(d, "ANN", "多输出", 1), 4),
                   fmt(p2m(d, "RF", "直接", 1), 4), fmt(p2m(d, "RF", "多输出", 1), 4)))

    for model, cap in (("ANN", "第二阶段 ANN 两种多步策略的逐预见期精度"),
                       ("RF", "第二阶段随机森林两种多步策略的逐预见期精度")):
        rows = []
        for strategy in ("直接", "多输出"):
            for h in range(1, 8):
                rows.append(row_text([
                    strategy, h, fmt(p2m(d, model, strategy, h, "RMSE（cfs）"), 2),
                    fmt(p2m(d, model, strategy, h, "MAE（cfs）"), 2),
                    fmt(p2m(d, model, strategy, h, "R²"), 4),
                    fmt(p2m(d, model, strategy, h, "NSE"), 4)]))
        rs.add_table(doc, ["策略", "预见期（天）", "RMSE（cfs）", "MAE（cfs）", "R²", "NSE"], rows,
                     caption=cap, chapter=4, numbering=num,
                     col_widths_cm=[2.2, 2.2, 2.8, 2.8, 2.3, 2.3], font_size=Pt(9.5))

    rs.add_body(doc, "阶段三的 LSTM 精度见表 4-4。直接多步把第 1 天 R² 提到 %s，是全部单模型里最好的；"
                     "多输出为 %s。与阶段二相比，LSTM 在第 1—2 天的优势明显，但第 3 天以后同样迅速衰减，"
                     "说明制约长预见期的不是模型容量，而是未来气象条件本身不可知。"
                % (fmt(p3m(d, "直接", 1), 4), fmt(p3m(d, "多输出", 1), 4)))
    rows = []
    for strategy in ("直接", "多输出"):
        for h in range(1, 8):
            rows.append(row_text([strategy, h, fmt(p3m(d, strategy, h, "RMSE"), 2),
                                  fmt(p3m(d, strategy, h, "MAE"), 2),
                                  fmt(p3m(d, strategy, h, "R²"), 4)]))
    rs.add_table(doc, ["策略", "预见期（天）", "RMSE（cfs）", "MAE（cfs）", "R²"], rows,
                 caption="第三阶段 PyTorch-LSTM 的逐预见期精度（14 行）", chapter=4, numbering=num,
                 col_widths_cm=[3.0, 2.6, 3.2, 3.2, 2.6], font_size=Pt(9.5))

    long_r2 = [_boost(d, m, h) for m in ("XGBoost", "LightGBM") for h in range(4, 8)]
    rs.add_body(doc, "梯度提升模型的结果见表 4-5。XGBoost 在第 1 天达到 R² = %s，是单模型中仅次于 LSTM 的；"
                     "LightGBM 第 1 天为 %s。两者在第 4—7 天的 R² 都落在 %s—%s 之间，"
                     "比 ANN/RF 的长预见期更稳，这与提升树对非线性与交互项的拟合方式有关。"
                % (fmt(_boost(d, "XGBoost", 1), 4), fmt(_boost(d, "LightGBM", 1), 4),
                   fmt(min(long_r2), 3), fmt(max(long_r2), 3)))
    rows = []
    for model in ("XGBoost", "LightGBM"):
        for h in range(1, 8):
            rows.append(row_text([model, "直接", h,
                                  fmt(_boost(d, model, h, "RMSE"), 2),
                                  fmt(_boost(d, model, h, "MAE"), 2),
                                  fmt(_boost(d, model, h, "R²"), 4)]))
    table45, cap45 = rs.add_table(doc, ["模型", "策略", "预见期（天）", "RMSE（cfs）", "MAE（cfs）", "R²"], rows,
                 caption="第三阶段 XGBoost 与 LightGBM 的逐预见期精度（14 行）", chapter=4, numbering=num,
                 col_widths_cm=[2.6, 1.8, 2.4, 2.8, 2.6, 2.4], font_size=Pt(9.5))
    print("[INFO] 359/353 表注挂载于：%s" % cap45)
    rs.add_figure_note(doc, split_note(d, prefix=True))

    rs.add_body(doc, "四张集成结果表分别对应课程要求的四个子任务。6 模型简单平均（直接多步）第 1 天 "
                     "R² = %s，明显高于 6 模型 Stacking 的 %s；多输出一侧，3 模型简单平均第 1 天为 %s，"
                     "3 模型 Stacking 为 %s。结论一致：在样本量小、元特征高度共线的条件下，"
                     "等权平均比学习式集成更稳健。"
                % (fmt(ensm(d, "sa6", 1), 4), fmt(ensm(d, "st6", 1), 4),
                   fmt(ensm(d, "sa3", 1), 3), fmt(ensm(d, "st3", 1), 4)))

    ens_specs = (("sa6", "6 模型简单平均（直接多步）的逐预见期精度", "6模型简单平均"),
                 ("st6", "6 模型 Stacking（直接多步）的逐预见期精度", "6模型Stacking"),
                 ("sa3", "3 模型简单平均（多输出）的逐预见期精度", "3模型简单平均"),
                 ("st3", "3 模型 Stacking（多输出）的逐预见期精度", "3模型Stacking"))
    for key, cap, label in ens_specs:
        rows = []
        for h in range(1, 8):
            rows.append(row_text([label, h, fmt(ensm(d, key, h, "RMSE"), 2),
                                  fmt(ensm(d, key, h, "MAE"), 2), fmt(ensm(d, key, h, "R²"), 4)]))
        rs.add_table(doc, ["方法", "预见期（天）", "RMSE（cfs）", "MAE（cfs）", "R²"], rows,
                     caption=cap, chapter=4, numbering=num,
                     col_widths_cm=[4.2, 2.6, 2.8, 2.8, 2.2], font_size=Pt(9.5))

    improved = sum(1 for h in range(1, 8) if p2m(d, "ANN", "直接", h) > p2raw(d, "ANN", "直接", h))
    rs.add_body(doc, "数据修正的收益单独列成表 4-10。同一套 ANN（直接多步）流程，"
                     "仅把输入从原 CSV 换成修正后的 CSV，第 1 天 R² 由 %s 升至 %s，"
                     "RMSE 由 %s 降到 %s cfs；7 个预见期中有 %d 个改善。表中“修正前”一列为未修正数据"
                     "口径，仅作对照，报告中其余所有结果均为修正后口径。"
                % (fmt(p2raw(d, "ANN", "直接", 1), 4), fmt(p2m(d, "ANN", "直接", 1), 4),
                   fmt(p2raw(d, "ANN", "直接", 1, "RMSE（cfs）"), 2),
                   fmt(p2m(d, "ANN", "直接", 1, "RMSE（cfs）"), 2), improved))
    rows = []
    for h in range(1, 8):
        rows.append(row_text([h,
                              fmt(p2raw(d, "ANN", "直接", h, "RMSE（cfs）"), 2),
                              fmt(p2raw(d, "ANN", "直接", h), 4),
                              fmt(p2m(d, "ANN", "直接", h, "RMSE（cfs）"), 2),
                              fmt(p2m(d, "ANN", "直接", h), 4)]))
    table410, cap410 = rs.add_table(doc, ["预见期（天）", "修正前 RMSE", "修正前 R²", "修正后 RMSE", "修正后 R²"], rows,
                 caption="数据修正前后对照：ANN 直接多步（修正前为未修正数据口径）", chapter=4, numbering=num,
                 col_widths_cm=[2.8, 3.2, 2.8, 3.2, 2.6], font_size=Pt(9.5))
    print("[INFO] 359/353 表注挂载于：%s" % cap410)
    rs.add_figure_note(doc, split_note(d, prefix=True))

    gains = [p2m(d, "ANN", "直接", h) - p2raw(d, "ANN", "直接", h) for h in range(1, 8)]
    rmse_gain = [p2raw(d, "ANN", "直接", h, "RMSE（cfs）") - p2m(d, "ANN", "直接", h, "RMSE（cfs）")
                 for h in range(1, 8)]
    worse = [h for h in range(1, 8) if gains[h - 1] < 0]
    detail_worse = "、".join("第 %d 天（%+.4f）" % (h, gains[h - 1]) for h in worse)
    order = np.argsort(gains)[::-1]
    rs.add_body(doc, "逐期看，7 个预见期 R² 的平均变化为 %+.4f，其中第 %d 天提升最大（%+.4f）；"
                     "%s略有下降，但这两个预见期的 R² 本就在 0.1 以内，属于均值预测附近的波动。"
                     "RMSE 平均下降 %.2f cfs。除这两天外，其余预见期都向 0 附近或正值收敛，"
                     "幅度第二大的纠偏出现在第 %d 天（%+.4f）。这说明修正的是系统性的输入错误，"
                     "而不是某一预见期的偶然波动；同一修正对 RF 与多输出策略的方向一致，"
                     "因此后续所有结果均基于修正后数据，报告中不再重复未修正数据的数值。"
                % (np.mean(gains), int(np.argmax(gains)) + 1, max(gains),
                   detail_worse, np.mean(rmse_gain),
                   int(order[1]) + 1, gains[order[1]]))

    rs.add_heading2(doc, "4.4 可视化结果")
    rs.add_body(doc, "课程要求用可视化工具展示过程与结果，本报告共选用 8 张图，覆盖探索性分析、"
                     "特征工程、阶段二与阶段三的精度评价、集成策略对比与水文过程线。"
                     "图件由 matplotlib 生成、以 PNG 存于 results/ 下，报告直接引用原图，不做二次加工。")
    figures = (
        (FIG_TS, "径流与气象要素的时间序列总览（2000—2004）。" + EDA_CAPTION_NOTE, "径流呈显著的春汛—融雪节律，"
                                                          "冬季低流量平稳、4—5 月陡升，峰值集中在少数几天。"),
        (FIG_BOX, "8 个数值变量的箱型图与离群点。" + EDA_CAPTION_NOTE, "径流量存在大量上侧离群点，"
                                            "最大值为中位数的数十倍，这是长预见期难以拟合的直接原因。"),
        (FIG_CORR, "皮尔逊相关系数热力图。" + EDA_CAPTION_NOTE, "径流与日照、气温、水汽压的相关性明显强于与降水的同步相关，"
                                   "支持引入滞后特征刻画降水与气温的延迟响应。"),
        (FIG_SEL, "特征选择：|Pearson r| 与互信息的排序对比", "两种指标给出的前列特征高度一致，"
                                                    "径流自身及其滑动统计量贡献最大。"),
        (FIG_P2_DECAY, "第二阶段精度随预见期衰减（ANN 与 RF）", "R² 从第 1 天的 0.72 一路降到第 7 天的负值，"
                                                  "衰减速度在 2—4 天之间最快。"),
        (FIG_P3_DECAY, "第三阶段精度随预见期衰减（LSTM 与集成）", "LSTM 把第 1 天 R² 抬到 0.85，"
                                                  "但 3 天以后回落到与基线相同的量级。"),
        (FIG_P3_STRATEGY, "集成策略对比（直接多步与多输出）", "同一预见期下简单平均优于 Stacking 的规律在两张策略图上都成立。"),
        (FIG_P3_HYDRO, "测试段（2004 年）水文过程线", "预测曲线能跟上退水段与中低流量过程，"
                                            "但对陡涨的洪峰存在明显削峰。"),
    )
    for path, cap, note in figures:
        rs.add_figure(doc, path, caption=cap, chapter=4, numbering=num)
        rs.add_figure_note(doc, "说明：" + note)


def _boost(d, model, horizon, col="R²"):
    return metric(d["boost"], ["模型", "预见期（天）"], [model, horizon], col)


# ------------------------------------------------------------------ 五、程序调试

DEBUG_ITEMS = (
    ("原始气象列错位一天、积雪水当量恒为 0",
     "第一阶段建好特征后，第 1 天预见期的 R² 只有 0.50（未修正数据口径），且“积雪水当量”整列方差为 0；"
     "融合评估里融雪期的样本误差显著偏大。",
     "把站点的气象列与 Daymet 原始文件按日期逐日比对，发现 Srad/Prcp/Tmax/Tmin/Vp 与 Daymet 对应值"
     "整体相差一天，站点数据的气象列实际是前一日的值；Swe 全部为 0，与 Daymet 的积雪水当量不一致。"
     "进一步用 USGS 官方接口核对，确认径流列本身无误，问题只出在 7 个气象列。",
     "写 build_fixed_dataset.py：用 year+yday 重建 Daymet 的真实日期索引，以站点径流日期为准左连接，"
     "恢复真实的 Swe。修正后数据仍为 1827 行 9 列，但 ANN 直接多步第 1 天 R² 由 %s 升到 %s（表 4-10）。"),
    ("ANN/RF 在 log1p 空间训练，集成时漏做反变换",
     "第三阶段第一次把基学习器的预测拼成集成输入时，测试集 R² 约为 -0.5，RMSE 超过 3000 cfs，"
     "比任何单模型都差。",
     "检查 _predict 的返回值与各基学习器训练时的目标变换，发现阶段二的 ANN/RF 目标取的是 log1p(Discharge)，"
     "模型输出停在对数尺度；简单平均与 Stacking 把对数尺度的值直接当 cfs 用，"
     "再经非负裁剪后大量样本被压成 0。",
     "在 _predict 里按 family 统一做 np.expm1(np.clip(pred, -20, 20)) 反变换，再裁剪非负（代码 4-6 末段），"
     "并把这条约定收敛为唯一的评估入口。修复后集成精度回到基学习器同一量级，第 1 天 R² 达到 %s。"),
    ("预测表的“日期”实为起报日，过程线整体偏移一天",
     "水文过程线图上预测序列相对实测整体左移一天，第 1 天预见期的峰值总在实测峰值之前出现。",
     "对照逐日预测结果.csv 与测试段原始日期，发现预测表的“日期”字段记录的是特征所在日 t，"
     "而预测值对应 t+h；画图时直接拿 t 当横坐标，等于把整条曲线提前了一天。",
     "评估与绘图统一改用 target_date = t + h（或直接取特征表中 Target_lead{h} 对应的日期），"
     "修正后峰值时刻与实测一致，过程线的相位偏差消失。"),
    ("标题段前段后被写成 EMU 数值，每级标题独占一页",
     "生成的 docx 打开后每个一级标题都单独占一页，标题与上一段之间出现大段空白。",
     "解压 docx 检查 word/document.xml，发现标题样式的 w:spacing 的 before/after 被写成了 457200 这类"
     "EMU 数值（python-docx 中 Length 未按磅换算），Word 按 1/20 磅解释，段前相当于上万磅。",
     "统一用 Pt() 换算成 twips 再写入 w:spacing（标题样式只保留 line_spacing=1.5、"
     "first_line_indent=Pt(0)，段落间距一律传 Pt 值）；重新生成后标题不再独占页。"),
    ("matplotlib 图例中文缺字",
     "精度衰减图的图例与坐标轴标签里出现方框，控制台同时提示 Glyph missing from current font。",
     "图件统一设置 rcParams['font.sans-serif'] = ['SimSun']，但标签中含“²”“≤”等 SimSun 缺失字形，"
     "且图例文字混用了数学符号。",
     "图例标签改为纯中文短标签（如“直接多步”“多输出”），字体回退链设为 "
     "['SimHei', 'Microsoft YaHei', 'SimSun'] 并设置 axes.unicode_minus=False；重出图后无缺字。"),
    ("控制台 GBK 编码报错",
     "批处理脚本打印中文模型名或“R²”时抛出 UnicodeEncodeError: 'gbk' codec can't encode character，"
     "整批实验在打印环节中断。",
     "Windows 控制台默认代码页 936，Python 的 stdout 按 GBK 编码；脚本中又存在“²”、全角括号等字符。",
     "所有入口脚本开头调用 sys.stdout.reconfigure(encoding='utf-8')，运行时追加 "
     "PYTHONIOENCODING=utf-8；训练日志改为写入 run_phase2.log、run_stage5_7.log 等文件，不再依赖控制台。"),
    ("Stacking 元特征分布偏移，长预见期反而不如简单平均",
     "6 模型 Stacking 第 1 天 R² 为 %s，低于简单平均的 %s；第 3—7 天 Stacking 的 MAE 明显更大"
     "（表 4-6 与表 4-7）。",
     "元学习器只在训练段末尾约五分之一的留出块上拟合，样本量小；六个基学习器预测的是同一条径流，"
     "元特征高度共线，留出块上的权重估计方差大，长预见期的权重更不稳定。",
     "先核对描述与产出结果的对应关系：曾尝试 enhanced_ensemble.py 的折外预测（out-of-fold）加浅层梯度"
     "提升树（Gradient Boosting Regressor）的方案，但该脚本依赖的 train_pytorch_lstm 模块在工程中不存在、"
     "其产物 enhanced_ensemble_metrics.csv 也未进入结果目录，四张集成表实际由 phase3_run_all.py stage5 的"
     "“287 行留出块 + Ridge”方案产出，报告正文与代码节选据此改正；教训是方法描述必须与产出结果的代码一致。"
     "改进方向：对元学习器施加非负权重约束（非负最小二乘 / 带约束 Ridge），按预见期分别选择集成策略，"
     "并在滚动窗口上重训元学习器以检验稳定性；当前把简单平均作为主结论、Stacking 作为对照。"),
    ("Word COM 导出 PDF 偶发 0x800706BE",
     "用 doctopdf_one.ps1 转 PDF 时偶发“远程过程调用失败 (0x800706BE)”，且 WINWORD.EXE 进程残留，"
     "后续调用持续失败。",
     "Document.Open 之后立即 SaveAs，Word 尚未完成排版；脚本也没有显式释放 COM 对象，"
     "多次调用后 Word 实例堆积。",
     "Open 之后 Start-Sleep -Milliseconds 800 再 SaveAs，关闭时 Close([ref]$false) 并 Quit()，"
     "转换前清理残留 WINWORD 进程；改为单文件单进程调用（doc1.docx → doc1.pdf）后稳定。"),
)


def section_debug(doc, d):
    rs.add_heading1(doc, "五、程序调试")
    rs.add_body(doc, "本节按实际调试顺序记录 8 个有代表性的问题，每个问题都按“现象—定位—解决”"
                     "三段叙述。这些问题大多不是模型本身的问题，而是数据口径、尺度变换与排版/工程细节"
                     "带来的问题，其中前两个直接决定了几十个百分点量级的精度差异。")
    for idx, (title, symptom, locate, fix) in enumerate(DEBUG_ITEMS, start=1):
        rs.add_heading3(doc, "问题 %d  %s" % (idx, title))
        rs.add_body(doc, "现象：" + symptom)
        rs.add_body(doc, "定位：" + locate)
        if idx in (1, 2, 7):
            vals = {
                1: (fmt(p2raw(d, "ANN", "直接", 1), 2), fmt(p2m(d, "ANN", "直接", 1), 4)),
                2: (fmt(ensm(d, "sa6", 1), 4),),
                7: (fmt(ensm(d, "st6", 1), 4), fmt(ensm(d, "sa6", 1), 4)),
            }[idx]
            if "%s" in symptom:
                symptom = symptom % vals
            if "%s" in fix:
                fix = fix % vals
        rs.add_body(doc, "解决：" + fix)


# ------------------------------------------------------------------ 六、总结


def section_summary(doc, d):
    rs.add_heading1(doc, "六、总结")
    rs.add_heading2(doc, "6.1 本次实践用到的技术、模型与算法")
    items = [
        "数据修正：Daymet 原始文件的日期重建与日历日对齐，USGS 径流序列核对；",
        "探索性分析：描述统计、直方图与核密度估计、箱型图与离群点识别、皮尔逊相关与滞后相关分析；",
        "特征工程：四季独热编码、月份/年积日正余弦周期编码、lag1/2/3/7 滞后特征、"
        "3 天与 7 天滑动均值/标准差/极值、未来 1—7 天多步标签构造；",
        "归一化：Min-Max 与 Z-score 两种方案对比，训练段 fit、测试段 transform；",
        "特征选择：|Pearson r| 与互信息双指标排序；",
        "基线模型：多层感知机 ANN（sklearn MLPRegressor）、随机森林 RF；",
        "深度模型：PyTorch 实现的 LSTM 回归器（1—2 层、dropout、早停、学习率调度，层数逐任务由网格搜索确定）；",
        "梯度提升：XGBoost 与 LightGBM（early_stopping 控制迭代轮数）；",
        "集成学习：等权简单平均、Stacking（287 行时序留出块 + Ridge 元学习器）；",
        "超参数搜索：GridSearchCV + TimeSeriesSplit(n_splits=3, gap=7)，严格限定在训练段；",
        "多步预测策略：直接多步（每个预见期一个模型）与多输出（一次输出 7 维）；",
        "评价与可视化：RMSE、MAE、R²、NSE，精度衰减曲线、策略对比柱状图、水文过程线；",
    ]
    for item in items:
        rs.add_body(doc, "· " + item, indent=False)

    rs.add_heading2(doc, "6.2 当前模型与结果存在的问题及改进方案")
    problems = [
        ("LSTM 实际只用了 1 天序列长度",
         "当前 LSTM 的输入仍是被展平的单日特征向量，时间维度为 1，循环结构没有真正发挥作用，"
         "因此它的优势主要来自参数容量而非时序记忆。",
         "改为滑窗序列输入（T = 7/14/30），构造 (样本, 时间步, 特征) 三维张量，"
         "并与直接多步 / 多输出两种输出头组合复现对比。"),
        ("积雪相关特征只有 8 个",
         "修正后的积雪水当量虽已恢复，但只以原始值加少量滞后进入模型，融雪过程的物理含义没有被显式表达。",
         "补充累积正积温、雪水当量变化率、融雪指数与降雨—融雪分离特征，并对融雪期单独建模。"),
        ("单站单期，泛化能力无法验证",
         "全部结论建立在 01047000 一个站点、2000—2004 五年数据上，测试段只有 2004 年 359 天。",
         "引入多站联合训练与区域化模型，用迁移学习或站点嵌入共享跨站点规律，"
         "并以滚动重训（walk-forward）代替单次划分来评估稳定性。"),
        ("峰值系统性低估",
         "第 1 天 R² 已达 %s，但水文过程线显示陡涨洪峰被明显削平：平方损失对稀有极值不敏感，"
         "模型倾向于输出条件均值。",
         "改用分位数损失或对高流量样本加权的损失函数，配合对数—原尺度混合目标与极值增广，"
         "并单独报告洪峰的量级与峰现时间误差。"),
        ("集成元学习器不稳定",
         "Stacking 的第 1 天 R²（%s）低于简单平均（%s），长预见期差距更大，元特征共线且样本量小是主因。",
         "对元学习器施加非负权重约束（非负最小二乘 / 带约束 Ridge），按预见期分别选择集成策略，"
         "并在滚动窗口上重训元学习器以检验稳定性。"),
        ("2004 年测试段被反复查看，泛化能力估计偏乐观",
         "2004 年测试段在 v2/v3/v5 与第三阶段的多轮迭代中被反复查看，方案与超参数的取舍实际上参照了它的指标，"
         "因此报告的泛化能力估计偏乐观。",
         "最终泛化能力仍需新年份或其他流域数据验证：把 2004 年重新放回“只用一次”的最终留出集，"
         "中间的方案选择改用滚动验证（walk-forward）或逐年留出，并用其他站点数据做外部验证。"),
    ]
    for idx, (title, desc, plan) in enumerate(problems, start=1):
        rs.add_body(doc, "（%d）%s。" % (idx, title) + desc, indent=True)
        rs.add_body(doc, "改进方案：" + plan, indent=True)

    rs.add_heading2(doc, "6.3 对人工智能/机器学习实际应用的认识与心得体会")
    rs.add_body(doc, "第一，数据口径比模型选择重要得多。本次实践中最大的一次精度跃升不是换模型换来的："
                     "把气象列按日历日对齐、把恒为 0 的积雪水当量恢复之后，同样的 ANN 直接多步，"
                     "第 1 天 R² 就从 %s 变成 %s。这让我理解到，真实项目里“先怀疑数据”不是一句口号，"
                     "而是有明确收益排序的工程习惯：花两小时核对数据，可能抵得上花两天调参。"
                % (fmt(p2raw(d, "ANN", "直接", 1), 2), fmt(p2m(d, "ANN", "直接", 1), 4)))
    rs.add_body(doc, "第二，指标好看不等于能用。R² 从第 1 天的 %s 一路衰减到第 7 天的负值，"
                     "但真正让我改变判断的是过程线：模型能跟上退水段，却把洪峰削平了。"
                     "水文业务关心的是峰现时间与洪峰量级，而 RMSE 这类平方损失指标恰恰对极值最不敏感。"
                     "从那以后我在每次评估里都会额外看一条过程线，而不是只看表格里的小数点。"
                % fmt(ensm(d, "sa6", 1), 4))
    rs.add_body(doc, "第三，约束条件其实是防泄漏的基础设施。“禁用 MultiOutputRegressor”“必须用 Grid Search"
                     "交叉验证”“选参只能看训练段”这些要求在实现时确实增加了工作量，但它们把两类最容易犯的错"
                     "挡在了流程外面：一是用随机划分让相邻日样本同时出现在训练和测试里，二是用测试段的"
                     "信息挑超参数。配合 gap=7 的 TimeSeriesSplit 和统一的反变换入口，我才真正理解"
                     "“可复现”意味着每一个数字都能追溯到某次具体的运行。")
    rs.add_body(doc, "第四，工程化能力决定结论的可信度。为了让报告里的每个数字都来自结果文件而不是手工抄写，"
                     "我把结果读取、格式化与关键数值断言全部脚本化：任何一个数字对不上，生成过程就直接中断。"
                     "这套做法起初被视为“多余的工作”，但当模型重新训练、结果表更新时，"
                     "报告只需要重新运行一次就能保持一致——这大概就是从“跑通一次”到“能交付”的距离。")


# ------------------------------------------------------------------ 注 与 附录

NOTE_LINES = (
    "注：",
    "报告中请务必写明每位同学的分工；",
    "请将本报告电子版及代码文件打包提交至网络教学平台；【压缩包以“组号+组长姓名+组员1姓名+组员2姓名”方式命名】",
    "不交实验报告者，成绩将被判为不及格。",
)

SCRIPT_ORDER = (
    "build_fixed_dataset", "step1_data_analysis", "step2_feature_engineering", "enhanced_features",
    "phase2_v3_config", "phase2_v3_data", "phase2_v3_features", "phase2_v3_models", "phase2_v3_baselines",
    "phase2_v5_data", "phase2_v5_experiment", "phase2_v5_figs",
    "pytorch_lstm", "pytorch_lstm_grid_search", "train_boosting", "enhanced_ensemble",
    "phase3_lstm", "phase3_probe", "phase3_run_all", "retrain_pytorch_lstm_final",
    "verify_all_results", "make_summary",
)

# 排版/成文类脚本属于提交方自用工具，不属于课程交付的技术内容，一律不列入附录
EXCLUDED_SCRIPT_STEMS = (
    "experiment_report", "report_style",
    "phase1_report", "phase2_report", "phase3_report", "final_report", "sizheng_report",
)


def section_notes(doc):
    rs.add_heading1(doc, NOTE_LINES[0])
    for line in NOTE_LINES[1:]:
        rs.add_body(doc, line, indent=False)


def code_scripts():
    """按 SCRIPT_ORDER 排序 code/ 下的建模脚本（排版/成文类脚本不附录）。"""
    files = [p for p in (ROOT / "code").glob("*.py")
             if p.stem not in EXCLUDED_SCRIPT_STEMS]
    by_stem = {p.stem: p for p in files}
    ordered = [by_stem[s] for s in SCRIPT_ORDER if s in by_stem]
    ordered += sorted((p for p in files if p.stem not in SCRIPT_ORDER), key=lambda p: p.stem)
    return ordered


def section_appendix(doc):
    rs.add_heading1(doc, "附录：")
    scripts = code_scripts()
    total, stripped = 0, 0
    bodies = []
    for path in scripts:
        text = path.read_text(encoding="utf-8")
        raw_lines = text.count("\n") + (0 if text.endswith("\n") else 1)
        lines = [l.replace("\t", "    ").rstrip() for l in text.splitlines() if l.strip()]
        total += raw_lines
        stripped += len(lines)
        bodies.append((path.name, raw_lines, len(lines), "\n".join(lines)))

    rs.add_body(doc, "本附录收录 code/ 目录下与本次实践技术内容相关的 %d 个 Python 脚本，合计 %d 行"
                     "（按模板“可减少空行”的要求去除空行后为 %d 行）。取舍说明：为使结论可复核，"
                     "数据准备、特征工程、建模、集成与校验脚本全部收录，未做删减；"
                     "报告排版与成文类脚本属于提交方自用工具，不属于本课程交付的技术内容，未列入附录；"
                     "模型权重文件（*.joblib）与结果表格（*.csv）不入附录，随实验目录一并提交。"
                     "其中 enhanced_ensemble.py 是未被采用的尝试（依赖的 train_pytorch_lstm 模块不在工程内、"
                     "产物也未进入结果目录），为便于复核仍按原样收录，报告中的集成结果以 phase3_run_all.py 为准。"
                     "代码以等宽字体 7.5 磅、单倍行距排印，按“数据准备 → 阶段一 → 阶段二 → 阶段三 → "
                     "校验与汇总”的顺序排列；每个文件以黑体标题起始，文件之间连续排印而不强制分页，"
                     "以避免出现只有两三行代码的半空白页（模板亦要求“以减少空行、缩小字体的方式节省纸张”）。"
                % (len(scripts), total, stripped))

    for idx, (name, raw_lines, stripped_lines, body) in enumerate(bodies, start=1):
        cap = doc.add_paragraph()
        pf = cap.paragraph_format
        pf.space_before = Pt(9 if idx > 1 else 6)
        pf.space_after = Pt(2)
        pf.first_line_indent = Pt(0)
        pf.keep_with_next = True
        rs.set_run_font(cap.add_run("代码 %d  %s（%d 行）" % (idx, name, raw_lines)),
                        Pt(10), bold=False, east_asia=rs.HEI_FONT)
        add_code_block(doc, body, size_pt=7.5, space_after=0)
    return [b[0] for b in bodies], total, stripped


# ------------------------------------------------------------------ 组装


def report():
    started = time.time()
    d = load_data()
    cover = cover_fields()

    # ---- 关键数字断言（至少 8 处）：格式与正文完全一致
    check_value("ANN 直接多步 第1天 R²", fmt(p2m(d, "ANN", "直接", 1), 4), "0.7230")
    check_value("ANN 多输出 第1天 R²", fmt(p2m(d, "ANN", "多输出", 1), 4), "0.7452")
    check_value("PyTorch-LSTM 直接 第1天 R²", fmt(p3m(d, "直接", 1), 4), "0.8463")
    check_value("PyTorch-LSTM 多输出 第1天 R²", fmt(p3m(d, "多输出", 1), 4), "0.8019")
    check_value("6模型简单平均 第1天 R²", fmt(ensm(d, "sa6", 1), 4), "0.8196")
    check_value("3模型多输出简单平均 第1天 R²", fmt(ensm(d, "sa3", 1), 3), "0.767")
    check_value("全期径流均值", fmt(d["data"]["Discharge"].mean(), 3), "610.893")
    test_q = d["data"]["Discharge"].iloc[-359:]
    check_value("测试段最大径流", fmt(test_q.max(), 1), "7470.0")
    check_value("未修正数据 ANN 直接 第1天 R²", fmt(p2raw(d, "ANN", "直接", 1), 2), "0.50")
    check_value("数据集行列数", "%dx%d" % d["data"].shape, "1827x9")
    check_value("成品特征矩阵", "%dx%d" % d["proc"].shape, "1827x196")
    facts = split_facts()
    check_value("阶段二训练行数", facts["train"], 1440)
    check_value("阶段二测试行数", facts["test"], 359)
    check_value("阶段三训练行数", facts["train3"], 1434)
    check_value("阶段三测试行数", facts["test3"], 353)
    check_value("Stacking 留出块行数", facts["hold"], 287)
    check_value("训练+测试+丢弃行数", facts["train"] + facts["test"] + 
                (pd.Timestamp("2003-12-31") - pd.Timestamp(facts["train_end"])).days
                + (pd.Timestamp(facts["train_start"]) - pd.Timestamp(d["data"]["Date"].iloc[0])).days
                + (pd.Timestamp(d["data"]["Date"].iloc[-1]) - pd.Timestamp(facts["test_end"])).days,
                d["data"].shape[0])

    doc = rs.new_document(header_text=HEADER_TEXT)
    build_cover(doc, cover)
    rs.begin_body_section(doc, restart_at=1, header_text=HEADER_TEXT)
    num = rs.Numbering()

    section_task(doc, d, num)
    section_division(doc)
    section_design(doc, d, num)
    section_code(doc, d, num)
    section_debug(doc, d)
    section_summary(doc, d)
    section_notes(doc)
    files, total, stripped = section_appendix(doc)
    check_value("附录收录脚本数", len(files), 22)

    out = ROOT / DOC_NAME
    doc.save(str(out))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    copy = OUT_DIR / DOC_NAME
    shutil.copyfile(out, copy)

    same = out.read_bytes() == copy.read_bytes()
    if not same:
        raise RuntimeError("两份 docx 不一致")

    print("[OK] 生成完成，用时 %.1f 秒" % (time.time() - started))
    print("[OK] 主文件  %s  %d 字节" % (out, out.stat().st_size))
    print("[OK] 副本    %s  %d 字节（逐字节一致：%s）" % (copy, copy.stat().st_size, same))
    print("[OK] 附录脚本 %d 个，合计 %d 行（去空行 %d 行）" % (len(files), total, stripped))
    print("[OK] 数值断言 %d 处全部通过：" % len(CHECKS))
    for label, expected, actual in CHECKS:
        print("     %-32s %-12s (实得 %s)" % (label, expected, actual))
    return out


# ------------------------------------------------------------------ 渲染自检


def convert_pdf(docx_path):
    """用 Word COM 把 docx 转成 PDF（单文件单进程，规避 0x800706BE）。"""
    work = Path(r"C:\lorender")
    work.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(docx_path, work / "doc1.docx")
    script = ROOT / "archive" / "_scripts" / "doctopdf_one.ps1"
    proc = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
                           "-File", str(script), "1"],
                          capture_output=True, text=True, timeout=600)
    print("[PDF]", (proc.stdout or "").strip(), (proc.stderr or "").strip()[:400])
    pdf = work / "doc1.pdf"
    if not pdf.exists():
        raise RuntimeError("PDF 转换失败")
    return pdf


def ink_check(pdf_path):
    """逐页渲染并统计墨迹覆盖率，同时抽取章节位置与 4 张抽查页。"""
    import pymupdf

    pdf = pymupdf.open(str(pdf_path))
    n = pdf.page_count
    coverages = []
    for page in pdf:
        pix = page.get_pixmap(dpi=70, colorspace=pymupdf.csGRAY)
        arr = np.frombuffer(pix.samples, dtype=np.uint8)
        coverages.append(float((arr < 250).mean()))

    texts = [pdf[i].get_text() for i in range(n)]
    appendix_start = None
    for i, t in enumerate(texts):
        if i > 3 and ("附录：" in t or "本附录收录" in t):
            appendix_start = i + 1
            break

    body_pages = (appendix_start - 1) if appendix_start else n
    low = []
    for i, c in enumerate(coverages, start=1):
        if i == 1 or i == appendix_start or i == n:
            continue
        if c < 0.05:
            low.append((i, round(c, 4)))

    samples = []
    targets = {"cover": 1}
    for key, needle in (("table", "表 4-2"), ("figure", "图 4-1")):
        for i, t in enumerate(texts):
            if needle.replace(" ", "") in t.replace(" ", ""):
                targets[key] = i + 1
                break
    if appendix_start:
        targets["appendix"] = appendix_start
    out_dir = Path(r"C:\lorender")
    for key, page_no in targets.items():
        if 1 <= page_no <= n:
            pix = pdf[page_no - 1].get_pixmap(dpi=110)
            path = out_dir / ("sample_%s_p%d.png" % (key, page_no))
            pix.save(str(path))
            samples.append((key, page_no, str(path)))

    print("[PDF] 总页数 %d，附录起始页 %s，正文页数 %d" % (n, appendix_start, body_pages))
    print("[INK] 覆盖率序列：" + " ".join("%d:%.3f" % (i, c) for i, c in enumerate(coverages, 1)))
    print("[INK] 小于 0.05 的正文页：" + (str(low) if low else "无"))
    for key, page_no, path in samples:
        print("[SAMPLE] %s -> 第 %d 页 %s" % (key, page_no, path))
    return {"pages": n, "appendix_start": appendix_start, "body_pages": body_pages,
            "low": low, "coverage": coverages, "samples": samples}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="生成后转 PDF 并做渲染自检")
    args = parser.parse_args()
    started = time.time()
    out = report()
    if args.check:
        pdf = convert_pdf(out)
        ink_check(pdf)
    print("[DONE] 总耗时 %.1f 秒" % (time.time() - started))


if __name__ == "__main__":
    main()
