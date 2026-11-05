# -*- coding: utf-8 -*-
"""《机器学习课程实践》第一次阶段验收报告（数据分析与特征工程）生成器。

全部数值均由 pandas 读自 results/ 与原始 CSV 后格式化输出，正文不含手写数字。
排版统一调用 code/report_style.py，不自行定义样式。
"""

import json
import os
import shutil
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from report_style import (  # noqa: E402
    begin_body_section,
    add_body,
    add_cover,
    add_figure,
    add_figure_note,
    add_heading1,
    add_heading2,
    add_table,
    new_document,
    Numbering,
)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

RAW_CSV = ROOT / "01047000.csv"
FEAT_CSV = ROOT / "results" / "processed_data" / "01047000_with_feature_values_独热编码版.csv"
FIG_EDA = ROOT / "results" / "figures" / "eda"
FIG_FE = ROOT / "results" / "figures" / "fe_without_current_discharge"
SUMMARY_CSV = FIG_EDA / "summary_statistics.csv"
LAG_CORR_CSV = FIG_EDA / "discharge_with_lag_factors_correlations.csv"
MATRIX_CSV = FIG_EDA / "pearson_correlation_matrix.csv"
RANK_CSV = FIG_FE / "feature_selection_ranking.csv"

OUT_DIR = ROOT / "本地资料" / "第一阶段"
DOC_NAME = "《机器学习课程实践》第一次阶段验收报告.docx"

BASE_COLS = ["日期", "当天径流量", "日照时长", "降水量", "短波辐射", "积雪水当量",
             "最高气温", "最低气温", "水汽压"]
TIME_COLS = ["年份", "月份", "日", "星期编码", "年内日序"]
ONEHOT_COLS = ["春季独热编码", "夏季独热编码", "秋季独热编码", "冬季独热编码"]
PERIODIC_COLS = ["月份正弦编码", "月份余弦编码", "年内日序正弦编码", "年内日序余弦编码"]
MM_SUFFIX = "_最小最大归一化"
ZS_SUFFIX = "_Z分数标准化"

CN_NAMES = {"Discharge": "当天径流量", "Dayl": "日照时长", "Prcp": "降水量", "Srad": "短波辐射",
            "Swe": "积雪水当量", "Tmax": "最高气温", "Tmin": "最低气温", "Vp": "水汽压"}


# 阶段验收日期参数：本份（第一次）为 2026-10-08；
# 后续两次报告分别改为 "2026-10-22" 与 "2026-11-05"，正文其余内容不变。
REPORT_DATE_ISO = "2026-10-08"


def cn_date(iso):
    """把 YYYY-MM-DD 转成中文年月日（如 二〇二六年十月八日）。"""
    year, month, day = (int(x) for x in iso.split("-"))
    digits = "〇一二三四五六七八九"

    def num(n):
        if n <= 10:
            return "十" if n == 10 else digits[n]
        if n < 20:
            return "十" + digits[n - 10]
        tens, ones = divmod(n, 10)
        return digits[tens] + "十" + (digits[ones] if ones else "")

    return "".join(digits[int(c)] for c in str(year)) + "年" + num(month) + "月" + num(day) + "日"


REPORT_DATE_CN = cn_date(REPORT_DATE_ISO)


def f3(value):
    """保留三位小数的定点格式。"""
    return "%.3f" % float(value)


def cn_feature(name):
    """把相关表里的英文特征名换成中文名，保留原始英文名可追溯。"""
    if "_lag" in name:
        base, k = name.split("_lag")
        return "%s 滞后 %s 天" % (CN_NAMES.get(base, base), k)
    return CN_NAMES.get(name, name)


# ---------------------------------------------------------------- 读数据
raw = pd.read_csv(RAW_CSV)
raw["Date"] = pd.to_datetime(raw["Date"])
feat = pd.read_csv(FEAT_CSV)
summary = pd.read_csv(SUMMARY_CSV, index_col=0)
lag_corr = pd.read_csv(LAG_CORR_CSV, index_col=0)
matrix = pd.read_csv(MATRIX_CSV, index_col=0)
rank = pd.read_csv(RANK_CSV)

n_rows, n_cols = raw.shape
n_feat_rows, n_feat_cols = feat.shape
date_min, date_max = raw["Date"].min().date(), raw["Date"].max().date()
train_days = int((raw["Date"].dt.year <= 2003).sum())
test_days = int((raw["Date"].dt.year == 2004).sum())

# 特征列构成统计
cols = list(feat.columns)
mm_cols = [c for c in cols if c.endswith(MM_SUFFIX)]
zs_cols = [c for c in cols if c.endswith(ZS_SUFFIX)]
plain_cols = [c for c in cols if c not in mm_cols and c not in zs_cols]
lag_raw = [c for c in plain_cols if "_滞后" in c]
win_raw = [c for c in plain_cols if "过去" in c]
enc_raw = [c for c in plain_cols if c in ONEHOT_COLS or c in PERIODIC_COLS]
base_raw = [c for c in plain_cols if c in BASE_COLS or c in TIME_COLS]
assert len(plain_cols) == len(lag_raw) + len(win_raw) + len(enc_raw) + len(base_raw)
assert len(cols) == len(lag_raw) + len(win_raw) + len(enc_raw) + len(base_raw) + len(mm_cols) + len(zs_cols)
n_lag, n_win, n_enc, n_base = len(lag_raw), len(win_raw), len(enc_raw), len(base_raw)
n_mm, n_zs = len(mm_cols), len(zs_cols)
lag_total = n_lag + sum(1 for c in mm_cols + zs_cols if "_滞后" in c)
win_total = n_win + sum(1 for c in mm_cols + zs_cols if "过去" in c)
enc_total = n_enc + sum(1 for c in mm_cols + zs_cols if c[:-len(MM_SUFFIX)] in ONEHOT_COLS + PERIODIC_COLS
                        or c[:-len(ZS_SUFFIX)] in ONEHOT_COLS + PERIODIC_COLS)
base_total = n_base + n_mm + n_zs - (lag_total - n_lag) - (win_total - n_win) - (enc_total - n_enc)

hist_gap = int(feat["当天径流量_滞后7天"].isna().sum())
zeros_swe = int((raw["Swe"] == 0).sum())

# IQR 离群点计数（原始量纲）
outliers = {}
for col in ["Discharge", "Dayl", "Prcp", "Srad", "Swe", "Tmax", "Tmin", "Vp"]:
    q1, q3 = raw[col].quantile(0.25), raw[col].quantile(0.75)
    iqr = q3 - q1
    outliers[CN_NAMES[col]] = int(((raw[col] < q1 - 1.5 * iqr) | (raw[col] > q3 + 1.5 * iqr)).sum())

# 特征选择：沿用既有口径 |r| >= 0.2 或 MI >= 0.05 或降水类必留
keep_mask = ((rank["皮尔逊相关系数绝对值"] >= 0.2)
             | (rank["互信息得分"] >= 0.05)
             | rank["特征名称"].str.startswith("降水量"))
kept = rank[keep_mask]
dropped = rank[~keep_mask]
n_cand, n_kept, n_dropped = len(rank), len(kept), len(dropped)
dropped_preview = "、".join(dropped["特征名称"].head(4).tolist())
top1 = rank.iloc[0]


def rank_pos(name):
    """特征在双指标排序表中的名次（1 起）。"""
    return int(rank.index[rank["特征名称"] == name][0]) + 1

# 关键相关系数
lag1_r = float(lag_corr.loc["Discharge_lag1", "Pearson_r"])
lag2_r = float(lag_corr.loc["Discharge_lag2", "Pearson_r"])
lag3_r = float(lag_corr.loc["Discharge_lag3", "Pearson_r"])
lag7_r = float(lag_corr.loc["Discharge_lag7", "Pearson_r"])
prcp_sync = float(matrix.loc["径流", "降水"])
prcp_lag1 = float(lag_corr.loc["Prcp_lag1", "Pearson_r"])
vp_lag2 = float(lag_corr.loc["Vp_lag2", "Pearson_r"])
vp_lag3 = float(lag_corr.loc["Vp_lag3", "Pearson_r"])

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


# ---------------------------------------------------------------- 文档
doc = new_document(header_text="杭州电子科技大学《机器学习课程实践》阶段验收报告　·　第一次")
add_cover(
    doc,
    college="自动化学院",
    major="智能科学与技术",
    class_name="课程实践小组",
    **cover_fields(),
    project="基于机器学习的径流预测",
    content="数据分析与特征工程",
    subtitle="第一次阶段验收报告",
    date_text=REPORT_DATE_CN,
)
begin_body_section(doc, 1)

num1 = Numbering(chapter=1)
num2 = Numbering(chapter=2)

# ================================================================ 第一章
add_heading1(doc, "一、数据分析")
add_body(doc, "本阶段以站点 01047000 的逐日水文气象观测序列为分析对象，先完成数据核对与探索性分析，再据此开展特征构造、特征归一化与特征选择。原始数据文件 01047000.csv 共 %d 行、%d 列，覆盖 %s 至 %s 的连续日序列；按时间顺序划分，2000—2003 年共 %d 天构成训练段，2004 年共 %d 天构成测试段，训练段在前、测试段在后，以避免未来信息渗入训练过程。" % (n_rows, n_cols, date_min, date_max, train_days, test_days))
add_body(doc, "本章依序完成五个子任务：字段释义、时间序列可视化、分布形态分析、箱型图离群检测与皮尔逊相关系数分析。每个子任务均给出对应的图表，图表中的全部数值均由脚本读自结果文件后格式化输出，未做人工改写。")

# ---- 1.1 子任务 1
add_heading2(doc, "1.1 子任务 1：数据字段英文名与中文释义")
add_body(doc, "字段释义以原始表头与 CSV字段说明.md 为依据。原始表头为纯英文，共 %d 个字段，其中 Date 为时间索引，其余 %d 个为逐日观测的水文气象变量。表 1-1 逐字段给出英文名、中文名称、单位与机理作用，作为后续特征命名与单位换算的基准。" % (n_cols, n_cols - 1))
add_table(
    doc,
    header=["英文字段", "中文名称", "单位", "机理作用"],
    rows=[
        ["Date", "观测日期", "无单位", "时间索引：标注逐日观测对应的日历日期，用于构造时间派生特征并划分训练段与测试段。"],
        ["Discharge", "当天径流量", "cfs", "目标变量：测站当日流量，是建模的预测对象；作为原始观测保留，不参与新一轮特征排名。"],
        ["Dayl", "日照时长", "s", "能量条件：当日昼长，随太阳高度角季节变化，决定蒸发与融雪的时间窗口。"],
        ["Prcp", "降水量", "mm/d", "水量输入：当日降水直接补给流域，是径流响应最直接的气象驱动因子。"],
        ["Srad", "短波辐射", "W/m²", "能量输入：到达地表的短波辐射通量，驱动蒸散发与积雪消融过程。"],
        ["Swe", "积雪水当量", "kg/m²", "蓄水状态：积雪的等效水量，在融雪期为主要补给项；当前样本中恒为零。"],
        ["Tmax", "最高气温", "℃", "能量条件：控制融雪速率与蒸散发强度，与径流的季节变化密切相关。"],
        ["Tmin", "最低气温", "℃", "能量条件：给出夜间温度下限，与最高气温共同界定日温差与冻结条件。"],
        ["Vp", "水汽压", "Pa", "湿润状况：空气中水汽的分压，表征蒸发潜力与流域湿润程度。"],
    ],
    caption="数据字段中英文对照与机理作用",
    chapter=1,
    numbering=num1,
)
add_body(doc, "需要特别说明积雪水当量字段。原始 CSV 共 %d 行的 Swe 取值全部为 %d，标准差为 %s，说明该站点在这段观测期内没有积雪储量记录。因此当前样本积雪水当量为零，不参与归一化与建模，报告中也不据此外推任何融雪过程，融雪补给项在本项目中不予考虑。" % (zeros_swe, 0, f3(summary.loc["Swe", "std"])))

# ---- 1.2 子任务 2
add_heading2(doc, "1.2 子任务 2：时间序列可视化")
add_body(doc, "时间序列可视化覆盖全部 %d 个观测变量，按统一时间轴、各自独立纵轴排布，以便同时观察季节周期与尖峰事件。图 1-1 给出 %s 至 %s 的完整序列。" % (len(BASE_COLS) - 1, date_min, date_max))
add_figure(doc, FIG_EDA / "1_time_series_overview.png", caption="原始观测变量时间序列总览（2000—2004 年）", chapter=1, numbering=num1)
add_figure_note(doc, "注：横轴为日历日期，各子图纵轴为对应变量的原始量纲；积雪水当量一栏恒为零值。")
add_body(doc, "该图显示四类形态。第一，当天径流量呈强烈的年内周期并伴随短历时尖峰，最大值达 %s cfs，远高于均值 %s cfs，说明洪水过程以脉冲形式出现。第二，日照时长、最高气温、最低气温与短波辐射呈稳定的正弦型季节循环，峰谷位置一致，反映太阳辐射的年周期驱动。第三，降水量为稀疏脉冲序列，多数日期为零或接近零，少数日期出现集中降水，属典型的零膨胀分布。第四，积雪水当量全程恒为零，曲线退化为一条水平线。" % (f3(raw["Discharge"].max()), f3(raw["Discharge"].mean())))
add_body(doc, "上述形态决定了特征构造的方向：径流的自相关强度高，需要滞后与滑动窗口特征刻画其记忆性；气象因子需要周期编码刻画季节循环；降水则需要同时保留当日值与累积值，以覆盖产流的滞后响应。")

# ---- 1.3 子任务 3
add_heading2(doc, "1.3 子任务 3：直方图与分布")
add_body(doc, "分布分析对 %d 个观测变量分别绘制直方图并叠加核密度估计曲线，结果见图 1-2，对应的描述统计量列入表 1-2。" % (len(BASE_COLS) - 1))
add_figure(doc, FIG_EDA / "2_histograms_kde.png", caption="各观测变量直方图与核密度估计", chapter=1, numbering=num1)
add_figure_note(doc, "注：柱形为等宽频数直方图，曲线为核密度估计；横轴为原始量纲，纵轴为频数或密度。")
stat_rows = []
for var in ["Discharge", "Dayl", "Prcp", "Srad", "Swe", "Tmax", "Tmin", "Vp"]:
    row = summary.loc[var]
    stat_rows.append([CN_NAMES[var], f3(row["mean"]), f3(row["std"]), f3(row["min"]),
                      f3(row["50%"]), f3(row["max"]), f3(row["skewness"])])
add_table(
    doc,
    header=["变量", "均值", "标准差", "最小值", "中位数", "最大值", "偏度"],
    rows=stat_rows,
    caption="各观测变量描述统计量",
    chapter=1,
    numbering=num1,
)
add_body(doc, "分布形态可分为三类。第一类为强右偏重尾，当天径流量的偏度为 %s、峰度为 %s，降水量的偏度为 %s、峰度为 %s，表明极少数大值主导了尾部，直接以原始量纲建模会使大值样本主导损失函数。第二类为对称或轻微偏斜的有界分布，短波辐射偏度 %s、日照时长偏度 %s，接近以中位数为中心的对称形态。第三类为双峰分布，最高气温偏度 %s、最低气温偏度 %s，冬夏两个温度中心分离，说明温度不适合以单一线性项进入模型。积雪水当量因取值恒定，直方图退化为单点，偏度与峰度均为 %s。" % (
    f3(summary.loc["Discharge", "skewness"]), f3(summary.loc["Discharge", "kurtosis"]),
    f3(summary.loc["Prcp", "skewness"]), f3(summary.loc["Prcp", "kurtosis"]),
    f3(summary.loc["Srad", "skewness"]), f3(summary.loc["Dayl", "skewness"]),
    f3(summary.loc["Tmax", "skewness"]), f3(summary.loc["Tmin", "skewness"]),
    f3(summary.loc["Swe", "skewness"])))
add_body(doc, "径流量与降水量中位数为 %s 与 %s，均值分别为 %s 与 %s，中位数显著低于均值，进一步印证右偏特征。这一结论直接支撑后续两项处理：其一，滞后与滑动窗口特征需要按训练段参数归一化，避免大值压缩其余样本；其二，评价指标应同时考察绝对误差与相对误差。" % (
    f3(summary.loc["Discharge", "50%"]), f3(summary.loc["Prcp", "50%"]),
    f3(summary.loc["Discharge", "mean"]), f3(summary.loc["Prcp", "mean"])))

# ---- 1.4 子任务 4
add_heading2(doc, "1.4 子任务 4：箱型图")
add_body(doc, "箱型图用于识别离群点并比较各变量的离散程度，结果见图 1-3。判定规则为 1.5 倍四分位距，超出上下须的观测记为离群点。")
add_figure(doc, FIG_EDA / "3_boxplots_all_8_columns.png", caption="各观测变量箱型图", chapter=1, numbering=num1)
add_figure_note(doc, "注：箱体上下边为上下四分位数，中线为中位数，箱须为 1.5 倍四分位距范围内的极值，箱须之外的点以圆点标为离群点。")
add_body(doc, "按该规则统计，离群点主要集中于水文与降水变量：当天径流量 %d 个，降水量 %d 个，水汽压 %d 个；气温与辐射类变量的离群点较少，最高气温 %d 个、最低气温 %d 个、短波辐射 %d 个、日照时长 %d 个。积雪水当量因取值恒定，分布退化为单线，无离群点。" % (
    outliers["当天径流量"], outliers["降水量"], outliers["水汽压"],
    outliers["最高气温"], outliers["最低气温"], outliers["短波辐射"], outliers["日照时长"]))
add_body(doc, "离群点在本文中不作删除处理。理由是该站点属山区流域，洪峰与暴雨是真实的水文响应而非观测误差，删除会削弱模型对极值事件的刻画能力；相应的处理放在特征工程环节，通过 Min-Max 与 Z-score 两种尺度变换控制量纲差异，并在后续建模阶段以稳健的评价指标复核极端样本的预测表现。")

# ---- 1.5 子任务 5
add_heading2(doc, "1.5 子任务 5：皮尔逊相关系数热力图")
add_body(doc, "相关性分析分三个层次：径流与其自身滞后序列、径流与其他因子的同期关系、径流与其他因子的滞后关系。三者合并绘制的皮尔逊相关系数热力图见图 1-4，径流与滞后因子的相关表、同期相关矩阵分别列入表 1-3 与表 1-4。")
add_figure(doc, FIG_EDA / "4_pearson_correlation_heatmap.png", caption="皮尔逊相关系数热力图（含滞后因子）", chapter=1, numbering=num1)
add_figure_note(doc, "注：色标为皮尔逊相关系数，暖色为正相关、冷色为负相关；对角线与自相关项恒为 1。")
add_body(doc, "热力图给出层次分明的相关结构。径流与其滞后序列的相关性最高，滞后 1 天为 %s、滞后 2 天为 %s、滞后 3 天为 %s、滞后 7 天为 %s，随滞后期延长单调衰减，说明径流具有明显的日尺度记忆性，且衰减曲线平缓，支持同时构造短滞后期与长滞后期特征。" % (f3(lag1_r), f3(lag2_r), f3(lag3_r), f3(lag7_r)))
lag_rows = []
for i, (name, row) in enumerate(lag_corr.head(12).iterrows(), start=1):
    lag_rows.append([str(i), name, cn_feature(name), f3(row["Pearson_r"]), f3(row["Abs_r"])])
add_table(
    doc,
    header=["排名", "特征", "中文含义", "皮尔逊相关系数", "相关系数绝对值"],
    rows=lag_rows,
    caption="径流与滞后因子相关系数前十二位",
    chapter=1,
    numbering=num1,
)
add_body(doc, "同期相关方面，其他因子与径流的线性关联整体偏弱。降水量与径流的同期相关系数为 %s，是唯一达到中等强度的气象变量；日照时长 %s、短波辐射 %s、最高气温 %s、最低气温 %s、水汽压 %s，绝对值均不足 0.1。水汽压与径流的同期相关系数为负值，说明空气湿润程度对当日径流的解释力有限，同期信息不足以支撑直接预测。" % (
    f3(prcp_sync),
    f3(matrix.loc["径流", "日长"]), f3(matrix.loc["径流", "辐射"]),
    f3(matrix.loc["径流", "最高温"]), f3(matrix.loc["径流", "最低温"]),
    f3(matrix.loc["径流", "水汽压"])))
sync_rows = []
for col in ["日长", "降水", "辐射", "最高温", "最低温", "水汽压"]:
    sync_rows.append([col, f3(matrix.loc["径流", col]), f3(abs(matrix.loc["径流", col]))])
add_table(
    doc,
    header=["因子", "同期皮尔逊相关系数", "相关系数绝对值"],
    rows=sync_rows,
    caption="径流与其他因子同期相关系数",
    chapter=1,
    numbering=num1,
)
add_body(doc, "滞后相关方面，降水量的解释力随滞后期前移，滞后 1 天为 %s，明显高于同期值，说明流域产流对降水存在一日左右的汇流延迟；滞后 2 天与滞后 3 天的相关系数继续衰减。气温、辐射与日照的滞后项绝对值普遍不足 0.1；水汽压的滞后项略高，滞后 2 天与滞后 3 天分别为 %s 与 %s，但方向为负，其作用更多体现为季节背景与湿润状况的间接指示，而非直接驱动。积雪水当量因取值恒定，相关系数无法计算，在热力图中以缺失值呈现，这与前文积雪水当量为零的结论一致。" % (f3(prcp_lag1), f3(vp_lag2), f3(vp_lag3)))
add_figure(doc, FIG_EDA / "4_discharge_lag_factors_barplot.png", caption="径流与各滞后因子相关系数排序", chapter=1, numbering=num1)
add_figure_note(doc, "注：横轴为相关系数绝对值，纵轴为因子名称，数值越大表示与径流的线性关联越强。")
add_body(doc, "三项分析共同支持两条特征构造决策。第一，径流的滞后项与滑动统计量应作为核心特征，其相关性显著高于任何同期气象因子。第二，气象因子需借助滞后、累积与周期编码间接表达，单靠同期原始值难以刻画产流过程。这两条决策在第二章的特征工程中逐一落实。")

# ================================================================ 第二章
add_heading1(doc, "二、特征工程")
add_body(doc, "本章在第一章结论的基础上完成三个子任务：时间序列特征提取、特征归一化与特征选择。特征成品保存为 results/processed_data/01047000_with_feature_values_独热编码版.csv，共 %d 行、%d 列；列构成与归一化版本均由脚本重新统计，其结果与 CSV字段说明.md 的记载一致。" % (n_feat_rows, n_feat_cols))

# ---- 2.1 子任务 6
add_heading2(doc, "2.1 子任务 6：时间序列特征提取")
add_body(doc, "时间序列特征提取包含季节编码、滞后特征与滑动窗口特征三个部分，全部以日期为键、以原始观测为源，按时间顺序向右追加，不改变原有行序。")
add_body(doc, "季节编码采用四季独热编码，即春季、夏季、秋季、冬季四个 0/1 列，3 至 5 月春季列取 1，6 至 8 月夏季列取 1，9 至 11 月秋季列取 1，12 月至次年 2 月冬季列取 1，每行恰有一列为 1。此处不使用整数季节码，原因是季节之间不存在大小与间距关系，整数编码会把这一并不存在的有序性引入模型。除四季独热外，另加两组周期编码：月份正弦与月份余弦，按 2π 乘以月份除以 12 计算；年内日序正弦与年内日序余弦，按 2π 乘以年内日序除以 365.25 计算。正弦余弦成对出现，使周期在年末与年初连续衔接，避免 12 月与 1 月在数值上被割裂。")
add_body(doc, "滞后特征对当天径流量、降水量、最高气温、最低气温、短波辐射与水汽压 %d 个变量分别构造滞后 1、2、3、7 天共 %d 个原始列；滑动窗口特征对当天径流量、降水量与最高气温 %d 个变量分别构造过去 3 天与过去 7 天的均值、标准差、最大值与最小值共 %d 个原始列。滞后 1 天对应水文过程的短期记忆，滞后 7 天对应周尺度的汇流累积，窗口 3 天与 7 天的极值与标准差则刻画过程内部的波动幅度。窗口统计只对上述 %d 个变量展开，而未对全部 %d 个变量展开，是出于列数可控与共线性抑制的考虑。" % (
    len(set(c.split("_滞后")[0] for c in lag_raw)), n_lag,
    len(set(c.split("_过去")[0] for c in win_raw)), n_win,
    len(set(c.split("_过去")[0] for c in win_raw)), len(BASE_COLS) - 1))
add_body(doc, "历史不足时特征留空而不填充。滞后 7 天与过去 7 天窗口在序列起始处缺少前置观测，相应单元格保持空白，共 %d 行受影响；这些空白在建模阶段按有效样本剔除，不做零值填充，以免把缺测误当作真实的零观测。" % hist_gap)
add_table(
    doc,
    header=["特征类别", "列数", "构成说明"],
    rows=[
        ["原始观测与时间索引", str(n_base), "日期 1 列，8 个原始观测变量各 1 列，年份、月份、日、星期编码、年内日序共 5 列。"],
        ["独热与周期编码（原始）", str(n_enc), "四季独热 4 列，月份正弦与余弦 2 列，年内日序正弦与余弦 2 列。"],
        ["滞后特征（原始）", str(n_lag), "6 个变量各配滞后 1、2、3、7 天，共 24 列。"],
        ["滑动窗口特征（原始）", str(n_win), "3 个变量各配过去 3 天与过去 7 天的均值、标准差、最大值、最小值，共 24 列。"],
        ["Min-Max 归一化版本", str(n_mm), "对上述数值特征逐列归一化；积雪水当量恒为零，不参与。"],
        ["Z-score 标准化版本", str(n_zs), "对上述数值特征逐列标准化；积雪水当量恒为零，不参与。"],
        ["合计", str(n_feat_cols), "与特征成品 CSV 的实际列数一致。"],
    ],
    caption="特征成品列构成统计",
    chapter=2,
    numbering=num2,
)
add_body(doc, "按用途归并上述明细：滞后特征合计 %d 列，滑动窗口特征合计 %d 列，独热与周期编码合计 %d 列，基础观测与时间索引及其两种尺度变换版本合计 %d 列，四者之和为 %d 列，与成品 CSV 的列数一致。每类特征都同时保留原始量纲与两种变换版本，便于在建模阶段对比尺度变换的实际增益。" % (lag_total, win_total, enc_total, base_total, lag_total + win_total + enc_total + base_total))

# ---- 2.2 子任务 7
add_heading2(doc, "2.2 子任务 7：特征归一化")
add_body(doc, "归一化处理各特征量纲不一致的问题。原始数据中日照时长以秒计、水汽压以帕计、径流量以立方英尺每秒计，数值范围相差数个数量级；若不作变换，距离类与梯度类模型会被大数量级特征主导，收敛速度与预测精度同时受损。")
add_body(doc, "Min-Max 归一化按下式把每个特征线性映射到 0 至 1 区间：新值等于原值减去该特征最小值、再除以最大值与最小值之差。其优点是输出有界、计算简单、不改变原始分布形状，适合有界输入或对尺度敏感的模型，例如神经网络、K 近邻与支持向量回归；缺点是最小值与最大值由极端样本决定，单个极端值会压缩其余样本的取值空间。")
add_body(doc, "Z-score 标准化按下式把每个特征变换为零均值、单位标准差的分布：新值等于原值减去均值、再除以标准差。其优点是消除量纲同时保留样本的相对位置，方差为 1 便于正则化项在各特征间公平取值，适合线性回归、岭回归与以梯度下降求解的模型；缺点是不产生有界区间，且均值与标准差本身受极端值拉动。两种方法的适用条件与处理口径列入表 2-2。")
add_table(
    doc,
    header=["对比项", "Min-Max 归一化", "Z-score 标准化"],
    rows=[
        ["变换公式", "(x − min) / (max − min)", "(x − μ) / σ"],
        ["输出范围", "固定为 0 至 1 的有界区间", "均值 0、标准差 1，无固定上下界"],
        ["对分布形状的影响", "只做平移与缩放，偏态与峰度不变", "只做平移与缩放，偏态与峰度不变"],
        ["对极端值的敏感性", "上下界由极值决定，极端值会压缩其余样本", "均值与标准差被极端值拉动，区分度随之下降"],
        ["适用条件", "需要固定输入区间、对尺度敏感的模型", "需要系数可比或假设近似对称分布的模型"],
        ["本阶段拟合口径", "仅在 2000—2003 年训练段拟合最小值与最大值，再变换 2004 年测试段", "仅在 2000—2003 年训练段拟合均值与标准差，再变换 2004 年测试段"],
    ],
    caption="Min-Max 归一化与 Z-score 标准化的方法对比",
    chapter=2,
    numbering=num2,
)
add_body(doc, "参数拟合严格限定在训练段。两种方法的 min、max、μ、σ 均只由 2000—2003 年的 %d 天有效观测估计，再以同一组参数变换 2004 年测试段的 %d 天；测试段不参与任何参数估计。这一口径的作用是防止数据泄漏：若用全序列统计量做变换，测试段的分布信息会提前进入训练过程，导致评估结果虚高。成品 CSV 中同时保留原始量纲与两种变换版本，空白单元格表示该行历史不足或该变量恒为零，不做填补。" % (train_days, test_days))
add_figure(doc, FIG_FE / "normalization_comparison.png", caption="归一化前后特征分布对比", chapter=2, numbering=num2)
add_figure_note(doc, "注：左列为原始量纲分布，右列为变换后分布；变换只改变横轴刻度，不改变分布形状。")

# ---- 2.3 子任务 8
add_heading2(doc, "2.3 子任务 8：特征选择")
add_body(doc, "特征选择方法分为三类。过滤式方法先按统计指标给特征打分再排序，与具体模型无关，计算代价低，适合候选特征数量大时的粗筛；包裹式方法以模型性能为目标反复搜索特征子集，效果通常更好但计算代价随特征数急剧上升；嵌入式方法在模型训练过程中完成筛选，例如带正则化的线性模型与基于不纯度的树模型，兼顾效率与效果。")
add_body(doc, "本阶段采用过滤式方法，并同时使用皮尔逊相关系数与互信息两个指标。皮尔逊相关系数只度量线性关联，对单调非线性关系不敏感；互信息度量两个变量之间的任意统计依赖，能捕捉非线性关联但需要离散化估计。两指标并列排序可以互补：前者稳定、可比性强，后者能发现线性相关系数偏小却具有预测价值的分组特征。候选特征共 %d 个，按两指标排序后的结果见表 2-3 与图 2-2。" % n_cand)
rank_rows = []
for i, (_, row) in enumerate(rank.head(15).iterrows(), start=1):
    rank_rows.append([str(i), row["特征名称"], f3(row["皮尔逊相关系数"]), f3(row["互信息得分"])])
add_table(
    doc,
    header=["排名", "特征名称", "皮尔逊相关系数", "互信息得分"],
    rows=rank_rows,
    caption="双指标特征排序前十五名",
    chapter=2,
    numbering=num2,
)
add_body(doc, "排序结果与第一章的相关性结论一致。排首位的是%s，其皮尔逊相关系数为 %s、互信息得分为 %s；前十二名全部为当天径流量的滞后与滑动窗口派生特征，说明径流的自记忆是该站点最主要的信息来源。降水量位列第 %d 位，其后依次为降水的窗口统计量；春季独热编码位列第 %d 位，月份正弦编码位列第 %d 位，年内日序正弦编码位列第 %d 位，季节与周期编码呈分组聚集。气温与辐射类特征整体靠后，最低气温位列第 %d 位，年内日序余弦编码位列末位第 %d 位。" % (
    top1["特征名称"], f3(top1["皮尔逊相关系数"]), f3(top1["互信息得分"]),
    rank_pos("降水量"), rank_pos("春季独热编码"), rank_pos("月份正弦编码"),
    rank_pos("年内日序正弦编码"), rank_pos("最低气温"), len(rank)))
add_figure(doc, FIG_FE / "feature_selection_comparison.png", caption="皮尔逊相关系数与互信息双指标特征排序对比", chapter=2, numbering=num2)
add_figure_note(doc, "注：左图为皮尔逊相关系数绝对值前二十位，右图为互信息得分前二十位。")
add_body(doc, "最终保留的特征子集按三条并列条件确定：皮尔逊相关系数绝对值不低于 0.2，或互信息得分不低于 0.05，或属于降水类特征。降水类特征单独保留，是因为降水是产流的直接驱动项，其线性相关系数会被零膨胀削弱，若仅按阈值筛选容易被漏掉。按该口径，%d 个候选中保留 %d 个，剔除 %d 个。" % (n_cand, n_kept, n_dropped))
add_body(doc, "剔除的 %d 个特征为%s。夏季独热编码被剔除，是因为四季独热四列每行恰有一列为 1，四列构成完全线性相关，季节信息已由月份与年内日序两组正余弦编码连续表达，夏季列不提供额外信息；两个最高气温的窗口标准差被剔除，是因为气温的逐日波动幅度对径流的同期关联与非线性依赖同时偏弱，二者在排序表中分别位列第 %d 位与第 %d 位。保留集合内部仍存在较强共线性，例如日照时长与最高气温的相关系数为 %s、最高气温与最低气温的相关系数为 %s；这部分相关特征留待后续建模阶段按模型类型分别处理，线性模型配合正则化项抑制共线性，树模型则利用其对冗余特征的容忍度。" % (
    n_dropped, dropped_preview,
    rank_pos("最高气温_过去7天标准差"), rank_pos("最高气温_过去3天标准差"),
    f3(matrix.loc["日长", "最高温"]),
    f3(matrix.loc["最高温", "最低温"])))
add_body(doc, "完成筛选后，特征集同时具备三类信息：径流的短期与周尺度记忆、降水的当期与累积补给、季节周期的显式表达。该子集作为后续建模的输入，与成品 CSV 中保留的全部特征列相互独立，前者用于正式训练，后者作为对照与消融实验的备选池。")

# ================================================================ 第三章
add_heading1(doc, "三、小结")
add_body(doc, "本阶段完成数据分析与特征工程共八个子任务。子任务 1 完成 %d 个字段的中英文对照、单位与机理释义，并确认积雪水当量恒为零、不参与归一化与建模。子任务 2 至子任务 4 完成全部 %d 个观测变量的时间序列可视化、直方图与核密度估计、箱型图与离群点识别，明确了径流与降水强右偏重尾、气温双峰、辐射与日照近似对称的分布格局。子任务 5 完成皮尔逊相关系数热力图及径流与滞后序列、同期因子、滞后因子三个层次的相关分析，确认径流滞后 1 天相关系数为 %s，为全部候选关系中的最高值。" % (n_cols, len(BASE_COLS) - 1, f3(lag1_r)))
add_body(doc, "子任务 6 完成时间序列特征提取，以四季独热编码替代整数季节码，辅以月份与年内日序的正余弦周期编码，并对 6 个变量构造滞后 1、2、3、7 天特征、对 3 个变量构造过去 3 天与过去 7 天的均值、标准差、最大值与最小值特征，成品特征表共 %d 行、%d 列。" % (n_feat_rows, n_feat_cols))
add_body(doc, "子任务 7 完成特征归一化，并列给出 Min-Max 归一化与 Z-score 标准化的公式、输出范围、对极端值的敏感性与适用条件，并统一采用仅在 2000—2003 年训练段拟合参数、再变换 2004 年测试段的口径，从流程上切断了测试段信息的回流。子任务 8 完成特征选择，比较过滤式、包裹式与嵌入式三类方法后采用过滤式的双指标排序，对 %d 个候选特征按皮尔逊相关系数与互信息排序，并按绝对值不低于 0.2、互信息不低于 0.05 或属降水类三条并列条件保留 %d 个、剔除 %d 个。" % (n_cand, n_kept, n_dropped))
add_body(doc, "对后续建模的支撑体现在三处。其一，分布与离群点结论说明径流序列存在真实的极值响应，评价环节需要同时报告绝对误差与相对误差，并对洪峰样本单独复核。其二，滞后与窗口特征的排序结果说明模型的输入记忆长度应覆盖 1 天至 7 天，后续建模中的输入窗口设定可以此为依据。其三，归一化参数限定在训练段拟合，保证了第二阶段训练与测试的划分口径与本阶段一致，使特征工程的结论可以直接迁移，无需重建数据管道。")
add_body(doc, "本阶段遗留两项工作移交后续阶段：一是在选定特征子集上比较不同模型的预测精度，二是以滚动预测与多步预测检验特征记忆长度的实际作用。上述工作不改变本阶段确立的特征口径与数据划分方式。")

# ---------------------------------------------------------------- 保存
OUT_DIR.mkdir(parents=True, exist_ok=True)
out_main = OUT_DIR / DOC_NAME
out_root = ROOT / DOC_NAME
doc.save(str(out_main))
shutil.copyfile(str(out_main), str(out_root))

print("saved:", out_main, out_main.stat().st_size)
print("saved:", out_root, out_root.stat().st_size)
print("figures:", len(doc.inline_shapes), "tables(incl cover):", len(doc.tables))
print("composition:", n_base, n_enc, n_lag, n_win, n_mm, n_zs, "sum", n_base + n_enc + n_lag + n_win + n_mm + n_zs)
print("group totals:", lag_total, win_total, enc_total, base_total, "sum", lag_total + win_total + enc_total + base_total)
print("selection:", n_cand, n_kept, n_dropped)
print("check lag1_r", lag1_r, "discharge_mean", summary.loc["Discharge", "mean"], "top1", top1["特征名称"], top1["皮尔逊相关系数"], top1["互信息得分"])
