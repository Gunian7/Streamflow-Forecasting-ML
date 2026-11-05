# -*- coding: utf-8 -*-
"""生成《机器学习课程实践》第二次阶段验收报告（修正数据版结果）。

数据口径（全部可回溯到结果文件，报告中的数字一律由脚本读出，不在此处写死）：
    results/phase2_v5/测试精度指标.csv              28 行测试精度指标
    results/phase2_v5/选参结果.csv                  16 行选参结果
    results/phase2_v5/*_网格搜索.csv                16 份网格搜索结果
    results/phase2_v5/逐日预测结果.csv              测试段逐日预测
    results/phase2_v5/run_phase2.log                数据边界（样本数、特征数、日期）
    results/phase2_v5/figures/*.png                 三张结果图，直接引用，不重绘
    archive/results_phase2_v5_raw/测试精度指标.csv  未修正数据版精度（只读对照）
"""

import json
import re
import shutil
import sys
from pathlib import Path

import pandas as pd
from docx.shared import Pt

import report_style as rs

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
P2 = ROOT / "results" / "phase2_v5"
FIG_DIR = P2 / "figures"
RAW_DIR = ROOT / "archive" / "results_phase2_v5_raw"
OUT_DIR = ROOT / "本地资料" / "第二阶段"
DOC_NAME = "《机器学习课程实践》第二次阶段验收报告.docx"
TEST_START = pd.Timestamp("2004-01-01")
HORIZONS = list(range(1, 8))
GROUPS = (("ANN", "直接"), ("ANN", "多输出"), ("RF", "直接"), ("RF", "多输出"))

FIG1 = FIG_DIR / "fig1_精度随预见期衰减.png"
FIG2 = FIG_DIR / "fig2_策略对比.png"
FIG3 = FIG_DIR / "fig3_2004年水文过程线.png"


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


# ------------------------------------------------------------------ 数据读取

def load_meta():
    """从运行日志读出数据边界：训练与测试样本数、特征数、起止日期。"""
    text = (P2 / "run_phase2.log").read_text(encoding="utf-8")
    block = re.search(r'\{\s*"数据文件".*?\n\}', text, re.S)
    if block is None:
        raise ValueError("run_phase2.log 中未找到数据边界记录")
    meta = json.loads(block.group(0))
    needed = ("训练起", "训练止", "训练样本", "测试起", "测试止", "测试样本", "特征数量")
    missing = [k for k in needed if k not in meta]
    if missing:
        raise KeyError("数据边界缺少字段: %s" % missing)
    return meta


def load_metrics(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    table = {}
    for _, row in df.iterrows():
        table[(row["模型"], row["策略"], int(row["预见期（天）"]))] = row
    return table


def load_predictions():
    pred = pd.read_csv(P2 / "逐日预测结果.csv", encoding="utf-8-sig")
    pred["日期"] = TEST_START + pd.to_timedelta(pred["样本序号"] + pred["预见期（天）"], "D")
    return pred


def pick(pred, model, strategy, h):
    seg = pred[(pred["模型"] == model) & (pred["策略"] == strategy) & (pred["预见期（天）"] == h)]
    return seg.sort_values("样本序号").reset_index(drop=True)


def find_col(frame, needle):
    for name in frame.columns:
        if needle in name:
            return name
    raise KeyError(needle)


# ------------------------------------------------------------------ 指标取值

def field(table, model, strategy, h, name):
    return float(table[(model, strategy, h)][name])


def r2(table, model, strategy, h):
    return field(table, model, strategy, h, "R²")


def rmse(table, model, strategy, h):
    return field(table, model, strategy, h, "RMSE（cfs）")


def mae(table, model, strategy, h):
    return field(table, model, strategy, h, "MAE（cfs）")


def mean_r2(table, model, strategy):
    return sum(r2(table, model, strategy, h) for h in HORIZONS) / len(HORIZONS)


def days_text(days):
    return "、".join("第 %d 天" % h for h in days)


def require(text, value, digits, where):
    """断言格式化后的 CSV 数值确实出现在生成的段落里（防占位符错位）。"""
    token = ("%%.%df" % digits) % value
    if token not in text:
        raise AssertionError("%s：段落中缺少 CSV 值 %s" % (where, token))
    return token


# ------------------------------------------------------------------ 表格构造

def fmt_hidden(values):
    return "(" + ", ".join(str(int(v)) for v in values) + ",)"


def params_table(grid, ann=True):
    """网格搜索明细表：各折交叉验证 RMSE（取负号还原）与平均 RMSE。"""
    rows = []
    if ann:
        header = ["alpha", "隐层结构", "折 1 RMSE", "折 2 RMSE", "折 3 RMSE", "平均 RMSE", "排名"]
        alpha_col = find_col(grid, "alpha")
        hidden_col = find_col(grid, "hidden_layer_sizes")
        for _, row in grid.iterrows():
            rows.append(["%.1f" % row[alpha_col],
                         str(row[hidden_col]).replace(" ", ""),
                         "%.4f" % (-row["split0_test_score"]),
                         "%.4f" % (-row["split1_test_score"]),
                         "%.4f" % (-row["split2_test_score"]),
                         "%.4f" % (-row["mean_test_score"]),
                         "%d" % int(row["rank_test_score"])])
    else:
        header = ["最大深度", "叶节点最小样本数", "折 1 RMSE", "折 2 RMSE", "折 3 RMSE", "平均 RMSE", "排名"]
        depth_col = find_col(grid, "max_depth")
        leaf_col = find_col(grid, "min_samples_leaf")
        for _, row in grid.iterrows():
            depth = row[depth_col]
            depth_text = "None" if pd.isna(depth) else "%d" % int(depth)
            rows.append([depth_text, "%d" % int(row[leaf_col]),
                         "%.4f" % (-row["split0_test_score"]),
                         "%.4f" % (-row["split1_test_score"]),
                         "%.4f" % (-row["split2_test_score"]),
                         "%.4f" % (-row["mean_test_score"]),
                         "%d" % int(row["rank_test_score"])])
    return header, rows


def sel_items(sel, model):
    """选参结果：某模型的 8 组（7 个直接预见期 + 1 个多输出）。"""
    items = []
    for _, row in sel[sel["模型"] == model].iterrows():
        items.append({
            "策略": row["策略"],
            "预见期": row["选参预见期"],
            "参数": json.loads(row["最佳参数"]),
            "cv": float(row["交叉验证RMSE（对数尺度）"]),
            "秒": float(row["耗时秒"]),
        })
    return items


def selected_rows(items, ann=True):
    header = ["策略", "预见期（天）", "alpha" if ann else "最大深度",
              "隐层结构" if ann else "叶节点最小样本数", "交叉验证 RMSE", "耗时（秒）"]
    rows = []
    for item in items:
        params = item["参数"]
        horizon = "1–7" if item["策略"] == "多输出" else "%d" % int(item["预见期"])
        if ann:
            first = "%.1f" % params["regressor__model__alpha"]
            second = fmt_hidden(params["regressor__model__hidden_layer_sizes"])
        else:
            depth = params["max_depth"]
            first = "None" if depth is None else "%d" % depth
            second = "%d" % params["min_samples_leaf"]
        rows.append([item["策略"], horizon, first, second, "%.4f" % item["cv"], "%.1f" % item["秒"]])
    return header, rows


PRECISION_HEADER = ["预见期（天）", "RMSE（cfs）", "MAE（cfs）", "R²", "NSE", "样本数"]


def precision_rows(table, model, strategy):
    rows = []
    for h in HORIZONS:
        rows.append(["%d" % h,
                     "%.2f" % rmse(table, model, strategy, h),
                     "%.2f" % mae(table, model, strategy, h),
                     "%.4f" % r2(table, model, strategy, h),
                     "%.4f" % field(table, model, strategy, h, "NSE"),
                     "%d" % int(field(table, model, strategy, h, "样本数"))])
    return rows


# ------------------------------------------------------------------ 报告正文

def build():
    for path in (FIG1, FIG2, FIG3):
        if not path.exists():
            raise FileNotFoundError("缺少结果图：%s" % path)

    meta = load_meta()
    n_train, n_test, n_feat = meta["训练样本"], meta["测试样本"], meta["特征数量"]
    t_lo, t_hi = meta["训练起"], meta["训练止"]
    e_lo, e_hi = meta["测试起"], meta["测试止"]

    m = load_metrics(P2 / "测试精度指标.csv")
    raw = load_metrics(RAW_DIR / "测试精度指标.csv")
    sel = pd.read_csv(P2 / "选参结果.csv", encoding="utf-8-sig")
    ann_grid = pd.read_csv(P2 / "ANN_1_网格搜索.csv", encoding="utf-8-sig")
    rf_grid = pd.read_csv(P2 / "RF_1_网格搜索.csv", encoding="utf-8-sig")
    pred = load_predictions()

    for key in GROUPS:
        for h in HORIZONS:
            if key + (h,) not in m:
                raise KeyError("精度指标缺少 %s 第 %d 天" % (key, h))
            got = int(field(m, key[0], key[1], h, "样本数"))
            if got != n_test:
                raise ValueError("样本数不一致：%s 第 %d 天为 %d" % (key, h, got))

    means = {(mo, st): mean_r2(m, mo, st) for mo, st in GROUPS}
    raw_means = {(mo, st): mean_r2(raw, mo, st) for mo, st in GROUPS}
    ann_win = [h for h in HORIZONS if r2(m, "ANN", "多输出", h) > r2(m, "ANN", "直接", h)]
    ann_lose = [h for h in HORIZONS if h not in ann_win]
    rf_win = [h for h in HORIZONS if r2(m, "RF", "多输出", h) > r2(m, "RF", "直接", h)]
    rf_lose = [h for h in HORIZONS if h not in rf_win]

    best_ann = ann_grid.loc[ann_grid["rank_test_score"].idxmin()]
    worst_ann = ann_grid.nsmallest(2, "mean_test_score")
    ann_alpha_col = find_col(ann_grid, "alpha")
    ann_hidden_col = find_col(ann_grid, "hidden_layer_sizes")

    rf_best_mean = float(rf_grid["mean_test_score"].max())
    rf_best_rows = rf_grid[rf_grid["mean_test_score"] == rf_best_mean]
    rf_depth_col = find_col(rf_grid, "max_depth")
    rf_leaf_col = find_col(rf_grid, "min_samples_leaf")

    def depth_text(value):
        return "None" if pd.isna(value) else "%d" % int(value)

    ann_items = sel_items(sel, "ANN")
    rf_items = sel_items(sel, "RF")
    ann_direct = [it for it in ann_items if it["策略"] == "直接"]
    ann_multi_item = [it for it in ann_items if it["策略"] == "多输出"][0]
    rf_direct = [it for it in rf_items if it["策略"] == "直接"]
    rf_multi_item = [it for it in rf_items if it["策略"] == "多输出"][0]

    if len(ann_items) != 8 or len(rf_items) != 8:
        raise AssertionError("选参结果应为 8 行：ANN=%d RF=%d" % (len(ann_items), len(rf_items)))

    def label_of(item):
        return "多输出策略" if item["策略"] == "多输出" else "第 %d 天" % int(item["预见期"])

    def split_labels(items, predicate):
        """按条件把全部选参行分成两组，断言两组互不重叠且覆盖完整。"""
        yes = [it for it in items if predicate(it)]
        no = [it for it in items if not predicate(it)]
        labels = [label_of(it) for it in yes] + [label_of(it) for it in no]
        if not yes or not no or len(labels) != len(items) or len(set(labels)) != len(items):
            raise AssertionError("选参结构分组不完整：%s" % labels)
        return "、".join(label_of(it) for it in yes), "、".join(label_of(it) for it in no), len(no)

    ann_alpha_1 = ann_direct[0]["参数"]["regressor__model__alpha"]
    ann_alpha_rest = sorted({it["参数"]["regressor__model__alpha"] for it in ann_items} - {ann_alpha_1})
    if len(ann_alpha_rest) != 1:
        raise AssertionError("ANN 惩罚项系数除第 1 天外不唯一：%s" % ann_alpha_rest)
    ann_alpha_other = ann_alpha_rest[0]

    ann_alpha_yes, _, _ = split_labels(
        ann_items, lambda it: it["参数"]["regressor__model__alpha"] == ann_alpha_1)
    ann_h16_txt, ann_h32_txt, n_ann_h32 = split_labels(
        ann_items, lambda it: tuple(it["参数"]["regressor__model__hidden_layer_sizes"]) == (16,))
    ann_leaf_txt, ann_alpha_txt, n_ann_alpha_rest = split_labels(
        ann_items, lambda it: it["参数"]["regressor__model__alpha"] == ann_alpha_1)

    rf_leaf5_txt, rf_leaf10_txt, n_rf_leaf10 = split_labels(
        rf_items, lambda it: it["参数"]["min_samples_leaf"] == 5)
    rf_d8_txt, rf_d16_txt, n_rf_d16 = split_labels(
        rf_items, lambda it: it["参数"]["max_depth"] == 8)

    ann_secs = [it["秒"] for it in ann_items]
    rf_secs = [it["秒"] for it in rf_items]

    # 测试段第 1 天预测的洪峰特征（用于过程线段落）
    ann_day1 = pick(pred, "ANN", "直接", 1)
    rf_day1 = pick(pred, "RF", "直接", 1)
    peak_row = ann_day1.loc[ann_day1["实测径流（cfs）"].idxmax()]
    peak_date = peak_row["日期"]
    peak_obs = float(peak_row["实测径流（cfs）"])
    peak_ann = float(ann_day1.loc[ann_day1["日期"] == peak_date, "预测径流（cfs）"].iloc[0])
    peak_rf = float(rf_day1.loc[rf_day1["日期"] == peak_date, "预测径流（cfs）"].iloc[0])
    spring = ann_day1[(ann_day1["日期"] >= pd.Timestamp("2004-03-01")) &
                      (ann_day1["日期"] <= pd.Timestamp("2004-06-30"))]
    s_row = spring.loc[spring["实测径流（cfs）"].idxmax()]
    s_date = s_row["日期"]
    s_obs = float(s_row["实测径流（cfs）"])
    s_ann = float(ann_day1.loc[ann_day1["日期"] == s_date, "预测径流（cfs）"].iloc[0])
    s_rf = float(rf_day1.loc[rf_day1["日期"] == s_date, "预测径流（cfs）"].iloc[0])

    doc = rs.new_document()
    rs.add_cover(
        doc,
        college="自动化学院",
        major="智能科学与技术",
        class_name="课程实践小组",
        **cover_fields(),
        project="基于机器学习的径流预测",
        content="基于神经网络与随机森林的径流预测",
        subtitle="第二次阶段验收报告",
        date_text="二〇二六年十月二十二日",
    )
    rs.begin_body_section(doc, 1)
    num = rs.Numbering()

    def P(text):
        return rs.add_body(doc, text)

    def T(header, rows, caption, chapter):
        return rs.add_table(doc, header, rows, caption=caption, chapter=chapter, numbering=num)

    def NOTE(text):
        return rs.add_body(doc, text, indent=False, size=Pt(9))

    def FIG(path, caption, chapter, note):
        rs.add_figure(doc, path, caption=caption, chapter=chapter, numbering=num)
        rs.add_figure_note(doc, note)

    # ---------------- 第一章 ----------------
    rs.add_heading1(doc, "一、基于神经网络的径流预测")
    P("本章围绕基于神经网络的径流预测展开，先梳理适用于径流预测的机器学习方法，"
      "再以人工神经网络完成超参数搜索、直接策略与多输出策略的建模与评估。")
    P("实验数据取自 01047000 流域 2000 年 1 月 1 日至 2004 年 12 月 31 日的逐日水文气象序列。"
      "气象变量按 Daymet v4 官方日值以年份与年积日为键与水文序列逐日对齐，"
      "积雪水当量取原始观测值，缺测的两日以前后相邻观测补齐，全部特征在该修正数据上重建。"
      "经特征工程与缺失行清理后，训练段为 %s 至 %s，共 %d 个有效样本；"
      "测试段为 %s 至 %s，共 %d 个样本；统一使用 %d 个特征，其中积雪水当量及其派生量共 8 个。"
      "训练段在前、测试段在后，二者在时间上不重叠，测试集仅在最终评估时使用一次，"
      "不参与任何超参数选择或模型筛选。"
      % (t_lo, t_hi, n_train, e_lo, e_hi, n_test, n_feat))
    P("超参数选择统一采用 GridSearchCV 配合 TimeSeriesSplit(n_splits=3, gap=7)，且只在训练段内部进行："
      "训练块与验证块之间空出 7 天，使验证样本的预见期目标不与会话块重叠，避免相邻样本的目标变量互相渗漏。"
      "目标变量取 log1p 变换后建模，评估时经 expm1 反变换回物理量纲，评价指标在原始量纲上计算。"
      "精度指标包括均方根误差（RMSE）、平均绝对误差（MAE）、决定系数（R²）与纳什效率系数（NSE）。")

    rs.add_heading2(doc, "1.1 子任务 1：适用于径流预测的机器学习方法综述")
    P("径流序列同时受降水、气温、融雪与流域蓄水状态的共同驱动，具有显著的非线性、非平稳与滞后特征，"
      "且观测样本量通常有限。按模型假设与学习方式，可将径流预测中应用较广的机器学习方法归为以下五类。")
    P("（1）人工神经网络。以多层感知机为代表，通过输入层、隐含层与输出层的加权连接及非线性激活函数逐层复合，"
      "逼近从气象水文驱动到径流响应的映射关系；训练以反向传播配合梯度类优化器最小化损失函数，"
      "模型复杂度由惩罚项系数与隐层规模共同控制。该类方法适用于样本量中等、驱动因子维度较高且响应关系强非线性的流域，"
      "对特征之间的共线性不敏感。优势在于表示能力强，能够同时吸收降水累积、气温、前期径流等多源信息，"
      "单一网络即可输出多个预见期。局限在于对超参数与初始化较敏感，训练结果存在随机性，"
      "外推能力受训练样本覆盖范围约束，且对极端值的响应倾向于被平滑。")
    P("（2）随机森林。以决策树为基学习器，通过自助重采样构造多棵回归树，"
      "并在每次分裂时随机抽取特征子集，最终以各树预测的均值作为输出。"
      "该类方法适用于样本量有限、特征含噪且存在异常值的流域，对特征尺度与分布假设不敏感，无需标准化。"
      "优势在于通过集成显著降低方差，训练与预测速度快，超参数少且结果稳健，可直接给出特征重要性，"
      "并原生支持多输出回归。局限在于难以外推到训练覆盖范围以外的量级，"
      "分段常数式的拟合在低流量平缓段易出现台阶，整体结构不透明。")
    P("（3）支持向量回归。在核函数映射的高维空间中寻找以 ε 不敏感带为界的回归超平面，"
      "对超出带宽的样本施加惩罚，并以结构风险最小化约束模型复杂度。"
      "该类方法适用于中小样本、特征维度适中的回归问题。优势在于泛化误差上界清晰，"
      "对高维特征与非线性核映射处理良好。局限在于核矩阵规模随样本数呈平方增长，长序列建模成本高，"
      "核参数与惩罚系数需要精细调优，且标准形式只支持单输出，多预见期须分别建模。")
    P("（4）梯度提升树。以加法模型的形式串行训练一系列弱回归树，每棵树拟合前一轮的残差或损失函数的负梯度，"
      "并以学习率控制单棵树的贡献。该类方法适用于特征异构、样本量从数百到数万、"
      "对拟合精度要求较高的场景。优势在于损失函数可按任务定制，精度通常高于单一模型，"
      "能自适应地筛选有效特征，对缺失值与异常值较为稳健。局限在于串行训练耗时较长，"
      "超参数较多（树的数量、深度、学习率、子采样比例），缺少早停时容易过拟合，"
      "在长预见期目标信噪比较低时更需谨慎控制训练轮数。")
    P("（5）循环神经网络与长短期记忆网络。循环网络在时间维度上共享参数，以隐状态递推传递历史信息；"
      "长短期记忆网络在此基础上引入输入门、遗忘门、输出门与细胞状态，缓解长序列训练中的梯度消失问题。"
      "该类方法适用于以序列整体为输入、需要显式刻画时序依赖的场景，如多步递归预测与多变量序列到序列建模。"
      "优势在于无需人工构造大量滞后特征即可学习长程依赖，适合序列较长的流域。"
      "局限在于训练成本与调参难度高于前馈网络，对样本量与序列长度要求较高，"
      "递归多步预测存在误差累积，可解释性弱于树模型。")
    P("综合训练样本规模（%d 个）、特征维度（%d 个）与实现成本，本报告以人工神经网络与随机森林为主要建模对象："
      "两者均可原生输出多个预见期，便于在同一数据口径下比较直接策略与多输出策略；"
      "支持向量回归与梯度提升树在本报告中仅作方法层面的梳理，未纳入实验；"
      "循环神经网络类方法对序列长度与训练资源的要求更高，本报告不作展开。" % (n_train, n_feat))

    rs.add_heading2(doc, "1.2 子任务 2：ANN 网格搜索确定超参数")
    P("人工神经网络采用 scikit-learn 的 MLPRegressor 实现，输入特征先经标准化，"
      "目标变量取 log1p 变换以压缩洪峰量级对损失的支配。超参数搜索空间为惩罚项系数 "
      "alpha ∈ {0.1, 1.0, 10.0} 与隐层结构 hidden_layer_sizes ∈ {(16,), (32,)}，共 6 组组合；"
      "验证方式为 GridSearchCV 配合 TimeSeriesSplit(n_splits=3, gap=7)，评分指标为对数尺度上的均方根误差，"
      "以负值形式供最大化使用。直接策略对 7 个预见期分别独立选参，多输出策略对 7 维输出整体选参一次。")
    header, rows = params_table(ann_grid, ann=True)
    T(header, rows, "ANN 网格搜索交叉验证结果（直接策略，预见期 1 天）", 1)
    mean_col = header.index("平均 RMSE")
    if sorted(r[mean_col] for r in rows) != sorted("%.4f" % (-float(v))
                                                   for v in ann_grid["mean_test_score"]):
        raise AssertionError("表 1-1 的平均 RMSE 列与网格搜索 CSV 不一致")
    worst_sorted = ["%.4f" % (-float(v)) for v in ann_grid["mean_test_score"].nsmallest(2)]
    if {float(r[ann_alpha_col]) for _, r in worst_ann.iterrows()} != {0.1}:
        raise AssertionError("排名最末的两组并非 alpha=0.1：%s" % worst_sorted)
    if "%.4f" % -float(best_ann["mean_test_score"]) in worst_sorted:
        raise AssertionError("最优组合与最差组合混淆")
    text = ("表 1-1 给出预见期 1 天的 6 组组合在三个时间折上的交叉验证 RMSE。"
            "alpha 取 0.1 的两组排名最末，平均交叉验证 RMSE 分别为 %s 与 %s，"
            "说明修正数据下过弱的正则化并不稳定；"
            "最优组合为 alpha=%.1f、隐层 %s，平均 RMSE 为 %.4f。"
            "与早期数据口径下各预见期普遍取 alpha=10.0 的结果不同，本版在部分预见期上选择了更弱的惩罚项，"
            "说明选参结果对数据对齐方式较为敏感，必须在最终数据口径下于训练段内部重新执行。"
            % (worst_sorted[0], worst_sorted[1], float(best_ann[ann_alpha_col]),
               str(best_ann[ann_hidden_col]).replace(" ", ""), -float(best_ann["mean_test_score"])))
    require(text, -float(best_ann["mean_test_score"]), 4, "1.2 网格搜索段")
    for _, row in worst_ann.iterrows():
        require(text, -float(row["mean_test_score"]), 4, "1.2 网格搜索段")
    P(text)
    header, rows = selected_rows(ann_items, ann=True)
    T(header, rows, "ANN 各预见期与多输出策略的选参结果", 1)
    text = ("8 组选参结果如表 1-2 所示。惩罚项系数在%s取 %.1f，其余 %d 组取 %.1f；"
            "隐层结构在%s取 16 个单元，其余 %d 组（%s）取 32 个单元。"
            "交叉验证 RMSE 由 1 天的 %.4f 单调上升至 7 天的 %.4f，与后文测试段精度随预见期衰减的方向一致。"
            "单次选参耗时 %.1f 至 %.1f 秒，多输出策略只需搜索一次，耗时降至 %.1f 秒。"
            "选参全程只在训练段内部完成，测试段未参与任何环节。"
            % (ann_alpha_yes, ann_alpha_1, n_ann_alpha_rest, ann_alpha_other,
               ann_h16_txt, n_ann_h32, ann_h32_txt,
               ann_direct[0]["cv"], ann_direct[-1]["cv"],
               min(ann_secs), max(ann_secs), ann_multi_item["秒"]))
    for value, digits in ((ann_direct[0]["cv"], 4), (ann_direct[-1]["cv"], 4),
                          (min(ann_secs), 1), (max(ann_secs), 1), (ann_multi_item["秒"], 1)):
        require(text, value, digits, "1.2 选参段")
    P(text)

    rs.add_heading2(doc, "1.3 子任务 3：ANN 直接策略 1–7 天精度")
    P("直接策略为每个预见期单独训练一个回归器：输入为同一组 %d 个特征，"
      "目标为该预见期对应的 log1p 径流，训练完成后经 expm1 反变换得到物理量纲的预测值。"
      "7 个预见期共训练 7 个模型，测试段 %d 个样本全部参与评估。" % (n_feat, n_test))
    T(PRECISION_HEADER, precision_rows(m, "ANN", "直接"), "ANN 直接策略 1–7 天测试精度", 1)
    NOTE("注：R² 与 NSE 在本项目中采用同一公式 1−SS_res/SS_tot，两列数值恒等，实为同一指标；"
         "RMSE 与 MAE 的单位为 cfs，样本数为测试段天数。")
    P("第 1 天的 R² 为 %.4f，RMSE 为 %.2f cfs，MAE 为 %.2f cfs；"
      "第 2 天至第 4 天的 R² 依次为 %.4f、%.4f、%.4f，呈单调递减；"
      "第 5 天降至 %.4f（RMSE %.2f cfs），第 6 天与第 7 天分别为 %.4f 与 %.4f。"
      "R² 为负值意味着预测误差平方和已超过以测试段均值作为预测的基准，"
      "即该预见期上的模型输出未能提供有效的水文信息。"
      "RMSE 与 MAE 的差距逐期扩大，反映误差主要由洪峰段的幅度偏差贡献。"
      % (r2(m, "ANN", "直接", 1), rmse(m, "ANN", "直接", 1), mae(m, "ANN", "直接", 1),
         r2(m, "ANN", "直接", 2), r2(m, "ANN", "直接", 3), r2(m, "ANN", "直接", 4),
         r2(m, "ANN", "直接", 5), rmse(m, "ANN", "直接", 5),
         r2(m, "ANN", "直接", 6), r2(m, "ANN", "直接", 7)))

    rs.add_heading2(doc, "1.4 子任务 4：ANN 多输出策略 1–7 天精度")
    P("多输出策略一次训练一个回归器，输出层直接给出 7 个预见期的 log1p 径流，即 7 维输出，"
      "由 MLPRegressor 原生支持，不借助任何逐输出独立的包装器。"
      "7 个预见期共享同一隐层表示，参数规模与训练轮次均少于直接策略，"
      "各输出之间通过共享表示相互约束，相当于对长预见期的低信噪比目标施加了正则化。")
    T(PRECISION_HEADER, precision_rows(m, "ANN", "多输出"), "ANN 多输出策略 1–7 天测试精度", 1)
    NOTE("注：R² 与 NSE 在本项目中采用同一公式 1−SS_res/SS_tot，两列数值恒等，实为同一指标；"
         "RMSE 与 MAE 的单位为 cfs，样本数为测试段天数。")
    P("在修正后的数据上，多输出策略第 1 天的 R² 为 %.4f，高于直接策略的 %.4f（领先 %.4f），"
      "RMSE 由 %.2f cfs 降至 %.2f cfs，这是本版与早期数据口径之间最明显的一处结论变化："
      "短预见期不再是直接策略的专属优势区间。"
      "第 2 天至第 4 天直接策略重新领先，差距由 %.4f 扩大到 %.4f；"
      "第 5 天多输出策略再次反超（%.4f 对 %.4f），第 6 天小幅领先 %.4f，"
      "第 7 天则由直接策略以 %.4f 对 %.4f 领先。"
      "7 个预见期的平均 R² 两者几乎持平（%.4f 对 %.4f），说明多输出策略的收益主要来自第 1 天与第 5 天，"
      "而不再表现为长预见期的系统性优势。"
      % (r2(m, "ANN", "多输出", 1), r2(m, "ANN", "直接", 1),
         r2(m, "ANN", "多输出", 1) - r2(m, "ANN", "直接", 1),
         rmse(m, "ANN", "直接", 1), rmse(m, "ANN", "多输出", 1),
         r2(m, "ANN", "直接", 2) - r2(m, "ANN", "多输出", 2),
         r2(m, "ANN", "直接", 4) - r2(m, "ANN", "多输出", 4),
         r2(m, "ANN", "多输出", 5), r2(m, "ANN", "直接", 5),
         r2(m, "ANN", "多输出", 6) - r2(m, "ANN", "直接", 6),
         r2(m, "ANN", "直接", 7), r2(m, "ANN", "多输出", 7),
         means[("ANN", "多输出")], means[("ANN", "直接")]))

    # ---------------- 第二章 ----------------
    rs.add_heading1(doc, "二、基于随机森林的径流预测")
    P("随机森林以回归树为基学习器，通过自助重采样与特征随机化构造集成，"
      "对特征尺度不敏感，因此无需标准化即可直接使用同一组特征。"
      "本章沿用与第一章完全一致的数据划分、特征集合与评估口径，仅替换模型与超参数搜索空间，"
      "使两种模型之间的比较不受数据口径差异的干扰。")
    P("数据划分保持不变：训练段 %d 个样本（%s 至 %s），测试段 %d 个样本（%s 至 %s），"
      "特征 %d 个，训练在前、测试在后，测试集只在最终评估时使用一次。"
      "超参数选择同样采用 GridSearchCV 配合 TimeSeriesSplit(n_splits=3, gap=7)，且只在训练段内部进行。"
      % (n_train, t_lo, t_hi, n_test, e_lo, e_hi, n_feat))

    rs.add_heading2(doc, "2.1 子任务 5：RF 网格搜索确定超参数")
    P("随机森林的搜索空间为树的最大深度 max_depth ∈ {8, 16, None} 与叶节点最小样本数 "
      "min_samples_leaf ∈ {5, 10}，共 6 组组合；其余参数取 scikit-learn 的默认值。"
      "验证方式与第一章一致，为 GridSearchCV 配合 TimeSeriesSplit(n_splits=3, gap=7)，"
      "评分指标为对数尺度上的均方根误差；直接策略逐预见期选参，多输出策略对 7 维输出整体选参一次。")
    header, rows = params_table(rf_grid, ann=False)
    T(header, rows, "RF 网格搜索交叉验证结果（直接策略，预见期 1 天）", 2)
    mean_col = header.index("平均 RMSE")
    if sorted(r[mean_col] for r in rows) != sorted("%.4f" % (-float(v))
                                                   for v in rf_grid["mean_test_score"]):
        raise AssertionError("表 2-1 的平均 RMSE 列与网格搜索 CSV 不一致")
    best_pairs = "、".join("（%s, %d）" % (depth_text(r[rf_depth_col]), int(r[rf_leaf_col]))
                          for _, r in rf_best_rows.iterrows())
    best_leaf = sorted({int(r[rf_leaf_col]) for _, r in rf_best_rows.iterrows()})
    best_depths = {depth_text(r[rf_depth_col]) for _, r in rf_best_rows.iterrows()}
    tail = ("" if len(best_depths) == 1 else
            "深度不同而交叉验证误差完全相同，说明在叶节点规模取 %d 时深度已不构成生长约束，"
            "与不限深度的配置等价。" % best_leaf[0])
    P("表 2-1 中平均交叉验证 RMSE 最低的配置为 %s，均为 %.4f；%s"
      "叶节点最小样本数取 %d 的组合占据排名前列，说明叶节点规模才是控制单棵树方差的关键参数，"
      "与随机森林在小样本条件下的方差控制需求相符。"
      % (best_pairs, -rf_best_mean, tail, best_leaf[0]))
    header, rows = selected_rows(rf_items, ann=False)
    T(header, rows, "RF 各预见期与多输出策略的选参结果", 2)
    text = ("8 组选参结果如表 2-2 所示。叶节点最小样本数在%s取 5，其余 %d 组取 10；"
            "最大深度在%s取 8，其余 %d 组（%s）取 16。"
            "交叉验证 RMSE 由 1 天的 %.4f 上升至 7 天的 %.4f，"
            "单次选参耗时 %.1f 至 %.1f 秒，高于人工神经网络，原因是每折都要生长完整森林。"
            % (rf_leaf5_txt, n_rf_leaf10, rf_d8_txt, n_rf_d16, rf_d16_txt,
               rf_direct[0]["cv"], rf_direct[-1]["cv"], min(rf_secs), max(rf_secs)))
    for value, digits in ((rf_direct[0]["cv"], 4), (rf_direct[-1]["cv"], 4),
                          (min(rf_secs), 1), (max(rf_secs), 1)):
        require(text, value, digits, "2.1 选参段")
    P(text)

    rs.add_heading2(doc, "2.2 子任务 6：RF 直接策略 1–7 天精度")
    P("直接策略为每个预见期单独训练一片森林，7 个预见期共 7 个模型，"
      "目标为该预见期的 log1p 径流，预测值经 expm1 反变换后参与评估。"
      "随机森林的预测是叶节点样本均值的加权组合，其取值受训练样本范围约束，"
      "这一特性对洪峰幅度的估计影响明显。")
    T(PRECISION_HEADER, precision_rows(m, "RF", "直接"), "RF 直接策略 1–7 天测试精度", 2)
    NOTE("注：R² 与 NSE 在本项目中采用同一公式 1−SS_res/SS_tot，两列数值恒等，实为同一指标；"
         "RMSE 与 MAE 的单位为 cfs，样本数为测试段天数。")
    P("第 1 天的 R² 为 %.4f，RMSE 为 %.2f cfs，MAE 为 %.2f cfs；"
      "第 2 天至第 7 天的 R² 依次为 %.4f、%.4f、%.4f、%.4f、%.4f、%.4f，7 个预见期全部为正，"
      "全年平均 R² 为 %.4f。与 ANN 直接策略相比，RF 仅在预见到第 1 天时落后（%.4f 对 %.4f），"
      "自第 2 天起即全面反超，衰减过程明显更平缓，第 7 天仍有 %.4f，而 ANN 直接策略已降至 %.4f。"
      % (r2(m, "RF", "直接", 1), rmse(m, "RF", "直接", 1), mae(m, "RF", "直接", 1),
         r2(m, "RF", "直接", 2), r2(m, "RF", "直接", 3), r2(m, "RF", "直接", 4),
         r2(m, "RF", "直接", 5), r2(m, "RF", "直接", 6), r2(m, "RF", "直接", 7),
         means[("RF", "直接")],
         r2(m, "RF", "直接", 1), r2(m, "ANN", "直接", 1),
         r2(m, "RF", "直接", 7), r2(m, "ANN", "直接", 7)))

    rs.add_heading2(doc, "2.3 子任务 7：RF 多输出策略 1–7 天精度")
    P("随机森林原生支持多输出回归：每棵树的叶节点同时保存 7 个预见期的样本均值，"
      "森林的输出即为各树叶节点向量预测的平均，因此多输出建模不需要额外封装，"
      "也不改变单棵树的训练方式。多输出策略只训练一次，参数规模与耗时约为直接策略的七分之一。")
    T(PRECISION_HEADER, precision_rows(m, "RF", "多输出"), "RF 多输出策略 1–7 天测试精度", 2)
    NOTE("注：R² 与 NSE 在本项目中采用同一公式 1−SS_res/SS_tot，两列数值恒等，实为同一指标；"
         "RMSE 与 MAE 的单位为 cfs，样本数为测试段天数。")
    P("多输出策略第 1 天的 R² 为 %.4f，明显低于直接策略的 %.4f（落后 %.4f），第 2 天仍然落后 %.4f；"
      "但自第 3 天起持续反超，第 3 天至第 7 天的领先幅度依次为 %.4f、%.4f、%.4f、%.4f、%.4f，"
      "其中第 7 天为四组方案中的最高值。"
      "该策略的全年平均 R² 为 %.4f，与直接策略的 %.4f 基本相当，"
      "说明随机森林的多输出策略以短预见期的精度损失换取长预见期的稳健性。"
      % (r2(m, "RF", "多输出", 1), r2(m, "RF", "直接", 1),
         r2(m, "RF", "直接", 1) - r2(m, "RF", "多输出", 1),
         r2(m, "RF", "直接", 2) - r2(m, "RF", "多输出", 2),
         r2(m, "RF", "多输出", 3) - r2(m, "RF", "直接", 3),
         r2(m, "RF", "多输出", 4) - r2(m, "RF", "直接", 4),
         r2(m, "RF", "多输出", 5) - r2(m, "RF", "直接", 5),
         r2(m, "RF", "多输出", 6) - r2(m, "RF", "直接", 6),
         r2(m, "RF", "多输出", 7) - r2(m, "RF", "直接", 7),
         means[("RF", "多输出")], means[("RF", "直接")]))

    # ---------------- 第三章 ----------------
    rs.add_heading1(doc, "三、结果分析与讨论")
    rs.add_heading2(doc, "3.1 两种模型与两种策略的横向比较")
    P("为在同一口径下比较各方案，表 3-1 汇总了四组方案在第 1 天与第 7 天的 R² 与 RMSE。"
      "四组方案的测试样本均为 %d 个，特征与数据划分完全相同，差异仅来自模型类型与建模策略。" % n_test)
    rows = []
    for model, strategy in GROUPS:
        rows.append(["%s %s策略" % (model, strategy),
                     "%.4f" % r2(m, model, strategy, 1),
                     "%.2f" % rmse(m, model, strategy, 1),
                     "%.4f" % r2(m, model, strategy, 7),
                     "%.2f" % rmse(m, model, strategy, 7),
                     "%.4f" % means[(model, strategy)]])
    T(["方案", "第1天 R²", "第1天 RMSE", "第7天 R²", "第7天 RMSE", "平均 R²"],
      rows, "四组方案在第 1 天与第 7 天以及全期平均的精度对比", 3)
    NOTE("注：RMSE 的单位为 cfs；平均 R² 为 7 个预见期 R² 的算术平均。")
    P("从第 1 天看，ANN 多输出策略以 R² %.4f、RMSE %.2f cfs 居首，"
      "ANN 直接策略以 %.4f 位居第二，两者相差 %.4f；RF 直接策略 %.4f 排列第三，RF 多输出策略 %.4f 垫底。"
      "这一排序与前两个阶段的数据口径下的结论不同：修正数据后，人工神经网络的多输出策略在最短预见期上反超了逐期独立训练的直接策略，"
      "而随机森林仍然是直接策略占优。两种模型因此呈现出不同的策略偏好，不能套用同一条结论。"
      % (r2(m, "ANN", "多输出", 1), rmse(m, "ANN", "多输出", 1),
         r2(m, "ANN", "直接", 1),
         r2(m, "ANN", "多输出", 1) - r2(m, "ANN", "直接", 1),
         r2(m, "RF", "直接", 1), r2(m, "RF", "多输出", 1)))
    P("从第 7 天看，排序完全改变：RF 多输出策略以 %.4f 居首，RF 直接策略 %.4f 次之，"
      "ANN 直接策略 %.4f 与 ANN 多输出策略 %.4f 均为负值，已落在零基准线以下。"
      "按 7 个预见期的平均 R² 计，RF 直接策略 %.4f 最高，RF 多输出策略 %.4f 紧随其后，"
      "ANN 多输出策略 %.4f 与 ANN 直接策略 %.4f 基本并列末位。"
      "整体规律是：随机森林的集成平均在全部预见期上都更稳定，ANN 的表示能力集中在最短预见期，"
      "一旦目标日的气象过程离开起报日的信息窗口，两类模型的差异就迅速缩小。"
      % (r2(m, "RF", "多输出", 7), r2(m, "RF", "直接", 7),
         r2(m, "ANN", "直接", 7), r2(m, "ANN", "多输出", 7),
         means[("RF", "直接")], means[("RF", "多输出")],
         means[("ANN", "多输出")], means[("ANN", "直接")]))
    P("逐预见期比较更能说明两种模型的分歧。ANN 的多输出策略在%s高于直接策略，"
      "在%s低于直接策略；RF 的多输出策略在%s高于直接策略，"
      "在%s低于直接策略。同一套策略组合在两种模型上给出了相反的胜负分布，"
      "说明策略的收益取决于模型类型：神经网络的多输出共享表示在样本有限时起到了正则作用，"
      "而随机森林的多输出只是把同一批叶节点均值打包输出，短预见期上反而稀释了单目标的拟合精度。"
      % (days_text(ann_win), days_text(ann_lose), days_text(rf_win), days_text(rf_lose)))

    FIG(FIG1, "四组方案 R² 随预见期的衰减曲线", 3,
        "图注：横轴为预见期天数，纵轴为决定系数 R²，四条曲线分别对应" 
        "ANN 直接、ANN 多输出、RF 直接与 RF 多输出，点线为零基准线。")
    FIG(FIG2, "第 1 天与第 7 天各方案的 R² 对比", 3,
        "图注：柱高为测试段（%d 个样本）上的决定系数 R²，柱顶数值保留四位小数，"
        "水平实线为零基准线。" % n_test)
    P("图 3-1 与图 3-2 分别以曲线和分组柱状图给出同一组结果。第 1 天四组方案的柱高分布在 %.2f 至 %.2f 之间，"
      "第 7 天则收缩到 %.2f 至 %.2f 的区间，且其中两根柱体已落到零线以下。"
      "可见短预见期上各方案的分差虽小但次序稳定，而在长预见期，R² 是否为正才是更关键的判断。"
      % (min(r2(m, mo, st, 1) for mo, st in GROUPS), max(r2(m, mo, st, 1) for mo, st in GROUPS),
         min(r2(m, mo, st, 7) for mo, st in GROUPS), max(r2(m, mo, st, 7) for mo, st in GROUPS)))

    rs.add_heading2(doc, "3.2 数据口径修正与效果对比")
    P("本版结果基于修正后的数据文件 01047000_fixed.csv 重新训练与评估，修正集中在三处。"
      "其一是气象列的对齐方式：原实现把 Daymet 气象变量与水文序列按行号直接拼接，整列错位一天，"
      "本版改为以年份（year）与年积日（yday）两个键与 Daymet v4 官方日值逐日对齐，并据此重建全部特征。"
      "其二是积雪水当量的恢复：原数据处理的积雪水当量在全部观测日上恒为零，积雪与融雪信息实际缺失；"
      "修正后恢复真实观测值，并派生融雪量、3 日与 7 日累积融雪、7 日滑动均值、正积温与雪盖指示共 8 个积雪特征，"
      "最终进入模型的特征为 %d 个。其三是缺测补齐：原始序列中有两日缺测，"
      "会使滚动窗口特征产生额外的空值行并连带删除样本，修正版以前后相邻观测补齐这两日。"
      % n_feat)
    P("三项修正共同改变的是起报时刻可获得的信息：气象驱动的日历位置被摆正，"
      "融雪过程第一次真正进入特征集合，样本也因缺测补齐而不再被额外裁掉。"
      "表 3-2 给出第 1 天与第 7 天四组方案在未修正数据版与修正后数据上的对照。")
    cmp_rows = []
    for model, strategy in GROUPS:
        for h in (1, 7):
            o_r2, n_r2 = r2(raw, model, strategy, h), r2(m, model, strategy, h)
            o_rmse, n_rmse = rmse(raw, model, strategy, h), rmse(m, model, strategy, h)
            cmp_rows.append(["%s %s" % (model, strategy), "%d" % h,
                             "%.3f" % o_r2, "%.3f" % n_r2, "%+.3f" % (n_r2 - o_r2),
                             "%.2f" % o_rmse, "%.2f" % n_rmse, "%+.2f" % (n_rmse - o_rmse)])
    T(["模型与策略", "预见期（天）", "未修正 R²", "修正后 R²", "ΔR²",
       "未修正 RMSE", "修正后 RMSE", "ΔRMSE"],
      cmp_rows, "未修正数据版与修正后数据的精度对照（第 1 天与第 7 天）", 3)
    NOTE("注：未修正数据版的数值取自归档结果目录（archive/results_phase2_v5_raw），"
         "修正后数值取自本版结果目录；RMSE 的单位为 cfs，Δ 为修正后减去未修正；"
         "为便于跨版本对照，本表的 R² 统一保留三位小数，四位小数的精度见正文各精度表。")
    P("第 1 天四组方案的精度全部上升：ANN 直接策略的 R² 由 %.3f 升至 %.3f（提高 %.3f），"
      "RMSE 由 %.2f cfs 降至 %.2f cfs；ANN 多输出策略由 %.3f 升至 %.3f（提高 %.3f）；"
      "RF 直接策略由 %.3f 升至 %.3f（提高 %.3f）；RF 多输出策略由 %.3f 升至 %.3f（提高 %.3f）。"
      "其中 ANN 多输出策略的升幅最大，也正是这一升幅改变了第 1 天两种策略的相对次序。"
      % (r2(raw, "ANN", "直接", 1), r2(m, "ANN", "直接", 1),
         r2(m, "ANN", "直接", 1) - r2(raw, "ANN", "直接", 1),
         rmse(raw, "ANN", "直接", 1), rmse(m, "ANN", "直接", 1),
         r2(raw, "ANN", "多输出", 1), r2(m, "ANN", "多输出", 1),
         r2(m, "ANN", "多输出", 1) - r2(raw, "ANN", "多输出", 1),
         r2(raw, "RF", "直接", 1), r2(m, "RF", "直接", 1),
         r2(m, "RF", "直接", 1) - r2(raw, "RF", "直接", 1),
         r2(raw, "RF", "多输出", 1), r2(m, "RF", "多输出", 1),
         r2(m, "RF", "多输出", 1) - r2(raw, "RF", "多输出", 1)))
    P("第 7 天的变化方向并不一致：RF 多输出策略的 R² 由 %.3f 微升至 %.3f，"
      "是四组方案中唯一改善的一组；ANN 直接策略由 %.3f 降至 %.3f，RF 直接策略由 %.3f 降至 %.3f，"
      "ANN 多输出策略由 %.3f 降至 %.3f，降幅最大。"
      "7 个预见期的平均 R² 也呈现同样格局：ANN 直接策略 %.3f 对 %.3f、ANN 多输出策略 %.3f 对 %.3f、"
      "RF 直接策略 %.3f 对 %.3f、RF 多输出策略 %.3f 对 %.3f。"
      "这说明气象列对齐与积雪特征主要改善了起报时刻附近的信息可用性，对长预见期的增益有限甚至略有代价，"
      "长预见期的误差仍主要由目标时段未观测的气象过程决定。"
      % (r2(raw, "RF", "多输出", 7), r2(m, "RF", "多输出", 7),
         r2(raw, "ANN", "直接", 7), r2(m, "ANN", "直接", 7),
         r2(raw, "RF", "直接", 7), r2(m, "RF", "直接", 7),
         r2(raw, "ANN", "多输出", 7), r2(m, "ANN", "多输出", 7),
         raw_means[("ANN", "直接")], means[("ANN", "直接")],
         raw_means[("ANN", "多输出")], means[("ANN", "多输出")],
         raw_means[("RF", "直接")], means[("RF", "直接")],
         raw_means[("RF", "多输出")], means[("RF", "多输出")]))

    rs.add_heading2(doc, "3.3 测试段水文过程线")
    P("图 3-3 给出 2004 年整个测试段（%s 至 %s）实测径流与两种模型第 1 天预测的过程线。" % (e_lo, e_hi))
    FIG(FIG3, "2004 年测试段实测与第 1 天预测径流过程线", 3,
        "图注：黑实线为实测径流，两条虚线分别为 ANN 与 RF 直接策略的第 1 天预测，"
        "数据取自测试段逐日预测结果。")
    P("全年最大洪峰出现在 %d 月 %d 日，实测 %.0f cfs，ANN 与 RF 的第 1 天预测分别为 %.0f cfs 与 %.0f cfs，"
      "两者对量级最大的洪峰均显著低估，预测值分别为实测的 %.0f%% 与 %.0f%%；"
      "而 3 月至 6 月的融雪期最大洪峰（%d 月 %d 日，实测 %.0f cfs）处，ANN 与 RF 的预测分别为 %.0f cfs 与 %.0f cfs，"
      "ANN 反而出现过冲。基流段与退水段的过程线与实测基本吻合，"
      "第 1 天整体 MAE 为 %.2f cfs，远小于 RMSE %.2f cfs，说明误差集中在少数洪峰日。"
      "这一现象说明两类模型给出的都是条件期望意义上的平滑估计："
      "训练段中出现过的量级被较好地复现，超出训练覆盖范围的极值则被系统性压缩，"
      "误差因此集中于少数峰值日。"
      % (peak_date.month, peak_date.day, peak_obs, peak_ann, peak_rf,
         100.0 * peak_ann / peak_obs, 100.0 * peak_rf / peak_obs,
         s_date.month, s_date.day, s_obs, s_ann, s_rf,
         mae(m, "ANN", "直接", 1), rmse(m, "ANN", "直接", 1)))

    rs.add_heading2(doc, "3.4 误差随预见期衰减的原因分析")
    P("从表 1-3、表 1-4、表 2-3、表 2-4 可见，四组方案的精度总体随预见期下降，衰减幅度与模型类型相关。"
      "其原因可从以下几个方面分析。")
    P("第一，输入信息在起报时刻固定。模型使用的 %d 个特征只包含起报日及更早的观测，"
      "预见期越长，起报与目标日之间新增的降水、气温与融雪过程就越长，这部分信息在预测时完全未知，"
      "可解释方差随之下降。这是精度衰减的根本原因，也是任何纯数据驱动方法共有的约束。" % n_feat)
    P("第二，条件期望的平滑效应。模型在 log 尺度上最小化均方误差，反变换后给出的是条件均值而非条件众数，"
      "对洪峰这类低频高幅事件系统性压缩。预见期越长，目标的条件分布越分散，"
      "均值收缩越强，峰值误差在 RMSE 中的权重随之上升，表现为 R² 快速下降而 MAE 上升较慢。")
    P("第三，样本量与事件覆盖不足。训练段仅 %d 个样本，覆盖约 4 个水文年，"
      "其中高流量事件数量有限，模型难以学到极端值的响应幅度。"
      "直接策略为每个预见期单独训练，长预见期目标信噪比更低，单折验证的偶然性也更大，"
      "ANN 直接策略在第 5 天出现的 R² 异常低值即属此类波动。" % n_train)
    P("第四，策略差异带来的正则效应。多输出策略让 7 个预见期共享同一表示，"
      "短预见期的高信噪比目标在联合训练中起到了约束作用，抑制了长预见期上的过拟合。"
      "本版数据上这种约束对人工神经网络有效：多输出在第 1 天即取得最好的短期精度，并在第 5、第 6 天形成第二轮反超；"
      "但对随机森林，多输出的收益只出现在第 3 天及以后，短预见期仍以直接策略更准。"
      "随机森林的集成平均本身即具有方差削减作用，这也是其衰减曲线比神经网络更平缓的原因。")

    # ---------------- 第四章 ----------------
    rs.add_heading1(doc, "四、阶段小结")
    P("本阶段完成了第二次阶段验收要求的 7 个子任务：梳理了人工神经网络、随机森林、支持向量回归、"
      "梯度提升树与循环神经网络五类适用于径流预测的机器学习方法；"
      "在训练段内部以 GridSearchCV 配合 TimeSeriesSplit(3, gap=7) 完成 ANN 与 RF 的超参数搜索，"
      "分别得到 8 组选参结果，合计 16 组；按直接策略与多输出策略分别训练并评估了两种模型在 1 至 7 天预见期上的精度，"
      "共形成 28 组测试精度指标。全部实验基于按 Daymet v4 官方日值对齐、恢复真实积雪水当量并补齐缺测两日后的修正数据集，"
      "训练段 %d 个样本、测试段 %d 个样本、%d 个特征。" % (n_train, n_test, n_feat))
    P("主要结论如下：第一，短预见期上人工神经网络的多输出策略精度最高，第 1 天 R² 为 %.4f、RMSE 为 %.2f cfs，"
      "高于同模型的直接策略（%.4f），修正数据后两种策略的次序在 ANN 上发生了反转；"
      "第二，随机森林的结论未变，第 1 天与第 2 天仍是直接策略占优（%.4f 对 %.4f），"
      "第 3 天起改由多输出策略领先；第三，长预见期上随机森林全面优于神经网络，"
      "第 7 天 RF 多输出策略 R² 为 %.4f，为四组方案中的最高值，其平均 R² %.4f 也仅次于 RF 直接策略的 %.4f；"
      "第四，四组方案第 1 天的 R² 分布在 %.2f 至 %.2f 之间，第 7 天收缩到 %.2f 至 %.2f 之间，"
      "预见期仍是决定预测可用性的首要因素。"
      % (r2(m, "ANN", "多输出", 1), rmse(m, "ANN", "多输出", 1), r2(m, "ANN", "直接", 1),
         r2(m, "RF", "直接", 1), r2(m, "RF", "多输出", 1),
         r2(m, "RF", "多输出", 7), means[("RF", "多输出")], means[("RF", "直接")],
         min(r2(m, mo, st, 1) for mo, st in GROUPS), max(r2(m, mo, st, 1) for mo, st in GROUPS),
         min(r2(m, mo, st, 7) for mo, st in GROUPS), max(r2(m, mo, st, 7) for mo, st in GROUPS)))
    P("本阶段工作仍存在局限：训练样本仅覆盖约 4 个水文年，高流量事件样本有限；"
      "所有方案对量级最大的洪峰仍存在系统性压缩，第 1 天预测的最大峰值仅为实测的 %.0f%% 与 %.0f%%；"
      "长预见期的 R² 已接近零甚至为负，尚不足以支撑实际调度决策；"
      "积雪特征虽已恢复，但 %d 个特征中与融雪直接相关的仅 8 个，"
      "融雪过程在模型中的贡献尚未做定量归因。"
      % (100.0 * peak_ann / peak_obs, 100.0 * peak_rf / peak_obs, n_feat))
    P("后续工作将从三方面展开：一是引入能够显式刻画时序依赖的深度序列模型，"
      "在相同数据口径下与本文方案对比；二是扩充样本年份与极端事件样本，"
      "并尝试针对峰值的加权损失或分位数回归以缓解幅度压缩；"
      "三是开展多模型融合，检验融合能否在长预见期上稳定提升精度。")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    primary = OUT_DIR / DOC_NAME
    mirror = ROOT / DOC_NAME
    doc.save(str(primary))
    shutil.copyfile(primary, mirror)

    return [primary, mirror, FIG1, FIG2, FIG3]


if __name__ == "__main__":
    paths = build()
    for p in paths:
        print("%s\t%d" % (p, p.stat().st_size))
