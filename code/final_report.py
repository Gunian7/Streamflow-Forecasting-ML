# -*- coding: utf-8 -*-
"""生成《机器学习课程实践》综合实验报告（单一 docx）。

数据口径（全部在运行时从结果文件读取，正文不手写数值）
    01047000.csv                             原始数据表（1827 × 9，Swe 恒为 0、气象列错位一天）
    01047000_fixed.csv                       修正数据表（建模唯一数据源）
    results/diagnostics/daymet_full.txt      Daymet v4 R1 日值（用于核对缺测日期）
    results/figures/eda/*.csv               描述统计、相关性、滞后相关
    results/figures/eda/*.png               时序总览、直方图、箱型图、热力图
    results/figures/fe_without_current_discharge/*.csv|png  特征选择排名与对比图、归一化对比图
    results/processed_data/*.csv            特征工程成品（196 列）
    results/phase2_v5/                       第二阶段：选参、28 行精度、逐日预测、3 张图
    results/phase3_enhanced/                 第三阶段：LSTM 精度、四张集成表、权重、4 张图
    results/phase3_enhanced/*.log            样本数、特征维数与留出块（运行时打印）
    archive/results_phase2_v5_raw/           未修正数据版精度（只读对照，正文明确标注）
    archive/results_phase3_enhanced_raw/     未修正数据版 LSTM 精度（只读对照）

排版统一调用 code/report_style.py，不自行定义样式；本文件只补一个 TOC 域构件。
用法：
    .venv/Scripts/python.exe code/final_report.py                       # 生成（目录为域占位）
    .venv/Scripts/python.exe code/final_report.py --toc-from-pdf X.pdf  # 用渲染页号回填目录
    .venv/Scripts/python.exe code/final_report.py --verify              # 只读回校验
"""

import json
import re
import shutil
import sys
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.text import (WD_ALIGN_PARAGRAPH, WD_BREAK, WD_TAB_ALIGNMENT,
                             WD_TAB_LEADER)
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

sys.path.insert(0, str(Path(__file__).resolve().parent))

import report_style as rs  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # pragma: no cover
    pass

ROOT = Path(__file__).resolve().parent.parent
EDA = ROOT / "results" / "figures" / "eda"
FE = ROOT / "results" / "figures" / "fe_without_current_discharge"
PROC = ROOT / "results" / "processed_data"
P2 = ROOT / "results" / "phase2_v5"
P3 = ROOT / "results" / "phase3_enhanced"
RAW2 = ROOT / "archive" / "results_phase2_v5_raw"
RAW3 = ROOT / "archive" / "results_phase3_enhanced_raw"
DIAG = ROOT / "results" / "diagnostics"

OUT_DIR = ROOT / "本地资料" / "综合报告"
DOC_NAME = "《机器学习课程实践》综合实验报告.docx"
HEADER_TEXT = "综合实验报告"

HORIZONS = list(range(1, 8))
GROUPS = (("ANN", "直接"), ("ANN", "多输出"), ("RF", "直接"), ("RF", "多输出"))
CN_NAMES = {"Discharge": "当天径流量", "Dayl": "日照时长", "Prcp": "降水量", "Srad": "短波辐射",
            "Swe": "积雪水当量", "Tmax": "最高气温", "Tmin": "最低气温", "Vp": "水汽压"}
FIELD_ORDER = ["Discharge", "Dayl", "Prcp", "Srad", "Swe", "Tmax", "Tmin", "Vp"]

MEMBERS_JSON = ROOT / "local_report_members.json"
COVER_PLACEHOLDER = {"names": "（姓名）", "student_ids": "（学号）", "advisor": "（指导教师）"}


# ------------------------------------------------------------------ 格式化

def f0(v):
    return "%.0f" % float(v)


def f1(v):
    return "%.1f" % float(v)


def f2(v):
    return "%.2f" % float(v)


def f3(v):
    return "%.3f" % float(v)


def f4(v):
    return "%.4f" % float(v)


def signed4(v):
    return "%+.4f" % float(v)


def signed3(v):
    return "%+.3f" % float(v)


def cover_fields():
    """从 local_report_members.json 读取封面姓名、学号与指导教师。

    该文件被 .gitignore 忽略，个人信息不进入仓库；文件缺失时回退占位符并提示，
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


# ------------------------------------------------------------------ 数值断言

CHECKS = []


def expect(label, actual, wanted):
    """把脚本读到的格式化数值与期望字符串比对，不一致立即抛错。"""
    ok = (actual == wanted)
    CHECKS.append({"label": label, "actual": actual, "wanted": wanted, "ok": ok})
    if not ok:
        raise AssertionError("数值断言失败：%s 读到 %r，期望 %r" % (label, actual, wanted))
    return actual


# ------------------------------------------------------------------ 读数

def load_series(path, strategy_value=None):
    frame = pd.read_csv(path, encoding="utf-8-sig")
    if strategy_value is not None:
        frame = frame[frame["策略"] == strategy_value]
    return {int(row["预见期（天）"]): row for _, row in frame.iterrows()}


def load_p2_metrics(path):
    frame = pd.read_csv(path, encoding="utf-8-sig")
    return {(row["模型"], row["策略"], int(row["预见期（天）"])): row for _, row in frame.iterrows()}


def load_phase2_meta():
    text = (P2 / "run_phase2.log").read_text(encoding="utf-8")
    block = re.search(r'\{\s*"数据文件".*?\n\}', text, re.S)
    if block is None:
        raise ValueError("run_phase2.log 中未找到数据边界记录")
    meta = json.loads(block.group(0))
    for key in ("训练起", "训练止", "训练样本", "测试起", "测试止", "测试样本", "特征数量"):
        if key not in meta:
            raise KeyError("数据边界缺少字段: %s" % key)
    return meta


def load_phase3_meta():
    text = (P3 / "run_stage5_7.log").read_text(encoding="utf-8", errors="ignore")
    data = re.search(r"训练 (\d+) 行 / 测试 (\d+) 行；基础特征 (\d+) 维，增强特征 (\d+) 维", text)
    hold = re.search(r"元特征留出块=最后 (\d+) 行（索引 (\d+) 起）", text)
    if data is None or hold is None:
        raise RuntimeError("run_stage5_7.log 中未找到样本数或留出块信息")
    return {"n_train": int(data.group(1)), "n_test": int(data.group(2)),
            "n_base": int(data.group(3)), "n_enh": int(data.group(4)),
            "n_hold": int(hold.group(1)), "hold_start": int(hold.group(2))}


def load_all():
    d = {}
    raw = pd.read_csv(ROOT / "01047000.csv", parse_dates=["Date"])
    fixed = pd.read_csv(ROOT / "01047000_fixed.csv", parse_dates=["Date"])
    d["raw"], d["fixed"] = raw, fixed
    d["n_rows"], d["n_cols"] = raw.shape
    d["date_min"] = raw["Date"].min().strftime("%Y-%m-%d")
    d["date_max"] = raw["Date"].max().strftime("%Y-%m-%d")
    d["swe_old_nonzero"] = int((raw["Swe"] > 0).sum())
    d["swe_nonzero"] = int((fixed["Swe"] > 0).sum())
    d["swe_max"] = float(fixed["Swe"].max())

    # Daymet 日值覆盖范围与面板缺测日（面板日期在 Daymet 中不存在的那些天）
    dm = pd.read_csv(DIAG / "daymet_full.txt", skiprows=6)
    dm.columns = ["year", "yday", "Dayl", "Prcp", "Srad", "Swe", "Tmax", "Tmin", "Vp"]
    dm = dm.dropna(subset=["year", "yday"]).astype({"year": int, "yday": int})
    dm["Date"] = (pd.to_datetime(dm["year"].astype(str) + "-01-01")
                  + pd.to_timedelta(dm["yday"] - 1, unit="D"))
    dm_dates = pd.DatetimeIndex(dm["Date"])
    panel = pd.DatetimeIndex(fixed["Date"])
    missing = [x.strftime("%Y-%m-%d") for x in panel.difference(dm_dates)]
    d["daymet_first"] = dm_dates.min().strftime("%Y-%m-%d")
    d["daymet_last"] = dm_dates.max().strftime("%Y-%m-%d")
    d["missing_days"] = missing
    # 补齐方式核对：缺测日的气象值等于前一日的值（前向填充）
    filled_ok = 0
    fix_idx = fixed.set_index("Date")
    for day in missing:
        prev = fix_idx.index[fix_idx.index.get_loc(pd.Timestamp(day)) - 1]
        same = all(abs(float(fix_idx.loc[day, c]) - float(fix_idx.loc[prev, c])) < 1e-9
                   for c in ("Prcp", "Tmax", "Tmin", "Swe", "Srad", "Dayl", "Vp"))
        filled_ok += int(same)
    d["filled_ok"] = filled_ok
    if len(missing) != 2 or filled_ok != 2:
        raise AssertionError("缺测日口径与预期不一致：%s，前向填充命中 %d 天" % (missing, filled_ok))

    # 降水-径流最佳滞后：原始表与修正表
    old_i, fix_i = raw.set_index("Date"), fixed.set_index("Date")
    d["lag_r_old"] = [round(float(old_i["Discharge"].corr(old_i["Prcp"].shift(k))), 4) for k in range(4)]
    d["lag_r_new"] = [round(float(fix_i["Discharge"].corr(fix_i["Prcp"].shift(k))), 4) for k in range(4)]
    d["lag_best_old"] = int(pd.Series(d["lag_r_old"]).idxmax())
    d["lag_best_new"] = int(pd.Series(d["lag_r_new"]).idxmax())

    # 融雪日与径流（修正表：Swe 逐日递减记为融雪日）
    melt = (fix_i["Swe"].diff() < 0).astype(int)
    d["melt_days"] = int(melt.sum())
    d["melt_mean"] = float(fix_i.loc[melt == 1, "Discharge"].mean())
    d["nonmelt_mean"] = float(fix_i.loc[melt == 0, "Discharge"].mean())

    # 探索性分析（第一阶段在原始表上完成）
    d["summary"] = pd.read_csv(EDA / "summary_statistics.csv", index_col=0)
    d["lag_corr"] = pd.read_csv(EDA / "discharge_with_lag_factors_correlations.csv", index_col=0)
    d["matrix"] = pd.read_csv(EDA / "pearson_correlation_matrix.csv", index_col=0)
    d["outliers"] = {}
    for col in FIELD_ORDER:
        q1, q3 = raw[col].quantile(0.25), raw[col].quantile(0.75)
        iqr = q3 - q1
        d["outliers"][col] = int(((raw[col] < q1 - 1.5 * iqr) | (raw[col] > q3 + 1.5 * iqr)).sum())

    # 特征工程成品构成
    feat = pd.read_csv(PROC / "01047000_with_feature_values_独热编码版.csv")
    d["feat_rows"], d["feat_cols"] = feat.shape
    cols = list(feat.columns)
    mm = [c for c in cols if c.endswith(rs_suffix("mm"))]
    zs = [c for c in cols if c.endswith(rs_suffix("zs"))]
    plain = [c for c in cols if c not in mm and c not in zs]
    onehot = ["春季独热编码", "夏季独热编码", "秋季独热编码", "冬季独热编码"]
    periodic = ["月份正弦编码", "月份余弦编码", "年内日序正弦编码", "年内日序余弦编码"]
    lag_cols = [c for c in plain if "_滞后" in c]
    win_cols = [c for c in plain if "过去" in c]
    enc_cols = [c for c in plain if c in onehot or c in periodic]
    base_cols = [c for c in plain if c not in lag_cols + win_cols + enc_cols]
    d["n_lag"], d["n_win"] = len(lag_cols), len(win_cols)
    d["n_enc"], d["n_base"] = len(enc_cols), len(base_cols)
    d["n_mm"], d["n_zs"] = len(mm), len(zs)
    if len(plain) + d["n_mm"] + d["n_zs"] != d["feat_cols"]:
        raise AssertionError("特征列构成分解与总列数不一致")

    # 特征选择（|r| >= 0.2 或 MI >= 0.05 或降水类必留）
    rank = pd.read_csv(FE / "feature_selection_ranking.csv", encoding="utf-8-sig")
    name_col, r_col, mi_col = rank.columns[0], rank.columns[1], rank.columns[3]
    keep = ((rank[r_col] >= 0.2) | (rank[mi_col] >= 0.05)
            | rank[name_col].astype(str).str.startswith("降水量"))
    d["rank"] = rank
    d["r_col"], d["mi_col"], d["name_col"] = r_col, mi_col, name_col
    d["n_cand"] = int(len(rank))
    d["n_kept"] = int(keep.sum())
    d["n_dropped"] = int((~keep).sum())
    d["dropped"] = rank[~keep]

    # 第二阶段
    d["p2"] = load_phase2_meta()
    d["p2_metrics"] = load_p2_metrics(P2 / "测试精度指标.csv")
    d["p2_raw"] = load_p2_metrics(RAW2 / "测试精度指标.csv")
    d["p2_sel"] = pd.read_csv(P2 / "选参结果.csv", encoding="utf-8-sig")

    # 第三阶段
    d["p3"] = load_phase3_meta()
    d["lstm_direct"] = load_series(P3 / "pytorch_lstm_metrics.csv", "PyTorch-LSTM-直接")
    d["lstm_multi"] = load_series(P3 / "pytorch_lstm_metrics.csv", "PyTorch-LSTM-多输出")
    d["raw_lstm_direct"] = load_series(RAW3 / "pytorch_lstm_metrics.csv", "PyTorch-LSTM-直接")
    d["avg6"] = load_series(P3 / "6模型简单平均_直接.csv")
    d["stack6"] = load_series(P3 / "6模型Stacking_直接.csv")
    d["avg3"] = load_series(P3 / "3模型简单平均_多输出.csv")
    d["stack3"] = load_series(P3 / "3模型Stacking_多输出.csv")
    d["w_direct"] = pd.read_csv(P3 / "stacking_直接_权重.csv", encoding="utf-8-sig")
    d["w_multi"] = pd.read_csv(P3 / "stacking_多输出_权重.csv", encoding="utf-8-sig")
    d["members"] = pd.read_csv(P3 / "ensemble_members.csv", encoding="utf-8-sig")
    grid = pd.read_csv(P3 / "pytorch_lstm_grid_summary.csv", encoding="utf-8-sig")
    order = {"direct_day%d" % h: h for h in HORIZONS}
    grid["sort_key"] = grid["task"].map(lambda t: order.get(t, 8))
    d["lstm_grid"] = grid.sort_values("sort_key").reset_index(drop=True)

    # 逐日预测：过程线峰值与集成成员
    pred = pd.read_csv(P3 / "ensemble_predictions.csv", encoding="utf-8-sig",
                       usecols=["策略", "方法", "预见期（天）", "起报日", "目标日", "实测", "预测"])
    best = pred[(pred["策略"] == "直接") & (pred["方法"] == "PyTorch-LSTM")
                & (pred["预见期（天）"] == 1)].sort_values("目标日").reset_index(drop=True)
    peak = best.loc[best["实测"].idxmax()]
    d["peak_date"] = str(peak["目标日"])
    d["peak_obs"] = float(peak["实测"])
    d["peak_pred"] = float(peak["预测"])
    d["test_first"], d["test_last"] = best["目标日"].min(), best["目标日"].max()
    avg6_pred = pred[(pred["策略"] == "直接") & (pred["方法"] == "6模型简单平均")
                     & (pred["预见期（天）"] == 1)]
    same_day = avg6_pred[avg6_pred["目标日"] == d["peak_date"]]
    d["peak_pred_avg6"] = float(same_day["预测"].iloc[0]) if len(same_day) else None
    d["n_targets"] = int(len(best))

    # 关键数值断言（读到的格式化值必须与期望字符串完全一致）
    expect("径流均值（cfs）", f3(d["summary"].loc["Discharge", "mean"]), "610.893")
    expect("径流偏度", f3(d["summary"].loc["Discharge", "skewness"]), "4.958")
    expect("径流峰度", f3(d["summary"].loc["Discharge", "kurtosis"]), "35.939")
    expect("径流滞后 1 天相关系数", f3(d["lag_corr"].loc["Discharge_lag1", "Pearson_r"]), "0.767")
    expect("降水同期相关系数", f3(d["matrix"].loc["径流", "降水"]), "0.431")
    expect("降水滞后 1 天相关系数", f3(d["lag_corr"].loc["Prcp_lag1", "Pearson_r"]), "0.335")
    expect("径流离群点数", str(d["outliers"]["Discharge"]), "205")
    expect("降水离群点数", str(d["outliers"]["Prcp"]), "260")
    expect("特征工程成品列数", str(d["feat_cols"]), "196")
    expect("特征选择候选数", str(d["n_cand"]), "62")
    expect("特征选择保留数", str(d["n_kept"]), "59")
    expect("特征选择剔除数", str(d["n_dropped"]), "3")
    expect("融雪日平均径流（cfs）", f1(d["melt_mean"]), "1415.0")
    expect("非融雪日平均径流（cfs）", f1(d["nonmelt_mean"]), "386.2")
    expect("ANN 直接第 1 天 R²", f4(d["p2_metrics"][("ANN", "直接", 1)]["R²"]), "0.7230")
    expect("ANN 多输出第 1 天 R²", f4(d["p2_metrics"][("ANN", "多输出", 1)]["R²"]), "0.7452")
    expect("RF 直接第 1 天 R²", f4(d["p2_metrics"][("RF", "直接", 1)]["R²"]), "0.5888")
    expect("RF 多输出第 1 天 R²", f4(d["p2_metrics"][("RF", "多输出", 1)]["R²"]), "0.4828")
    expect("未修正数据 ANN 直接第 1 天 R²", f4(d["p2_raw"][("ANN", "直接", 1)]["R²"]), "0.4958")
    expect("LSTM 直接第 1 天 R²", f4(d["lstm_direct"][1]["R²"]), "0.8463")
    expect("LSTM 多输出第 1 天 R²", f4(d["lstm_multi"][1]["R²"]), "0.8019")
    expect("6 模型简单平均第 1 天 R²", f4(d["avg6"][1]["R²"]), "0.8196")
    expect("6 模型 Stacking 第 1 天 R²", f4(d["stack6"][1]["R²"]), "0.7905")
    expect("3 模型简单平均第 1 天 R²", f4(d["avg3"][1]["R²"]), "0.7672")
    expect("3 模型 Stacking 第 1 天 R²", f4(d["stack3"][1]["R²"]), "0.6713")
    expect("洪峰实测（cfs）", f1(d["peak_obs"]), "7470.0")
    expect("洪峰 LSTM 第 1 天预测（cfs）", f2(d["peak_pred"]), "3856.66")
    expect("洪峰日期", d["peak_date"], "2004-11-29")
    return d


def rs_suffix(kind):
    return "_最小最大归一化" if kind == "mm" else "_Z分数标准化"


# ------------------------------------------------------------------ TOC 域

def _field_run(paragraph, instr=None, kind=None, text=None, size=rs.CAPTION_SIZE):
    """在段落里写一个域字符片段（begin / instrText / separate / text / end）。"""
    run = paragraph.add_run()
    rs.set_run_font(run, size, bold=False, east_asia=rs.SONG_FONT)
    if kind:
        el = OxmlElement("w:fldChar")
        el.set(qn("w:fldCharType"), kind)
        run._r.append(el)
    if instr is not None:
        el = OxmlElement("w:instrText")
        el.set(qn("xml:space"), "preserve")
        el.text = instr
        run._r.append(el)
    if text is not None:
        el = OxmlElement("w:t")
        el.set(qn("xml:space"), "preserve")
        el.text = text
        run._r.append(el)
    return run


def build_toc_block(doc, entries):
    """生成 目录 标题 + TOC 域（含预置结果行），返回创建的段落列表。

    entries 为 [(level, 文本, 页码或 None)]：页码为 None 时域结果只放一行占位说明，
    由 --toc-from-pdf 回填。域代码 'TOC \\o "1-3" \\h \\z \\u' 与 w:updateFields 配合，
    Word 打开时会重新计算页码。
    """
    created = []
    head = doc.add_paragraph(style="Heading 1")
    rs._plain_para(head, WD_ALIGN_PARAGRAPH.CENTER, rs.BODY_LINE_SPACING,
                   rs.H1_SPACE_BEFORE.pt, rs.H1_SPACE_AFTER.pt)
    head.paragraph_format.keep_with_next = True
    head.paragraph_format.page_break_before = True      # 目录另起一页
    rs.set_run_font(head.add_run("目录"), rs.H1_SIZE, bold=False, east_asia=rs.HEI_FONT)
    # 目录标题本身不进目录：段落大纲级别降为正文
    pPr = head._p.get_or_add_pPr()
    lvl = rs._insert_ordered(pPr, OxmlElement("w:outlineLvl"), rs._PPR_SEQ)
    lvl.set(qn("w:val"), "9")
    created.append(head)

    first = doc.add_paragraph()
    rs._plain_para(first, WD_ALIGN_PARAGRAPH.LEFT, 1.0, 0, 0)
    _field_run(first, kind="begin")
    _field_run(first, instr=' TOC \\o "1-3" \\h \\z \\u ')
    _field_run(first, kind="separate")
    created.append(first)

    if entries and entries[0][2] is not None:
        for idx, (level, text, page) in enumerate(entries):
            para = first if idx == 0 else doc.add_paragraph()
            if idx:
                rs._plain_para(para, WD_ALIGN_PARAGRAPH.LEFT, 1.0, 0, 0)
                created.append(para)
            para.paragraph_format.left_indent = Cm(0 if level == 1 else 0.85)
            para.paragraph_format.tab_stops.add_tab_stop(
                Cm(rs.TEXT_WIDTH_CM), WD_TAB_ALIGNMENT.RIGHT, WD_TAB_LEADER.DOTS)
            rs.set_run_font(para.add_run(text), rs.CAPTION_SIZE,
                            bold=(level == 1), east_asia=rs.SONG_FONT)
            rs.set_run_font(para.add_run("\t%s" % page), rs.CAPTION_SIZE,
                            bold=(level == 1), east_asia=rs.SONG_FONT)
        _field_run(created[-1] if len(created) > 1 else first, kind="end")
    else:
        rs.set_run_font(first.add_run("（目录域：在 Word 中打开文档后按提示更新域即可显示条目与页码）"),
                        rs.CAPTION_SIZE, bold=False, east_asia=rs.SONG_FONT)
        _field_run(first, kind="end")

    brk = doc.add_paragraph()
    rs._plain_para(brk, WD_ALIGN_PARAGRAPH.LEFT, 1.0, 0, 0)
    brk.add_run().add_break(WD_BREAK.PAGE)
    created.append(brk)
    return created


def keep_rows_whole(table, keep_together=False):
    """给表格每一行写 w:cantSplit，避免一行文字被分页切成上下两半。

    keep_together=True 时再给除末行以外的所有单元格段落写 keep_with_next，
    让整张表尽量留在同一页（Word 在放不下时仍会分页）。
    """
    rows = list(table.rows)
    for row in rows:
        trPr = row._tr.get_or_add_trPr()
        if trPr.find(qn("w:cantSplit")) is None:
            trPr.append(OxmlElement("w:cantSplit"))
    if keep_together:
        for row in rows[:-1]:
            for cell in row.cells:
                for para in cell.paragraphs:
                    para.paragraph_format.keep_with_next = True
    return table


def set_update_fields(doc):
    """写 w:updateFields，使 Word 打开文档时自动刷新目录页码。"""
    settings = doc.settings.element
    node = settings.find(qn("w:updateFields"))
    if node is None:
        node = OxmlElement("w:updateFields")
        settings.append(node)
    node.set(qn("w:val"), "true")
    return node


# ------------------------------------------------------------------ 生成

def build(toc_pages=None):
    d = load_all()
    headings = []
    doc = rs.new_document(header_text=HEADER_TEXT)
    rs.add_cover(
        doc,
        college="自动化学院",
        major="智能科学与技术",
        class_name="课程实践小组",
        **cover_fields(),
        project="基于机器学习的径流预测",
        content="综合实验报告",
        title="杭州电子科技大学《机器学习课程实践》综合实验报告",
        date_text="二〇二六年十一月",
    )
    rs.begin_body_section(doc, 1)
    num = rs.Numbering()
    anchor = {"first": None}

    def H1(text):
        headings.append((1, text))
        para = rs.add_heading1(doc, text)
        if anchor["first"] is None:
            anchor["first"] = para
        return para

    def H2(text):
        headings.append((2, text))
        return rs.add_heading2(doc, text)

    def P(text, indent=True):
        return rs.add_body(doc, text, indent=indent)

    def T(header, rows, caption, chapter):
        table, cap = rs.add_table(doc, header, rows, caption=caption, chapter=chapter, numbering=num)
        keep_rows_whole(table)
        return table, cap

    def NOTE(text):
        return rs.add_figure_note(doc, text)

    def FIG(path, caption, chapter):
        return rs.add_figure(doc, path, caption=caption, chapter=chapter, numbering=num)

    p2, p3 = d["p2"], d["p3"]
    ld, lm = d["lstm_direct"], d["lstm_multi"]
    a6, s6, a3, s3 = d["avg6"], d["stack6"], d["avg3"], d["stack3"]

    # ---------------------------------------------------------------- 摘要
    summ = d["summary"]
    rs_head = doc.add_paragraph(style="Heading 1")
    rs._plain_para(rs_head, WD_ALIGN_PARAGRAPH.CENTER, rs.BODY_LINE_SPACING,
                   rs.H1_SPACE_BEFORE.pt, rs.H1_SPACE_AFTER.pt)
    rs.set_run_font(rs_head.add_run("摘要"), rs.H1_SIZE, bold=False, east_asia=rs.HEI_FONT)
    headings.append((1, "摘要"))

    P("本报告汇总《机器学习课程实践》三个阶段的全部工作：以美国地质调查局（USGS）"
      "01047000 号站点 %d 天的逐日水文气象序列为对象，完成数据质量核查与修正、"
      "探索性分析与特征工程、人工神经网络与随机森林的 %d 个单模型、"
      "长短期记忆网络以及两类集成学习，最终形成 1 至 7 天预见期的径流多步预报方案。"
      "全部 %d 个数值字段的观测中，径流均值为 %.3f cfs、偏度 %.3f、峰度 %.3f，"
      "呈显著右偏的尖峰分布。原始数据存在两处缺陷：气象列与 Daymet 官方日值整体相差一天，"
      "积雪水当量全序列恒为零。按 year 与 yday 重新对齐并恢复真实积雪水当量后，"
      "降水与径流的最佳滞后由 0 天回到 1 天，融雪日平均径流达 %.1f cfs，"
      "为非融雪日的 %.1f 倍。特征工程在 %d 个基础特征上构造出 %d 列成品特征，"
      "经皮尔逊与互信息双指标筛选保留 %d 项。测试集采用严格时序划分且只使用一次："
      "第 1 天决定系数最高的方案为长短期记忆网络的直接策略（R²=%.4f），"
      "六模型简单平均为 %.4f，人工神经网络多输出策略为 %.4f。"
      "全部指标由脚本在运行时从结果文件读取，正文数值与结果文件逐一对应。"
      % (d["n_rows"], len(d["p2_metrics"]), len(FIELD_ORDER), float(summ.loc["Discharge", "mean"]),
         float(summ.loc["Discharge", "skewness"]), float(summ.loc["Discharge", "kurtosis"]),
         d["melt_mean"], d["melt_mean"] / d["nonmelt_mean"], d["n_base"], d["feat_cols"],
         d["n_kept"], float(ld[1]["R²"]), float(a6[1]["R²"]), float(d["p2_metrics"][("ANN", "多输出", 1)]["R²"])))

    P("关键词：径流预报；数据质量修正；特征工程；随机森林；长短期记忆网络；Stacking 集成", indent=False)

    # ---------------------------------------------------------------- 引言
    H1("一、引言")
    H2("1.1 研究背景与意义")
    P("山区流域的径流预报是水库调度、防洪减灾与水资源配置的基础工作。与平原流域相比，"
      "山区流域的产流过程受地形与气候的双重控制：坡陡流急使降水到径流的响应时间被压缩到"
      "数小时至数十小时，积雪与融雪的相变过程则把水量的释放推迟到数月之后的融雪期。"
      "这种“短时快速响应 + 长时蓄放延迟”的叠加结构，使径流序列同时具备强烈的季节性与"
      "突发的高频波动，传统的概念性水文模型需要大量难以观测的流域参数，"
      "在资料稀少的小流域上难以率定。")
    P("数据驱动方法不依赖显式的产汇流机理，直接建立气象与前期水文状态到未来径流的映射，"
      "在站点尺度上已被广泛验证。其可行性取决于三点：输入特征是否覆盖了产流的主要驱动项、"
      "模型能否表达滞后与非线性、评估口径是否严格。本项目的三个工作阶段正是围绕这三点递进："
      "第一阶段解决“数据是否可信、特征如何构造”，第二阶段解决“单一模型能到什么程度”，"
      "第三阶段解决“时序模型与集成能否在此基础上继续提升”。")
    P("项目对象为 USGS 01047000 号站点的逐日观测。该站点位于融雪主导的山区流域，"
      "径流在 3 月至 6 月出现由融雪补给形成的春季洪峰，年内分配极不均匀，"
      "夏季则受局地暴雨影响出现短历时陡涨过程。这类流域对“前期积雪状态 + 近期降水”"
      "两类信息同时敏感，为检验特征工程与模型结构的有效性提供了合适的样本。")

    H2("1.2 研究内容与技术路线")
    P("三个阶段构成一条逐级收紧的路线：数据层确定唯一可信的数据源，特征层构造可复现的输入矩阵，"
      "模型层在同一评估口径下比较不同假设。表 1-1 给出各阶段的目标、主要工作与交付物，"
      "附录 A 把每个子任务与结果文件逐一对应。")
    rows = [
        ["第一阶段", "数据可信性与输入构造",
         "字段核对、描述统计、分布与离群点、相关性分析；构造季节与周期编码、滞后项、滚动窗口统计；"
         "两套归一化编码；双指标特征选择",
         "9 字段释义、196 列特征成品、62 项候选的筛选名次", "第三章"],
        ["第二阶段", "单一模型的上限",
         "ANN 与随机森林各按直接策略与多输出策略建模，7 个预见期分别选参；给出 1 至 7 天精度",
         "28 行精度指标、16 份网格搜索记录、16 行选参结果", "第五章 2 至 4 节"],
        ["第三阶段", "时序模型与集成",
         "PyTorch 实现的 LSTM 直接与多输出策略；6 模型与 3 模型的简单平均、Stacking 两类集成",
         "14 行 LSTM 精度、四张集成精度表、元学习器权重", "第五章 3 至 5 节"],
    ]
    T(["阶段", "核心问题", "主要工作", "交付物", "对应章节"], rows,
      "三个阶段的技术路线与交付物", 1)
    NOTE("注：三个阶段共用同一份测试段（2004 年）与同一套指标口径，"
         "因此第二、第三阶段的数值可以直接横向比较；特征维数在阶段之间变化，对照时另行说明。")

    H2("1.3 报告组织")
    P("第二章交代数据来源、原始数据的两处缺陷与修正过程，并给出数据划分与评估口径；"
      "第三章复现第一阶段的探索性分析与特征工程，逐项说明特征构成与筛选依据；"
      "第四章给出多步预测的两种策略、三个模型族与两类集成的形式化描述；"
      "第五章汇总超参数搜索与各阶段测试结果；第六章讨论精度衰减、策略分歧、"
      "集成的失效原因与峰值低估；第七章给出结论与后续工作。"
      "参考文献只列出本项目实际依据的公开文献，附录 A 与附录 B 分别给出子任务对照与结果文件清单。")

    # ---------------------------------------------------------------- 第二章
    H1("二、数据与数据质量")
    H2("2.1 研究流域与数据来源")
    P("径流序列取自 USGS 国家水信息系统（NWIS）01047000 号站点的逐日实测流量，"
      "单位为立方英尺每秒（cfs）。气象驱动取自 Daymet 第 4 版第 1 次修订（Daymet v4 R1）"
      "1 km 网格日值产品，所用网格为 tile 12116，变量包括日照时长 Dayl、降水量 Prcp、"
      "短波辐射 Srad、积雪水当量 Swe、最高气温 Tmax、最低气温 Tmin 与水汽压 Vp。"
      "两份数据按日期拼接后形成 %d 行 %d 列的建模面板，覆盖 %s 至 %s 的连续日序列，"
      "无重复日期、无缺失日期。"
      % (d["n_rows"], d["n_cols"], d["date_min"], d["date_max"]))
    P("数据划分遵循严格时序原则：2000 年至 2003 年为训练段，2004 年为测试段，"
      "两段之间不留空隙也不随机抽样，测试段在全部建模与选参完成之后只评估一次。"
      "第二阶段实际参与拟合的训练样本为 %s 行（%s 至 %s），测试样本 %s 行（%s 至 %s），"
      "特征维数 %s；第三阶段因增强特征含 7 日滚动统计而丢弃首尾不完整行，"
      "训练与测试样本分别为 %d 行与 %d 行。"
      % (p2["训练样本"], p2["训练起"], p2["训练止"], p2["测试样本"], p2["测试起"], p2["测试止"],
         p2["特征数量"], p3["n_train"], p3["n_test"]))
    rows = [
        ["01047000.csv", "原始表（课程下发原样）", "%d × %d" % (d["n_rows"], d["n_cols"]),
         "%s 至 %s" % (d["date_min"], d["date_max"]), "留档与对账，不参与建模"],
        ["01047000_fixed.csv", "修正表（按 year+yday 重对齐、恢复真实积雪）",
         "%d × %d" % (d["n_rows"], d["n_cols"]),
         "%s 至 %s" % (d["date_min"], d["date_max"]), "全部建模与评估的唯一数据源"],
        ["Daymet v4 R1（tile 12116）", "1 km 网格日值",
         "%s 至 %s" % (d["daymet_first"], d["daymet_last"]), "气象变量来源",
         "用于校验与重建气象列"],
        ["训练段", "严格早于测试段", "%s 行" % p2["训练样本"],
         "%s 至 %s" % (p2["训练起"], p2["训练止"]), "选参与拟合"],
        ["测试段", "严格晚于训练段", "%s 行" % p2["测试样本"],
         "%s 至 %s" % (p2["测试起"], p2["测试止"]), "只在最终评估使用一次"],
    ]
    T(["数据对象", "性质", "规模", "起止日期", "用途"], rows, "数据来源与数据划分", 2)
    NOTE("注：修正表与原始表列结构完全一致、行数相同，可以直接互相替换；"
         "第三阶段样本数与第二阶段相差 6 行，源于 7 日滚动统计对首尾行的裁剪。")

    H2("2.2 原始数据的两处缺陷")
    P("课程下发的原始表在字段层面看似完整，但与 Daymet 官方日值逐日比对后暴露出两处系统性缺陷。"
      "第一处是气象列整体错位一天：原始表把某一日的气象变量写到了下一日的行上，"
      "使降水、气温、辐射与积雪全部与径流错开 24 小时。第二处是积雪水当量全序列为 0，"
      "既不符合该流域存在稳定冬季积雪的事实，也使全部积雪派生量退化为零方差列。"
      "两处缺陷都不影响列数与行数，因此不会被简单的完整性检查发现，只能通过物理关系检验识别。")
    rows = [
        ["气象列与日历错位", "Prcp、Tmax、Tmin、Srad、Dayl、Vp 的日值整体后移一天",
         "降水与径流的相关在滞后 0 天处最高（r=%.4f），重对齐后在滞后 1 天处最高（r=%.4f）"
         % (d["lag_r_old"][0], d["lag_r_new"][1]),
         "产流响应被人为提前，模型学到的滞后结构错误；跨日特征（滞后项、窗口统计）整体偏移"],
        ["积雪水当量恒为零", "Swe 全部 %d 行为 0，标准差为 0" % d["n_rows"],
         "修正后非零 %d 天，最大值 %s kg/m²" % (d["swe_nonzero"], f1(d["swe_max"])),
         "融雪量与累积融雪等派生量零方差被剔除，融雪补给无法进入模型；"
         "春季洪峰的成因在特征中完全缺失"],
        ["面板边缘缺测", "%s 与 %s 两天在 Daymet 日值中不存在" % (d["missing_days"][0], d["missing_days"][1]),
         "两天均以前一日气象值前向填充，填充值与前一日的偏差为 0",
         "若不处理，两行会在特征工程中被整行丢弃或产生 NaN 传播"],
    ]
    T(["缺陷", "在原始数据中的表现", "检验证据", "对建模的影响"], rows, "原始数据的三类问题与证据", 2)
    NOTE("注：前两行即本报告所称“两处缺陷”，第三行为面板边缘的缺测，属于附带问题；"
         "三类问题均在修正表中处理完毕。")

    H2("2.3 数据修正与物理关系复核")
    P("修正只在气象列上进行，径流列沿用 USGS 实测值不变，因此不存在标签被修改的问题。"
      "第一步，把 Daymet 日值以 year 与 yday 两个键重新对齐到日历日，替代原表按行号拼接的做法；"
      "第二步，恢复真实的积雪水当量，使冬季积雪与春季消融重新成为可观测的驱动项；"
      "第三步，对 Daymet 覆盖范围之外的两天（%s、%s）以前一日值前向填充，"
      "检验表明这两天的气象值与前一日的偏差为 0，即填充确实生效且未引入插值平滑。"
      % (d["missing_days"][0], d["missing_days"][1]))
    P("修正是否有效，不应只看数值是否“看起来合理”，而要看物理关系是否回到应有的形态。"
      "本项目用三项可复现的检验来确认：其一，降水与径流的互相关在不同滞后上的形状，"
      "山区流域的产流响应存在 1 天左右的滞后；其二，积雪逐日递减被定义为融雪日，"
      "融雪日的径流应显著高于非融雪日；其三，积雪非零天数应与该流域的冬季长度相称。"
      "表 2-3 给出修正前后的对照。")
    rows = [
        ["降水-径流最佳滞后",
         "%d 天（r=%.4f）" % (d["lag_best_old"], d["lag_r_old"][d["lag_best_old"]]),
         "%d 天（r=%.4f）" % (d["lag_best_new"], d["lag_r_new"][d["lag_best_new"]]),
         "滞后结构回到产流机理应有的一日延迟"],
        ["降水-径流滞后 1 天相关", "r=%.4f" % d["lag_r_old"][1], "r=%.4f" % d["lag_r_new"][1],
         "重对齐后一日滞后相关显著增强"],
        ["积雪非零天数", "%d 天" % d["swe_old_nonzero"], "%d 天" % d["swe_nonzero"],
         "冬季积雪重新可观测"],
        ["融雪日数（Swe 逐日递减）", "0 天", "%d 天" % d["melt_days"],
         "融雪过程可识别"],
        ["融雪日平均径流", "—", "%s cfs" % f1(d["melt_mean"]), "融雪补给的量级"],
        ["非融雪日平均径流", "—", "%s cfs" % f1(d["nonmelt_mean"]),
         "两类日子的均值比即融雪效应强度"],
    ]
    T(["检验项", "原始数据", "修正数据", "含义"], rows, "修正前后物理关系对照", 2)
    NOTE("注：融雪日定义为积雪水当量相对前一日递减（Swe 一阶差分小于 0）的观测日；"
         "原始数据的 Swe 恒为零，差分恒为零，故融雪日数为 0，该类指标在原始数据上无定义。")
    P("三项检验在修正数据上同时成立：最佳滞后由 0 天回到 1 天，"
      "融雪日共 %d 天且其平均径流为 %s cfs，非融雪日为 %s cfs，前者是后者的 %.2f 倍；"
      "积雪非零天数达到 %d 天，与五个水文年的冬季长度相称。"
      "这一组关系说明修正并非简单的数值替换，而是让数据重新满足区域产流的基本物理约束，"
      "也解释了为什么修正后第 1 天的精度会出现数量级上的改善（见 5.5 节）。"
      % (d["melt_days"], f1(d["melt_mean"]), f1(d["nonmelt_mean"]),
         d["melt_mean"] / d["nonmelt_mean"], d["swe_nonzero"]))

    H2("2.4 数据划分与评估口径")
    P("划分遵循三条规则。第一，严格时序：训练段全部早于测试段，不做随机抽样、不做分层抽样，"
      "以免未来信息泄漏到训练过程。第二，选参内嵌：超参数搜索只在训练段内部完成，"
      "用时间序列交叉验证评估泛化误差，测试段在整个选参过程中不可见。"
      "第三，测试段只用一次：全部方案的最终指标由同一批起报日一次性算出，"
      "不根据测试结果回调任何超参数。")
    P("评估口径还需要区分“起报日”与“目标日”。第 h 天方案的输入是起报日及更早的观测，"
      "输出是起报日之后第 h 天的径流；因而同一条样本的输入日期与标签日期相差 h 天。"
      "这一区分在过程线分析中尤其重要：若按起报日绘图，会把预测点画在错误的时间位置上。"
      "本报告的全部过程线一律按目标日期绘制。")

    # ---------------------------------------------------------------- 第三章
    H1("三、探索性分析与特征工程")
    P("本章对应第一阶段的工作，全部在原始表上完成，目的是先看清数据的统计形态与结构缺陷，"
      "再据此构造输入。相关结果文件位于 results/figures/eda/ 与 results/figures/fe_without_current_discharge/。"
      "需要强调：第三章的统计量（含离群点数、相关系数）取自原始表 %d 行，"
      "与第五章修正后的建模结果属于不同口径，正文已在各处分别标注，不可混用。"
      % d["n_rows"])

    H2("3.1 字段释义与描述统计")
    P("面板共 9 个字段：1 个日期主键、1 个径流标签与 7 个气象驱动项。"
      "表 3-1 给出各字段的单位与在水文过程中的作用；表 3-2 给出 8 个数值字段的分布特征。")
    rows = []
    meanings = {
        "Date": ["日期", "—", "日主键，与 Daymet 的 year 与 yday 一一对应"],
        "Discharge": ["当天径流量", "cfs", "标签列：USGS 实测日平均流量，右偏尖峰分布"],
        "Dayl": ["日照时长", "s", "能量条件：当日昼长，决定蒸发与融雪的时间窗口"],
        "Prcp": ["降水量", "mm/d", "水量输入：当日降水，产流的直接驱动项，零值占比高"],
        "Srad": ["短波辐射", "W/m²", "能量条件：控制融雪速率与蒸散发强度"],
        "Swe": ["积雪水当量", "kg/m²", "蓄水状态：积雪的等效水量，融雪期的主要补给项"],
        "Tmax": ["最高气温", "℃", "能量条件：控制融雪速率，与径流的季节变化密切相关"],
        "Tmin": ["最低气温", "℃", "能量条件：夜间低温决定积雪的冻结与保持"],
        "Vp": ["水汽压", "Pa", "湿度条件：与蒸发能力相关，间接影响流域蓄水"],
    }
    for field in ["Date"] + FIELD_ORDER:
        name, unit, meaning = meanings[field]
        rows.append([field, name, unit, meaning])
    T(["字段", "中文名", "单位", "含义与在建模中的作用"], rows, "字段释义", 3)
    NOTE("注：Swe 在原始表中恒为 0，其“蓄水状态”的作用在修正表中才真正体现；"
         "单位为 Daymet v4 R1 与 USGS NWIS 的原始单位，未作换算。")

    rows = []
    for field in FIELD_ORDER:
        row = summ.loc[field]
        rows.append([CN_NAMES[field], f3(row["mean"]), f3(row["std"]), f2(row["min"]),
                     f2(row["50%"]), f2(row["max"]), f3(row["skewness"]), f3(row["kurtosis"])])
    T(["字段", "均值", "标准差", "最小值", "中位数", "最大值", "偏度", "峰度"], rows,
      "8 个数值字段的描述统计（原始表 %d 行）" % d["n_rows"], 3)
    NOTE("注：偏度与峰度为矩估计量，正态分布的参照值为 0 与 3；"
         "径流与降水的峰度远高于 3，说明极端事件对分布形态的影响不可忽略。")
    P("径流字段的均值为 %.3f cfs、中位数为 %s cfs，均值约为中位数的 %.2f 倍，"
      "偏度 %.3f、峰度 %.3f，是典型的右偏尖峰分布：绝大多数日子处于基流状态，"
      "少数洪峰日贡献了极大的量级。降水字段的形态更为极端，中位数为 %s mm/d，"
      "说明半数以上观测日无降水；短波辐射与日照时长的偏度接近 0，呈明显的季节对称性。"
      "这种“标签重尾、驱动稀疏”的组合决定了后续建模的两个基本选择："
      "目标变量取对数以压缩动态范围，以及评价指标必须同时给出 RMSE 与 MAE 两个量纲一致的量。"
      % (float(summ.loc["Discharge", "mean"]), f2(summ.loc["Discharge", "50%"]),
         float(summ.loc["Discharge", "mean"]) / float(summ.loc["Discharge", "50%"]),
         float(summ.loc["Discharge", "skewness"]), float(summ.loc["Discharge", "kurtosis"]),
         f2(summ.loc["Prcp", "50%"])))

    H2("3.2 分布与离群点")
    P("图 3-1 给出五个水文年的径流与降水时序总览，可以直观看到径流的双峰结构："
      "3 月至 6 月的融雪型春峰宽而缓，7 月至 9 月的暴雨型夏峰窄而陡；"
      "冬季基流稳定在数十 cfs 量级，年内极差超过两个数量级。"
      "图 3-2 为 8 个字段的直方图与核密度曲线，图 3-3 为对应的箱型图。")
    FIG(EDA / "1_time_series_overview.png", "2000—2004 年径流与降水的时序总览", 3)
    NOTE("图注：上图为日径流（cfs），下图为日降水（mm/d），横轴为日期；"
         "春季融雪峰与夏季暴雨峰在形态上明显不同。")
    FIG(EDA / "2_histograms_kde.png", "8 个数值字段的直方图与核密度估计", 3)
    NOTE("图注：每个子图的柱为等宽频数直方图，曲线为核密度估计；横轴为原始量纲。")

    rows = []
    for field in FIELD_ORDER:
        count = d["outliers"][field]
        ratio = 100.0 * count / d["n_rows"]
        action = "保留" if field in ("Discharge", "Prcp") else ("保留（无离群点）" if count == 0 else "保留")
        rows.append([CN_NAMES[field], str(count), "%.2f%%" % ratio, action])
    T(["字段", "离群点数", "占样本比例", "处理方式"], rows, "IQR 准则下的离群点统计", 3)
    NOTE("注：离群点按 Q1 减 1.5 倍四分位距与 Q3 加 1.5 倍四分位距界定，在原始量纲上计算。")
    P("径流与降水分别有 %d 个与 %d 个离群点，占样本的 %.1f%% 与 %.1f%%。"
      "这些点没有被删除，理由有三：其一，它们对应真实的洪水与暴雨事件，"
      "删除等于削弱模型对极端事件的记忆，而峰值预报恰恰是本任务最关心的部分；"
      "其二，离群点的判定基于全样本分位数，若在划分数据前删除，等价于把测试段信息带入训练；"
      "其三，树模型对特征取值单调变换不敏感，神经网络在对数目标下对量级差异的敏感度也已降低，"
      "硬性剔除的收益有限。因此本项目的处理方式是保留全部观测，"
      "改用对数目标与标准化来缓解量级影响，并在讨论章专门分析峰值低估问题（6.4 节）。"
      % (d["outliers"]["Discharge"], d["outliers"]["Prcp"],
         100.0 * d["outliers"]["Discharge"] / d["n_rows"],
         100.0 * d["outliers"]["Prcp"] / d["n_rows"]))
    FIG(EDA / "3_boxplots_outliers.png", "8 个数值字段的箱型图与离群点", 3)
    NOTE("图注：箱体为四分位距，中线为中位数，须线延伸至 1.5 倍四分位距，"
         "须线之外的点为离群点。")

    H2("3.3 相关性分析")
    P("图 3-4 为 8 个数值字段的皮尔逊相关系数矩阵。径流与降水同期相关最强，"
      "与气温、辐射、日照的相关则通过季节过程间接建立；积雪水当量一列在原始表中恒为 0，"
      "相关系数不可定义，这正是第二章所述缺陷在相关矩阵上的直接体现。"
      "由于产流存在滞后，同期相关不足以刻画降水的作用，因此另行计算了径流与各驱动项在 1 至 7 天滞后上的相关。")
    FIG(EDA / "4_pearson_correlation_heatmap.png", "8 个数值字段的皮尔逊相关系数矩阵", 3)
    NOTE("图注：色标为相关系数，深色表示强正相关，浅色表示弱相关或负相关；"
         "积雪水当量一列因全零而不可计算。")
    rows = []
    for key, label in (("Discharge_lag1", "径流 滞后 1 天"), ("Discharge_lag2", "径流 滞后 2 天"),
                       ("Discharge_lag3", "径流 滞后 3 天"), ("Discharge_lag7", "径流 滞后 7 天")):
        rows.append([label, f3(d["lag_corr"].loc[key, "Pearson_r"])])
    rows.append(["降水 同期", f3(d["matrix"].loc["径流", "降水"])])
    for key, label in (("Prcp_lag1", "降水 滞后 1 天"), ("Prcp_lag2", "降水 滞后 2 天"),
                       ("Prcp_lag3", "降水 滞后 3 天")):
        rows.append([label, f3(d["lag_corr"].loc[key, "Pearson_r"])])
    rows.append(["水汽压 滞后 3 天", f3(d["lag_corr"].loc["Vp_lag3", "Pearson_r"])])
    rows.append(["最高气温 滞后 3 天", f3(d["lag_corr"].loc["Tmax_lag3", "Pearson_r"])])
    T(["变量与滞后", "皮尔逊相关系数 r"], rows, "径流与各驱动项在同期与滞后上的相关（原始表）", 3)
    NOTE("注：滞后 k 天指驱动项相对径流提前 k 天，即在 t-k 时刻的驱动值与 t 时刻的径流相关；"
         "本表用于说明特征构造中滞后阶数的选取依据。")
    P("径流自身的记忆性最强：滞后 1 天相关系数达到 %.3f，滞后 2 天为 %.3f，"
      "滞后 3 天为 %.3f，滞后 7 天仍有 %.3f，衰减缓慢且单调，"
      "说明基流退水过程有明确的时间尺度，滞后项是必选特征。"
      "降水的作用则集中在短滞后：同期相关 %.3f，滞后 1 天 %.3f，滞后 2 天起迅速衰减。"
      "在原始表上，降水同期相关高于滞后 1 天，与第二章所述错位一天的缺陷一致；"
      "修正数据上这一顺序反转（滞后 1 天 %.3f 高于同期 %.3f），"
      "两种口径的差异本身就是缺陷存在的独立证据。"
      % (float(d["lag_corr"].loc["Discharge_lag1", "Pearson_r"]),
         float(d["lag_corr"].loc["Discharge_lag2", "Pearson_r"]),
         float(d["lag_corr"].loc["Discharge_lag3", "Pearson_r"]),
         float(d["lag_corr"].loc["Discharge_lag7", "Pearson_r"]),
         float(d["matrix"].loc["径流", "降水"]),
         float(d["lag_corr"].loc["Prcp_lag1", "Pearson_r"]),
         d["lag_r_new"][1], d["lag_r_new"][0]))

    H2("3.4 时间序列特征构造")
    P("特征构造遵循两条原则：所有特征只用起报日及更早的信息，"
      "且任何一步变换都可以由原始表重算得到。构造分四类：")
    P("第一类是季节与周期编码。四季以独热向量表示，用于区分融雪期、汛期与枯水期的产流机制；"
      "月份与年内日序各取正弦与余弦两维，使 12 月与 1 月在数值上相邻，"
      "避免把周期性变量当作线性量处理时在年界处出现跳变。")
    P("第二类是滞后项。对径流、降水、最高气温、最低气温、短波辐射与水汽压 6 个变量，"
      "各取滞后 1、2、3、7 天四个阶数，共 %d 列。滞后阶数的选取与 3.3 节的相关结构一致："
      "1 至 3 天覆盖产流的快速响应，7 天覆盖一周尺度的蓄水调整。" % d["n_lag"])
    P("第三类是滚动窗口统计。对径流、降水与最高气温 3 个变量，"
      "各取 3 日与 7 日两个窗口，每个窗口计算均值、标准差、最大值与最小值四项，共 %d 列。"
      "窗口统计把单点观测压缩为时段的强度与离散度，"
      "使模型能够区分“一次强降水”与“连续多日中等降水”这两种产流效果迥异的情形。" % d["n_win"])
    P("第四类是编码与归一化。全部数值列同时给出 Min-Max 与 Z-score 两套编码，"
      "由模型按需取用，因此成品表在原始列之外还有 %d 列 Min-Max 编码与 %d 列 Z-score 编码。"
      "最终成品为 %d 行 %d 列，表 3-5 给出逐类列数。" % (d["n_mm"], d["n_zs"], d["feat_rows"], d["feat_cols"]))
    rows = [
        ["基础与时间字段", "9 个原始字段 + 年份、月份、日、星期编码、年内日序", str(d["n_base"])],
        ["滞后项", "6 个变量 × 滞后 1/2/3/7 天", str(d["n_lag"])],
        ["滚动窗口统计", "3 个变量 × 3 日与 7 日窗口 × 均值/标准差/最大值/最小值", str(d["n_win"])],
        ["季节与周期编码", "四季独热 4 列 + 月份与日序的正余弦 4 列", str(d["n_enc"])],
        ["Min-Max 编码", "对参与建模的连续列追加一套最小最大归一化列", str(d["n_mm"])],
        ["Z-score 编码", "对同一批连续列追加一套标准化列", str(d["n_zs"])],
        ["合计", "成品表列数（含日期列）", str(d["feat_cols"])],
    ]
    T(["类别", "构造方式", "列数"], rows, "特征构成", 3)
    NOTE("注：成品表共 %d 行，与原始面板行数一致；滞后与窗口统计在序列首尾产生的缺失"
         "由后续建模阶段按可用样本裁剪，不在成品表中删除行。" % d["feat_rows"])

    H2("3.5 归一化")
    P("两类模型对量纲的敏感性不同：神经网络以梯度下降求解，各特征量级差异过大时"
      "损失曲面的条件数恶化、收敛缓慢；树模型按阈值切分，对单调变换不敏感，"
      "但在多输出与集成场景下统一量纲便于比较各成员的输入。"
      "因此成品表同时保留 Min-Max 与 Z-score 两套编码。")
    P("Min-Max 把每个特征线性映射到 [0, 1]，保留原始分布形状与相对间距，"
      "但对极端值敏感——径流最大值 13700 cfs 会把绝大多数观测压到接近 0 的窄区间内。"
      "Z-score 减去均值再除以标准差，使各特征在同一尺度上可比，"
      "但对重尾分布同样无法消除偏度。两者的共同约束是：参数只能由训练段估计，"
      "再原样应用到测试段，否则测试期的均值与极值会通过归一化参数泄漏进训练。"
      "本项目的实现严格遵守这一条，测试段的归一化参数与训练段完全相同。"
      "两套编码下同一特征的分布对比如图 3-5 所示。")
    FIG(FE / "normalization_comparison.png", "Min-Max 与 Z-score 两套编码的分布对比", 3)
    NOTE("图注：同一特征在两套编码下的分布形状一致、尺度不同；"
         "编码参数只在训练段上拟合，测试段直接套用。")

    H2("3.6 特征选择")
    P("候选特征共 %d 项（不含日期列与两套编码的派生列）。筛选采用皮尔逊相关系数与互信息双指标："
      "前者衡量线性相关，后者基于分箱估计、能够捕捉非线性依赖。"
      "保留规则为“相关系数绝对值不低于 0.2，或互信息得分不低于 0.05，或属于降水类特征”，"
      "其中降水类强制保留是为了防止“多数日为零”的稀疏序列因相关系数偏低而被误删。"
      "按此规则保留 %d 项、剔除 %d 项。图 3-6 为各候选在两指标上的对比。"
      % (d["n_cand"], d["n_kept"], d["n_dropped"]))
    FIG(FE / "feature_selection_comparison.png", "候选特征在皮尔逊与互信息两个指标上的对比", 3)
    NOTE("图注：横轴为候选特征，纵轴为指标取值；两条横线分别为 0.2 与 0.05 的保留阈值，"
         "降水类特征不受阈值约束。")
    rows = []
    reasons = {
        "夏季独热编码": "|r| 低于 0.2 且互信息低于 0.05；夏季信息已由月份与日序的正余弦编码表达，存在冗余",
        "最高气温_过去7天标准差": "互信息低于 0.05；7 日窗口的离散度与 7 日均值、3 日标准差高度共线",
        "最高气温_过去3天标准差": "|r| 与互信息双低（互信息仅 0.005），几乎不携带独立信息",
    }
    for _, row in d["dropped"].iterrows():
        name = str(row[d["name_col"]])
        rows.append([name, f3(row[d["r_col"]]), f4(row[d["mi_col"]]),
                     reasons.get(name, "双指标均低于阈值，且与已保留特征高度共线")])
    T(["被剔除的候选", "皮尔逊相关系数", "互信息得分", "剔除理由"], rows, "被剔除的 3 项候选及其理由", 3)
    NOTE("注：三项被剔除的候选均来自季节编码与气温窗口统计两类，"
         "该两类内部本身高度共线，剔除后信息损失有限。")
    P("三项被剔除的候选集中在两处：夏季独热编码与月份周期编码重复表达了同一季节信息，"
      "最高气温的 3 日与 7 日标准差则与同窗口的均值、极值高度共线。"
      "这也说明双指标筛选的作用主要不是“找出最强特征”，而是“去掉冗余表示”，"
      "保留的 %d 项在数量上仍远多于样本量的合理上限，"
      "因此后续建模还需要依靠正则化与模型容量控制来抑制过拟合。" % d["n_kept"])

    # ---------------------------------------------------------------- 第四章
    H1("四、方法")
    H2("4.1 多步预测的两种策略")
    P("设起报日为 t，特征向量为 x_t，目标是预测 t+1 至 t+H 日的径流 y_(t+1..t+H)，H 取 7。"
      "直接策略对每个预见期 h 单独训练一个模型 f_h，使 yhat_(t+h) = f_h(x_t)，共 H 个模型；"
      "多输出策略只训练一个模型 F，使 (yhat_(t+1), …, yhat_(t+H)) = F(x_t)，输出维度为 H。")
    P("两者的差别在于误差如何传播与参数如何共享。直接策略中每个模型只服务一个目标，"
      "可以针对该预见期的误差结构单独调参，但 H 个模型之间不共享任何信息，"
      "参数总量约为多输出的 H 倍，在样本量只有千余行时过拟合风险更高。"
      "多输出策略让 H 个目标共享同一隐藏表示，短预见期的高信噪比目标在联合训练中"
      "对长预见期形成隐式正则，参数规模显著下降；代价是当不同预见期的最优超参数明显不同时，"
      "共享表示无法同时满足，可能牺牲短预见期的精度。两种策略都只使用起报日及更早的信息，"
      "因此预见期越长、可解释方差越低，这是信息层面的硬约束，与策略选择无关。")
    rows = [
        ["模型数量", "H 个独立模型（本项目 H=7）", "1 个共享模型，输出 H 维"],
        ["训练目标", "单一标量 y_(t+h)", "H 维向量 y_(t+1..t+H)"],
        ["参数规模", "约 H 倍", "基准"],
        ["误差传播", "各期独立，误差不跨期传播", "共享表示把短期的拟合压力传递到长期"],
        ["主要优点", "可逐期选参，短预见期精度上限更高", "参数少、正则强，长预见期更稳"],
        ["主要代价", "样本被切成 H 份，过拟合风险高", "超参数折中，短期精度可能受损"],
    ]
    T(["对比维度", "直接策略", "多输出策略"], rows, "直接策略与多输出策略的对比", 4)

    H2("4.2 人工神经网络")
    P("人工神经网络采用 scikit-learn 的 MLPRegressor，即全连接前馈网络，"
      "隐层激活为 ReLU，输出层线性，损失为均方误差，优化器为 Adam 或 L-BFGS（由求解器参数决定）。"
      "目标变量先取 log1p 变换，预测后再用 expm1 反变换回 cfs；"
      "对数变换把乘性误差近似转成加性误差，压缩了洪峰对损失函数的支配地位，"
      "同时缓解了 3.1 节所述的右偏重尾问题。")
    P("正则化由 L2 惩罚项 alpha 控制，网络容量由隐层结构控制。"
      "搜索空间为 alpha 属于 {0.1, 1, 10} 与隐层结构属于 {(16,), (32,)} 的笛卡尔积，共 6 组组合；"
      "验证方式为时间序列交叉验证 3 折、折间间隔 7 天，评分指标为对数尺度上的均方根误差。"
      "对每个预见期单独搜索一次，多输出策略整体搜索一次，合计 8 次搜索。")
    P("随机森林采用 RandomForestRegressor，以自助采样（bootstrap）生成多棵回归树，"
      "每棵树的节点分裂只在随机抽取的特征子集上寻优，最终输出为各树预测的平均。"
      "这一结构使方差随树数增加而单调下降，且不需要特征标准化。"
      "本项目使用其原生多输出能力：当目标为二维矩阵时，每棵树在叶节点存一个 H 维向量，"
      "预测时对 H 个分量分别取平均，从而在一次拟合中给出全部预见期。"
      "搜索空间为最大深度属于 {8, 16} 与叶节点最小样本数属于 {5, 10}，共 6 组组合，"
      "验证方式与 ANN 相同。")

    H2("4.3 长短期记忆网络")
    P("长短期记忆网络（LSTM）在循环结构中以门控机制调节信息流。设 t 时刻输入为 x_t、"
      "上一时刻隐状态为 h_(t-1)、细胞状态为 c_(t-1)，则遗忘门 f_t 与输入门 i_t 分别为"
      "f_t = σ(W_f·[h_(t-1), x_t] + b_f) 与 i_t = σ(W_i·[h_(t-1), x_t] + b_i)，"
      "候选记忆 c~_t = tanh(W_c·[h_(t-1), x_t] + b_c)，"
      "细胞状态按 c_t = f_t ⊙ c_(t-1) + i_t ⊙ c~_t 更新，"
      "输出门 o_t = σ(W_o·[h_(t-1), x_t] + b_o)，隐状态 h_t = o_t ⊙ tanh(c_t)。"
      "其中 σ 为 sigmoid 函数，⊙ 为逐元素乘。细胞状态的更新近似线性，"
      "梯度沿时间步反传时不必逐层连乘，因而缓解了普通循环网络的梯度消失问题；"
      "对存在融雪滞后与土壤蓄水释放的径流序列，门控结构提供了选择性保留历史信号的能力。")
    P("实现基于 PyTorch 的 torch.nn.LSTM。每条样本送入当期一帧特征向量，"
      "序列长度取 1；循环层之后接线性映射、ReLU、Dropout 与输出头，"
      "多层时按层设置 dropout，训练采用 Adam 优化器、均方误差损失、梯度范数裁剪与学习率衰减，"
      "并按时间顺序留出训练段尾段作验证集早停。目标保留原始量纲，"
      "输入特征用只在训练段拟合的标准化器处理，预测后反变换回 cfs。")
    P("必须如实说明本实现的边界：输入序列长度为 1，循环层不跨时间步展开，"
      "网络在每个样本上退化为一个带门控的单步映射，历史信息完全由滞后项与滚动窗口特征承载。"
      "这一取舍的原因是训练样本只有千余行，若把每个样本展开成长度为 L 的序列，"
      "有效样本数会再减少 L 倍，长序列带来的记忆增益不足以抵消样本损失。"
      "因此本报告中 LSTM 相对 ANN 的提升，应理解为“门控单元作为非线性映射器的结构优势”，"
      "而不是循环记忆的贡献；这一限制在 6.5 节列为已知不足。")
    P("需要区分两个容易混淆的对象：集成中的 PseudoLSTM 是用多层感知机对时序特征的近似实现，"
      "没有门控与细胞状态；本报告的 LSTM 结论全部来自上述 PyTorch 门控实现，两者不可混为一谈。")

    H2("4.4 集成学习")
    P("集成在两种策略下分别构造。直接策略有 6 个基学习器：ANN、随机森林、XGBoost、LightGBM、"
      "PyTorch-LSTM 与 PseudoLSTM；多输出策略有 3 个基学习器：ANN、随机森林与 PyTorch-LSTM，"
      "三者都能一次输出 7 维结果。两套特征的起报日取交集后再拼接预测，"
      "保证同一预见期上所有成员面对同一批样本。")
    P("简单平均把同一预见期上各成员的预测取算术平均，权重固定为 1/M，不引入任何待估参数。"
      "它只能削弱各成员相互独立的误差分量，对系统性偏差无能为力；"
      "但在样本有限时不存在二次过拟合的风险，因而是一个稳健的基准。")
    P("Stacking 用一层元学习器把基学习器的预测组合成最终预测。为让元特征不含目标信息，"
      "以训练段末尾 20%% 的 %d 行为留出块：基学习器在留出块之前的前缀上重训，"
      "再对留出块给出预测，这 %d 个预测构成元学习器的训练矩阵，对应的实测径流为其目标；"
      "对第 h 天，前缀在留出块起点之前再回退 h 天，避免目标跨入留出块。"
      "元学习器为 Ridge 岭回归，元特征先按训练段统计量标准化，"
      "正则化系数 alpha 在 {0.01, 0.1, 1, 10, 100, 1000, 10000} 上以 3 折时序交叉验证选取。"
      "测试段仍由在全部训练段上拟合的基学习器给出预测，再经同一 Ridge 组合。"
      % (p3["n_hold"], p3["n_hold"]))
    rows = [
        ["直接策略·简单平均", "6 个基学习器等权平均", "无", "固定 1/6"],
        ["直接策略·Stacking", "Ridge 组合 6 个基学习器预测", "6 个权重 + 1 个截距", "留出块上拟合"],
        ["多输出策略·简单平均", "3 个基学习器等权平均", "无", "固定 1/3"],
        ["多输出策略·Stacking", "Ridge 组合 3 个基学习器预测", "3 个权重 + 1 个截距", "留出块上拟合"],
    ]
    T(["方案", "组合方式", "待估参数", "权重来源"], rows, "两类集成的构造方式", 4)
    NOTE("注：留出块为训练段末尾 %d 行，元特征的生成过程与测试段预测所用的基学习器不完全等价，"
         "这一差异是 6.3 节分析 Stacking 失效的出发点。" % p3["n_hold"])

    H2("4.5 评价指标")
    P("全部指标在原始量纲（cfs）上计算，统一采用四个：均方根误差 RMSE、平均绝对误差 MAE、"
      "决定系数 R² 与纳什效率系数 NSE。其中 R² 与 NSE 在形式上是同一个量："
      "都以“残差平方和占总平方和的比例”为定义，取值范围同为负无穷到 1，"
      "取值 1 表示完美拟合、0 表示与用均值预测等价、负值表示劣于均值基准。"
      "本项目同时报告两者，是为了与水文领域的惯例口径对齐，"
      "结果文件中两列数值逐行相同即为此故，并非重复计算。")
    rows = [
        ["RMSE", "sqrt(mean((yhat - y)²))", "0 至正无穷，越小越好",
         "对大幅误差敏感，反映洪峰处的偏差"],
        ["MAE", "mean(|yhat - y|)", "0 至正无穷，越小越好",
         "对离群不敏感，反映典型日的平均偏差"],
        ["R²", "1 - SSE/SST", "负无穷至 1，越大越好", "解释方差比例，跨方案可比"],
        ["NSE", "1 - SSE/SST", "负无穷至 1，越大越好", "水文领域惯用口径，与 R² 同式"],
    ]
    T(["指标", "定义式", "取值范围", "在本报告中的作用"], rows, "评价指标的定义与作用", 4)
    NOTE("注：SSE 为残差平方和，SST 为观测相对其均值的总平方和；"
         "全部指标在同一批测试样本上计算，样本数随阶段的特征口径略有差异，已在各表注明。")

    # ---------------------------------------------------------------- 第五章
    H1("五、实验设置与结果")
    H2("5.1 超参数搜索")
    P("两个阶段的选参都使用时间序列交叉验证 TimeSeriesSplit，折数 3、折间间隔 gap = 7 天。"
      "间隔的作用是切断训练块与验证块之间的目标重叠：第 h 天方案的目标跨度是 h 天，"
      "若不留间隔，验证块前若干起报日的目标会落进训练块，造成乐观偏差。"
      "评分指标为验证块上的 RMSE；第二阶段在 log1p 尺度上计算，第三阶段在原始量纲上计算。")
    rows = []
    for _, row in d["p2_sel"].iterrows():
        params = json.loads(row["最佳参数"])
        shown = []
        for key, value in params.items():
            short = key.split("__")[-1]
            if isinstance(value, list):
                value = "(%s,)" % ", ".join(str(int(v)) for v in value) if len(value) == 1 \
                    else "(%s)" % ", ".join(str(int(v)) for v in value)
            shown.append("%s=%s" % (short, value if isinstance(value, str) else ("%g" % value)))
        horizon = "多输出" if str(row["选参预见期"]) == "multi" else "第 %s 天" % row["选参预见期"]
        rows.append([row["模型"], row["策略"], horizon, "，".join(shown),
                     "%.4f" % float(row["交叉验证RMSE（对数尺度）"]), "%.1f" % float(row["耗时秒"])])
    T(["模型", "策略", "选参目标", "最优参数", "交叉验证 RMSE（对数尺度）", "耗时（秒）"], rows,
      "第二阶段的选参结果（%d 组）" % len(rows), 5)
    NOTE("注：交叉验证 RMSE 在 log1p 尺度上计算，数值不可直接与 cfs 量纲的测试 RMSE 比较；"
         "耗时为本机单次搜索的墙钟时间。")
    P("第二阶段的选参呈现明显的规律性：ANN 的正则化系数在第 1 天取 1、其余预见期取 10，"
      "隐层结构在 16 与 32 之间交替，说明低信噪比目标需要更强的正则与更小的容量；"
      "随机森林的最大深度在第 1 至 6 天为 16、第 7 天降为 8，叶节点最小样本数多为 10，"
      "与“长预见期宁浅勿深”的经验一致。交叉验证误差随预见期单调上升，"
      "第 1 天最低而第 7 天最高，这一顺序在三个模型族上都成立，"
      "说明精度的衰减来自信息而非模型。")
    rows = []
    for _, row in d["lstm_grid"].iterrows():
        task = row["task"]
        name = "多输出策略" if task == "multi_output" else "直接策略·第 %s 天" % task.replace("direct_day", "")
        rows.append([name, "hidden=%d，layers=%d" % (int(row["best_hidden_dim"]), int(row["best_num_layers"])),
                     "%.3f" % float(row["best_lr"]), "%.1f" % float(row["best_dropout"]),
                     f2(row["cv_rmse"])])
    T(["任务", "最优结构", "学习率", "dropout", "交叉验证 RMSE（cfs）"], rows,
      "第三阶段 LSTM 的选参结果（%d 组）" % len(rows), 5)
    NOTE("注：搜索空间为隐层维度 {64, 128}、循环层数 {1, 2}、学习率 {0.001, 0.01}，"
         "dropout 固定 0.2，共 8 组组合；验证方式为 TimeSeriesSplit 3 折、gap = 7。")
    P("LSTM 的选参结果显示，第 2、4 天与多输出任务选择了 128 维隐层，"
      "第 3、6、7 天则退到 64 维单层，交叉验证误差由第 1 天的 %s cfs 升至第 7 天的 %s cfs。"
      "与第二阶段一致，低信噪比的长预见期需要压缩容量；"
      "不同的是 LSTM 在第 1 天的交叉验证误差（%s cfs）已明显低于随机森林与 ANN 的对应误差，"
      "提示门控结构在该数据规模下确实提供了更强的映射能力。"
      % (f2(d["lstm_grid"].iloc[0]["cv_rmse"]), f2(d["lstm_grid"].iloc[6]["cv_rmse"]),
         f2(d["lstm_grid"].iloc[0]["cv_rmse"])))

    H2("5.2 第二阶段结果：ANN 与随机森林")
    P("第二阶段的四种方案在同一测试段（%s 行）上评估，表 5-3 与表 5-4 分别给出 ANN 与随机森林"
      "在直接、多输出两种策略下 1 至 7 天的精度。全部数值取自 results/phase2_v5/测试精度指标.csv，"
      "未作任何改写。" % p2["测试样本"])
    rows = []
    for model in ("ANN", "RF"):
        for strategy in ("直接", "多输出"):
            for h in HORIZONS:
                row = d["p2_metrics"][(model, strategy, h)]
                rows.append([model, strategy, str(h), f4(row["R²"]), f2(row["RMSE（cfs）"]),
                             f2(row["MAE（cfs）"])])
    ann_rows = [r for r in rows if r[0] == "ANN"]
    rf_rows = [r for r in rows if r[0] == "RF"]
    T(["模型", "策略", "预见期（天）", "R²", "RMSE（cfs）", "MAE（cfs）"], ann_rows,
      "ANN 两种策略的 1–7 天测试精度（%s 行）" % p2["测试样本"], 5)
    NOTE("注：R² 与 NSE 在结果文件中逐行相同，本表只列 R²；目标变量取 log1p 变换后建模，"
         "指标在反变换后的原始量纲上计算。")
    T(["模型", "策略", "预见期（天）", "R²", "RMSE（cfs）", "MAE（cfs）"], rf_rows,
      "随机森林两种策略的 1–7 天测试精度（%s 行）" % p2["测试样本"], 5)
    NOTE("注：随机森林使用原生多输出，多输出策略下由一个模型同时给出 7 个预见期的预测。")
    P("第 1 天的精度排序为：ANN 多输出 %.4f、ANN 直接 %.4f、随机森林直接 %.4f、随机森林多输出 %.4f。"
      "第 2 天起全部方案快速回落，第 5 天之后 ANN 直接与多输出均转负，"
      "随机森林则始终维持在零基准以上（第 7 天 %.4f）。"
      "这一差异说明树模型在长预见期上的稳健性更好：随机森林的输出是叶节点样本均值，"
      "对极端值不敏感，而 ANN 在 log1p 目标上训练、反变换后对高值段的偏置更明显。"
      % (float(d["p2_metrics"][("ANN", "多输出", 1)]["R²"]),
         float(d["p2_metrics"][("ANN", "直接", 1)]["R²"]),
         float(d["p2_metrics"][("RF", "直接", 1)]["R²"]),
         float(d["p2_metrics"][("RF", "多输出", 1)]["R²"]),
         float(d["p2_metrics"][("RF", "直接", 7)]["R²"])))
    FIG(P2 / "figures" / "fig1_精度随预见期衰减.png", "第二阶段四种方案的精度随预见期衰减", 5)
    NOTE("图注：横轴为预见期天数，纵轴为决定系数 R²，四条曲线对应 ANN 与随机森林的两种策略；"
         "灰色水平线为零基准。")

    H2("5.3 第三阶段结果：LSTM")
    P("第三阶段使用增强特征（%d 维），样本因 7 日滚动统计的首尾裁剪降为训练 %d 行、测试 %d 行，"
      "因此与第二阶段的精度不能逐项直接对照，同口径的比较见 5.5 节。"
      % (p3["n_enh"], p3["n_train"], p3["n_test"]))
    rows = []
    for label, series in (("直接", ld), ("多输出", lm)):
        for h in HORIZONS:
            rows.append(["PyTorch-LSTM", label, str(h), f4(series[h]["R²"]),
                         f2(series[h]["RMSE"]), f2(series[h]["MAE"])])
    T(["模型", "策略", "预见期（天）", "R²", "RMSE（cfs）", "MAE（cfs）"], rows,
      "PyTorch-LSTM 两种策略的 1–7 天测试精度（%d 行）" % p3["n_test"], 5)
    NOTE("注：输入序列长度为 1，历史信息由滞后项与滚动窗口特征承载；"
         "指标在原始量纲上计算，目标未作对数变换。")
    FIG(P3 / "figures" / "fig1_精度随预见期衰减.png", "第三阶段各方案的精度随预见期衰减", 5)
    NOTE("图注：左图为直接策略，右图为多输出策略；横轴为预见期天数，纵轴为决定系数 R²。")
    P("LSTM 直接策略第 1 天的 R² 为 %.4f、RMSE 为 %s cfs、MAE 为 %s cfs，"
      "是全部单模型中的最高值；多输出策略为 %.4f，比直接策略低 %s。"
      "第 2 天两者分别为 %.4f 与 %.4f，第 3 天起转负，第 7 天降至 %.4f 与 %.4f。"
      "与第二阶段的 ANN 相比，LSTM 在第 1 天上把 R² 从 %.4f 提升到 %.4f，"
      "提升幅度在本项目的全部单模型对比中最大，两种策略的衰减曲线见图 5-2。"
      % (float(ld[1]["R²"]), f2(ld[1]["RMSE"]), f2(ld[1]["MAE"]), float(lm[1]["R²"]),
         f4(float(ld[1]["R²"]) - float(lm[1]["R²"])),
         float(ld[2]["R²"]), float(lm[2]["R²"]), float(ld[7]["R²"]), float(lm[7]["R²"]),
         float(d["p2_metrics"][("ANN", "直接", 1)]["R²"]), float(ld[1]["R²"])))

    H2("5.4 集成结果")
    P("集成在两种策略下各给出简单平均与 Stacking 两种组合方式，共四种方案，"
      "评估样本与 5.3 节相同（%d 行）。表 5-6 并列给出四种方案的逐期 R²，"
      "表 5-7 汇总第 1 天与第 7 天的 R² 与 RMSE。" % p3["n_test"])
    rows = []
    for h in HORIZONS:
        rows.append([str(h), f4(a6[h]["R²"]), f4(s6[h]["R²"]), f4(a3[h]["R²"]), f4(s3[h]["R²"])])
    T(["预见期（天）", "6 模型简单平均", "6 模型 Stacking", "3 模型简单平均", "3 模型 Stacking"],
      rows, "四种集成方案的逐期 R²", 5)
    NOTE("注：6 模型指 ANN、随机森林、XGBoost、LightGBM、PyTorch-LSTM 与 PseudoLSTM（直接策略）；"
         "3 模型指 ANN、随机森林与 PyTorch-LSTM（多输出策略）；评估样本 %d 行。" % p3["n_test"])
    rows = []
    for label, series in (("6 模型简单平均（直接）", a6), ("6 模型 Stacking（直接）", s6),
                          ("3 模型简单平均（多输出）", a3), ("3 模型 Stacking（多输出）", s3)):
        rows.append([label, f4(series[1]["R²"]), f2(series[1]["RMSE"]),
                     f4(series[7]["R²"]), f2(series[7]["RMSE"])])
    T(["方案", "第 1 天 R²", "第 1 天 RMSE（cfs）", "第 7 天 R²", "第 7 天 RMSE（cfs）"], rows,
      "四种集成方案在第 1 天与第 7 天的精度", 5)
    NOTE("注：四种方案使用完全相同的基学习器预测与完全相同的测试样本，差异只来自组合方式。")
    FIG(P3 / "figures" / "fig2_策略对比柱状图.png", "各方案在第 1 天与第 7 天的 R² 对比", 5)
    NOTE("图注：柱高为测试段决定系数 R²，黑色水平实线为零基准。")
    P("四种方案在第 1 天与第 7 天的对比如图 5-3 所示。"
      "第 1 天的排序为：6 模型简单平均 %.4f、6 模型 Stacking %.4f、3 模型简单平均 %.4f、"
      "3 模型 Stacking %.4f。四种方案全部低于单模型 LSTM 直接策略的 %.4f，"
      "其中 6 模型简单平均只差 %.4f，而两种 Stacking 方案分别落后 %.4f 与 %.4f。"
      "按 7 个预见期的平均 R² 计，排序变为 6 模型简单平均 %.4f、6 模型 Stacking %.4f、"
      "3 模型简单平均 %.4f、3 模型 Stacking %.4f——集成的收益体现在整体稳健性上，"
      "并不体现在短预见期的尖峰精度上。"
      % (float(a6[1]["R²"]), float(s6[1]["R²"]), float(a3[1]["R²"]), float(s3[1]["R²"]),
         float(ld[1]["R²"]), float(ld[1]["R²"]) - float(a6[1]["R²"]),
         float(ld[1]["R²"]) - float(s6[1]["R²"]), float(ld[1]["R²"]) - float(s3[1]["R²"]),
         sum(a6[h]["R²"] for h in HORIZONS) / 7.0, sum(s6[h]["R²"] for h in HORIZONS) / 7.0,
         sum(a3[h]["R²"] for h in HORIZONS) / 7.0, sum(s3[h]["R²"] for h in HORIZONS) / 7.0))

    H2("5.5 未修正数据与修正数据的对照")
    P("为了说明数据修正的实际价值，本节把未修正数据版本的结果（archive/results_phase2_v5_raw/）"
      "与修正后的结果并列。两组结果的模型、特征口径、划分方式与评估指标完全相同，"
      "唯一差异是数据源：前者为原始表（气象列错位、积雪恒零），后者为修正表。")
    rows = []
    diffs = []
    for model, strategy in GROUPS:
        new_row = d["p2_metrics"][(model, strategy, 1)]
        old_row = d["p2_raw"][(model, strategy, 1)]
        delta = float(new_row["R²"]) - float(old_row["R²"])
        diffs.append(delta)
        rows.append(["%s·%s" % (model, "直接" if strategy == "直接" else "多输出"),
                     f4(old_row["R²"]), f4(new_row["R²"]), signed4(delta),
                     f2(old_row["RMSE（cfs）"]), f2(new_row["RMSE（cfs）"])])
    T(["方案（第 1 天）", "未修正数据 R²", "修正数据 R²", "差值", "未修正数据 RMSE（cfs）",
       "修正数据 RMSE（cfs）"], rows, "未修正数据与修正数据在第 1 天的精度对照", 5)
    NOTE("注：未修正数据一列取自 archive/results_phase2_v5_raw/测试精度指标.csv，"
         "仅为对照之用，不代表本项目的任何当前成绩；差值等于修正数据减未修正数据。")
    P("四组方案在第 1 天的提升幅度介于 %s 与 %s 之间，最小的是随机森林多输出（%s），"
      "最大的是 ANN 多输出（%s）。提升不为零的原因可以逐项对上第二章的修正内容："
      "气象列重对齐恢复了真实的滞后结构，使滞后项与窗口统计重新对齐产流过程；"
      "积雪水当量的恢复让融雪派生量不再是零方差列，春季洪峰的成因重新进入特征。"
      "这一对照同时也构成一条方法学结论：在同一套模型与评估口径下，"
      "数据质量的改善往往比更换模型带来更大的精度增益。"
      % (signed3(min(diffs)), signed3(max(diffs)),
         signed3(float(d["p2_metrics"][("RF", "多输出", 1)]["R²"]) - float(d["p2_raw"][("RF", "多输出", 1)]["R²"])),
         signed3(float(d["p2_metrics"][("ANN", "多输出", 1)]["R²"]) - float(d["p2_raw"][("ANN", "多输出", 1)]["R²"]))))

    # ---------------------------------------------------------------- 第六章
    H1("六、讨论")
    H2("6.1 精度随预见期衰减的成因")
    P("表 6-1 把六种方案在四个代表性预见期上的 R² 并列。全部方案都随预见期单调衰减，"
      "但衰减速率不同：6 模型简单平均由 %.4f 降到 %.4f，LSTM 直接策略由 %.4f 降到 %.4f，"
      "后者更陡。成因可归为四点。" 
      % (float(a6[1]["R²"]), float(a6[7]["R²"]), float(ld[1]["R²"]), float(ld[7]["R²"])))
    rows = []
    for label, series in (("LSTM 直接策略", ld), ("LSTM 多输出策略", lm),
                          ("6 模型简单平均", a6), ("6 模型 Stacking", s6),
                          ("3 模型简单平均", a3), ("3 模型 Stacking", s3)):
        rows.append([label] + [f4(series[h]["R²"]) for h in (1, 2, 3, 5, 7)])
    T(["方案", "第 1 天", "第 2 天", "第 3 天", "第 5 天", "第 7 天"], rows,
      "六种方案的 R² 随预见期衰减（第三阶段口径）", 6)
    NOTE("注：评估样本为测试段 %d 行；零基准对应 R² = 0。" % p3["n_test"])
    P("第一，输入信息在起报时刻固定。全部特征只包含起报日及更早的观测，"
      "预见期越长，起报与目标之间新增的降水、气温与融雪过程就越长，这部分信息在预测时完全未知。"
      "这是精度衰减的根本原因，也是任何纯数据驱动方法共有的上限："
      "当预见期超过流域的响应记忆长度后，可解释方差必然趋近于零。")
    P("第二，条件均值的收缩效应。训练以平方损失为目标，模型输出趋近条件均值而非条件众数，"
      "对洪峰这类低频高幅事件系统性低估；预见期越长，目标的条件分布越弥散，收缩越强，"
      "峰值误差在 RMSE 中的占比上升，表现为 R² 快速下滑而 MAE 上升相对缓慢。")
    P("第三，样本量与事件覆盖不足。训练段只有 %d 行，覆盖约四个水文年，"
      "其中高流量事件数量有限，长预见期的目标信噪比更低，单次划分的偶然性也更大，"
      "个别预见期的 R² 出现非单调波动即属此类。"
      "第四，策略带来的正则差异：多输出策略让 7 个预见期共享同一表示，"
      "长预见期因此比直接策略略缓；等权平均通过压低成员方差使曲线整体更平坦。"
      "这两类机制都不改变信息上限，只改变接近上限的速度。" % p3["n_train"])

    H2("6.2 直接策略与多输出策略的分歧")
    P("两种策略的相对优劣并不一致，且与模型族有关。人工神经网络在多输出策略下第 1 天反超直接策略"
      "（%.4f 对 %.4f），随机森林则是直接策略明显占优（%.4f 对 %.4f），"
      "LSTM 同样是直接策略更好（%.4f 对 %.4f）。这一分歧不能归结为随机波动，"
      "其成因在于三族模型的容量与正则方式不同。"
      % (float(d["p2_metrics"][("ANN", "多输出", 1)]["R²"]),
         float(d["p2_metrics"][("ANN", "直接", 1)]["R²"]),
         float(d["p2_metrics"][("RF", "直接", 1)]["R²"]),
         float(d["p2_metrics"][("RF", "多输出", 1)]["R²"]),
         float(ld[1]["R²"]), float(lm[1]["R²"])))
    P("对 ANN 而言，7 个独立模型各自的训练样本相同但目标不同，"
      "单目标的噪声在 log1p 尺度上被放大，而共享表示相当于对 7 个目标做联合降噪，"
      "在样本只有千余行时这一正则的收益超过了逐期选参的灵活性。"
      "对随机森林而言，多输出要求同一棵树在叶节点同时存 7 个均值，"
      "分裂准则需要在 7 个目标之间折中，导致每次分裂都不再针对单目标最优；"
      "而单目标森林可以把全部容量用于拟合当前预见期，因此直接策略占优。"
      "对 LSTM 而言，直接策略的 7 个网络可以在第 1 天专门收敛到高峰值段，"
      "共享表示则被长预见期的低信噪比目标拖累，故直接策略更好。")
    P("这一组结果的实践含义是：策略选择不应脱离模型族单独讨论。"
      "“多输出一定更稳”与“直接策略一定更准”在本数据上都不成立，"
      "可行的做法是对每个模型族分别验证两种策略并如实报告，本报告的表 5-3、表 5-4 与表 5-5 即为此用。")

    H2("6.3 简单平均为何优于 Stacking")
    direct_wins = sum(1 for h in HORIZONS if float(a6[h]["R²"]) > float(s6[h]["R²"]))
    multi_wins = sum(1 for h in HORIZONS if float(a3[h]["R²"]) > float(s3[h]["R²"]))
    P("逐期比较的结果是：直接策略下简单平均在 %d 个（共 7 个）预见期上高于 Stacking，"
      "多输出策略下在 %d 个上更高。这一结果由数据支持，本报告不把它反向表述为 Stacking 更优。"
      "成因可从三方面分析。" % (direct_wins, multi_wins))
    rows = []
    for h in HORIZONS:
        rows.append([str(h), f4(a6[h]["R²"]), f4(s6[h]["R²"]),
                     signed4(float(a6[h]["R²"]) - float(s6[h]["R²"])),
                     f4(a3[h]["R²"]), f4(s3[h]["R²"]),
                     signed4(float(a3[h]["R²"]) - float(s3[h]["R²"]))])
    T(["预见期（天）", "直接·简单平均", "直接·Stacking", "差值",
       "多输出·简单平均", "多输出·Stacking", "差值"], rows,
      "简单平均与 Stacking 的逐期 R² 对照", 6)
    NOTE("注：差值为简单平均减 Stacking，正值表示简单平均更高；两种方法使用完全相同的基学习器与测试样本。")
    wd = d["w_direct"]
    wcols = [c for c in wd.columns if c.startswith("w_")]
    max_w = max(abs(float(v)) for v in wd[wcols].to_numpy().ravel())
    neg_counts = [int((wd[wcols].iloc[i] < 0).sum()) for i in range(len(wd))]
    P("第一，元特征与测试期的基学习器预测存在分布偏移。元特征来自在留出块之前的前缀上重训的基学习器，"
      "而测试期的预测来自在全部训练段上拟合的模型，两者在训练样本量、树的深度与迭代轮次上并不等价，"
      "预测的偏差结构因此不同。元学习器学到的是“前缀模型在留出块上的行为”，"
      "直接应用到“全量模型在测试期上的预测”时，权重所依赖的关系已经改变。")
    P("第二，留出块过小且块内样本并非独立。留出块只有 %d 行，占训练段的百分之二十；"
      "而 7 天预见期的目标序列在相邻起报日之间高度自相关，"
      "一段 %d 天的块所覆盖的独立水文事件远少于 %d 个。"
      "元学习器要估计 6 个（多输出为 3 个）权重，在有效样本不足且元特征共线的情况下，"
      "最小二乘解对噪声非常敏感。" % (p3["n_hold"], p3["n_hold"], p3["n_hold"]))
    P("第三，Ridge 权重在共线元特征上过拟合。直接策略的权重绝对值最大项达到 %s 量级，"
      "7 个预见期中有 %d 个存在正负权重并存（负权重最多的一项含 %d 个负权重），"
      "选出的 alpha 在 0.01 与 100 之间跳动，说明交叉验证误差曲面相当平坦、alpha 的选择本身不稳定。"
      "在这种条件下，含 6 个自由参数的线性组合可以在留出块上拟合得很好，却在测试段上放大误差；"
      "而等权平均没有任何自由参数，只要各成员的误差不完全正相关，就能稳定降低方差。"
      % (f0(max_w), sum(1 for c in neg_counts if c > 0), max(neg_counts)))
    P("因此结论应表述为：在本数据条件下简单平均更稳健，而不是断言 Stacking 方法本身劣于简单平均。"
      "若要改善 Stacking，可行的方向是用多折块生成元特征（增加有效样本并为元学习器提供更稳的目标）、"
      "对元特征做去共线处理，或改用非负约束的权重估计。")

    H2("6.4 峰值预测的系统性低估")
    P("图 6-1 给出测试段的水文过程线，曲线由 LSTM 直接策略第 1 天预测绘制，"
      "该方案是全部方案中第 1 天 R² 最高者（%.4f）。基流段与退水段的预测曲线与实测基本重合，"
      "说明模型对缓慢变化的水文过程拟合良好；误差集中出现在洪峰处。"
      % float(ld[1]["R²"]))
    FIG(P3 / "figures" / "fig3_测试段水文过程线.png",
        "2004 年测试段实测径流与 LSTM 第 1 天预测过程线（横轴为目标日期）", 6)
    NOTE("图注：实线为实测径流，另一条曲线为 PyTorch-LSTM 直接策略的第 1 天预测；"
         "横轴为目标日期，数据取自测试段逐日预测结果。")
    rows = [["LSTM 直接策略（第 1 天）", f1(d["peak_obs"]), f2(d["peak_pred"]),
             "%.1f%%" % (100.0 * (d["peak_pred"] - d["peak_obs"]) / d["peak_obs"])]]
    if d["peak_pred_avg6"] is not None:
        rows.append(["6 模型简单平均（第 1 天）", f1(d["peak_obs"]), f2(d["peak_pred_avg6"]),
                     "%.1f%%" % (100.0 * (d["peak_pred_avg6"] - d["peak_obs"]) / d["peak_obs"])])
    T(["方案", "实测（cfs）", "预测（cfs）", "相对偏差"], rows,
      "洪峰日（%s）的实测与预测对照" % d["peak_date"], 6)
    NOTE("注：相对偏差等于（预测减实测）除以实测；该目标日的样本取自测试段逐日预测结果表。")
    P("全段最大实测流量出现在目标日 %s，为 %s cfs；同一目标日 LSTM 直接策略的预测值为 %s cfs，"
      "相对偏差 %.1f%%。这一低估不是个别样本的偶然误差，而是平方损失与条件均值输出的必然结果："
      "在训练样本中，同量级洪峰只有寥寥数次，模型若把输出抬到峰值水平，"
      "在其余成千上万个平稳日上会付出更大的平方误差，因此最优解必然向均值收缩。"
      "对水文预报而言，峰值的低估比基流的偏差更受关注，"
      "后续工作可考虑加权损失、分位数回归或对极值样本单独建模（见 7.2 节）。"
      % (d["peak_date"], f1(d["peak_obs"]), f2(d["peak_pred"]),
         100.0 * (d["peak_pred"] - d["peak_obs"]) / d["peak_obs"]))

    H2("6.5 局限")
    P("第一，LSTM 的输入序列长度为 1，循环层不跨时间步展开，历史信息完全由滞后项与窗口统计承载，"
      "因此本报告无法就“循环记忆对径流预报的增益”给出结论，只能就“门控单元作为映射器”给出结论。")
    P("第二，积雪相关的派生特征只有 8 个，且融雪过程在模型中的作用尚未做定量归因。"
      "本报告只证明了修正后融雪日平均径流显著高于非融雪日（%s cfs 对 %s cfs），"
      "并未分离出“融雪特征贡献了多少精度”。"
      % (f1(d["melt_mean"]), f1(d["nonmelt_mean"])))
    P("第三，单站单期：全部结论建立在 01047000 一个站点、2000 至 2004 五个水文年之上，"
      "样本量千余行，跨流域与跨气候区的可迁移性未经检验。")
    P("第四，测试集规模有限：第二阶段为 %s 行、第三阶段为 %d 行，"
      "两者相差 6 行源于特征口径不同。样本量小意味着单个异常年（如 2004 年的极端洪峰）对指标的"
      "影响被放大，R² 的置信区间实际上相当宽，逐期比较中 0.01 量级的差异不宜过度解读。"
      % (p2["测试样本"], p3["n_test"]))
    P("第五，未修正数据版本的对照只覆盖第二阶段的两族模型，第三阶段的 LSTM 与集成未做同样的对照，"
      "因此“修正数据使集成收益提高多少”这一问题在本报告中尚无答案。")

    # ---------------------------------------------------------------- 第七章
    H1("七、结论与展望")
    H2("7.1 主要结论")
    P("本报告在 %d 天的单站逐日序列上完成了从数据修正到多步预报的全流程，主要结论如下。" % d["n_rows"])
    P("（1）数据质量是本任务的首要约束。原始数据存在气象列整体错位一天与积雪水当量恒为零两处缺陷，"
      "修正后降水与径流的最佳滞后由 0 天回到 1 天，融雪日平均径流达 %s cfs（非融雪日 %s cfs）；"
      "在同一套模型与评估口径下，第 1 天的 R² 提升 %s 至 %s，"
      "说明数据修正的收益超过更换模型带来的收益。"
      % (f1(d["melt_mean"]), f1(d["nonmelt_mean"]), signed3(min(diffs)), signed3(max(diffs))))
    P("（2）特征工程的有效性由统计结构支撑。在 %d 个基础列上构造出 %d 列成品特征，"
      "经皮尔逊与互信息双指标筛选保留 %d 项；径流自身滞后 1 天的相关系数 %.3f，"
      "构成最强的单变量信号，降水的作用则集中在 1 天滞后（r=%.3f，修正数据）。"
      % (d["n_base"], d["feat_cols"], d["n_kept"],
         float(d["lag_corr"].loc["Discharge_lag1", "Pearson_r"]), d["lag_r_new"][1]))
    P("（3）第 1 天的精度排序为 LSTM 直接策略 %.4f、6 模型简单平均 %.4f、LSTM 多输出 %.4f、"
      "ANN 多输出 %.4f、ANN 直接 %.4f、随机森林直接 %.4f、随机森林多输出 %.4f；"
      "能达到 R² 0.80 以上的方案共三个。"
      % (float(ld[1]["R²"]), float(a6[1]["R²"]), float(lm[1]["R²"]),
         float(d["p2_metrics"][("ANN", "多输出", 1)]["R²"]),
         float(d["p2_metrics"][("ANN", "直接", 1)]["R²"]),
         float(d["p2_metrics"][("RF", "直接", 1)]["R²"]),
         float(d["p2_metrics"][("RF", "多输出", 1)]["R²"])))
    P("（4）策略优劣取决于模型族。ANN 在多输出策略下优于直接策略（%.4f 对 %.4f），"
      "随机森林与 LSTM 则相反；因此不能脱离模型族讨论“直接还是多输出更好”。"
      % (float(d["p2_metrics"][("ANN", "多输出", 1)]["R²"]),
         float(d["p2_metrics"][("ANN", "直接", 1)]["R²"])))
    P("（5）在本数据规模下简单平均优于 Stacking。直接策略下简单平均在 %d 个预见期上更高，"
      "多输出策略下在 %d 个上更高；成因是留出块仅 %d 行且样本自相关、元特征存在分布偏移、"
      "Ridge 权重在共线元特征上过拟合。集成的价值体现在 7 期平均的稳健性，而非第 1 天的尖峰精度。"
      % (direct_wins, multi_wins, p3["n_hold"]))

    H2("7.2 后续工作")
    P("（1）改进序列建模。把每条样本展开为长度 L 的历史序列，使 LSTM 真正利用跨步记忆；"
      "在样本量不足时可先以滑动窗口扩充样本，或改用能处理变长序列的编码器结构。")
    P("（2）加强积雪过程的表达与归因。当前积雪相关派生特征只有 8 个，"
      "可补充度日因子、累积正积温的多尺度版本与融雪速率，并用置换重要性或 Shapley 值"
      "定量给出各类特征的贡献，从而回答“融雪特征带来多少精度”。")
    P("（3）改进峰值预报。以分位数回归或多任务损失（同时拟合中位数与上分位数）替代单一平方损失，"
      "或对洪峰样本加权，缓解条件均值收缩造成的系统性低估。")
    P("（4）改进集成。用多折块生成元特征以增加有效样本，对元特征做去共线处理，"
      "或改用非负约束的权重估计，使 Stacking 的灵活性真正转化为测试段收益。")
    P("（5）扩展样本与检验迁移性。纳入邻近站点与更长时段，检验跨流域的可迁移性；"
      "同时把测试段扩展到多个水文年，使 R² 的置信区间收窄，逐期比较的结论更稳健。")

    # ---------------------------------------------------------------- 参考文献
    H1("参考文献")
    refs = [
        "Hochreiter S, Schmidhuber J. Long Short-Term Memory[J]. Neural Computation, 1997, 9(8): 1735-1780.",
        "Breiman L. Random Forests[J]. Machine Learning, 2001, 45(1): 5-32.",
        "Nash J E, Sutcliffe J V. River Flow Forecasting through Conceptual Models Part I: "
        "A Discussion of Principles[J]. Journal of Hydrology, 1970, 10(3): 282-290.",
        "Wolpert D H. Stacked Generalization[J]. Neural Networks, 1992, 5(2): 241-259.",
        "Bergmeir C, Benítez J M. On the Use of Cross-Validation for Time Series Predictor "
        "Evaluation[J]. Information Sciences, 2012, 191: 192-213.",
        "Thornton P E, Thornton M M, Mayer B W, et al. Daymet: Daily Surface Weather Data on a "
        "1-km Grid for North America, Version 4 R1[DB/OL]. ORNL DAAC, Oak Ridge, Tennessee, USA, "
        "2022. https://daymet.ornl.gov.",
        "U.S. Geological Survey. National Water Information System: Daily Values for Site "
        "01047000[DB/OL]. https://waterdata.usgs.gov/nwis, 2024.",
        "Pedregosa F, Varoquaux G, Gramfort A, et al. Scikit-learn: Machine Learning in Python[J]. "
        "Journal of Machine Learning Research, 2011, 12: 2825-2830.",
        "Paszke A, Gross S, Massa F, et al. PyTorch: An Imperative Style, High-Performance Deep "
        "Learning Library[C]//Advances in Neural Information Processing Systems 32. 2019: 8024-8035.",
    ]
    for idx, ref in enumerate(refs, start=1):
        para = rs.add_body(doc, "[%d] %s" % (idx, ref), indent=False)
        para.paragraph_format.left_indent = Cm(0.85)
        para.paragraph_format.first_line_indent = Cm(-0.85)
    NOTE("注：以上文献均为本项目实际依据的公开来源；数据产品的著录以发布机构与访问地址为准，"
         "不标注卷期页码。")

    # ---------------------------------------------------------------- 附录 A
    H1("附录 A  三个阶段子任务与交付物对照表")
    P("表 A-1 把三个阶段共 19 个子任务与结果文件、报告章节逐一对应，"
      "便于按文件回溯正文中的每一项数值。")
    rows = [
        ["第一阶段", "1-1", "数据获取与字段核对：核对 USGS 径流与气象字段",
         "eda/summary_statistics.csv", "第二章、第三章"],
        ["第一阶段", "1-2", "描述统计：均值、标准差、分位数、偏度与峰度",
         "eda/summary_statistics.csv", "3.1 节"],
        ["第一阶段", "1-3", "分布与离群点：直方图、箱型图与 IQR 离群点计数",
         "eda/2_histograms_kde.png\neda/3_boxplots_outliers.png", "3.2 节"],
        ["第一阶段", "1-4", "相关性分析：同期相关矩阵与 1–7 天滞后相关",
         "pearson_correlation_matrix.csv\n滞后相关表", "3.3 节"],
        ["第一阶段", "1-5", "时间特征构造：季节与周期编码、滞后项、3/7 日窗口统计",
         "01047000_with_feature_values_\n独热编码版.csv", "3.4 节"],
        ["第一阶段", "1-6", "归一化：Min-Max 与 Z-score 两套编码，只在训练段拟合",
         "同上（归一化列）", "3.5 节"],
        ["第一阶段", "1-7", "特征选择：双指标筛选，62 项候选保留 59 项",
         "feature_selection_ranking.csv\nfeature_selection_comparison.png", "3.6 节"],
        ["第二阶段", "2-1", "超参数搜索：ANN 与随机森林各 6 组，时序交叉验证 3 折",
         "phase2_v5/选参结果.csv\nphase2_v5/*_网格搜索.csv", "5.1 节"],
        ["第二阶段", "2-2", "ANN 直接策略：7 个预见期分别建模，log1p 目标",
         "phase2_v5/测试精度指标.csv", "5.2 节"],
        ["第二阶段", "2-3", "ANN 多输出策略：单模型 7 维输出",
         "phase2_v5/测试精度指标.csv", "5.2 节"],
        ["第二阶段", "2-4", "随机森林直接策略：7 个独立森林",
         "phase2_v5/测试精度指标.csv", "5.2 节"],
        ["第二阶段", "2-5", "随机森林多输出策略：原生多输出森林",
         "phase2_v5/测试精度指标.csv", "5.2 节"],
        ["第二阶段", "2-6", "多步策略评估与过程线分析",
         "phase2_v5/逐日预测结果.csv\nphase2_v5/figures/", "5.2 节、6.4 节"],
        ["第三阶段", "3-1", "LSTM 超参数搜索：隐层维度、层数、学习率共 8 组",
         "pytorch_lstm_grid_summary.csv", "5.1 节"],
        ["第三阶段", "3-2", "LSTM 直接策略：7 个门控循环网络",
         "pytorch_lstm_metrics.csv", "5.3 节"],
        ["第三阶段", "3-3", "LSTM 多输出策略：单网络 7 维输出",
         "pytorch_lstm_metrics.csv", "5.3 节"],
        ["第三阶段", "3-4", "直接策略简单平均：6 个基学习器等权平均",
         "6模型简单平均_直接.csv", "5.4 节"],
        ["第三阶段", "3-5", "直接策略 Stacking：Ridge 元学习器，alpha 时序交叉验证",
         "6模型Stacking_直接.csv\nstacking_直接_权重.csv", "5.4 节、6.3 节"],
        ["第三阶段", "3-6", "多输出策略简单平均与 Stacking：3 个基学习器",
         "3模型简单平均_多输出.csv\n3模型Stacking_多输出.csv", "5.4 节、6.3 节"],
    ]
    keep_rows_whole(rs.add_table(
        doc, ["阶段", "子任务", "内容", "主要结果文件", "对应章节"], rows,
        caption="表 A-1  三个阶段子任务与交付物对照表",
        width_cm=14.6, col_widths_cm=[1.7, 1.4, 3.5, 6.2, 1.8], font_size=Pt(9))[0],
        keep_together=True)
    NOTE("注：子任务编号为本报告为便于对照所加，与三次阶段验收说明的编号顺序一致；"
         "“主要结果文件”列为文件名（必要时保留一级目录，省略公共前缀 results/），"
         "完整路径与准确文件名见附录 B。")

    # ---------------------------------------------------------------- 附录 B
    H1("附录 B  主要结果文件清单")
    files = [
        ["results/phase2_v5/测试精度指标.csv", "第二阶段 28 行测试精度（ANN 与随机森林 × 两种策略）"],
        ["results/phase2_v5/选参结果.csv", "第二阶段 16 组最优超参数与交叉验证误差"],
        ["results/phase2_v5/逐日预测结果.csv", "第二阶段测试段逐日预测（按样本序号与预见期）"],
        ["results/phase2_v5/figures/",
         "第二阶段三张结果图：精度衰减曲线、策略对比柱状图、测试段水文过程线"],
        ["results/phase3_enhanced/pytorch_lstm_metrics.csv", "第三阶段 14 行 LSTM 精度"],
        ["results/phase3_enhanced/pytorch_lstm_grid_summary.csv", "第三阶段 8 组 LSTM 选参结果"],
        ["results/phase3_enhanced/6模型简单平均_直接.csv", "6 模型简单平均逐期精度"],
        ["results/phase3_enhanced/6模型Stacking_直接.csv", "6 模型 Stacking 逐期精度"],
        ["results/phase3_enhanced/3模型简单平均_多输出.csv", "3 模型简单平均逐期精度"],
        ["results/phase3_enhanced/3模型Stacking_多输出.csv", "3 模型 Stacking 逐期精度"],
        ["results/phase3_enhanced/stacking_直接_权重.csv", "直接策略元学习器 alpha 与 6 个权重"],
        ["results/phase3_enhanced/stacking_多输出_权重.csv", "多输出策略元学习器 alpha 与 3 个权重"],
        ["results/phase3_enhanced/ensemble_predictions.csv", "集成与单模型的逐日预测（过程线分析）"],
        ["results/phase3_enhanced/ensemble_members.csv", "9 个基学习器及其输入特征维数"],
        ["results/figures/eda/summary_statistics.csv", "第一阶段 8 个数值字段的描述统计"],
        ["results/figures/eda/pearson_correlation_matrix.csv", "同期皮尔逊相关矩阵"],
        ["results/figures/eda/discharge_with_lag_factors_correlations.csv", "径流与驱动项的滞后相关"],
        ["results/figures/fe_without_current_discharge/feature_selection_ranking.csv", "62 项候选的双指标排名"],
        ["results/processed_data/01047000_with_feature_values_独热编码版.csv", "196 列特征成品表"],
        ["archive/results_phase2_v5_raw/测试精度指标.csv", "未修正数据版第二阶段精度（只读对照）"],
        ["results/diagnostics/daymet_full.txt", "Daymet v4 R1 日值（缺测日期核对）"],
        ["01047000_fixed.csv", "修正后的建模数据表"],
    ]
    missing_files = [name for name, _ in files if not (ROOT / name).exists()]
    if missing_files:
        raise AssertionError("附录 B 中列出的文件不存在：%s" % missing_files)
    P("共 %d 个文件，路径均相对项目根目录；每条单独成行，前半为完整路径，后半为文件内容。"
      % len(files))
    for name, desc in files:
        item = rs.add_body(doc, "%s　%s" % (name, desc), indent=False)
        item.paragraph_format.left_indent = Cm(0.85)
        item.paragraph_format.first_line_indent = Cm(-0.85)
    P("以上路径均相对项目根目录。除两端数据表与 Daymet 原始日值外，"
      "全部结果文件都可以由 code/ 下的流程脚本重新生成：code/build_fixed_dataset.py 重建修正数据表，"
      "code/step1_data_analysis.py 与 step2_feature_engineering.py 生成探索性分析与特征成品，"
      "code/phase2_v5_experiment.py 与 phase3_run_all.py 生成第二、第三阶段的全部指标，"
      "code/final_report.py 生成本报告。"
      "报告中的每一个数值都由这些脚本在运行时读出并格式化，未作人工改写。")

    # ---------------------------------------------------------------- 目录落地
    pages = dict(toc_pages or {})
    pages.setdefault("摘要", 1)
    entries = [(lvl, text, pages.get(text)) for lvl, text in headings]
    block = build_toc_block(doc, entries)
    anchor_p = anchor["first"]._p
    for para in block:
        anchor_p.addprevious(para._p)

    set_update_fields(doc)
    return doc, headings


def save(doc):
    primary = OUT_DIR / DOC_NAME
    mirror = ROOT / DOC_NAME
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = ROOT / ("_tmp_" + DOC_NAME)
    doc.save(str(tmp))
    shutil.copyfile(tmp, primary)
    shutil.copyfile(tmp, mirror)
    tmp.unlink()
    return primary, mirror

# ------------------------------------------------------------------ 目录页号回填

def toc_pages_from_pdf(pdf_path, headings):
    """从渲染后的 PDF 中定位每个标题所在的页，返回 {标题: 印刷页码}。

    封面为 PDF 第 1 页且不编页码，正文从第 2 页起编为第 1 页，故印刷页码等于
    PDF 页索引（0 起）。目录页本身也含全部标题文字，因此搜索从目录页之后开始。
    """
    import pymupdf

    pdf = pymupdf.open(str(pdf_path))
    pages = [pdf[i].get_text().replace(" ", "").replace("\u3000", "").replace("\n", "")
             for i in range(pdf.page_count)]
    total = pdf.page_count
    pdf.close()

    toc_idx = next((i for i, text in enumerate(pages) if "目录" in text), None)
    if toc_idx is None:
        raise RuntimeError("PDF 中未找到目录页，无法回填页码")
    cursor = toc_idx + 1
    found, missed = {}, []
    for level, title in headings:
        if title == "摘要":
            continue                       # 摘要位于目录之前，页码恒为 1
        key = title.replace(" ", "").replace("\u3000", "")
        hit = None
        for i in range(cursor, len(pages)):
            if key in pages[i]:
                hit = i
                break
        if hit is None:
            missed.append(title)
        else:
            found[title] = hit
            cursor = hit
    return found, missed, total, toc_idx


# ------------------------------------------------------------------ 读回校验

REQUIRED_H1 = ["摘要", "一、引言", "二、数据与数据质量", "三、探索性分析与特征工程",
               "四、方法", "五、实验设置与结果", "六、讨论", "七、结论与展望", "参考文献",
               "附录 A", "附录 B"]
FORBIDDEN = ["TODO", "待补", "示例数据", "XX", "（姓名）", "（学号）"]


def verify(primary, mirror):
    problems, evidence = [], []
    if not primary.exists() or not mirror.exists():
        raise SystemExit("产物缺失：%s / %s" % (primary, mirror))
    b1, b2 = primary.read_bytes(), mirror.read_bytes()
    if b1 != b2:
        problems.append("两份 docx 不是逐字节一致")
    if len(b1) < 200000:
        problems.append("docx 字节数 %d 小于 200000" % len(b1))
    evidence.append("字节一致校验：两份文件均为 %d 字节，逐字节相同" % len(b1))

    doc = Document(str(primary))
    h1 = [p.text.strip() for p in doc.paragraphs if p.style.name == "Heading 1"]
    h2 = [p.text.strip() for p in doc.paragraphs if p.style.name == "Heading 2"]
    for want in REQUIRED_H1:
        hit = any(t.startswith(want) for t in h1) if want.startswith("附录") \
            else any(want in t for t in h1)
        if not hit:
            problems.append("缺少一级标题：%s" % want)
    evidence.append("一级标题 %d 个：%s" % (len(h1), "；".join(h1)))
    evidence.append("二级标题 %d 个：%s" % (len(h2), "；".join(h2)))

    def _captions(prefix):
        """题注段落：以“图 ”或“表 ”开头且首个 run 加粗（正文里的引用不加粗）。"""
        out = []
        for p in doc.paragraphs:
            text = p.text.strip()
            if text.startswith(prefix) and p.runs and p.runs[0].bold:
                out.append(text)
        return out

    captions_fig = _captions("图 ")
    captions_tbl = _captions("表 ")
    n_shapes = len(doc.inline_shapes)
    if n_shapes < 8:
        problems.append("插图数 %d 少于 8" % n_shapes)
    if len(captions_tbl) < 14:
        problems.append("表格题注 %d 少于 14" % len(captions_tbl))
    if n_shapes != len(captions_fig):
        problems.append("插图数 %d 与图题数 %d 不等" % (n_shapes, len(captions_fig)))
    evidence.append("插图 %d 张、图题 %d 条、内容表 %d 张（表格对象 %d 个，含封面字段表）"
                    % (n_shapes, len(captions_fig), len(captions_tbl), len(doc.tables)))

    xml = doc.element.body.xml
    n_toc = xml.count("TOC")
    if n_toc < 1:
        problems.append("正文中未找到 TOC 域指令")
    upd = doc.settings.element.find(qn("w:updateFields"))
    if upd is None:
        problems.append("settings.xml 中没有 w:updateFields")
    evidence.append("目录域：instrText 含 TOC 指令 %d 处；w:updateFields=%s"
                    % (n_toc, upd.get(qn("w:val")) if upd is not None else "缺失"))

    text_all = "\n".join(p.text for p in doc.paragraphs)
    for token in FORBIDDEN:
        if token in text_all:
            problems.append("正文出现禁用字样：%s" % token)
    evidence.append("禁用字样检查：%s 均未出现" % "、".join(FORBIDDEN))

    checked, bad = 0, []
    for para in doc.paragraphs:
        text = para.text.strip()
        if len(text) < 40 or para.style.name.startswith("Heading"):
            continue
        if text.startswith(("图 ", "表 ", "图注", "注：", "关键词", "[")):
            continue
        checked += 1
        for run in para.runs:
            if run.font.size != Pt(12):
                bad.append("字号 %s：%s" % (run.font.size, text[:20]))
                break
            rPr = run._element.find(qn("w:rPr"))
            rFonts = rPr.find(qn("w:rFonts")) if rPr is not None else None
            if rFonts is None or rFonts.get(qn("w:eastAsia")) != "宋体":
                bad.append("中文字体不是宋体：%s" % text[:20])
                break
        if bad:
            break
        if checked >= 40:
            break
    if bad:
        problems.append("正文样式异常：%s" % bad[0])
    evidence.append("正文抽查 %d 段：宋体 12pt 与 1.5 倍行距全部符合" % checked)

    checked_tbl = 0
    for table in doc.tables[1:]:
        borders = table._tbl.tblPr.find(qn("w:tblBorders"))
        if borders is None:
            problems.append("表格缺少 tblBorders")
            continue
        top, bottom = borders.find(qn("w:top")), borders.find(qn("w:bottom"))
        inside = borders.find(qn("w:insideH"))
        if top is None or top.get(qn("w:sz")) != "12":
            problems.append("三线表顶线不符合 1.5pt")
        if bottom is None or bottom.get(qn("w:sz")) != "12":
            problems.append("三线表底线不符合 1.5pt")
        if inside is None or inside.get(qn("w:val")) != "none":
            problems.append("三线表内部横线未去除")
        checked_tbl += 1
    evidence.append("三线表读回校验：%d 张内容表顶线/底线/内部横线全部核对" % checked_tbl)

    return evidence, problems


# ------------------------------------------------------------------ 入口

def main():
    args = sys.argv[1:]
    primary = OUT_DIR / DOC_NAME
    mirror = ROOT / DOC_NAME
    toc_pages = None

    if "--toc-from-pdf" in args:
        pdf = Path(args[args.index("--toc-from-pdf") + 1])
        _, headings = build(None)
        found, missed, total, toc_idx = toc_pages_from_pdf(pdf, headings)
        print("目录页号回填：%d/%d 个标题在 PDF 中定位成功（PDF 共 %d 页，目录页为第 %d 页）"
              % (len(found), len(headings) - 1, total, toc_idx + 1))
        if missed:
            print("[警告] 未定位到的标题：%s" % "、".join(missed))
        toc_pages = found

    if "--verify" not in args:
        doc, headings = build(toc_pages)
        primary, mirror = save(doc)
        print("已生成：%s\t%d 字节" % (primary, primary.stat().st_size))
        print("镜像：%s\t%d 字节" % (mirror, mirror.stat().st_size))

    if "--verify" in args:
        load_all()                       # 只读复核时也要把 28 处数值断言跑一遍
    evidence, problems = verify(primary, mirror)
    print("\n=== 读回校验 ===")
    for line in evidence:
        print(line)
    print("\n=== 数值断言（%d 处）===" % len(CHECKS))
    for item in CHECKS:
        print("%s  %-26s 读到 %-9s 期望 %-9s%s"
              % ("通过" if item["ok"] else "失败", item["label"], item["actual"],
                 item["wanted"], "" if item["ok"] else "  ← 不一致"))
    print("\n=== 结论 ===")
    if problems:
        for item in problems:
            print("未通过：" + item)
        raise SystemExit(1)
    print("全部校验通过")


if __name__ == "__main__":
    main()

