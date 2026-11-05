# -*- coding: utf-8 -*-
"""第二阶段 3 张中文图表（数据全部读自 results/phase2_v5/*.csv）。"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
P2 = ROOT / "results" / "phase2_v5"
FIG = P2 / "figures"
FIG.mkdir(parents=True, exist_ok=True)
TEST_START = pd.Timestamp("2004-01-01")


def setup_font():
    from matplotlib import font_manager
    installed = {f.name for f in font_manager.fontManager.ttflist}
    chosen = next((c for c in ["SimSun", "Microsoft YaHei", "SimHei"] if c in installed), "DejaVu Sans")
    plt.rcParams["font.sans-serif"] = [chosen, "Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.unicode_minus"] = False
    legend_family = "Microsoft YaHei" if (chosen == "SimSun" and "Microsoft YaHei" in installed) else chosen
    print(f"[字体] 正文 {chosen} / 图例 {legend_family}", flush=True)
    return legend_family


def main():
    legend_family = setup_font()
    met = pd.read_csv(P2 / "测试精度指标.csv", encoding="utf-8-sig")
    if "R\u00b2" not in met.columns:
        print("列名:", list(met.columns))

    def series(model, strategy, col="R\u00b2"):
        d = met[(met["模型"] == model) & (met["策略"] == strategy)].sort_values("预见期（天）")
        return d["预见期（天）"].to_numpy(), d[col].to_numpy()

    # 图1 精度随预见期衰减
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    for model, strategy, color, marker, ls in [
            ("ANN", "直接", "#1f4e79", "o", "-"), ("ANN", "多输出", "#2e75b6", "s", "--"),
            ("RF", "直接", "#843c0c", "^", "-"), ("RF", "多输出", "#c55a11", "D", "--")]:
        x, y = series(model, strategy)
        ax.plot(x, y, marker=marker, linestyle=ls, color=color, lw=1.8, ms=6,
                label=f"{model} {strategy}策略")
    ax.axhline(0, color="0.45", lw=1.0, ls=":")
    ax.set_xlabel("预见期（天）", fontsize=12)
    ax.set_ylabel("决定系数 R\u00b2", fontsize=12)
    ax.set_title("图1  测试集（2004 年）R\u00b2 随预见期衰减", fontsize=14, fontweight="bold")
    ax.set_xticks(list(range(1, 8)))
    ax.grid(alpha=0.3, ls="--")
    ax.legend(prop={"family": legend_family, "size": 10}, frameon=False)
    fig.tight_layout()
    p1 = FIG / "fig1_精度随预见期衰减.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)

    # 图2 策略对比（第1天 / 第7天）
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    groups = [1, 7]
    combos = [("ANN", "直接", "#1f4e79"), ("ANN", "多输出", "#5b9bd5"),
              ("RF", "直接", "#843c0c"), ("RF", "多输出", "#ed7d31")]
    width = 0.2
    for i, (model, strategy, color) in enumerate(combos):
        vals = [float(met[(met["模型"] == model) & (met["策略"] == strategy)
                          & (met["预见期（天）"] == h)]["R\u00b2"].iloc[0]) for h in groups]
        pos = [g + (i - 1.5) * width for g in groups]
        bars = ax.bar(pos, vals, width=width, color=color, edgecolor="white", lw=0.6,
                      label=f"{model} {strategy}策略")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + (0.008 if v >= 0 else -0.008),
                    f"{v:.4f}", ha="center", va="bottom" if v >= 0 else "top",
                    rotation=90, fontsize=8)
    ax.axhline(0, color="0.3", lw=0.9)
    ax.set_xticks(groups)
    ax.set_xticklabels(["第 1 天预见期", "第 7 天预见期"], fontsize=12)
    ax.set_ylabel("决定系数 R\u00b2", fontsize=12)
    ttl_max = float(met["R\u00b2"].max())
    ax.set_ylim(min(-0.05, float(met["R\u00b2"].min()) - 0.06), ttl_max + 0.10)
    ax.set_title("图2  第 1 天与第 7 天各模型/策略的 R\u00b2 对比", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, ls="--")
    ax.legend(prop={"family": legend_family, "size": 9}, ncol=2, frameon=False, loc="upper right")
    fig.tight_layout()
    p2 = FIG / "fig2_策略对比.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)

    # 图3 2004 年水文过程线
    det = pd.read_csv(P2 / "逐日预测结果.csv", encoding="utf-8-sig")
    det["日期"] = TEST_START + pd.to_timedelta(det["样本序号"] + det["预见期（天）"], "D")
    def pick(model, strategy, h):
        d = det[(det["模型"] == model) & (det["策略"] == strategy) & (det["预见期（天）"] == h)]
        return d.sort_values("样本序号")
    ann1 = pick("ANN", "直接", 1)
    rf1 = pick("RF", "直接", 1)
    fig, ax = plt.subplots(figsize=(12.5, 5.0))
    ax.plot(ann1["日期"], ann1["实测径流（cfs）"], color="black", lw=1.8, label="实测径流")
    ax.plot(ann1["日期"], ann1["预测径流（cfs）"], color="#1f4e79", lw=1.4, ls="--", label="ANN 直接策略 第1天预测")
    ax.plot(rf1["日期"], rf1["预测径流（cfs）"], color="#c55a11", lw=1.2, ls="-.", label="RF 直接策略 第1天预测")
    ax.set_xlabel("日期（2004 年测试段）", fontsize=12)
    ax.set_ylabel("日径流量（cfs）", fontsize=12)
    ax.set_title("图3  2004 年测试段实测与第 1 天预测水文过程线", fontsize=14, fontweight="bold")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=15))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)
    ax.grid(alpha=0.3, ls="--")
    ax.legend(prop={"family": legend_family, "size": 10}, frameon=False, loc="upper left")
    fig.tight_layout()
    p3 = FIG / "fig3_2004年水文过程线.png"
    fig.savefig(p3, dpi=150)
    plt.close(fig)

    for p in (p1, p2, p3):
        print(f"[图] {p}  {p.stat().st_size} 字节", flush=True)


if __name__ == "__main__":
    main()
