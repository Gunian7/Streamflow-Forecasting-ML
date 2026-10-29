"""Generate a day-1 accuracy ranking for the enhanced phase-3 models."""
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "phase3_enhanced"
TARGET_R2 = 0.80
BASELINE_MODEL = "PseudoLSTM"


def _read_csv(path):
    if not path.exists():
        return None
    return pd.read_csv(path, encoding="utf-8-sig")


def _first_row(frame, filters, source):
    if frame is None:
        return None
    matches = frame
    for column, value in filters.items():
        if column not in matches.columns:
            raise ValueError(f"{source} 缺少必要字段: {column}")
        if isinstance(value, (int, float)):
            numeric = pd.to_numeric(matches[column], errors="coerce")
            matches = matches[numeric == value]
        else:
            matches = matches[matches[column].astype(str) == str(value)]
    if matches.empty:
        return None
    if len(matches) > 1:
        raise ValueError(f"{source} 中第1天结果不唯一: {filters}")
    row = matches.iloc[0]
    missing = [name for name in ("R²", "RMSE", "MAE") if name not in row.index]
    if missing:
        raise ValueError(f"{source} 缺少指标字段: {', '.join(missing)}")
    return row


def _record(model, model_type, row, source):
    return {
        "模型": model,
        "类型": model_type,
        "R²": float(row["R²"]) if row is not None else float("nan"),
        "RMSE": float(row["RMSE"]) if row is not None else float("nan"),
        "MAE": float(row["MAE"]) if row is not None else float("nan"),
        "数据状态": "可用" if row is not None else f"缺失: {source}",
    }


def build_ranking():
    pseudo_path = ROOT / "results" / "phase3_lstm" / "lstm_metrics.csv"
    pytorch_path = OUT / "pytorch_lstm_metrics.csv"
    boosting_path = OUT / "boosting_metrics.csv"
    ensemble_path = OUT / "enhanced_ensemble_metrics.csv"

    pseudo = _read_csv(pseudo_path)
    pytorch = _read_csv(pytorch_path)
    boosting = _read_csv(boosting_path)
    ensemble = _read_csv(ensemble_path)

    rows = [
        _record(
            "PseudoLSTM",
            "原始基线",
            _first_row(pseudo, {"策略": "直接", "预见期（天）": 1}, str(pseudo_path)),
            str(pseudo_path),
        ),
        _record(
            "PyTorch-LSTM",
            "真实深度学习",
            _first_row(
                pytorch,
                {"策略": "PyTorch-LSTM-直接", "预见期（天）": 1},
                str(pytorch_path),
            ),
            str(pytorch_path),
        ),
        _record(
            "XGBoost",
            "梯度提升树",
            _first_row(boosting, {"模型": "XGBoost", "预见期（天）": 1}, str(boosting_path)),
            str(boosting_path),
        ),
        _record(
            "LightGBM",
            "梯度提升树",
            _first_row(boosting, {"模型": "LightGBM", "预见期（天）": 1}, str(boosting_path)),
            str(boosting_path),
        ),
        _record(
            "6模型 Stacking",
            "增强集成",
            _first_row(ensemble, {"预见期（天）": 1}, str(ensemble_path)),
            str(ensemble_path),
        ),
    ]
    ranking = pd.DataFrame(rows)

    baseline_rows = ranking.loc[ranking["模型"] == BASELINE_MODEL, "R²"]
    baseline = baseline_rows.iloc[0]
    if pd.isna(baseline):
        raise ValueError(f"基线模型 {BASELINE_MODEL} 的第1天 R² 缺失，无法计算提升幅度")

    ranking["绝对提升ΔR²"] = ranking["R²"] - baseline
    ranking["相对提升（%）"] = (ranking["绝对提升ΔR²"] / abs(baseline)) * 100
    ranking["距目标差值"] = (TARGET_R2 - ranking["R²"]).clip(lower=0)
    ranking["达标状态"] = ranking["R²"].map(
        lambda value: "✅ 达标" if value >= TARGET_R2 else "⚠️ 未达标"
    )
    ranking.loc[ranking["R²"].isna(), "达标状态"] = "无法判定（指标缺失）"
    ranking = ranking.sort_values("R²", ascending=False, na_position="last").reset_index(drop=True)
    ranking.insert(
        0,
        "排名",
        [None if pd.isna(value) else index + 1 for index, value in enumerate(ranking["R²"])],
    )
    return ranking


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ranking = build_ranking()
    output_path = OUT / "精度排行榜_第1天.csv"
    ranking.to_csv(output_path, index=False, encoding="utf-8-sig")

    available = ranking.dropna(subset=["R²"])
    print("=" * 94)
    print("第三阶段精度提升总结（第1天）")
    print("=" * 94)
    display = ranking.copy()
    display["排名"] = display["排名"].map(lambda value: "" if pd.isna(value) else str(int(value)))
    print(display.to_string(index=False, na_rep="N/A", float_format=lambda value: f"{value:.4f}"))

    baseline = ranking.loc[ranking["模型"] == BASELINE_MODEL].iloc[0]
    if not available.empty:
        best = available.iloc[0]
        print("\n提升幅度（相对 PseudoLSTM 基线）:")
        print(f"  基线 R²: {baseline['R²']:.4f}")
        print(f"  当前已知最优: {best['模型']}，R²={best['R²']:.4f}")
        print(f"  绝对提升: ΔR²={best['绝对提升ΔR²']:+.4f}")
        print(f"  相对提升: {best['相对提升（%）']:+.2f}%")

    missing_count = int(ranking["R²"].isna().sum())
    if missing_count:
        print(f"\n⚠️ 有 {missing_count} 个模型缺少第1天指标；全模型目标状态无法最终判定。")
    else:
        best_r2 = float(available.iloc[0]["R²"])
        if best_r2 >= TARGET_R2:
            print(f"\n✅ 目标达成！最优 R²={best_r2:.4f} ≥ {TARGET_R2:.2f}")
        else:
            print(f"\n⚠️ 目标未达成，距离目标还差 {TARGET_R2 - best_r2:.4f} R²。")
    print(f"\n排行榜已保存: {output_path}")


if __name__ == "__main__":
    main()
