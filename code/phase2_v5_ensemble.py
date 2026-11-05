"""v5 等权融合评估（固定规则，不在测试集上调权重）"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'results/phase2_v5'


def metrics(y, p):
    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return {'RMSE（cfs）': float(np.sqrt(np.mean((y - p) ** 2))),
            'MAE（cfs）': float(np.mean(np.abs(y - p))),
            'R²': 1 - ss_res / ss_tot,
            'NSE': 1 - ss_res / ss_tot}


def main():
    det = pd.read_csv(OUT / '逐日预测结果.csv')
    rows = []

    schemes = {
        'ANN两策略等权': [('ANN', '直接'), ('ANN', '多输出')],
        '四模型等权': [('ANN', '直接'), ('ANN', '多输出'), ('RF', '直接'), ('RF', '多输出')],
        'ANN多输出+RF多输出': [('ANN', '多输出'), ('RF', '多输出')],
    }

    for h in range(1, 8):
        sub = det[det['预见期（天）'] == h]
        piv = sub.pivot_table(index=['模型', '策略'], columns='样本序号',
                              values='预测径流（cfs）')
        y = sub.groupby('样本序号')['实测径流（cfs）'].first().sort_index().to_numpy(float)

        for name, members in schemes.items():
            P = np.array([piv.loc[m].to_numpy(float) for m in members])
            pred = P.mean(axis=0)
            m = metrics(y, pred)
            rows.append({'融合方案': name, '预见期（天）': h, **m})
            print(f'第{h}天 {name:18s} NSE={m["NSE"]:+.4f}  RMSE={m["RMSE（cfs）"]:.1f}')

    df = pd.DataFrame(rows)
    df.to_csv(OUT / '融合评估.csv', index=False, encoding='utf-8-sig')
    print()
    print(df.pivot_table(index='预见期（天）', columns='融合方案',
                         values='NSE').round(4).to_string())
    print(f'\n已保存: {OUT / "融合评估.csv"}')


if __name__ == '__main__':
    main()
