"""Phase-2 v5 全部图表（英文标签，避免字体缺失）"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'results/phase2_v5'
OUT.mkdir(parents=True, exist_ok=True)

sns.set_style('whitegrid')
plt.rcParams['axes.unicode_minus'] = False


def plot_metrics():
    df = pd.read_csv(OUT / '测试精度指标.csv')
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    for r, model in enumerate(['ANN', 'RF']):
        sub = df[df['模型'] == model]
        for c, (col, ylab) in enumerate([('RMSE（cfs）', 'RMSE (cfs)'), ('NSE', 'NSE')]):
            ax = axes[r, c]
            for strat, lab in [('直接', 'Direct'), ('多输出', 'Multi-output')]:
                d = sub[sub['策略'] == strat]
                ax.plot(d['预见期（天）'], d[col], marker='o' if c == 0 else 's',
                        linewidth=2, markersize=6, label=lab)
            if c == 1:
                ax.axhline(0, color='crimson', linestyle='--', linewidth=1)
            ax.set_xlabel('Forecast Horizon (days)', fontsize=11)
            ax.set_ylabel(ylab, fontsize=11)
            ax.set_title(f'{model} — {ylab}', fontsize=12, fontweight='bold')
            ax.set_xticks(range(1, 8))
            ax.legend()
            ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT / '测试集精度指标对比.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('测试集精度指标对比.png')


def plot_strategy():
    df = pd.read_csv(OUT / '测试精度指标.csv')
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for r, model in enumerate(['ANN', 'RF']):
        sub = df[df['模型'] == model]
        for c, (col, ylab) in enumerate([('RMSE（cfs）', 'RMSE (cfs)'), ('NSE', 'NSE')]):
            ax = axes[r, c]
            x = np.arange(7)
            w = 0.36
            d = sub[sub['策略'] == '直接'].set_index('预见期（天）')
            m = sub[sub['策略'] == '多输出'].set_index('预见期（天）')
            ax.bar(x - w / 2, [d.loc[h, col] for h in range(1, 8)], w,
                   label='Direct', alpha=0.85)
            ax.bar(x + w / 2, [m.loc[h, col] for h in range(1, 8)], w,
                   label='Multi-output', alpha=0.85)
            if c == 1:
                ax.axhline(0, color='crimson', linestyle='--', linewidth=1)
            ax.set_xticks(x)
            ax.set_xticklabels(range(1, 8))
            ax.set_xlabel('Forecast Horizon (days)', fontsize=11)
            ax.set_ylabel(ylab, fontsize=11)
            ax.set_title(f'{model} Strategy Comparison — {ylab}',
                         fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / '策略对比柱状图.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('策略对比柱状图.png')


def plot_baseline():
    met = pd.read_csv(OUT / '测试精度指标.csv')
    base = pd.read_csv(OUT / '基线指标.csv')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, col, ylab in [(axes[0], 'RMSE（cfs）', 'RMSE (cfs)'), (axes[1], 'NSE', 'NSE')]:
        for model, ls in [('ANN', '-'), ('RF', '-')]:
            for strat, mk in [('直接', 'o'), ('多输出', 's')]:
                d = met[(met['模型'] == model) & (met['策略'] == strat)]
                ax.plot(d['预见期（天）'], d[col], linestyle=ls, marker=mk,
                        linewidth=2, markersize=5, alpha=0.85,
                        label=f'{model} {"Direct" if strat=="直接" else "Multi"}')
        for name, lab in [('真持续性', 'Persistence'), ('月份气候均值', 'Monthly Climate')]:
            d = base[base['基线'] == name]
            ax.plot(d['预见期（天）'], d[col], linestyle='--', linewidth=2,
                    color='gray' if name == '真持续性' else 'black', alpha=0.7, label=lab)
        if col == 'NSE':
            ax.axhline(0, color='crimson', linestyle=':', linewidth=1)
        ax.set_xlabel('Forecast Horizon (days)', fontsize=11)
        ax.set_ylabel(ylab, fontsize=11)
        ax.set_title(f'Models vs Baselines — {ylab}', fontsize=12, fontweight='bold')
        ax.set_xticks(range(1, 8))
        ax.legend(fontsize=8, ncol=2)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT / '模型与基线对比.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('模型与基线对比.png')


def plot_heatmap():
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, f, title in [
        (axes[0], 'ANN_1_网格搜索.csv', 'ANN Direct (Day 1) — Hyperparameter Search'),
        (axes[1], 'ANN_multi_网格搜索.csv', 'ANN Multi-output — Hyperparameter Search'),
    ]:
        g = pd.read_csv(OUT / f)
        piv = g.pivot_table(index='param_regressor__model__hidden_layer_sizes',
                            columns='param_regressor__model__alpha',
                            values='mean_test_score') * -1
        sns.heatmap(piv, annot=True, fmt='.3f', cmap='RdYlGn_r', ax=ax,
                    cbar_kws={'label': 'CV RMSE (log scale)'})
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('Regularization (alpha)', fontsize=10)
        ax.set_ylabel('Hidden Layer Sizes', fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / 'ANN超参数搜索热力图.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('ANN超参数搜索热力图.png')


def _process_plot(model):
    det = pd.read_csv(OUT / '逐日预测结果.csv')
    sub = det[det['模型'] == model]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)

    for r, strat in enumerate(['直接', '多输出']):
        for c, h in enumerate([1, 7]):
            ax = axes[r, c]
            d = sub[(sub['策略'] == strat) & (sub['预见期（天）'] == h)]
            ax.plot(range(len(d)), d['实测径流（cfs）'], color='tab:blue',
                    linewidth=1.2, label='Observed')
            ax.plot(range(len(d)), d['预测径流（cfs）'], color='tab:orange',
                    linewidth=1.2, linestyle='--', label='Predicted')
            tag = 'Direct' if strat == '直接' else 'Multi-output'
            ax.set_title(f'{model} {tag} — Day {h}', fontsize=11, fontweight='bold')
            ax.set_xlabel('Test sample index (2004)', fontsize=9)
            ax.set_ylabel('Discharge (cfs)', fontsize=9)
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT / f'{model}_预测过程.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'{model}_预测过程.png')


def _scatter_plot(model):
    det = pd.read_csv(OUT / '逐日预测结果.csv')
    sub = det[(det['模型'] == model) & (det['预见期（天）'] == 1)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, strat in zip(axes, ['直接', '多输出']):
        d = sub[sub['策略'] == strat]
        ax.scatter(d['实测径流（cfs）'], d['预测径流（cfs）'], s=14, alpha=0.6)
        hi = max(d['实测径流（cfs）'].max(), d['预测径流（cfs）'].max())
        ax.plot([0, hi], [0, hi], 'r--', linewidth=1)
        tag = 'Direct' if strat == '直接' else 'Multi-output'
        ax.set_title(f'{model} {tag} — Day 1', fontsize=11, fontweight='bold')
        ax.set_xlabel('Observed (cfs)', fontsize=10)
        ax.set_ylabel('Predicted (cfs)', fontsize=10)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / f'{model}_散点图.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'{model}_散点图.png')


if __name__ == '__main__':
    plot_metrics()
    plot_strategy()
    plot_baseline()
    plot_heatmap()
    for m in ['ANN', 'RF']:
        _process_plot(m)
        _scatter_plot(m)
    print('\n全部图表已生成')
