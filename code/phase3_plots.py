"""Phase-3 可视化：LSTM 和集成学习结果图表"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parent.parent
OUT_LSTM = ROOT / 'results/phase3_lstm'
OUT_ENS = ROOT / 'results/phase3_ensemble'
OUT_FIG = ROOT / 'results/phase3_figures'
OUT_FIG.mkdir(parents=True, exist_ok=True)

sns.set_style('whitegrid')
plt.rcParams['axes.unicode_minus'] = False


def plot_lstm_metrics():
    """LSTM 直接策略与多输出策略对比"""
    df = pd.read_csv(OUT_LSTM / 'lstm_metrics.csv')
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    
    for ax, metric, ylabel in zip(axes, ['R²', 'RMSE', 'MAE'], 
                                   ['R²', 'RMSE (cfs)', 'MAE (cfs)']):
        for strategy, marker, label in [('直接', 'o', 'Direct'), 
                                         ('多输出', 's', 'Multi-output')]:
            sub = df[df['策略'] == strategy]
            ax.plot(sub['预见期（天）'], sub[metric], marker=marker, 
                   linewidth=2, markersize=6, label=label)
        
        if metric == 'R²':
            ax.axhline(0, color='crimson', linestyle='--', linewidth=1, alpha=0.6)
        
        ax.set_xlabel('Forecast Horizon (days)', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f'LSTM — {ylabel}', fontsize=12, fontweight='bold')
        ax.set_xticks(range(1, 8))
        ax.legend()
        ax.grid(alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(OUT_FIG / 'LSTM_metrics.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('✓ LSTM_metrics.png')


def plot_ensemble_comparison():
    """集成学习方法对比"""
    df = pd.read_csv(OUT_ENS / 'ensemble_metrics.csv')
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for row, strategy in enumerate(['direct', 'multi']):
        sub = df[df['策略'] == strategy]
        strat_label = 'Direct' if strategy == 'direct' else 'Multi-output'
        
        for col, metric, ylabel in [(0, 'R²', 'R²'), (1, 'RMSE', 'RMSE (cfs)')]:
            ax = axes[row, col]
            
            for method, marker in [('简单平均', 'o'), ('Stacking', 's')]:
                d = sub[sub['方法'] == method]
                label = 'Simple Average' if method == '简单平均' else 'Stacking'
                ax.plot(d['预见期（天）'], d[metric], marker=marker,
                       linewidth=2, markersize=6, label=label)
            
            if metric == 'R²':
                ax.axhline(0, color='crimson', linestyle='--', linewidth=1, alpha=0.6)
            
            ax.set_xlabel('Forecast Horizon (days)', fontsize=10)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.set_title(f'Ensemble ({strat_label}) — {ylabel}', 
                        fontsize=11, fontweight='bold')
            ax.set_xticks(range(1, 8))
            ax.legend(fontsize=9)
            ax.grid(alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(OUT_FIG / 'Ensemble_comparison.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('✓ Ensemble_comparison.png')


def plot_all_models_comparison():
    """所有模型综合对比（LSTM vs 集成学习 vs ANN vs RF）"""
    lstm = pd.read_csv(OUT_LSTM / 'lstm_metrics.csv')
    ens = pd.read_csv(OUT_ENS / 'ensemble_metrics.csv')
    v5 = pd.read_csv(ROOT / 'results/phase2_v5/测试精度指标.csv')
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 只比较直接策略的第1天
    for ax, metric, ylabel in [(axes[0], 'R²', 'R²'), (axes[1], 'RMSE', 'RMSE (cfs)')]:
        data = []
        
        # LSTM
        lstm_d = lstm[(lstm['策略'] == '直接') & (lstm['预见期（天）'] == 1)]
        if len(lstm_d) > 0:
            data.append(('LSTM\nDirect', lstm_d.iloc[0][metric]))
        
        # ANN
        ann_d = v5[(v5['模型'] == 'ANN') & (v5['策略'] == '直接') & (v5['预见期（天）'] == 1)]
        if len(ann_d) > 0:
            col = 'RMSE（cfs）' if metric == 'RMSE' else metric
            data.append(('ANN\nDirect', ann_d.iloc[0][col]))
        
        # RF
        rf_d = v5[(v5['模型'] == 'RF') & (v5['策略'] == '直接') & (v5['预见期（天）'] == 1)]
        if len(rf_d) > 0:
            col = 'RMSE（cfs）' if metric == 'RMSE' else metric
            data.append(('RF\nDirect', rf_d.iloc[0][col]))
        
        # Ensemble Simple Average
        ens_avg = ens[(ens['策略'] == 'direct') & (ens['方法'] == '简单平均') & (ens['预见期（天）'] == 1)]
        if len(ens_avg) > 0:
            data.append(('Ensemble\nAvg', ens_avg.iloc[0][metric]))
        
        # Ensemble Stacking
        ens_stack = ens[(ens['策略'] == 'direct') & (ens['方法'] == 'Stacking') & (ens['预见期（天）'] == 1)]
        if len(ens_stack) > 0:
            data.append(('Ensemble\nStacking', ens_stack.iloc[0][metric]))
        
        if data:
            labels, values = zip(*data)
            x = np.arange(len(labels))
            bars = ax.bar(x, values, width=0.6, alpha=0.85)
            
            # 标注数值
            for i, (bar, val) in enumerate(zip(bars, values)):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{val:.3f}' if metric == 'R²' else f'{val:.0f}',
                       ha='center', va='bottom', fontsize=9)
            
            if metric == 'R²':
                ax.axhline(0, color='crimson', linestyle='--', linewidth=1, alpha=0.5)
            
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=10)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_title(f'Model Comparison (Day 1) — {ylabel}', 
                        fontsize=12, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(OUT_FIG / 'All_models_comparison.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('✓ All_models_comparison.png')


def plot_stacking_weights():
    """Stacking 元学习器权重可视化"""
    df = pd.read_csv(OUT_ENS / 'ensemble_metrics.csv')
    stacking = df[df['方法'] == 'Stacking'].copy()
    
    if '元模型权重' not in stacking.columns or stacking.empty:
        print('⚠ 跳过权重图：元模型权重列不存在')
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for ax, strategy, title in [(axes[0], 'direct', 'Direct Strategy'),
                                 (axes[1], 'multi', 'Multi-output Strategy')]:
        sub = stacking[stacking['策略'] == strategy]
        if sub.empty:
            continue
        
        weights_matrix = []
        for _, row in sub.iterrows():
            try:
                w = eval(row['元模型权重'])
                weights_matrix.append(w)
            except:
                weights_matrix.append([0, 0, 0])
        
        weights_matrix = np.array(weights_matrix).T  # shape: (3, 7)
        
        x = np.arange(1, 8)
        width = 0.25
        
        for i, (model, color) in enumerate([('ANN', 'tab:blue'), 
                                            ('RF', 'tab:orange'), 
                                            ('LSTM', 'tab:green')]):
            ax.bar(x + (i-1)*width, weights_matrix[i], width, 
                  label=model, alpha=0.85, color=color)
        
        ax.axhline(0, color='black', linewidth=0.8, linestyle='-', alpha=0.3)
        ax.set_xlabel('Forecast Horizon (days)', fontsize=11)
        ax.set_ylabel('Meta-learner Weight', fontsize=11)
        ax.set_title(f'Stacking Weights — {title}', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(OUT_FIG / 'Stacking_weights.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('✓ Stacking_weights.png')


if __name__ == '__main__':
    print('生成 Phase-3 图表...\n')
    plot_lstm_metrics()
    plot_ensemble_comparison()
    plot_all_models_comparison()
    plot_stacking_weights()
    print(f'\n所有图表已保存至: {OUT_FIG}')
