# -*- coding: utf-8 -*-
"""
五、模型精度全面评估与多维度对比分析模块
1. 评估指标体系：
   - 均方根误差 (RMSE, Root Mean Squared Error)
   - 平均绝对误差 (MAE, Mean Absolute Error)
   - 决定系数 (R², Coefficient of Determination)
   - 纳什效率系数 (NSE, Nash-Sutcliffe Efficiency)
2. 覆盖全部 5 种算法类别 * 2 种预测策略 * 7 个预测步长：
   - ANN (直接预测 / 多输出预测)
   - Random Forest (直接预测 / 多输出预测)
   - LSTM (直接预测 / 多输出预测)
   - Simple Average (直接预测 / 多输出预测)
   - Stacking (直接预测 / 多输出预测)
3. 导出精细汇总统计表格 (CSV + Excel)
4. 生成高质量对比图表：
   - 随预测步长 (Lead 1-7) 变化的误差衰减与退化曲线 (RMSE, NSE 变化折线图)
   - 直接预测 vs 多输出预测在各算法下的综合指标雷达/柱状图
   - 2004测试集典型时段（洪峰期与退水期）的拟合效果水文过程线对比图
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

# 绘图字体与风格
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style='whitegrid', font='Microsoft YaHei')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRED_PATH = os.path.join(BASE_DIR, 'results', 'predictions', 'all_models_unified_predictions.pkl')
FIG_OUT_DIR = os.path.join(BASE_DIR, 'results', 'figures', 'evaluation')
METRICS_OUT_DIR = os.path.join(BASE_DIR, 'results', 'metrics')
os.makedirs(FIG_OUT_DIR, exist_ok=True)
os.makedirs(METRICS_OUT_DIR, exist_ok=True)

def calc_nse(y_true, y_pred):
    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    if denom == 0:
        return 0.0
    numer = np.sum((y_true - y_pred) ** 2)
    return 1.0 - (numer / denom)

def compute_all_metrics():
    with open(PRED_PATH, 'rb') as f:
        data = pickle.load(f)
        
    y_test = data['y_test'] # shape: [357, 7]
    
    models_dict = {
        ('ANN', '多步直接'): data['ann_direct'],
        ('ANN', '多输出'): data['ann_multi'],
        ('随机森林', '多步直接'): data['rf_direct'],
        ('随机森林', '多输出'): data['rf_multi'],
        ('LSTM', '多步直接'): data['lstm_direct'],
        ('LSTM', '多输出'): data['lstm_multi'],
        ('简单平均集成', '多步直接'): data['avg_direct'],
        ('简单平均集成', '多输出'): data['avg_multi'],
        ('Stacking集成', '多步直接'): data['stacking_direct'],
        ('Stacking集成', '多输出'): data['stacking_multi']
    }
    
    rows = []
    for (model_name, strategy), preds in models_dict.items():
        # 分步评估 1-7 天
        step_rmses, step_maes, step_r2s, step_nses = [], [], [], []
        for step in range(7):
            y_t = y_test[:, step]
            y_p = preds[:, step]
            
            rmse = root_mean_squared_error(y_t, y_p)
            mae = mean_absolute_error(y_t, y_p)
            r2 = r2_score(y_t, y_p)
            nse = calc_nse(y_t, y_p)
            
            step_rmses.append(rmse)
            step_maes.append(mae)
            step_r2s.append(r2)
            step_nses.append(nse)
            
            rows.append({
                '算法': model_name,
                '策略': strategy,
                '预测步长': f'未来第{step+1}天',
                'Step': step + 1,
                'RMSE (cfs)': round(rmse, 2),
                'MAE (cfs)': round(mae, 2),
                'R2': round(r2, 4),
                'NSE': round(nse, 4)
            })
            
        # 整体平均评估
        rows.append({
            '算法': model_name,
            '策略': strategy,
            '预测步长': '7天整体均值',
            'Step': 0,
            'RMSE (cfs)': round(float(np.mean(step_rmses)), 2),
            'MAE (cfs)': round(float(np.mean(step_maes)), 2),
            'R2': round(float(np.mean(step_r2s)), 4),
            'NSE': round(float(np.mean(step_nses)), 4)
        })
        
    df_metrics = pd.DataFrame(rows)
    df_metrics.to_csv(os.path.join(METRICS_OUT_DIR, 'detailed_model_metrics.csv'), index=False, encoding='utf-8-sig')
    
    # 透视表：7天整体均值指标对照表
    df_summary = df_metrics[df_metrics['Step'] == 0].pivot(index='算法', columns='策略', values=['NSE', 'RMSE (cfs)', 'R2', 'MAE (cfs)'])
    df_summary.to_csv(os.path.join(METRICS_OUT_DIR, 'overall_performance_summary.csv'), encoding='utf-8-sig')
    
    print("=== 各模型 7 天平均表现汇总表 ===")
    print(df_metrics[df_metrics['Step'] == 0][['算法', '策略', 'RMSE (cfs)', 'MAE (cfs)', 'NSE', 'R2']])
    
    return data, df_metrics

def plot_performance_curves(df_metrics):
    """绘制各模型随预见期（未来1-7天）的性能演变折线图 (RMSE & NSE)"""
    df_step = df_metrics[df_metrics['Step'] > 0].copy()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 直接预测下的 RMSE 变化
    ax1 = axes[0, 0]
    sns.lineplot(data=df_step[df_step['策略'] == '多步直接'], x='Step', y='RMSE (cfs)', hue='算法', marker='o', linewidth=2.5, ax=ax1)
    ax1.set_title('多步直接预测策略下各模型 RMSE 随预见期的变化', fontsize=13, fontweight='bold')
    ax1.set_xlabel('预见期 (未来第 N 天)', fontsize=11)
    ax1.set_ylabel('RMSE (cfs) - 越低越好', fontsize=11)
    ax1.set_xticks(range(1, 8))
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # 2. 多输出预测下的 RMSE 变化
    ax2 = axes[0, 1]
    sns.lineplot(data=df_step[df_step['策略'] == '多输出'], x='Step', y='RMSE (cfs)', hue='算法', marker='s', linewidth=2.5, ax=ax2)
    ax2.set_title('多输出预测策略下各模型 RMSE 随预见期的变化', fontsize=13, fontweight='bold')
    ax2.set_xlabel('预见期 (未来第 N 天)', fontsize=11)
    ax2.set_ylabel('RMSE (cfs) - 越低越好', fontsize=11)
    ax2.set_xticks(range(1, 8))
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    # 3. 直接预测下的 NSE 变化
    ax3 = axes[1, 0]
    sns.lineplot(data=df_step[df_step['策略'] == '多步直接'], x='Step', y='NSE', hue='算法', marker='^', linewidth=2.5, ax=ax3)
    ax3.set_title('多步直接预测策略下各模型 NSE 随预见期的变化', fontsize=13, fontweight='bold')
    ax3.set_xlabel('预见期 (未来第 N 天)', fontsize=11)
    ax3.set_ylabel('纳什效率系数 (NSE) - 越高越好', fontsize=11)
    ax3.set_xticks(range(1, 8))
    ax3.grid(True, linestyle='--', alpha=0.5)
    
    # 4. 多输出预测下的 NSE 变化
    ax4 = axes[1, 1]
    sns.lineplot(data=df_step[df_step['策略'] == '多输出'], x='Step', y='NSE', hue='算法', marker='v', linewidth=2.5, ax=ax4)
    ax4.set_title('多输出预测策略下各模型 NSE 随预见期的变化', fontsize=13, fontweight='bold')
    ax4.set_xlabel('预见期 (未来第 N 天)', fontsize=11)
    ax4.set_ylabel('纳什效率系数 (NSE) - 越高越好', fontsize=11)
    ax4.set_xticks(range(1, 8))
    ax4.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    save_path = os.path.join(FIG_OUT_DIR, 'metrics_decay_lead1_to_7.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"预见期误差演变曲线已保存: {save_path}")

def plot_strategy_comparison(df_metrics):
    """对比直接预测策略 vs 多输出预测策略在各个算法上的均值表现"""
    df_avg = df_metrics[df_metrics['Step'] == 0].copy()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # RMSE 对比
    sns.barplot(data=df_avg, x='算法', y='RMSE (cfs)', hue='策略', palette='Blues_r', ax=ax1)
    ax1.set_title('各算法在直接预测与多输出策略下的平均 RMSE 对比', fontsize=13, fontweight='bold')
    ax1.set_ylabel('平均 RMSE (cfs) [越低越优]', fontsize=11)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=20)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # NSE 对比
    sns.barplot(data=df_avg, x='算法', y='NSE', hue='策略', palette='Greens_r', ax=ax2)
    ax2.set_title('各算法在直接预测与多输出策略下的平均 NSE 对比', fontsize=13, fontweight='bold')
    ax2.set_ylabel('平均 NSE [越高越优]', fontsize=11)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=20)
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    save_path = os.path.join(FIG_OUT_DIR, 'strategy_comparison_bar.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"策略对比柱状图已保存: {save_path}")

def plot_hydrograph_comparison(data):
    """绘制测试集（2004年春季洪峰期）真实值与各模型预测水文过程线对比"""
    y_test = data['y_test']
    
    # 选取未来第1天 (Lead 1) 与未来第3天 (Lead 3) 的径流模拟曲线
    # 取测试集第 60 到 160 天（约覆盖 2004 年 3 月至 6 月融雪与春汛高峰期）
    start_idx, end_idx = 60, 160
    time_idx = np.arange(start_idx, end_idx)
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    
    # Lead 1
    ax1 = axes[0]
    ax1.plot(time_idx, y_test[start_idx:end_idx, 0], label='观测值 (Observed)', color='black', linewidth=2.2, linestyle='-')
    ax1.plot(time_idx, data['ann_direct'][start_idx:end_idx, 0], label='ANN 直接预测', color='#1f77b4', linestyle='--', alpha=0.8)
    ax1.plot(time_idx, data['rf_direct'][start_idx:end_idx, 0], label='随机森林 直接预测', color='#2ca02c', linestyle='-.', alpha=0.8)
    ax1.plot(time_idx, data['lstm_direct'][start_idx:end_idx, 0], label='LSTM 直接预测', color='#ff7f0e', linestyle=':', alpha=0.8)
    ax1.plot(time_idx, data['avg_direct'][start_idx:end_idx, 0], label='简单平均集成', color='#9467bd', linestyle='-', alpha=0.9)
    ax1.plot(time_idx, data['stacking_direct'][start_idx:end_idx, 0], label='Stacking 集成', color='#d62728', linewidth=2.0, linestyle='-')
    ax1.set_title('测试集春汛期径流预测过程线对比 (未来第1天预测 Lead 1)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('径流量 Discharge (cfs)', fontsize=11)
    ax1.legend(loc='upper right', frameon=True)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Lead 3
    ax2 = axes[1]
    ax2.plot(time_idx, y_test[start_idx:end_idx, 2], label='观测值 (Observed)', color='black', linewidth=2.2, linestyle='-')
    ax2.plot(time_idx, data['ann_direct'][start_idx:end_idx, 2], label='ANN 直接预测', color='#1f77b4', linestyle='--', alpha=0.8)
    ax2.plot(time_idx, data['rf_direct'][start_idx:end_idx, 2], label='随机森林 直接预测', color='#2ca02c', linestyle='-.', alpha=0.8)
    ax2.plot(time_idx, data['lstm_direct'][start_idx:end_idx, 2], label='LSTM 直接预测', color='#ff7f0e', linestyle=':', alpha=0.8)
    ax2.plot(time_idx, data['avg_direct'][start_idx:end_idx, 2], label='简单平均集成', color='#9467bd', linestyle='-', alpha=0.9)
    ax2.plot(time_idx, data['stacking_direct'][start_idx:end_idx, 2], label='Stacking 集成', color='#d62728', linewidth=2.0, linestyle='-')
    ax2.set_title('测试集春汛期径流预测过程线对比 (未来第3天预测 Lead 3)', fontsize=13, fontweight='bold')
    ax2.set_xlabel('测试集连续观测序列时间步 (Days in 2004 Spring Season)', fontsize=11)
    ax2.set_ylabel('径流量 Discharge (cfs)', fontsize=11)
    ax2.legend(loc='upper right', frameon=True)
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    save_path = os.path.join(FIG_OUT_DIR, 'hydrograph_comparison_spring_flood.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"水文洪峰过程线对比图已保存: {save_path}")

if __name__ == '__main__':
    print("开始执行模型评估与对比制图任务...")
    data, df_metrics = compute_all_metrics()
    plot_performance_curves(df_metrics)
    plot_strategy_comparison(df_metrics)
    plot_hydrograph_comparison(data)
    print("全部评估指标计算与图表生成完成！")
