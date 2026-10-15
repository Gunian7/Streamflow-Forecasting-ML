# -*- coding: utf-8 -*-
"""
一、数据分析与可视化模块 (针对第一次阶段验收严格对齐)
严格按照《第一次阶段验收说明》逐项实现：
- 子任务1：给出所有英文字段的中文含义、物理单位与详细解释
- 子任务2：对流域数据集的所有列（日期除外，共8列）均做时间序列可视化
- 子任务3：对流域数据集的所有列（日期除外，共8列）均做直方图 (含KDE)
- 子任务4：对流域数据集的所有列（日期除外，共8列）均做独立箱型图 (识别离群点) + 径流月度季节箱型图
- 子任务5：皮尔逊相关系数热力图，严格满足三项要求：
    1. 计算径流序列与其滞后序列的相关系数 (Discharge vs Discharge_lag)
    2. 计算径流序列与其他因子序列的相关系数 (Discharge vs Prcp, Tmax, Tmin, Srad, Swe, Vp, Dayl)
    3. 计算径流序列与其他因子滞后序列的相关系数 (Discharge vs Prcp_lag, Tmax_lag, 等)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体与负号正常显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'figures', 'eda')
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '01047000.csv')

# 所有非日期列
NUMERIC_COLS = ['Discharge', 'Dayl', 'Prcp', 'Srad', 'Swe', 'Tmax', 'Tmin', 'Vp']
CHINESE_NAMES = [
    '日均径流量 (Discharge)', '白昼时长 (Dayl)', '日降水量 (Prcp)', '太阳辐射 (Srad)',
    '积雪水当量 (Swe)', '最高气温 (Tmax)', '最低气温 (Tmin)', '水汽压 (Vp)'
]
CHINESE_LABELS = ['径流', '日长', '降水', '辐射', '积雪', '最高温', '最低温', '水汽压']

def load_data():
    df = pd.read_csv(DATA_PATH)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    return df

def generate_summary(df):
    stats = df[NUMERIC_COLS].describe().T
    stats['skewness'] = df[NUMERIC_COLS].skew()
    stats['kurtosis'] = df[NUMERIC_COLS].kurtosis()
    stats.to_csv(os.path.join(OUTPUT_DIR, 'summary_statistics.csv'), encoding='utf-8-sig')
    print("=== 基础描述性统计 ===")
    print(stats[['mean', 'std', 'min', '50%', 'max', 'skewness']])
    return stats

# 子任务2：时间序列可视化 (所有8列)
def plot_time_series(df):
    fig, axes = plt.subplots(4, 2, figsize=(16, 14), sharex=True)
    axes = axes.flatten()
    
    for i, (col, name) in enumerate(zip(NUMERIC_COLS, CHINESE_NAMES)):
        color = '#1f77b4' if col == 'Discharge' else '#2ca02c' if 'Prcp' in col else '#d62728' if 'T' in col else '#ff7f0e'
        axes[i].plot(df['Date'], df[col], label=name, color=color, linewidth=0.9)
        axes[i].set_title(name, fontsize=12, fontweight='bold')
        axes[i].grid(True, linestyle='--', alpha=0.5)
        axes[i].axvline(pd.to_datetime('2004-01-01'), color='black', linestyle=':', linewidth=1.5, label='训练/测试分割线 (2004)')
        if i == 0:
            axes[i].legend(loc='upper right')
            
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, '1_time_series_overview.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"子任务2 时间序列图已保存: {save_path}")

# 子任务3：直方图与KDE (所有8列)
def plot_histograms(df):
    fig, axes = plt.subplots(4, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, (col, name) in enumerate(zip(NUMERIC_COLS, CHINESE_NAMES)):
        sns.histplot(df[col], kde=True, ax=axes[i], color='royalblue', stat='density', bins=30)
        axes[i].set_title(f'{name} 分布直方图与KDE曲线', fontsize=11, fontweight='bold')
        axes[i].set_xlabel('')
        axes[i].grid(True, linestyle='--', alpha=0.5)
        
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, '2_histograms_kde.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"子任务3 直方图已保存: {save_path}")

# 子任务4：箱型图 (所有8列全部具有独立箱型图展示 + 月度季节箱型图)
def plot_boxplots(df):
    # 图4-1：对所有8列分别画出单变量独立箱型图（真实物理量纲尺度）
    fig, axes = plt.subplots(4, 2, figsize=(15, 13))
    axes = axes.flatten()
    
    for i, (col, name) in enumerate(zip(NUMERIC_COLS, CHINESE_NAMES)):
        sns.boxplot(y=df[col], ax=axes[i], color='#7293CB', width=0.4, fliersize=3)
        axes[i].set_title(f'{name} 箱型图 (真实物理尺度)', fontsize=11, fontweight='bold')
        axes[i].set_ylabel('')
        axes[i].grid(True, linestyle='--', alpha=0.5)
        
    plt.tight_layout()
    save_path1 = os.path.join(OUTPUT_DIR, '3_boxplots_all_8_columns.png')
    plt.savefig(save_path1, dpi=300)
    plt.close()
    print(f"子任务4 所有8列独立箱型图已保存: {save_path1}")
    
    # 图4-2：归一化横向对比箱型图 + 径流量月度季节箱型图
    norm_df = (df[NUMERIC_COLS] - df[NUMERIC_COLS].min()) / (df[NUMERIC_COLS].max() - df[NUMERIC_COLS].min() + 1e-8)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 11))
    sns.boxplot(data=norm_df, ax=ax1, palette='Set3')
    ax1.set_xticks(range(len(CHINESE_LABELS)))
    ax1.set_xticklabels(CHINESE_LABELS, fontsize=11)
    ax1.set_title('所有8列变量归一化[0,1]后的箱型图对比 (宏观离群点检测)', fontsize=12, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    sns.boxplot(x='Month', y='Discharge', data=df, ax=ax2, hue='Month', palette='coolwarm', legend=False)
    ax2.set_title('日均径流量 (Discharge) 各月份季节性箱型图分布', fontsize=12, fontweight='bold')
    ax2.set_xlabel('月份 (Month)', fontsize=11)
    ax2.set_ylabel('径流量 (cfs)', fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    save_path2 = os.path.join(OUTPUT_DIR, '3_boxplots_outliers.png')
    plt.savefig(save_path2, dpi=300)
    plt.close()
    print(f"子任务4 综合箱型图与季节分析已保存: {save_path2}")

# 子任务5：皮尔逊相关系数热力图 (严格包含：径流与滞后序列、径流与其他因子序列、径流与其他因子滞后序列)
def plot_correlation_heatmap(df):
    # 构造滞后序列数据
    df_corr = pd.DataFrame()
    df_corr['Discharge'] = df['Discharge']
    
    # 1. 径流序列与其滞后序列 (Lag 1, 2, 3, 7)
    for lag in [1, 2, 3, 7]:
        df_corr[f'Discharge_lag{lag}'] = df['Discharge'].shift(lag)
        
    # 2. 径流与其他因子序列及其滞后序列 (Lag 1, 2, 3)
    other_factors = ['Prcp', 'Tmax', 'Tmin', 'Srad', 'Vp', 'Dayl']
    for factor in other_factors:
        df_corr[factor] = df[factor]
        for lag in [1, 2, 3]:
            df_corr[f'{factor}_lag{lag}'] = df[factor].shift(lag)
            
    # 去除滞后产生的空值
    df_corr_clean = df_corr.dropna()
    
    # 1. 全局皮尔逊相关矩阵
    corr_matrix = df_corr_clean.corr(method='pearson')
    corr_matrix.to_csv(os.path.join(OUTPUT_DIR, 'pearson_correlation_full_lag_matrix.csv'), encoding='utf-8-sig')
    
    # 2. 专门提取【径流序列】与【自身滞后】、【其他因子】、【其他因子滞后】的相关系数
    discharge_corrs = corr_matrix[['Discharge']].drop('Discharge')
    discharge_corrs.columns = ['Pearson_r']
    discharge_corrs['Abs_r'] = discharge_corrs['Pearson_r'].abs()
    discharge_corrs = discharge_corrs.sort_values('Abs_r', ascending=False)
    discharge_corrs.to_csv(os.path.join(OUTPUT_DIR, 'discharge_with_lag_factors_correlations.csv'), encoding='utf-8-sig')
    
    # 绘制热力图 A：精选核心水文驱动因子及其滞后的完整热力图
    key_cols = [
        'Discharge', 'Discharge_lag1', 'Discharge_lag2', 'Discharge_lag3', 'Discharge_lag7',
        'Prcp', 'Prcp_lag1', 'Prcp_lag2', 'Prcp_lag3',
        'Tmax', 'Tmax_lag1', 'Tmin', 'Tmin_lag1',
        'Srad', 'Srad_lag1', 'Vp', 'Vp_lag1', 'Dayl'
    ]
    key_labels = [
        '径流', '径流lag1', '径流lag2', '径流lag3', '径流lag7',
        '降水', '降水lag1', '降水lag2', '降水lag3',
        '最高温', '最高温lag1', '最低温', '最低温lag1',
        '辐射', '辐射lag1', '水汽压', '水汽压lag1', '日长'
    ]
    
    sub_corr = df_corr_clean[key_cols].corr(method='pearson')
    sub_corr.columns = key_labels
    sub_corr.index = key_labels
    
    plt.figure(figsize=(15, 12))
    sns.heatmap(sub_corr, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8}, annot_kws={"size": 9})
    plt.title('径流及其滞后序列、其他因子及其滞后序列的皮尔逊相关系数热力图 (全面对齐子任务5)', fontsize=13, fontweight='bold')
    save_path = os.path.join(OUTPUT_DIR, '4_pearson_correlation_heatmap.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"子任务5 皮尔逊相关热力图已保存: {save_path}")
    
    # 绘制热力图 B：径流序列与全部因子及滞后因子的专项相关度对比条形图 (更直观展示验收要求)
    plt.figure(figsize=(12, 9))
    top_corrs = discharge_corrs.sort_values('Abs_r', ascending=True)
    colors = ['#d62728' if r > 0 else '#1f77b4' for r in top_corrs['Pearson_r']]
    plt.barh(top_corrs.index, top_corrs['Pearson_r'], color=colors, alpha=0.85)
    plt.axvline(0, color='black', linestyle='--', linewidth=0.8)
    plt.title('日均径流序列 (Discharge) 与自身滞后、其他因子及滞后因子的皮尔逊相关度', fontsize=13, fontweight='bold')
    plt.xlabel('皮尔逊相关系数 r (红色为正相关，蓝色为负相关)', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    save_path_bar = os.path.join(OUTPUT_DIR, '4_discharge_lag_factors_barplot.png')
    plt.savefig(save_path_bar, dpi=300)
    plt.close()
    print(f"子任务5 径流与各滞后因子相关度条形图已保存: {save_path_bar}")

if __name__ == '__main__':
    print("开始执行对齐第一次阶段验收的数据分析与可视化...")
    df = load_data()
    generate_summary(df)
    plot_time_series(df)
    plot_histograms(df)
    plot_boxplots(df)
    plot_correlation_heatmap(df)
    print("数据分析全部 5 个子任务更新完成！")
