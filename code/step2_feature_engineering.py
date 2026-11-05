# -*- coding: utf-8 -*-
"""
二、特征工程模块 (Feature Engineering)
1. 时间序列特征提取：
   - 四季编码 (Season: 春=1, 夏=2, 秋=3, 冬=0 及 One-Hot/正弦余弦周期编码)
   - 月份编码 (Month: 1-12，正弦余弦循环编码 month_sin, month_cos)
   - 滞后特征 (Lag features: 径流与气象变量的前1, 2, 3, 7天滞后值)
   - 窗口特征 (Rolling features: 径流与降水等在3天、7天的滑动均值、滑动标准差、滑动最大值、滑动最小值)
2. 特征归一化：
   - 总结常用归一化方法 (Min-Max Scaling, Z-Score Standardization)
   - 严格在训练集(2000-2003)上 fit，再 transform 训练集与测试集(2004)，防止数据穿越
3. 特征选择：
   - 总结常用特征选择方法 (Filter法: 皮尔逊相关系数、互信息法 Mutual Information; Wrapper法; Embedded法)
   - 分别实现皮尔逊相关系数法与互信息法并对比输出重要性排序，筛选最佳特征子集
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import mutual_info_regression

# 设置字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'figures', 'fe')
DATA_OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'processed_data')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_OUT_DIR, exist_ok=True)

DATA_PATH = os.environ.get(
    'RUNOFF_DATA_PATH',
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '01047000.csv'))

def get_season(month):
    if month in [3, 4, 5]:
        return 1 # 春
    elif month in [6, 7, 8]:
        return 2 # 夏
    elif month in [9, 10, 11]:
        return 3 # 秋
    else:
        return 0 # 冬

def export_feature_values(original, full_features, data_dict, output_path):
    """保留原始行列，按日期在右侧追加未筛选特征及训练集 scaler 的输出。"""
    dates = pd.to_datetime(original['Date'])
    indexed = full_features.set_index('Date')
    if indexed.index.has_duplicates or dates.duplicated().any():
        raise ValueError('日期必须唯一，不能进行有歧义的特征对齐')
    aligned = indexed.reindex(dates)
    derived = aligned[[c for c in aligned.columns
                       if c not in original.columns and not c.startswith('Target_lead')]].copy()
    derived.index = original.index
    cols = data_dict['feature_cols']
    values = aligned[cols].to_numpy(dtype=float)
    mm = pd.DataFrame(data_dict['scaler_mm_X'].transform(values),
                      index=original.index, columns=[f'{c}_minmax' for c in cols])
    std = pd.DataFrame(data_dict['scaler_std_X'].transform(values),
                       index=original.index, columns=[f'{c}_zscore' for c in cols])
    result = pd.concat([original, derived, mm, std], axis=1)
    result.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f'逐日特征数值 CSV: {output_path} ({len(result)} 行, {len(result.columns)} 列)')
    return result


def build_features(return_full=False):
    df = pd.read_csv(DATA_PATH)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # 1. 时间周期特征提取
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['DayOfYear'] = df['Date'].dt.dayofyear
    
    # 四季独热编码，不赋予季节数值大小关系
    for name, months in [('spring', [3, 4, 5]), ('summer', [6, 7, 8]),
                         ('autumn', [9, 10, 11]), ('winter', [12, 1, 2])]:
        df[f'Season_{name}'] = df['Month'].isin(months).astype(int)
    
    # 周期性正弦余弦编码 (Cyclical encoding，保证12月到1月的连续平滑)
    df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12.0)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12.0)
    df['DayOfYear_sin'] = np.sin(2 * np.pi * df['DayOfYear'] / 365.25)
    df['DayOfYear_cos'] = np.cos(2 * np.pi * df['DayOfYear'] / 365.25)
    
    # 2. 滞后特征 (Lag features)
    # 径流与气象变量的历史滞后
    lag_cols = ['Discharge', 'Prcp', 'Tmax', 'Tmin', 'Srad', 'Vp']
    lags = [1, 2, 3, 7]
    for col in lag_cols:
        for lag in lags:
            df[f'{col}_lag{lag}'] = df[col].shift(lag)
            
    # 3. 窗口统计特征 (Rolling features)
    # 计算过去 3 天与 7 天的滑动均值、标准差、最大值、最小值
    roll_cols = ['Discharge', 'Prcp', 'Tmax']
    windows = [3, 7]
    for col in roll_cols:
        for w in windows:
            # 使用 shift(1) 防止把当天未来信息算进特征（针对滞后特征预测场景）
            rolled = df[col].shift(1).rolling(window=w)
            df[f'{col}_roll_mean_{w}'] = rolled.mean()
            df[f'{col}_roll_std_{w}'] = rolled.std()
            df[f'{col}_roll_max_{w}'] = rolled.max()
            df[f'{col}_roll_min_{w}'] = rolled.min()
            
    # 4. 构建未来 1-7 天的目标径流变量 (Discharge_lead1 ~ Discharge_lead7)
    for lead in range(1, 8):
        df[f'Target_lead{lead}'] = df['Discharge'].shift(-lead)
        
    # 由于构建了 lag 7 和 lead 7，以及 rolling 7，数据头部 7 天和尾部 7 天会有 NaN
    # 为保证特征与目标严格对齐，去除含 NaN 的行
    df_clean = df.dropna().reset_index(drop=True)
    print(f"特征提取完毕: 原始样本数 {len(df)}, 清洗后有效样本数 {len(df_clean)}, 特征总列数 {df_clean.shape[1]}")
    
    return (df_clean, df) if return_full else df_clean

def split_and_scale(df_clean):
    """
    按指导书要求严格按照时间顺序划分：
    前4年 (2000-2003) 为训练集，第5年 (2004) 为测试集。
    归一化 scaler 严格在训练集 fit，再 transform 训练和测试集。
    """
    # 确定特征列
    drop_cols = ['Date', 'Year', 'Month', 'Day', 'DayOfWeek', 'DayOfYear', 'Swe'] + [f'Target_lead{i}' for i in range(1, 8)]
    feature_cols = [c for c in df_clean.columns if c not in drop_cols]
    target_cols = [f'Target_lead{i}' for i in range(1, 8)]
    
    train_mask = df_clean['Year'] < 2004
    test_mask = df_clean['Year'] == 2004
    
    train_df = df_clean[train_mask].copy()
    test_df = df_clean[test_mask].copy()
    
    print(f"训练集区间: {train_df['Date'].min().strftime('%Y-%m-%d')} 至 {train_df['Date'].max().strftime('%Y-%m-%d')}, 样本数: {len(train_df)}")
    print(f"测试集区间: {test_df['Date'].min().strftime('%Y-%m-%d')} 至 {test_df['Date'].max().strftime('%Y-%m-%d')}, 样本数: {len(test_df)}")
    
    # 比较 Min-Max 与 Standard 两种归一化方法
    scaler_mm_X = MinMaxScaler()
    scaler_mm_y = MinMaxScaler()
    
    scaler_std_X = StandardScaler()
    scaler_std_y = StandardScaler()
    
    X_train_raw = train_df[feature_cols].values
    y_train_raw = train_df[target_cols].values
    X_test_raw = test_df[feature_cols].values
    y_test_raw = test_df[target_cols].values
    
    # 拟合 Min-Max
    X_train_mm = scaler_mm_X.fit_transform(X_train_raw)
    y_train_mm = scaler_mm_y.fit_transform(y_train_raw)
    X_test_mm = scaler_mm_X.transform(X_test_raw)
    y_test_mm = scaler_mm_y.transform(y_test_raw)
    
    # 拟合 Standard
    X_train_std = scaler_std_X.fit_transform(X_train_raw)
    y_train_std = scaler_std_y.fit_transform(y_train_raw)
    X_test_std = scaler_std_X.transform(X_test_raw)
    y_test_std = scaler_std_y.transform(y_test_raw)
    
    # 保存归一化比较图（以径流特征为例）
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.hist(train_df['Discharge'], bins=30, color='skyblue', edgecolor='black')
    plt.title('原始径流分布 (Original)')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.subplot(1, 3, 2)
    plt.hist(y_train_mm[:, 0], bins=30, color='lightgreen', edgecolor='black')
    plt.title('Min-Max 归一化后 [0, 1]')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.subplot(1, 3, 3)
    plt.hist(y_train_std[:, 0], bins=30, color='salmon', edgecolor='black')
    plt.title('Z-Score 标准化后 (Mean=0, Std=1)')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'normalization_comparison.png'), dpi=300)
    plt.close()
    
    return {
        'feature_cols': feature_cols,
        'target_cols': target_cols,
        'train_df': train_df,
        'test_df': test_df,
        'X_train_raw': X_train_raw,
        'y_train_raw': y_train_raw,
        'X_test_raw': X_test_raw,
        'y_test_raw': y_test_raw,
        'X_train_mm': X_train_mm,
        'y_train_mm': y_train_mm,
        'X_test_mm': X_test_mm,
        'y_test_mm': y_test_mm,
        'scaler_mm_X': scaler_mm_X,
        'scaler_mm_y': scaler_mm_y,
        'scaler_std_X': scaler_std_X,
        'scaler_std_y': scaler_std_y
    }

def selection_feature_indices(feature_cols):
    return [i for i, col in enumerate(feature_cols) if col != 'Discharge']


def feature_selection(data_dict):
    """
    实现两种特征选择方法：
    1. 皮尔逊相关系数法 (Pearson Correlation with Target)
    2. 互信息法 (Mutual Information Regression)
    以 Target_lead1 (未来第1天径流) 为基准计算特征重要度，并完成筛选与图表导出
    """
    indices = selection_feature_indices(data_dict['feature_cols'])
    feature_cols = [data_dict['feature_cols'][i] for i in indices]
    train_df = data_dict['train_df']
    
    # 1. 皮尔逊相关系数
    pearson_corrs = []
    for col in feature_cols:
        r = train_df[col].corr(train_df['Target_lead1'])
        pearson_corrs.append(r)
        
    df_pearson = pd.DataFrame({
        'Feature': feature_cols,
        'Pearson_r': pearson_corrs,
        'Abs_Pearson_r': np.abs(pearson_corrs)
    }).sort_values('Abs_Pearson_r', ascending=False).reset_index(drop=True)
    
    # 2. 互信息法
    X_train = data_dict['X_train_raw'][:, indices]
    y_target1 = data_dict['y_train_raw'][:, 0]
    
    mi_scores = mutual_info_regression(X_train, y_target1, random_state=42)
    df_mi = pd.DataFrame({
        'Feature': feature_cols,
        'Mutual_Info': mi_scores
    }).sort_values('Mutual_Info', ascending=False).reset_index(drop=True)
    
    # 合并特征选择评分表
    df_fs = pd.merge(df_pearson, df_mi, on='Feature')
    df_fs.to_csv(os.path.join(OUTPUT_DIR, 'feature_selection_ranking.csv'), index=False, encoding='utf-8-sig')
    
    # 可视化前20个最重要特征
    plt.figure(figsize=(14, 8))
    
    plt.subplot(1, 2, 1)
    top20_pearson = df_fs.sort_values('Abs_Pearson_r', ascending=True).tail(20)
    plt.barh(top20_pearson['Feature'], top20_pearson['Abs_Pearson_r'], color='steelblue')
    plt.title('皮尔逊相关系数绝对值 Top 20 特征', fontsize=12, fontweight='bold')
    plt.xlabel('|r|')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.subplot(1, 2, 2)
    top20_mi = df_fs.sort_values('Mutual_Info', ascending=True).tail(20)
    plt.barh(top20_mi['Feature'], top20_mi['Mutual_Info'], color='darkorange')
    plt.title('互信息法评分 (Mutual Info) Top 20 特征', fontsize=12, fontweight='bold')
    plt.xlabel('MI Score')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'feature_selection_comparison.png'), dpi=300)
    plt.close()
    
    print("=== 特征选择 Top 10 (皮尔逊相关系数) ===")
    print(df_fs[['Feature', 'Pearson_r', 'Abs_Pearson_r']].head(10))
    print("=== 特征选择 Top 10 (互信息法) ===")
    print(df_fs[['Feature', 'Mutual_Info']].sort_values('Mutual_Info', ascending=False).head(10))
    
    # 选取最佳特征子集：综合排名靠前，保留滞后径流、滑动统计以及关键气象特征(降水、气温、太阳辐射、周期编码等)
    # 取互信息 > 0.05 或 皮尔逊相关绝对值 > 0.25 的优质特征，同时保证降水等驱动变量不被丢弃
    selected_features = df_fs[
        (df_fs['Abs_Pearson_r'] >= 0.2) | (df_fs['Mutual_Info'] >= 0.05) | (df_fs['Feature'].str.startswith('Prcp'))
    ]['Feature'].tolist()
    
    print(f"\n选定的优质特征数量: {len(selected_features)} / {len(feature_cols)}")
    print("选定特征列表:", selected_features)
    
    return df_fs, selected_features

def save_preprocessed_dataset(data_dict, selected_features):
    """
    保存处理好的数据集和特征信息到 results/processed_data，便于后续统一调用
    """
    import pickle
    
    feature_cols = data_dict['feature_cols']
    selected_indices = [feature_cols.index(f) for f in selected_features]
    
    # 提取选定特征的归一化矩阵
    X_train_selected_mm = data_dict['X_train_mm'][:, selected_indices]
    X_test_selected_mm = data_dict['X_test_mm'][:, selected_indices]
    
    X_train_selected_raw = data_dict['X_train_raw'][:, selected_indices]
    X_test_selected_raw = data_dict['X_test_raw'][:, selected_indices]
    
    data_bundle = {
        'all_features': feature_cols,
        'selected_features': selected_features,
        'selected_indices': selected_indices,
        'target_cols': data_dict['target_cols'],
        'train_df': data_dict['train_df'],
        'test_df': data_dict['test_df'],
        'X_train_raw': X_train_selected_raw,
        'X_test_raw': X_test_selected_raw,
        'y_train_raw': data_dict['y_train_raw'],
        'y_test_raw': data_dict['y_test_raw'],
        'X_train_mm': X_train_selected_mm,
        'X_test_mm': X_test_selected_mm,
        'y_train_mm': data_dict['y_train_mm'],
        'y_test_mm': data_dict['y_test_mm'],
        'scaler_X': data_dict['scaler_mm_X'],
        'scaler_y': data_dict['scaler_mm_y']
    }
    
    save_file = os.path.join(DATA_OUT_DIR, 'processed_dataset.pkl')
    with open(save_file, 'wb') as f:
        pickle.dump(data_bundle, f)
    print(f"数据包已打包保存至: {save_file}")

if __name__ == '__main__':
    print("执行特征工程任务...")
    df_clean, df_full = build_features(return_full=True)
    data_dict = split_and_scale(df_clean)
    export_feature_values(pd.read_csv(DATA_PATH), df_full, data_dict,
                          os.path.join(DATA_OUT_DIR, '01047000_with_feature_values_独热编码版.csv'))
    df_fs, selected_features = feature_selection(data_dict)
    save_preprocessed_dataset(data_dict, selected_features)
    print("特征工程任务全部圆满完成！")
