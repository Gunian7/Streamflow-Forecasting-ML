# -*- coding: utf-8 -*-
"""
四、集成学习多步预测建模模块 (Simple Average & Stacking)
严格按照指导书规范：
1. 两种集成学习方法：
   - 基于简单平均 (Simple Average) 的径流多步预测模型
   - 基于 Stacking 的径流多步预测模型 (元学习器使用 Ridge / LinearRegression)
2. 预测策略覆盖：
   - (1) 多步直接预测策略: 针对1-7天，简单平均7个模型 + Stacking 7个模型 = 14个模型
   - (2) 多输出预测策略: 简单平均1个多输出模型 + Stacking 1个多输出模型 = 2个模型
   共 2 * 8 = 16 个集成学习模型！
3. 数据严格对齐：
   - 将 ANN、RF 与 LSTM 的输出按统一索引对齐 (LSTM 序列丢弃了前2天)
   - Stacking 元学习器严格基于基学习器在训练集上的预测输出拟合，再在测试集预测，防止标签泄露
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
import torch.nn as nn

# 为兼容 step3 在 __main__ 下序列化的 LSTM 模型类而保留同名定义。
# 本步骤仅使用已导出的预测矩阵，不会再调用该模型。
class LSTMPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2):
        super(LSTMPredictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRED_OUT_DIR = os.path.join(BASE_DIR, 'results', 'predictions')

def calc_nse(y_true, y_pred):
    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    if denom == 0:
        return 0.0
    numer = np.sum((y_true - y_pred) ** 2)
    return 1.0 - (numer / denom)

def evaluate_metrics(y_true, y_pred):
    return {
        'RMSE': float(root_mean_squared_error(y_true, y_pred)),
        'MAE': float(mean_absolute_error(y_true, y_pred)),
        'R2': float(r2_score(y_true, y_pred)),
        'NSE': float(calc_nse(y_true, y_pred))
    }

def run_ensemble():
    in_file = os.path.join(PRED_OUT_DIR, 'individual_models_results.pkl')
    with open(in_file, 'rb') as f:
        bundle = pickle.load(f)
        
    ann = bundle['ann']
    rf = bundle['rf']
    lstm = bundle['lstm']
    
    seq_len = lstm['seq_len']
    offset = seq_len - 1 # 3 - 1 = 2
    
    # 对齐训练集与测试集真实目标
    # y_train 和 y_test 裁剪前 offset 行与 LSTM 严格对齐
    y_train_align = bundle['data_bundle']['y_train_raw'][offset:]
    y_test_align = bundle['data_bundle']['y_test_raw'][offset:]
    
    # 提取对齐后的三个基模型在训练集与测试集上的预测值
    # 1. 直接预测对齐
    ann_tr_dir = ann['preds_train_direct'][offset:]
    ann_te_dir = ann['preds_direct'][offset:]
    
    rf_tr_dir = rf['preds_train_direct'][offset:]
    rf_te_dir = rf['preds_direct'][offset:]
    
    lstm_tr_dir = lstm['preds_train_direct']
    lstm_te_dir = lstm['preds_direct']
    
    # 2. 多输出预测对齐
    ann_tr_mul = ann['preds_train_multi'][offset:]
    ann_te_mul = ann['preds_multi'][offset:]
    
    rf_tr_mul = rf['preds_train_multi'][offset:]
    rf_te_mul = rf['preds_multi'][offset:]
    
    lstm_tr_mul = lstm['preds_train_multi']
    lstm_te_mul = lstm['preds_multi']
    
    print(f"数据对齐完毕: 训练集有效样本 {len(y_train_align)}, 测试集有效样本 {len(y_test_align)}")
    
    # ========================================================
    # 集成模型 1: 简单平均法 (Simple Average)
    # ========================================================
    print("\n--- 构建简单平均法 (Simple Average) 集成模型 ---")
    # (1) 多步直接预测 (7步，分别对应7个直接集成预测)
    avg_preds_direct = np.zeros_like(y_test_align)
    for step in range(7):
        # 简单平均融合: (ANN + RF + LSTM) / 3
        avg_preds_direct[:, step] = (ann_te_dir[:, step] + rf_te_dir[:, step] + lstm_te_dir[:, step]) / 3.0
        
    # (2) 多输出预测 (1个多输出集成模型)
    avg_preds_multi = (ann_te_mul + rf_te_mul + lstm_te_mul) / 3.0
    
    # ========================================================
    # 集成模型 2: Stacking 集成法
    # ========================================================
    print("\n--- 构建 Stacking 集成模型 (Meta-Learner: Ridge Regression) ---")
    stacking_models_direct = {}
    stacking_preds_direct = np.zeros_like(y_test_align)
    
    # (1) 多步直接预测 (7个 Stacking 模型)
    for step in range(7):
        # 构建当前步元学习器的特征矩阵: [ANN_pred, RF_pred, LSTM_pred]
        meta_X_train = np.column_stack([ann_tr_dir[:, step], rf_tr_dir[:, step], lstm_tr_dir[:, step]])
        meta_y_train = y_train_align[:, step]
        
        meta_X_test = np.column_stack([ann_te_dir[:, step], rf_te_dir[:, step], lstm_te_dir[:, step]])
        
        meta_learner = Ridge(alpha=1.0, random_state=42)
        meta_learner.fit(meta_X_train, meta_y_train)
        stacking_models_direct[f'stacking_direct_day{step+1}'] = meta_learner
        
        pred_step = meta_learner.predict(meta_X_test)
        # 径流非负物理约束
        pred_step = np.clip(pred_step, a_min=0, a_max=None)
        stacking_preds_direct[:, step] = pred_step
        print(f"Stacking 直接预测 Day {step+1} 元学习器权重: ANN={meta_learner.coef_[0]:.3f}, RF={meta_learner.coef_[1]:.3f}, LSTM={meta_learner.coef_[2]:.3f}, 截距={meta_learner.intercept_:.3f}")
        
    # (2) 多输出预测 (1个 Stacking 模型)
    stacking_preds_multi = np.zeros_like(y_test_align)
    stacking_models_multi = []
    # 多输出下的 Stacking: 对每个输出步建立元回归映射
    for step in range(7):
        meta_X_train = np.column_stack([ann_tr_mul[:, step], rf_tr_mul[:, step], lstm_tr_mul[:, step]])
        meta_y_train = y_train_align[:, step]
        meta_X_test = np.column_stack([ann_te_mul[:, step], rf_te_mul[:, step], lstm_te_mul[:, step]])
        
        meta_model = Ridge(alpha=1.0, random_state=42)
        meta_model.fit(meta_X_train, meta_y_train)
        stacking_models_multi.append(meta_model)
        pred_m_step = np.clip(meta_model.predict(meta_X_test), a_min=0, a_max=None)
        stacking_preds_multi[:, step] = pred_m_step
        
    # 保存所有预测与对齐数据
    ensemble_results = {
        'y_test': y_test_align,
        'y_train': y_train_align,
        # 基模型对齐测试集预测
        'ann_direct': ann_te_dir,
        'ann_multi': ann_te_mul,
        'rf_direct': rf_te_dir,
        'rf_multi': rf_te_mul,
        'lstm_direct': lstm_te_dir,
        'lstm_multi': lstm_te_mul,
        # 简单平均集成预测
        'avg_direct': avg_preds_direct,
        'avg_multi': avg_preds_multi,
        # Stacking 集成预测
        'stacking_direct': stacking_preds_direct,
        'stacking_multi': stacking_preds_multi,
        # 元学习器模型集合
        'stacking_models_direct': stacking_models_direct,
        'stacking_models_multi': stacking_models_multi
    }
    
    save_path = os.path.join(PRED_OUT_DIR, 'all_models_unified_predictions.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(ensemble_results, f)
        
    print(f"\n全部16个集成模型预测构建完毕并保存至: {save_path}")

if __name__ == '__main__':
    run_ensemble()
