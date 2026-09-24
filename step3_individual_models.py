# -*- coding: utf-8 -*-
"""
三、径流多步预测建模与网格搜索调优模块 (ANN, Random Forest, LSTM)
严格对应指导书要求：
1. 预测要求：
   - 多步直接预测 (Direct Multi-step Forecasting): 针对未来1-7天，分别训练7个独立模型；
   - 多输出预测 (Multi-output Forecasting): 1个模型一次性联合输出未来1-7天的预测值。
2. 三种基础模型：
   - ANN (人工神经网络 - MLPRegressor)
   - Random Forest (随机森林 - RandomForestRegressor)
   - LSTM (长短期记忆递归神经网络 - PyTorch LSTM)
3. 模型数量要求：3种方法 * 8（7个直接 + 1个多输出）= 24个个体模型。
4. 使用基于时间序列的时间切分进行网格搜索交叉验证 (TimeSeriesSplit Grid Search CV)，确定最佳超参数。
5. 保存所有模型的预测结果、训练历史与评价指标。
"""

import os
import pickle
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

# 设置随机种子以保证实验可复现性
np.random.seed(42)
torch.manual_seed(42)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'results', 'processed_data', 'processed_dataset.pkl')
MODEL_OUT_DIR = os.path.join(BASE_DIR, 'results', 'models')
PRED_OUT_DIR = os.path.join(BASE_DIR, 'results', 'predictions')
os.makedirs(MODEL_OUT_DIR, exist_ok=True)
os.makedirs(PRED_OUT_DIR, exist_ok=True)

def calc_nse(y_true, y_pred):
    """计算水文领域的黄金评价指标: 纳什效率系数 (Nash-Sutcliffe Efficiency)"""
    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    if denom == 0:
        return 0.0
    numer = np.sum((y_true - y_pred) ** 2)
    return 1.0 - (numer / denom)

def evaluate_predictions(y_true, y_pred):
    """计算单个目标或多输出的四大评价指标"""
    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    nse = calc_nse(y_true, y_pred)
    return {'RMSE': rmse, 'MAE': mae, 'R2': r2, 'NSE': nse}

def load_data():
    with open(DATA_PATH, 'rb') as f:
        data_bundle = pickle.load(f)
    return data_bundle

# ==========================================
# 1. 人工神经网络 (ANN)
# ==========================================
def run_ann(data_bundle):
    print("\n==================== 开始 ANN 建模 (MLPRegressor) ====================")
    X_train = data_bundle['X_train_mm']
    X_test = data_bundle['X_test_mm']
    y_train_mm = data_bundle['y_train_mm']
    y_test_mm = data_bundle['y_test_mm']
    scaler_y = data_bundle['scaler_y']
    
    # 时序网格搜索
    tscv = TimeSeriesSplit(n_splits=3)
    param_grid = {
        'hidden_layer_sizes': [(64, 32), (128, 64)],
        'alpha': [0.001, 0.01],
        'learning_rate_init': [0.005, 0.01]
    }
    
    print("正在进行 ANN 网格搜索 (TimeSeriesSplit)...")
    base_mlp = MLPRegressor(max_iter=300, random_state=42, early_stopping=True)
    grid = GridSearchCV(base_mlp, param_grid, cv=tscv, scoring='neg_root_mean_squared_error', n_jobs=-1)
    grid.fit(X_train, y_train_mm[:, 0])
    best_params = grid.best_params_
    print(f"ANN 最佳超参数: {best_params}")
    
    ann_models = {}
    preds_direct_scaled = np.zeros((len(X_test), 7))
    preds_train_direct_scaled = np.zeros((len(X_train), 7))
    
    # (1) 多步直接预测: 训练7个独立模型
    print("训练 ANN 直接预测模型 (1-7天)...")
    for step in range(7):
        y_train_step = y_train_mm[:, step]
        model = MLPRegressor(**best_params, max_iter=400, random_state=42 + step)
        model.fit(X_train, y_train_step)
        ann_models[f'ann_direct_day{step+1}'] = model
        preds_train_direct_scaled[:, step] = model.predict(X_train)
        preds_direct_scaled[:, step] = model.predict(X_test)
        
    # 反归一化
    preds_direct = scaler_y.inverse_transform(preds_direct_scaled)
    preds_train_direct = scaler_y.inverse_transform(preds_train_direct_scaled)
    
    # (2) 多输出预测: 1个模型输出所有7步
    print("训练 ANN 多输出预测模型...")
    mlp_multi = MLPRegressor(**best_params, max_iter=500, random_state=42)
    mlp_multi.fit(X_train, y_train_mm)
    ann_models['ann_multi'] = mlp_multi
    preds_train_multi_scaled = mlp_multi.predict(X_train)
    preds_multi_scaled = mlp_multi.predict(X_test)
    
    preds_multi = scaler_y.inverse_transform(preds_multi_scaled)
    preds_train_multi = scaler_y.inverse_transform(preds_train_multi_scaled)
    
    return {
        'models': ann_models,
        'best_params': best_params,
        'preds_direct': preds_direct,
        'preds_train_direct': preds_train_direct,
        'preds_multi': preds_multi,
        'preds_train_multi': preds_train_multi
    }

# ==========================================
# 2. 随机森林 (Random Forest)
# ==========================================
def run_rf(data_bundle):
    print("\n==================== 开始 Random Forest 建模 ====================")
    # 随机森林基于树模型，直接使用原始尺度训练便于解释且树结构对尺度不敏感
    X_train = data_bundle['X_train_raw']
    X_test = data_bundle['X_test_raw']
    y_train = data_bundle['y_train_raw']
    y_test = data_bundle['y_test_raw']
    
    tscv = TimeSeriesSplit(n_splits=3)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 15, 20],
        'min_samples_split': [2, 5]
    }
    
    print("正在进行 Random Forest 网格搜索...")
    base_rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    grid = GridSearchCV(base_rf, param_grid, cv=tscv, scoring='neg_root_mean_squared_error', n_jobs=-1)
    grid.fit(X_train, y_train[:, 0])
    best_params = grid.best_params_
    print(f"Random Forest 最佳超参数: {best_params}")
    
    rf_models = {}
    preds_direct = np.zeros((len(X_test), 7))
    preds_train_direct = np.zeros((len(X_train), 7))
    
    # (1) 多步直接预测
    print("训练 Random Forest 直接预测模型 (1-7天)...")
    for step in range(7):
        y_train_step = y_train[:, step]
        model = RandomForestRegressor(**best_params, random_state=42 + step, n_jobs=-1)
        model.fit(X_train, y_train_step)
        rf_models[f'rf_direct_day{step+1}'] = model
        preds_train_direct[:, step] = model.predict(X_train)
        preds_direct[:, step] = model.predict(X_test)
        
    # (2) 多输出预测
    print("训练 Random Forest 多输出预测模型...")
    rf_multi = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
    rf_multi.fit(X_train, y_train)
    rf_models['rf_multi'] = rf_multi
    preds_train_multi = rf_multi.predict(X_train)
    preds_multi = rf_multi.predict(X_test)
    
    return {
        'models': rf_models,
        'best_params': best_params,
        'preds_direct': preds_direct,
        'preds_train_direct': preds_train_direct,
        'preds_multi': preds_multi,
        'preds_train_multi': preds_train_multi
    }

# ==========================================
# 3. 深度学习长短期记忆网络 (LSTM)
# ==========================================
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
        # x shape: [batch_size, seq_len, input_dim]
        out, _ = self.lstm(x)
        # 取最后一个时间步输出
        out = self.fc(out[:, -1, :])
        return out

def train_lstm_model(model, train_loader, val_loader, epochs=60, lr=0.005, weight_decay=1e-4):
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_loss = float('inf')
    best_weights = None
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for bx, by in train_loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * bx.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # 验证
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for bx, by in val_loader:
                    out = model(bx)
                    loss = criterion(out, by)
                    val_loss += loss.item() * bx.size(0)
            val_loss /= len(val_loader.dataset)
            scheduler.step(val_loss)
            
            if val_loss < best_loss:
                best_loss = val_loss
                best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
    if best_weights is not None:
        model.load_state_dict(best_weights)
    return model

def run_lstm(data_bundle):
    print("\n==================== 开始 LSTM 建模 (PyTorch) ====================")
    X_train = data_bundle['X_train_mm']
    X_test = data_bundle['X_test_mm']
    y_train_mm = data_bundle['y_train_mm']
    y_test_mm = data_bundle['y_test_mm']
    scaler_y = data_bundle['scaler_y']
    
    # 构造 LSTM 的 3D 输入序列 [Batch, Seq_len, Features]
    # 由于我们在特征工程中已经构建了全面的历史滞后和滑动特征，单个样本本身包含完整动态上下文
    # 为保持时间序列因果性，我们设计 Seq_len=3 的滑动窗口输入
    SEQ_LEN = 3
    def make_sequences(X, y):
        xs, ys = [], []
        for i in range(len(X) - SEQ_LEN + 1):
            xs.append(X[i : i + SEQ_LEN])
            ys.append(y[i + SEQ_LEN - 1])
        return np.array(xs), np.array(ys)
        
    X_tr_seq, y_tr_seq = make_sequences(X_train, y_train_mm)
    X_te_seq, y_te_seq = make_sequences(X_test, y_test_mm)
    
    # 划分内部验证集进行超参数检验
    val_size = int(len(X_tr_seq) * 0.2)
    X_tr_inner, y_tr_inner = X_tr_seq[:-val_size], y_tr_seq[:-val_size]
    X_val_inner, y_val_inner = X_tr_seq[-val_size:], y_tr_seq[-val_size:]
    
    input_dim = X_train.shape[1]
    
    # 网格搜索简选超参数
    param_candidates = [
        {'hidden_dim': 64, 'num_layers': 1, 'lr': 0.005},
        {'hidden_dim': 128, 'num_layers': 2, 'lr': 0.003},
        {'hidden_dim': 64, 'num_layers': 2, 'lr': 0.005}
    ]
    
    print("正在进行 LSTM 超参数网格搜索...")
    best_cfg = None
    min_val_err = float('inf')
    
    for cfg in param_candidates:
        model = LSTMPredictor(input_dim, cfg['hidden_dim'], cfg['num_layers'], output_dim=1)
        t_loader = DataLoader(TensorDataset(torch.FloatTensor(X_tr_inner), torch.FloatTensor(y_tr_inner[:, 0:1])), batch_size=32, shuffle=False)
        v_loader = DataLoader(TensorDataset(torch.FloatTensor(X_val_inner), torch.FloatTensor(y_val_inner[:, 0:1])), batch_size=32, shuffle=False)
        train_lstm_model(model, t_loader, v_loader, epochs=25, lr=cfg['lr'])
        
        model.eval()
        with torch.no_grad():
            v_pred = model(torch.FloatTensor(X_val_inner)).numpy()
        err = root_mean_squared_error(y_val_inner[:, 0], v_pred[:, 0])
        print(f"配置 {cfg} 验证集 RMSE: {err:.4f}")
        if err < min_val_err:
            min_val_err = err
            best_cfg = cfg
            
    print(f"LSTM 最佳超参数配置: {best_cfg}")
    
    lstm_models = {}
    train_full_loader = DataLoader(TensorDataset(torch.FloatTensor(X_tr_seq), torch.FloatTensor(y_tr_seq)), batch_size=32, shuffle=False)
    
    preds_direct_scaled = np.zeros((len(X_te_seq), 7))
    preds_train_direct_scaled = np.zeros((len(X_tr_seq), 7))
    
    # (1) 多步直接预测 (7个模型)
    print("训练 LSTM 直接预测模型 (1-7天)...")
    for step in range(7):
        model = LSTMPredictor(input_dim, best_cfg['hidden_dim'], best_cfg['num_layers'], output_dim=1)
        sub_loader = DataLoader(TensorDataset(torch.FloatTensor(X_tr_seq), torch.FloatTensor(y_tr_seq[:, step:step+1])), batch_size=32, shuffle=False)
        train_lstm_model(model, sub_loader, val_loader=None, epochs=40, lr=best_cfg['lr'])
        lstm_models[f'lstm_direct_day{step+1}'] = model
        
        model.eval()
        with torch.no_grad():
            preds_train_direct_scaled[:, step] = model(torch.FloatTensor(X_tr_seq)).numpy().flatten()
            preds_direct_scaled[:, step] = model(torch.FloatTensor(X_te_seq)).numpy().flatten()
            
    # (2) 多输出预测 (1个模型)
    print("训练 LSTM 多输出预测模型...")
    lstm_multi = LSTMPredictor(input_dim, best_cfg['hidden_dim'], best_cfg['num_layers'], output_dim=7)
    train_lstm_model(lstm_multi, train_full_loader, val_loader=None, epochs=45, lr=best_cfg['lr'])
    lstm_models['lstm_multi'] = lstm_multi
    
    lstm_multi.eval()
    with torch.no_grad():
        preds_train_multi_scaled = lstm_multi(torch.FloatTensor(X_tr_seq)).numpy()
        preds_multi_scaled = lstm_multi(torch.FloatTensor(X_te_seq)).numpy()
        
    # 反归一化
    preds_direct = scaler_y.inverse_transform(preds_direct_scaled)
    preds_train_direct = scaler_y.inverse_transform(preds_train_direct_scaled)
    preds_multi = scaler_y.inverse_transform(preds_multi_scaled)
    preds_train_multi = scaler_y.inverse_transform(preds_train_multi_scaled)
    
    # 对齐测试集真实值：由于序列窗口为 SEQ_LEN=3，前 (SEQ_LEN-1) 个点没有预测，对齐对应的真实值
    y_test_aligned = data_bundle['y_test_raw'][SEQ_LEN-1:]
    y_train_aligned = data_bundle['y_train_raw'][SEQ_LEN-1:]
    
    return {
        'models': lstm_models,
        'best_params': best_cfg,
        'seq_len': SEQ_LEN,
        'y_test_aligned': y_test_aligned,
        'y_train_aligned': y_train_aligned,
        'preds_direct': preds_direct,
        'preds_train_direct': preds_train_direct,
        'preds_multi': preds_multi,
        'preds_train_multi': preds_train_multi
    }

if __name__ == '__main__':
    data_bundle = load_data()
    
    # 运行三种个体方法
    start_time = time.time()
    res_ann = run_ann(data_bundle)
    res_rf = run_rf(data_bundle)
    res_lstm = run_lstm(data_bundle)
    
    # 保存预测结果与模型信息
    results_bundle = {
        'ann': res_ann,
        'rf': res_rf,
        'lstm': res_lstm,
        'data_bundle': data_bundle
    }
    
    out_file = os.path.join(PRED_OUT_DIR, 'individual_models_results.pkl')
    with open(out_file, 'wb') as f:
        pickle.dump(results_bundle, f)
        
    print(f"\n全部24个个体预测模型训练评估完毕！总耗时: {time.time() - start_time:.2f} 秒")
    print(f"结果已存储至: {out_file}")
