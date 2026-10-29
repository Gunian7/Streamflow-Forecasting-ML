# 提升第三阶段精度至 R² ≥ 0.80

**目标**：将第三阶段第1天预测精度从 R²=0.7778 提升至 ≥0.80，同时改善整体 1-7 天精度。

---

## 当前状态与假设

### 现状
- **当前最优**：LSTM 直接策略第1天 R²=0.7778, RMSE=396.9
- **实现方式**：PseudoLSTM（sklearn MLPRegressor + 时序特征模拟）
- **数据**：已修正（真实积雪 + 当天径流 + 正确日期对齐）
- **特征数**：71 维（包含 lag、滚动统计、积雪派生）

### 瓶颈分析
1. **模型限制**：PseudoLSTM 本质是 MLP，无真正的时序记忆机制
2. **特征饱和**：当前特征工程已较完善，单纯增加特征空间有限
3. **集成潜力**：Stacking R²=0.7712，接近单模型上限

### 假设
- 训练集 1440 条，测试集 359 条，数据量足够训练深度模型
- 可以安装 PyTorch（轻量、易用）
- 保留现有代码，新方法独立实现在 `phase3_enhanced/` 目录

---

## 架构与方法

采用**四管齐下**策略：

1. **真实深度学习模型**：实现 PyTorch LSTM/GRU，替代 PseudoLSTM
2. **Temporal Fusion Transformer**：尝试时序预测的 SOTA 架构
3. **增强特征工程**：时序分解、交互特征、窗口统计优化
4. **集成优化**：加入 XGBoost/LightGBM，优化 Stacking 元学习器

预期提升路径：
- 真实 LSTM：0.778 → 0.79-0.81（+0.01-0.03）
- TFT：0.778 → 0.80-0.82（+0.02-0.04）
- 特征增强：额外 +0.01-0.02
- 集成优化：额外 +0.01-0.02

**目标可达性**：高（多方法叠加，至少一条路径能达到 0.80）

---

## 实施步骤

### 阶段一：环境准备与数据增强（10分钟）

#### 任务 1.1：安装 PyTorch 和辅助库
```bash
cd "F:/Desktop/学习/机器学习课程实践"
uv pip install --python .venv/Scripts/python.exe torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/Scripts/python.exe optuna xgboost lightgbm
```

**验证**：
```python
import torch
import xgboost as xgb
import lightgbm as lgb
import optuna
print(f"PyTorch: {torch.__version__}")
print(f"XGBoost: {xgb.__version__}")
print(f"LightGBM: {lgb.__version__}")
print(f"Optuna: {optuna.__version__}")
# 预期输出：所有库版本号，无报错
```

#### 任务 1.2：创建增强特征模块
**文件**：`code/enhanced_features.py`

```python
"""增强特征工程：时序分解、交互项、窗口优化"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def add_enhanced_features(df):
    """在现有特征基础上添加增强特征
    
    新增特征：
    - 指数加权移动平均（EWMA）
    - 趋势与季节分解
    - 交互特征（降水×温度、积雪×正积温）
    - 滞后差分（捕获变化率）
    """
    df = df.copy()
    
    # 1. 指数加权移动平均（捕获短期趋势）
    for col in ['Discharge', 'Prcp', 'Tmax']:
        df[f'{col}_ewm3'] = df[col].ewm(span=3, adjust=False).mean()
        df[f'{col}_ewm7'] = df[col].ewm(span=7, adjust=False).mean()
    
    # 2. 滞后差分（捕获变化率）
    for col in ['Discharge', 'Prcp', 'Swe']:
        df[f'{col}_diff1'] = df[col].diff(1)
        df[f'{col}_diff3'] = df[col].diff(3)
    
    # 3. 交互特征（物理意义）
    df['Prcp_x_Tmax'] = df['Prcp'] * df['Tmax']  # 雨热同期指数
    df['Swe_x_PDD'] = df['Swe'] * df['Tmax'].clip(lower=0)  # 融雪潜力
    df['Prcp_x_Swe'] = df['Prcp'] * df['Swe']  # 降水对积雪的补充
    
    # 4. 滚动标准差（捕获波动性）
    for win in [3, 7]:
        df[f'Discharge_std_{win}'] = df['Discharge'].rolling(win).std()
        df[f'Prcp_std_{win}'] = df['Prcp'].rolling(win).std()
    
    # 5. 累积特征
    df['Prcp_cum3'] = df['Prcp'].rolling(3).sum()
    df['Prcp_cum7'] = df['Prcp'].rolling(7).sum()
    df['PDD_cum3'] = df['Tmax'].clip(lower=0).rolling(3).sum()
    
    # 6. 比率特征
    df['Q_to_P_ratio'] = df['Discharge'] / (df['Prcp'].rolling(7).sum() + 1)
    
    return df


def get_enhanced_dataset():
    """获取增强特征后的训练集和测试集"""
    import os
    os.environ.setdefault('RUNOFF_DATA_PATH',
                          str(ROOT / '01047000_fixed.csv'))
    import sys
    sys.path.insert(0, str(ROOT / 'code'))
    import phase2_v5_data as d5
    
    train, test, base_cols, meta = d5.build_dataset_v5()
    
    # 添加增强特征
    train_enh = add_enhanced_features(train)
    test_enh = add_enhanced_features(test)
    
    # 合并列名
    new_cols = [c for c in train_enh.columns 
                if c not in base_cols and not c.startswith('Target_') 
                and c != 'Date']
    enhanced_cols = base_cols + new_cols
    
    # 去除 NaN（由于 diff 和 rolling 产生）
    train_enh = train_enh.dropna(subset=enhanced_cols)
    test_enh = test_enh.dropna(subset=enhanced_cols)
    
    meta['enhanced_features'] = len(new_cols)
    meta['total_features'] = len(enhanced_cols)
    
    return train_enh, test_enh, enhanced_cols, meta


if __name__ == '__main__':
    train, test, cols, meta = get_enhanced_dataset()
    print(f"训练集: {len(train)} 条")
    print(f"测试集: {len(test)} 条")
    print(f"基础特征: {meta['特征数量']}")
    print(f"新增特征: {meta['enhanced_features']}")
    print(f"总特征数: {meta['total_features']}")
    print(f"\n新增特征列表:")
    new = [c for c in cols if c not in ['Date'] and 
           ('_ewm' in c or '_diff' in c or '_x_' in c or 
            '_std_' in c or '_cum' in c or 'ratio' in c)]
    for c in new:
        print(f"  - {c}")
```

**验证**：
```bash
cd "F:/Desktop/学习/机器学习课程实践"
.venv/Scripts/python code/enhanced_features.py
# 预期输出：
# 训练集: ~1430 条
# 测试集: ~355 条
# 基础特征: 71
# 新增特征: ~20
# 总特征数: ~91
# 新增特征列表: [Discharge_ewm3, Prcp_x_Tmax, ...]
```

---

### 阶段二：真实 LSTM/GRU 模型（20分钟）

#### 任务 2.1：实现 PyTorch LSTM 模型
**文件**：`code/pytorch_lstm.py`

```python
"""真实的 PyTorch LSTM/GRU 模型"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


class TimeSeriesDataset(Dataset):
    """时序数据集"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMRegressor(nn.Module):
    """LSTM 回归模型（支持单输出和多输出）"""
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, 
                 output_dim=1, dropout=0.2, bidirectional=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        lstm_out_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Sequential(
            nn.Linear(lstm_out_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        # x shape: (batch, features) -> reshape to (batch, seq_len=1, features)
        x = x.unsqueeze(1)
        
        # LSTM
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]  # 取最后时刻
        
        # 全连接
        out = self.fc(lstm_out)
        return out.squeeze(-1) if out.shape[-1] == 1 else out


class LSTMTrainer:
    """LSTM 训练器（支持早停和学习率调度）"""
    def __init__(self, model, device='cpu', lr=0.001):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=False
        )
        self.criterion = nn.MSELoss()
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
    
    def fit(self, X, y, X_val=None, y_val=None, epochs=100, batch_size=32, 
            early_stopping_patience=15, verbose=True):
        """训练模型"""
        # 标准化
        X_scaled = self.scaler_X.fit_transform(X)
        y = y.reshape(-1, 1) if y.ndim == 1 else y
        y_scaled = self.scaler_y.fit_transform(y)
        
        train_dataset = TimeSeriesDataset(X_scaled, y_scaled)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        if X_val is not None:
            X_val_scaled = self.scaler_X.transform(X_val)
            y_val = y_val.reshape(-1, 1) if y_val.ndim == 1 else y_val
            y_val_scaled = self.scaler_y.transform(y_val)
            val_dataset = TimeSeriesDataset(X_val_scaled, y_val_scaled)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # 训练
            self.model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                self.optimizer.zero_grad()
                pred = self.model(X_batch)
                loss = self.criterion(pred, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # 验证
            if X_val is not None:
                self.model.eval()
                val_loss = 0
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch = X_batch.to(self.device)
                        y_batch = y_batch.to(self.device)
                        pred = self.model(X_batch)
                        loss = self.criterion(pred, y_batch)
                        val_loss += loss.item()
                val_loss /= len(val_loader)
                
                self.scheduler.step(val_loss)
                
                # 早停
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self.best_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1
                
                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    self.model.load_state_dict(self.best_state)
                    break
                
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        return self
    
    def predict(self, X):
        """预测"""
        self.model.eval()
        X_scaled = self.scaler_X.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        with torch.no_grad():
            pred_scaled = self.model(X_tensor).cpu().numpy()
        
        pred = self.scaler_y.inverse_transform(
            pred_scaled.reshape(-1, 1) if pred_scaled.ndim == 1 else pred_scaled
        )
        return pred.ravel() if pred.shape[1] == 1 else pred


# 使用示例（供测试）
if __name__ == '__main__':
    # 简单测试
    X_dummy = np.random.randn(100, 10)
    y_dummy = np.random.randn(100)
    
    model = LSTMRegressor(input_dim=10, hidden_dim=32, num_layers=2, output_dim=1)
    trainer = LSTMTrainer(model, lr=0.01)
    trainer.fit(X_dummy, y_dummy, epochs=5, verbose=True)
    
    pred = trainer.predict(X_dummy[:5])
    print(f"预测形状: {pred.shape}")
    print(f"预测示例: {pred[:3]}")
    print("✓ PyTorch LSTM 模型测试通过")
```

**验证**：
```bash
cd "F:/Desktop/学习/机器学习课程实践"
.venv/Scripts/python code/pytorch_lstm.py
# 预期输出：
# Epoch 5/5 - Train Loss: ...
# 预测形状: (5,)
# 预测示例: [...]
# ✓ PyTorch LSTM 模型测试通过
```

#### 任务 2.2：训练真实 LSTM 并评估
**文件**：`code/train_pytorch_lstm.py`

```python
"""训练 PyTorch LSTM 并与 PseudoLSTM 对比"""
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import TimeSeriesSplit

sys.path.insert(0, str(Path(__file__).resolve().parent))

from enhanced_features import get_enhanced_dataset
from pytorch_lstm import LSTMRegressor, LSTMTrainer

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'results/phase3_enhanced'
OUT.mkdir(parents=True, exist_ok=True)


def metrics(y, p):
    y = np.asarray(y, float).ravel()
    p = np.asarray(p, float).ravel()
    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return {
        'RMSE': float(np.sqrt(np.mean((y - p) ** 2))),
        'MAE': float(np.mean(np.abs(y - p))),
        'R²': 1 - ss_res / ss_tot
    }


def train_lstm_direct():
    """直接策略：为每个预见期训练独立 LSTM"""
    train, test, cols, meta = get_enhanced_dataset()
    
    X = train[cols].to_numpy(float)
    Xt = test[cols].to_numpy(float)
    
    # 使用时序交叉验证选择超参数
    tscv = TimeSeriesSplit(n_splits=3, gap=7)
    
    results = []
    detail = []
    
    print('\n真实 LSTM 直接策略训练:')
    for h in range(1, 8):
        y_train = train[f'Target_lead{h}'].to_numpy(float)
        y_test = test[f'Target_lead{h}'].to_numpy(float)
        
        # 创建模型（基于经验的较优配置）
        model = LSTMRegressor(
            input_dim=len(cols),
            hidden_dim=128,
            num_layers=2,
            output_dim=1,
            dropout=0.3,
            bidirectional=False
        )
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        trainer = LSTMTrainer(model, device=device, lr=0.001)
        
        # 训练（使用 20% 作为验证集）
        n_val = int(len(X) * 0.2)
        X_tr, X_val = X[:-n_val], X[-n_val:]
        y_tr, y_val = y_train[:-n_val], y_train[-n_val:]
        
        trainer.fit(X_tr, y_tr, X_val, y_val, epochs=100, 
                   batch_size=32, early_stopping_patience=15, verbose=False)
        
        # 预测
        pred = np.clip(trainer.predict(Xt), 0, None)
        
        m = metrics(y_test, pred)
        results.append({
            '策略': 'PyTorch-LSTM-直接',
            '预见期（天）': h,
            **m
        })
        
        for i, (obs, pre) in enumerate(zip(y_test, pred)):
            detail.append({
                '策略': 'PyTorch-LSTM-直接',
                '预见期（天）': h,
                '样本序号': i,
                '实测': obs,
                '预测': pre
            })
        
        # 保存模型
        joblib.dump(trainer, OUT / f'pytorch_lstm_direct_{h}.joblib', compress=3)
        print(f'  第{h}天: R²={m["R²"]:.4f}, RMSE={m["RMSE"]:.1f}')
    
    return pd.DataFrame(results), pd.DataFrame(detail)


def train_lstm_multi():
    """多输出策略：单模型预测 7 天"""
    train, test, cols, meta = get_enhanced_dataset()
    
    X = train[cols].to_numpy(float)
    Xt = test[cols].to_numpy(float)
    Y_train = np.column_stack([train[f'Target_lead{h}'].to_numpy(float) 
                                for h in range(1, 8)])
    Y_test = np.column_stack([test[f'Target_lead{h}'].to_numpy(float) 
                              for h in range(1, 8)])
    
    print('\n真实 LSTM 多输出策略训练:')
    
    model = LSTMRegressor(
        input_dim=len(cols),
        hidden_dim=128,
        num_layers=2,
        output_dim=7,
        dropout=0.3,
        bidirectional=False
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = LSTMTrainer(model, device=device, lr=0.001)
    
    n_val = int(len(X) * 0.2)
    X_tr, X_val = X[:-n_val], X[-n_val:]
    Y_tr, Y_val = Y_train[:-n_val], Y_train[-n_val:]
    
    trainer.fit(X_tr, Y_tr, X_val, Y_val, epochs=100, 
               batch_size=32, early_stopping_patience=15, verbose=False)
    
    pred_multi = np.clip(trainer.predict(Xt), 0, None)
    
    results = []
    detail = []
    
    for h in range(1, 8):
        y_test = Y_test[:, h-1]
        pred_h = pred_multi[:, h-1]
        
        m = metrics(y_test, pred_h)
        results.append({
            '策略': 'PyTorch-LSTM-多输出',
            '预见期（天）': h,
            **m
        })
        
        for i, (obs, pre) in enumerate(zip(y_test, pred_h)):
            detail.append({
                '策略': 'PyTorch-LSTM-多输出',
                '预见期（天）': h,
                '样本序号': i,
                '实测': obs,
                '预测': pre
            })
        
        print(f'  第{h}天: R²={m["R²"]:.4f}, RMSE={m["RMSE"]:.1f}')
    
    joblib.dump(trainer, OUT / 'pytorch_lstm_multi.joblib', compress=3)
    
    return pd.DataFrame(results), pd.DataFrame(detail)


def main():
    t0 = time.time()
    
    print('='*74)
    print('增强版：真实 PyTorch LSTM 训练')
    print('='*74)
    
    # 直接策略
    r1, d1 = train_lstm_direct()
    
    # 多输出策略
    r2, d2 = train_lstm_multi()
    
    # 合并保存
    all_results = pd.concat([r1, r2], ignore_index=True)
    all_detail = pd.concat([d1, d2], ignore_index=True)
    
    all_results.to_csv(OUT / 'pytorch_lstm_metrics.csv', 
                       index=False, encoding='utf-8-sig')
    all_detail.to_csv(OUT / 'pytorch_lstm_predictions.csv', 
                      index=False, encoding='utf-8-sig')
    
    print('\n' + '='*74)
    print('PyTorch LSTM 结果汇总')
    print('='*74)
    pivot = all_results.pivot_table(
        index='预见期（天）',
        columns='策略',
        values='R²'
    ).round(4)
    print(pivot.to_string())
    
    # 对比 PseudoLSTM
    print('\n与 PseudoLSTM 对比（第1天）:')
    old = pd.read_csv(ROOT / 'results/phase3_lstm/lstm_metrics.csv')
    old_d1 = old[(old['策略']=='直接') & (old['预见期（天）']==1)].iloc[0]
    new_d1 = all_results[(all_results['策略']=='PyTorch-LSTM-直接') & 
                         (all_results['预见期（天）']==1)].iloc[0]
    
    print(f"  PseudoLSTM:   R²={old_d1['R²']:.4f}, RMSE={old_d1['RMSE']:.1f}")
    print(f"  PyTorch LSTM: R²={new_d1['R²']:.4f}, RMSE={new_d1['RMSE']:.1f}")
    print(f"  提升: ΔR²={new_d1['R²']-old_d1['R²']:+.4f}, ΔRMSE={new_d1['RMSE']-old_d1['RMSE']:+.1f}")
    
    if new_d1['R²'] >= 0.80:
        print(f"\n🎉 已达成目标！第1天 R² = {new_d1['R²']:.4f} ≥ 0.80")
    else:
        print(f"\n⚠ 需继续优化，当前 R² = {new_d1['R²']:.4f} < 0.80")
    
    print(f'\n总耗时: {time.time()-t0:.0f}s')
    print(f'结果保存至: {OUT}')


if __name__ == '__main__':
    main()
```

**验证**：
```bash
cd "F:/Desktop/学习/机器学习课程实践"
.venv/Scripts/python code/train_pytorch_lstm.py
# 预期输出：
# 真实 LSTM 直接策略训练:
#   第1天: R²=0.XXXX, RMSE=XXX.X
#   ...
# PyTorch LSTM 结果汇总
# ...
# 与 PseudoLSTM 对比（第1天）:
#   提升: ΔR²=+0.XXXX
# 总耗时: XXs
```

---

### 阶段三：XGBoost/LightGBM 集成（15分钟）

#### 任务 3.1：训练 XGBoost 和 LightGBM
**文件**：`code/train_boosting.py`

```python
"""训练 XGBoost 和 LightGBM 基学习器"""
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit

sys.path.insert(0, str(Path(__file__).resolve().parent))

from enhanced_features import get_enhanced_dataset

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'results/phase3_enhanced'
OUT.mkdir(parents=True, exist_ok=True)


def metrics(y, p):
    y = np.asarray(y, float).ravel()
    p = np.asarray(p, float).ravel()
    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return {
        'RMSE': float(np.sqrt(np.mean((y - p) ** 2))),
        'MAE': float(np.mean(np.abs(y - p))),
        'R²': 1 - ss_res / ss_tot
    }


def train_xgboost():
    """训练 XGBoost 直接策略"""
    train, test, cols, meta = get_enhanced_dataset()
    
    X = train[cols].to_numpy(float)
    Xt = test[cols].to_numpy(float)
    
    results = []
    
    print('\nXGBoost 直接策略训练:')
    for h in range(1, 8):
        y_train = train[f'Target_lead{h}'].to_numpy(float)
        y_test = test[f'Target_lead{h}'].to_numpy(float)
        
        model = xgb.XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            early_stopping_rounds=30
        )
        
        # 使用验证集早停
        n_val = int(len(X) * 0.2)
        X_tr, X_val = X[:-n_val], X[-n_val:]
        y_tr, y_val = y_train[:-n_val], y_train[-n_val:]
        
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        
        pred = np.clip(model.predict(Xt), 0, None)
        
        m = metrics(y_test, pred)
        results.append({
            '模型': 'XGBoost',
            '策略': '直接',
            '预见期（天）': h,
            **m
        })
        
        joblib.dump(model, OUT / f'xgboost_direct_{h}.joblib', compress=3)
        print(f'  第{h}天: R²={m["R²"]:.4f}, RMSE={m["RMSE"]:.1f}')
    
    return pd.DataFrame(results)


def train_lightgbm():
    """训练 LightGBM 直接策略"""
    train, test, cols, meta = get_enhanced_dataset()
    
    X = train[cols].to_numpy(float)
    Xt = test[cols].to_numpy(float)
    
    results = []
    
    print('\nLightGBM 直接策略训练:')
    for h in range(1, 8):
        y_train = train[f'Target_lead{h}'].to_numpy(float)
        y_test = test[f'Target_lead{h}'].to_numpy(float)
        
        model = lgb.LGBMRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        
        n_val = int(len(X) * 0.2)
        X_tr, X_val = X[:-n_val], X[-n_val:]
        y_tr, y_val = y_train[:-n_val], y_train[-n_val:]
        
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
        
        pred = np.clip(model.predict(Xt), 0, None)
        
        m = metrics(y_test, pred)
        results.append({
            '模型': 'LightGBM',
            '策略': '直接',
            '预见期（天）': h,
            **m
        })
        
        joblib.dump(model, OUT / f'lightgbm_direct_{h}.joblib', compress=3)
        print(f'  第{h}天: R²={m["R²"]:.4f}, RMSE={m["RMSE"]:.1f}')
    
    return pd.DataFrame(results)


def main():
    t0 = time.time()
    
    print('='*74)
    print('训练梯度提升树模型（XGBoost & LightGBM）')
    print('='*74)
    
    r1 = train_xgboost()
    r2 = train_lightgbm()
    
    results = pd.concat([r1, r2], ignore_index=True)
    results.to_csv(OUT / 'boosting_metrics.csv', 
                   index=False, encoding='utf-8-sig')
    
    print('\n' + '='*74)
    print('梯度提升树结果汇总')
    print('='*74)
    pivot = results.pivot_table(
        index='预见期（天）',
        columns='模型',
        values='R²'
    ).round(4)
    print(pivot.to_string())
    
    print(f'\n总耗时: {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
```

**验证**：
```bash
cd "F:/Desktop/学习/机器学习课程实践"
.venv/Scripts/python code/train_boosting.py
# 预期输出：
# XGBoost 直接策略训练:
#   第1天: R²=0.XXXX, RMSE=XXX.X
# LightGBM 直接策略训练:
#   第1天: R²=0.XXXX, RMSE=XXX.X
# 梯度提升树结果汇总
```

#### 任务 3.2：增强集成（6模型 Stacking）
**文件**：`code/enhanced_ensemble.py`

```python
"""增强集成：ANN + RF + PseudoLSTM + PyTorch-LSTM + XGBoost + LightGBM"""
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit

sys.path.insert(0, str(Path(__file__).resolve().parent))

from enhanced_features import get_enhanced_dataset

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'results/phase3_enhanced'
OUT.mkdir(parents=True, exist_ok=True)


def metrics(y, p):
    y = np.asarray(y, float).ravel()
    p = np.asarray(p, float).ravel()
    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return {
        'RMSE': float(np.sqrt(np.mean((y - p) ** 2))),
        'MAE': float(np.mean(np.abs(y - p))),
        'R²': 1 - ss_res / ss_tot
    }


def load_all_base_models():
    """加载所有基学习器"""
    models = {
        'ANN': {},
        'RF': {},
        'PseudoLSTM': {},
        'PyTorch-LSTM': {},
        'XGBoost': {},
        'LightGBM': {}
    }
    
    for h in range(1, 8):
        # 第二阶段模型
        models['ANN'][h] = joblib.load(ROOT / f'results/phase2_v5/ANN_{h}.joblib')
        models['RF'][h] = joblib.load(ROOT / f'results/phase2_v5/RF_{h}.joblib')
        
        # 第三阶段原始 LSTM
        models['PseudoLSTM'][h] = joblib.load(
            ROOT / f'results/phase3_lstm/lstm_direct_{h}.joblib')
        
        # 增强版模型
        models['PyTorch-LSTM'][h] = joblib.load(OUT / f'pytorch_lstm_direct_{h}.joblib')
        models['XGBoost'][h] = joblib.load(OUT / f'xgboost_direct_{h}.joblib')
        models['LightGBM'][h] = joblib.load(OUT / f'lightgbm_direct_{h}.joblib')
    
    return models


def get_base_predictions(models, X, h):
    """获取所有基模型的预测（6个基学习器）"""
    preds = []
    for family in ['ANN', 'RF', 'PseudoLSTM', 'PyTorch-LSTM', 'XGBoost', 'LightGBM']:
        model = models[family][h]
        pred = model.predict(X)
        if hasattr(pred, 'ravel'):
            pred = pred.ravel()
        preds.append(pred)
    return np.column_stack(preds)


def stacking_6models():
    """6模型 Stacking 集成（非线性元学习器）"""
    train_orig, test_orig, base_cols, _ = get_enhanced_dataset()
    
    # 需要映射到原始特征空间（因为旧模型用71维特征）
    import phase2_v5_data as d5
    train_71, test_71, cols_71, _ = d5.build_dataset_v5()
    
    X_train_71 = train_71[cols_71].to_numpy(float)
    X_test_71 = test_71[cols_71].to_numpy(float)
    
    X_train_enh = train_orig[base_cols + [c for c in train_orig.columns 
                                           if c.startswith('Target_')]].to_numpy(float)
    X_test_enh = test_orig[base_cols + [c for c in test_orig.columns 
                                         if c.startswith('Target_')]].to_numpy(float)
    
    # 对齐索引（增强特征会丢失开头几行）
    idx_train = train_71['Date'].isin(train_orig['Date'])
    idx_test = test_71['Date'].isin(test_orig['Date'])
    
    X_train_71 = X_train_71[idx_train]
    X_test_71 = X_test_71[idx_test]
    
    models = load_all_base_models()
    
    results = []
    
    print('\n6模型增强 Stacking:')
    for h in range(1, 8):
        y_train = train_orig[f'Target_lead{h}'].to_numpy(float)
        y_test = test_orig[f'Target_lead{h}'].to_numpy(float)
        
        # 生成元特征（交叉验证）
        tscv = TimeSeriesSplit(n_splits=3, gap=7)
        n_train = len(X_train_71)
        meta_train = np.zeros((n_train, 6))
        
        for train_idx, val_idx in tscv.split(X_train_71):
            # 使用已训练好的模型预测
            for i, family in enumerate(['ANN', 'RF', 'PseudoLSTM', 
                                        'PyTorch-LSTM', 'XGBoost', 'LightGBM']):
                model = models[family][h]
                
                # 根据模型类型选择特征
                if family in ['PyTorch-LSTM', 'XGBoost', 'LightGBM']:
                    X_val = X_train_enh[val_idx]
                else:
                    X_val = X_train_71[val_idx]
                
                pred = model.predict(X_val)
                if hasattr(pred, 'ravel'):
                    pred = pred.ravel()
                meta_train[val_idx, i] = pred
        
        # 测试集元特征
        meta_test = np.zeros((len(X_test_71), 6))
        for i, family in enumerate(['ANN', 'RF', 'PseudoLSTM', 
                                    'PyTorch-LSTM', 'XGBoost', 'LightGBM']):
            model = models[family][h]
            
            if family in ['PyTorch-LSTM', 'XGBoost', 'LightGBM']:
                X_t = X_test_enh
            else:
                X_t = X_test_71
            
            pred = model.predict(X_t)
            if hasattr(pred, 'ravel'):
                pred = pred.ravel()
            meta_test[:, i] = pred
        
        # 训练元学习器（使用 Gradient Boosting 非线性）
        meta_model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            random_state=42
        )
        meta_model.fit(meta_train, y_train)
        
        # 预测
        ensemble_pred = np.clip(meta_model.predict(meta_test), 0, None)
        
        m = metrics(y_test, ensemble_pred)
        results.append({
            '集成方法': '6模型-Stacking(GBR)',
            '预见期（天）': h,
            **m,
            '基学习器': 'ANN+RF+PseudoLSTM+PyTorch-LSTM+XGB+LGB'
        })
        
        joblib.dump(meta_model, OUT / f'stacking6_meta_{h}.joblib', compress=3)
        print(f'  第{h}天: R²={m["R²"]:.4f}, RMSE={m["RMSE"]:.1f}')
    
    return pd.DataFrame(results)


def main():
    t0 = time.time()
    
    print('='*74)
    print('6模型增强集成（Stacking with GBR Meta-Learner）')
    print('='*74)
    
    results = stacking_6models()
    results.to_csv(OUT / 'enhanced_ensemble_metrics.csv', 
                   index=False, encoding='utf-8-sig')
    
    print('\n' + '='*74)
    print('增强集成结果汇总')
    print('='*74)
    print(results[['预见期（天）', 'R²', 'RMSE', 'MAE']].to_string(index=False))
    
    # 对比原 Stacking
    print('\n与原 Stacking 对比（第1天）:')
    old = pd.read_csv(ROOT / 'results/phase3_ensemble/ensemble_metrics.csv')
    old_s1 = old[(old['策略']=='direct') & (old['方法']=='Stacking') & 
                 (old['预见期（天）']==1)].iloc[0]
    new_s1 = results[results['预见期（天）']==1].iloc[0]
    
    print(f"  原 Stacking (3模型):    R²={old_s1['R²']:.4f}, RMSE={old_s1['RMSE']:.1f}")
    print(f"  增强 Stacking (6模型):  R²={new_s1['R²']:.4f}, RMSE={new_s1['RMSE']:.1f}")
    print(f"  提升: ΔR²={new_s1['R²']-old_s1['R²']:+.4f}")
    
    if new_s1['R²'] >= 0.80:
        print(f"\n🎉🎉 目标达成！第1天 R² = {new_s1['R²']:.4f} ≥ 0.80")
    
    print(f'\n总耗时: {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
```

**验证**：
```bash
cd "F:/Desktop/学习/机器学习课程实践"
.venv/Scripts/python code/enhanced_ensemble.py
# 预期输出：
# 6模型增强 Stacking:
#   第1天: R²=0.XXXX, RMSE=XXX.X
# ...
# 与原 Stacking 对比（第1天）:
#   提升: ΔR²=+0.XXXX
# 🎉🎉 目标达成！第1天 R² = 0.XXXX ≥ 0.80 （如果达到）
```

---

### 阶段四：生成增强版验收文档（10分钟）

#### 任务 4.1：生成对比报告
**文件**：`code/generate_enhanced_report.py`

```python
"""生成增强版精度对比报告"""
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'results/phase3_enhanced'


def main():
    # 读取所有结果
    pseudo = pd.read_csv(ROOT / 'results/phase3_lstm/lstm_metrics.csv')
    pytorch = pd.read_csv(OUT / 'pytorch_lstm_metrics.csv')
    boosting = pd.read_csv(OUT / 'boosting_metrics.csv')
    ensemble = pd.read_csv(OUT / 'enhanced_ensemble_metrics.csv')
    
    print('='*80)
    print('第三阶段精度提升总结（第1天对比）')
    print('='*80)
    
    # 第1天精度对比
    results = []
    
    # PseudoLSTM
    r = pseudo[(pseudo['策略']=='直接') & (pseudo['预见期（天）']==1)].iloc[0]
    results.append(['PseudoLSTM 直接', r['R²'], r['RMSE'], r['MAE'], '原始版本'])
    
    # PyTorch LSTM
    r = pytorch[(pytorch['策略']=='PyTorch-LSTM-直接') & (pytorch['预见期（天）']==1)].iloc[0]
    results.append(['PyTorch-LSTM 直接', r['R²'], r['RMSE'], r['MAE'], '真实深度学习'])
    
    # XGBoost
    r = boosting[(boosting['模型']=='XGBoost') & (boosting['预见期（天）']==1)].iloc[0]
    results.append(['XGBoost', r['R²'], r['RMSE'], r['MAE'], '梯度提升树'])
    
    # LightGBM
    r = boosting[(boosting['模型']=='LightGBM') & (boosting['预见期（天）']==1)].iloc[0]
    results.append(['LightGBM', r['R²'], r['RMSE'], r['MAE'], '梯度提升树'])
    
    # 6模型集成
    r = ensemble[ensemble['预见期（天）']==1].iloc[0]
    results.append(['6模型 Stacking', r['R²'], r['RMSE'], r['MAE'], '增强集成'])
    
    df = pd.DataFrame(results, columns=['模型', 'R²', 'RMSE', 'MAE', '类型'])
    df = df.sort_values('R²', ascending=False)
    
    print('\n第1天精度排行榜:')
    print(df.to_string(index=False))
    
    # 计算提升幅度
    baseline = pseudo[(pseudo['策略']=='直接') & (pseudo['预见期（天）']==1)].iloc[0]['R²']
    best = df.iloc[0]
    
    print(f'\n提升幅度:')
    print(f'  基线 (PseudoLSTM): R² = {baseline:.4f}')
    print(f'  最优 ({best["模型"]}): R² = {best["R²"]:.4f}')
    print(f'  绝对提升: ΔR² = +{best["R²"]-baseline:.4f}')
    print(f'  相对提升: {(best["R²"]-baseline)/baseline*100:.1f}%')
    
    if best['R²'] >= 0.80:
        print(f'\n✅ 目标达成！R² = {best["R²"]:.4f} ≥ 0.80')
    else:
        print(f'\n⚠️  距离目标还差: {0.80-best["R²"]:.4f}')
    
    # 保存排行榜
    df.to_csv(OUT / '精度排行榜_第1天.csv', index=False, encoding='utf-8-sig')
    
    print(f'\n报告已保存: {OUT}/精度排行榜_第1天.csv')


if __name__ == '__main__':
    main()
```

**验证**：
```bash
cd "F:/Desktop/学习/机器学习课程实践"
.venv/Scripts/python code/generate_enhanced_report.py
# 预期输出：
# 第1天精度排行榜
# ... （按 R² 降序排列）
# ✅ 目标达成！R² = 0.XXXX ≥ 0.80 （如果成功）
```

---

## 测试与验证

### 功能测试
每个任务完成后运行其验证命令，确保：
1. 无 Python 报错
2. 输出结果在合理范围（R² ∈ [0, 1]，RMSE > 0）
3. 文件成功保存

### 精度验证
- **最低要求**：第1天 R² ≥ 0.80
- **理想目标**：第1天 R² ≥ 0.82，且 1-7 天平均 R² 提升 > 0.05

### 回归测试
生成报告后，确认：
- 原第三阶段结果未被覆盖（独立目录 `phase3_enhanced/`）
- 所有模型可序列化/反序列化
- CSV 文件编码为 UTF-8-sig（Excel 可读）

---

## 风险、权衡与开放问题

### 风险
1. **计算时间**：PyTorch LSTM 训练可能需 5-10 分钟，整体流程约 40-60 分钟
   - 缓解：使用 GPU（如果可用），调小 epochs
   
2. **过拟合**：增强特征后维度达 ~91，可能在小数据集上过拟合
   - 缓解：使用早停、Dropout、交叉验证

3. **模型不兼容**：旧模型（ANN/RF）用 71 维，新模型用 91 维
   - 已处理：在集成时映射到对应特征空间

### 权衡
- **复杂度 vs 精度**：6模型集成复杂度高，但精度最有保障
- **训练时间 vs 效果**：PyTorch LSTM 慢但效果好，XGBoost 快但可能不如深度学习

### 开放问题
1. **是否需要 Transformer**？
   - 如果 LSTM 已达 0.80，可跳过
   - 如果不足，阶段五可实现 Temporal Fusion Transformer

2. **超参数优化**？
   - 当前用经验值，如时间允许可加 Optuna 贝叶斯优化

3. **时序窗口**？
   - 当前特征是扁平的，未用滑动窗口序列
   - 真实 LSTM 可改为序列输入（需改造数据格式）

---

## 执行顺序

1. **阶段一**（10 分钟）：安装依赖 → 增强特征
2. **阶段二**（20 分钟）：PyTorch LSTM 实现 → 训练评估
3. **阶段三**（15 分钟）：XGBoost/LightGBM → 6模型集成
4. **阶段四**（10 分钟）：生成对比报告
5. **检查点**：R² 是否 ≥ 0.80？
   - 是：完成，生成最终文档
   - 否：继续阶段五（Transformer 或超参数优化）

---

## 后续可选（如未达 0.80）

### 阶段五：Temporal Fusion Transformer（预留）
如果上述方法仍未达 0.80，可实现 TFT（需额外 30 分钟）

**关键点**：
- 多头注意力机制
- 可解释性（变量重要性输出）
- 需要序列化输入（窗口 = 7-14 天）

**实现文件**：`code/temporal_fusion_transformer.py`（待需要时编写）

---

**计划完成。预计总耗时：55-90 分钟（取决于计算资源和是否需要阶段五）。**
