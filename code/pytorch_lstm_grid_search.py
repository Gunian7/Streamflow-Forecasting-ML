"""PyTorch-LSTM Grid Search for all days"""
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
from pytorch_lstm import LSTMRegressor, LSTMTrainer, TimeSeriesDataset

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'results/phase3_enhanced'
OUT.mkdir(parents=True, exist_ok=True)


def grid_search_pytorch_lstm():
    """为PyTorch-LSTM做Grid Search"""
    
    print('='*70)
    print('PyTorch-LSTM Grid Search（第1-7天 + 多输出）')
    print('='*70)
    
    # 加载数据
    train, test, cols, _ = get_enhanced_dataset()
    X_train = train[cols].to_numpy(float)
    
    # 参数空间（简化版，保证30分钟内完成）
    param_grid = {
        'hidden_dim': [64, 128],
        'num_layers': [1, 2],
        'lr': [0.001, 0.01],
        'dropout': [0.2]
    }
    
    tscv = TimeSeriesSplit(n_splits=3, gap=7)
    results = []
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'使用设备: {device}')
    print(f'参数组合数: {len(param_grid["hidden_dim"]) * len(param_grid["num_layers"]) * len(param_grid["lr"])} = 8')
    print()
    
    # 直接策略第1-7天
    for h in range(1, 8):
        print(f'\n{"="*70}')
        print(f'第{h}天 Grid Search')
        print(f'{"="*70}')
        
        y_train = train[f'Target_lead{h}'].to_numpy(float)
        
        best_score = float('inf')
        best_params = None
        cv_scores = []
        
        for hidden_dim in param_grid['hidden_dim']:
            for num_layers in param_grid['num_layers']:
                for lr in param_grid['lr']:
                    dropout = param_grid['dropout'][0]
                    
                    fold_scores = []
                    
                    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
                        X_tr = X_train[train_idx]
                        y_tr = y_train[train_idx]
                        X_val = X_train[val_idx]
                        y_val = y_train[val_idx]
                        
                        # 训练模型
                        model = LSTMRegressor(
                            input_dim=X_tr.shape[1],
                            hidden_dim=hidden_dim,
                            num_layers=num_layers,
                            output_dim=1,
                            dropout=dropout
                        )
                        
                        trainer = LSTMTrainer(model, device=device, lr=lr)
                        trainer.fit(X_tr, y_tr, X_val, y_val, 
                                   epochs=50, batch_size=32,
                                   early_stopping_patience=10,
                                   verbose=False)
                        
                        # 验证集RMSE
                        pred = trainer.predict(X_val)
                        rmse = np.sqrt(np.mean((y_val - pred) ** 2))
                        fold_scores.append(rmse)
                    
                    mean_rmse = np.mean(fold_scores)
                    cv_scores.append(mean_rmse)
                    
                    print(f'  hidden={hidden_dim}, layers={num_layers}, lr={lr}: CV_RMSE={mean_rmse:.1f}')
                    
                    if mean_rmse < best_score:
                        best_score = mean_rmse
                        best_params = {
                            'hidden_dim': hidden_dim,
                            'num_layers': num_layers,
                            'lr': lr,
                            'dropout': dropout
                        }
        
        print(f'\n✓ 最优: {best_params}, CV_RMSE={best_score:.1f}')
        
        results.append({
            'task': f'direct_day{h}',
            'best_hidden_dim': best_params['hidden_dim'],
            'best_num_layers': best_params['num_layers'],
            'best_lr': best_params['lr'],
            'best_dropout': best_params['dropout'],
            'cv_rmse': best_score
        })
    
    # 多输出策略
    print(f'\n{"="*70}')
    print(f'多输出策略 Grid Search')
    print(f'{"="*70}')
    
    # 准备多输出目标
    y_train_multi = np.column_stack([
        train[f'Target_lead{h}'].to_numpy(float)
        for h in range(1, 8)
    ])
    
    best_score = float('inf')
    best_params = None
    
    for hidden_dim in param_grid['hidden_dim']:
        for num_layers in param_grid['num_layers']:
            for lr in param_grid['lr']:
                dropout = param_grid['dropout'][0]
                
                fold_scores = []
                
                for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
                    X_tr = X_train[train_idx]
                    y_tr = y_train_multi[train_idx]
                    X_val = X_train[val_idx]
                    y_val = y_train_multi[val_idx]
                    
                    # 训练模型（输出7维）
                    model = LSTMRegressor(
                        input_dim=X_tr.shape[1],
                        hidden_dim=hidden_dim,
                        num_layers=num_layers,
                        output_dim=7,
                        dropout=dropout
                    )
                    
                    trainer = LSTMTrainer(model, device=device, lr=lr)
                    trainer.fit(X_tr, y_tr, X_val, y_val,
                               epochs=50, batch_size=32,
                               early_stopping_patience=10,
                               verbose=False)
                    
                    # 验证集RMSE（平均所有输出）
                    pred = trainer.predict(X_val)
                    rmse = np.sqrt(np.mean((y_val - pred) ** 2))
                    fold_scores.append(rmse)
                
                mean_rmse = np.mean(fold_scores)
                
                print(f'  hidden={hidden_dim}, layers={num_layers}, lr={lr}: CV_RMSE={mean_rmse:.1f}')
                
                if mean_rmse < best_score:
                    best_score = mean_rmse
                    best_params = {
                        'hidden_dim': hidden_dim,
                        'num_layers': num_layers,
                        'lr': lr,
                        'dropout': dropout
                    }
    
    print(f'\n✓ 最优: {best_params}, CV_RMSE={best_score:.1f}')
    
    results.append({
        'task': 'multi_output',
        'best_hidden_dim': best_params['hidden_dim'],
        'best_num_layers': best_params['num_layers'],
        'best_lr': best_params['lr'],
        'best_dropout': best_params['dropout'],
        'cv_rmse': best_score
    })
    
    # 保存结果
    df = pd.DataFrame(results)
    df.to_csv(OUT / 'pytorch_lstm_grid_summary.csv', index=False, encoding='utf-8-sig')
    
    print()
    print('='*70)
    print('完整汇总:')
    print(df.to_string(index=False))
    
    return df


if __name__ == '__main__':
    t0 = time.time()
    grid_search_pytorch_lstm()
    print(f'\n总耗时: {time.time()-t0:.0f}s')
