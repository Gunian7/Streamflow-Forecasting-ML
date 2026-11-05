"""使用Grid Search最优参数重新训练PyTorch-LSTM"""
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from enhanced_features import get_enhanced_dataset
from pytorch_lstm import LSTMRegressor, LSTMTrainer

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'results/phase3_enhanced'


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


def retrain_direct():
    """重新训练直接策略"""
    print('='*70)
    print('重新训练PyTorch-LSTM直接策略（使用各自最优参数）')
    print('='*70)
    
    # 读取Grid Search结果
    grid = pd.read_csv(OUT / 'pytorch_lstm_grid_summary.csv')
    
    # 加载数据
    train, test, cols, _ = get_enhanced_dataset()
    X_train = train[cols].to_numpy(float)
    X_test = test[cols].to_numpy(float)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = []
    
    for h in range(1, 8):
        print(f'\n第{h}天:')
        
        # 获取最优参数
        row = grid[grid['task'] == f'direct_day{h}'].iloc[0]
        hidden_dim = int(row['best_hidden_dim'])
        num_layers = int(row['best_num_layers'])
        lr = float(row['best_lr'])
        dropout = float(row['best_dropout'])
        
        print(f'  最优参数: hidden={hidden_dim}, layers={num_layers}, lr={lr}, dropout={dropout}')
        
        # 训练
        y_train = train[f'Target_lead{h}'].to_numpy(float)
        y_test = test[f'Target_lead{h}'].to_numpy(float)
        
        model = LSTMRegressor(
            input_dim=X_train.shape[1],
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=1,
            dropout=dropout
        )
        
        trainer = LSTMTrainer(model, device=device, lr=lr)
        
        t0 = time.time()
        trainer.fit(X_train, y_train, 
                   epochs=200, batch_size=32,
                   early_stopping_patience=20,
                   verbose=False)
        
        # 预测
        pred = trainer.predict(X_test)
        pred = np.clip(pred, 0, None)
        
        m = metrics(y_test, pred)
        elapsed = time.time() - t0
        
        print(f'  R²={m["R²"]:.4f}, RMSE={m["RMSE"]:.1f}, 耗时{elapsed:.1f}s')
        
        results.append({
            '策略': 'PyTorch-LSTM-直接',
            '预见期（天）': h,
            **m
        })
        
        # 保存模型
        joblib.dump(trainer, OUT / f'pytorch_lstm_direct_{h}.joblib', compress=3)
    
    df = pd.DataFrame(results)
    df.to_csv(OUT / 'pytorch_lstm_direct_final.csv', index=False, encoding='utf-8-sig')
    
    return results


def retrain_multi():
    """重新训练多输出策略"""
    print()
    print('='*70)
    print('重新训练PyTorch-LSTM多输出策略')
    print('='*70)
    
    # 读取最优参数
    grid = pd.read_csv(OUT / 'pytorch_lstm_grid_summary.csv')
    row = grid[grid['task'] == 'multi_output'].iloc[0]
    hidden_dim = int(row['best_hidden_dim'])
    num_layers = int(row['best_num_layers'])
    lr = float(row['best_lr'])
    dropout = float(row['best_dropout'])
    
    print(f'最优参数: hidden={hidden_dim}, layers={num_layers}, lr={lr}, dropout={dropout}')
    
    # 加载数据
    train, test, cols, _ = get_enhanced_dataset()
    X_train = train[cols].to_numpy(float)
    X_test = test[cols].to_numpy(float)
    
    # 多输出目标
    y_train = np.column_stack([
        train[f'Target_lead{h}'].to_numpy(float)
        for h in range(1, 8)
    ])
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 训练
    model = LSTMRegressor(
        input_dim=X_train.shape[1],
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        output_dim=7,
        dropout=dropout
    )
    
    trainer = LSTMTrainer(model, device=device, lr=lr)
    
    t0 = time.time()
    trainer.fit(X_train, y_train,
               epochs=200, batch_size=32,
               early_stopping_patience=20,
               verbose=False)
    elapsed = time.time() - t0
    
    print(f'训练完成，耗时{elapsed:.1f}s')
    
    # 预测
    y_pred = trainer.predict(X_test)
    
    results = []
    
    print()
    for h in range(1, 8):
        y_test = test[f'Target_lead{h}'].to_numpy(float)
        pred = np.clip(y_pred[:, h-1], 0, None)
        
        m = metrics(y_test, pred)
        
        print(f'  第{h}天: R²={m["R²"]:.4f}, RMSE={m["RMSE"]:.1f}')
        
        results.append({
            '策略': 'PyTorch-LSTM-多输出',
            '预见期（天）': h,
            **m
        })
    
    # 保存
    joblib.dump(trainer, OUT / 'pytorch_lstm_multi.joblib', compress=3)
    
    df = pd.DataFrame(results)
    df.to_csv(OUT / 'pytorch_lstm_multi_final.csv', index=False, encoding='utf-8-sig')
    
    return results


def update_metrics():
    """更新汇总指标"""
    print()
    print('='*70)
    print('更新pytorch_lstm_metrics.csv')
    print('='*70)
    
    direct = pd.read_csv(OUT / 'pytorch_lstm_direct_final.csv')
    multi = pd.read_csv(OUT / 'pytorch_lstm_multi_final.csv')
    
    all_metrics = pd.concat([direct, multi], ignore_index=True)
    all_metrics.to_csv(OUT / 'pytorch_lstm_metrics.csv', index=False, encoding='utf-8-sig')
    
    print('已更新 pytorch_lstm_metrics.csv')
    print()
    print('对比（第1天）:')
    print(f'  直接策略: R²={direct.iloc[0]["R²"]:.4f}')
    print(f'  多输出策略: R²={multi.iloc[0]["R²"]:.4f}')


def main():
    t0 = time.time()
    
    r1 = retrain_direct()
    r2 = retrain_multi()
    update_metrics()
    
    print()
    print('='*70)
    print(f'总耗时: {time.time()-t0:.0f}s')
    print('='*70)


if __name__ == '__main__':
    main()
