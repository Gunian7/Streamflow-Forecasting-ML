"""简化版：子任务6和7只用3模型（ANN+RF+PseudoLSTM）"""
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit

sys.path.insert(0, str(Path(__file__).resolve().parent))

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


def simple_average_multi_3models():
    """子任务6: 3模型简单平均-多输出"""
    
    print('='*70)
    print('子任务6: 3模型简单平均-多输出（ANN+RF+PseudoLSTM）')
    print('='*70)
    
    import phase2_v5_data as d5
    train, test, cols, _ = d5.build_dataset_v5()
    
    X_test = test[cols].to_numpy(float)
    
    # 填充到71维
    X_test = np.concatenate([X_test, np.zeros((len(X_test), 6))], axis=1)
    
    # 加载3个多输出模型
    import phase3_lstm
    sys.modules['__main__'].PseudoLSTM = phase3_lstm.PseudoLSTM
    
    models = {
        'ANN': joblib.load(ROOT / 'results/phase2_v5/ANN_multi.joblib'),
        'RF': joblib.load(ROOT / 'results/phase2_v5/RF_multi.joblib'),
        'PseudoLSTM': joblib.load(ROOT / 'results/phase3_lstm/lstm_multi.joblib'),
    }
    
    results = []
    
    print()
    for h in range(1, 8):
        y_test = test[f'Target_lead{h}'].to_numpy(float)
        
        # 收集预测
        preds = []
        for name, model in models.items():
            pred = model.predict(X_test)
            if pred.ndim > 1:
                pred = pred[:, h-1]
            elif hasattr(pred, 'ravel'):
                pred = pred.ravel()
            preds.append(pred)
        
        # 简单平均
        avg_pred = np.mean(preds, axis=0)
        avg_pred = np.clip(avg_pred, 0, None)
        
        m = metrics(y_test, avg_pred)
        results.append({
            '策略': '多输出',
            '方法': '3模型简单平均',
            '预见期（天）': h,
            **m,
            '基学习器': 'ANN+RF+PseudoLSTM'
        })
        
        r2 = m['R²']
        rmse = m['RMSE']
        print(f'  第{h}天: R²={r2:.4f}, RMSE={rmse:.1f}')
    
    df = pd.DataFrame(results)
    df.to_csv(OUT / '3模型简单平均_多输出.csv', index=False, encoding='utf-8-sig')
    
    return results


def stacking_multi_3models():
    """子任务7: 3模型Stacking-多输出"""
    
    print()
    print('='*70)
    print('子任务7: 3模型Stacking-多输出（ANN+RF+PseudoLSTM）')
    print('='*70)
    
    import phase2_v5_data as d5
    train, test, cols, _ = d5.build_dataset_v5()
    
    X_train = train[cols].to_numpy(float)
    X_test = test[cols].to_numpy(float)
    
    # 填充到71维
    X_train = np.concatenate([X_train, np.zeros((len(X_train), 6))], axis=1)
    X_test = np.concatenate([X_test, np.zeros((len(X_test), 6))], axis=1)
    
    # 加载3个多输出模型
    import phase3_lstm
    sys.modules['__main__'].PseudoLSTM = phase3_lstm.PseudoLSTM
    
    models = {
        'ANN': joblib.load(ROOT / 'results/phase2_v5/ANN_multi.joblib'),
        'RF': joblib.load(ROOT / 'results/phase2_v5/RF_multi.joblib'),
        'PseudoLSTM': joblib.load(ROOT / 'results/phase3_lstm/lstm_multi.joblib'),
    }
    
    results = []
    
    print()
    for h in range(1, 8):
        y_train = train[f'Target_lead{h}'].to_numpy(float)
        y_test = test[f'Target_lead{h}'].to_numpy(float)
        
        n_models = 3
        n_train = len(X_train)
        
        # 生成OOF元特征
        meta_train = np.zeros((n_train, n_models))
        tscv = TimeSeriesSplit(n_splits=3, gap=7)
        
        for train_idx, val_idx in tscv.split(X_train):
            for i, (name, model) in enumerate(models.items()):
                pred_val = model.predict(X_train[val_idx])
                if pred_val.ndim > 1:
                    pred_val = pred_val[:, h-1]
                elif hasattr(pred_val, 'ravel'):
                    pred_val = pred_val.ravel()
                meta_train[val_idx, i] = pred_val
        
        # 测试集元特征
        meta_test = np.zeros((len(X_test), n_models))
        
        for i, (name, model) in enumerate(models.items()):
            pred_test = model.predict(X_test)
            if pred_test.ndim > 1:
                pred_test = pred_test[:, h-1]
            elif hasattr(pred_test, 'ravel'):
                pred_test = pred_test.ravel()
            meta_test[:, i] = pred_test
        
        # 训练元学习器
        meta_model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            random_state=42
        )
        meta_model.fit(meta_train, y_train)
        
        # 预测
        stacking_pred = np.clip(meta_model.predict(meta_test), 0, None)
        
        m = metrics(y_test, stacking_pred)
        results.append({
            '策略': '多输出',
            '方法': '3模型Stacking',
            '预见期（天）': h,
            **m,
            '基学习器': 'ANN+RF+PseudoLSTM'
        })
        
        r2 = m['R²']
        rmse = m['RMSE']
        print(f'  第{h}天: R²={r2:.4f}, RMSE={rmse:.1f}')
        
        joblib.dump(meta_model, OUT / f'stacking_3model_multi_{h}.joblib', compress=3)
    
    df = pd.DataFrame(results)
    df.to_csv(OUT / '3模型Stacking_多输出.csv', index=False, encoding='utf-8-sig')
    
    return results


def main():
    t0 = time.time()
    
    try:
        r6 = simple_average_multi_3models()
        r7 = stacking_multi_3models()
        
        print()
        print('='*70)
        print('汇总')
        print('='*70)
        print()
        print('子任务6（3模型简单平均-多输出）第1天:')
        print(f'  R²={r6[0]["R²"]:.4f}, RMSE={r6[0]["RMSE"]:.1f}')
        print()
        print('子任务7（3模型Stacking-多输出）第1天:')
        print(f'  R²={r7[0]["R²"]:.4f}, RMSE={r7[0]["RMSE"]:.1f}')
        
        print()
        print(f'总耗时: {time.time()-t0:.0f}s')
        
    except Exception as e:
        print(f'错误: {e}')
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
