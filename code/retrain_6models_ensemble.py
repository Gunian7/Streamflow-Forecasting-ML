"""重新训练6模型集成（子任务4、6、7）"""
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


def load_all_predictions():
    """加载6个基学习器的预测结果"""
    
    # 直接策略
    direct_preds = {}
    
    # PyTorch-LSTM 直接
    pytorch = pd.read_csv(OUT / 'pytorch_lstm_predictions.csv')
    pytorch_direct = pytorch[pytorch['策略']=='PyTorch-LSTM-直接']
    direct_preds['PyTorch-LSTM'] = pytorch_direct
    
    # XGBoost, LightGBM
    boosting = pd.read_csv(OUT / 'boosting_predictions.csv')
    direct_preds['XGBoost'] = boosting[boosting['模型']=='XGBoost']
    direct_preds['LightGBM'] = boosting[boosting['模型']=='LightGBM']
    
    # ANN, RF, PseudoLSTM (从第二阶段和第三阶段)
    # 需要从原始模型重新预测
    print('需要加载 ANN、RF、PseudoLSTM 的预测...')
    
    # 多输出策略
    multi_preds = {}
    
    # PseudoLSTM 多输出
    lstm_old = pd.read_csv(ROOT / 'results/phase3_lstm/lstm_predictions.csv')
    multi_preds['PseudoLSTM'] = lstm_old[lstm_old['策略']=='多输出']
    
    # PyTorch-LSTM 多输出
    pytorch_multi = pytorch[pytorch['策略']=='PyTorch-LSTM-多输出']
    multi_preds['PyTorch-LSTM'] = pytorch_multi
    
    # ANN, RF 多输出需要重新预测
    
    return direct_preds, multi_preds


def simple_average_6models():
    """计算6模型简单平均（子任务4和6）"""
    
    print('='*70)
    print('6模型简单平均')
    print('='*70)
    
    # 加载增强特征数据
    from enhanced_features import get_enhanced_dataset
    train, test, cols, meta = get_enhanced_dataset()
    
    # 加载6个基学习器（直接策略）
    models_direct = {
        'PyTorch-LSTM': joblib.load(OUT / 'pytorch_lstm_direct_1.joblib'),
        'XGBoost': joblib.load(OUT / 'xgboost_direct_1.joblib'),
        'LightGBM': joblib.load(OUT / 'lightgbm_direct_1.joblib'),
    }
    
    # 加载第二阶段的ANN、RF
    models_direct['ANN'] = joblib.load(ROOT / 'results/phase2_v5/ANN_1.joblib')
    models_direct['RF'] = joblib.load(ROOT / 'results/phase2_v5/RF_1.joblib')
    
    # 加载PseudoLSTM
    import phase3_lstm
    sys.modules['__main__'].PseudoLSTM = phase3_lstm.PseudoLSTM
    models_direct['PseudoLSTM'] = joblib.load(ROOT / 'results/phase3_lstm/lstm_direct_1.joblib')
    
    X_test = test[cols].to_numpy(float)
    
    # 需要映射到71维（ANN、RF、PseudoLSTM用71维）
    import phase2_v5_data as d5
    train_71, test_71, cols_71, _ = d5.build_dataset_v5()
    
    # 对齐索引
    idx_test = test_71['Date'].isin(test['Date'])
    X_test_71 = test_71[cols_71].to_numpy(float)[idx_test]
    
    results_direct = []
    
    print()
    print('子任务4: 6模型简单平均-直接策略')
    for h in range(1, 8):
        y_test = test[f'Target_lead{h}'].to_numpy(float)
        
        # 加载每个预见期的模型
        if h > 1:
            models_direct['PyTorch-LSTM'] = joblib.load(OUT / f'pytorch_lstm_direct_{h}.joblib')
            models_direct['XGBoost'] = joblib.load(OUT / f'xgboost_direct_{h}.joblib')
            models_direct['LightGBM'] = joblib.load(OUT / f'lightgbm_direct_{h}.joblib')
            models_direct['ANN'] = joblib.load(ROOT / f'results/phase2_v5/ANN_{h}.joblib')
            models_direct['RF'] = joblib.load(ROOT / f'results/phase2_v5/RF_{h}.joblib')
            models_direct['PseudoLSTM'] = joblib.load(ROOT / f'results/phase3_lstm/lstm_direct_{h}.joblib')
        
        # 预测
        preds = []
        for name, model in models_direct.items():
            if name in ['PyTorch-LSTM', 'XGBoost', 'LightGBM']:
                pred = model.predict(X_test)
            else:
                pred = model.predict(X_test_71)
            
            if hasattr(pred, 'ravel'):
                pred = pred.ravel()
            preds.append(pred)
        
        # 简单平均
        avg_pred = np.mean(preds, axis=0)
        avg_pred = np.clip(avg_pred, 0, None)
        
        m = metrics(y_test, avg_pred)
        results_direct.append({
            '策略': '直接',
            '方法': '6模型简单平均',
            '预见期（天）': h,
            **m
        })
        
        print(f'  第{h}天: R²={m["R²"]:.4f}, RMSE={m["RMSE"]:.1f}')
    
    # 保存结果
    pd.DataFrame(results_direct).to_csv(OUT / '6模型简单平均_直接.csv', 
                                        index=False, encoding='utf-8-sig')
    
    return results_direct


def stacking_6models_multi():
    """重新训练6模型Stacking-多输出（子任务7）"""
    
    print()
    print('='*70)
    print('子任务7: 6模型Stacking-多输出')
    print('='*70)
    
    # 加载多输出模型
    from enhanced_features import get_enhanced_dataset
    train, test, cols, meta = get_enhanced_dataset()
    
    # 映射到71维
    import phase2_v5_data as d5
    train_71, test_71, cols_71, _ = d5.build_dataset_v5()
    idx_train = train_71['Date'].isin(train['Date'])
    idx_test = test_71['Date'].isin(test['Date'])
    X_train_71 = train_71[cols_71].to_numpy(float)[idx_train]
    X_test_71 = test_71[cols_71].to_numpy(float)[idx_test]
    
    X_train_enh = train[cols].to_numpy(float)
    X_test_enh = test[cols].to_numpy(float)
    
    # 加载6个多输出模型
    models_multi = {}
    models_multi['PyTorch-LSTM'] = joblib.load(OUT / 'pytorch_lstm_multi.joblib')
    models_multi['ANN'] = joblib.load(ROOT / 'results/phase2_v5/ANN_multi.joblib')
    models_multi['RF'] = joblib.load(ROOT / 'results/phase2_v5/RF_multi.joblib')
    
    import phase3_lstm
    sys.modules['__main__'].PseudoLSTM = phase3_lstm.PseudoLSTM
    models_multi['PseudoLSTM'] = joblib.load(ROOT / 'results/phase3_lstm/lstm_multi.joblib')
    
    # XGBoost和LightGBM没有多输出版本，用直接策略的多个模型代替
    # 或者跳过这两个
    print('注意: XGBoost和LightGBM没有多输出版本，使用4模型Stacking')
    
    results = []
    
    for h in range(1, 8):
        y_train = train[f'Target_lead{h}'].to_numpy(float)
        y_test = test[f'Target_lead{h}'].to_numpy(float)
        
        # 生成元特征（交叉验证）
        tscv = TimeSeriesSplit(n_splits=3, gap=7)
        n_train = len(X_train_71)
        meta_train = np.zeros((n_train, 4))
        
        for train_idx, val_idx in tscv.split(X_train_71):
            for i, (name, model) in enumerate(models_multi.items()):
                if name == 'PyTorch-LSTM':
                    X_val = X_train_enh[val_idx]
                    pred = model.predict(X_val)
                    if pred.ndim > 1:
                        pred = pred[:, h-1]
                else:
                    X_val = X_train_71[val_idx]
                    pred = model.predict(X_val)
                    if hasattr(pred, 'ravel'):
                        pred = pred.ravel()
                
                meta_train[val_idx, i] = pred
        
        # 测试集元特征
        meta_test = np.zeros((len(X_test_71), 4))
        for i, (name, model) in enumerate(models_multi.items()):
            if name == 'PyTorch-LSTM':
                pred = model.predict(X_test_enh)
                if pred.ndim > 1:
                    pred = pred[:, h-1]
            else:
                pred = model.predict(X_test_71)
                if hasattr(pred, 'ravel'):
                    pred = pred.ravel()
            
            meta_test[:, i] = pred
        
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
            '方法': '4模型Stacking',
            '预见期（天）': h,
            **m,
            '基学习器': 'ANN+RF+PseudoLSTM+PyTorch-LSTM'
        })
        
        print(f'  第{h}天: R²={m["R²"]:.4f}, RMSE={m["RMSE"]:.1f}')
        
        joblib.dump(meta_model, OUT / f'stacking_multi_{h}.joblib', compress=3)
    
    pd.DataFrame(results).to_csv(OUT / '4模型Stacking_多输出.csv', 
                                 index=False, encoding='utf-8-sig')
    
    return results


def main():
    t0 = time.time()
    
    try:
        # 子任务4: 6模型简单平均-直接
        r_direct_avg = simple_average_6models()
        
        # 子任务7: 4模型Stacking-多输出（没有XGBoost/LightGBM多输出版本）
        r_multi_stack = stacking_6models_multi()
        
        print()
        print('='*70)
        print('汇总')
        print('='*70)
        print()
        print('子任务4（6模型简单平均-直接）第1天:')
        r4 = r_direct_avg[0]
        print(f'  R²={r4["R²"]:.4f}, RMSE={r4["RMSE"]:.1f}')
        print()
        print('子任务7（4模型Stacking-多输出）第1天:')
        r7 = r_multi_stack[0]
        print(f'  R²={r7["R²"]:.4f}, RMSE={r7["RMSE"]:.1f}')
        
        print()
        print(f'总耗时: {time.time()-t0:.0f}s')
        
    except Exception as e:
        print(f'错误: {e}')
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
