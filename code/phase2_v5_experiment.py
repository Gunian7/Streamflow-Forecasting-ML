"""Phase-2 v5 实验：修正数据 + 当天径流 + 积雪特征 + 对数目标

与 v3 的差异全部来自已核实的数据/特征缺陷修正，模型族仍为 ANN 与 RF，
仍包含直接策略与多输出策略，满足第二阶段验收要求。
"""
import json
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import phase2_v5_data as d5

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'results/phase2_v5'
OUT.mkdir(parents=True, exist_ok=True)


def make_ann():
    pipe = Pipeline([('scale', StandardScaler()),
                     ('model', MLPRegressor(solver='lbfgs', max_iter=5000,
                                            max_fun=100000, tol=1e-5,
                                            early_stopping=False,
                                            random_state=42))])
    return TransformedTargetRegressor(regressor=pipe, transformer=StandardScaler())


def make_rf(n_estimators=300):
    return RandomForestRegressor(n_estimators=n_estimators, n_jobs=1, random_state=42)


ANN_GRID = {
    'regressor__model__alpha': [0.1, 1.0, 10.0],
    'regressor__model__hidden_layer_sizes': [(16,), (32,)],
}
RF_GRID = {
    'max_depth': [8, 16, None],
    'min_samples_leaf': [5, 10],
}


def metrics(y, p):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    rmse = float(np.sqrt(np.mean((y - p) ** 2)))
    mae = float(np.mean(np.abs(y - p)))
    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return {'RMSE（cfs）': rmse, 'MAE（cfs）': mae,
            'R²': 1 - ss_res / ss_tot, 'NSE': 1 - ss_res / ss_tot,
            '样本数': int(len(y))}


def run_search(X, Y, log_target=True):
    tscv = TimeSeriesSplit(n_splits=3, gap=7)
    rows = []
    preds = {}

    for family, maker, grid in [('ANN', make_ann, ANN_GRID), ('RF', make_rf, RF_GRID)]:
        for task in list(range(1, 8)) + ['multi']:
            t0 = time.time()
            label = f'{family}_{task}'
            target = Y if task == 'multi' else Y[:, task - 1]
            y_fit = np.log1p(target) if log_target else target

            gs = GridSearchCV(maker(), grid, cv=tscv,
                              scoring='neg_root_mean_squared_error',
                              n_jobs=2, error_score='raise')
            gs.fit(X, y_fit)
            joblib.dump(gs, OUT / f'{label}.joblib', compress=3)
            pd.DataFrame(gs.cv_results_).to_csv(
                OUT / f'{label}_网格搜索.csv', index=False, encoding='utf-8-sig')

            rows.append({'模型': family,
                         '策略': '多输出' if task == 'multi' else '直接',
                         '选参预见期': str(task),
                         '最佳参数': json.dumps(gs.best_params_, ensure_ascii=False),
                         '交叉验证RMSE（对数尺度）': float(-gs.best_score_),
                         '耗时秒': round(time.time() - t0, 1)})
            print(f'{label:10s} CV={-gs.best_score_:.4f} ({time.time()-t0:.0f}s)  '
                  f'{gs.best_params_}', flush=True)
    return pd.DataFrame(rows)


def evaluate(Xt, Yt, log_target=True):
    rows, detail = [], []
    for family in ['ANN', 'RF']:
        for task in list(range(1, 8)) + ['multi']:
            path = OUT / f'{family}_{task}.joblib'
            if not path.exists():
                continue
            gs = joblib.load(path)
            raw = np.asarray(gs.best_estimator_.predict(Xt))
            if task == 'multi':
                mat = raw
            else:
                mat = raw.reshape(-1, 1)

            for pos, h in enumerate(range(1, 8) if task == 'multi' else [task]):
                y_true = Yt[:, h - 1]
                y_pred = mat[:, pos] if task == 'multi' else mat[:, 0]
                if log_target:
                    y_pred = np.expm1(y_pred)
                # 流量非负
                y_pred = np.clip(y_pred, 0, None)
                m = metrics(y_true, y_pred)
                rows.append({'模型': family,
                             '策略': '多输出' if task == 'multi' else '直接',
                             '预见期（天）': h, **m})
                for i in range(len(y_true)):
                    detail.append({'模型': family,
                                   '策略': '多输出' if task == 'multi' else '直接',
                                   '预见期（天）': h,
                                   '样本序号': i,
                                   '实测径流（cfs）': y_true[i],
                                   '预测径流（cfs）': y_pred[i]})
    return pd.DataFrame(rows), pd.DataFrame(detail)


def main():
    t_start = time.time()
    train, test, cols, meta = d5.build_dataset_v5()
    print('=' * 74)
    print('v5 数据边界')
    print('=' * 74)
    print(json.dumps({k: v for k, v in meta.items() if k != '特征'}, ensure_ascii=False, indent=2))

    X = train[cols].to_numpy(float)
    Y = d5.target_matrix(train, log=False)
    Xt = test[cols].to_numpy(float)
    Yt = d5.target_matrix(test, log=False)

    print()
    print('=' * 74)
    print('超参数搜索（对数目标）')
    print('=' * 74)
    cv = run_search(X, Y, log_target=True)
    cv.to_csv(OUT / '选参结果.csv', index=False, encoding='utf-8-sig')

    print()
    print('=' * 74)
    print('2004 测试集评估')
    print('=' * 74)
    met, det = evaluate(Xt, Yt, log_target=True)
    met.to_csv(OUT / '测试精度指标.csv', index=False, encoding='utf-8-sig')
    det.to_csv(OUT / '逐日预测结果.csv', index=False, encoding='utf-8-sig')

    pivot = met.pivot_table(index='预见期（天）', columns=['模型', '策略'],
                            values='NSE').round(4)
    print(pivot.to_string())

    print()
    print('最优逐日:')
    for h in range(1, 8):
        sub = met[met['预见期（天）'] == h]
        b = sub.loc[sub['NSE'].idxmax()]
        print(f'  第{h}天 {b["模型"]} {b["策略"]}: NSE={b["NSE"]:.4f}, '
              f'RMSE={b["RMSE（cfs）"]:.1f}')

    print(f'\n总耗时 {time.time()-t_start:.0f}s')
    print(f'结果目录: {OUT}')


if __name__ == '__main__':
    main()
