"""Phase-two v3 experiment runner: CV-only feature/target ablation.

Selection uses training-period cross-validation only. The 2004 test segment is
evaluated once at the `final` stage and is never used to pick a configuration.
"""
import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase2_v3_config as cfg
import phase2_v3_data as d3
import phase2_v3_features as f3
import phase2_v3_baselines as b3

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'results/phase2_v3'
SMOKE = ROOT / 'results/phase2_v3_smoke'


def groups_for(cols):
    base = f3.feature_groups(cols)
    base['D_加长窗口与温差'] = f3.group_d_columns(cols)
    return base


def cv_frames(train, test, cols, group_cols, with_d):
    """Build the shared-date frames for one feature group."""
    frame = f3.add_long_windows(train) if with_d else train
    missing = [c for c in group_cols if c not in frame.columns]
    if missing:
        raise KeyError(missing)
    frame = frame.dropna(subset=group_cols).reset_index(drop=True)
    return frame


def run_ablation(smoke=False):
    out = SMOKE if smoke else OUT
    out.mkdir(parents=True, exist_ok=True)
    train, test, cols, meta = d3.build_dataset()
    groups = groups_for(cols)
    tscv = TimeSeriesSplit(n_splits=3, gap=cfg.PROTOCOL['max_horizon'])
    rows = []
    for name, members in groups.items():
        with_d = name.startswith('D_')
        frame = cv_frames(train, test, cols, members, with_d)
        # Shared dates across groups keep the comparison fair.
        X = frame[members].to_numpy(dtype=float)
        Y = frame[[f'Target_lead{h}' for h in range(1, 8)]].to_numpy(dtype=float)
        oof = np.full_like(Y, np.nan)
        for fold, (tr, va) in enumerate(tscv.split(X), 1):
            model = RandomForestRegressor(n_estimators=60 if smoke else 200,
                                          min_samples_leaf=3, n_jobs=1,
                                          random_state=42)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                model.fit(X[tr], Y[tr])
            oof[va] = model.predict(X[va])
        for h in range(1, 8):
            ok = ~np.isnan(oof[:, h - 1])
            row = {'特征组': name, '特征数': len(members), '样本数': int(ok.sum()),
                   '预见期（天）': h}
            row.update(b3.metrics(Y[ok, h - 1], oof[ok, h - 1]))
            rows.append(row)
        print(f'{name}: {len(members)} 特征, {int(ok.sum())} 个OOF样本', flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out / '特征消融_交叉验证.csv', index=False, encoding='utf-8-sig')
    print(frame.pivot_table(index='特征组', columns='预见期（天）',
                            values='RMSE（cfs）').round(1).to_string())
    return frame


def run_ann(smoke=False):
    """CV-only search over ANN regularisation on the best feature group."""
    import joblib
    import phase2_v3_models as m3
    from sklearn.model_selection import GridSearchCV
    out = SMOKE if smoke else OUT
    out.mkdir(parents=True, exist_ok=True)
    train, test, cols, meta = d3.build_dataset()
    groups = groups_for(cols)
    members = groups['C_加窗口统计']
    frame = train.dropna(subset=members).reset_index(drop=True)
    X = frame[members].to_numpy(dtype=float)
    Y = frame[[f'Target_lead{h}' for h in range(1, 8)]].to_numpy(dtype=float)
    tscv = TimeSeriesSplit(n_splits=3, gap=cfg.PROTOCOL['max_horizon'])
    grid = m3.ann_param_grid()
    rows = []
    for horizon in list(range(1, 8)) + ['multi']:
        target = Y if horizon == 'multi' else Y[:, horizon - 1]
        label = f'ANN_{horizon}'
        search = GridSearchCV(m3.make_ann(), grid, cv=tscv,
                              scoring='neg_root_mean_squared_error',
                              n_jobs=2, error_score='raise',
                              return_train_score=True)
        search.fit(X, target)
        joblib.dump(search, out / f'{label}.joblib', compress=3)
        scores = pd.DataFrame(search.cv_results_)
        scores.to_csv(out / f'{label}_网格搜索.csv', index=False,
                      encoding='utf-8-sig')
        rows.append({'模型': 'ANN', '策略': '多输出' if horizon == 'multi' else '直接',
                     '选参预见期': str(horizon),
                     '最佳参数': json.dumps(search.best_params_, ensure_ascii=False),
                     '交叉验证RMSE（cfs）': -search.best_score_,
                     **{k: v for k, v in m3.ann_diagnostics(search).items()
                        if k != '交叉验证RMSE（cfs）'}})
        print(label, round(-search.best_score_, 2), flush=True)
    frame_out = pd.DataFrame(rows)
    frame_out.to_csv(out / 'ANN选参与诊断.csv', index=False, encoding='utf-8-sig')
    return frame_out


def run_rf(smoke=False):
    """CV-only two-stage RF search on the best feature group."""
    import joblib
    import phase2_v3_models as m3
    from sklearn.model_selection import GridSearchCV
    out = SMOKE if smoke else OUT
    out.mkdir(parents=True, exist_ok=True)
    train, test, cols, meta = d3.build_dataset()
    members = groups_for(cols)['C_加窗口统计']
    frame = train.dropna(subset=members).reset_index(drop=True)
    X = frame[members].to_numpy(dtype=float)
    Y = frame[[f'Target_lead{h}' for h in range(1, 8)]].to_numpy(dtype=float)
    tscv = TimeSeriesSplit(n_splits=3, gap=cfg.PROTOCOL['max_horizon'])
    rows = []
    for horizon in list(range(1, 8)) + ['multi']:
        target = Y if horizon == 'multi' else Y[:, horizon - 1]
        coarse = GridSearchCV(m3.make_rf(200), m3.rf_param_grid('coarse'),
                              cv=tscv, scoring='neg_root_mean_squared_error',
                              n_jobs=2, error_score='raise')
        coarse.fit(X, target)
        best = dict(coarse.best_params_)
        refine = GridSearchCV(m3.make_rf(200), m3.rf_param_grid('refine'),
                              cv=tscv, scoring='neg_root_mean_squared_error',
                              n_jobs=2, error_score='raise')
        refine.set_params(**{'estimator__' + k: v for k, v in best.items()})
        refine.fit(X, target)
        if -refine.best_score_ > -coarse.best_score_:
            chosen, source = coarse, '粗搜'
        else:
            chosen, source = refine, '细化'
        joblib.dump(chosen, out / f'RF_{horizon}.joblib', compress=3)
        pd.DataFrame(chosen.cv_results_).to_csv(
            out / f'RF_{horizon}_网格搜索.csv', index=False, encoding='utf-8-sig')
        rows.append({'模型': 'RF', '策略': '多输出' if horizon == 'multi' else '直接',
                     '选参预见期': str(horizon),
                     '最佳参数': json.dumps(chosen.best_params_, ensure_ascii=False),
                     '交叉验证RMSE（cfs）': -chosen.best_score_, '来源': source})
        print(f'RF_{horizon}', round(-chosen.best_score_, 2), source, flush=True)
    frame_out = pd.DataFrame(rows)
    frame_out.to_csv(out / 'RF选参.csv', index=False, encoding='utf-8-sig')
    return frame_out


def run_final(smoke=False):
    """Evaluate frozen v3 selections once on the 2004 test segment."""
    import joblib
    import phase2_v3_models as m3
    out = SMOKE if smoke else OUT
    train, test, cols, meta = d3.build_dataset()
    members = groups_for(cols)['C_加窗口统计']
    frame = train.dropna(subset=members).reset_index(drop=True)
    X = frame[members].to_numpy(dtype=float)
    Y = frame[[f'Target_lead{h}' for h in range(1, 8)]].to_numpy(dtype=float)
    Xt = test[members].to_numpy(dtype=float)
    Yt = test[[f'Target_lead{h}' for h in range(1, 8)]].to_numpy(dtype=float)
    rows, preds = [], []
    for family in ['ANN', 'RF']:
        for horizon in list(range(1, 8)) + ['multi']:
            path = out / f'{family}_{horizon}.joblib'
            if not path.exists():
                raise FileNotFoundError(path)
            search = joblib.load(path)
            raw = np.asarray(search.best_estimator_.predict(Xt))
            if horizon == 'multi':
                # Native multi-output must yield one column per horizon.
                if raw.ndim != 2 or raw.shape[1] != 7:
                    raise ValueError(f'{family}_multi 输出形状异常: {raw.shape}')
                matrix = raw
            else:
                # Direct models are trained on one target, so predict is 1-D.
                if raw.ndim != 1:
                    raise ValueError(f'{family}_{horizon} 输出形状异常: {raw.shape}')
                matrix = raw.reshape(-1, 1)
            for h in (range(1, 8) if horizon == 'multi' else [horizon]):
                # Direct models keep their single target in column 0.
                col = h - 1 if horizon == 'multi' else 0
                row = {'模型': family,
                       '策略': '多输出' if horizon == 'multi' else '直接',
                       '预见期（天）': h}
                row.update(b3.metrics(Yt[:, h - 1], matrix[:, col]))
                rows.append(row)
                for date, a, p in zip(test['Date'], Yt[:, h - 1], matrix[:, col]):
                    preds.append({'模型': family,
                                  '策略': '多输出' if horizon == 'multi' else '直接',
                                  '起报日期': str(date.date()),
                                  '目标日期': str((date + pd.Timedelta(days=h)).date()),
                                  '预见期（天）': h,
                                  '实测径流（cfs）': a, '预测径流（cfs）': p})
            print(f'{family}_{horizon} done', flush=True)
    metrics = pd.DataFrame(rows)
    assert len(metrics) == 28
    metrics.to_csv(out / '测试精度指标.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(preds).to_csv(out / '逐日预测结果.csv', index=False,
                               encoding='utf-8-sig')
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', default='features',
                    choices=['baseline', 'features', 'ann', 'rf', 'final'])
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    if args.stage == 'baseline':
        train, test, cols, meta = d3.build_dataset()
        frame, _ = b3.evaluate_baselines(train, test)
        (SMOKE if args.smoke else OUT).mkdir(parents=True, exist_ok=True)
        frame.to_csv((SMOKE if args.smoke else OUT) / '基线指标.csv',
                     index=False, encoding='utf-8-sig')
        print(frame.to_string(index=False))
        return
    if args.stage == 'ann':
        run_ann(smoke=args.smoke)
        return
    if args.stage == 'rf':
        run_rf(smoke=args.smoke)
        return
    if args.stage == 'final':
        run_final(smoke=args.smoke)
        return
    run_ablation(smoke=args.smoke)


if __name__ == '__main__':
    main()
