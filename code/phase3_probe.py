# -*- coding: utf-8 -*-
"""Phase-3 可行性探针：比较"现用 01047000.csv"与"恢复的 01047000_fixed.csv（Daymet v4
真实积雪）"在第 1 天预见期上的精度差异。划分、特征口径与超参数完全一致，只换数据源。

用法：
  python phase3_probe.py                 # 对齐校验 + 两个子进程跑两组数据源 + 汇总
  python phase3_probe.py --source base   # 内部使用：只跑一个数据源
控制台只输出 ASCII；所有中文结论写入 results/probe/。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CODE = Path(__file__).resolve().parent
BASE_CSV = ROOT / '01047000.csv'
FIXED_CSV = ROOT / '01047000_fixed.csv'
DAYMET = ROOT / 'results/diagnostics/daymet_full.txt'
OUT = ROOT / 'results/probe'

MET = ['Dayl', 'Prcp', 'Srad', 'Swe', 'Tmax', 'Tmin', 'Vp']
DM_COLS = ['dayl', 'prcp', 'srad', 'swe', 'tmax', 'tmin', 'vp']
SOURCES = {'base': BASE_CSV, 'fixed': FIXED_CSV}
BASE_NAME, FIXED_NAME = '01047000.csv', '01047000_fixed.csv'
BASE_LABEL = '现用 01047000.csv'
FIXED_LABEL = '重建 01047000_fixed.csv'

# 现用最优超参数（results/phase2_v5/选参结果.csv 直接策略第 1 天）
ANN_ALPHA, ANN_HIDDEN = 10.0, (32,)
RF_DEPTH, RF_LEAF, RF_TREES = 16, 10, 300
# 现用最优超参数（results/phase3_enhanced/pytorch_lstm_grid_summary.csv direct_day1）
LSTM_HIDDEN, LSTM_LAYERS, LSTM_LR, LSTM_DROPOUT = 128, 1, 0.001, 0.2
LSTM_EPOCHS, LSTM_BATCH, LSTM_PATIENCE = 200, 32, 20
SEED = 42


# ------------------------------------------------------------ 对齐校验
def check_fixed():
    fx = pd.read_csv(FIXED_CSV)
    fx['Date'] = pd.to_datetime(fx['Date'])
    raw = DAYMET.read_text(encoding='utf-8').splitlines()
    start = next(i for i, l in enumerate(raw) if l.strip().lower().startswith('year,yday'))
    dm = pd.read_csv(io_string('\n'.join(raw[start:])))
    dm.columns = ['year', 'yday'] + DM_COLS
    # Daymet 的 yday 在闰年包含 2 月 29 日（fixed 文件与 daymet 逐值比对为 0 误差即为证据）
    dm['Date'] = [pd.Timestamp(year=int(y), month=1, day=1) + pd.Timedelta(days=int(d) - 1)
                  for y, d in zip(dm['year'], dm['yday'])]

    src = dm.set_index('Date')[DM_COLS].rename(columns=dict(zip(DM_COLS, MET)))
    err = {}
    joined = fx.set_index('Date').join(src, how='left', suffixes=('', '@daymet'))
    for c in MET:
        diff = (joined[c] - joined[f'{c}@daymet']).abs()
        err[c] = {'max_abs': float(diff.max()) if diff.notna().any() else None,
                  'rows_over_0.005': int((diff > 0.005).sum())}

    bench = pd.read_csv(BASE_CSV)
    info = {
        'path': str(FIXED_CSV),
        'shape': [int(fx.shape[0]), int(fx.shape[1])],
        'columns': list(fx.columns),
        'date_span': [str(fx['Date'].min().date()), str(fx['Date'].max().date())],
        'consecutive_days': bool((fx['Date'].diff().dropna().dt.days == 1).all()),
        'swe': {'min': float(fx['Swe'].min()), 'mean': float(fx['Swe'].mean()),
                'max': float(fx['Swe'].max()), 'nonzero': int((fx['Swe'] > 0).sum())},
        'daymet_missing_dates': [str(d.date()) for d in joined.index[joined['Swe@daymet'].isna()]],
        'align_max_abs': err,
        'align_max_abs_all': max(v['max_abs'] or 0.0 for v in err.values()),
        'discharge_identical_to_base': bool(
            np.allclose(fx['Discharge'].to_numpy(float), bench['Discharge'].to_numpy(float))),
        'date_identical_to_base': bool(
            (fx['Date'].to_numpy() == pd.to_datetime(bench['Date']).to_numpy()).all()),
    }

    samples = ['2000-01-01', '2000-06-15', '2002-03-01', '2004-03-01', '2004-12-30']
    rows = []
    for s in samples:
        t = pd.Timestamp(s)
        a = fx[fx['Date'] == t].iloc[0]
        b = dm[dm['Date'] == t].iloc[0]
        rows.append({'日期': s,
                     'Dayl(fixed)': float(a['Dayl']), 'Dayl(daymet)': float(b['dayl']),
                     'Prcp(fixed)': float(a['Prcp']), 'Prcp(daymet)': float(b['prcp']),
                     'Swe(fixed)': float(a['Swe']), 'Swe(daymet)': float(b['swe']),
                     '一致': bool(abs(a['Dayl'] - b['dayl']) < 1e-6
                                and abs(a['Prcp'] - b['prcp']) < 1e-6
                                and abs(a['Swe'] - b['swe']) < 1e-6)})
    sample_df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    sample_df.to_csv(OUT / 'alignment_samples.csv', index=False, encoding='utf-8-sig')
    (OUT / 'fixed_check.json').write_text(json.dumps(info, ensure_ascii=False, indent=2),
                                          encoding='utf-8')
    return info, sample_df


def io_string(text):
    import io
    return io.StringIO(text)


def md_table(df):
    """不依赖 tabulate 的 Markdown 表格渲染。"""
    def fmt(v):
        if isinstance(v, (bool, np.bool_)):
            return '是' if v else '否'
        if isinstance(v, (int, np.integer)):
            return str(int(v))
        if isinstance(v, (float, np.floating)):
            return f'{v:.6g}'
        return str(v)
    cols = list(df.columns)
    out = ['| ' + ' | '.join(cols) + ' |', '|' + '|'.join(['---'] * len(cols)) + '|']
    for _, row in df.iterrows():
        out.append('| ' + ' | '.join(fmt(row[c]) for c in cols) + ' |')
    return '\n'.join(out)


# ------------------------------------------------------------ 模型
def metrics(y, p):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return {'R2': 1 - ss_res / ss_tot, 'RMSE': float(np.sqrt(np.mean((y - p) ** 2)))}


def run_ann(X, y, Xt, yt):
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    pipe = Pipeline([('scale', StandardScaler()),
                     ('model', MLPRegressor(solver='lbfgs', max_iter=5000, max_fun=100000,
                                            tol=1e-5, early_stopping=False, random_state=SEED,
                                            alpha=ANN_ALPHA, hidden_layer_sizes=ANN_HIDDEN))])
    est = TransformedTargetRegressor(regressor=pipe, transformer=StandardScaler())
    t0 = time.time()
    est.fit(X, np.log1p(y))
    pred = np.clip(np.expm1(np.asarray(est.predict(Xt))), 0, None)
    return metrics(yt, pred), time.time() - t0


def run_rf(X, y, Xt, yt):
    from sklearn.ensemble import RandomForestRegressor
    est = RandomForestRegressor(n_estimators=RF_TREES, n_jobs=1, random_state=SEED,
                                max_depth=RF_DEPTH, min_samples_leaf=RF_LEAF)
    t0 = time.time()
    est.fit(X, np.log1p(y))
    pred = np.clip(np.expm1(np.asarray(est.predict(Xt))), 0, None)
    return metrics(yt, pred), time.time() - t0


def run_lstm(X, y, Xt, yt, n_feat):
    import torch
    from pytorch_lstm import LSTMRegressor, LSTMTrainer
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    n = len(X)
    split = int(n * 0.8)
    fit_hi = split - 7                      # 训练段与验证段之间留 7 天 gap（对应 7 天目标）
    model = LSTMRegressor(input_dim=n_feat, hidden_dim=LSTM_HIDDEN, num_layers=LSTM_LAYERS,
                          output_dim=1, dropout=LSTM_DROPOUT)
    trainer = LSTMTrainer(model, device='cpu', lr=LSTM_LR)
    t0 = time.time()
    trainer.fit(X[:fit_hi], y[:fit_hi], X_val=X[split:], y_val=y[split:],
                epochs=LSTM_EPOCHS, batch_size=LSTM_BATCH,
                early_stopping_patience=LSTM_PATIENCE, verbose=False)
    pred = np.clip(np.asarray(trainer.predict(Xt), float), 0, None)
    elapsed = time.time() - t0
    return (metrics(yt, pred), elapsed,
            {'fit_rows': int(fit_hi), 'val_rows': int(n - split),
             'epochs_run': int(len(trainer.history))})


# ------------------------------------------------------------ 单数据源
def run_source(tag):
    os.environ['RUNOFF_DATA_PATH'] = str(SOURCES[tag])
    sys.path.insert(0, str(CODE))
    import phase2_v5_data as d5
    import enhanced_features as ef

    train, test, cols, meta = d5.build_dataset_v5()          # 现有流水线，未改动
    X = train[cols].to_numpy(float)
    y = train['Target_lead1'].to_numpy(float)
    Xt = test[cols].to_numpy(float)
    yt = test['Target_lead1'].to_numpy(float)

    name = SOURCES[tag].name
    rows = []
    m, s = run_ann(X, y, Xt, yt)
    rows.append({'数据源': name, '模型': 'ANN（MLPRegressor, log1p）', '特征维度': len(cols),
                 'R²': m['R2'], 'RMSE': m['RMSE'], '训练耗时秒': round(s, 1)})
    m, s = run_rf(X, y, Xt, yt)
    rows.append({'数据源': name, '模型': 'RF（随机森林, log1p）', '特征维度': len(cols),
                 'R²': m['R2'], 'RMSE': m['RMSE'], '训练耗时秒': round(s, 1)})

    m = metrics(yt, test['Discharge'].to_numpy(float))
    rows.append({'数据源': name, '模型': '持久性基线（前一日流量）', '特征维度': 1,
                 'R²': m['R2'], 'RMSE': m['RMSE'], '训练耗时秒': 0.0})

    train_e, test_e, cols_e, meta_e = ef.get_enhanced_dataset()   # 88 维 enhanced 口径
    m, s, info = run_lstm(train_e[cols_e].to_numpy(float), train_e['Target_lead1'].to_numpy(float),
                          test_e[cols_e].to_numpy(float), test_e['Target_lead1'].to_numpy(float),
                          len(cols_e))
    rows.append({'数据源': name, '模型': 'LSTM（PyTorch, 早停）', '特征维度': len(cols_e),
                 'R²': m['R2'], 'RMSE': m['RMSE'], '训练耗时秒': round(s, 1)})

    df = pd.DataFrame(rows)
    meta_out = {'source': name, 'v5': meta, 'enhanced': meta_e, 'lstm': info,
                'test_rows_v5': int(len(test)), 'test_rows_enhanced': int(len(test_e)),
                'feature_cols_v5_n': int(len(cols)), 'feature_cols_enhanced_n': int(len(cols_e))}
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f'partial_{tag}.csv', index=False, encoding='utf-8-sig')
    (OUT / f'meta_{tag}.json').write_text(json.dumps(meta_out, ensure_ascii=False, indent=2,
                                                     default=str), encoding='utf-8')
    return df


# ------------------------------------------------------------ 汇总
def main():
    t_all = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    fixed_info, sample_df = check_fixed()

    for tag in SOURCES:
        p = OUT / f'partial_{tag}.csv'
        if p.exists():
            p.unlink()

    env = dict(os.environ)
    env['PYTHONIOENCODING'] = 'utf-8'
    env.pop('RUNOFF_DATA_PATH', None)
    timings = {}
    for tag in SOURCES:
        t0 = time.time()
        r = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--source', tag],
                           cwd=str(ROOT), env=env, capture_output=True, text=True,
                           encoding='utf-8', errors='replace')
        timings[tag] = round(time.time() - t0, 1)
        (OUT / f'run_{tag}.log').write_text(
            f'rc={r.returncode} elapsed={timings[tag]}s\n--- stdout ---\n{r.stdout}\n'
            f'--- stderr ---\n{r.stderr}', encoding='utf-8')
        if r.returncode != 0:
            raise RuntimeError(f'source {tag} failed, see results/probe/run_{tag}.log')

    df = pd.concat([pd.read_csv(OUT / f'partial_{tag}.csv') for tag in SOURCES], ignore_index=True)
    order = ['ANN（MLPRegressor, log1p）', 'RF（随机森林, log1p）',
             'LSTM（PyTorch, 早停）', '持久性基线（前一日流量）']
    df['_o'] = df['模型'].map({k: i for i, k in enumerate(order)})
    df = df.sort_values(['_o', '_o'], kind='stable')
    df = df.sort_values(['_o', '数据源'], kind='stable').drop(columns='_o')
    df.to_csv(OUT / 'probe_results.csv', index=False, encoding='utf-8-sig')

    metas = {t: json.loads((OUT / f'meta_{t}.json').read_text(encoding='utf-8')) for t in SOURCES}
    model_rows = df[df['模型'] != '持久性基线（前一日流量）']
    best_fixed = model_rows[model_rows['数据源'] == FIXED_NAME].sort_values('R²', ascending=False).iloc[0]
    best_base = model_rows[model_rows['数据源'] == BASE_NAME].sort_values('R²', ascending=False).iloc[0]

    deltas = []
    for model in order[:3]:
        b = df[(df['模型'] == model) & (df['数据源'] == BASE_NAME)].iloc[0]
        f = df[(df['模型'] == model) & (df['数据源'] == FIXED_NAME)].iloc[0]
        deltas.append({'模型': model, 'R²(现用)': float(b['R²']), 'R²(重建)': float(f['R²']),
                       'ΔR²': float(f['R²'] - b['R²']),
                       'RMSE(现用)': float(b['RMSE']), 'RMSE(重建)': float(f['RMSE']),
                       'ΔRMSE': float(f['RMSE'] - b['RMSE'])})
    delta_df = pd.DataFrame(deltas)
    delta_df.to_csv(OUT / 'probe_delta.csv', index=False, encoding='utf-8-sig')

    ok = bool(float(best_fixed['R²']) >= 0.70)
    conclusion = (
        f'重建数据源（01047000_fixed.csv）下最佳第 1 天 R² = {float(best_fixed["R²"]):.4f}'
        f'（{best_fixed["模型"]}，RMSE={float(best_fixed["RMSE"]):.1f} cfs），'
        f'{"达到" if ok else "未达到"} 0.70 阈值；'
        f'相对现用数据最佳（{best_base["模型"]}，R²={float(best_base["R²"]):.4f}，'
        f'RMSE={float(best_base["RMSE"]):.1f} cfs）'
        f'整体提升 ΔR²={float(best_fixed["R²"]) - float(best_base["R²"]):+.4f}。')
    (OUT / 'conclusion.txt').write_text(conclusion + '\n', encoding='utf-8')

    summary = {'total_seconds': round(time.time() - t_all, 1), 'source_seconds': timings,
               'fixed': fixed_info, 'metas': metas,
               'best_fixed': {'模型': best_fixed['模型'], 'R²': float(best_fixed['R²']),
                              'RMSE': float(best_fixed['RMSE'])},
               'best_base': {'模型': best_base['模型'], 'R²': float(best_base['R²']),
                             'RMSE': float(best_base['RMSE'])},
               'threshold_0.70': ok, 'conclusion': conclusion}
    (OUT / 'probe_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2,
                                                       default=str), encoding='utf-8')

    lines = ['# Phase-3 探针报告：Daymet v4 真实积雪数据集 vs 现用数据集', '',
             '## 一、重建数据与 daymet_full 的对齐校验', '',
             f'- 文件：{fixed_info["path"]}',
             f'- 形状：{fixed_info["shape"][0]} 行 × {fixed_info["shape"][1]} 列，'
             f'列：{fixed_info["columns"]}',
             f'- 日期：{fixed_info["date_span"][0]} ~ {fixed_info["date_span"][1]}，逐日连续='
             f'{fixed_info["consecutive_days"]}',
             f'- Swe：min={fixed_info["swe"]["min"]:.2f}，mean={fixed_info["swe"]["mean"]:.2f}，'
             f'max={fixed_info["swe"]["max"]:.2f}，非零行数={fixed_info["swe"]["nonzero"]}',
             f'- Daymet 无值日（已 ffill）：{fixed_info["daymet_missing_dates"]}',
             f'- 与 daymet 逐值最大绝对误差（7 个气象列）：{fixed_info["align_max_abs_all"]:.10f}',
             f'- Discharge/Date 与 01047000.csv 完全一致：'
             f'{fixed_info["discharge_identical_to_base"]} / {fixed_info["date_identical_to_base"]}',
             '', '### 抽样对照（5 个日期，"一致"列要求 dayl/prcp/swe 三值均为 0 误差）', '',
             md_table(sample_df), '',
             '## 二、探针结果（第 1 天预见期，2004 年测试集）', '',
             md_table(df), '',
             '### 相对现用数据的增量', '',
             md_table(delta_df), '',
             '## 三、口径核对', '']
    for t in SOURCES:
        m = metas[t]
        v5, en = m['v5'], m['enhanced']
        lines.append(f'- {m["source"]}：v5 训练 {v5["训练样本"]} 行（{v5["训练起"]}~{v5["训练止"]}）、'
                     f'测试 {m["test_rows_v5"]} 行（{v5["测试起"]}~{v5["测试止"]}）、特征 {m["feature_cols_v5_n"]} 维'
                     f'（剔除零方差列 {len(v5["剔除零方差列"])} 个）')
        lines.append(f'  - enhanced 测试 {m["test_rows_enhanced"]} 行、特征 {m["feature_cols_enhanced_n"]} 维；'
                     f'LSTM 早停：{m["lstm"]}')
    lines += ['', '## 四、结论', '', conclusion, '',
              f'- 总耗时 {summary["total_seconds"]}s（现用数据源 {timings["base"]}s，'
              f'重建数据源 {timings["fixed"]}s）']
    (OUT / 'probe_report.md').write_text('\n'.join(lines), encoding='utf-8')
    print('OK probe_report.md written')


if __name__ == '__main__':
    if '--source' in sys.argv:
        tag = sys.argv[sys.argv.index('--source') + 1]
        run_source(tag)
        print(f'OK source {tag}')
    else:
        main()
