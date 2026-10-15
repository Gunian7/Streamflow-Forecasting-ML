"""Fixed feature groups for ablation (no supervision, no leakage)."""
import numpy as np
import pandas as pd

from phase2_v3_config import PROTOCOL

HORIZON = PROTOCOL['max_horizon']
LAGS = [1, 2, 3, 7]
WINDOWS = [3, 7]


def _rolling_stats(col, prefix):
    out = [f'{col}_roll_{stat}_{w}' for w in WINDOWS
           for stat in ['mean', 'std', 'max', 'min']]
    return [c for c in out if c.startswith(prefix)]


def feature_groups(all_cols):
    """Return {group_name: [columns]}. Groups are cumulative (A ⊂ B ⊂ C ⊂ D)."""
    available = set(all_cols)

    def pick(candidates):
        missing = [c for c in candidates if c not in available]
        if missing:
            raise KeyError(f'特征缺失: {missing}')
        return list(candidates)

    lag_flow = pick([f'Discharge_lag{k}' for k in LAGS])
    seasons = pick(['Season_spring', 'Season_summer', 'Season_autumn', 'Season_winter'])
    cycles = pick(['Month_sin', 'Month_cos', 'DayOfYear_sin', 'DayOfYear_cos'])

    met_lag = []
    for var in ['Prcp', 'Tmax', 'Tmin', 'Srad', 'Vp']:
        met_lag += pick([f'{var}_lag{k}' for k in LAGS])
    met_now = pick(['Prcp', 'Tmax', 'Tmin', 'Srad', 'Vp', 'Dayl'])

    roll = []
    for var in ['Discharge', 'Prcp', 'Tmax']:
        roll += _rolling_stats(var, var + '_roll_')

    group_a = lag_flow + seasons + cycles
    group_b = group_a + met_now + met_lag
    group_c = group_b + roll
    return {'A_历史径流与季节': group_a, 'B_加气象观测': group_b, 'C_加窗口统计': group_c}


def add_long_windows(df):
    """Add 14/30-day cumulative precipitation and mean flow (history only)."""
    out = df.copy()
    for w in [14, 30]:
        out[f'Prcp_roll_sum_{w}'] = out['Prcp'].shift(1).rolling(window=w).sum()
        out[f'Discharge_roll_mean_{w}'] = out['Discharge'].shift(1).rolling(window=w).mean()
    out['Tmax_minus_Tmin'] = out['Tmax'] - out['Tmin']
    return out


def group_d_columns(all_cols):
    base = feature_groups(all_cols)['C_加窗口统计']
    return base + ['Prcp_roll_sum_14', 'Prcp_roll_sum_30',
                   'Discharge_roll_mean_14', 'Discharge_roll_mean_30',
                   'Tmax_minus_Tmin']


def common_dates(frames):
    """Intersection of issue dates so window length cannot change the sample."""
    idx = None
    for f in frames:
        dates = set(f['Date'])
        idx = dates if idx is None else (idx & dates)
    if not idx:
        raise ValueError('各组没有共同日期，无法公平比较')
    return pd.DatetimeIndex(sorted(idx))


if __name__ == '__main__':
    import phase2_v3_data as d3
    train, test, cols, meta = d3.build_dataset()
    groups = feature_groups(cols)
    for name, members in groups.items():
        print(name, len(members))
