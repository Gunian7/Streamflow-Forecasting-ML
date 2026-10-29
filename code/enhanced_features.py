"""Causal time-series and physically meaningful runoff feature engineering."""
from pathlib import Path
import os
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def enhanced_feature_columns(existing_columns=()):
    """Return the added feature names in deterministic order."""
    existing = set(existing_columns)
    candidates = []
    for col in ['Discharge', 'Prcp', 'Tmax']:
        candidates.extend([f'{col}_ewm3', f'{col}_ewm7'])
    for col in ['Discharge', 'Prcp', 'Swe']:
        candidates.extend([f'{col}_diff1', f'{col}_diff3'])
    candidates.extend(['Prcp_x_Tmax', 'Swe_x_PDD', 'Prcp_x_Swe'])
    for window in [3, 7]:
        candidates.extend([f'Discharge_std_{window}', f'Prcp_std_{window}'])
    candidates.extend(['Prcp_cum3', 'Prcp_cum7', 'PDD_cum3', 'Q_to_P_ratio'])
    return [name for name in candidates if name not in existing]


def add_enhanced_features(df):
    """Append EWMA, change, interaction, variability and accumulation features."""
    missing = sorted({'Discharge', 'Prcp', 'Tmax', 'Swe'} - set(df.columns))
    if missing:
        raise ValueError(f'Missing required source columns: {missing}')

    out = df.copy()

    for col in ['Discharge', 'Prcp', 'Tmax']:
        out[f'{col}_ewm3'] = out[col].ewm(span=3, adjust=False).mean()
        out[f'{col}_ewm7'] = out[col].ewm(span=7, adjust=False).mean()

    for col in ['Discharge', 'Prcp', 'Swe']:
        out[f'{col}_diff1'] = out[col].diff(1)
        out[f'{col}_diff3'] = out[col].diff(3)

    positive_degree_days = out['Tmax'].clip(lower=0)
    out['Prcp_x_Tmax'] = out['Prcp'] * out['Tmax']
    out['Swe_x_PDD'] = out['Swe'] * positive_degree_days
    out['Prcp_x_Swe'] = out['Prcp'] * out['Swe']

    for window in [3, 7]:
        out[f'Discharge_std_{window}'] = out['Discharge'].rolling(window).std()
        out[f'Prcp_std_{window}'] = out['Prcp'].rolling(window).std()

    out['Prcp_cum3'] = out['Prcp'].rolling(3).sum()
    out['Prcp_cum7'] = out['Prcp'].rolling(7).sum()
    out['PDD_cum3'] = positive_degree_days.rolling(3).sum()
    out['Q_to_P_ratio'] = out['Discharge'] / (out['Prcp'].rolling(7).sum() + 1)
    return out


def get_enhanced_dataset():
    """Return enhanced train/test frames, model feature columns and metadata."""
    os.environ.setdefault('RUNOFF_DATA_PATH', str(ROOT / '01047000_fixed.csv'))
    code_dir = str(ROOT / 'code')
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)
    import phase2_v5_data as d5

    train, test, base_cols, meta = d5.build_dataset_v5()
    added = enhanced_feature_columns(base_cols)
    train_enhanced = add_enhanced_features(train)
    test_enhanced = add_enhanced_features(test)
    feature_cols = list(base_cols) + added

    train_enhanced = train_enhanced.dropna(subset=added)
    test_enhanced = test_enhanced.dropna(subset=added)
    meta = dict(meta)
    meta['enhanced_features'] = len(added)
    meta['total_features'] = len(feature_cols)
    meta['added_feature_names'] = added
    return train_enhanced, test_enhanced, feature_cols, meta


if __name__ == '__main__':
    train, test, cols, meta = get_enhanced_dataset()
    added = meta['added_feature_names']
    print(f"训练集: {len(train)} 条")
    print(f"测试集: {len(test)} 条")
    print(f"基础特征: {meta['特征数量']}")
    print(f"新增特征: {meta['enhanced_features']}")
    print(f"总特征数: {meta['total_features']}")
    print('新增特征列表:')
    for name in added:
        print(f'  - {name}')
