"""Tests for physically meaningful enhanced runoff features."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))

import enhanced_features as ef


def sample_frame(n=12):
    return pd.DataFrame({
        'Date': pd.date_range('2000-01-01', periods=n, freq='D'),
        'Discharge': np.arange(n, dtype=float) * 2,
        'Prcp': np.arange(n, dtype=float),
        'Tmax': np.arange(n, dtype=float) - 3,
        'Swe': np.arange(n, dtype=float) * 10,
    })


def test_adds_expected_feature_families_without_mutating_input():
    frame = sample_frame()
    original = frame.copy(deep=True)

    enhanced = ef.add_enhanced_features(frame)

    assert len(enhanced.columns) - len(frame.columns) == 23
    assert {'Discharge_ewm3', 'Prcp_ewm7', 'Swe_diff3',
            'Prcp_x_Tmax', 'Swe_x_PDD', 'Prcp_x_Swe',
            'Discharge_std_7', 'Prcp_cum7', 'PDD_cum3',
            'Q_to_P_ratio'} <= set(enhanced.columns)
    pd.testing.assert_frame_equal(frame, original)


def test_interactions_and_cumulative_features_match_physical_definitions():
    enhanced = ef.add_enhanced_features(sample_frame())

    assert enhanced.loc[4, 'Prcp_x_Tmax'] == 4 * 1
    assert enhanced.loc[2, 'Swe_x_PDD'] == 20 * 0
    assert enhanced.loc[5, 'Swe_x_PDD'] == 50 * 2
    assert enhanced.loc[4, 'Prcp_x_Swe'] == 4 * 40
    assert enhanced.loc[6, 'Prcp_cum3'] == 4 + 5 + 6
    assert enhanced.loc[6, 'PDD_cum3'] == 1 + 2 + 3


def test_feature_list_matches_enhanced_feature_count():
    frame = sample_frame()
    enhanced = ef.add_enhanced_features(frame)

    added = ef.enhanced_feature_columns(frame.columns)

    assert len(added) == 23
    assert set(added) == set(enhanced.columns) - set(frame.columns)
