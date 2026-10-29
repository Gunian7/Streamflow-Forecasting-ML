import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))

from enhanced_ensemble import align_by_date, metrics, select_feature_matrix


def test_align_by_date_keeps_only_common_dates_in_order():
    old = pd.DataFrame({"Date": pd.to_datetime(["2020-01-01", "2020-01-02"]), "x": [1, 2]})
    enhanced = pd.DataFrame({"Date": pd.to_datetime(["2020-01-02", "2020-01-03"]), "x": [20, 30]})

    old_aligned, enhanced_aligned = align_by_date(old, enhanced)

    assert old_aligned["Date"].tolist() == [pd.Timestamp("2020-01-02")]
    assert enhanced_aligned["Date"].tolist() == [pd.Timestamp("2020-01-02")]
    assert old_aligned["x"].tolist() == [2]
    assert enhanced_aligned["x"].tolist() == [20]


def test_select_feature_matrix_rejects_dimension_mismatch():
    frame = pd.DataFrame({"a": [1.0], "b": [2.0]})

    with pytest.raises(ValueError, match="feature dimension mismatch"):
        select_feature_matrix(frame, ["a"], expected_dim=2)


def test_metrics_returns_finite_regression_scores():
    result = metrics(np.array([1.0, 2.0]), np.array([1.0, 3.0]))

    assert result["RMSE"] == pytest.approx(2**-0.5)
    assert result["MAE"] == pytest.approx(0.5)
    assert result["R²"] == pytest.approx(-1.0)
