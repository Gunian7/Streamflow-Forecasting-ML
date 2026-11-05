"""Phase-two v3 baselines and shared metric helpers."""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

from phase2_v3_config import PROTOCOL

MAX_HORIZON = PROTOCOL['max_horizon']


def metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float).ravel()
    predicted = np.asarray(predicted, dtype=float).ravel()
    denom = float(((actual - actual.mean()) ** 2).sum())
    return {
        'RMSE（cfs）': float(np.sqrt(mean_squared_error(actual, predicted))),
        'MAE（cfs）': float(mean_absolute_error(actual, predicted)),
        'R²': float(1 - ((actual - predicted) ** 2).sum() / denom),
        'NSE': float(1 - ((actual - predicted) ** 2).sum() / denom),
        '样本数': int(actual.size),
    }


def persistence(last_known, horizon=MAX_HORIZON):
    """Repeat the last observed flow for every future step."""
    last = np.asarray(last_known, dtype=float).reshape(-1, 1)
    if horizon != MAX_HORIZON:
        raise ValueError('本实验固定预测 1-7 天')
    return np.repeat(last, horizon, axis=1)


def climatology_month_mean(issue_dates, train_df):
    """Predict with the training-set mean flow of each target month."""
    table = train_df.assign(_m=train_df['Date'].dt.month).groupby('_m')['Discharge'].mean()
    overall = float(train_df['Discharge'].mean())
    out = np.zeros((len(issue_dates), MAX_HORIZON))
    for i, date in enumerate(pd.to_datetime(pd.Series(list(issue_dates))).items()):
        _, d = date
        for h in range(1, MAX_HORIZON + 1):
            month = (d + pd.Timedelta(days=h)).month
            out[i, h - 1] = float(table.get(month, overall))
    return out


def evaluate_baselines(train_df, test_df):
    """Score both baselines per horizon on the held-out test segment."""
    rows = []
    preds = {
        '持续性': persistence(test_df['Discharge_lag1'].to_numpy()),
        '月份气候均值': climatology_month_mean(test_df['Date'], train_df),
    }
    for name, matrix in preds.items():
        for h in range(1, MAX_HORIZON + 1):
            row = {'基线': name, '预见期（天）': h}
            row.update(metrics(test_df[f'Target_lead{h}'], matrix[:, h - 1]))
            rows.append(row)
    return pd.DataFrame(rows), preds


if __name__ == '__main__':
    import phase2_v3_data as d3
    train, test, cols, meta = d3.build_dataset()
    frame, _ = evaluate_baselines(train, test)
    print(frame.to_string(index=False))
