# -*- coding: utf-8 -*-
"""贵州茅台次日开盘价预测：滚动时间验证（Walk-forward validation）

Compare four methods on four non-overlapping 150-row test windows:
  1) Naive persistence: predict next Open = previous observed Open
  2) ARIMA(5,1,0): one-step-ahead forecast, update state with each newly observed Open
  3) RAW-LSTM-Price: predict the next absolute Open price
  4) RAW-LSTM-Return: predict Open-to-Open return, then convert back to price

Data split (for 1500 rows):
  Fold 1 train [0:900],   test [900:1050]
  Fold 2 train [0:1050],  test [1050:1200]
  Fold 3 train [0:1200],  test [1200:1350]
  Fold 4 train [0:1350],  test [1350:1500]

The model/hyperparameters are fixed in advance. For each test point, predictions
only use data available before that point. This script writes into a new output
folder and does not overwrite previous experiments.

Run from the folder containing ashare_600519.csv:
    python ashare_walk_forward_validation.py
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import statsmodels
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings("ignore")

# -------------------- Fixed experiment settings --------------------
DATA_CANDIDATES = ["ashare_600519.csv", "ashare_600519(1).csv"]
OUTPUT_DIR = Path("results_walk_forward_validation")
PAST_DAYS = 90
EPOCHS = 100
BATCH_SIZE = 32
SEED = 42
ARIMA_ORDER = (5, 1, 0)
FOLD_TEST_SIZE = 150
INITIAL_TRAIN_SIZE = 900
MODEL_NAMES = ["Naive_Persistence", "ARIMA_510", "RAW_LSTM_Price", "RAW_LSTM_Return"]


def set_seed(seed=SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def load_data():
    data_path = next((Path(p) for p in DATA_CANDIDATES if Path(p).exists()), None)
    if data_path is None:
        raise FileNotFoundError(
            "找不到 ashare_600519.csv 或 ashare_600519(1).csv。"
            "请将脚本放在行情 CSV 所在目录，或修改 DATA_CANDIDATES。"
        )
    df = pd.read_csv(data_path)
    # Normalize case and whitespace for basic column compatibility.
    rename = {}
    for col in df.columns:
        low = str(col).strip().lower()
        if low in {"date", "open"}:
            rename[col] = low
    df = df.rename(columns=rename)
    if not {"date", "open"}.issubset(df.columns):
        raise ValueError(f"CSV 至少需要 date 和 open 两列，当前字段：{list(df.columns)}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["open"] = pd.to_numeric(df["open"], errors="coerce")
    df = (df.dropna(subset=["date", "open"])
            .sort_values("date")
            .drop_duplicates("date", keep="last")
            .reset_index(drop=True))
    if len(df) < INITIAL_TRAIN_SIZE + 4 * FOLD_TEST_SIZE:
        raise ValueError(
            f"有效数据只有 {len(df)} 行；当前配置至少需要 {INITIAL_TRAIN_SIZE + 4 * FOLD_TEST_SIZE} 行。"
        )
    print(f"数据文件：{data_path}")
    print(f"有效数据：{len(df)} 行；日期范围：{df.date.iloc[0].date()} 至 {df.date.iloc[-1].date()}")
    return df, data_path


def build_lstm():
    model = Sequential([
        Input(shape=(PAST_DAYS, 1)),
        LSTM(50, return_sequences=True), Dropout(0.2),
        LSTM(50, return_sequences=True), Dropout(0.2),
        LSTM(50, return_sequences=True), Dropout(0.2),
        LSTM(50, return_sequences=False), Dropout(0.2),
        Dense(1),
    ])
    model.compile(optimizer=Adam(), loss="mean_squared_error")
    return model


def make_price_train_sequences(x_scaled, y_scaled):
    X, y = [], []
    for target_idx in range(PAST_DAYS, len(y_scaled)):
        X.append(x_scaled[target_idx - PAST_DAYS:target_idx, :])
        y.append(y_scaled[target_idx, 0])
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)


def make_test_sequences(all_x_scaled, train_end, test_end):
    """Each test target i receives only inputs in [i-PAST_DAYS, i)."""
    X = []
    for target_idx in range(train_end, test_end):
        X.append(all_x_scaled[target_idx - PAST_DAYS:target_idx, :])
    return np.asarray(X, dtype=np.float32)


def fit_lstm_predict(x_raw, opens, train_end, test_end, target_type, fold_num):
    """Train one model on [0, train_end), forecast [train_end, test_end)."""
    input_scaler = MinMaxScaler(feature_range=(0, 1))
    x_train_scaled = input_scaler.fit_transform(x_raw[:train_end].reshape(-1, 1))
    x_all_scaled = input_scaler.transform(x_raw[:test_end].reshape(-1, 1))
    X_train = x_train_scaled
    X_test = make_test_sequences(x_all_scaled, train_end, test_end)

    if target_type == "price":
        target_scaler = MinMaxScaler(feature_range=(0, 1))
        y_train_all = target_scaler.fit_transform(opens[:train_end].reshape(-1, 1))
    elif target_type == "return":
        all_returns = np.zeros(len(opens), dtype=np.float64)
        all_returns[1:] = (opens[1:] - opens[:-1]) / opens[:-1]
        y_train_values = all_returns[PAST_DAYS:train_end].reshape(-1, 1)
        target_scaler = StandardScaler()
        target_scaler.fit(y_train_values)
        y_train_all = np.full((train_end, 1), np.nan, dtype=np.float64)
        y_train_all[PAST_DAYS:train_end] = target_scaler.transform(y_train_values)
    else:
        raise ValueError("target_type 必须为 price 或 return")

    X_train_list, y_train_list = [], []
    for target_idx in range(PAST_DAYS, train_end):
        X_train_list.append(x_train_scaled[target_idx - PAST_DAYS:target_idx, :])
        if target_type == "price":
            y_train_list.append(y_train_all[target_idx, 0])
        else:
            y_train_list.append(y_train_all[target_idx, 0])
    X_train_seq = np.asarray(X_train_list, dtype=np.float32)
    y_train_seq = np.asarray(y_train_list, dtype=np.float32)

    tf.keras.backend.clear_session()
    set_seed(SEED)
    model = build_lstm()
    model.fit(X_train_seq, y_train_seq, epochs=EPOCHS, batch_size=BATCH_SIZE,
              shuffle=False, verbose=0)
    preds_scaled = model.predict(X_test, verbose=0).reshape(-1, 1)
    pred_target = target_scaler.inverse_transform(preds_scaled).reshape(-1)

    if target_type == "price":
        return pred_target
    # Return at target index i is based on Open[i-1]; then convert to predicted Open[i].
    previous_open = opens[train_end - 1:test_end - 1]
    return previous_open * (1.0 + pred_target)


def arima_walk_forward_predict(opens, train_end, test_end):
    """Fit ARIMA on training history; sequentially forecast one step and append observed values."""
    fitted = ARIMA(opens[:train_end], order=ARIMA_ORDER).fit()
    predictions = []
    for target_idx in range(train_end, test_end):
        forecast = float(np.asarray(fitted.forecast(steps=1)).reshape(-1)[0])
        predictions.append(forecast)
        # After the day's Open is observed, append it before predicting the next day.
        fitted = fitted.append(endog=[float(opens[target_idx])], refit=False)
    return np.asarray(predictions, dtype=np.float64)


def metric_row(model_name, fold, dates, previous_open, actual_open, predicted_open):
    prev = np.asarray(previous_open, dtype=float)
    actual = np.asarray(actual_open, dtype=float)
    pred = np.asarray(predicted_open, dtype=float)
    true_up = (actual > prev).astype(int)
    pred_up = (pred > prev).astype(int)
    return {
        "Fold": int(fold["fold_num"]),
        "Model": model_name,
        "Train_End_Index_Exclusive": int(fold["train_end"]),
        "Test_Start_Date": str(dates[0]),
        "Test_End_Date": str(dates[-1]),
        "Samples": int(len(actual)),
        "MAE": float(mean_absolute_error(actual, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, pred))),
        "Direction_Accuracy": float(accuracy_score(true_up, pred_up)),
        "Balanced_Accuracy": float(balanced_accuracy_score(true_up, pred_up)),
        "Precision_Up": float(precision_score(true_up, pred_up, zero_division=0)),
        "Recall_Up": float(recall_score(true_up, pred_up, zero_division=0)),
        "F1_Up": float(f1_score(true_up, pred_up, zero_division=0)),
        "Actual_Up_Days": int(true_up.sum()),
        "Predicted_Up_Days": int(pred_up.sum()),
        "Mean_Prediction_Bias": float(np.mean(pred - actual)),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df, data_path = load_data()
    dates = df["date"].dt.strftime("%Y-%m-%d").to_numpy()
    opens = df["open"].to_numpy(dtype=np.float64)
    n = len(opens)

    if n != 1500:
        print(f"提示：当前有效行数为 {n}，将按固定900行起始训练窗口和150行测试窗运行；最后剩余部分按可用完整窗口处理。")

    # Generate expanding training windows and consecutive, non-overlapping test windows.
    folds = []
    train_end = INITIAL_TRAIN_SIZE
    fold_num = 1
    while train_end + FOLD_TEST_SIZE <= n:
        folds.append({"fold_num": fold_num, "train_end": train_end, "test_end": train_end + FOLD_TEST_SIZE})
        train_end += FOLD_TEST_SIZE
        fold_num += 1
    if not folds:
        raise ValueError("未能构造有效的滚动验证窗口。")

    rows = []
    prediction_frames = []
    print(f"\n固定参数：Past_Days={PAST_DAYS}, Epochs={EPOCHS}, Batch_Size={BATCH_SIZE}, Seed={SEED}, ARIMA{ARIMA_ORDER}")
    print(f"共 {len(folds)} 个测试窗口，每个窗口 {FOLD_TEST_SIZE} 个交易日。")

    for fold in folds:
        f = fold["fold_num"]
        train_end, test_end = fold["train_end"], fold["test_end"]
        idx = np.arange(train_end, test_end)
        test_dates = dates[idx]
        actual = opens[idx]
        previous = opens[idx - 1]
        print(f"\n--- Fold {f}: train rows [0:{train_end}], test rows [{train_end}:{test_end}] | {test_dates[0]} -> {test_dates[-1]} ---")

        predictions = {}
        predictions["Naive_Persistence"] = previous.copy()
        print("Naive 完成")
        predictions["ARIMA_510"] = arima_walk_forward_predict(opens, train_end, test_end)
        print("ARIMA(5,1,0) 完成")
        predictions["RAW_LSTM_Price"] = fit_lstm_predict(opens, opens, train_end, test_end, "price", f)
        print("RAW-LSTM-Price 完成")
        predictions["RAW_LSTM_Return"] = fit_lstm_predict(opens, opens, train_end, test_end, "return", f)
        print("RAW-LSTM-Return 完成")

        for name in MODEL_NAMES:
            pred = predictions[name]
            rows.append(metric_row(name, fold, test_dates, previous, actual, pred))
            true_up = (actual > previous).astype(int)
            pred_up = (pred > previous).astype(int)
            prediction_frames.append(pd.DataFrame({
                "Fold": f,
                "Model": name,
                "date": test_dates,
                "previous_actual_open": previous,
                "actual_open": actual,
                "predicted_open": pred,
                "true_return": (actual - previous) / previous,
                "predicted_return": (pred - previous) / previous,
                "true_up": true_up,
                "predicted_up": pred_up,
            }))

    metrics = pd.DataFrame(rows)
    pred_all = pd.concat(prediction_frames, ignore_index=True)
    metrics.to_csv(OUTPUT_DIR / "fold_metrics.csv", index=False, encoding="utf-8-sig")
    pred_all.to_csv(OUTPUT_DIR / "predictions_all_folds.csv", index=False, encoding="utf-8-sig")

    # Summary over folds: mean/std of fold metrics; use ddof=1 for between-fold std.
    metric_cols = ["MAE", "RMSE", "Direction_Accuracy", "Balanced_Accuracy", "Precision_Up", "Recall_Up", "F1_Up", "Predicted_Up_Days", "Mean_Prediction_Bias"]
    summary_rows = []
    for name, group in metrics.groupby("Model", sort=False):
        item = {"Model": name, "Folds": int(group["Fold"].nunique()), "Total_Test_Samples": int(group["Samples"].sum())}
        for col in metric_cols:
            vals = group[col].astype(float)
            item[f"{col}_mean"] = float(vals.mean())
            item[f"{col}_std_across_folds"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        summary_rows.append(item)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUT_DIR / "summary_by_model.csv", index=False, encoding="utf-8-sig")

    config = pd.DataFrame([{
        "data_file": data_path.name,
        "rows": n,
        "initial_train_size": INITIAL_TRAIN_SIZE,
        "test_window_size": FOLD_TEST_SIZE,
        "folds": len(folds),
        "past_days": PAST_DAYS,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "seed": SEED,
        "arima_order": str(ARIMA_ORDER),
        "target": "next trading day original Open",
        "previous_price_for_direction": "previous actual Open",
        "arima_policy": "fit once per fold; one-step forecast; append observed Open with refit=False",
        "note": "Expanding-window exploratory evaluation; test windows follow earlier windows chronologically. Prior development has viewed overlapping historical periods."
    }])
    config.to_csv(OUTPUT_DIR / "experiment_config.csv", index=False, encoding="utf-8-sig")

    print("\n===== 各测试窗口结果 =====")
    print(metrics[["Fold", "Model", "MAE", "RMSE", "Direction_Accuracy", "Balanced_Accuracy", "Predicted_Up_Days"]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n===== 跨窗口汇总（均值 / 窗口间标准差） =====")
    print(summary[["Model", "MAE_mean", "MAE_std_across_folds", "RMSE_mean", "RMSE_std_across_folds", "Balanced_Accuracy_mean", "Direction_Accuracy_mean"]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\n结果已保存至：{OUTPUT_DIR.resolve()}")
    print("请提交 fold_metrics.csv、summary_by_model.csv、experiment_config.csv。")

if __name__ == "__main__":
    main()
