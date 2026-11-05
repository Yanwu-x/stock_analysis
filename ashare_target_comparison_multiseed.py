# -*- coding: utf-8 -*-
"""贵州茅台次日开盘价预测：价格目标 vs 收益率目标，多随机种子对比。

比较：
1) Naive：预测次日 Open = 前一交易日 Open
2) RAW-LSTM-Price：过去90日 Open -> 下一日绝对 Open
3) RAW-LSTM-Return：过去90日 Open -> 下一日开盘收益率，再还原为预测 Open

统一设置：按日期升序，前80%训练、后20%测试；90日窗口；4层LSTM，每层50单元，Dropout=0.2；
100 epochs，batch_size=32；种子42、123、2026。所有Scaler只用训练区间拟合。

输出目录：results_target_comparison_multiseed
说明：当前 300 日测试段此前已用于探索性实验，因此本程序用于目标变换的稳健性检查，
不能被描述为完全未接触过的最终独立测试。
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, accuracy_score,
    balanced_accuracy_score, precision_score, recall_score, f1_score
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings("ignore")

DATA_CANDIDATES = ["ashare_600519.csv", "ashare_600519(1).csv"]
OUTPUT_DIR = Path("results_target_comparison_multiseed")
TRAIN_RATIO = 0.80
PAST_DAYS = 90
EPOCHS = 100
BATCH_SIZE = 32
SEEDS = [42, 123, 2026]
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass



def locate_data() -> Path:
    for name in DATA_CANDIDATES:
        p = Path(name)
        if p.exists():
            return p
    raise FileNotFoundError(
        "找不到 ashare_600519.csv 或 ashare_600519(1).csv。请把脚本放到行情CSV所在目录。"
    )


def load_data() -> pd.DataFrame:
    path = locate_data()
    df = pd.read_csv(path)
    required = {"date", "open"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV至少需要 date、open 两列，当前列：{list(df.columns)}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["open"] = pd.to_numeric(df["open"], errors="coerce")
    df = (df.dropna(subset=["date", "open"])
            .sort_values("date")
            .drop_duplicates("date", keep="last")
            .reset_index(drop=True))
    if len(df) <= PAST_DAYS + 10:
        raise ValueError(f"有效样本不足：{len(df)}")
    print(f"数据文件：{path} | 行数：{len(df)} | 日期：{df.date.iloc[0].date()} 至 {df.date.iloc[-1].date()}")
    return df


def build_model() -> tf.keras.Model:
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


def make_train_sequences(x_scaled: np.ndarray, y_scaled: np.ndarray, end: int):
    X, y = [], []
    for i in range(PAST_DAYS, end):
        X.append(x_scaled[i-PAST_DAYS:i, :])
        y.append(y_scaled[i, 0])
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)


def make_test_sequences(train_x_scaled: np.ndarray, test_x_scaled: np.ndarray):
    combined = np.concatenate([train_x_scaled[-PAST_DAYS:], test_x_scaled], axis=0)
    X = [combined[j:j+PAST_DAYS, :] for j in range(len(test_x_scaled))]
    return np.asarray(X, dtype=np.float32)


def train_predict(X_train, y_train, X_test, seed: int, label: str):
    tf.keras.backend.clear_session()
    set_seed(seed)
    model = build_model()
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=False, verbose=0)
    preds = model.predict(X_test, verbose=0).reshape(-1, 1)
    print(f"  seed={seed} | {label}完成 | train={len(X_train)} test={len(X_test)}")
    return preds


def evaluate(name, seed, dates, previous_open, actual_open, predicted_open, predicted_return=None):
    previous_open = np.asarray(previous_open, dtype=float)
    actual_open = np.asarray(actual_open, dtype=float)
    predicted_open = np.asarray(predicted_open, dtype=float)
    true_return = (actual_open - previous_open) / previous_open
    pred_return = ((predicted_open - previous_open) / previous_open) if predicted_return is None else np.asarray(predicted_return, dtype=float)
    true_up = (actual_open > previous_open).astype(int)
    pred_up = (predicted_open > previous_open).astype(int)
    row = {
        "Model": name, "Seed": seed,
        "MAE": float(mean_absolute_error(actual_open, predicted_open)),
        "RMSE": float(np.sqrt(mean_squared_error(actual_open, predicted_open))),
        "Strict_Direction_Accuracy": float(accuracy_score(true_up, pred_up)),
        "Balanced_Accuracy": float(balanced_accuracy_score(true_up, pred_up)),
        "Precision_Up": float(precision_score(true_up, pred_up, zero_division=0)),
        "Recall_Up": float(recall_score(true_up, pred_up, zero_division=0)),
        "F1_Up": float(f1_score(true_up, pred_up, zero_division=0)),
        "Actual_Up_Days": int(true_up.sum()),
        "Predicted_Up_Days": int(pred_up.sum()),
        "Mean_Prediction_Bias": float(np.mean(predicted_open - actual_open)),
        "Return_Correlation": float(np.corrcoef(true_return, pred_return)[0, 1]) if np.std(pred_return) > 0 else 0.0,
        "Test_Samples": int(len(actual_open)),
    }
    pred_df = pd.DataFrame({
        "date": dates,
        "previous_actual_open": previous_open,
        "actual_open": actual_open,
        "predicted_open": predicted_open,
        "actual_return": true_return,
        "predicted_return": pred_return,
        "true_up": true_up,
        "predicted_up": pred_up,
    })
    return row, pred_df


def main():
    df = load_data()
    opens = df["open"].to_numpy(dtype=np.float64)
    dates = df["date"].dt.strftime("%Y-%m-%d").to_numpy()
    n = len(opens)
    split = int(n * TRAIN_RATIO)
    if split <= PAST_DAYS or n - split < 2:
        raise ValueError("训练集或测试集不足，检查 TRAIN_RATIO 与 PAST_DAYS。")

    open_train, open_test = opens[:split], opens[split:]
    test_dates = dates[split:]
    previous_open = np.concatenate(([opens[split-1]], open_test[:-1]))

    input_scaler = MinMaxScaler(feature_range=(0, 1))
    x_train_scaled = input_scaler.fit_transform(open_train.reshape(-1, 1))
    x_test_scaled = input_scaler.transform(open_test.reshape(-1, 1))
    X_train = None
    X_test = make_test_sequences(x_train_scaled, x_test_scaled)

    # 价格目标：与同一输入序列对齐。
    price_scaler = MinMaxScaler(feature_range=(0, 1))
    y_price_train_all = price_scaler.fit_transform(open_train.reshape(-1, 1))
    X_train, y_price_train = make_train_sequences(x_train_scaled, y_price_train_all, split)

    # 收益率目标 r_t=(Open_t-Open_{t-1})/Open_{t-1}；第t个标签仅用前一日Open作基准。
    returns = np.full(n, np.nan, dtype=np.float64)
    returns[1:] = (opens[1:] - opens[:-1]) / opens[:-1]
    valid_train_returns = returns[PAST_DAYS:split].reshape(-1, 1)
    return_scaler = StandardScaler().fit(valid_train_returns)
    y_return_train_all = np.zeros((split, 1), dtype=np.float64)
    y_return_train_all[PAST_DAYS:, 0] = return_scaler.transform(valid_train_returns).reshape(-1)
    X_train_return, y_return_train = make_train_sequences(x_train_scaled, y_return_train_all, split)

    rows = []
    pred_paths = []
    # Naive is deterministic, but record once as seed=0 to avoid duplicating a static baseline across seeds.
    naive_row, naive_df = evaluate("Naive_Persistence", 0, test_dates, previous_open, open_test, previous_open,
                                   predicted_return=np.zeros_like(previous_open))
    rows.append(naive_row)
    naive_path = OUTPUT_DIR / "naive_predictions.csv"
    naive_df.to_csv(naive_path, index=False, encoding="utf-8-sig")
    pred_paths.append(str(naive_path))

    for seed in SEEDS:
        print(f"\n开始 seed={seed}")
        pred_price_scaled = train_predict(X_train, y_price_train, X_test, seed, "RAW-LSTM-Price")
        pred_price = price_scaler.inverse_transform(pred_price_scaled).reshape(-1)
        row, pred = evaluate("RAW_LSTM_Price", seed, test_dates, previous_open, open_test, pred_price)
        rows.append(row)
        p = OUTPUT_DIR / f"seed_{seed}_price_predictions.csv"
        pred.to_csv(p, index=False, encoding="utf-8-sig")
        pred_paths.append(str(p))

        pred_return_scaled = train_predict(X_train_return, y_return_train, X_test, seed, "RAW-LSTM-Return")
        pred_return = return_scaler.inverse_transform(pred_return_scaled).reshape(-1)
        pred_return_price = previous_open * (1.0 + pred_return)
        row, pred = evaluate("RAW_LSTM_Return", seed, test_dates, previous_open, open_test,
                             pred_return_price, predicted_return=pred_return)
        rows.append(row)
        p = OUTPUT_DIR / f"seed_{seed}_return_predictions.csv"
        pred.to_csv(p, index=False, encoding="utf-8-sig")
        pred_paths.append(str(p))

    all_results = pd.DataFrame(rows)
    all_path = OUTPUT_DIR / "all_results.csv"
    all_results.to_csv(all_path, index=False, encoding="utf-8-sig")
    summary = (all_results[all_results["Model"] != "Naive_Persistence"]
               .groupby("Model")
               .agg(
                   MAE_mean=("MAE", "mean"), MAE_std=("MAE", "std"),
                   RMSE_mean=("RMSE", "mean"), RMSE_std=("RMSE", "std"),
                   Strict_Direction_Accuracy_mean=("Strict_Direction_Accuracy", "mean"),
                   Strict_Direction_Accuracy_std=("Strict_Direction_Accuracy", "std"),
                   Balanced_Accuracy_mean=("Balanced_Accuracy", "mean"),
                   Balanced_Accuracy_std=("Balanced_Accuracy", "std"),
                   F1_Up_mean=("F1_Up", "mean"), F1_Up_std=("F1_Up", "std"),
                   Predicted_Up_Days_mean=("Predicted_Up_Days", "mean"),
                   Return_Correlation_mean=("Return_Correlation", "mean"),
               ).reset_index())
    # Put the deterministic naive baseline in its own summary row with zero standard deviations.
    na = all_results[all_results["Model"] == "Naive_Persistence"].iloc[0]
    naive_summary = pd.DataFrame([{
        "Model": "Naive_Persistence",
        "MAE_mean": na["MAE"], "MAE_std": 0.0,
        "RMSE_mean": na["RMSE"], "RMSE_std": 0.0,
        "Strict_Direction_Accuracy_mean": na["Strict_Direction_Accuracy"],
        "Strict_Direction_Accuracy_std": 0.0,
        "Balanced_Accuracy_mean": na["Balanced_Accuracy"], "Balanced_Accuracy_std": 0.0,
        "F1_Up_mean": na["F1_Up"], "F1_Up_std": 0.0,
        "Predicted_Up_Days_mean": na["Predicted_Up_Days"],
        "Return_Correlation_mean": na["Return_Correlation"],
    }])
    summary = pd.concat([naive_summary, summary], ignore_index=True)
    summary_path = OUTPUT_DIR / "summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    config = pd.DataFrame([{
        "data_file": locate_data().name, "rows": n,
        "train_rows": split, "test_rows": n-split,
        "train_start": dates[0], "train_end": dates[split-1],
        "test_start": dates[split], "test_end": dates[-1],
        "train_ratio": TRAIN_RATIO, "past_days": PAST_DAYS,
        "epochs": EPOCHS, "batch_size": BATCH_SIZE,
        "seeds": ",".join(map(str, SEEDS)),
        "target_price": "Open[t+1] absolute price",
        "target_return": "(Open[t+1]-Open[t])/Open[t]",
        "note": "Exploratory test period previously viewed during development.",
    }])
    config.to_csv(OUTPUT_DIR / "experiment_config.csv", index=False, encoding="utf-8-sig")

    print("\n===== 多随机种子汇总 =====")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\n所有逐种子结果：{all_path}")
    print(f"汇总：{summary_path}")
    print(f"配置：{OUTPUT_DIR / 'experiment_config.csv'}")


if __name__ == "__main__":
    main()
