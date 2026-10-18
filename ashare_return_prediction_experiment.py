# -*- coding: utf-8 -*-
"""贵州茅台次日开盘价预测：绝对价格目标 vs. 次日收益率目标。

目的：诊断当前 LSTM 预测价系统性偏低、很少预测上涨的问题。

对比：
1) Naive: 次日 Open = 前一交易日真实 Open
2) RAW-LSTM-Price: 使用过去90天 Open 预测下一日绝对 Open
3) RAW-LSTM-Return: 使用相同输入窗口/网络结构预测下一日收益率，再还原为预测 Open

统一设置：按时间排序，前80%训练、后20%测试；过去90个交易日；4层LSTM，每层50单元、Dropout=0.2；100 epochs；batch=32；seed=42。
Scaler只在训练数据上拟合。测试时预测每一天都只使用此前已知的Open值。

用法：将本脚本与 ashare_600519.csv 放在同一目录，然后运行：
    python ashare_return_prediction_experiment.py

注意：收益率预测只是待验证的实验方案，不保证方向准确率一定提高。
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings("ignore")

DATA_FILE = "ashare_600519.csv"
OUTPUT_DIR = "results_return_prediction"
TRAIN_RATIO = 0.80
PAST_DAYS = 90
EPOCHS = 100
BATCH_SIZE = 32
SEED = 42

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


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
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"找不到 {DATA_FILE}。请将脚本放在该CSV同一目录，或修改 DATA_FILE。")
    df = pd.read_csv(DATA_FILE)
    if not {"date", "open"}.issubset(df.columns):
        raise ValueError(f"CSV必须至少包含 date、open 两列；实际列为：{list(df.columns)}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["open"] = pd.to_numeric(df["open"], errors="coerce")
    df = df.dropna(subset=["date", "open"]).sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    if len(df) <= PAST_DAYS + 10:
        raise ValueError(f"有效数据太少：{len(df)} 行")
    return df


def build_model():
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(PAST_DAYS, 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(1),
    ])
    model.compile(optimizer=Adam(), loss="mean_squared_error")
    return model


def build_train_sequences(x_scaled, y_scaled, end_index):
    X, y = [], []
    for i in range(PAST_DAYS, end_index):
        X.append(x_scaled[i - PAST_DAYS:i, :])
        y.append(y_scaled[i, 0])
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)


def build_test_sequences(train_x_scaled, test_x_scaled):
    combined = np.concatenate([train_x_scaled[-PAST_DAYS:], test_x_scaled], axis=0)
    X = []
    for j in range(len(test_x_scaled)):
        X.append(combined[j:j + PAST_DAYS, :])
    return np.asarray(X, dtype=np.float32)


def train_and_predict(X_train, y_train, X_test, label):
    tf.keras.backend.clear_session()
    set_seed(SEED)
    model = build_model()
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=False, verbose=0)
    preds = model.predict(X_test, verbose=0).reshape(-1, 1)
    print(f"{label} 完成：训练样本 {len(X_train)}，测试样本 {len(X_test)}")
    return preds


def evaluate(name, dates, previous_open, actual_open, predicted_open, predicted_return=None):
    previous_open = np.asarray(previous_open, dtype=float)
    actual_open = np.asarray(actual_open, dtype=float)
    predicted_open = np.asarray(predicted_open, dtype=float)
    true_return = (actual_open - previous_open) / previous_open
    if predicted_return is None:
        predicted_return = (predicted_open - previous_open) / previous_open
    else:
        predicted_return = np.asarray(predicted_return, dtype=float)
    true_up = (actual_open > previous_open).astype(int)
    pred_up = (predicted_open > previous_open).astype(int)
    tp = int(((true_up == 1) & (pred_up == 1)).sum())
    fp = int(((true_up == 0) & (pred_up == 1)).sum())
    fn = int(((true_up == 1) & (pred_up == 0)).sum())
    tn = int(((true_up == 0) & (pred_up == 0)).sum())
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    try:
        corr_returns = float(np.corrcoef(true_return, predicted_return)[0, 1])
        if not np.isfinite(corr_returns):
            corr_returns = 0.0
    except Exception:
        corr_returns = 0.0

    row = {
        "Model": name,
        "MAE": float(mean_absolute_error(actual_open, predicted_open)),
        "RMSE": float(np.sqrt(mean_squared_error(actual_open, predicted_open))),
        "Direction_Accuracy": float(np.mean(true_up == pred_up)),
        "Balanced_Accuracy": float((recall + specificity) / 2),
        "Precision_Up": float(precision_score(true_up, pred_up, zero_division=0)),
        "Recall_Up": float(recall_score(true_up, pred_up, zero_division=0)),
        "F1_Up": float(f1_score(true_up, pred_up, zero_division=0)),
        "Actual_Up_Days": int(true_up.sum()),
        "Predicted_Up_Days": int(pred_up.sum()),
        "Mean_Prediction_Bias": float(np.mean(predicted_open - actual_open)),
        "Predicted_Below_Previous_Pct": float(np.mean(predicted_open < previous_open)),
        "Mean_True_Return": float(np.mean(true_return)),
        "Mean_Predicted_Return": float(np.mean(predicted_return)),
        "Return_Correlation": corr_returns,
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "Test_Samples": len(actual_open),
    }
    pred_df = pd.DataFrame({
        "date": dates,
        "previous_actual_open": previous_open,
        "actual_open": actual_open,
        "predicted_open": predicted_open,
        "actual_return": true_return,
        "predicted_return": predicted_return,
        "true_up": true_up,
        "predicted_up": pred_up,
    })
    return row, pred_df


def main():
    set_seed(SEED)
    df = load_data()
    dates = df["date"].dt.strftime("%Y-%m-%d").to_numpy()
    opens = df["open"].to_numpy(dtype=np.float64)
    n = len(df)
    split = int(n * TRAIN_RATIO)
    if split <= PAST_DAYS or n - split < 2:
        raise ValueError("训练集或测试集过小，请检查 TRAIN_RATIO / PAST_DAYS")

    open_train = opens[:split]
    open_test = opens[split:]
    test_dates = dates[split:]
    previous_open = np.concatenate(([opens[split - 1]], open_test[:-1]))

    # 输入：只使用过去的Open值；输入Scaler只使用训练区间fit。
    input_scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_all = input_scaler.fit_transform(open_train.reshape(-1, 1))
    X_test_all = input_scaler.transform(open_test.reshape(-1, 1))
    X_test = build_test_sequences(X_train_all, X_test_all)

    print(f"数据：{len(df)} 行，日期 {dates[0]} 至 {dates[-1]}")
    print(f"训练集 {split} 行 / 测试集 {n - split} 行；窗口 {PAST_DAYS}；seed={SEED}")

    rows, pred_dfs = [], {}

    # 1) Naive：预测价等于前一交易日真实Open。
    naive_pred = previous_open.copy()
    row, pred_df = evaluate("Naive_Persistence", test_dates, previous_open, open_test, naive_pred,
                            predicted_return=np.zeros_like(naive_pred))
    rows.append(row)
    pred_dfs[row["Model"]] = pred_df
    pred_df.to_csv(Path(OUTPUT_DIR) / "naive_predictions.csv", index=False, encoding="utf-8-sig")

    # 2) 对照组：直接预测绝对Open价。
    price_scaler = MinMaxScaler(feature_range=(0, 1))
    y_price_train_all = price_scaler.fit_transform(open_train.reshape(-1, 1))
    X_price_train, y_price_train = build_train_sequences(X_train_all, y_price_train_all, split)
    pred_price_scaled = train_and_predict(X_price_train, y_price_train, X_test, "RAW-LSTM-Price")
    pred_price = price_scaler.inverse_transform(pred_price_scaled).reshape(-1)
    row, pred_df = evaluate("RAW_LSTM_Price", test_dates, previous_open, open_test, pred_price)
    rows.append(row)
    pred_dfs[row["Model"]] = pred_df
    pred_df.to_csv(Path(OUTPUT_DIR) / "raw_lstm_price_predictions.csv", index=False, encoding="utf-8-sig")

    # 3) 新实验：预测下一交易日收益率，而不是绝对价格。
    returns = np.zeros_like(opens, dtype=np.float64)
    returns[1:] = (opens[1:] - opens[:-1]) / opens[:-1]
    ret_scaler = StandardScaler()
    # 只用训练期中实际作为训练标签的收益率拟合 scaler。
    train_return_targets = returns[PAST_DAYS:split].reshape(-1, 1)
    ret_scaler.fit(train_return_targets)
    y_return_train_all = ret_scaler.transform(returns[:split].reshape(-1, 1))
    X_return_train, y_return_train = build_train_sequences(X_train_all, y_return_train_all, split)
    pred_return_scaled = train_and_predict(X_return_train, y_return_train, X_test, "RAW-LSTM-Return")
    pred_return = ret_scaler.inverse_transform(pred_return_scaled).reshape(-1)
    # 将预测收益率转换为预测Open，便于与Naive和价格回归模型对照。
    pred_return_price = previous_open * (1.0 + pred_return)
    row, pred_df = evaluate("RAW_LSTM_Return", test_dates, previous_open, open_test,
                            pred_return_price, predicted_return=pred_return)
    rows.append(row)
    pred_dfs[row["Model"]] = pred_df
    pred_df.to_csv(Path(OUTPUT_DIR) / "raw_lstm_return_predictions.csv", index=False, encoding="utf-8-sig")

    results = pd.DataFrame(rows)
    results_path = Path(OUTPUT_DIR) / "final_results.csv"
    results.to_csv(results_path, index=False, encoding="utf-8-sig")
    try:
        results.to_excel(Path(OUTPUT_DIR) / "final_results.xlsx", index=False)
    except Exception as exc:
        print(f"提示：Excel 保存失败（CSV已保存）：{exc}")

    plt.figure(figsize=(14, 6))
    plt.plot(test_dates, open_test, label="Actual Open", linewidth=2)
    for name in ["Naive_Persistence", "RAW_LSTM_Price", "RAW_LSTM_Return"]:
        plt.plot(pred_dfs[name]["date"], pred_dfs[name]["predicted_open"], label=name, alpha=0.85)
    plt.xlabel("Date")
    plt.ylabel("Open Price")
    plt.title("Next-day Open Forecast: Price Target vs Return Target")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plot_path = Path(OUTPUT_DIR) / "prediction_comparison.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()

    print("\n===== 结果汇总 =====")
    show_cols = ["Model", "MAE", "RMSE", "Direction_Accuracy", "Balanced_Accuracy",
                 "Precision_Up", "Recall_Up", "F1_Up", "Predicted_Up_Days",
                 "Mean_Prediction_Bias", "Return_Correlation"]
    print(results[show_cols].to_string(index=False))
    print(f"\n结果：{results_path.resolve()}")
    print(f"预测曲线：{plot_path.resolve()}")
    print("注意：模型选择与阈值不可根据测试集结果反复调节；该实验用于检验改用收益率目标后表现是否改善。")


if __name__ == "__main__":
    main()
