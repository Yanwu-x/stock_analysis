# -*- coding: utf-8 -*-
"""贵州茅台开盘价预测：因果小波辅助特征消融实验。

实验组：
1) RAW-LSTM: Open
2) OPEN_TREND_LSTM: Open + 因果小波 Trend
3) OPEN_DETAIL_LSTM: Open + 高频残差 Detail (Open - Trend)

统一规范：
- 按日期升序排列，前80%训练，后20%测试；
- 过去90个交易日预测下一交易日真实 Open；
- 小波特征用截至各时点的滚动历史窗口构造，不用未来数据；
- 输入与目标分开归一化，Scaler 仅在训练集拟合；
- 所有模型的训练标签、测试标签均为原始真实 Open；
- 输出独立保存到 results_wavelet_feature_experiment/，不覆盖旧结果。

运行：将本脚本放在 ashare_600519.csv 同一目录，在项目 Python 环境执行：
    python ashare_wavelet_feature_experiment.py
"""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pywt
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings("ignore")

# ------------------------- 参数 -------------------------
DATA_FILE = "ashare_600519.csv"
OUTPUT_DIR = "results_wavelet_feature_experiment"
TRAIN_RATIO = 0.8
PAST_DAYS = 90
EPOCHS = 100
BATCH_SIZE = 32
SEED = 42
WAVELET = "db4"
WAVELET_LEVEL = 3
DENOISE_WINDOW = 128

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
        raise FileNotFoundError(
            f"找不到 {DATA_FILE}。请将脚本放到该 CSV 所在目录，或修改 DATA_FILE。"
        )
    df = pd.read_csv(DATA_FILE)
    required = ["date", "open"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"数据缺少必要字段 {missing}；当前字段为：{list(df.columns)}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["open"] = pd.to_numeric(df["open"], errors="coerce")
    df = df.dropna(subset=["date", "open"]).sort_values("date").reset_index(drop=True)
    if len(df) <= PAST_DAYS + 10:
        raise ValueError(f"数据量过少：{len(df)} 行，无法构造 {PAST_DAYS} 日窗口。")
    print(f"数据文件：{DATA_FILE}")
    print(f"有效样本：{len(df)}")
    print(f"日期范围：{df.date.iloc[0].date()} 至 {df.date.iloc[-1].date()}")
    return df


def denoise_single_window(values, wavelet=WAVELET, level=WAVELET_LEVEL):
    """对当前及之前的一个窗口去噪，并仅返回该窗口的最后一个重构值。"""
    values = np.asarray(values, dtype=np.float64)
    wavelet_obj = pywt.Wavelet(wavelet)
    max_level = pywt.dwt_max_level(len(values), wavelet_obj.dec_len)
    if max_level < 1:
        return float(values[-1])
    actual_level = min(level, max_level)
    coeffs = pywt.wavedec(values, wavelet=wavelet, level=actual_level, mode="symmetric")
    finest_detail = coeffs[-1]
    sigma = np.median(np.abs(finest_detail)) / 0.6745
    if not np.isfinite(sigma):
        return float(values[-1])
    threshold = sigma * np.sqrt(2.0 * np.log(max(2, len(values))))
    denoised_coeffs = [coeffs[0]]
    for detail_coeff in coeffs[1:]:
        denoised_coeffs.append(pywt.threshold(detail_coeff, threshold, mode="soft"))
    reconstructed = pywt.waverec(denoised_coeffs, wavelet=wavelet, mode="symmetric")[:len(values)]
    result = float(reconstructed[-1])
    return result if np.isfinite(result) else float(values[-1])


def make_causal_trend(values):
    """Trend[t] 仅由 values[max(0,t-window+1):t+1] 计算。"""
    values = np.asarray(values, dtype=np.float64)
    trend = np.empty_like(values)
    n = len(values)
    for t in range(n):
        start = max(0, t - DENOISE_WINDOW + 1)
        history = values[start:t + 1]
        if len(history) < 16:
            trend[t] = values[t]
        else:
            trend[t] = denoise_single_window(history)
        if (t + 1) % 200 == 0 or t == n - 1:
            print(f"因果小波特征进度：{t + 1}/{n}")
    return trend


def build_model(n_features):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(PAST_DAYS, n_features)),
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


def create_train_sequences(input_scaled, target_scaled):
    X, y = [], []
    for i in range(PAST_DAYS, len(input_scaled)):
        X.append(input_scaled[i - PAST_DAYS:i, :])
        y.append(target_scaled[i, 0])
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)


def create_test_sequences(train_input_scaled, test_input_scaled, test_target_scaled):
    combined_input = np.concatenate([train_input_scaled[-PAST_DAYS:], test_input_scaled], axis=0)
    X, y = [], []
    for j in range(len(test_target_scaled)):
        i = PAST_DAYS + j
        X.append(combined_input[i - PAST_DAYS:i, :])
        y.append(test_target_scaled[j, 0])
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)


def compute_metrics(y_true, y_pred, previous_actual_open):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    # 与此前 corrected 脚本相同口径：比较预测序列与真实序列的相邻变化方向。
    if len(y_true) > 1:
        directional_accuracy = float(np.mean(
            np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred))
        ))
    else:
        directional_accuracy = float("nan")

    # 严格次日涨跌口径：均以“前一交易日真实 Open”为判断阈值。
    true_up = (y_true > previous_actual_open).astype(int)
    pred_up = (y_pred > previous_actual_open).astype(int)
    return {
        "MAE": float(mae),
        "RMSE": rmse,
        "Directional_Accuracy": directional_accuracy,
        "Strict_Direction_Accuracy": float(accuracy_score(true_up, pred_up)),
        "Strict_Precision": float(precision_score(true_up, pred_up, zero_division=0)),
        "Strict_Recall": float(recall_score(true_up, pred_up, zero_division=0)),
        "Strict_F1": float(f1_score(true_up, pred_up, zero_division=0)),
        "Strict_Predicted_Up_Days": int(pred_up.sum()),
        "Strict_Actual_Up_Days": int(true_up.sum()),
        "Test_Samples": int(len(y_true)),
    }


def run_one_experiment(name, feature_columns, feature_df, train_end, test_dates, train_open, test_open):
    print("\n" + "=" * 76)
    print(f"开始实验：{name}")
    print(f"输入特征：{feature_columns}")
    print("预测目标：原始真实 Open")

    input_values = feature_df[feature_columns].to_numpy(dtype=np.float64)
    train_input_values = input_values[:train_end]
    test_input_values = input_values[train_end:]

    # 输入 scaler 只 fit 训练区间。目标 scaler 同样只用训练区间真实 Open fit。
    input_scaler = MinMaxScaler(feature_range=(0, 1))
    train_input_scaled = input_scaler.fit_transform(train_input_values)
    test_input_scaled = input_scaler.transform(test_input_values)

    target_scaler = MinMaxScaler(feature_range=(0, 1))
    train_target_scaled = target_scaler.fit_transform(train_open.reshape(-1, 1))
    test_target_scaled = target_scaler.transform(test_open.reshape(-1, 1))

    X_train, y_train = create_train_sequences(train_input_scaled, train_target_scaled)
    X_test, y_test_scaled = create_test_sequences(
        train_input_scaled, test_input_scaled, test_target_scaled
    )
    if len(X_train) == 0 or len(X_test) != len(test_open):
        raise RuntimeError(f"序列构造异常：X_train={X_train.shape}, X_test={X_test.shape}")

    tf.keras.backend.clear_session()
    set_seed(SEED)
    model = build_model(n_features=len(feature_columns))
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=False, verbose=0)
    predicted_scaled = model.predict(X_test, verbose=0)

    y_pred = target_scaler.inverse_transform(predicted_scaled).reshape(-1)
    y_true = target_scaler.inverse_transform(y_test_scaled.reshape(-1, 1)).reshape(-1)
    previous_actual_open = np.concatenate(([train_open[-1]], test_open[:-1]))
    metrics = compute_metrics(y_true, y_pred, previous_actual_open)

    pred_df = pd.DataFrame({
        "date": test_dates,
        "previous_actual_open": previous_actual_open,
        "actual_open": y_true,
        "predicted_open": y_pred,
        "true_up": (y_true > previous_actual_open).astype(int),
        "predicted_up": (y_pred > previous_actual_open).astype(int),
    })
    pred_path = Path(OUTPUT_DIR) / f"{name}_predictions.csv"
    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")

    result = {
        "Model": name,
        "Features": "+".join(feature_columns),
        "Target": "original_open",
        **metrics,
    }
    print(f"MAE={metrics['MAE']:.4f} | RMSE={metrics['RMSE']:.4f} | "
          f"DA={metrics['Directional_Accuracy']:.2%} | "
          f"StrictAcc={metrics['Strict_Direction_Accuracy']:.2%} | "
          f"F1={metrics['Strict_F1']:.4f}")
    print(f"预测明细：{pred_path}")
    return result, pred_df


def main():
    set_seed(SEED)
    df = load_data()
    open_values = df["open"].to_numpy(dtype=np.float64)
    split_index = int(len(df) * TRAIN_RATIO)
    if split_index <= PAST_DAYS or len(df) - split_index < 2:
        raise ValueError("训练集或测试集太小，请检查数据量和 PAST_DAYS。")

    print(f"训练集：{split_index} 行 | 测试集：{len(df) - split_index} 行")
    print(f"参数：past_days={PAST_DAYS}, epochs={EPOCHS}, batch={BATCH_SIZE}, seed={SEED}")
    print(f"构造滚动因果小波特征：wavelet={WAVELET}, level={WAVELET_LEVEL}, window={DENOISE_WINDOW}")

    trend = make_causal_trend(open_values)
    detail = open_values - trend
    feature_df = pd.DataFrame({"Open": open_values, "Trend": trend, "Detail": detail})
    feature_df.insert(0, "date", df["date"].dt.strftime("%Y-%m-%d"))
    feature_path = Path(OUTPUT_DIR) / "causal_wavelet_features.csv"
    feature_df.to_csv(feature_path, index=False, encoding="utf-8-sig")
    print(f"特征序列已保存：{feature_path}")

    # 三组使用相同数据划分、预测目标和训练参数。
    feature_matrix = feature_df[["Open", "Trend", "Detail"]]
    train_open = open_values[:split_index]
    test_open = open_values[split_index:]
    test_dates = df["date"].iloc[split_index:].dt.strftime("%Y-%m-%d").to_numpy()

    experiments = [
        ("RAW_LSTM", ["Open"]),
        ("OPEN_TREND_LSTM", ["Open", "Trend"]),
        ("OPEN_DETAIL_LSTM", ["Open", "Detail"]),
    ]

    results = []
    predictions = {}
    for name, cols in experiments:
        result, pred_df = run_one_experiment(
            name=name,
            feature_columns=cols,
            feature_df=feature_matrix,
            train_end=split_index,
            test_dates=test_dates,
            train_open=train_open,
            test_open=test_open,
        )
        results.append(result)
        predictions[name] = pred_df

    results_df = pd.DataFrame(results)
    results_path = Path(OUTPUT_DIR) / "final_results.csv"
    results_df.to_csv(results_path, index=False, encoding="utf-8-sig")
    try:
        results_df.to_excel(Path(OUTPUT_DIR) / "final_results.xlsx", index=False)
    except Exception as exc:
        print(f"提示：Excel 汇总未保存（不影响 CSV）：{exc}")

    # 预测曲线比较图
    plt.figure(figsize=(14, 6))
    first = predictions["RAW_LSTM"]
    plt.plot(first["date"], first["actual_open"], label="Actual Open", linewidth=2)
    for name in ["RAW_LSTM", "OPEN_TREND_LSTM", "OPEN_DETAIL_LSTM"]:
        plt.plot(predictions[name]["date"], predictions[name]["predicted_open"], label=name, alpha=0.85)
    plt.xlabel("Date")
    plt.ylabel("Open Price")
    plt.title("A-share Open Prediction: Wavelet Feature Ablation")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plot_path = Path(OUTPUT_DIR) / "prediction_comparison.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()

    print("\n" + "=" * 76)
    print("实验结束，结果汇总：")
    show_cols = ["Model", "Features", "MAE", "RMSE", "Directional_Accuracy",
                 "Strict_Direction_Accuracy", "Strict_Precision", "Strict_Recall", "Strict_F1"]
    print(results_df[show_cols].to_string(index=False))
    print(f"\n结果 CSV：{results_path.resolve()}")
    print(f"预测对比图：{plot_path.resolve()}")
    print("注意：本脚本中的 Directional_Accuracy 延续此前相邻变化方向口径；Strict_* 指标是以此前一交易日真实 Open 为阈值的次日涨跌口径。")


if __name__ == "__main__":
    main()
