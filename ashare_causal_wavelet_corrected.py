# -*- coding: utf-8 -*-

"""
A-share Stock Price Prediction
Causal Wavelet Denoising + LSTM

修正版说明：
- 小波去噪仅作用于模型输入。
- 预测目标始终为真实 original Open。
- 因此 Wavelet-LSTM 与 RAW-LSTM/Naive 的 MAE、RMSE 可以公平比较。

9.21阶段实验

实验：
1. Naive
   明天 Open = 今天 Open

2. RAW-LSTM
   原始 Open + LSTM

3. Causal DB4-L2 + LSTM
4. Causal DB4-L3 + LSTM
5. Causal DB4-L4 + LSTM

核心区别：
之前的小波实验是对整个区间进行一次性去噪，
本实验采用滚动窗口，只允许使用当前时刻及之前的数据。

预测方式：
过去90个交易日 -> 预测下一交易日 Open

固定：
TRAIN_RATIO = 0.8
PAST_DAYS = 90
EPOCHS = 100
BATCH_SIZE = 32
SEED = 42

输出：
results_causal_wavelet/
    final_results.csv
    naive_predictions.csv
    raw_lstm_predictions.csv
    causal_DB4_L2_predictions.csv
    causal_DB4_L3_predictions.csv
    causal_DB4_L4_predictions.csv

    denoised_series_DB4_L2.csv
    denoised_series_DB4_L3.csv
    denoised_series_DB4_L4.csv

    comparison_prediction.png
    causal_denoising_test.png
"""


# ============================================================
# 1. 环境
# ============================================================

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import random
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pywt
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam


# ============================================================
# 2. 参数
# ============================================================

DATA_FILE = "ashare_600519.csv"

OUTPUT_DIR = "results_causal_wavelet_corrected"

TRAIN_RATIO = 0.8

PAST_DAYS = 90

EPOCHS = 100

BATCH_SIZE = 32

SEED = 42

WAVELET = "db4"

# 本次严格因果小波实验只测试三个层数
WAVELET_LEVELS = [2, 3, 4]

# 滚动去噪窗口
# 128个交易日只允许使用历史数据
DENOISE_WINDOW = 128


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 3. 固定随机种子
# ============================================================

def set_seed(seed=42):

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)

    np.random.seed(seed)

    tf.random.set_seed(seed)

    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


set_seed(SEED)


# ============================================================
# 4. 读取数据
# ============================================================

def load_data():

    print("=" * 80)
    print("读取 A 股数据")
    print("=" * 80)

    if not os.path.exists(DATA_FILE):

        raise FileNotFoundError(
            f"找不到文件：{DATA_FILE}"
        )

    df = pd.read_csv(
        DATA_FILE
    )

    required = [
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            f"缺少字段：{missing}"
        )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    df["open"] = pd.to_numeric(
        df["open"],
        errors="coerce"
    )

    df = df.dropna(
        subset=[
            "date",
            "open"
        ]
    )

    df = (
        df.sort_values("date")
        .reset_index(drop=True)
    )

    print(
        f"数据量：{len(df)}"
    )

    print(
        f"时间范围："
        f"{df['date'].iloc[0].date()}"
        f" -> "
        f"{df['date'].iloc[-1].date()}"
    )

    return df


# ============================================================
# 5. 计算单个历史窗口的小波去噪
# ============================================================

def denoise_single_window(
    values,
    wavelet,
    level
):
    """
    对一个历史窗口进行小波去噪。

    重要：
    values 中只有当前时刻及其之前的数据。

    因此不会使用未来信息。
    """

    values = np.asarray(
        values,
        dtype=np.float64
    )

    wavelet_obj = pywt.Wavelet(
        wavelet
    )

    max_level = pywt.dwt_max_level(
        len(values),
        wavelet_obj.dec_len
    )

    if max_level < 1:
        return values[-1]

    actual_level = min(
        level,
        max_level
    )

    coeffs = pywt.wavedec(
        values,
        wavelet=wavelet,
        level=actual_level,
        mode="symmetric"
    )

    # --------------------------------------------------------
    # 根据当前历史窗口估计噪声
    # --------------------------------------------------------

    detail = coeffs[-1]

    sigma = (
        np.median(
            np.abs(detail)
        ) / 0.6745
    )

    # 防止sigma出现极端情况
    if not np.isfinite(sigma):

        return values[-1]

    threshold = (
        sigma
        * np.sqrt(
            2 * np.log(
                len(values)
            )
        )
    )

    # --------------------------------------------------------
    # 对 detail coefficients 做软阈值
    # --------------------------------------------------------

    denoised_coeffs = [
        coeffs[0]
    ]

    for detail_coeff in coeffs[1:]:

        denoised_detail = pywt.threshold(
            detail_coeff,
            threshold,
            mode="soft"
        )

        denoised_coeffs.append(
            denoised_detail
        )

    # --------------------------------------------------------
    # 重构
    # --------------------------------------------------------

    reconstructed = pywt.waverec(
        denoised_coeffs,
        wavelet=wavelet,
        mode="symmetric"
    )

    reconstructed = reconstructed[
        :len(values)
    ]

    # --------------------------------------------------------
    # 我们只取当前窗口的最后一个值
    #
    # 这意味着：
    # 当前时刻的去噪值只使用过去数据
    # --------------------------------------------------------

    result = float(
        reconstructed[-1]
    )

    if not np.isfinite(result):

        return float(values[-1])

    return result


# ============================================================
# 6. 构建严格因果的小波序列
# ============================================================

def causal_wavelet_denoise(
    values,
    wavelet,
    level,
    window
):
    """
    对整个价格序列进行“滚动因果小波去噪”。

    对每一个时间点 t：

        values[max(0,t-window+1) : t+1]

    只包含：
        当前时刻
        +
        当前时刻之前的数据

    不会使用 t 之后的数据。

    最终得到：
        denoised[t]

    在后续预测 t+1 时，只使用：
        denoised[t-89:t+1]
    """

    values = np.asarray(
        values,
        dtype=np.float64
    )

    n = len(values)

    denoised = np.zeros(
        n,
        dtype=np.float64
    )

    print()
    print(
        f"开始 Causal Wavelet："
        f"{wavelet} Level {level}"
    )

    print(
        f"窗口大小：{window}"
    )

    for t in range(n):

        start = max(
            0,
            t - window + 1
        )

        history = values[
            start:t + 1
        ]

        # 历史数据太少时直接保留原值
        if len(history) < 16:

            denoised[t] = values[t]

        else:

            denoised[t] = (
                denoise_single_window(
                    history,
                    wavelet,
                    level
                )
            )

        if (
            (t + 1) % 100 == 0
            or t == 0
            or t == n - 1
        ):

            print(
                f"Progress: "
                f"{t + 1}/{n}"
            )

    return denoised


# ============================================================
# 7. LSTM训练序列（输入与目标分离）
# ============================================================

def create_train_sequences(
    input_scaled,
    target_scaled,
    past_days
):
    """
    构造训练序列。

    input_scaled：模型输入序列。
        RAW-LSTM 时是原始 Open。
        Causal Wavelet-LSTM 时是因果去噪后的 Open。

    target_scaled：预测目标。
        始终对应真实的 original Open。

    关键：小波只处理输入，不处理预测目标。
    """

    X = []
    y = []

    for i in range(
        past_days,
        len(input_scaled)
    ):

        X.append(
            input_scaled[
                i - past_days:i
            ]
        )

        y.append(
            target_scaled[i]
        )

    return (
        np.array(X),
        np.array(y)
    )


# ============================================================
# 8. 测试序列
# ============================================================

def create_test_sequences(
    train_input_scaled,
    test_input_scaled,
    test_target_scaled,
    past_days
):
    """
    构造测试序列。

    X 使用过去90天的 input：
        RAW-LSTM：原始 Open
        Wavelet-LSTM：因果去噪 Open

    y 始终使用测试集当天的真实 original Open。
    """

    combined_input = np.concatenate(
        [
            train_input_scaled[-past_days:],
            test_input_scaled
        ]
    )

    X = []
    y = []

    for j in range(
        len(test_target_scaled)
    ):

        i = past_days + j

        X.append(
            combined_input[
                i - past_days:i
            ]
        )

        y.append(
            test_target_scaled[j]
        )

    return (
        np.array(X),
        np.array(y)
    )


# ============================================================
# 9. LSTM模型
# ============================================================

def build_model():

    model = Sequential()

    model.add(
        LSTM(
            50,
            return_sequences=True,
            input_shape=(
                PAST_DAYS,
                1
            )
        )
    )

    model.add(
        Dropout(0.2)
    )

    model.add(
        LSTM(
            50,
            return_sequences=True
        )
    )

    model.add(
        Dropout(0.2)
    )

    model.add(
        LSTM(
            50,
            return_sequences=True
        )
    )

    model.add(
        Dropout(0.2)
    )

    model.add(
        LSTM(
            50,
            return_sequences=False
        )
    )

    model.add(
        Dropout(0.2)
    )

    model.add(
        Dense(1)
    )

    model.compile(
        optimizer=Adam(),
        loss="mean_squared_error"
    )

    return model


# ============================================================
# 10. Directional Accuracy
# ============================================================

def calculate_directional_accuracy(
    y_true,
    y_pred
):

    true_diff = np.diff(
        y_true
    )

    pred_diff = np.diff(
        y_pred
    )

    return np.mean(
        np.sign(true_diff)
        == np.sign(pred_diff)
    )


# ============================================================
# 11. Naive Baseline
# ============================================================

def run_naive(
    train_values,
    test_values,
    test_dates
):

    print()
    print("=" * 80)
    print("Naive Baseline")
    print("=" * 80)

    """
    最简单的基线：

        预测明天 Open = 今天 Open

    第一个测试样本：
        使用训练集最后一天 Open

    后续测试样本：
        使用前一天真实 Open
    """

    y_true = test_values.copy()

    y_pred = np.zeros_like(
        y_true
    )

    # 第一个测试日
    y_pred[0] = train_values[-1]

    # 后续测试日
    for i in range(
        1,
        len(test_values)
    ):

        y_pred[i] = test_values[
            i - 1
        ]

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    da = calculate_directional_accuracy(
        y_true,
        y_pred
    )

    result = pd.DataFrame(
        {
            "date": test_dates,
            "actual_open": y_true,
            "predicted_open": y_pred
        }
    )

    result.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "naive_predictions.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )

    print(
        f"MAE = {mae:.4f}"
    )

    print(
        f"RMSE = {rmse:.4f}"
    )

    print(
        f"Directional Accuracy = "
        f"{da:.4f}"
        f" ({da * 100:.2f}%)"
    )

    return {
        "Model": "Naive",
        "MAE": mae,
        "RMSE": rmse,
        "Directional_Accuracy": da,
        "Test_Samples": len(y_true)
    }


# ============================================================
# 12. LSTM实验（小波只处理输入，目标为真实 Open）
# ============================================================

def run_lstm_experiment(
    train_input_values,
    train_target_values,
    test_input_values,
    test_target_values,
    test_dates,
    model_name
):

    print()
    print("=" * 80)
    print(
        f"LSTM Experiment: {model_name}"
    )
    print("=" * 80)

    # --------------------------------------------------------
    # scaler
    # --------------------------------------------------------

    # --------------------------------------------------------
    # 输入 scaler：只用训练集的输入数据 fit
    # --------------------------------------------------------

    input_scaler = MinMaxScaler(
        feature_range=(0, 1)
    )

    train_input_scaled = input_scaler.fit_transform(
        train_input_values.reshape(-1, 1)
    )

    test_input_scaled = input_scaler.transform(
        test_input_values.reshape(-1, 1)
    )

    # --------------------------------------------------------
    # 目标 scaler：只用训练集的真实 Open fit
    # --------------------------------------------------------

    target_scaler = MinMaxScaler(
        feature_range=(0, 1)
    )

    train_target_scaled = target_scaler.fit_transform(
        train_target_values.reshape(-1, 1)
    )

    test_target_scaled = target_scaler.transform(
        test_target_values.reshape(-1, 1)
    )

    # --------------------------------------------------------
    # train
    # --------------------------------------------------------

    X_train, y_train = (
        create_train_sequences(
            train_input_scaled,
            train_target_scaled,
            PAST_DAYS
        )
    )

    # --------------------------------------------------------
    # test
    # --------------------------------------------------------

    X_test, y_test = (
        create_test_sequences(
            train_input_scaled,
            test_input_scaled,
            test_target_scaled,
            PAST_DAYS
        )
    )

    print(
        f"X_train = {X_train.shape}"
    )

    print(
        f"X_test  = {X_test.shape}"
    )

    # --------------------------------------------------------
    # seed
    # --------------------------------------------------------

    set_seed(SEED)

    # --------------------------------------------------------
    # model
    # --------------------------------------------------------

    model = build_model()

    # --------------------------------------------------------
    # train
    # --------------------------------------------------------

    model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=False,
        verbose=0
    )

    print(
        "Training finished."
    )

    # --------------------------------------------------------
    # prediction
    # --------------------------------------------------------

    predicted_scaled = model.predict(
        X_test,
        verbose=0
    )

    # --------------------------------------------------------
    # inverse
    # --------------------------------------------------------

    y_pred = target_scaler.inverse_transform(
        predicted_scaled
    ).flatten()

    y_true = target_scaler.inverse_transform(
        y_test.reshape(-1, 1)
    ).flatten()

    # --------------------------------------------------------
    # metrics
    # --------------------------------------------------------

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    da = calculate_directional_accuracy(
        y_true,
        y_pred
    )

    # --------------------------------------------------------
    # save
    # --------------------------------------------------------

    prediction_df = pd.DataFrame(
        {
            "date": test_dates,
            "actual_open": y_true,
            "predicted_open": y_pred
        }
    )

    prediction_path = os.path.join(
        OUTPUT_DIR,
        f"{model_name}_predictions.csv"
    )

    prediction_df.to_csv(
        prediction_path,
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print(
        f"{model_name}"
    )

    print(
        f"MAE = {mae:.4f}"
    )

    print(
        f"RMSE = {rmse:.4f}"
    )

    print(
        f"Directional Accuracy = "
        f"{da:.4f}"
        f" ({da * 100:.2f}%)"
    )

    return {
        "Model": model_name,
        "Target": "original_open",
        "Input_Type": "raw_open" if model_name == "raw_lstm" else "causal_denoised_open",
        "MAE": mae,
        "RMSE": rmse,
        "Directional_Accuracy": da,
        "Test_Samples": len(y_true)
    }


# ============================================================
# 13. 去噪数据可视化
# ============================================================

def save_causal_denoising_plot(
    df,
    denoised_dict,
    split_index
):

    test_dates = df[
        "date"
    ].iloc[
        split_index:
    ]

    original_test = df[
        "open"
    ].iloc[
        split_index:
    ].values

    plt.figure(
        figsize=(14, 6)
    )

    plt.plot(
        test_dates,
        original_test,
        label="Original Open",
        alpha=0.5
    )

    for level, denoised in denoised_dict.items():

        plt.plot(
            test_dates,
            denoised[
                split_index:
            ],
            label=f"Causal db4-L{level}"
        )

    plt.xlabel(
        "Date"
    )

    plt.ylabel(
        "Open Price"
    )

    plt.title(
        "Causal Wavelet Denoising on Test Period"
    )

    plt.legend()

    plt.grid(True)

    plt.xticks(
        rotation=45
    )

    plt.tight_layout()

    path = os.path.join(
        OUTPUT_DIR,
        "causal_denoising_test.png"
    )

    plt.savefig(
        path,
        dpi=200
    )

    plt.close()

    print(
        f"去噪对比图已保存：{path}"
    )


# ============================================================
# 14. 预测结果对比
# ============================================================

def save_prediction_comparison(
    prediction_results
):

    plt.figure(
        figsize=(14, 6)
    )

    first = True

    actual = None
    dates = None

    for name, df_pred in prediction_results.items():

        if actual is None:

            actual = df_pred[
                "actual_open"
            ].values

            dates = pd.to_datetime(
                df_pred["date"]
            )

            plt.plot(
                dates,
                actual,
                label="Actual Open",
                linewidth=2
            )

        plt.plot(
            dates,
            df_pred[
                "predicted_open"
            ].values,
            label=name,
            alpha=0.8
        )

    plt.xlabel(
        "Date"
    )

    plt.ylabel(
        "Open Price"
    )

    plt.title(
        "Prediction Comparison"
    )

    plt.legend()

    plt.grid(True)

    plt.xticks(
        rotation=45
    )

    plt.tight_layout()

    path = os.path.join(
        OUTPUT_DIR,
        "comparison_prediction.png"
    )

    plt.savefig(
        path,
        dpi=200
    )

    plt.close()

    print(
        f"预测对比图已保存：{path}"
    )


# ============================================================
# 15. 主函数
# ============================================================

def main():

    print()
    print("=" * 100)
    print("Causal Wavelet + LSTM Experiment")
    print("=" * 100)

    # --------------------------------------------------------
    # 读取数据
    # --------------------------------------------------------

    df = load_data()

    # --------------------------------------------------------
    # 80/20
    # --------------------------------------------------------

    split_index = int(
        len(df) * TRAIN_RATIO
    )

    train_df = df.iloc[
        :split_index
    ].copy()

    test_df = df.iloc[
        split_index:
    ].copy()

    train_values = (
        train_df["open"]
        .values
        .astype(float)
    )

    test_values = (
        test_df["open"]
        .values
        .astype(float)
    )

    test_dates = (
        test_df["date"]
        .dt.strftime("%Y-%m-%d")
        .values
    )

    print()
    print(
        f"Train Samples = "
        f"{len(train_values)}"
    )

    print(
        f"Test Samples = "
        f"{len(test_values)}"
    )

    # ========================================================
    # A. Naive
    # ========================================================

    all_results = []

    naive_result = run_naive(
        train_values,
        test_values,
        test_dates
    )

    all_results.append(
        naive_result
    )

    # ========================================================
    # B. RAW LSTM
    # ========================================================

    raw_result = run_lstm_experiment(
        train_input_values=train_values,
        train_target_values=train_values,
        test_input_values=test_values,
        test_target_values=test_values,
        test_dates=test_dates,
        model_name="raw_lstm"
    )

    all_results.append(
        raw_result
    )

    # 保存预测结果
    prediction_results = {}

    naive_prediction = pd.read_csv(
        os.path.join(
            OUTPUT_DIR,
            "naive_predictions.csv"
        )
    )

    raw_prediction = pd.read_csv(
        os.path.join(
            OUTPUT_DIR,
            "raw_lstm_predictions.csv"
        )
    )

    prediction_results["Naive"] = (
        naive_prediction
    )

    prediction_results["RAW-LSTM"] = (
        raw_prediction
    )

    # ========================================================
    # C. Causal Wavelet
    # ========================================================

    denoised_dict = {}

    for level in WAVELET_LEVELS:

        print()
        print()
        print(
            "#" * 100
        )

        print(
            f"Causal db4 Level {level}"
        )

        print(
            "#" * 100
        )

        # ----------------------------------------------------
        # 因果小波去噪
        # ----------------------------------------------------

        denoised = causal_wavelet_denoise(
            values=df["open"].values,
            wavelet=WAVELET,
            level=level,
            window=DENOISE_WINDOW
        )

        denoised_dict[level] = denoised

        # ----------------------------------------------------
        # 保存去噪序列
        # ----------------------------------------------------

        denoised_df = pd.DataFrame(
            {
                "date": df["date"],
                "original_open": df["open"],
                f"causal_db4_l{level}": denoised
            }
        )

        denoised_path = os.path.join(
            OUTPUT_DIR,
            f"denoised_series_DB4_L{level}.csv"
        )

        denoised_df.to_csv(
            denoised_path,
            index=False,
            encoding="utf-8-sig"
        )

        # ----------------------------------------------------
        # 提取训练/测试
        # ----------------------------------------------------

        denoised_train = (
            denoised[
                :split_index
            ]
        )

        denoised_test = (
            denoised[
                split_index:
            ]
        )

        # ----------------------------------------------------
        # LSTM
        # ----------------------------------------------------

        model_name = (
            f"causal_DB4_L{level}"
        )

        # 关键修正：
        #   输入 = causal denoised Open
        #   目标 = original Open
        # 这样才能评价“小波输入预处理是否帮助预测真实股价”。
        result = run_lstm_experiment(
            train_input_values=denoised_train,
            train_target_values=train_values,
            test_input_values=denoised_test,
            test_target_values=test_values,
            test_dates=test_dates,
            model_name=model_name
        )

        all_results.append(
            result
        )

        # ----------------------------------------------------
        # 读取预测结果用于最终画图
        # ----------------------------------------------------

        prediction_path = os.path.join(
            OUTPUT_DIR,
            f"{model_name}_predictions.csv"
        )

        prediction_results[
            f"Causal DB4-L{level}"
        ] = pd.read_csv(
            prediction_path
        )

    # ========================================================
    # D. 去噪图
    # ========================================================

    save_causal_denoising_plot(
        df=df,
        denoised_dict=denoised_dict,
        split_index=split_index
    )

    # ========================================================
    # E. 预测对比图
    # ========================================================

    save_prediction_comparison(
        prediction_results
    )

    # ========================================================
    # F. 保存结果
    # ========================================================

    results_df = pd.DataFrame(
        all_results
    )

    results_path = os.path.join(
        OUTPUT_DIR,
        "final_results.csv"
    )

    results_df.to_csv(
        results_path,
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # G. 输出
    # ========================================================

    print()
    print()
    print("=" * 100)
    print("最终实验结果")
    print("=" * 100)

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    print()
    print(
        f"结果保存：{results_path}"
    )

    print()
    print("=" * 100)


if __name__ == "__main__":
    main()  