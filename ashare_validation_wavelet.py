# -*- coding: utf-8 -*-

"""
A-share Stock Price Prediction
Validation-based Causal Wavelet + LSTM

实验目的：
使用 Train / Validation / Test 三段时间序列划分，
通过 Validation 集选择最合适的小波分解层数和去噪窗口，
然后在 Train + Validation 上重新训练，
最后在 Test 集进行一次最终评估。

数据：
    ashare_600519.csv

数据划分：
    70% Train
    10% Validation
    20% Test

LSTM：
    PAST_DAYS = 90
    EPOCHS = 100
    BATCH_SIZE = 32
    SEED = 42
    4层LSTM
    每层50 units
    Dropout = 0.2

Causal Wavelet：
    Wavelet = db4
    Level = 2 / 3 / 4
    Window = 64 / 128 / 256

选择标准：
    Validation MAE 最小

最终输出：
    results_validation/
        validation_results.csv
        best_config.csv
        final_test_results.csv
        validation_mae_heatmap.png
        final_prediction_comparison.png
        validation_predictions_*.csv
        test_predictions_*.csv
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

from tensorflow.keras.layers import (
    LSTM,
    Dropout,
    Dense
)

from tensorflow.keras.optimizers import Adam


# ============================================================
# 2. 参数
# ============================================================

DATA_FILE = "ashare_600519.csv"

OUTPUT_DIR = "results_validation"

TRAIN_RATIO = 0.70

VAL_RATIO = 0.10

TEST_RATIO = 0.20

PAST_DAYS = 90

EPOCHS = 100

BATCH_SIZE = 32

SEED = 42

WAVELET = "db4"

WAVELET_LEVELS = [2, 3, 4]

DENOISE_WINDOWS = [64, 128, 256]


# ============================================================
# 3. 创建输出目录
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 4. 固定随机种子
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


# ============================================================
# 5. 读取数据
# ============================================================

def load_data():

    print("=" * 80)
    print("读取 A 股数据")
    print("=" * 80)

    if not os.path.exists(DATA_FILE):

        raise FileNotFoundError(
            f"找不到数据文件：{DATA_FILE}\n"
            f"请检查 CSV 与 Python 文件是否位于同一目录。"
        )

    df = pd.read_csv(
        DATA_FILE
    )

    required_columns = [
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]

    missing = [
        col
        for col in required_columns
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
        f"日期范围："
        f"{df['date'].iloc[0].date()}"
        f" -> "
        f"{df['date'].iloc[-1].date()}"
    )

    return df


# ============================================================
# 6. 数据划分
# ============================================================

def split_data(df):

    n = len(df)

    train_end = int(
        n * TRAIN_RATIO
    )

    val_end = int(
        n * (TRAIN_RATIO + VAL_RATIO)
    )

    train_df = df.iloc[
        :train_end
    ].copy()

    val_df = df.iloc[
        train_end:val_end
    ].copy()

    test_df = df.iloc[
        val_end:
    ].copy()

    print()
    print("=" * 80)
    print("Train / Validation / Test 划分")
    print("=" * 80)

    print(
        f"Train      : {len(train_df)}"
    )

    print(
        f"Validation : {len(val_df)}"
    )

    print(
        f"Test       : {len(test_df)}"
    )

    print()

    print(
        f"Train："
        f"{train_df['date'].iloc[0].date()}"
        f" -> "
        f"{train_df['date'].iloc[-1].date()}"
    )

    print(
        f"Validation："
        f"{val_df['date'].iloc[0].date()}"
        f" -> "
        f"{val_df['date'].iloc[-1].date()}"
    )

    print(
        f"Test："
        f"{test_df['date'].iloc[0].date()}"
        f" -> "
        f"{test_df['date'].iloc[-1].date()}"
    )

    return (
        train_df,
        val_df,
        test_df
    )


# ============================================================
# 7. 单窗口因果小波去噪
# ============================================================

def denoise_single_window(
    values,
    wavelet,
    level
):

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

        return float(
            values[-1]
        )

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

    detail = coeffs[-1]

    sigma = (
        np.median(
            np.abs(detail)
        ) / 0.6745
    )

    if not np.isfinite(
        sigma
    ):

        return float(
            values[-1]
        )

    threshold = (
        sigma
        * np.sqrt(
            2 * np.log(
                len(values)
            )
        )
    )

    new_coeffs = [
        coeffs[0]
    ]

    for detail_coeff in coeffs[1:]:

        thresholded = pywt.threshold(
            detail_coeff,
            threshold,
            mode="soft"
        )

        new_coeffs.append(
            thresholded
        )

    reconstructed = pywt.waverec(
        new_coeffs,
        wavelet=wavelet,
        mode="symmetric"
    )

    reconstructed = reconstructed[
        :len(values)
    ]

    result = float(
        reconstructed[-1]
    )

    if not np.isfinite(
        result
    ):

        return float(
            values[-1]
        )

    return result


# ============================================================
# 8. Causal Wavelet
# ============================================================

def causal_wavelet_denoise(
    values,
    wavelet,
    level,
    window
):

    values = np.asarray(
        values,
        dtype=np.float64
    )

    n = len(values)

    denoised = np.zeros(
        n,
        dtype=np.float64
    )

    print(
        f"开始：{wavelet}-L{level}, "
        f"Window={window}"
    )

    for t in range(n):

        start = max(
            0,
            t - window + 1
        )

        historical_values = values[
            start:t + 1
        ]

        if len(historical_values) < 16:

            denoised[t] = values[t]

        else:

            denoised[t] = (
                denoise_single_window(
                    historical_values,
                    wavelet,
                    level
                )
            )

        if (
            (t + 1) % 200 == 0
            or t == n - 1
        ):

            print(
                f"  Progress: "
                f"{t + 1}/{n}"
            )

    return denoised


# ============================================================
# 9. 创建 LSTM 模型
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
# 10. 创建训练序列
# ============================================================

def create_sequences(
    scaled_values,
    past_days
):

    X = []
    y = []

    for i in range(
        past_days,
        len(scaled_values)
    ):

        X.append(
            scaled_values[
                i - past_days:i
            ]
        )

        y.append(
            scaled_values[i]
        )

    return (
        np.array(X),
        np.array(y)
    )


# ============================================================
# 11. 创建验证 / 测试序列
# ============================================================

def create_eval_sequences(
    history_scaled,
    eval_scaled,
    past_days
):

    combined = np.concatenate(
        [
            history_scaled[-past_days:],
            eval_scaled
        ]
    )

    X = []
    y = []

    for i in range(
        past_days,
        len(combined)
    ):

        X.append(
            combined[
                i - past_days:i
            ]
        )

        y.append(
            combined[i]
        )

    return (
        np.array(X),
        np.array(y)
    )


# ============================================================
# 12. Directional Accuracy
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
# 13. 评价指标
# ============================================================

def evaluate_predictions(
    y_true,
    y_pred
):

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

    return (
        mae,
        rmse,
        da
    )


# ============================================================
# 14. 运行 LSTM
# ============================================================

def train_and_predict(
    train_values,
    eval_values,
    fit_reference_values,
    seed
):

    """
    train_values：
        用于真正 fit 模型的数据

    eval_values：
        当前要预测的数据

    fit_reference_values：
        scaler fit 使用的数据

    这里单独传出来，是为了保证：
        scaler 永远只在训练数据上 fit
    """

    set_seed(seed)

    # --------------------------------------------------------
    # scaler
    # --------------------------------------------------------

    scaler = MinMaxScaler(
        feature_range=(0, 1)
    )

    train_scaled = scaler.fit_transform(
        fit_reference_values.reshape(-1, 1)
    )

    # --------------------------------------------------------
    # 当 train_values 与 fit_reference_values 相同
    # --------------------------------------------------------

    if (
        len(train_values)
        == len(fit_reference_values)
        and np.allclose(
            train_values,
            fit_reference_values
        )
    ):

        actual_train_scaled = (
            train_scaled
        )

    else:

        actual_train_scaled = (
            scaler.transform(
                train_values.reshape(-1, 1)
            )
        )

    eval_scaled = (
        scaler.transform(
            eval_values.reshape(-1, 1)
        )
    )

    # --------------------------------------------------------
    # train sequences
    # --------------------------------------------------------

    X_train, y_train = (
        create_sequences(
            actual_train_scaled,
            PAST_DAYS
        )
    )

    # --------------------------------------------------------
    # eval sequences
    # --------------------------------------------------------

    X_eval, y_eval = (
        create_eval_sequences(
            actual_train_scaled,
            eval_scaled,
            PAST_DAYS
        )
    )

    # --------------------------------------------------------
    # model
    # --------------------------------------------------------

    model = build_model()

    model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=False,
        verbose=0
    )

    # --------------------------------------------------------
    # prediction
    # --------------------------------------------------------

    pred_scaled = model.predict(
        X_eval,
        verbose=0
    )

    y_pred = scaler.inverse_transform(
        pred_scaled
    ).flatten()

    y_true = scaler.inverse_transform(
        y_eval.reshape(-1, 1)
    ).flatten()

    return (
        y_true,
        y_pred
    )


# ============================================================
# 15. Validation 阶段
# ============================================================

def run_validation_experiments(
    raw_df,
    denoised_cache
):

    print()
    print("=" * 100)
    print("Validation 参数搜索")
    print("=" * 100)

    n = len(raw_df)

    train_end = int(
        n * TRAIN_RATIO
    )

    val_end = int(
        n * (TRAIN_RATIO + VAL_RATIO)
    )

    raw_train = (
        raw_df["open"]
        .iloc[:train_end]
        .values
        .astype(float)
    )

    raw_val = (
        raw_df["open"]
        .iloc[train_end:val_end]
        .values
        .astype(float)
    )

    results = []

    # --------------------------------------------------------
    # RAW baseline
    # --------------------------------------------------------

    print()
    print(
        "Validation: RAW-LSTM"
    )

    y_true, y_pred = train_and_predict(
        train_values=raw_train,
        eval_values=raw_val,
        fit_reference_values=raw_train,
        seed=SEED
    )

    mae, rmse, da = evaluate_predictions(
        y_true,
        y_pred
    )

    raw_val_dates = (
        raw_df["date"]
        .iloc[
            train_end:val_end
        ]
        .dt.strftime("%Y-%m-%d")
        .values
    )

    pd.DataFrame(
        {
            "date": raw_val_dates,
            "actual_open": y_true,
            "predicted_open": y_pred
        }
    ).to_csv(
        os.path.join(
            OUTPUT_DIR,
            "validation_predictions_RAW.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )

    results.append(
        {
            "Model": "RAW-LSTM",
            "Wavelet": "None",
            "Level": 0,
            "Window": 0,
            "MAE": mae,
            "RMSE": rmse,
            "Directional_Accuracy": da,
            "Samples": len(y_true)
        }
    )

    # --------------------------------------------------------
    # Wavelet 参数
    # --------------------------------------------------------

    for level in WAVELET_LEVELS:

        for window in DENOISE_WINDOWS:

            key = (
                level,
                window
            )

            denoised = denoised_cache[
                key
            ]

            train_values = (
                denoised[
                    :train_end
                ]
            )

            val_values = (
                denoised[
                    train_end:val_end
                ]
            )

            print()
            print(
                f"Validation: "
                f"DB4-L{level}, "
                f"Window={window}"
            )

            y_true, y_pred = (
                train_and_predict(
                    train_values=train_values,
                    eval_values=val_values,
                    fit_reference_values=train_values,
                    seed=SEED
                )
            )

            mae, rmse, da = (
                evaluate_predictions(
                    y_true,
                    y_pred
                )
            )

            model_name = (
                f"DB4_L{level}_W{window}"
            )

            val_dates = (
                raw_df["date"]
                .iloc[
                    train_end:val_end
                ]
                .dt.strftime(
                    "%Y-%m-%d"
                )
                .values
            )

            pd.DataFrame(
                {
                    "date": val_dates,
                    "actual_open": y_true,
                    "predicted_open": y_pred
                }
            ).to_csv(
                os.path.join(
                    OUTPUT_DIR,
                    f"validation_predictions_{model_name}.csv"
                ),
                index=False,
                encoding="utf-8-sig"
            )

            results.append(
                {
                    "Model": "Causal Wavelet-LSTM",
                    "Wavelet": WAVELET,
                    "Level": level,
                    "Window": window,
                    "MAE": mae,
                    "RMSE": rmse,
                    "Directional_Accuracy": da,
                    "Samples": len(y_true)
                }
            )

    results_df = pd.DataFrame(
        results
    )

    path = os.path.join(
        OUTPUT_DIR,
        "validation_results.csv"
    )

    results_df.to_csv(
        path,
        index=False,
        encoding="utf-8-sig"
    )

    return results_df


# ============================================================
# 16. 选择最优配置
# ============================================================

def select_best_config(
    validation_results
):

    print()
    print("=" * 100)
    print("Validation 结果")
    print("=" * 100)

    print(
        validation_results.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    # --------------------------------------------------------
    # 只从 Wavelet 模型中选择
    #
    # RAW 只是 baseline，不参与小波参数选择
    # --------------------------------------------------------

    wavelet_results = (
        validation_results[
            validation_results["Model"]
            == "Causal Wavelet-LSTM"
        ]
        .copy()
    )

    best = (
        wavelet_results
        .sort_values("MAE")
        .iloc[0]
    )

    best_config = pd.DataFrame(
        [
            {
                "Wavelet": best["Wavelet"],
                "Level": int(
                    best["Level"]
                ),
                "Window": int(
                    best["Window"]
                ),
                "Validation_MAE": best["MAE"],
                "Validation_RMSE": best["RMSE"],
                "Validation_DA": best[
                    "Directional_Accuracy"
                ]
            }
        ]
    )

    path = os.path.join(
        OUTPUT_DIR,
        "best_config.csv"
    )

    best_config.to_csv(
        path,
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print(
        "根据 Validation MAE 选择的小波配置："
    )

    print(
        f"Wavelet = {best['Wavelet']}"
    )

    print(
        f"Level   = {int(best['Level'])}"
    )

    print(
        f"Window  = {int(best['Window'])}"
    )

    print(
        f"Validation MAE = "
        f"{best['MAE']:.4f}"
    )

    return (
        int(best["Level"]),
        int(best["Window"])
    )


# ============================================================
# 17. 最终 Test
# ============================================================

def run_final_test(
    raw_df,
    denoised_cache,
    best_level,
    best_window
):

    print()
    print("=" * 100)
    print("最终 Test 实验")
    print("=" * 100)

    n = len(raw_df)

    train_end = int(
        n * TRAIN_RATIO
    )

    val_end = int(
        n * (TRAIN_RATIO + VAL_RATIO)
    )

    # --------------------------------------------------------
    # Train + Validation
    # --------------------------------------------------------

    raw_trainval = (
        raw_df["open"]
        .iloc[:val_end]
        .values
        .astype(float)
    )

    test_values = (
        raw_df["open"]
        .iloc[val_end:]
        .values
        .astype(float)
    )

    test_dates = (
        raw_df["date"]
        .iloc[val_end:]
        .dt.strftime(
            "%Y-%m-%d"
        )
        .values
    )

    results = []

    prediction_dict = {}

    # ========================================================
    # 1. Naive
    # ========================================================

    y_true_naive = test_values.copy()

    y_pred_naive = np.zeros_like(
        y_true_naive
    )

    # 第一个测试日：
    # 上一个交易日是 Validation 最后一天
    y_pred_naive[0] = (
        raw_trainval[-1]
    )

    for i in range(
        1,
        len(test_values)
    ):

        y_pred_naive[i] = (
            test_values[i - 1]
        )

    mae, rmse, da = (
        evaluate_predictions(
            y_true_naive,
            y_pred_naive
        )
    )

    results.append(
        {
            "Model": "Naive",
            "Level": 0,
            "Window": 0,
            "MAE": mae,
            "RMSE": rmse,
            "Directional_Accuracy": da,
            "Test_Samples": len(
                y_true_naive
            )
        }
    )

    prediction_dict["Naive"] = (
        y_pred_naive
    )

    pd.DataFrame(
        {
            "date": test_dates,
            "actual_open": y_true_naive,
            "predicted_open": y_pred_naive
        }
    ).to_csv(
        os.path.join(
            OUTPUT_DIR,
            "test_predictions_Naive.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # 2. RAW-LSTM
    # ========================================================

    print()
    print(
        "最终 Test：RAW-LSTM"
    )

    y_true_raw, y_pred_raw = (
        train_and_predict(
            train_values=raw_trainval,
            eval_values=test_values,
            fit_reference_values=raw_trainval,
            seed=SEED
        )
    )

    mae, rmse, da = (
        evaluate_predictions(
            y_true_raw,
            y_pred_raw
        )
    )

    results.append(
        {
            "Model": "RAW-LSTM",
            "Level": 0,
            "Window": 0,
            "MAE": mae,
            "RMSE": rmse,
            "Directional_Accuracy": da,
            "Test_Samples": len(
                y_true_raw
            )
        }
    )

    prediction_dict["RAW-LSTM"] = (
        y_pred_raw
    )

    pd.DataFrame(
        {
            "date": test_dates,
            "actual_open": y_true_raw,
            "predicted_open": y_pred_raw
        }
    ).to_csv(
        os.path.join(
            OUTPUT_DIR,
            "test_predictions_RAW-LSTM.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # 3. 最优 Causal Wavelet-LSTM
    # ========================================================

    print()
    print(
        f"最终 Test："
        f"DB4-L{best_level}, "
        f"Window={best_window}"
    )

    best_denoised = (
        denoised_cache[
            (
                best_level,
                best_window
            )
        ]
    )

    trainval_denoised = (
        best_denoised[
            :val_end
        ]
    )

    test_denoised = (
        best_denoised[
            val_end:
        ]
    )

    y_true_wavelet, y_pred_wavelet = (
        train_and_predict(
            train_values=trainval_denoised,
            eval_values=test_denoised,
            fit_reference_values=trainval_denoised,
            seed=SEED
        )
    )

    mae, rmse, da = (
        evaluate_predictions(
            y_true_wavelet,
            y_pred_wavelet
        )
    )

    results.append(
        {
            "Model": "Causal Wavelet-LSTM",
            "Level": best_level,
            "Window": best_window,
            "MAE": mae,
            "RMSE": rmse,
            "Directional_Accuracy": da,
            "Test_Samples": len(
                y_true_wavelet
            )
        }
    )

    prediction_dict[
        f"Causal DB4-L{best_level}"
    ] = y_pred_wavelet

    pd.DataFrame(
        {
            "date": test_dates,
            "actual_open": y_true_wavelet,
            "predicted_open": y_pred_wavelet
        }
    ).to_csv(
        os.path.join(
            OUTPUT_DIR,
            f"test_predictions_Causal_DB4_L{best_level}_W{best_window}.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )

    final_results = pd.DataFrame(
        results
    )

    path = os.path.join(
        OUTPUT_DIR,
        "final_test_results.csv"
    )

    final_results.to_csv(
        path,
        index=False,
        encoding="utf-8-sig"
    )

    return (
        final_results,
        test_dates,
        y_true_wavelet,
        prediction_dict
    )


# ============================================================
# 18. Validation MAE 热力图
# ============================================================

def plot_validation_heatmap(
    validation_results
):

    wavelet = validation_results[
        validation_results["Model"]
        == "Causal Wavelet-LSTM"
    ].copy()

    if wavelet.empty:
        return

    pivot = wavelet.pivot(
        index="Level",
        columns="Window",
        values="MAE"
    )

    plt.figure(
        figsize=(8, 6)
    )

    plt.imshow(
        pivot.values,
        aspect="auto"
    )

    plt.colorbar(
        label="Validation MAE"
    )

    plt.xticks(
        range(len(pivot.columns)),
        pivot.columns
    )

    plt.yticks(
        range(len(pivot.index)),
        pivot.index
    )

    plt.xlabel(
        "Denoising Window"
    )

    plt.ylabel(
        "Wavelet Level"
    )

    plt.title(
        "Validation MAE - "
        "Causal Wavelet"
    )

    # 在热力图中标数值
    for i in range(
        len(pivot.index)
    ):

        for j in range(
            len(pivot.columns)
        ):

            value = pivot.iloc[
                i,
                j
            ]

            plt.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center"
            )

    plt.tight_layout()

    path = os.path.join(
        OUTPUT_DIR,
        "validation_mae_heatmap.png"
    )

    plt.savefig(
        path,
        dpi=200
    )

    plt.close()


# ============================================================
# 19. 最终预测对比图
# ============================================================

def plot_final_predictions(
    test_dates,
    y_true,
    prediction_dict
):

    dates = pd.to_datetime(
        test_dates
    )

    plt.figure(
        figsize=(14, 6)
    )

    plt.plot(
        dates,
        y_true,
        label="Actual Open",
        linewidth=2
    )

    for name, values in (
        prediction_dict.items()
    ):

        plt.plot(
            dates,
            values,
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
        "Final Test Prediction Comparison"
    )

    plt.legend()

    plt.grid(True)

    plt.xticks(
        rotation=45
    )

    plt.tight_layout()

    path = os.path.join(
        OUTPUT_DIR,
        "final_prediction_comparison.png"
    )

    plt.savefig(
        path,
        dpi=200
    )

    plt.close()


# ============================================================
# 20. 主函数
# ============================================================

def main():

    set_seed(SEED)

    print()
    print("=" * 100)
    print("Validation-based Causal Wavelet Experiment")
    print("=" * 100)

    # --------------------------------------------------------
    # 读取
    # --------------------------------------------------------

    df = load_data()

    # --------------------------------------------------------
    # 划分
    # --------------------------------------------------------

    train_df, val_df, test_df = (
        split_data(df)
    )

    # --------------------------------------------------------
    # 计算所有 causal wavelet
    #
    # 注意：
    # 每个时间点只使用自己和过去数据
    # --------------------------------------------------------

    values = (
        df["open"]
        .values
        .astype(float)
    )

    denoised_cache = {}

    print()
    print("=" * 100)
    print("生成 Causal Wavelet 数据")
    print("=" * 100)

    for level in WAVELET_LEVELS:

        for window in DENOISE_WINDOWS:

            print()
            print(
                "-" * 80
            )

            denoised = (
                causal_wavelet_denoise(
                    values=values,
                    wavelet=WAVELET,
                    level=level,
                    window=window
                )
            )

            denoised_cache[
                (
                    level,
                    window
                )
            ] = denoised

    # ========================================================
    # Validation
    # ========================================================

    validation_results = (
        run_validation_experiments(
            raw_df=df,
            denoised_cache=denoised_cache
        )
    )

    # ========================================================
    # 选择参数
    # ========================================================

    best_level, best_window = (
        select_best_config(
            validation_results
        )
    )

    # ========================================================
    # Validation 图
    # ========================================================

    plot_validation_heatmap(
        validation_results
    )

    # ========================================================
    # Final Test
    # ========================================================

    (
        final_results,
        test_dates,
        y_true,
        prediction_dict
    ) = run_final_test(
        raw_df=df,
        denoised_cache=denoised_cache,
        best_level=best_level,
        best_window=best_window
    )

    # ========================================================
    # Final 图
    # ========================================================

    plot_final_predictions(
        test_dates,
        y_true,
        prediction_dict
    )

    # ========================================================
    # 输出最终结果
    # ========================================================

    print()
    print()
    print("=" * 100)
    print("最终 Test 结果")
    print("=" * 100)

    print(
        final_results.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    print()
    print(
        f"Validation 最优配置："
        f"db4-L{best_level}, "
        f"Window={best_window}"
    )

    print()
    print(
        f"结果目录："
        f"{OUTPUT_DIR}"
    )

    print()
    print(
        "主要文件："
    )

    print(
        "  validation_results.csv"
    )

    print(
        "  best_config.csv"
    )

    print(
        "  final_test_results.csv"
    )

    print(
        "  validation_mae_heatmap.png"
    )

    print(
        "  final_prediction_comparison.png"
    )

    print("=" * 100)


# ============================================================
# 21. 程序入口
# ============================================================

if __name__ == "__main__":

    main()