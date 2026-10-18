# -*- coding: utf-8 -*-

"""
A-share Stock Price Prediction
Wavelet Stability Experiment + LSTM

目的：
验证不同随机种子下，小波去噪 LSTM 的结果是否稳定。

实验：
Seed = 42
Seed = 123
Seed = 2026

每个 Seed：
    RAW
    DB4_L2
    DB4_L3
    DB4_L4
    DB4_L5

共：
    3 Seeds × 5 Models = 15 experiments

固定：
    PAST_DAYS = 90
    EPOCHS = 100
    BATCH_SIZE = 32
    TRAIN_RATIO = 0.8
    Wavelet = db4
"""

import os

# 减少 TensorFlow 输出
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
# 1. 参数
# ============================================================

DATA_FILE = "ashare_600519.csv"

OUTPUT_DIR = "results_wavelet_stability"

TRAIN_RATIO = 0.8

PAST_DAYS = 90

EPOCHS = 100

BATCH_SIZE = 32

WAVELET = "db4"

WAVELET_LEVELS = [2, 3, 4, 5]

# 本次稳定性实验使用 3 个随机种子
SEEDS = [42, 123, 2026]


# ============================================================
# 2. 创建输出目录
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 3. 固定随机种子
# ============================================================

def set_seed(seed):

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)

    np.random.seed(seed)

    tf.random.set_seed(seed)

    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


# ============================================================
# 4. 读取数据
# ============================================================

def load_data():

    print("=" * 80)
    print("读取 A 股数据")
    print("=" * 80)

    if not os.path.exists(DATA_FILE):

        raise FileNotFoundError(
            f"找不到数据文件：{DATA_FILE}\n"
            f"请检查 csv 文件和 Python 文件是否在同一目录。"
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
        subset=["date", "open"]
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
        f"{df['date'].iloc[0].date()} "
        f"-> "
        f"{df['date'].iloc[-1].date()}"
    )

    return df


# ============================================================
# 5. 小波阈值
# ============================================================

def estimate_threshold(
    train_values,
    wavelet,
    level
):

    wavelet_obj = pywt.Wavelet(
        wavelet
    )

    max_level = pywt.dwt_max_level(
        len(train_values),
        wavelet_obj.dec_len
    )

    actual_level = min(
        level,
        max_level
    )

    coeffs = pywt.wavedec(
        train_values,
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

    threshold = (
        sigma
        * np.sqrt(
            2 * np.log(
                len(train_values)
            )
        )
    )

    return threshold, actual_level


# ============================================================
# 6. 小波去噪
# ============================================================

def wavelet_denoise(
    values,
    wavelet,
    level,
    threshold
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

    denoised_coeffs = [
        coeffs[0]
    ]

    for detail in coeffs[1:]:

        denoised_detail = pywt.threshold(
            detail,
            threshold,
            mode="soft"
        )

        denoised_coeffs.append(
            denoised_detail
        )

    reconstructed = pywt.waverec(
        denoised_coeffs,
        wavelet=wavelet,
        mode="symmetric"
    )

    reconstructed = reconstructed[
        :len(values)
    ]

    if np.isnan(
        reconstructed
    ).any():

        raise ValueError(
            "小波去噪后出现 NaN"
        )

    if np.isinf(
        reconstructed
    ).any():

        raise ValueError(
            "小波去噪后出现 Inf"
        )

    return reconstructed


# ============================================================
# 7. 创建训练序列
# ============================================================

def create_train_sequences(
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
# 8. 创建测试序列
# ============================================================

def create_test_sequences(
    train_scaled,
    test_scaled,
    past_days
):

    combined = np.concatenate(
        [
            train_scaled[-past_days:],
            test_scaled
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
# 9. LSTM 模型
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

    true_direction = np.sign(
        true_diff
    )

    pred_direction = np.sign(
        pred_diff
    )

    return np.mean(
        true_direction
        == pred_direction
    )


# ============================================================
# 11. 保存单次预测结果
# ============================================================

def save_predictions(
    test_dates,
    y_true,
    y_pred,
    seed,
    experiment_name
):

    prediction_df = pd.DataFrame(
        {
            "date": test_dates,
            "actual_open": y_true,
            "predicted_open": y_pred
        }
    )

    filename = (
        f"predictions_"
        f"{experiment_name}_seed{seed}.csv"
    )

    path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    prediction_df.to_csv(
        path,
        index=False,
        encoding="utf-8-sig"
    )


# ============================================================
# 12. 保存单次 Loss 曲线
# ============================================================

def save_loss_plot(
    history,
    seed,
    experiment_name
):

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        history.history["loss"],
        label="Training Loss"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "MSE Loss"
    )

    plt.title(
        f"{experiment_name} - Seed {seed}"
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    filename = (
        f"loss_"
        f"{experiment_name}_seed{seed}.png"
    )

    path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    plt.savefig(
        path,
        dpi=200
    )

    plt.close()


# ============================================================
# 13. 单次实验
# ============================================================

def run_single_experiment(
    df,
    experiment_name,
    seed,
    train_values,
    test_values
):

    print()
    print(
        "=" * 80
    )

    print(
        f"Experiment : {experiment_name}"
    )

    print(
        f"Seed       : {seed}"
    )

    print(
        "=" * 80
    )

    # --------------------------------------------------------
    # 固定随机种子
    # --------------------------------------------------------

    set_seed(seed)

    # --------------------------------------------------------
    # scaler
    # --------------------------------------------------------

    scaler = MinMaxScaler(
        feature_range=(0, 1)
    )

    train_scaled = scaler.fit_transform(
        train_values.reshape(-1, 1)
    )

    test_scaled = scaler.transform(
        test_values.reshape(-1, 1)
    )

    # --------------------------------------------------------
    # train sequences
    # --------------------------------------------------------

    X_train, y_train = (
        create_train_sequences(
            train_scaled,
            PAST_DAYS
        )
    )

    # --------------------------------------------------------
    # test sequences
    # --------------------------------------------------------

    X_test, y_test = (
        create_test_sequences(
            train_scaled,
            test_scaled,
            PAST_DAYS
        )
    )

    print(
        f"X_train: {X_train.shape}"
    )

    print(
        f"X_test : {X_test.shape}"
    )

    # --------------------------------------------------------
    # model
    # --------------------------------------------------------

    model = build_model()

    # --------------------------------------------------------
    # training
    # --------------------------------------------------------

    history = model.fit(
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
    # loss
    # --------------------------------------------------------

    save_loss_plot(
        history,
        seed,
        experiment_name
    )

    # --------------------------------------------------------
    # prediction
    # --------------------------------------------------------

    pred_scaled = model.predict(
        X_test,
        verbose=0
    )

    # --------------------------------------------------------
    # inverse transform
    # --------------------------------------------------------

    y_pred = scaler.inverse_transform(
        pred_scaled
    ).flatten()

    y_true = scaler.inverse_transform(
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
    # save predictions
    # --------------------------------------------------------

    test_dates = (
        df["date"]
        .iloc[
            len(df) - len(test_values):
        ]
        .dt.strftime("%Y-%m-%d")
        .values
    )

    save_predictions(
        test_dates,
        y_true,
        y_pred,
        seed,
        experiment_name
    )

    # --------------------------------------------------------
    # print
    # --------------------------------------------------------

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
        "Seed": seed,
        "Experiment": experiment_name,
        "MAE": mae,
        "RMSE": rmse,
        "Directional_Accuracy": da,
        "Test_Samples": len(y_true)
    }


# ============================================================
# 14. 汇总稳定性结果
# ============================================================

def create_summary(
    results_df
):

    print()
    print()
    print("=" * 100)
    print("稳定性实验汇总")
    print("=" * 100)

    # --------------------------------------------------------
    # 平均值
    # --------------------------------------------------------

    summary = (
        results_df
        .groupby("Experiment")
        .agg(
            Mean_MAE=("MAE", "mean"),
            Std_MAE=("MAE", "std"),
            Mean_RMSE=("RMSE", "mean"),
            Std_RMSE=("RMSE", "std"),
            Mean_DA=(
                "Directional_Accuracy",
                "mean"
            ),
            Std_DA=(
                "Directional_Accuracy",
                "std"
            ),
            Seeds=("Seed", "count")
        )
        .reset_index()
    )

    # --------------------------------------------------------
    # 排序
    # --------------------------------------------------------

    summary = summary.sort_values(
        "Mean_MAE"
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # 保存
    # --------------------------------------------------------

    summary_path = os.path.join(
        OUTPUT_DIR,
        "wavelet_stability_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    print()
    print(
        f"稳定性汇总已保存："
        f"{summary_path}"
    )

    return summary


# ============================================================
# 15. 绘制稳定性对比图
# ============================================================

def plot_stability(
    summary
):

    experiments = (
        summary["Experiment"]
        .tolist()
    )

    x = np.arange(
        len(experiments)
    )

    # --------------------------------------------------------
    # MAE
    # --------------------------------------------------------

    plt.figure(
        figsize=(10, 6)
    )

    plt.errorbar(
        x,
        summary["Mean_MAE"],
        yerr=summary["Std_MAE"],
        fmt="o-",
        capsize=5
    )

    plt.xticks(
        x,
        experiments
    )

    plt.xlabel(
        "Model"
    )

    plt.ylabel(
        "MAE"
    )

    plt.title(
        "Wavelet Stability - MAE"
    )

    plt.grid(True)

    plt.tight_layout()

    path = os.path.join(
        OUTPUT_DIR,
        "stability_MAE.png"
    )

    plt.savefig(
        path,
        dpi=200
    )

    plt.close()

    # --------------------------------------------------------
    # RMSE
    # --------------------------------------------------------

    plt.figure(
        figsize=(10, 6)
    )

    plt.errorbar(
        x,
        summary["Mean_RMSE"],
        yerr=summary["Std_RMSE"],
        fmt="o-",
        capsize=5
    )

    plt.xticks(
        x,
        experiments
    )

    plt.xlabel(
        "Model"
    )

    plt.ylabel(
        "RMSE"
    )

    plt.title(
        "Wavelet Stability - RMSE"
    )

    plt.grid(True)

    plt.tight_layout()

    path = os.path.join(
        OUTPUT_DIR,
        "stability_RMSE.png"
    )

    plt.savefig(
        path,
        dpi=200
    )

    plt.close()

    # --------------------------------------------------------
    # Directional Accuracy
    # --------------------------------------------------------

    plt.figure(
        figsize=(10, 6)
    )

    plt.errorbar(
        x,
        summary["Mean_DA"] * 100,
        yerr=summary["Std_DA"] * 100,
        fmt="o-",
        capsize=5
    )

    plt.xticks(
        x,
        experiments
    )

    plt.xlabel(
        "Model"
    )

    plt.ylabel(
        "Directional Accuracy (%)"
    )

    plt.title(
        "Wavelet Stability - Directional Accuracy"
    )

    plt.grid(True)

    plt.tight_layout()

    path = os.path.join(
        OUTPUT_DIR,
        "stability_Directional_Accuracy.png"
    )

    plt.savefig(
        path,
        dpi=200
    )

    plt.close()


# ============================================================
# 16. 主函数
# ============================================================

def main():

    print()
    print("=" * 100)
    print("Wavelet-LSTM 稳定性实验")
    print("=" * 100)

    # --------------------------------------------------------
    # 数据
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

    train_open = (
        train_df["open"]
        .values
        .astype(float)
    )

    test_open = (
        test_df["open"]
        .values
        .astype(float)
    )

    print()
    print(
        f"Train = {len(train_open)}"
    )

    print(
        f"Test  = {len(test_open)}"
    )

    # --------------------------------------------------------
    # 所有结果
    # --------------------------------------------------------

    all_results = []

    # ========================================================
    # 每个 Seed
    # ========================================================

    for seed in SEEDS:

        print()
        print()
        print(
            "#" * 100
        )

        print(
            f"当前 Seed = {seed}"
        )

        print(
            "#" * 100
        )

        # ----------------------------------------------------
        # RAW
        # ----------------------------------------------------

        raw_result = run_single_experiment(
            df=df,
            experiment_name="RAW",
            seed=seed,
            train_values=train_open,
            test_values=test_open
        )

        all_results.append(
            raw_result
        )

        # ----------------------------------------------------
        # Wavelet
        # ----------------------------------------------------

        # threshold 由训练集估计
        # 每个 level 单独计算
        for level in WAVELET_LEVELS:

            threshold, actual_level = (
                estimate_threshold(
                    train_open,
                    WAVELET,
                    level
                )
            )

            print()
            print(
                f"db4 Level {level}"
            )

            print(
                f"Actual Level = "
                f"{actual_level}"
            )

            print(
                f"Threshold = "
                f"{threshold:.6f}"
            )

            # ------------------------------------------------
            # 训练集去噪
            # ------------------------------------------------

            denoised_train = (
                wavelet_denoise(
                    train_open,
                    WAVELET,
                    level,
                    threshold
                )
            )

            # ------------------------------------------------
            # 测试集去噪
            # ------------------------------------------------

            denoised_test = (
                wavelet_denoise(
                    test_open,
                    WAVELET,
                    level,
                    threshold
                )
            )

            experiment_name = (
                f"DB4_L{level}"
            )

            result = run_single_experiment(
                df=df,
                experiment_name=experiment_name,
                seed=seed,
                train_values=denoised_train,
                test_values=denoised_test
            )

            all_results.append(
                result
            )

    # ========================================================
    # 保存每一次实验的结果
    # ========================================================

    results_df = pd.DataFrame(
        all_results
    )

    raw_results_path = os.path.join(
        OUTPUT_DIR,
        "wavelet_stability_all_results.csv"
    )

    results_df.to_csv(
        raw_results_path,
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # 汇总
    # ========================================================

    summary = create_summary(
        results_df
    )

    # ========================================================
    # 图
    # ========================================================

    plot_stability(
        summary
    )

    # ========================================================
    # 最终结论
    # ========================================================

    best_mae = summary.iloc[0]

    best_rmse = summary.loc[
        summary["Mean_RMSE"].idxmin()
    ]

    best_da = summary.loc[
        summary["Mean_DA"].idxmax()
    ]

    print()
    print()
    print("=" * 100)
    print("稳定性实验最终结果")
    print("=" * 100)

    print(
        f"平均 MAE 最优："
        f"{best_mae['Experiment']}"
        f" -> "
        f"{best_mae['Mean_MAE']:.4f}"
    )

    print(
        f"平均 RMSE 最优："
        f"{best_rmse['Experiment']}"
        f" -> "
        f"{best_rmse['Mean_RMSE']:.4f}"
    )

    print(
        f"平均 Directional Accuracy 最优："
        f"{best_da['Experiment']}"
        f" -> "
        f"{best_da['Mean_DA'] * 100:.2f}%"
    )

    print()
    print(
        f"全部实验结果："
        f"{raw_results_path}"
    )

    print(
        f"汇总结果："
        f"{os.path.join(OUTPUT_DIR, 'wavelet_stability_summary.csv')}"
    )

    print()
    print("=" * 100)


if __name__ == "__main__":
    main()