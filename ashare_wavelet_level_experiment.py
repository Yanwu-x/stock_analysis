# -*- coding: utf-8 -*-

"""
A-share Stock Price Prediction
Wavelet Denoising Level Experiment + LSTM

实验：
1. RAW      : 原始 Open + LSTM
2. DB4_L2   : db4 Level 2 + LSTM
3. DB4_L3   : db4 Level 3 + LSTM
4. DB4_L4   : db4 Level 4 + LSTM
5. DB4_L5   : db4 Level 5 + LSTM

固定参数：
- PAST_DAYS = 90
- EPOCHS = 100
- BATCH_SIZE = 32
- TRAIN_RATIO = 0.8
- SEED = 42
- Wavelet = db4

输出：
results_wavelet_level/
    experiment_results.csv

    predictions_RAW.csv
    predictions_DB4_L2.csv
    predictions_DB4_L3.csv
    predictions_DB4_L4.csv
    predictions_DB4_L5.csv

    loss_RAW.png
    loss_DB4_L2.png
    loss_DB4_L3.png
    loss_DB4_L4.png
    loss_DB4_L5.png

    denoising_comparison_DB4_L2.png
    denoising_comparison_DB4_L3.png
    denoising_comparison_DB4_L4.png
    denoising_comparison_DB4_L5.png

    level_comparison.png
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pywt
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam


# ============================================================
# 1. 全局参数
# ============================================================

DATA_FILE = "ashare_600519.csv"

OUTPUT_DIR = "results_wavelet_level"

TARGET_COLUMN = "open"

TRAIN_RATIO = 0.8

PAST_DAYS = 90
EPOCHS = 100
BATCH_SIZE = 32

SEED = 42

WAVELET = "db4"

# 这次重点测试 Level 2 / 3 / 4 / 5
WAVELET_LEVELS = [2, 3, 4, 5]


# ============================================================
# 2. 固定随机种子
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
# 3. 创建输出目录
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 4. 读取数据
# ============================================================

def load_data():

    print("=" * 70)
    print("读取 A 股数据")
    print("=" * 70)

    if not os.path.exists(DATA_FILE):

        raise FileNotFoundError(
            f"\n找不到数据文件：{DATA_FILE}\n"
            f"请确认：\n"
            f"1. csv 文件名称正确\n"
            f"2. csv 和 Python 文件在同一个目录\n"
        )

    df = pd.read_csv(DATA_FILE)

    print(f"数据文件：{DATA_FILE}")
    print(f"原始数据形状：{df.shape}")

    required_columns = [
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]

    missing_columns = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"数据缺少必要列：{missing_columns}"
        )

    # 日期
    df["date"] = pd.to_datetime(df["date"])

    # 按时间排序
    df = df.sort_values(
        "date"
    ).reset_index(drop=True)

    # Open 转数字
    df["open"] = pd.to_numeric(
        df["open"],
        errors="coerce"
    )

    # 删除 Open 缺失
    df = df.dropna(
        subset=["open"]
    ).reset_index(drop=True)

    # NaN 检查
    if df["open"].isna().any():

        raise ValueError(
            "Open 中存在 NaN。"
        )

    # Inf 检查
    if np.isinf(
        df["open"].values
    ).any():

        raise ValueError(
            "Open 中存在 Inf。"
        )

    print(
        f"清洗后数据形状：{df.shape}"
    )

    print(
        f"开始日期："
        f"{df['date'].iloc[0].date()}"
    )

    print(
        f"结束日期："
        f"{df['date'].iloc[-1].date()}"
    )

    return df


# ============================================================
# 5. 时间序列 80/20 划分
# ============================================================

def split_data(df):

    split_index = int(
        len(df) * TRAIN_RATIO
    )

    train_df = df.iloc[
        :split_index
    ].copy()

    test_df = df.iloc[
        split_index:
    ].copy()

    print()
    print("=" * 70)
    print("训练集 / 测试集划分")
    print("=" * 70)

    print(
        f"总样本数：{len(df)}"
    )

    print(
        f"训练集：{len(train_df)}"
    )

    print(
        f"测试集：{len(test_df)}"
    )

    print()

    print(
        f"训练集："
        f"{train_df['date'].iloc[0].date()}"
        f" -> "
        f"{train_df['date'].iloc[-1].date()}"
    )

    print(
        f"测试集："
        f"{test_df['date'].iloc[0].date()}"
        f" -> "
        f"{test_df['date'].iloc[-1].date()}"
    )

    return train_df, test_df


# ============================================================
# 6. 计算小波去噪阈值
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

    # 最高频 detail coefficient
    detail_coeff = coeffs[-1]

    # 估计噪声标准差
    sigma = (
        np.median(
            np.abs(detail_coeff)
        ) / 0.6745
    )

    # Universal Threshold
    threshold = (
        sigma
        * np.sqrt(
            2 * np.log(len(train_values))
        )
    )

    return threshold, actual_level


# ============================================================
# 7. 小波去噪
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

    # 保留低频 approximation
    denoised_coeffs = [
        coeffs[0]
    ]

    # 对 detail coefficient 做软阈值
    for detail in coeffs[1:]:

        denoised_detail = pywt.threshold(
            detail,
            threshold,
            mode="soft"
        )

        denoised_coeffs.append(
            denoised_detail
        )

    # 小波重构
    reconstructed = pywt.waverec(
        denoised_coeffs,
        wavelet=wavelet,
        mode="symmetric"
    )

    # 保证与原数据长度一致
    reconstructed = reconstructed[
        :len(values)
    ]

    # 检查
    if np.isnan(
        reconstructed
    ).any():

        raise ValueError(
            "小波去噪后出现 NaN。"
        )

    if np.isinf(
        reconstructed
    ).any():

        raise ValueError(
            "小波去噪后出现 Inf。"
        )

    return reconstructed


# ============================================================
# 8. 构造训练序列
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

    X = np.array(X)
    y = np.array(y)

    return X, y


# ============================================================
# 9. 构造测试序列
# ============================================================

def create_test_sequences(
    train_scaled,
    test_scaled,
    past_days
):

    # 使用训练集末尾90天
    # 给测试集第一个样本提供历史窗口
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

    X = np.array(X)
    y = np.array(y)

    return X, y


# ============================================================
# 10. 构建 LSTM
# ============================================================

def build_model():

    model = Sequential()

    # 第一层
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

    # 第二层
    model.add(
        LSTM(
            50,
            return_sequences=True
        )
    )

    model.add(
        Dropout(0.2)
    )

    # 第三层
    model.add(
        LSTM(
            50,
            return_sequences=True
        )
    )

    model.add(
        Dropout(0.2)
    )

    # 第四层
    model.add(
        LSTM(
            50,
            return_sequences=False
        )
    )

    model.add(
        Dropout(0.2)
    )

    # 输出层
    model.add(
        Dense(1)
    )

    model.compile(
        optimizer=Adam(),
        loss="mean_squared_error"
    )

    return model


# ============================================================
# 11. Directional Accuracy
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

    accuracy = np.mean(
        true_direction
        == pred_direction
    )

    return accuracy


# ============================================================
# 12. 保存 Loss 曲线
# ============================================================

def save_loss_plot(
    history,
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
        f"LSTM Training Loss - "
        f"{experiment_name}"
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    save_path = os.path.join(
        OUTPUT_DIR,
        f"loss_{experiment_name}.png"
    )

    plt.savefig(
        save_path,
        dpi=200
    )

    plt.close()


# ============================================================
# 13. 保存去噪前后对比图
# ============================================================

def save_denoising_plot(
    original_train,
    denoised_train,
    original_test,
    denoised_test,
    train_dates,
    test_dates,
    level
):

    plt.figure(
        figsize=(14, 6)
    )

    # 原始训练集
    plt.plot(
        train_dates,
        original_train,
        label="Original Train",
        alpha=0.45
    )

    # 去噪训练集
    plt.plot(
        train_dates,
        denoised_train,
        label=(
            f"Wavelet Denoised Train "
            f"(db4-L{level})"
        ),
        linewidth=1.5
    )

    # 原始测试集
    plt.plot(
        test_dates,
        original_test,
        label="Original Test",
        alpha=0.45
    )

    # 去噪测试集
    plt.plot(
        test_dates,
        denoised_test,
        label=(
            f"Wavelet Denoised Test "
            f"(db4-L{level})"
        ),
        linewidth=1.5
    )

    plt.xlabel(
        "Date"
    )

    plt.ylabel(
        "Open Price"
    )

    plt.title(
        f"Wavelet Denoising "
        f"Comparison - db4 Level {level}"
    )

    plt.legend()

    plt.grid(True)

    plt.xticks(
        rotation=45
    )

    plt.tight_layout()

    save_path = os.path.join(
        OUTPUT_DIR,
        f"denoising_comparison_DB4_L{level}.png"
    )

    plt.savefig(
        save_path,
        dpi=200
    )

    plt.close()


# ============================================================
# 14. 保存预测结果
# ============================================================

def save_prediction_csv(
    dates,
    y_true,
    y_pred,
    experiment_name
):

    prediction_df = pd.DataFrame(
        {
            "date": dates,
            "actual_open": y_true,
            "predicted_open": y_pred
        }
    )

    save_path = os.path.join(
        OUTPUT_DIR,
        f"predictions_{experiment_name}.csv"
    )

    prediction_df.to_csv(
        save_path,
        index=False,
        encoding="utf-8-sig"
    )


# ============================================================
# 15. 单个实验
# ============================================================

def run_experiment(
    experiment_name,
    train_values,
    test_values
):

    print()
    print("=" * 70)
    print(
        f"开始实验：{experiment_name}"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # 数据转数组
    # --------------------------------------------------------

    train_values = np.asarray(
        train_values,
        dtype=np.float64
    )

    test_values = np.asarray(
        test_values,
        dtype=np.float64
    )

    # --------------------------------------------------------
    # NaN / Inf 检查
    # --------------------------------------------------------

    if np.isnan(
        train_values
    ).any():

        raise ValueError(
            f"{experiment_name} "
            f"训练数据存在 NaN。"
        )

    if np.isnan(
        test_values
    ).any():

        raise ValueError(
            f"{experiment_name} "
            f"测试数据存在 NaN。"
        )

    if np.isinf(
        train_values
    ).any():

        raise ValueError(
            f"{experiment_name} "
            f"训练数据存在 Inf。"
        )

    if np.isinf(
        test_values
    ).any():

        raise ValueError(
            f"{experiment_name} "
            f"测试数据存在 Inf。"
        )

    # --------------------------------------------------------
    # MinMaxScaler
    # --------------------------------------------------------

    scaler = MinMaxScaler(
        feature_range=(0, 1)
    )

    # 只使用训练集 fit
    train_scaled = scaler.fit_transform(
        train_values.reshape(-1, 1)
    )

    # 测试集只 transform
    test_scaled = scaler.transform(
        test_values.reshape(-1, 1)
    )

    # --------------------------------------------------------
    # 构造训练序列
    # --------------------------------------------------------

    X_train, y_train = (
        create_train_sequences(
            train_scaled,
            PAST_DAYS
        )
    )

    # --------------------------------------------------------
    # 构造测试序列
    # --------------------------------------------------------

    X_test, y_test = (
        create_test_sequences(
            train_scaled,
            test_scaled,
            PAST_DAYS
        )
    )

    print(
        f"X_train shape: {X_train.shape}"
    )

    print(
        f"y_train shape: {y_train.shape}"
    )

    print(
        f"X_test shape: {X_test.shape}"
    )

    print(
        f"y_test shape: {y_test.shape}"
    )

    # --------------------------------------------------------
    # 固定随机种子
    # --------------------------------------------------------

    set_seed(SEED)

    # --------------------------------------------------------
    # 建模
    # --------------------------------------------------------

    model = build_model()

    # --------------------------------------------------------
    # 训练
    # --------------------------------------------------------

    history = model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=False,
        verbose=1
    )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    save_loss_plot(
        history,
        experiment_name
    )

    # --------------------------------------------------------
    # 预测
    # --------------------------------------------------------

    predicted_scaled = model.predict(
        X_test,
        verbose=0
    )

    # --------------------------------------------------------
    # 反归一化
    # --------------------------------------------------------

    y_pred = scaler.inverse_transform(
        predicted_scaled
    ).flatten()

    y_true = scaler.inverse_transform(
        y_test.reshape(-1, 1)
    ).flatten()

    # --------------------------------------------------------
    # MAE
    # --------------------------------------------------------

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    # --------------------------------------------------------
    # RMSE
    # --------------------------------------------------------

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    # --------------------------------------------------------
    # Directional Accuracy
    # --------------------------------------------------------

    directional_accuracy = (
        calculate_directional_accuracy(
            y_true,
            y_pred
        )
    )

    # --------------------------------------------------------
    # 保存预测结果
    # --------------------------------------------------------

    save_prediction_csv(
        test_dates,
        y_true,
        y_pred,
        experiment_name
    )

    # --------------------------------------------------------
    # 输出结果
    # --------------------------------------------------------

    print()
    print("-" * 70)

    print(
        f"实验：{experiment_name}"
    )

    print(
        f"MAE  = {mae:.4f}"
    )

    print(
        f"RMSE = {rmse:.4f}"
    )

    print(
        f"Directional Accuracy = "
        f"{directional_accuracy:.4f}"
        f" ({directional_accuracy * 100:.2f}%)"
    )

    print("-" * 70)

    return {
        "Experiment": experiment_name,
        "Wavelet": (
            "None"
            if experiment_name == "RAW"
            else WAVELET
        ),
        "Level": (
            0
            if experiment_name == "RAW"
            else int(
                experiment_name.split("_L")[1]
            )
        ),
        "MAE": mae,
        "RMSE": rmse,
        "Directional_Accuracy": (
            directional_accuracy
        ),
        "Test_Samples": len(y_true)
    }


# ============================================================
# 16. 绘制 Level 对比图
# ============================================================

def save_level_comparison(
    results_df
):

    wavelet_results = (
        results_df[
            results_df["Experiment"] != "RAW"
        ]
        .copy()
    )

    raw_result = (
        results_df[
            results_df["Experiment"] == "RAW"
        ]
        .iloc[0]
    )

    # --------------------------------------------------------
    # MAE
    # --------------------------------------------------------

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        wavelet_results["Level"],
        wavelet_results["MAE"],
        marker="o",
        label="Wavelet MAE"
    )

    plt.axhline(
        raw_result["MAE"],
        linestyle="--",
        label="RAW Baseline"
    )

    plt.xlabel(
        "Wavelet Decomposition Level"
    )

    plt.ylabel(
        "MAE"
    )

    plt.title(
        "MAE Comparison Across Wavelet Levels"
    )

    plt.xticks(
        wavelet_results["Level"]
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    save_path = os.path.join(
        OUTPUT_DIR,
        "level_comparison_MAE.png"
    )

    plt.savefig(
        save_path,
        dpi=200
    )

    plt.close()

    # --------------------------------------------------------
    # RMSE
    # --------------------------------------------------------

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        wavelet_results["Level"],
        wavelet_results["RMSE"],
        marker="o",
        label="Wavelet RMSE"
    )

    plt.axhline(
        raw_result["RMSE"],
        linestyle="--",
        label="RAW Baseline"
    )

    plt.xlabel(
        "Wavelet Decomposition Level"
    )

    plt.ylabel(
        "RMSE"
    )

    plt.title(
        "RMSE Comparison Across Wavelet Levels"
    )

    plt.xticks(
        wavelet_results["Level"]
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    save_path = os.path.join(
        OUTPUT_DIR,
        "level_comparison_RMSE.png"
    )

    plt.savefig(
        save_path,
        dpi=200
    )

    plt.close()

    # --------------------------------------------------------
    # Directional Accuracy
    # --------------------------------------------------------

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        wavelet_results["Level"],
        wavelet_results[
            "Directional_Accuracy"
        ],
        marker="o",
        label="Wavelet DA"
    )

    plt.axhline(
        raw_result[
            "Directional_Accuracy"
        ],
        linestyle="--",
        label="RAW Baseline"
    )

    plt.xlabel(
        "Wavelet Decomposition Level"
    )

    plt.ylabel(
        "Directional Accuracy"
    )

    plt.title(
        "Directional Accuracy "
        "Across Wavelet Levels"
    )

    plt.xticks(
        wavelet_results["Level"]
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    save_path = os.path.join(
        OUTPUT_DIR,
        "level_comparison_Directional_Accuracy.png"
    )

    plt.savefig(
        save_path,
        dpi=200
    )

    plt.close()


# ============================================================
# 17. 主函数
# ============================================================

def main():

    global test_dates

    set_seed(SEED)

    # --------------------------------------------------------
    # 读取数据
    # --------------------------------------------------------

    df = load_data()

    # --------------------------------------------------------
    # 80/20 时间划分
    # --------------------------------------------------------

    train_df, test_df = split_data(
        df
    )

    # --------------------------------------------------------
    # 原始 Open
    # --------------------------------------------------------

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

    test_dates = (
        test_df["date"]
        .dt.strftime(
            "%Y-%m-%d"
        )
        .values
    )

    train_dates = train_df["date"]

    # ========================================================
    # 1. RAW baseline
    # ========================================================

    raw_result = run_experiment(
        experiment_name="RAW",
        train_values=train_open,
        test_values=test_open
    )

    all_results = [
        raw_result
    ]

    # ========================================================
    # 2. Level 2 / 3 / 4 / 5
    # ========================================================

    for level in WAVELET_LEVELS:

        print()
        print()
        print("=" * 70)

        print(
            f"正在进行 db4 Level {level} "
            f"小波去噪"
        )

        print("=" * 70)

        # ----------------------------------------------------
        # 估计 threshold
        # ----------------------------------------------------

        threshold, actual_level = (
            estimate_threshold(
                train_open,
                WAVELET,
                level
            )
        )

        print(
            f"Wavelet           : {WAVELET}"
        )

        print(
            f"Requested Level   : {level}"
        )

        print(
            f"Actual Level      : {actual_level}"
        )

        print(
            f"Threshold         : {threshold:.6f}"
        )

        # ----------------------------------------------------
        # 训练集去噪
        # ----------------------------------------------------

        denoised_train = (
            wavelet_denoise(
                train_open,
                wavelet=WAVELET,
                level=level,
                threshold=threshold
            )
        )

        # ----------------------------------------------------
        # 测试集去噪
        # ----------------------------------------------------

        denoised_test = (
            wavelet_denoise(
                test_open,
                wavelet=WAVELET,
                level=level,
                threshold=threshold
            )
        )

        # ----------------------------------------------------
        # 保存去噪图
        # ----------------------------------------------------

        save_denoising_plot(
            original_train=train_open,
            denoised_train=denoised_train,
            original_test=test_open,
            denoised_test=denoised_test,
            train_dates=train_dates,
            test_dates=test_df["date"],
            level=level
        )

        # ----------------------------------------------------
        # LSTM 实验
        # ----------------------------------------------------

        experiment_name = (
            f"DB4_L{level}"
        )

        result = run_experiment(
            experiment_name=experiment_name,
            train_values=denoised_train,
            test_values=denoised_test
        )

        result["Wavelet"] = WAVELET
        result["Level"] = level

        all_results.append(
            result
        )

    # ========================================================
    # 3. 保存结果
    # ========================================================

    results_df = pd.DataFrame(
        all_results
    )

    results_df = results_df[
        [
            "Experiment",
            "Wavelet",
            "Level",
            "MAE",
            "RMSE",
            "Directional_Accuracy",
            "Test_Samples"
        ]
    ]

    result_path = os.path.join(
        OUTPUT_DIR,
        "experiment_results.csv"
    )

    results_df.to_csv(
        result_path,
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # 4. Level 对比图
    # ========================================================

    save_level_comparison(
        results_df
    )

    # ========================================================
    # 5. 最终结果
    # ========================================================

    print()
    print()
    print("=" * 90)
    print("全部小波分解层数实验完成")
    print("=" * 90)

    print()

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    print()

    # --------------------------------------------------------
    # 找最佳 MAE
    # --------------------------------------------------------

    best_mae_row = (
        results_df
        .loc[
            results_df["MAE"].idxmin()
        ]
    )

    # --------------------------------------------------------
    # 找最佳 RMSE
    # --------------------------------------------------------

    best_rmse_row = (
        results_df
        .loc[
            results_df["RMSE"].idxmin()
        ]
    )

    # --------------------------------------------------------
    # 找最佳 Directional Accuracy
    # --------------------------------------------------------

    best_da_row = (
        results_df
        .loc[
            results_df[
                "Directional_Accuracy"
            ].idxmax()
        ]
    )

    print(
        f"MAE 最优："
        f"{best_mae_row['Experiment']} "
        f"-> "
        f"{best_mae_row['MAE']:.4f}"
    )

    print(
        f"RMSE 最优："
        f"{best_rmse_row['Experiment']} "
        f"-> "
        f"{best_rmse_row['RMSE']:.4f}"
    )

    print(
        f"Directional Accuracy 最优："
        f"{best_da_row['Experiment']} "
        f"-> "
        f"{best_da_row['Directional_Accuracy'] * 100:.2f}%"
    )

    print()

    print(
        f"结果文件："
        f"{result_path}"
    )

    print()
    print("=" * 90)


# ============================================================
# 18. 程序入口
# ============================================================

if __name__ == "__main__":
    main()