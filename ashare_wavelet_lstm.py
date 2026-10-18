# -*- coding: utf-8 -*-

"""
A-share Stock Price Prediction
Wavelet Denoising + LSTM

实验：
1. RAW      : 原始 Open + LSTM
2. DB4_L2   : db4 小波去噪 Level 2 + LSTM
3. DB4_L4   : db4 小波去噪 Level 4 + LSTM

保持与之前实验一致：
- PAST_DAYS = 90
- EPOCHS = 100
- BATCH_SIZE = 32
- TRAIN_RATIO = 0.8
- SEED = 42
- 4层 LSTM
- 每层 Dropout = 0.2

输出：
results_wavelet/
    experiment_results.csv
    predictions_RAW.csv
    predictions_DB4_L2.csv
    predictions_DB4_L4.csv
    loss_RAW.png
    loss_DB4_L2.png
    loss_DB4_L4.png
    denoising_comparison_DB4_L2.png
    denoising_comparison_DB4_L4.png
"""

import os

# ============================================================
# 1. 环境变量
# ============================================================

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
# 2. 全局参数
# ============================================================

DATA_FILE = "ashare_600519.csv"

OUTPUT_DIR = "results_wavelet"

PAST_DAYS = 90
EPOCHS = 100
BATCH_SIZE = 32
TRAIN_RATIO = 0.8
SEED = 42

# 小波参数
WAVELET = "db4"
WAVELET_LEVELS = [2, 4]

# 预测列
TARGET_COLUMN = "open"


# ============================================================
# 3. 固定随机种子
# ============================================================

def set_seed(seed=42):
    """
    尽可能固定实验随机性，保证不同实验之间可以公平比较。
    """
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
# 4. 创建输出目录
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 5. 读取数据
# ============================================================

def load_data():
    print("=" * 60)
    print("读取数据")
    print("=" * 60)

    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(
            f"找不到数据文件：{DATA_FILE}\n"
            f"请确认 csv 和本 Python 文件位于同一个文件夹。"
        )

    df = pd.read_csv(DATA_FILE)

    print(f"数据文件：{DATA_FILE}")
    print(f"数据形状：{df.shape}")
    print(f"列名：{df.columns.tolist()}")

    # 检查必须列
    required_columns = ["date", "open", "high", "low", "close", "volume"]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"数据缺少以下必要列：{missing_columns}"
        )

    # 日期处理
    df["date"] = pd.to_datetime(df["date"])

    # 按日期升序排列
    df = df.sort_values("date").reset_index(drop=True)

    # Open 转成数值
    df[TARGET_COLUMN] = pd.to_numeric(
        df[TARGET_COLUMN],
        errors="coerce"
    )

    # 删除 Open 缺失
    df = df.dropna(
        subset=[TARGET_COLUMN]
    ).reset_index(drop=True)

    # 检查 NaN / Inf
    if df[TARGET_COLUMN].isna().any():
        raise ValueError("Open 中存在 NaN。")

    if np.isinf(df[TARGET_COLUMN].values).any():
        raise ValueError("Open 中存在 Inf。")

    print()
    print(f"清洗后数据形状：{df.shape}")
    print(f"开始日期：{df['date'].iloc[0].date()}")
    print(f"结束日期：{df['date'].iloc[-1].date()}")

    return df


# ============================================================
# 6. 80/20 时间序列划分
# ============================================================

def split_data(df):
    """
    按时间顺序划分：
    前80% -> Train
    后20% -> Test

    不进行随机打乱。
    """

    split_index = int(len(df) * TRAIN_RATIO)

    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()

    print()
    print("=" * 60)
    print("训练集 / 测试集划分")
    print("=" * 60)

    print(f"总样本数：{len(df)}")
    print(f"训练集：{len(train_df)}")
    print(f"测试集：{len(test_df)}")

    print()
    print(
        f"训练集时间："
        f"{train_df['date'].iloc[0].date()} "
        f"-> "
        f"{train_df['date'].iloc[-1].date()}"
    )

    print(
        f"测试集时间："
        f"{test_df['date'].iloc[0].date()} "
        f"-> "
        f"{test_df['date'].iloc[-1].date()}"
    )

    return train_df, test_df


# ============================================================
# 7. 小波去噪
# ============================================================

def estimate_threshold(train_values, wavelet, level):
    """
    使用训练集估计噪声水平和阈值。

    使用最高层 detail coefficient：
        sigma = median(|detail|) / 0.6745

    再使用 Universal Threshold：
        threshold = sigma * sqrt(2 * ln(N))

    注意：
    阈值只从训练集估计，避免根据测试集统计量调整参数。
    """

    max_level = pywt.dwt_max_level(
        len(train_values),
        pywt.Wavelet(wavelet).dec_len
    )

    actual_level = min(level, max_level)

    coeffs = pywt.wavedec(
        train_values,
        wavelet=wavelet,
        level=actual_level,
        mode="symmetric"
    )

    detail_coeff = coeffs[-1]

    sigma = np.median(
        np.abs(detail_coeff)
    ) / 0.6745

    threshold = sigma * np.sqrt(
        2 * np.log(len(train_values))
    )

    return threshold, actual_level


def wavelet_denoise(values, wavelet, level, threshold):
    """
    使用给定的小波和阈值进行软阈值去噪。

    参数：
        values     : 一维价格序列
        wavelet    : 小波名称，例如 db4
        level      : 分解层数
        threshold  : 已经由训练集计算好的阈值
    """

    values = np.asarray(
        values,
        dtype=np.float64
    )

    max_level = pywt.dwt_max_level(
        len(values),
        pywt.Wavelet(wavelet).dec_len
    )

    actual_level = min(level, max_level)

    coeffs = pywt.wavedec(
        values,
        wavelet=wavelet,
        level=actual_level,
        mode="symmetric"
    )

    # 保留低频 Approximation，
    # 对 Detail coefficients 做软阈值处理
    denoised_coeffs = [coeffs[0]]

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

    # waverec 有时长度会比原序列多1个点
    reconstructed = reconstructed[:len(values)]

    # 防止出现 NaN / Inf
    if np.isnan(reconstructed).any():
        raise ValueError("小波去噪后出现 NaN。")

    if np.isinf(reconstructed).any():
        raise ValueError("小波去噪后出现 Inf。")

    return reconstructed


# ============================================================
# 8. 构造 LSTM 序列
# ============================================================

def create_train_sequences(scaled_values, past_days):
    """
    训练数据：

    [t-90, ..., t-1] -> t

    即使用过去90个交易日预测第91个交易日。
    """

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


def create_test_sequences(
    train_scaled,
    test_scaled,
    past_days
):
    """
    测试集需要使用：
    最后 past_days 个训练数据
    +
    全部测试数据

    这样第一个测试样本也可以拥有完整的90天历史窗口。
    """

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
# 9. 构建 LSTM
# ============================================================

def build_model():
    """
    与之前的 LSTM 实验保持一致：
    4层 LSTM，每层50 units
    Dropout = 0.2
    """

    model = Sequential()

    model.add(
        LSTM(
            50,
            return_sequences=True,
            input_shape=(PAST_DAYS, 1)
        )
    )
    model.add(Dropout(0.2))

    model.add(
        LSTM(
            50,
            return_sequences=True
        )
    )
    model.add(Dropout(0.2))

    model.add(
        LSTM(
            50,
            return_sequences=True
        )
    )
    model.add(Dropout(0.2))

    model.add(
        LSTM(
            50,
            return_sequences=False
        )
    )
    model.add(Dropout(0.2))

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
    """
    与之前实验保持一致：

    比较相邻预测值的变化方向：

    sign(y_true[t] - y_true[t-1])
    vs
    sign(y_pred[t] - y_pred[t-1])
    """

    true_diff = np.diff(y_true)
    pred_diff = np.diff(y_pred)

    true_direction = np.sign(true_diff)
    pred_direction = np.sign(pred_diff)

    accuracy = np.mean(
        true_direction == pred_direction
    )

    return accuracy


# ============================================================
# 11. 更严格的下一日方向准确率
# ============================================================

def calculate_next_day_direction_accuracy(
    current_prices,
    y_true,
    y_pred
):
    """
    更严格地计算：

    预测的下一日 Open 是否相对于当天 Open 上涨/下跌。

    Actual direction:
        y_true[t] - current_price[t]

    Predicted direction:
        y_pred[t] - current_price[t]

    这个指标额外保存，方便以后论文使用。

    它和之前的 Directional Accuracy 不完全相同。
    """

    true_diff = y_true - current_prices
    pred_diff = y_pred - current_prices

    true_direction = np.sign(true_diff)
    pred_direction = np.sign(pred_diff)

    accuracy = np.mean(
        true_direction == pred_direction
    )

    return accuracy


# ============================================================
# 12. 绘制 Loss 曲线
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

    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")

    plt.title(
        f"LSTM Training Loss - {experiment_name}"
    )

    plt.legend()
    plt.grid(True)

    save_path = os.path.join(
        OUTPUT_DIR,
        f"loss_{experiment_name}.png"
    )

    plt.tight_layout()
    plt.savefig(
        save_path,
        dpi=200
    )

    plt.close()

    print(f"Loss曲线已保存：{save_path}")


# ============================================================
# 13. 绘制去噪前后对比
# ============================================================

def save_denoising_plot(
    original_train,
    denoised_train,
    original_test,
    denoised_test,
    dates_train,
    dates_test,
    level
):
    plt.figure(
        figsize=(14, 6)
    )

    # 为了观察趋势，不使用全部点，直接全部绘制
    plt.plot(
        dates_train,
        original_train,
        label="Original Train",
        alpha=0.45
    )

    plt.plot(
        dates_train,
        denoised_train,
        label=f"Wavelet Denoised Train (db4-L{level})",
        linewidth=1.5
    )

    plt.plot(
        dates_test,
        original_test,
        label="Original Test",
        alpha=0.45
    )

    plt.plot(
        dates_test,
        denoised_test,
        label=f"Wavelet Denoised Test (db4-L{level})",
        linewidth=1.5
    )

    plt.xlabel("Date")
    plt.ylabel("Open Price")

    plt.title(
        f"Wavelet Denoising Comparison - db4 Level {level}"
    )

    plt.legend()
    plt.grid(True)

    plt.xticks(rotation=45)

    save_path = os.path.join(
        OUTPUT_DIR,
        f"denoising_comparison_DB4_L{level}.png"
    )

    plt.tight_layout()
    plt.savefig(
        save_path,
        dpi=200
    )

    plt.close()

    print(
        f"去噪前后对比图已保存：{save_path}"
    )


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

    print(
        f"预测结果已保存：{save_path}"
    )


# ============================================================
# 15. 运行单个实验
# ============================================================

def run_experiment(
    experiment_name,
    train_values,
    test_values,
    threshold=None
):
    """
    运行：
        数据 -> MinMaxScaler -> LSTM -> 预测

    train_values:
        训练集 Open / 去噪 Open

    test_values:
        测试集 Open / 去噪 Open
    """

    print()
    print("=" * 60)
    print(f"开始实验：{experiment_name}")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. 检查数据
    # --------------------------------------------------------

    train_values = np.asarray(
        train_values,
        dtype=np.float64
    )

    test_values = np.asarray(
        test_values,
        dtype=np.float64
    )

    if np.isnan(train_values).any():
        raise ValueError(
            f"{experiment_name} 的训练数据出现 NaN。"
        )

    if np.isnan(test_values).any():
        raise ValueError(
            f"{experiment_name} 的测试数据出现 NaN。"
        )

    # --------------------------------------------------------
    # 2. MinMaxScaler
    # --------------------------------------------------------

    scaler = MinMaxScaler(
        feature_range=(0, 1)
    )

    # 只能用训练集 fit
    train_scaled = scaler.fit_transform(
        train_values.reshape(-1, 1)
    )

    # 测试集只能 transform
    test_scaled = scaler.transform(
        test_values.reshape(-1, 1)
    )

    # --------------------------------------------------------
    # 3. 构造训练数据
    # --------------------------------------------------------

    X_train, y_train = create_train_sequences(
        train_scaled,
        PAST_DAYS
    )

    # --------------------------------------------------------
    # 4. 构造测试数据
    # --------------------------------------------------------

    X_test, y_test = create_test_sequences(
        train_scaled,
        test_scaled,
        PAST_DAYS
    )

    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")

    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    # --------------------------------------------------------
    # 5. 固定随机种子
    # --------------------------------------------------------

    set_seed(SEED)

    # --------------------------------------------------------
    # 6. 创建模型
    # --------------------------------------------------------

    model = build_model()

    print()
    model.summary()

    # --------------------------------------------------------
    # 7. 训练
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
    # 8. 保存 Loss
    # --------------------------------------------------------

    save_loss_plot(
        history,
        experiment_name
    )

    # --------------------------------------------------------
    # 9. 预测
    # --------------------------------------------------------

    predicted_scaled = model.predict(
        X_test,
        verbose=0
    )

    # --------------------------------------------------------
    # 10. 反归一化
    # --------------------------------------------------------

    y_pred = scaler.inverse_transform(
        predicted_scaled
    ).flatten()

    y_true = scaler.inverse_transform(
        y_test.reshape(-1, 1)
    ).flatten()

    # --------------------------------------------------------
    # 11. 计算 MAE / RMSE
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

    # --------------------------------------------------------
    # 12. Directional Accuracy
    # --------------------------------------------------------

    directional_accuracy = (
        calculate_directional_accuracy(
            y_true,
            y_pred
        )
    )

    # --------------------------------------------------------
    # 13. 更严格的 Next-Day Direction Accuracy
    # --------------------------------------------------------

    # 测试集当天 Open
    current_prices = test_values

    next_day_direction_accuracy = (
        calculate_next_day_direction_accuracy(
            current_prices,
            y_true,
            y_pred
        )
    )

    # --------------------------------------------------------
    # 14. 保存预测结果
    # --------------------------------------------------------

    # 预测日期对应 test_df 的全部日期
    prediction_dates = test_dates

    save_prediction_csv(
        prediction_dates,
        y_true,
        y_pred,
        experiment_name
    )

    # --------------------------------------------------------
    # 15. 打印结果
    # --------------------------------------------------------

    print()
    print("-" * 60)
    print(f"{experiment_name} 实验结果")
    print("-" * 60)

    print(
        f"MAE  : {mae:.4f}"
    )

    print(
        f"RMSE : {rmse:.4f}"
    )

    print(
        f"Directional Accuracy : "
        f"{directional_accuracy:.4f} "
        f"({directional_accuracy * 100:.2f}%)"
    )

    print(
        f"Next-Day Direction Accuracy : "
        f"{next_day_direction_accuracy:.4f} "
        f"({next_day_direction_accuracy * 100:.2f}%)"
    )

    print("-" * 60)

    return {
        "Experiment": experiment_name,
        "Wavelet": "None",
        "Level": 0,
        "MAE": mae,
        "RMSE": rmse,
        "Directional_Accuracy": directional_accuracy,
        "Next_Day_Directional_Accuracy": (
            next_day_direction_accuracy
        ),
        "Test_Samples": len(y_true)
    }


# ============================================================
# 16. 主函数
# ============================================================

def main():

    global test_dates

    set_seed(SEED)

    # --------------------------------------------------------
    # 读取数据
    # --------------------------------------------------------

    df = load_data()

    # --------------------------------------------------------
    # 划分训练/测试
    # --------------------------------------------------------

    train_df, test_df = split_data(df)

    train_open = train_df[
        TARGET_COLUMN
    ].values.astype(float)

    test_open = test_df[
        TARGET_COLUMN
    ].values.astype(float)

    # 测试日期
    test_dates = test_df[
        "date"
    ].dt.strftime("%Y-%m-%d").values

    # --------------------------------------------------------
    # RAW baseline
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("实验 1：RAW Open + LSTM")
    print("=" * 60)

    raw_result = run_experiment(
        experiment_name="RAW",
        train_values=train_open,
        test_values=test_open
    )

    raw_result["Wavelet"] = "None"
    raw_result["Level"] = 0

    all_results = [
        raw_result
    ]

    # ========================================================
    # 小波实验
    # ========================================================

    for level in WAVELET_LEVELS:

        print()
        print("=" * 60)
        print(
            f"开始小波去噪："
            f"{WAVELET} Level {level}"
        )
        print("=" * 60)

        # ----------------------------------------------------
        # 1. 根据训练集估计 threshold
        # ----------------------------------------------------

        threshold, actual_level = (
            estimate_threshold(
                train_open,
                WAVELET,
                level
            )
        )

        print(
            f"Wavelet = {WAVELET}"
        )

        print(
            f"Requested Level = {level}"
        )

        print(
            f"Actual Level = {actual_level}"
        )

        print(
            f"Threshold = {threshold:.6f}"
        )

        # ----------------------------------------------------
        # 2. 对训练集做小波去噪
        # ----------------------------------------------------

        denoised_train = wavelet_denoise(
            train_open,
            wavelet=WAVELET,
            level=level,
            threshold=threshold
        )

        # ----------------------------------------------------
        # 3. 对测试集做小波去噪
        # ----------------------------------------------------

        denoised_test = wavelet_denoise(
            test_open,
            wavelet=WAVELET,
            level=level,
            threshold=threshold
        )

        # ----------------------------------------------------
        # 4. 保存去噪对比图
        # ----------------------------------------------------

        save_denoising_plot(
            original_train=train_open,
            denoised_train=denoised_train,
            original_test=test_open,
            denoised_test=denoised_test,
            dates_train=train_df["date"],
            dates_test=test_df["date"],
            level=level
        )

        # ----------------------------------------------------
        # 5. LSTM
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

        all_results.append(result)

    # ========================================================
    # 保存总实验结果
    # ========================================================

    results_df = pd.DataFrame(
        all_results
    )

    # 调整列顺序
    results_df = results_df[
        [
            "Experiment",
            "Wavelet",
            "Level",
            "MAE",
            "RMSE",
            "Directional_Accuracy",
            "Next_Day_Directional_Accuracy",
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
    # 打印最终结果
    # ========================================================

    print()
    print()
    print("=" * 80)
    print("所有实验完成")
    print("=" * 80)

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    print()
    print(
        f"实验结果已保存：{result_path}"
    )

    print()
    print("=" * 80)
    print("建议重点比较：")
    print("=" * 80)

    print(
        "1. RAW vs DB4_L2"
    )

    print(
        "2. RAW vs DB4_L4"
    )

    print(
        "3. MAE / RMSE 是否下降"
    )

    print(
        "4. Directional Accuracy 是否提升"
    )

    print(
        "5. Level 2 和 Level 4 哪个效果更好"
    )

    print("=" * 80)


# ============================================================
# 17. 程序入口
# ============================================================

if __name__ == "__main__":
    main()