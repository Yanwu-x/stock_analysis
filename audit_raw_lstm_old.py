# -*- coding: utf-8 -*-

"""
A股股票预测：LSTM 特征消融实验

数据：
    ashare_600519.csv

字段：
    date
    open
    high
    low
    close
    volume

实验目标：
    使用过去 PAST_DAYS 个交易日的信息
    预测下一交易日 Open

数据划分：
    前 80%：训练集
    后 20%：测试集

FEATURE_MODE：
    OPEN
    OHLCV
    MA5
    MA
    RSI
    MACD
    TECHNICAL
    ALL
"""

import os
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras import backend as K

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)


# ============================================================
# 0. 参数
# ============================================================

DATA_FILE = "ashare_600519.csv"

PAST_DAYS = 90
EPOCHS = 100
BATCH_SIZE = 32

TRAIN_RATIO = 0.80

SEED = 42

RESULT_DIR = "results_raw_lstm_audit"

# 一键运行全部实验
FEATURE_MODE = "OPEN"

# 是否显示图像
SHOW_PLOT = False


# ============================================================
# 1. 固定随机种子
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
# 2. 读取数据
# ============================================================

def load_data():

    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(
            f"找不到数据文件：{DATA_FILE}\n"
            "请确认 CSV 和 Python 文件在同一个文件夹。"
        )

    df = pd.read_csv(DATA_FILE)

    print("\n原始数据：")
    print(df.head())

    print("\n数据大小：")
    print(df.shape)

    print("\n字段：")
    print(df.columns.tolist())

    # 日期
    df["date"] = pd.to_datetime(df["date"])

    # 按时间排序
    df = df.sort_values("date").reset_index(drop=True)

    # 检查重复日期
    if df["date"].duplicated().any():
        print("警告：存在重复日期。")

    # 检查基础字段
    required = [
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]

    for col in required:
        if col not in df.columns:
            raise ValueError(
                f"缺少字段：{col}"
            )

    # 检查原始 NaN
    if df[required].isna().any().any():
        raise ValueError(
            "OHLCV 原始数据存在 NaN，请先检查 CSV。"
        )

    return df


# ============================================================
# 3. 技术指标
# ============================================================

def add_technical_indicators(df):

    df = df.copy()

    # -------------------------
    # MA5
    # -------------------------

    df["MA5"] = (
        df["close"]
        .rolling(
            window=5,
            min_periods=5
        )
        .mean()
    )

    # -------------------------
    # MA20
    # -------------------------

    df["MA20"] = (
        df["close"]
        .rolling(
            window=20,
            min_periods=20
        )
        .mean()
    )

    # -------------------------
    # RSI 14
    # -------------------------

    delta = df["close"].diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

    avg_gain = (
        gain
        .rolling(
            window=14,
            min_periods=14
        )
        .mean()
    )

    avg_loss = (
        loss
        .rolling(
            window=14,
            min_periods=14
        )
        .mean()
    )

    rs = avg_gain / (
        avg_loss + 1e-8
    )

    df["RSI"] = (
        100
        - 100 / (1 + rs)
    )

    # -------------------------
    # MACD
    # -------------------------

    ema12 = (
        df["close"]
        .ewm(
            span=12,
            adjust=False
        )
        .mean()
    )

    ema26 = (
        df["close"]
        .ewm(
            span=26,
            adjust=False
        )
        .mean()
    )

    df["MACD"] = ema12 - ema26

    return df


# ============================================================
# 4. 特征组合
# ============================================================

FEATURE_CONFIG = {

    "OPEN": [
        "open"
    ],

    "OHLCV": [
        "open",
        "high",
        "low",
        "close",
        "volume"
    ],

    "MA5": [
        "open",
        "MA5"
    ],

    "MA": [
        "open",
        "MA5",
        "MA20"
    ],

    "RSI": [
        "open",
        "RSI"
    ],

    "MACD": [
        "open",
        "MACD"
    ],

    "TECHNICAL": [
        "open",
        "MA5",
        "MA20",
        "RSI",
        "MACD"
    ]
}


ALL_MODES = [
    "OPEN",
    "OHLCV",
    "MA5",
    "MA",
    "RSI",
    "MACD",
    "TECHNICAL"
]


def get_features(mode):

    mode = mode.upper()

    if mode not in FEATURE_CONFIG:

        raise ValueError(
            f"未知 FEATURE_MODE：{mode}\n"
            f"可选模式："
            f"{list(FEATURE_CONFIG.keys())}"
        )

    return FEATURE_CONFIG[mode]


# ============================================================
# 5. 构建 LSTM
# ============================================================

def build_lstm(input_shape):

    model = Sequential()

    model.add(
        Input(
            shape=input_shape
        )
    )

    # Layer 1
    model.add(
        LSTM(
            units=50,
            return_sequences=True
        )
    )

    model.add(
        Dropout(0.2)
    )

    # Layer 2
    model.add(
        LSTM(
            units=50,
            return_sequences=True
        )
    )

    model.add(
        Dropout(0.2)
    )

    # Layer 3
    model.add(
        LSTM(
            units=50,
            return_sequences=True
        )
    )

    model.add(
        Dropout(0.2)
    )

    # Layer 4
    model.add(
        LSTM(
            units=50
        )
    )

    model.add(
        Dropout(0.2)
    )

    # Output
    model.add(
        Dense(
            units=1
        )
    )

    model.compile(
        optimizer="adam",
        loss="mean_squared_error"
    )

    return model


# ============================================================
# 6. 创建训练集序列
# ============================================================

def create_train_sequences(
    train_df,
    features
):

    feature_scaler = MinMaxScaler(
        feature_range=(0, 1)
    )

    target_scaler = MinMaxScaler(
        feature_range=(0, 1)
    )

    feature_data = (
        train_df[features]
        .values
        .astype(np.float64)
    )

    target_data = (
        train_df[["open"]]
        .values
        .astype(np.float64)
    )

    # --------------------------------------------------------
    # 因为技术指标最前面存在 NaN，
    # 所以 scaler 只使用 PAST_DAYS 之后的有效数据 fit。
    # --------------------------------------------------------
    # ⭐ 新增：删除 feature_data 中包含 NaN 的行，同时删除 target_data 中对应的行
    nan_mask = ~np.isnan(feature_data).any(axis=1)
    feature_data = feature_data[nan_mask]
    target_data = target_data[nan_mask]

    # ⭐ 新增：检查是否还有 NaN
    if np.isnan(feature_data).any():
        raise ValueError("feature_data 中仍然存在 NaN")
    if np.isnan(target_data).any():
        raise ValueError("target_data 中仍然存在 NaN")
    
    # ⭐ 新增：检查数据量是否足够
    if len(feature_data) <= PAST_DAYS:
        raise ValueError(f"过滤后数据仅 {len(feature_data)} 行，少于 PAST_DAYS ({PAST_DAYS})")

    scaler_feature_data = (
        feature_data[PAST_DAYS:]
    )

    if not np.isfinite(
        scaler_feature_data
    ).all():

        raise ValueError(
            f"{features} 在训练数据中存在 "
            "NaN/Inf。"
        )

    feature_scaler.fit(
        scaler_feature_data
    )

    target_scaler.fit(
        target_data[PAST_DAYS:]
    )

    feature_scaled = (
        feature_scaler
        .transform(feature_data)
    )

    target_scaled = (
        target_scaler
        .transform(target_data)
    )

    X_train = []
    y_train = []

    # 从 PAST_DAYS 开始
    for i in range(
        PAST_DAYS,
        len(feature_data)
    ):

        X_train.append(
            feature_scaled[
                i - PAST_DAYS:i,
                :
            ]
        )

        y_train.append(
            target_scaled[i, 0]
        )

    X_train = np.asarray(
        X_train,
        dtype=np.float32
    )

    y_train = np.asarray(
        y_train,
        dtype=np.float32
    )

    # 检查
    if not np.isfinite(
        X_train
    ).all():

        raise ValueError(
            "X_train 中存在 NaN/Inf。"
        )

    if not np.isfinite(
        y_train
    ).all():

        raise ValueError(
            "y_train 中存在 NaN/Inf。"
        )

    return (
        X_train,
        y_train,
        feature_scaler,
        target_scaler
    )


# ============================================================
# 7. 创建测试集序列
# ============================================================

def create_test_sequences(
    train_df,
    test_df,
    features,
    feature_scaler
):

    # 合并 train/test
    combined = pd.concat(
        [
            train_df,
            test_df
        ],
        axis=0
    ).reset_index(
        drop=True
    )

    feature_data = (
        combined[features]
        .values
        .astype(np.float64)
    )

    feature_scaled = (
        feature_scaler
        .transform(feature_data)
    )

    test_start = len(
        train_df
    )

    X_test = []

    for i in range(
        test_start,
        test_start + len(test_df)
    ):

        X_test.append(
            feature_scaled[
                i - PAST_DAYS:i,
                :
            ]
        )

    X_test = np.asarray(
        X_test,
        dtype=np.float32
    )

    if not np.isfinite(
        X_test
    ).all():

        raise ValueError(
            "X_test 中存在 NaN/Inf。"
        )

    real_open = (
        test_df[["open"]]
        .values
        .astype(np.float64)
    )

    dates = test_df[
        "date"
    ].values

    return (
        X_test,
        real_open,
        dates
    )


# ============================================================
# 8. Directional Accuracy
# ============================================================

def calculate_directional_accuracy(
    real_price,
    predicted_price
):

    real = real_price.flatten()

    pred = predicted_price.flatten()

    real_diff = np.diff(real)

    pred_diff = np.diff(pred)

    if len(real_diff) == 0:
        return np.nan

    return np.mean(
        (
            real_diff
            * pred_diff
        ) > 0
    )


# ============================================================
# 9. 保存结果
# ============================================================

def save_results(
    mode,
    features,
    mae,
    rmse,
    da,
    real_price,
    predicted_price,
    dates,
    history
):

    os.makedirs(
        RESULT_DIR,
        exist_ok=True
    )

    # -------------------------
    # 指标汇总
    # -------------------------

    result_file = os.path.join(
        RESULT_DIR,
        "experiment_results.csv"
    )

    result = pd.DataFrame([
        {
            "Feature_Mode": mode,
            "Features": "+".join(features),
            "Past_Days": PAST_DAYS,
            "Epochs": EPOCHS,
            "Batch_Size": BATCH_SIZE,
            "Train_Ratio": TRAIN_RATIO,
            "Seed": SEED,
            "MAE": mae,
            "RMSE": rmse,
            "Directional_Accuracy": da,
            "Test_Samples": len(real_price)
        }
    ])

    if os.path.exists(
        result_file
    ):

        result.to_csv(
            result_file,
            mode="a",
            header=False,
            index=False
        )

    else:

        result.to_csv(
            result_file,
            index=False
        )

    # -------------------------
    # 预测结果
    # -------------------------

    prediction_file = os.path.join(
        RESULT_DIR,
        f"prediction_{mode}.csv"
    )

    prediction_df = pd.DataFrame({

        "date":
            pd.to_datetime(dates),

        "real_open":
            real_price.flatten(),

        "predicted_open":
            predicted_price.flatten()
    })

    prediction_df.to_csv(
        prediction_file,
        index=False
    )

    # -------------------------
    # Loss
    # -------------------------

    loss_file = os.path.join(
        RESULT_DIR,
        f"loss_{mode}.csv"
    )

    loss_df = pd.DataFrame({

        "epoch":
            np.arange(
                1,
                len(
                    history.history["loss"]
                ) + 1
            ),

        "loss":
            history.history["loss"]
    })

    loss_df.to_csv(
        loss_file,
        index=False
    )


# ============================================================
# 10. 保存预测图
# ============================================================

def plot_prediction(
    mode,
    real_price,
    predicted_price,
    dates
):

    os.makedirs(
        RESULT_DIR,
        exist_ok=True
    )

    plt.figure(
        figsize=(12, 6)
    )

    plt.plot(
        pd.to_datetime(dates),
        real_price.flatten(),
        label="Real Open"
    )

    plt.plot(
        pd.to_datetime(dates),
        predicted_price.flatten(),
        label="Predicted Open"
    )

    plt.title(
        f"600519 {mode} LSTM "
        f"(PAST_DAYS={PAST_DAYS})"
    )

    plt.xlabel(
        "Date"
    )

    plt.ylabel(
        "Open Price"
    )

    plt.legend()

    plt.xticks(
        rotation=45
    )

    plt.tight_layout()

    file_path = os.path.join(
        RESULT_DIR,
        f"plot_{mode}.png"
    )

    plt.savefig(
        file_path,
        dpi=200
    )

    if SHOW_PLOT:

        plt.show()

    else:

        plt.close()


# ============================================================
# 11. 单个实验
# ============================================================

def run_experiment(
    mode,
    train_df,
    test_df
):

    print("\n")
    print("=" * 70)
    print(f"开始实验：{mode}")
    print("=" * 70)

    # 每组实验使用相同随机种子
    set_seed(SEED)

    # 清空上一个 Keras 模型
    K.clear_session()

    features = get_features(
        mode
    )

    print(
        "输入特征：",
        features
    )

    # -------------------------
    # 训练序列
    # -------------------------

    (
        X_train,
        y_train,
        feature_scaler,
        target_scaler
    ) = create_train_sequences(
        train_df,
        features
    )

    print(
        "X_train shape:",
        X_train.shape
    )

    print(
        "y_train shape:",
        y_train.shape
    )

    # -------------------------
    # 模型
    # -------------------------

    model = build_lstm(
        (
            X_train.shape[1],
            X_train.shape[2]
        )
    )

    # -------------------------
    # 训练
    # -------------------------

    history = model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
        shuffle=False
    )

    # -------------------------
    # 测试集
    # -------------------------

    (
        X_test,
        real_price,
        dates
    ) = create_test_sequences(
        train_df,
        test_df,
        features,
        feature_scaler
    )

    print(
        "X_test shape:",
        X_test.shape
    )

    # -------------------------
    # 预测
    # -------------------------

    predicted_scaled = (
        model.predict(
            X_test,
            verbose=0
        )
    )

    if not np.isfinite(
        predicted_scaled
    ).all():

        raise ValueError(
            "LSTM 输出出现 NaN/Inf。"
        )

    predicted_price = (
        target_scaler
        .inverse_transform(
            predicted_scaled
        )
    )

    # -------------------------
    # Metrics
    # -------------------------

    mae = mean_absolute_error(
        real_price,
        predicted_price
    )

    rmse = np.sqrt(
        mean_squared_error(
            real_price,
            predicted_price
        )
    )

    da = calculate_directional_accuracy(
        real_price,
        predicted_price
    )

    print("\n")
    print("-" * 60)
    print(f"{mode} 实验结果")
    print("-" * 60)

    print(
        f"MAE:  {mae:.4f}"
    )

    print(
        f"RMSE: {rmse:.4f}"
    )

    print(
        f"Directional Accuracy: {da:.4f}"
    )

    # -------------------------
    # 保存
    # -------------------------

    save_results(
        mode=mode,
        features=features,
        mae=mae,
        rmse=rmse,
        da=da,
        real_price=real_price,
        predicted_price=predicted_price,
        dates=dates,
        history=history
    )

    # -------------------------
    # 图
    # -------------------------

    plot_prediction(
        mode,
        real_price,
        predicted_price,
        dates
    )

    return {
        "Feature_Mode": mode,
        "Features": "+".join(features),
        "MAE": mae,
        "RMSE": rmse,
        "Directional_Accuracy": da
    }


# ============================================================
# 12. 主程序
# ============================================================

def main():

    print("=" * 70)
    print("600519 贵州茅台 LSTM 特征消融实验")
    print("=" * 70)

    # -------------------------
    # 读取数据
    # -------------------------

    df = load_data()

    # -------------------------
    # 计算指标
    # -------------------------

    df = add_technical_indicators(
        df
    )

    print("\n技术指标计算完成：")
    print(
        "MA5 / MA20 / RSI / MACD"
    )

    # -------------------------
    # 80% / 20%
    # -------------------------

    split_index = int(
        len(df) * TRAIN_RATIO
    )

    train_df = df[
        :split_index
    ].copy()

    test_df = df[
        split_index:
    ].copy()

    print("\n数据划分：")

    print(
        f"总数据：{len(df)}"
    )

    print(
        f"训练集：{len(train_df)}"
    )

    print(
        f"测试集：{len(test_df)}"
    )

    print(
        f"训练起始：{train_df['date'].iloc[0]}"
    )

    print(
        f"训练结束：{train_df['date'].iloc[-1]}"
    )

    print(
        f"测试起始：{test_df['date'].iloc[0]}"
    )

    print(
        f"测试结束：{test_df['date'].iloc[-1]}"
    )

    # -------------------------
    # 实验列表
    # -------------------------

    if FEATURE_MODE.upper() == "ALL":

        modes = ALL_MODES

    else:

        modes = [
            FEATURE_MODE.upper()
        ]

    print("\n本次实验：")

    for mode in modes:

        print(
            f"{mode}: "
            f"{get_features(mode)}"
        )

    # -------------------------
    # 运行
    # -------------------------

    all_results = []

    for mode in modes:

        result = run_experiment(
            mode,
            train_df,
            test_df
        )

        all_results.append(
            result
        )

    # -------------------------
    # 汇总
    # -------------------------

    summary = pd.DataFrame(
        all_results
    )

    print("\n")
    print("=" * 90)
    print("全部实验完成")
    print("=" * 90)

    print(
        summary.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}"
        )
    )

    summary.to_csv(
        os.path.join(
            RESULT_DIR,
            "latest_summary.csv"
        ),
        index=False
    )

    print(
        "\n结果保存目录：",
        os.path.abspath(
            RESULT_DIR
        )
    )


# ============================================================
# 13. Entry
# ============================================================

if __name__ == "__main__":

    main()