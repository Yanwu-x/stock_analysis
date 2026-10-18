# -*- coding: utf-8 -*-

"""
600519 资金流向特征 + LSTM 可行性实验

实验：
1. RAW_20
   Open

2. MAIN_FLOW_20
   Open + 主力净流入

3. ALL_FLOW_20
   Open
   + 主力净流入
   + 超大单净流入
   + 大单净流入
   + 中单净流入
   + 小单净流入

说明：
由于 efinance 当前获得的资金流向历史数据较短，
本实验作为“资金流向特征可行性实验”。

窗口：
过去20个交易日 -> 下一交易日 Open

评价：
MAE
RMSE
Directional Accuracy
"""

import os
import random
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import efinance as ef

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
# 1. 参数
# ============================================================

STOCK_CODE = "600519"

STOCK_DATA_FILE = "ashare_600519.csv"

OUTPUT_DIR = "results_fundflow_lstm"

TRAIN_RATIO = 0.8

# 注意：
# 资金流向只有约115个有效样本，
# 所以使用20天窗口，而不是90天
PAST_DAYS = 20

EPOCHS = 100

BATCH_SIZE = 16

SEED = 42


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
# 3. 输出目录
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 4. 获取资金流向
# ============================================================

def get_fund_flow():

    print("=" * 70)
    print("获取 efinance 资金流向数据")
    print("=" * 70)

    df = ef.stock.get_history_bill(
        STOCK_CODE
    )

    if df is None or len(df) == 0:

        raise ValueError(
            "没有获取到资金流向数据。"
        )

    print(
        f"获取到 {len(df)} 条资金流向数据"
    )

    print(
        "字段："
    )

    print(
        df.columns.tolist()
    )

    return df


# ============================================================
# 5. 读取股票价格
# ============================================================

def load_stock():

    print()
    print("=" * 70)
    print("读取股票日线数据")
    print("=" * 70)

    if not os.path.exists(
        STOCK_DATA_FILE
    ):

        raise FileNotFoundError(
            f"找不到：{STOCK_DATA_FILE}"
        )

    df = pd.read_csv(
        STOCK_DATA_FILE
    )

    required = [
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]

    for col in required:

        if col not in df.columns:

            raise ValueError(
                f"股票数据缺少字段：{col}"
            )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    df["open"] = pd.to_numeric(
        df["open"],
        errors="coerce"
    )

    df["high"] = pd.to_numeric(
        df["high"],
        errors="coerce"
    )

    df["low"] = pd.to_numeric(
        df["low"],
        errors="coerce"
    )

    df["close"] = pd.to_numeric(
        df["close"],
        errors="coerce"
    )

    df["volume"] = pd.to_numeric(
        df["volume"],
        errors="coerce"
    )

    df = (
        df.dropna(
            subset=[
                "date",
                "open"
            ]
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    return df


# ============================================================
# 6. 合并数据
# ============================================================

def prepare_data(
    fund_df,
    stock_df
):

    print()
    print("=" * 70)
    print("合并资金流向与股票数据")
    print("=" * 70)

    fund_df = fund_df.copy()

    # --------------------------------------------------------
    # 日期
    # --------------------------------------------------------

    fund_df["日期"] = pd.to_datetime(
        fund_df["日期"],
        errors="coerce"
    )

    fund_df = fund_df.dropna(
        subset=["日期"]
    )

    fund_df = fund_df.rename(
        columns={
            "日期": "date"
        }
    )

    # --------------------------------------------------------
    # 要使用的资金流字段
    # --------------------------------------------------------

    flow_columns = [
        "主力净流入",
        "超大单净流入",
        "大单净流入",
        "中单净流入",
        "小单净流入"
    ]

    missing = [
        col
        for col in flow_columns
        if col not in fund_df.columns
    ]

    if missing:

        raise ValueError(
            f"缺少资金流字段：{missing}"
        )

    # --------------------------------------------------------
    # 数值化
    # --------------------------------------------------------

    for col in flow_columns:

        fund_df[col] = pd.to_numeric(
            fund_df[col],
            errors="coerce"
        )

    fund_df = fund_df[
        ["date"] + flow_columns
    ]

    # --------------------------------------------------------
    # 股票数据
    # --------------------------------------------------------

    stock = stock_df[
        [
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume"
        ]
    ].copy()

    # --------------------------------------------------------
    # 合并
    # --------------------------------------------------------

    merged = pd.merge(
        stock,
        fund_df,
        on="date",
        how="inner"
    )

    merged = (
        merged
        .sort_values("date")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # 删除资金流缺失
    # --------------------------------------------------------

    merged = merged.dropna(
        subset=flow_columns
    ).reset_index(drop=True)

    print(
        f"股票数据：{len(stock)} 条"
    )

    print(
        f"资金流向：{len(fund_df)} 条"
    )

    print(
        f"最终有效匹配：{len(merged)} 条"
    )

    if len(merged) <= PAST_DAYS + 20:

        raise ValueError(
            f"有效资金流向数据太少：{len(merged)} 条，"
            f"不足以进行 {PAST_DAYS} 天窗口实验。"
        )

    print(
        f"时间范围："
        f"{merged['date'].iloc[0].date()} "
        f"-> "
        f"{merged['date'].iloc[-1].date()}"
    )

    return merged


# ============================================================
# 7. 构造序列
# ============================================================

def create_sequences(
    feature_values,
    target_values,
    past_days
):

    X = []
    y = []

    for i in range(
        past_days,
        len(feature_values)
    ):

        # 过去20天特征
        X.append(
            feature_values[
                i - past_days:i
            ]
        )

        # 第i天的 Open
        # 即使用前20个交易日预测下一交易日
        y.append(
            target_values[i]
        )

    return (
        np.asarray(X),
        np.asarray(y)
    )


# ============================================================
# 8. 构建 LSTM
# ============================================================

def build_model(
    num_features
):

    model = Sequential()

    model.add(
        LSTM(
            50,
            return_sequences=True,
            input_shape=(
                PAST_DAYS,
                num_features
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
# 9. Directional Accuracy
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
# 10. 单个 LSTM 实验
# ============================================================

def run_experiment(
    df,
    experiment_name,
    feature_columns
):

    print()
    print("=" * 80)
    print(
        f"开始实验：{experiment_name}"
    )
    print("=" * 80)

    print(
        f"输入特征：{feature_columns}"
    )

    data = df[
        ["date"] + feature_columns
    ].copy()

    # --------------------------------------------------------
    # 目标：下一交易日 Open
    # --------------------------------------------------------

    # 这里先获取每一天的 open
    target = df["open"].values.astype(float)

    # --------------------------------------------------------
    # 80/20 时间划分
    # --------------------------------------------------------

    split_index = int(
        len(data) * TRAIN_RATIO
    )

    train_data = data.iloc[
        :split_index
    ].copy()

    test_data = data.iloc[
        split_index:
    ].copy()

    train_target = target[
        :split_index
    ]

    test_target = target[
        split_index:
    ]

    # --------------------------------------------------------
    # 特征 scaler
    # --------------------------------------------------------

    feature_scaler = MinMaxScaler()

    train_features = (
        feature_scaler
        .fit_transform(
            train_data[
                feature_columns
            ]
        )
    )

    test_features = (
        feature_scaler
        .transform(
            test_data[
                feature_columns
            ]
        )
    )

    # --------------------------------------------------------
    # Target scaler
    # --------------------------------------------------------

    target_scaler = MinMaxScaler()

    train_target_scaled = (
        target_scaler
        .fit_transform(
            train_target.reshape(-1, 1)
        )
        .flatten()
    )

    test_target_scaled = (
        target_scaler
        .transform(
            test_target.reshape(-1, 1)
        )
        .flatten()
    )

    # --------------------------------------------------------
    # 构造训练序列
    # --------------------------------------------------------

    X_train, y_train = (
        create_sequences(
            train_features,
            train_target_scaled,
            PAST_DAYS
        )
    )

    # --------------------------------------------------------
    # 测试序列
    #
    # 第一个测试样本需要使用训练集最后20天作为历史
    # --------------------------------------------------------

    combined_features = np.concatenate(
        [
            train_features[-PAST_DAYS:],
            test_features
        ],
        axis=0
    )

    combined_target = np.concatenate(
        [
            train_target_scaled[-PAST_DAYS:],
            test_target_scaled
        ]
    )

    X_test = []
    y_test = []

    for i in range(
        PAST_DAYS,
        len(combined_features)
    ):

        X_test.append(
            combined_features[
                i - PAST_DAYS:i
            ]
        )

        y_test.append(
            combined_target[i]
        )

    X_test = np.asarray(
        X_test
    )

    y_test = np.asarray(
        y_test
    )

    print(
        f"X_train shape = "
        f"{X_train.shape}"
    )

    print(
        f"X_test shape = "
        f"{X_test.shape}"
    )

    # --------------------------------------------------------
    # 固定种子
    # --------------------------------------------------------

    set_seed(SEED)

    # --------------------------------------------------------
    # 模型
    # --------------------------------------------------------

    model = build_model(
        len(feature_columns)
    )

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
    # Loss 图
    # --------------------------------------------------------

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
        f"LSTM Loss - {experiment_name}"
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    loss_path = os.path.join(
        OUTPUT_DIR,
        f"loss_{experiment_name}.png"
    )

    plt.savefig(
        loss_path,
        dpi=200
    )

    plt.close()

    # --------------------------------------------------------
    # 预测
    # --------------------------------------------------------

    pred_scaled = model.predict(
        X_test,
        verbose=0
    )

    # --------------------------------------------------------
    # 反归一化
    # --------------------------------------------------------

    y_pred = (
        target_scaler
        .inverse_transform(
            pred_scaled
        )
        .flatten()
    )

    y_true = (
        target_scaler
        .inverse_transform(
            y_test.reshape(-1, 1)
        )
        .flatten()
    )

    # --------------------------------------------------------
    # 指标
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
    # 保存预测
    # --------------------------------------------------------

    prediction_df = pd.DataFrame(
        {
            "date": test_data[
                "date"
            ].dt.strftime(
                "%Y-%m-%d"
            ).values,
            "actual_open": y_true,
            "predicted_open": y_pred
        }
    )

    prediction_path = os.path.join(
        OUTPUT_DIR,
        f"predictions_{experiment_name}.csv"
    )

    prediction_df.to_csv(
        prediction_path,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # 打印
    # --------------------------------------------------------

    print()
    print(
        f"{experiment_name} 结果"
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
        "Experiment": experiment_name,
        "Feature_Count": len(
            feature_columns
        ),
        "Features": "+".join(
            feature_columns
        ),
        "Past_Days": PAST_DAYS,
        "MAE": mae,
        "RMSE": rmse,
        "Directional_Accuracy": da,
        "Test_Samples": len(y_true)
    }


# ============================================================
# 11. 主函数
# ============================================================

def main():

    print()
    print("=" * 90)
    print("600519 资金流向 + LSTM 可行性实验")
    print("=" * 90)

    # --------------------------------------------------------
    # 获取数据
    # --------------------------------------------------------

    fund_df = get_fund_flow()

    stock_df = load_stock()

    df = prepare_data(
        fund_df,
        stock_df
    )

    # --------------------------------------------------------
    # 保存最终实验数据
    # --------------------------------------------------------

    merged_path = os.path.join(
        OUTPUT_DIR,
        "fundflow_lstm_dataset.csv"
    )

    df.to_csv(
        merged_path,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # 三组实验
    # --------------------------------------------------------

    experiments = [

        (
            "RAW_20",
            [
                "open"
            ]
        ),

        (
            "MAIN_FLOW_20",
            [
                "open",
                "主力净流入"
            ]
        ),

        (
            "ALL_FLOW_20",
            [
                "open",
                "主力净流入",
                "超大单净流入",
                "大单净流入",
                "中单净流入",
                "小单净流入"
            ]
        )

    ]

    results = []

    for experiment_name, feature_columns in experiments:

        result = run_experiment(
            df=df,
            experiment_name=experiment_name,
            feature_columns=feature_columns
        )

        results.append(
            result
        )

    # --------------------------------------------------------
    # 汇总
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    results_path = os.path.join(
        OUTPUT_DIR,
        "fundflow_lstm_results.csv"
    )

    results_df.to_csv(
        results_path,
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print()
    print("=" * 100)
    print("资金流向 + LSTM 实验完成")
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

    print("=" * 100)


if __name__ == "__main__":
    main()