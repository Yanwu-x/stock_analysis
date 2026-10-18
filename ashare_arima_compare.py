# -*- coding: utf-8 -*-

"""
A-share Stock Price Prediction
ARIMA Parameter Comparison

比较：
ARIMA(1,1,0)
ARIMA(3,1,0)
ARIMA(5,1,0)
ARIMA(7,1,0)
ARIMA(5,1,1)

数据：
ashare_600519.csv

划分：
前80% Train
后20% Test

评价：
MAE
RMSE
Directional Accuracy
"""

import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

from statsmodels.tsa.arima.model import ARIMA


# ============================================================
# 1. 参数
# ============================================================

DATA_FILE = "ashare_600519.csv"

OUTPUT_DIR = "results_arima_compare"

TRAIN_RATIO = 0.8

# 本次比较的 ARIMA 参数
ARIMA_ORDERS = [
    (1, 1, 0),
    (3, 1, 0),
    (5, 1, 0),
    (7, 1, 0),
    (5, 1, 1)
]

TARGET_COLUMN = "open"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 2. 读取数据
# ============================================================

def load_data():

    if not os.path.exists(DATA_FILE):

        raise FileNotFoundError(
            f"找不到数据文件：{DATA_FILE}"
        )

    df = pd.read_csv(DATA_FILE)

    required = [
        "date",
        "open"
    ]

    for col in required:

        if col not in df.columns:

            raise ValueError(
                f"缺少字段：{col}"
            )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    df["open"] = pd.to_numeric(
        df["open"],
        errors="coerce"
    )

    df = (
        df.dropna(
            subset=["date", "open"]
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    return df


# ============================================================
# 3. Directional Accuracy
# ============================================================

def calculate_directional_accuracy(
    y_true,
    y_pred
):

    true_diff = np.diff(y_true)

    pred_diff = np.diff(y_pred)

    true_direction = np.sign(
        true_diff
    )

    pred_direction = np.sign(
        pred_diff
    )

    return np.mean(
        true_direction == pred_direction
    )


# ============================================================
# 4. 滚动 ARIMA 预测
# ============================================================

def rolling_forecast(
    train_values,
    test_values,
    order
):

    history = list(
        train_values
    )

    predictions = []

    total = len(test_values)

    print()
    print(
        f"ARIMA{order} 开始预测，"
        f"测试样本={total}"
    )

    for i in range(total):

        try:

            model = ARIMA(
                history,
                order=order
            )

            fitted_model = model.fit()

            forecast = fitted_model.forecast(
                steps=1
            )

            prediction = float(
                forecast[0]
            )

        except Exception as e:

            print(
                f"\nARIMA{order} 第 {i+1} 个样本预测失败：{e}"
            )

            prediction = history[-1]

        predictions.append(
            prediction
        )

        # 加入真实值，继续滚动
        history.append(
            float(test_values[i])
        )

        if (
            (i + 1) % 25 == 0
            or i == 0
            or i == total - 1
        ):

            print(
                f"Progress: "
                f"{i + 1}/{total}"
            )

    return np.array(predictions)


# ============================================================
# 5. 保存预测
# ============================================================

def save_prediction(
    test_df,
    y_true,
    y_pred,
    order
):

    name = (
        f"ARIMA_{order[0]}_{order[1]}_{order[2]}"
    )

    result = pd.DataFrame(
        {
            "date": test_df["date"]
                .dt.strftime("%Y-%m-%d"),
            "actual_open": y_true,
            "predicted_open": y_pred
        }
    )

    path = os.path.join(
        OUTPUT_DIR,
        f"{name}_predictions.csv"
    )

    result.to_csv(
        path,
        index=False,
        encoding="utf-8-sig"
    )


# ============================================================
# 6. 主函数
# ============================================================

def main():

    print("=" * 80)
    print("ARIMA 参数比较实验")
    print("=" * 80)

    df = load_data()

    # --------------------------------------------------------
    # 80/20 划分
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

    print()
    print(
        f"总样本：{len(df)}"
    )

    print(
        f"训练集：{len(train_df)}"
    )

    print(
        f"测试集：{len(test_df)}"
    )

    print(
        f"测试日期："
        f"{test_df['date'].iloc[0].date()} "
        f"-> "
        f"{test_df['date'].iloc[-1].date()}"
    )

    all_results = []

    # --------------------------------------------------------
    # 每个 ARIMA
    # --------------------------------------------------------

    for order in ARIMA_ORDERS:

        y_pred = rolling_forecast(
            train_values,
            test_values,
            order
        )

        y_true = test_values

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

        save_prediction(
            test_df,
            y_true,
            y_pred,
            order
        )

        all_results.append(
            {
                "Model": f"ARIMA{order}",
                "p": order[0],
                "d": order[1],
                "q": order[2],
                "MAE": mae,
                "RMSE": rmse,
                "Directional_Accuracy": da,
                "Test_Samples": len(y_true)
            }
        )

        print()
        print(
            f"ARIMA{order}"
        )
        print(
            f"MAE = {mae:.4f}"
        )
        print(
            f"RMSE = {rmse:.4f}"
        )
        print(
            f"Directional Accuracy = "
            f"{da:.4f} "
            f"({da * 100:.2f}%)"
        )

    # --------------------------------------------------------
    # 汇总
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        all_results
    )

    results_df = results_df.sort_values(
        "MAE"
    ).reset_index(
        drop=True
    )

    result_path = os.path.join(
        OUTPUT_DIR,
        "arima_parameter_results.csv"
    )

    results_df.to_csv(
        result_path,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # 打印最终结果
    # --------------------------------------------------------

    print()
    print()
    print("=" * 90)
    print("ARIMA 参数比较结果")
    print("=" * 90)

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    best_mae = results_df.iloc[0]

    best_rmse = results_df.loc[
        results_df["RMSE"].idxmin()
    ]

    best_da = results_df.loc[
        results_df[
            "Directional_Accuracy"
        ].idxmax()
    ]

    print()
    print(
        f"MAE 最优："
        f"{best_mae['Model']}"
        f" -> "
        f"{best_mae['MAE']:.4f}"
    )

    print(
        f"RMSE 最优："
        f"{best_rmse['Model']}"
        f" -> "
        f"{best_rmse['RMSE']:.4f}"
    )

    print(
        f"方向准确率最优："
        f"{best_da['Model']}"
        f" -> "
        f"{best_da['Directional_Accuracy'] * 100:.2f}%"
    )

    print()
    print(
        f"结果保存：{result_path}"
    )

    print("=" * 90)


if __name__ == "__main__":
    main()