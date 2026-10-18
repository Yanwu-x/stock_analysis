# -*- coding: utf-8 -*-

"""
A-share Stock Price Prediction
ARIMA Time Series Baseline

实验目标：
使用传统时间序列模型 ARIMA 对 A 股 Open 价格进行预测，
并与之前的 LSTM / Wavelet-LSTM 结果进行比较。

数据：
    ashare_600519.csv

数据划分：
    前 80% -> Train
    后 20% -> Test

模型：
    ARIMA(5, 1, 0)

评价指标：
    MAE
    RMSE
    Directional Accuracy

输出：
    results_arima/
        arima_results.csv
        arima_predictions.csv
        arima_prediction_plot.png
        arima_residual_plot.png
"""

import os

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller


# ============================================================
# 1. 参数设置
# ============================================================

DATA_FILE = "ashare_600519.csv"

OUTPUT_DIR = "results_arima"

TARGET_COLUMN = "open"

TRAIN_RATIO = 0.8

# ARIMA(p, d, q)
ARIMA_ORDER = (5, 1, 0)


# ============================================================
# 2. 创建输出目录
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 3. 读取数据
# ============================================================

def load_data():

    print("=" * 70)
    print("读取 A 股数据")
    print("=" * 70)

    if not os.path.exists(DATA_FILE):

        raise FileNotFoundError(
            f"\n找不到数据文件：{DATA_FILE}\n"
            f"请确认 csv 文件和 Python 文件位于同一个目录。"
        )

    df = pd.read_csv(
        DATA_FILE
    )

    print(
        f"数据文件：{DATA_FILE}"
    )

    print(
        f"原始数据形状：{df.shape}"
    )

    required_columns = [
        "date",
        "open"
    ]

    for col in required_columns:

        if col not in df.columns:

            raise ValueError(
                f"数据中缺少列：{col}"
            )

    # 日期
    df["date"] = pd.to_datetime(
        df["date"]
    )

    # Open 转数字
    df["open"] = pd.to_numeric(
        df["open"],
        errors="coerce"
    )

    # 按时间排序
    df = df.sort_values(
        "date"
    ).reset_index(
        drop=True
    )

    # 删除缺失值
    df = df.dropna(
        subset=["open"]
    ).reset_index(
        drop=True
    )

    # 检查 NaN
    if df["open"].isna().any():

        raise ValueError(
            "Open 中存在 NaN。"
        )

    # 检查 Inf
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
# 4. 训练集 / 测试集划分
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
    print("时间序列数据划分")
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
        "训练集："
        f"{train_df['date'].iloc[0].date()}"
        " -> "
        f"{train_df['date'].iloc[-1].date()}"
    )

    print(
        "测试集："
        f"{test_df['date'].iloc[0].date()}"
        " -> "
        f"{test_df['date'].iloc[-1].date()}"
    )

    return train_df, test_df


# ============================================================
# 5. ADF 平稳性检验
# ============================================================

def adf_test(series, name):

    print()
    print("=" * 70)
    print(
        f"ADF 平稳性检验：{name}"
    )
    print("=" * 70)

    result = adfuller(
        series,
        autolag="AIC"
    )

    adf_statistic = result[0]
    p_value = result[1]

    print(
        f"ADF Statistic : "
        f"{adf_statistic:.6f}"
    )

    print(
        f"p-value       : "
        f"{p_value:.6f}"
    )

    print(
        "Critical Values:"
    )

    for key, value in result[4].items():

        print(
            f"    {key}: {value:.6f}"
        )

    if p_value < 0.05:

        print(
            "\n结论："
            "在 5% 显著性水平下拒绝单位根假设，"
            "序列可以认为是平稳的。"
        )

    else:

        print(
            "\n结论："
            "在 5% 显著性水平下不能拒绝单位根假设，"
            "原始序列可能存在非平稳性。"
        )

    return {
        "ADF_Statistic": adf_statistic,
        "p_value": p_value
    }


# ============================================================
# 6. Directional Accuracy
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
# 7. ARIMA 滚动预测
# ============================================================

def rolling_arima_forecast(
    train_values,
    test_values,
    order
):

    """
    滚动预测：

    第一步：
        使用全部训练数据
        -> 预测第一个测试样本

    第二步：
        将真实的第一个测试值加入历史数据
        -> 预测第二个测试样本

    第三步：
        继续加入真实测试值
        -> 预测第三个测试样本

    ...

    这样可以避免一次性直接预测整个测试集，
    更符合时间序列逐步预测的形式。
    """

    history = list(
        train_values
    )

    predictions = []

    total = len(
        test_values
    )

    print()
    print("=" * 70)
    print("开始 ARIMA 滚动预测")
    print("=" * 70)

    print(
        f"ARIMA order = {order}"
    )

    print(
        f"测试样本数 = {total}"
    )

    for i in range(total):

        # ----------------------------------------------------
        # 建立当前 ARIMA 模型
        # ----------------------------------------------------

        model = ARIMA(
            history,
            order=order
        )

        model_fit = model.fit()

        # ----------------------------------------------------
        # 预测下一点
        # ----------------------------------------------------

        forecast = model_fit.forecast(
            steps=1
        )

        prediction = float(
            forecast[0]
        )

        predictions.append(
            prediction
        )

        # ----------------------------------------------------
        # 将真实测试值加入历史
        # ----------------------------------------------------

        actual_value = float(
            test_values[i]
        )

        history.append(
            actual_value
        )

        # ----------------------------------------------------
        # 打印进度
        # ----------------------------------------------------

        if (
            (i + 1) % 20 == 0
            or i == 0
            or i == total - 1
        ):

            print(
                f"Progress: "
                f"{i + 1}/{total}"
            )

    return np.array(
        predictions
    )


# ============================================================
# 8. 保存预测结果
# ============================================================

def save_predictions(
    test_df,
    y_true,
    y_pred
):

    prediction_df = pd.DataFrame(
        {
            "date": test_df[
                "date"
            ].dt.strftime(
                "%Y-%m-%d"
            ),
            "actual_open": y_true,
            "predicted_open": y_pred
        }
    )

    save_path = os.path.join(
        OUTPUT_DIR,
        "arima_predictions.csv"
    )

    prediction_df.to_csv(
        save_path,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        f"预测结果已保存：{save_path}"
    )

    return prediction_df


# ============================================================
# 9. 绘制预测结果
# ============================================================

def save_prediction_plot(
    test_df,
    y_true,
    y_pred,
    order
):

    plt.figure(
        figsize=(14, 6)
    )

    plt.plot(
        test_df["date"],
        y_true,
        label="Actual Open"
    )

    plt.plot(
        test_df["date"],
        y_pred,
        label="ARIMA Prediction"
    )

    plt.xlabel(
        "Date"
    )

    plt.ylabel(
        "Open Price"
    )

    plt.title(
        f"ARIMA{order} Prediction vs Actual"
    )

    plt.legend()

    plt.grid(True)

    plt.xticks(
        rotation=45
    )

    plt.tight_layout()

    save_path = os.path.join(
        OUTPUT_DIR,
        "arima_prediction_plot.png"
    )

    plt.savefig(
        save_path,
        dpi=200
    )

    plt.close()

    print(
        f"预测对比图已保存：{save_path}"
    )


# ============================================================
# 10. 绘制残差
# ============================================================

def save_residual_plot(
    test_df,
    y_true,
    y_pred
):

    residuals = (
        y_true - y_pred
    )

    plt.figure(
        figsize=(14, 5)
    )

    plt.plot(
        test_df["date"],
        residuals
    )

    plt.axhline(
        0,
        linestyle="--"
    )

    plt.xlabel(
        "Date"
    )

    plt.ylabel(
        "Residual"
    )

    plt.title(
        "ARIMA Prediction Residuals"
    )

    plt.grid(True)

    plt.xticks(
        rotation=45
    )

    plt.tight_layout()

    save_path = os.path.join(
        OUTPUT_DIR,
        "arima_residual_plot.png"
    )

    plt.savefig(
        save_path,
        dpi=200
    )

    plt.close()

    print(
        f"残差图已保存：{save_path}"
    )


# ============================================================
# 11. 保存最终指标
# ============================================================

def save_results(
    order,
    mae,
    rmse,
    directional_accuracy,
    test_samples,
    adf_result
):

    result_df = pd.DataFrame(
        [
            {
                "Model": "ARIMA",
                "Order": str(order),
                "MAE": mae,
                "RMSE": rmse,
                "Directional_Accuracy": (
                    directional_accuracy
                ),
                "Test_Samples": test_samples,
                "ADF_Statistic": (
                    adf_result[
                        "ADF_Statistic"
                    ]
                ),
                "ADF_p_value": (
                    adf_result[
                        "p_value"
                    ]
                )
            }
        ]
    )

    save_path = os.path.join(
        OUTPUT_DIR,
        "arima_results.csv"
    )

    result_df.to_csv(
        save_path,
        index=False,
        encoding="utf-8-sig"
    )

    return result_df


# ============================================================
# 12. 主函数
# ============================================================

def main():

    # --------------------------------------------------------
    # 1. 读取数据
    # --------------------------------------------------------

    df = load_data()

    # --------------------------------------------------------
    # 2. 80/20 时间划分
    # --------------------------------------------------------

    train_df, test_df = split_data(
        df
    )

    train_values = (
        train_df[
            TARGET_COLUMN
        ]
        .values
        .astype(float)
    )

    test_values = (
        test_df[
            TARGET_COLUMN
        ]
        .values
        .astype(float)
    )

    # --------------------------------------------------------
    # 3. ADF 检验
    # --------------------------------------------------------

    adf_original = adf_test(
        train_values,
        "Original Open"
    )

    # --------------------------------------------------------
    # 4. 一阶差分 ADF
    # --------------------------------------------------------

    diff_values = np.diff(
        train_values
    )

    adf_difference = adf_test(
        diff_values,
        "First Difference of Open"
    )

    # --------------------------------------------------------
    # 5. ARIMA 滚动预测
    # --------------------------------------------------------

    y_pred = rolling_arima_forecast(
        train_values=train_values,
        test_values=test_values,
        order=ARIMA_ORDER
    )

    y_true = test_values

    # --------------------------------------------------------
    # 6. MAE
    # --------------------------------------------------------

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    # --------------------------------------------------------
    # 7. RMSE
    # --------------------------------------------------------

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    # --------------------------------------------------------
    # 8. Directional Accuracy
    # --------------------------------------------------------

    directional_accuracy = (
        calculate_directional_accuracy(
            y_true,
            y_pred
        )
    )

    # --------------------------------------------------------
    # 9. 保存预测结果
    # --------------------------------------------------------

    save_predictions(
        test_df=test_df,
        y_true=y_true,
        y_pred=y_pred
    )

    # --------------------------------------------------------
    # 10. 保存预测图
    # --------------------------------------------------------

    save_prediction_plot(
        test_df=test_df,
        y_true=y_true,
        y_pred=y_pred,
        order=ARIMA_ORDER
    )

    # --------------------------------------------------------
    # 11. 保存残差图
    # --------------------------------------------------------

    save_residual_plot(
        test_df=test_df,
        y_true=y_true,
        y_pred=y_pred
    )

    # --------------------------------------------------------
    # 12. 保存指标
    # --------------------------------------------------------

    result_df = save_results(
        order=ARIMA_ORDER,
        mae=mae,
        rmse=rmse,
        directional_accuracy=(
            directional_accuracy
        ),
        test_samples=len(
            y_true
        ),
        adf_result=adf_original
    )

    # --------------------------------------------------------
    # 13. 输出结果
    # --------------------------------------------------------

    print()
    print()
    print("=" * 80)
    print("ARIMA 实验完成")
    print("=" * 80)

    print()

    print(
        f"Model : ARIMA{ARIMA_ORDER}"
    )

    print(
        f"MAE   : {mae:.4f}"
    )

    print(
        f"RMSE  : {rmse:.4f}"
    )

    print(
        f"Directional Accuracy : "
        f"{directional_accuracy:.4f}"
        f" ({directional_accuracy * 100:.2f}%)"
    )

    print(
        f"Test Samples : {len(y_true)}"
    )

    print()

    print(
        "ARIMA 结果："
    )

    print(
        result_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    print()

    print(
        f"结果目录：{OUTPUT_DIR}"
    )

    print(
        "包含："
    )

    print(
        "  - arima_results.csv"
    )

    print(
        "  - arima_predictions.csv"
    )

    print(
        "  - arima_prediction_plot.png"
    )

    print(
        "  - arima_residual_plot.png"
    )

    print("=" * 80)


# ============================================================
# 13. 程序入口
# ============================================================

if __name__ == "__main__":
    main()