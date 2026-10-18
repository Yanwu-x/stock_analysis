# -*- coding: utf-8 -*-

"""
股票预测模型简单交易回测

用途：
1. 读取已经训练好的预测 CSV，不重新训练模型。
2. 根据“预测当天 Open 相对前一交易日 Open 的方向”生成交易信号。
3. 采用简单日内策略：
   - 预测当天上涨 -> 当天开盘买入，收盘卖出
   - 预测当天下跌/不涨 -> 空仓
4. 计算：
   - 累计收益率
   - 最大回撤
   - 交易次数
   - 胜率
   - 最终资金
5. 与买入持有策略比较。

注意：
- 你的模型预测的是“下一交易日 Open”。
- 因此信号在当天开盘前已经可以由前一交易日及之前的信息生成。
- 为避免未来函数，本回测使用：
    signal_t = predicted_open_t > previous_open_t
  然后：
    若 signal_t = 1，在 t 日开盘买入，并在 t 日收盘卖出。
- 不使用当天收盘价来生成当天交易信号。
- 默认单边手续费 0.2%（0.002）。
  即买入收一次，卖出再收一次。
- 买入持有也考虑首次买入和最终卖出各一次手续费。

默认目录：

当前目录/
├─ ashare_600519.csv
├─ results_causal_wavelet_corrected/
│  ├─ raw_lstm_predictions.csv
│  ├─ causal_DB4_L2_predictions.csv
│  ├─ causal_DB4_L3_predictions.csv
│  └─ causal_DB4_L4_predictions.csv
└─ results_gru/
   └─ gru_predictions.csv

输出：
results_backtest/
├─ backtest_summary.csv
├─ backtest_summary.xlsx
├─ backtest_daily_*.csv
└─ backtest_equity_*.png

运行：
python stock_backtest.py
"""

from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# ============================================================
# 1. 参数
# ============================================================

SEARCH_ROOT = Path(".")

OUTPUT_DIR = SEARCH_ROOT / "results_backtest"

INITIAL_CAPITAL = 100000.0

# 单边手续费 0.2%
FEE_RATE = 0.002

# 印花税等其他成本暂不单独设置。
# 如后续老师要求，可以再增加。
OTHER_SELL_COST = 0.0

# 已知模型预测文件
PREDICTION_FILES = [
    ("Naive", "results_causal_wavelet_corrected/naive_predictions.csv"),
    ("RAW-LSTM", "results_causal_wavelet_corrected/raw_lstm_predictions.csv"),
    ("RAW-GRU", "results_gru/gru_predictions.csv"),
    ("Causal Wavelet-L2", "results_causal_wavelet_corrected/causal_DB4_L2_predictions.csv"),
    ("Causal Wavelet-L3", "results_causal_wavelet_corrected/causal_DB4_L3_predictions.csv"),
    ("Causal Wavelet-L4", "results_causal_wavelet_corrected/causal_DB4_L4_predictions.csv"),
]

# 如果你之后想加入 ARIMA，把下面解开并填写实际文件名：
EXTRA_FILES = [
    # ("ARIMA(5,1,0)", "results_arima/arima_predictions.csv"),
]


# ============================================================
# 2. 找原始数据
# ============================================================

def find_market_file():
    candidates = [
        SEARCH_ROOT / "ashare_600519.csv",
        SEARCH_ROOT / "ashare_600519(1).csv",
    ]

    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError(
        "找不到 ashare_600519.csv / ashare_600519(1).csv"
    )


# ============================================================
# 3. 读取原始行情
# ============================================================

def load_market_data(path):
    df = pd.read_csv(path)

    rename = {}

    for col in df.columns:
        low = str(col).strip().lower()

        if low == "date":
            rename[col] = "date"

        elif low == "open":
            rename[col] = "open"

        elif low == "close":
            rename[col] = "close"

    df = df.rename(columns=rename)

    required = ["date", "open", "close"]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"原始行情缺少字段：{missing}\n"
            f"当前字段：{list(df.columns)}"
        )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    df["open"] = pd.to_numeric(
        df["open"],
        errors="coerce"
    )

    df["close"] = pd.to_numeric(
        df["close"],
        errors="coerce"
    )

    df = (
        df.dropna(
            subset=["date", "open", "close"]
        )
        .sort_values("date")
        .drop_duplicates(
            subset=["date"],
            keep="last"
        )
        .reset_index(drop=True)
    )

    # 前一交易日真实 Open
    df["previous_open"] = df["open"].shift(1)

    # 日内 Open -> Close 收益率
    df["intraday_return"] = (
        df["close"] / df["open"] - 1.0
    )

    return df


# ============================================================
# 4. 读取预测文件
# ============================================================

def load_prediction(path):
    df = pd.read_csv(path)

    rename = {}

    for col in df.columns:
        low = str(col).strip().lower()

        if low == "date":
            rename[col] = "date"

        elif low in {
            "actual_open",
            "real_open",
        }:
            rename[col] = "actual_open"

        elif low in {
            "predicted_open",
            "pred_open",
            "predicted",
        }:
            rename[col] = "predicted_open"

    df = df.rename(columns=rename)

    required = [
        "date",
        "actual_open",
        "predicted_open",
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"预测文件缺少字段：{missing}\n"
            f"当前字段：{list(df.columns)}"
        )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    df["actual_open"] = pd.to_numeric(
        df["actual_open"],
        errors="coerce"
    )

    df["predicted_open"] = pd.to_numeric(
        df["predicted_open"],
        errors="coerce"
    )

    df = (
        df.dropna(
            subset=[
                "date",
                "actual_open",
                "predicted_open",
            ]
        )
        .sort_values("date")
        .drop_duplicates(
            subset=["date"],
            keep="last"
        )
        .reset_index(drop=True)
    )

    return df


# ============================================================
# 5. 回测一个模型
# ============================================================

def backtest_model(
    model_name,
    prediction_df,
    market_df,
):
    merged = prediction_df.merge(
        market_df[
            [
                "date",
                "open",
                "close",
                "previous_open",
                "intraday_return",
            ]
        ],
        on="date",
        how="left",
    )

    merged = merged.dropna(
        subset=[
            "previous_open",
            "open",
            "close",
        ]
    ).copy()

    if merged.empty:
        raise ValueError(
            "预测日期无法与原始行情数据匹配。"
        )

    # --------------------------------------------
    # 检查 prediction actual_open 是否和市场 open 一致
    # --------------------------------------------

    actual_open_diff = (
        merged["actual_open"]
        - merged["open"]
    )

    max_abs_diff = float(
        np.max(
            np.abs(actual_open_diff)
        )
    )

    # --------------------------------------------
    # 生成信号
    #
    # 预测当天 Open > 前一天真实 Open -> 看涨
    # --------------------------------------------

    merged["signal"] = (
        merged["predicted_open"]
        > merged["previous_open"]
    ).astype(int)

    # 真实方向
    merged["actual_up"] = (
        merged["open"]
        > merged["previous_open"]
    ).astype(int)

    # --------------------------------------------
    # 当日策略收益
    #
    # 如果 signal=1：
    #   开盘买入
    #   收盘卖出
    #
    # 扣除：
    #   买入手续费 FEE_RATE
    #   卖出手续费 FEE_RATE
    #   卖出其他成本 OTHER_SELL_COST
    #
    # 公式：
    #   gross = close/open
    #   net_factor = gross
    #               * (1-FEE)
    #               * (1-FEE-OTHER_SELL_COST)
    #
    # signal=0 -> 资金不变
    # --------------------------------------------

    fee_factor_buy = 1.0 - FEE_RATE
    fee_factor_sell = (
        1.0
        - FEE_RATE
        - OTHER_SELL_COST
    )

    gross_factor = (
        merged["close"]
        / merged["open"]
    )

    trade_factor = (
        gross_factor
        * fee_factor_buy
        * fee_factor_sell
    )

    merged["strategy_factor"] = np.where(
        merged["signal"] == 1,
        trade_factor,
        1.0,
    )

    # 单日净收益率
    merged["strategy_return"] = (
        merged["strategy_factor"] - 1.0
    )

    # 交易是否发生
    merged["trade"] = (
        merged["signal"] == 1
    ).astype(int)

    # 是否预测正确
    merged["correct_direction"] = (
        merged["signal"]
        == merged["actual_up"]
    ).astype(int)

    # --------------------------------------------
    # 资金曲线
    # --------------------------------------------

    merged["equity"] = (
        INITIAL_CAPITAL
        * merged["strategy_factor"].cumprod()
    )

    # 买入持有：
    # 第一个测试日开盘买入，最后一个测试日收盘卖出
    first_open = float(
        merged["open"].iloc[0]
    )

    last_close = float(
        merged["close"].iloc[-1]
    )

    buy_hold_factor = (
        last_close
        / first_open
        * fee_factor_buy
        * fee_factor_sell
    )

    buy_hold_final = (
        INITIAL_CAPITAL
        * buy_hold_factor
    )

    # --------------------------------------------
    # 累计收益
    # --------------------------------------------

    final_capital = float(
        merged["equity"].iloc[-1]
    )

    cumulative_return = (
        final_capital
        / INITIAL_CAPITAL
        - 1.0
    )

    buy_hold_return = (
        buy_hold_final
        / INITIAL_CAPITAL
        - 1.0
    )

    # --------------------------------------------
    # 最大回撤
    # --------------------------------------------

    equity = merged["equity"].to_numpy()

    running_max = np.maximum.accumulate(
        equity
    )

    drawdown = (
        equity / running_max - 1.0
    )

    max_drawdown = float(
        np.min(drawdown)
    )

    # --------------------------------------------
    # 交易统计
    # --------------------------------------------

    trade_days = int(
        merged["trade"].sum()
    )

    if trade_days > 0:
        winning_trades = int(
            (
                merged.loc[
                    merged["trade"] == 1,
                    "intraday_return"
                ]
                > 0
            ).sum()
        )

        win_rate = (
            winning_trades
            / trade_days
        )
    else:
        winning_trades = 0
        win_rate = np.nan

    direction_accuracy = float(
        merged["correct_direction"].mean()
    )

    # --------------------------------------------
    # 保存逐日数据
    # --------------------------------------------

    safe_name = re.sub(
        r"[^A-Za-z0-9_-]+",
        "_",
        model_name,
    )

    detail_path = (
        OUTPUT_DIR
        / f"backtest_daily_{safe_name}.csv"
    )

    merged[
        [
            "date",
            "previous_open",
            "actual_open",
            "predicted_open",
            "open",
            "close",
            "signal",
            "actual_up",
            "correct_direction",
            "intraday_return",
            "strategy_return",
            "equity",
        ]
    ].to_csv(
        detail_path,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------
    # 保存净值图
    # --------------------------------------------

    plt.figure(
        figsize=(12, 6)
    )

    plt.plot(
        merged["date"],
        merged["equity"],
        label=f"{model_name} Strategy",
    )

    # 买入持有净值，仅作为首尾对比的参考线
    # 用线性插值绘制不是买入持有真实逐日净值，因此图例明确标注为 reference。
    buy_hold_start = INITIAL_CAPITAL
    buy_hold_end = buy_hold_final

    plt.plot(
        [merged["date"].iloc[0], merged["date"].iloc[-1]],
        [buy_hold_start, buy_hold_end],
        linestyle="--",
        label="Buy & Hold (reference)",
    )

    plt.xlabel("Date")
    plt.ylabel("Capital")
    plt.title(
        f"{model_name} Backtest Equity Curve"
    )
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    figure_path = (
        OUTPUT_DIR
        / f"backtest_equity_{safe_name}.png"
    )

    plt.savefig(
        figure_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    return {
        "Model": model_name,
        "Initial_Capital": INITIAL_CAPITAL,
        "Final_Capital": final_capital,
        "Cumulative_Return": cumulative_return,
        "Buy_Hold_Final_Capital": buy_hold_final,
        "Buy_Hold_Return": buy_hold_return,
        "Excess_Return_vs_Buy_Hold": (
            cumulative_return - buy_hold_return
        ),
        "Max_Drawdown": max_drawdown,
        "Trade_Days": trade_days,
        "Win_Days": winning_trades,
        "Win_Rate": win_rate,
        "Direction_Accuracy": direction_accuracy,
        "Test_Samples": len(merged),
        "Fee_Rate_Per_Side": FEE_RATE,
        "Max_Actual_Open_Diff": max_abs_diff,
        "Detail_File": str(detail_path),
        "Equity_Figure": str(figure_path),
    }


# ============================================================
# 6. 主程序
# ============================================================

def main():
    print("=" * 90)
    print("股票预测模型简单交易回测")
    print("=" * 90)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------
    # 原始数据
    # --------------------------------------------

    market_path = find_market_file()

    print(
        f"\n原始行情：{market_path}"
    )

    market_df = load_market_data(
        market_path
    )

    print(
        f"数据量：{len(market_df)}"
    )

    print(
        f"时间范围："
        f"{market_df['date'].iloc[0].date()} "
        f"-> "
        f"{market_df['date'].iloc[-1].date()}"
    )

    print(
        f"初始资金：{INITIAL_CAPITAL:.2f}"
    )

    print(
        f"单边手续费：{FEE_RATE * 100:.2f}%"
    )

    # --------------------------------------------
    # 文件列表
    # --------------------------------------------

    all_files = (
        PREDICTION_FILES
        + EXTRA_FILES
    )

    existing_files = []

    print("\n预测文件：")

    for model_name, relative_path in all_files:
        path = SEARCH_ROOT / relative_path

        if path.exists():
            print(
                f"  [√] {model_name}: {path}"
            )
            existing_files.append(
                (model_name, path)
            )
        else:
            print(
                f"  [未找到] {model_name}: {path}"
            )

    if not existing_files:
        raise RuntimeError(
            "没有找到任何预测文件。"
        )

    # --------------------------------------------
    # 回测
    # --------------------------------------------

    results = []

    for model_name, path in existing_files:
        print("\n" + "-" * 90)
        print(
            f"开始回测：{model_name}"
        )

        try:
            prediction_df = load_prediction(
                path
            )

            result = backtest_model(
                model_name=model_name,
                prediction_df=prediction_df,
                market_df=market_df,
            )

            results.append(result)

            print(
                f"测试样本：{result['Test_Samples']}"
            )

            print(
                f"交易天数：{result['Trade_Days']}"
            )

            print(
                f"方向准确率："
                f"{result['Direction_Accuracy'] * 100:.2f}%"
            )

            print(
                f"累计收益率："
                f"{result['Cumulative_Return'] * 100:.2f}%"
            )

            print(
                f"买入持有收益率："
                f"{result['Buy_Hold_Return'] * 100:.2f}%"
            )

            print(
                f"相对买入持有："
                f"{result['Excess_Return_vs_Buy_Hold'] * 100:.2f}%"
            )

            print(
                f"最大回撤："
                f"{result['Max_Drawdown'] * 100:.2f}%"
            )

            print(
                f"胜率："
                f"{result['Win_Rate'] * 100:.2f}%"
                if not np.isnan(result["Win_Rate"])
                else "胜率：N/A"
            )

            print(
                f"实际 Open 对齐最大误差："
                f"{result['Max_Actual_Open_Diff']:.8f}"
            )

        except Exception as exc:
            print(
                f"[跳过] {model_name}: {exc}"
            )

    if not results:
        raise RuntimeError(
            "所有模型回测均失败。"
        )

    # --------------------------------------------
    # 汇总
    # --------------------------------------------

    summary = pd.DataFrame(
        results
    )

    preferred_order = [
        "Naive",
        "ARIMA(5,1,0)",
        "RAW-LSTM",
        "RAW-GRU",
        "Causal Wavelet-L2",
        "Causal Wavelet-L3",
        "Causal Wavelet-L4",
    ]

    order_map = {
        name: i
        for i, name in enumerate(
            preferred_order
        )
    }

    summary["_order"] = (
        summary["Model"]
        .map(order_map)
        .fillna(999)
    )

    summary = (
        summary
        .sort_values(
            ["_order", "Model"]
        )
        .drop(columns="_order")
        .reset_index(drop=True)
    )

    csv_path = (
        OUTPUT_DIR
        / "backtest_summary.csv"
    )

    summary.to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig",
    )

    xlsx_path = (
        OUTPUT_DIR
        / "backtest_summary.xlsx"
    )

    try:
        with pd.ExcelWriter(
            xlsx_path,
            engine="openpyxl",
        ) as writer:
            summary.to_excel(
                writer,
                sheet_name="Summary",
                index=False,
            )
    except Exception as exc:
        print(
            f"[提示] Excel 写入失败：{exc}"
        )
        xlsx_path = None

    print("\n" + "=" * 90)
    print("回测汇总")
    print("=" * 90)

    display_cols = [
        "Model",
        "Final_Capital",
        "Cumulative_Return",
        "Buy_Hold_Return",
        "Excess_Return_vs_Buy_Hold",
        "Max_Drawdown",
        "Trade_Days",
        "Win_Rate",
        "Direction_Accuracy",
    ]

    print(
        summary[display_cols].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\n结果已保存：")
    print(f"CSV：{csv_path}")

    if xlsx_path is not None:
        print(f"Excel：{xlsx_path}")

    print(
        f"明细与净值图：{OUTPUT_DIR}"
    )

    print("\n完成。")


if __name__ == "__main__":
    main()
