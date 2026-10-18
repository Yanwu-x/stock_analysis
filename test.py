# -*- coding: utf-8 -*-

"""
贵州茅台（600519）资金流向分析
数据来源：efinance

功能：
1. 获取 600519 历史资金流向数据
2. 保存原始资金流向数据
3. 与 A 股日线数据 ashare_600519.csv 合并
4. 计算：
   - 当日收益率
   - 下一交易日收益率
   - 各类资金净流入
5. 计算资金流向与股价变化的相关性
6. 分析当天资金流向与下一交易日收益之间的关系
7. 保存统计结果
8. 生成可视化图表

注意：
efinance 当前接口返回的是“资金净流入”等资金流向指标，
不能直接等同于真实逐笔买入量/卖出量。
"""


# ============================================================
# 1. 导入库
# ============================================================

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import efinance as ef


warnings.filterwarnings("ignore")


# ============================================================
# 2. 参数
# ============================================================

STOCK_CODE = "600519"

# 你的原始 A 股日线数据
STOCK_DATA_FILE = "ashare_600519.csv"

# 输出目录
OUTPUT_DIR = "results_fund_flow"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 3. 获取 efinance 资金流向数据
# ============================================================

def get_fund_flow():

    print("=" * 70)
    print("1. 获取贵州茅台历史资金流向")
    print("=" * 70)

    try:

        df = ef.stock.get_history_bill(
            STOCK_CODE
        )

    except Exception as e:

        raise RuntimeError(
            f"获取资金流向数据失败：{e}"
        )

    if df is None or len(df) == 0:

        raise ValueError(
            "efinance 没有返回资金流向数据。"
        )

    print(
        f"成功获取 {len(df)} 条资金流向数据"
    )

    print()

    print(
        "字段："
    )

    print(
        df.columns.tolist()
    )

    print()

    print(
        "前5行："
    )

    print(
        df.head()
    )

    # --------------------------------------------------------
    # 保存原始数据
    # --------------------------------------------------------

    raw_path = os.path.join(
        OUTPUT_DIR,
        "600519_fund_flow_raw.csv"
    )

    df.to_csv(
        raw_path,
        index=False,
        encoding="utf-8-sig"
    )

    print()

    print(
        f"原始资金流向数据已保存：{raw_path}"
    )

    return df


# ============================================================
# 4. 统一日期格式
# ============================================================

def prepare_fund_flow(df):

    df = df.copy()

    # --------------------------------------------------------
    # 检查日期列
    # --------------------------------------------------------

    if "日期" not in df.columns:

        raise ValueError(
            "资金流向数据中没有找到“日期”列。"
            f"\n当前字段：{df.columns.tolist()}"
        )

    # 日期
    df["日期"] = pd.to_datetime(
        df["日期"],
        errors="coerce"
    )

    # 删除日期异常
    df = df.dropna(
        subset=["日期"]
    ).copy()

    # 排序
    df = df.sort_values(
        "日期"
    ).reset_index(
        drop=True
    )

    return df


# ============================================================
# 5. 检查资金流向字段
# ============================================================

def check_flow_columns(df):

    expected_columns = [
        "主力净流入",
        "小单净流入",
        "中单净流入",
        "大单净流入",
        "超大单净流入"
    ]

    print()
    print("=" * 70)
    print("2. 检查资金流向字段")
    print("=" * 70)

    missing = []

    for col in expected_columns:

        if col in df.columns:

            print(
                f"[OK] {col}"
            )

        else:

            print(
                f"[缺失] {col}"
            )

            missing.append(col)

    if missing:

        print()

        print(
            "警告：以下字段不存在："
        )

        print(
            missing
        )

    return expected_columns


# ============================================================
# 6. 读取自己的 A 股日线数据
# ============================================================

def load_stock_data():

    print()
    print("=" * 70)
    print("3. 读取 A 股日线数据")
    print("=" * 70)

    if not os.path.exists(
        STOCK_DATA_FILE
    ):

        raise FileNotFoundError(
            f"\n找不到：{STOCK_DATA_FILE}\n"
            "请把它与 test.py 放在同一个目录。"
        )

    df = pd.read_csv(
        STOCK_DATA_FILE
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
            f"股票数据缺少字段：{missing}"
        )

    # 日期
    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    # 删除日期异常
    df = df.dropna(
        subset=["date"]
    ).copy()

    # 排序
    df = df.sort_values(
        "date"
    ).reset_index(
        drop=True
    )

    print(
        f"日线数据：{len(df)} 条"
    )

    print(
        f"日期范围："
        f"{df['date'].iloc[0].date()} "
        f"-> "
        f"{df['date'].iloc[-1].date()}"
    )

    return df


# ============================================================
# 7. 合并资金流向和股票数据
# ============================================================

def merge_data(
    fund_df,
    stock_df
):

    print()
    print("=" * 70)
    print("4. 按日期合并股票数据和资金流向")
    print("=" * 70)

    fund = fund_df.copy()

    stock = stock_df.copy()

    # --------------------------------------------------------
    # 日期统一成 date
    # --------------------------------------------------------

    fund = fund.rename(
        columns={
            "日期": "date"
        }
    )

    # --------------------------------------------------------
    # 合并
    # --------------------------------------------------------

    merged = pd.merge(
        stock,
        fund,
        on="date",
        how="inner"
    )

    merged = merged.sort_values(
        "date"
    ).reset_index(
        drop=True
    )

    print(
        f"股票数据：{len(stock)} 条"
    )

    print(
        f"资金流向：{len(fund)} 条"
    )

    print(
        f"成功匹配：{len(merged)} 条"
    )

    if len(merged) == 0:

        raise ValueError(
            "两份数据没有任何日期匹配成功，"
            "请检查日期格式。"
        )

    print()

    print(
        f"匹配日期："
        f"{merged['date'].iloc[0].date()} "
        f"-> "
        f"{merged['date'].iloc[-1].date()}"
    )

    return merged


# ============================================================
# 8. 构造收益率和资金流向特征
# ============================================================

def create_features(df):

    df = df.copy()

    print()
    print("=" * 70)
    print("5. 构造分析特征")
    print("=" * 70)

    # --------------------------------------------------------
    # 当日收盘价收益率
    #
    # Return_t = (Close_t / Close_(t-1)) - 1
    # --------------------------------------------------------

    df["return"] = (
        df["close"]
        .pct_change()
    )

    # --------------------------------------------------------
    # 下一交易日收益率
    #
    # Return_(t+1)
    #
    # shift(-1)
    # --------------------------------------------------------

    df["next_day_return"] = (
        df["return"]
        .shift(-1)
    )

    # --------------------------------------------------------
    # 下一交易日收盘价
    # --------------------------------------------------------

    df["next_day_close"] = (
        df["close"]
        .shift(-1)
    )

    # --------------------------------------------------------
    # 成交量变化率
    # --------------------------------------------------------

    df["volume_change"] = (
        df["volume"]
        .pct_change()
    )

    # --------------------------------------------------------
    # 资金流向与成交额比例
    #
    # 如果存在成交额字段则计算。
    # efinance 数据未必包含成交额，
    # 所以这里根据已有字段判断。
    # --------------------------------------------------------

    money_amount_col = None

    possible_amount_columns = [
        "成交额",
        "成交金额",
        "成交额(元)"
    ]

    for col in possible_amount_columns:

        if col in df.columns:

            money_amount_col = col
            break

    if money_amount_col is not None:

        df[
            "主力净流入占成交额"
        ] = (
            df["主力净流入"]
            / df[money_amount_col]
        )

    # --------------------------------------------------------
    # 将资金流向转换成“亿元”，更方便观察
    # --------------------------------------------------------

    flow_columns = [
        "主力净流入",
        "小单净流入",
        "中单净流入",
        "大单净流入",
        "超大单净流入"
    ]

    for col in flow_columns:

        if col in df.columns:

            df[
                col + "_亿元"
            ] = (
                df[col] / 100000000
            )

    # --------------------------------------------------------
    # 主力资金方向
    # --------------------------------------------------------

    if "主力净流入" in df.columns:

        df["main_flow_direction"] = np.where(
            df["主力净流入"] > 0,
            1,
            np.where(
                df["主力净流入"] < 0,
                -1,
                0
            )
        )

    return df


# ============================================================
# 9. 相关性分析
# ============================================================

def correlation_analysis(df):

    print()
    print("=" * 70)
    print("6. 资金流向与股票收益率相关性分析")
    print("=" * 70)

    flow_columns = [
        "主力净流入",
        "超大单净流入",
        "大单净流入",
        "中单净流入",
        "小单净流入"
    ]

    rows = []

    for col in flow_columns:

        if col not in df.columns:

            continue

        # ----------------------------------------------------
        # 当日相关性
        # ----------------------------------------------------

        temp_current = df[
            [col, "return"]
        ].dropna()

        if len(temp_current) >= 2:

            current_corr = (
                temp_current[
                    col
                ].corr(
                    temp_current[
                        "return"
                    ]
                )
            )

        else:

            current_corr = np.nan

        # ----------------------------------------------------
        # 下一交易日相关性
        # ----------------------------------------------------

        temp_next = df[
            [
                col,
                "next_day_return"
            ]
        ].dropna()

        if len(temp_next) >= 2:

            next_corr = (
                temp_next[
                    col
                ].corr(
                    temp_next[
                        "next_day_return"
                    ]
                )
            )

        else:

            next_corr = np.nan

        rows.append(
            {
                "资金流向指标": col,
                "与当日收益率相关系数": (
                    current_corr
                ),
                "与下一日收益率相关系数": (
                    next_corr
                ),
                "有效样本数": len(
                    temp_next
                )
            }
        )

    corr_df = pd.DataFrame(
        rows
    )

    print(
        corr_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    # --------------------------------------------------------
    # 保存
    # --------------------------------------------------------

    save_path = os.path.join(
        OUTPUT_DIR,
        "fund_flow_correlation.csv"
    )

    corr_df.to_csv(
        save_path,
        index=False,
        encoding="utf-8-sig"
    )

    print()

    print(
        f"相关性分析已保存：{save_path}"
    )

    return corr_df


# ============================================================
# 10. 分析净流入和上涨/下跌之间的关系
# ============================================================

def flow_direction_analysis(df):

    print()
    print("=" * 70)
    print("7. 主力资金净流入与涨跌关系")
    print("=" * 70)

    if "主力净流入" not in df.columns:

        print(
            "没有找到“主力净流入”字段，跳过。"
        )

        return None

    temp = df[
        [
            "主力净流入",
            "return"
        ]
    ].dropna().copy()

    # --------------------------------------------------------
    # 主力资金流入 / 流出
    # --------------------------------------------------------

    temp["flow_direction"] = np.where(
        temp["主力净流入"] > 0,
        "净流入",
        np.where(
            temp["主力净流入"] < 0,
            "净流出",
            "持平"
        )
    )

    # --------------------------------------------------------
    # 当日上涨 / 下跌
    # --------------------------------------------------------

    temp["price_direction"] = np.where(
        temp["return"] > 0,
        "上涨",
        np.where(
            temp["return"] < 0,
            "下跌",
            "持平"
        )
    )

    # --------------------------------------------------------
    # 交叉统计
    # --------------------------------------------------------

    crosstab = pd.crosstab(
        temp["flow_direction"],
        temp["price_direction"]
    )

    print(
        "\n主力资金方向 × 当日股价方向："
    )

    print(
        crosstab
    )

    # --------------------------------------------------------
    # 保存
    # --------------------------------------------------------

    save_path = os.path.join(
        OUTPUT_DIR,
        "main_flow_price_direction.csv"
    )

    crosstab.to_csv(
        save_path,
        encoding="utf-8-sig"
    )

    print()

    print(
        f"资金方向交叉统计已保存：{save_path}"
    )

    return crosstab


# ============================================================
# 11. 绘制股价 + 主力净流入
# ============================================================

def plot_price_and_main_flow(df):

    if "主力净流入" not in df.columns:

        return

    fig, ax1 = plt.subplots(
        figsize=(14, 7)
    )

    # --------------------------------------------------------
    # 左轴：股价
    # --------------------------------------------------------

    ax1.plot(
        df["date"],
        df["close"],
        label="Close Price"
    )

    ax1.set_xlabel(
        "Date"
    )

    ax1.set_ylabel(
        "Close Price"
    )

    # --------------------------------------------------------
    # 右轴：主力净流入
    # --------------------------------------------------------

    ax2 = ax1.twinx()

    ax2.plot(
        df["date"],
        df["主力净流入"] / 100000000,
        label="Main Net Flow"
    )

    ax2.axhline(
        0,
        linestyle="--"
    )

    ax2.set_ylabel(
        "Main Net Flow (100 Million Yuan)"
    )

    plt.title(
        "600519 Close Price and Main Net Flow"
    )

    fig.tight_layout()

    save_path = os.path.join(
        OUTPUT_DIR,
        "price_main_flow.png"
    )

    plt.savefig(
        save_path,
        dpi=200
    )

    plt.close()

    print(
        f"股价与主力净流入图已保存：{save_path}"
    )


# ============================================================
# 12. 绘制资金流向柱状图
# ============================================================

def plot_flow_types(df):

    flow_columns = [
        "主力净流入",
        "超大单净流入",
        "大单净流入",
        "中单净流入",
        "小单净流入"
    ]

    available = [
        col
        for col in flow_columns
        if col in df.columns
    ]

    if not available:

        return

    # --------------------------------------------------------
    # 最近数据
    # --------------------------------------------------------

    plot_df = df[
        [
            "date"
        ] + available
    ].copy()

    # 只画最近100个交易日
    plot_df = plot_df.tail(
        min(100, len(plot_df))
    )

    plt.figure(
        figsize=(14, 7)
    )

    for col in available:

        plt.plot(
            plot_df["date"],
            plot_df[col] / 100000000,
            label=col
        )

    plt.axhline(
        0,
        linestyle="--"
    )

    plt.xlabel(
        "Date"
    )

    plt.ylabel(
        "Net Flow (100 Million Yuan)"
    )

    plt.title(
        "600519 Different Capital Flow Categories"
    )

    plt.legend()

    plt.grid(
        True
    )

    plt.xticks(
        rotation=45
    )

    plt.tight_layout()

    save_path = os.path.join(
        OUTPUT_DIR,
        "different_flow_types.png"
    )

    plt.savefig(
        save_path,
        dpi=200
    )

    plt.close()

    print(
        f"不同资金类型图已保存：{save_path}"
    )


# ============================================================
# 13. 绘制资金流向与下一日收益关系
# ============================================================

def plot_main_flow_next_return(df):

    if (
        "主力净流入" not in df.columns
        or "next_day_return" not in df.columns
    ):

        return

    temp = df[
        [
            "主力净流入",
            "next_day_return"
        ]
    ].dropna()

    if len(temp) < 2:

        return

    plt.figure(
        figsize=(9, 7)
    )

    plt.scatter(
        temp["主力净流入"] / 100000000,
        temp["next_day_return"] * 100,
        alpha=0.6
    )

    plt.axhline(
        0,
        linestyle="--"
    )

    plt.axvline(
        0,
        linestyle="--"
    )

    plt.xlabel(
        "Main Net Flow (100 Million Yuan)"
    )

    plt.ylabel(
        "Next Day Return (%)"
    )

    plt.title(
        "Main Net Flow vs Next-Day Return"
    )

    plt.grid(
        True
    )

    plt.tight_layout()

    save_path = os.path.join(
        OUTPUT_DIR,
        "main_flow_vs_next_day_return.png"
    )

    plt.savefig(
        save_path,
        dpi=200
    )

    plt.close()

    print(
        f"主力净流入与下一日收益散点图已保存：{save_path}"
    )


# ============================================================
# 14. 基础统计
# ============================================================

def descriptive_statistics(df):

    print()
    print("=" * 70)
    print("8. 资金流向描述性统计")
    print("=" * 70)

    flow_columns = [
        "主力净流入",
        "超大单净流入",
        "大单净流入",
        "中单净流入",
        "小单净流入"
    ]

    available = [
        col
        for col in flow_columns
        if col in df.columns
    ]

    if not available:

        return None

    stats = (
        df[available]
        .describe()
        .T
    )

    stats[
        "mean_亿元"
    ] = stats["mean"] / 100000000

    stats[
        "std_亿元"
    ] = stats["std"] / 100000000

    stats[
        "min_亿元"
    ] = stats["min"] / 100000000

    stats[
        "max_亿元"
    ] = stats["max"] / 100000000

    print(
        stats[
            [
                "count",
                "mean_亿元",
                "std_亿元",
                "min_亿元",
                "max_亿元"
            ]
        ].to_string(
            float_format=lambda x: f"{x:.4f}"
        )
    )

    save_path = os.path.join(
        OUTPUT_DIR,
        "fund_flow_descriptive_statistics.csv"
    )

    stats.to_csv(
        save_path,
        encoding="utf-8-sig"
    )

    print()

    print(
        f"描述性统计已保存：{save_path}"
    )

    return stats


# ============================================================
# 15. 保存最终合并数据
# ============================================================

def save_merged_data(df):

    save_path = os.path.join(
        OUTPUT_DIR,
        "600519_stock_fund_flow_merged.csv"
    )

    df.to_csv(
        save_path,
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print(
        f"最终合并数据已保存：{save_path}"
    )


# ============================================================
# 16. 主函数
# ============================================================

def main():

    print()
    print("=" * 80)
    print("贵州茅台 600519 资金流向分析")
    print("=" * 80)

    # --------------------------------------------------------
    # 1. 获取资金流向
    # --------------------------------------------------------

    fund_df = get_fund_flow()

    # --------------------------------------------------------
    # 2. 处理资金流向
    # --------------------------------------------------------

    fund_df = prepare_fund_flow(
        fund_df
    )

    # --------------------------------------------------------
    # 3. 检查字段
    # --------------------------------------------------------

    check_flow_columns(
        fund_df
    )

    # --------------------------------------------------------
    # 4. 读取股票日线
    # --------------------------------------------------------

    stock_df = load_stock_data()

    # --------------------------------------------------------
    # 5. 合并
    # --------------------------------------------------------

    merged_df = merge_data(
        fund_df,
        stock_df
    )

    # --------------------------------------------------------
    # 6. 构造特征
    # --------------------------------------------------------

    merged_df = create_features(
        merged_df
    )

    # --------------------------------------------------------
    # 7. 相关性
    # --------------------------------------------------------

    corr_df = correlation_analysis(
        merged_df
    )

    # --------------------------------------------------------
    # 8. 主力资金方向分析
    # --------------------------------------------------------

    flow_direction_analysis(
        merged_df
    )

    # --------------------------------------------------------
    # 9. 描述性统计
    # --------------------------------------------------------

    descriptive_statistics(
        merged_df
    )

    # --------------------------------------------------------
    # 10. 保存合并数据
    # --------------------------------------------------------

    save_merged_data(
        merged_df
    )

    # --------------------------------------------------------
    # 11. 画图
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("9. 生成可视化结果")
    print("=" * 70)

    plot_price_and_main_flow(
        merged_df
    )

    plot_flow_types(
        merged_df
    )

    plot_main_flow_next_return(
        merged_df
    )

    # --------------------------------------------------------
    # 12. 完成
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("全部分析完成！")
    print("=" * 80)

    print()
    print(
        f"输出目录：{OUTPUT_DIR}"
    )

    print()

    print(
        "主要文件："
    )

    print(
        "1. 600519_fund_flow_raw.csv"
    )

    print(
        "2. 600519_stock_fund_flow_merged.csv"
    )

    print(
        "3. fund_flow_correlation.csv"
    )

    print(
        "4. fund_flow_descriptive_statistics.csv"
    )

    print(
        "5. main_flow_price_direction.csv"
    )

    print(
        "6. price_main_flow.png"
    )

    print(
        "7. different_flow_types.png"
    )

    print(
        "8. main_flow_vs_next_day_return.png"
    )

    print()

    print("=" * 80)


# ============================================================
# 17. 程序入口
# ============================================================

if __name__ == "__main__":
    main()