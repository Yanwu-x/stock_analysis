import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ===================== 配置区 =====================
INPUT_CSV = r"./results_walk_forward_validation/predictions_all_folds.csv"
FEE_RATE = 0.0015    # 双边手续费
INIT_CAPITAL = 100000
SAVE_FIG = True
SAVE_SUMMARY = True
OUTPUT_SUMMARY = "./backtest_wf_summary.csv"
# =================================================

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

df_raw = pd.read_csv(INPUT_CSV)
df_raw["date"] = pd.to_datetime(df_raw["date"])
df_raw = df_raw.sort_values("date").reset_index(drop=True)

model_list = df_raw["Model"].unique()
summary = []

def run_backtest(sub_df, model_name):
    sub_df = sub_df.sort_values("date").reset_index(drop=True)
    capital = INIT_CAPITAL
    share = 0.0
    nav = []
    bh_cap = INIT_CAPITAL
    bh_nav = []

    for _, row in sub_df.iterrows():
        ret = row["true_return"]
        signal = row["predicted_up"]
        open_p = row["actual_open"]

        # 策略交易逻辑
        if signal == 1 and share == 0:
            # 满仓买入
            share = capital / open_p * (1 - FEE_RATE)
            capital = 0
        elif signal == 0 and share > 0:
            # 全部卖出
            capital = share * open_p * (1 - FEE_RATE)
            share = 0
        # 持仓净值跟随价格变动
        if share > 0:
            share = share * (1 + ret)
        total = capital + share * open_p
        nav.append(total)

        # 买入持有基准
        bh_cap = bh_cap * (1 + ret)
        bh_nav.append(bh_cap)

    sub_df["strategy_nav"] = nav
    sub_df["bh_nav"] = bh_nav

    # 指标计算
    def metrics(ser):
        ret_ser = ser.pct_change().dropna()
        total_ret = ser.iloc[-1] / ser.iloc[0] - 1
        dd = (ser / ser.cummax() - 1).min()
        if ret_ser.std() == 0:
            sharpe = np.nan
        else:
            sharpe = np.sqrt(252) * ret_ser.mean() / ret_ser.std()
        return total_ret, dd, sharpe

    strat_ret, strat_dd, strat_sharpe = metrics(sub_df["strategy_nav"])
    bh_ret, bh_dd, bh_sharpe = metrics(sub_df["bh_nav"])

    # 绘图
    if SAVE_FIG:
        plt.figure(figsize=(10, 5))
        plt.plot(sub_df["date"], sub_df["strategy_nav"], label=f"{model_name}策略")
        plt.plot(sub_df["date"], sub_df["bh_nav"], label="买入持有", linestyle="--")
        plt.title(f"{model_name} 滚动窗口回测净值曲线")
        plt.xlabel("日期")
        plt.ylabel("账户净值")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        os.makedirs("./backtest_wf_figs", exist_ok=True)
        plt.savefig(f"./backtest_wf_figs/{model_name}.png", dpi=300)
        plt.close()

    return {
        "model": model_name,
        "strategy_total_return": strat_ret,
        "strategy_max_dd": strat_dd,
        "strategy_sharpe": strat_sharpe,
        "bh_total_return": bh_ret,
        "bh_max_dd": bh_dd,
        "bh_sharpe": bh_sharpe,
        "excess_return": strat_ret - bh_ret
    }

# 逐个模型跑
for m in model_list:
    data_m = df_raw[df_raw["Model"] == m].copy()
    res = run_backtest(data_m, m)
    summary.append(res)
    print(f"==== {m} ====")
    print(f"策略收益:{res['strategy_total_return']:.2%} | 最大回撤:{res['strategy_max_dd']:.2%} | 夏普:{res['strategy_sharpe']:.3f}")
    print(f"买入持有收益:{res['bh_total_return']:.2%} | 超额收益:{res['excess_return']:.2%}\n")

summary_df = pd.DataFrame(summary)
if SAVE_SUMMARY:
    summary_df.to_csv(OUTPUT_SUMMARY, index=False, encoding="utf-8-sig")
print(f"汇总结果已保存至 {OUTPUT_SUMMARY}")
