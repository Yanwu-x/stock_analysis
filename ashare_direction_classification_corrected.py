# -*- coding: utf-8 -*-
"""贵州茅台次日开盘方向分类实验（修正目标口径版）

目标定义：在第 t 个交易日收盘后预测次日开盘相对今日收盘的方向：
    target_up = 1 if Open[t+1] > Close[t], else 0

说明：上一个版本使用 Open[t+1] > Open[t]，但特征包含当天 Close[t]、High[t]、
Low[t]、Volume[t] 及 Close[t]/Open[t]-1。这会让标签覆盖当天已经发生的日内价格变动，
从而产生不公平的目标/特征重叠。此版本改用 Close[t] 作为预测起点，研究次日开盘的隔夜跳空方向。

切分：前70%训练、接下来10%验证、最后20%测试，严格按目标日期切分。
特征：只使用 t 日收盘后已知的 OHLCV 及其历史统计量；Scaler 仅由训练集拟合。
模型：多数类基线、当天 K 线方向简单规则、逻辑回归、随机森林。
分类阈值：概率模型统一使用0.5，不能用测试集挑阈值。
"""
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score, average_precision_score,
)

warnings.filterwarnings("ignore")
SEED = 42
TRAIN_RATIO = 0.70
VALIDATION_END_RATIO = 0.80
OUTPUT_DIR = Path(__file__).resolve().parent / "results_direction_classification_corrected"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def find_data_file():
    here = Path(__file__).resolve().parent
    for name in ["ashare_600519.csv", "ashare_600519(1).csv", "ashare_600519 (1).csv"]:
        candidate = here / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "未找到 ashare_600519.csv。请把行情 CSV 与脚本放在同一目录，"
        "或修改 find_data_file() 中的实际文件路径。"
    )


def load_data_and_features(path):
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    required = {"date", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"行情 CSV 缺少必要列：{sorted(missing)}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = (df.dropna(subset=["date", "open", "high", "low", "close", "volume"])
            .sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True))
    if len(df) < 200:
        raise ValueError(f"清理后的数据只有 {len(df)} 行，样本量可能不足。")

    op, hi, lo, cl = df["open"], df["high"], df["low"], df["close"]
    vol = df["volume"].clip(lower=0)
    open_ret = op.pct_change(fill_method=None)
    close_ret = cl.pct_change(fill_method=None)
    volume_log_change = np.log1p(vol).diff()

    f = pd.DataFrame(index=df.index)
    # 所有特征都截止到第 t 天收盘后，预测 Open[t+1] 与 Close[t] 的关系。
    f["open_ret_1d"] = open_ret
    f["close_ret_1d"] = close_ret
    f["intraday_return"] = cl / op - 1.0
    f["open_gap_vs_prev_close"] = op / cl.shift(1) - 1.0
    f["high_low_range_pct"] = (hi - lo) / op.replace(0, np.nan)
    f["volume_log_change"] = volume_log_change

    for lag in [1, 2, 3, 5, 10]:
        f[f"open_ret_lag_{lag}"] = open_ret.shift(lag)
        f[f"close_ret_lag_{lag}"] = close_ret.shift(lag)

    for w in [5, 10, 20]:
        f[f"open_ret_mean_{w}"] = open_ret.rolling(w).mean()
        f[f"open_ret_std_{w}"] = open_ret.rolling(w).std()
        f[f"close_ret_mean_{w}"] = close_ret.rolling(w).mean()
        f[f"close_ret_std_{w}"] = close_ret.rolling(w).std()
        f[f"volume_change_mean_{w}"] = volume_log_change.rolling(w).mean()

    next_open = op.shift(-1)
    # 核心修正：预测时点是第 t 日收盘后，因此与 Close[t] 比较。
    y = (next_open > cl).where(next_open.notna())
    data = f.copy()
    data["target_up"] = y
    data["target_index"] = np.arange(len(df)) + 1
    data["target_date"] = df["date"].shift(-1)
    data["origin_date"] = df["date"]
    data["origin_open"] = op
    data["origin_close"] = cl
    data["actual_next_open"] = next_open
    data["candle_continuation_pred"] = (cl > op).astype(int)
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=list(f.columns) + ["target_up", "target_date"]).copy()
    data["target_up"] = data["target_up"].astype(int)
    data["target_index"] = data["target_index"].astype(int)
    return df, data, list(f.columns)


def metrics_row(model_name, split_name, y_true, y_pred, y_prob):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else np.nan
    ap = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else np.nan
    return {
        "Model": model_name, "Split": split_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced_Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Precision_Up": precision_score(y_true, y_pred, zero_division=0),
        "Recall_Up": recall_score(y_true, y_pred, zero_division=0),
        "F1_Up": f1_score(y_true, y_pred, zero_division=0),
        "ROC_AUC": auc, "Average_Precision": ap,
        "Actual_Up_Days": int(y_true.sum()), "Predicted_Up_Days": int(y_pred.sum()),
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
        "Samples": int(len(y_true)), "Decision_Threshold": 0.5,
    }


def main():
    path = find_data_file()
    df, data, feature_cols = load_data_and_features(path)
    n = len(df)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * VALIDATION_END_RATIO)
    train = data[data["target_index"] < train_end].copy()
    valid = data[(data["target_index"] >= train_end) & (data["target_index"] < val_end)].copy()
    test = data[data["target_index"] >= val_end].copy()
    if min(len(train), len(valid), len(test)) == 0:
        raise ValueError("训练/验证/测试集为空，请检查数据切分。")

    X_train, y_train = train[feature_cols], train["target_up"].to_numpy()
    split_data = {
        "Validation": (valid[feature_cols], valid["target_up"].to_numpy(), valid),
        "Test": (test[feature_cols], test["target_up"].to_numpy(), test),
    }
    models = {
        "Majority_Baseline": DummyClassifier(strategy="most_frequent"),
        "Logistic_Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=3000, random_state=SEED)),
        ]),
        "Random_Forest": RandomForestClassifier(
            n_estimators=400, max_depth=None, min_samples_leaf=5,
            max_features="sqrt", class_weight=None, random_state=SEED, n_jobs=-1,
        ),
    }

    print(f"数据文件：{path}")
    print(f"原始数据行数：{n}，日期：{df.date.iloc[0].date()} 至 {df.date.iloc[-1].date()}")
    print("目标：预测第 t+1 日 Open 是否高于第 t 日 Close（隔夜跳空方向）")
    print(f"训练 {len(train)} 样本 | 验证 {len(valid)} 样本 | 测试 {len(test)} 样本")
    print(f"训练集目标上涨比例：{y_train.mean():.3f} | 特征数：{len(feature_cols)}")

    result_rows = []
    pred_output = pd.DataFrame({
        "date": test["target_date"].dt.strftime("%Y-%m-%d").to_numpy(),
        "origin_date": test["origin_date"].dt.strftime("%Y-%m-%d").to_numpy(),
        "origin_open": test["origin_open"].to_numpy(),
        "origin_close": test["origin_close"].to_numpy(),
        "actual_next_open": test["actual_next_open"].to_numpy(),
        "true_up": test["target_up"].to_numpy(),
        "Candle_Continuation_Baseline_pred_up": test["candle_continuation_pred"].to_numpy(),
    })

    # 同日K线延续规则：若当日 Close > Open，则预测次日 Open > 当日 Close。
    # 它只使用当前收盘后已知信息，作为简单启发式基线。
    for split_name, (_, y_split, meta) in split_data.items():
        pred = meta["candle_continuation_pred"].to_numpy(dtype=int)
        result_rows.append(metrics_row(
            "Candle_Continuation_Baseline", split_name, y_split, pred, pred.astype(float)
        ))

    for name, model in models.items():
        model.fit(X_train, y_train)
        for split_name, (X_split, y_split, meta) in split_data.items():
            prob = model.predict_proba(X_split)[:, 1]
            pred = (prob >= 0.5).astype(int)
            result_rows.append(metrics_row(name, split_name, y_split, pred, prob))
            if split_name == "Test":
                pred_output[f"{name}_prob_up"] = prob
                pred_output[f"{name}_pred_up"] = pred

    metrics = pd.DataFrame(result_rows)
    metrics.to_csv(OUTPUT_DIR / "classification_metrics.csv", index=False, encoding="utf-8-sig")
    pred_output.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"Feature": feature_cols}).to_csv(OUTPUT_DIR / "feature_list.csv", index=False, encoding="utf-8-sig")

    rf = RandomForestClassifier(
        n_estimators=400, max_depth=None, min_samples_leaf=5,
        max_features="sqrt", class_weight=None, random_state=SEED, n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    pd.DataFrame({"Feature": feature_cols, "Importance": rf.feature_importances_}) \
        .sort_values("Importance", ascending=False).to_csv(
            OUTPUT_DIR / "random_forest_feature_importance.csv", index=False, encoding="utf-8-sig"
        )

    config = pd.DataFrame([{
        "data_file": path.name, "rows": n, "train_ratio": TRAIN_RATIO,
        "validation_ratio": VALIDATION_END_RATIO - TRAIN_RATIO,
        "test_ratio": 1 - VALIDATION_END_RATIO,
        "target_definition": "Open[t+1] > Close[t]",
        "features_count": len(feature_cols), "seed": SEED, "probability_threshold": 0.5,
        "train_up_rate": float(y_train.mean()),
        "train_start": str(train.target_date.min().date()), "train_end": str(train.target_date.max().date()),
        "validation_start": str(valid.target_date.min().date()), "validation_end": str(valid.target_date.max().date()),
        "test_start": str(test.target_date.min().date()), "test_end": str(test.target_date.max().date()),
    }])
    config.to_csv(OUTPUT_DIR / "experiment_config.csv", index=False, encoding="utf-8-sig")

    display_cols = ["Model", "Split", "Accuracy", "Balanced_Accuracy", "Precision_Up", "Recall_Up", "F1_Up", "ROC_AUC", "Predicted_Up_Days", "Actual_Up_Days"]
    print("\n===== 修正目标口径后的分类结果 =====")
    print(metrics[display_cols].to_string(index=False, formatters={
        col: "{:.4f}".format for col in ["Accuracy", "Balanced_Accuracy", "Precision_Up", "Recall_Up", "F1_Up", "ROC_AUC"]
    }))
    print(f"\n结果目录：{OUTPUT_DIR.resolve()}")
    print("请保留 classification_metrics.csv、test_predictions.csv、experiment_config.csv 与 random_forest_feature_importance.csv。")


if __name__ == "__main__":
    main()
