# -*- coding: utf-8 -*-
"""贵州茅台次日开盘涨跌方向分类实验。

研究目的：在收益率回归+阈值法没有泛化的情况下，直接学习次日方向标签。
时间切分：前70%训练，接下来10%验证，最后20%测试。
重要：特征只使用预测日之前已知的 OHLCV 数据；分类器暂时固定使用0.5概率阈值，
不使用测试集选择阈值，以先判断“直接分类目标”是否比回归阈值法更合适。
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix, roc_auc_score,
    average_precision_score,
)

warnings.filterwarnings("ignore")
SEED = 42
TRAIN_RATIO = 0.70
VALIDATION_END_RATIO = 0.80
OUTPUT_DIR = Path(__file__).resolve().parent / "results_direction_classification"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def find_data_file():
    here = Path(__file__).resolve().parent
    candidates = [
        here / "ashare_600519.csv",
        here / "ashare_600519(1).csv",
        here / "ashare_600519 (1).csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "未找到 ashare_600519.csv。请把行情CSV与本脚本放在同一目录，"
        "或在 find_data_file() 中填写实际路径。"
    )


def load_and_make_features(path):
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    required = {"date", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"行情 CSV 缺少必要列：{sorted(missing)}；当前列：{list(df.columns)}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = (df.dropna(subset=["date", "open", "high", "low", "close", "volume"])
            .sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True))
    if len(df) < 200:
        raise ValueError(f"清理后的数据只有 {len(df)} 行，样本可能不足。")

    op = df["open"]
    cl = df["close"]
    hi = df["high"]
    lo = df["low"]
    vol = df["volume"].clip(lower=0)
    open_ret = op.pct_change(fill_method=None)
    close_ret = cl.pct_change(fill_method=None)
    volume_log_change = np.log1p(vol).diff()

    f = pd.DataFrame(index=df.index)
    # 预测起点当日已知的日内/隔夜信息
    f["open_ret_1d"] = open_ret
    f["close_ret_1d"] = close_ret
    f["intraday_return"] = cl / op - 1.0
    f["open_gap_vs_prev_close"] = op / cl.shift(1) - 1.0
    f["high_low_range_pct"] = (hi - lo) / op.replace(0, np.nan)
    f["volume_log_change"] = volume_log_change

    # 最近数日的收益率滞后项。当前行只用当前及过去的数据。
    for lag in [1, 2, 3, 5, 10]:
        f[f"open_ret_lag_{lag}"] = open_ret.shift(lag)
        f[f"close_ret_lag_{lag}"] = close_ret.shift(lag)

    for w in [5, 10, 20]:
        f[f"open_ret_mean_{w}"] = open_ret.rolling(w).mean()
        f[f"open_ret_std_{w}"] = open_ret.rolling(w).std()
        f[f"close_ret_mean_{w}"] = close_ret.rolling(w).mean()
        f[f"close_ret_std_{w}"] = close_ret.rolling(w).std()
        f[f"volume_change_mean_{w}"] = volume_log_change.rolling(w).mean()

    # 行 i 的特征在第 i 天收盘后可知；标签是 Open[i+1] 是否高于 Open[i]。
    next_open = op.shift(-1)
    y = (next_open > op).where(next_open.notna())
    target_index = np.arange(len(df)) + 1
    target_date = df["date"].shift(-1)
    data = f.copy()
    data["target_up"] = y
    data["target_index"] = target_index
    data["target_date"] = target_date
    data["origin_date"] = df["date"]
    data["origin_open"] = op
    data["next_open"] = next_open
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=list(f.columns) + ["target_up", "target_date"]).copy()
    data["target_up"] = data["target_up"].astype(int)
    data["target_index"] = data["target_index"].astype(int)
    return df, data, list(f.columns)


def metric_row(model_name, split_name, y_true, y_pred, y_prob):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    try:
        auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else np.nan
    except Exception:
        auc = np.nan
    try:
        ap = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else np.nan
    except Exception:
        ap = np.nan
    return {
        "Model": model_name, "Split": split_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced_Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Precision_Up": precision_score(y_true, y_pred, zero_division=0),
        "Recall_Up": recall_score(y_true, y_pred, zero_division=0),
        "F1_Up": f1_score(y_true, y_pred, zero_division=0),
        "ROC_AUC": auc, "Average_Precision": ap,
        "Actual_Up_Days": int(y_true.sum()),
        "Predicted_Up_Days": int(y_pred.sum()),
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
        "Samples": int(len(y_true)), "Decision_Threshold": 0.5,
    }


def main():
    path = find_data_file()
    df, data, feature_cols = load_and_make_features(path)
    n = len(df)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * VALIDATION_END_RATIO)

    # target_index is the index of the day being predicted. Split by target date.
    train = data[data["target_index"] < train_end].copy()
    valid = data[(data["target_index"] >= train_end) & (data["target_index"] < val_end)].copy()
    test = data[data["target_index"] >= val_end].copy()
    if min(len(train), len(valid), len(test)) == 0:
        raise ValueError("训练/验证/测试集为空，请检查数据切分和缺失值。")

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
    print(f"训练集：{len(train)} 个样本，目标日期 {train.target_date.min().date()} 至 {train.target_date.max().date()}")
    print(f"验证集：{len(valid)} 个样本，目标日期 {valid.target_date.min().date()} 至 {valid.target_date.max().date()}")
    print(f"测试集：{len(test)} 个样本，目标日期 {test.target_date.min().date()} 至 {test.target_date.max().date()}")
    print(f"训练集上涨比例：{y_train.mean():.3f}；特征数：{len(feature_cols)}")
    print("分类决策阈值固定为 0.5；不根据测试集选择阈值。")

    summary_rows = []
    test_output = pd.DataFrame({
        "date": test["target_date"].dt.strftime("%Y-%m-%d").to_numpy(),
        "origin_date": test["origin_date"].dt.strftime("%Y-%m-%d").to_numpy(),
        "origin_open": test["origin_open"].to_numpy(),
        "actual_next_open": test["next_open"].to_numpy(),
        "true_up": test["target_up"].to_numpy(),
    })

    model_details = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        model_details.append((name, model))
        for split_name, (X_split, y_split, meta) in split_data.items():
            prob = model.predict_proba(X_split)[:, 1]
            pred = (prob >= 0.5).astype(int)
            summary_rows.append(metric_row(name, split_name, y_split, pred, prob))
            if split_name == "Test":
                test_output[f"{name}_prob_up"] = prob
                test_output[f"{name}_pred_up"] = pred

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUT_DIR / "classification_metrics.csv", index=False, encoding="utf-8-sig")
    test_output.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"Feature": feature_cols}).to_csv(OUTPUT_DIR / "feature_list.csv", index=False, encoding="utf-8-sig")

    # Save Random Forest importance for exploratory interpretation only.
    rf_model = dict(model_details)["Random_Forest"]
    imp = pd.DataFrame({"Feature": feature_cols, "Importance": rf_model.feature_importances_})
    imp.sort_values("Importance", ascending=False).to_csv(
        OUTPUT_DIR / "random_forest_feature_importance.csv", index=False, encoding="utf-8-sig"
    )

    cfg = pd.DataFrame([{
        "data_file": path.name, "rows": n, "train_ratio": TRAIN_RATIO,
        "validation_ratio": VALIDATION_END_RATIO - TRAIN_RATIO,
        "test_ratio": 1.0 - VALIDATION_END_RATIO, "past_feature_windows": "lags 1,2,3,5,10; rolling 5,10,20",
        "features_count": len(feature_cols), "seed": SEED, "probability_threshold": 0.5,
        "train_up_rate": float(y_train.mean()),
        "train_start": str(train.target_date.min().date()), "train_end": str(train.target_date.max().date()),
        "validation_start": str(valid.target_date.min().date()), "validation_end": str(valid.target_date.max().date()),
        "test_start": str(test.target_date.min().date()), "test_end": str(test.target_date.max().date()),
    }])
    cfg.to_csv(OUTPUT_DIR / "experiment_config.csv", index=False, encoding="utf-8-sig")

    print("\n===== 分类模型结果（验证集及独立测试集）=====")
    cols = ["Model", "Split", "Accuracy", "Balanced_Accuracy", "Precision_Up", "Recall_Up", "F1_Up", "ROC_AUC", "Predicted_Up_Days", "Actual_Up_Days"]
    print(summary[cols].to_string(index=False, formatters={
        "Accuracy": "{:.4f}".format,
        "Balanced_Accuracy": "{:.4f}".format,
        "Precision_Up": "{:.4f}".format,
        "Recall_Up": "{:.4f}".format,
        "F1_Up": "{:.4f}".format,
        "ROC_AUC": lambda x: "nan" if pd.isna(x) else f"{x:.4f}",
    }))
    print(f"\n结果目录：{OUTPUT_DIR.resolve()}")
    print("请保留 classification_metrics.csv、test_predictions.csv 和 experiment_config.csv。")


if __name__ == "__main__":
    main()
