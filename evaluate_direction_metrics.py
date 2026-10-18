# -*- coding: utf-8 -*-
"""
统一计算已有股票预测结果的"严格次日方向指标"。
默认会读取：
  results_causal_wavelet_corrected/naive_predictions.csv
  results_causal_wavelet_corrected/raw_lstm_predictions.csv
  results_causal_wavelet_corrected/causal_DB4_L2_predictions.csv
  results_causal_wavelet_corrected/causal_DB4_L3_predictions.csv
  results_causal_wavelet_corrected/causal_DB4_L4_predictions.csv
  results_gru/gru_predictions.csv
ARIMA 暂时不自动扫描，避免把旧的多组 ARIMA CSV 重复识别。确认具体预测文件后再加入。
需要和 ashare_600519.csv 放在同一项目目录中运行。
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd

ROOT = Path(".")
OUT = ROOT / "results_direction_eval"
DATA_NAMES = ["ashare_600519.csv", "ashare_600519(1).csv"]

KNOWN = [
    ("Naive", "results_causal_wavelet_corrected/naive_predictions.csv"),
    ("RAW-LSTM", "results_causal_wavelet_corrected/raw_lstm_predictions.csv"),
    ("Causal Wavelet-L2", "results_causal_wavelet_corrected/causal_DB4_L2_predictions.csv"),
    ("Causal Wavelet-L3", "results_causal_wavelet_corrected/causal_DB4_L3_predictions.csv"),
    ("Causal Wavelet-L4", "results_causal_wavelet_corrected/causal_DB4_L4_predictions.csv"),
    ("RAW-GRU", "results_gru/gru_predictions.csv"),
]

def find_data():
    for name in DATA_NAMES:
        p = ROOT / name
        if p.exists():
            return p
    raise FileNotFoundError(f"找不到原始数据：{DATA_NAMES}")

def normalize_market(df):
    cols = {str(c).strip().lower(): c for c in df.columns}
    if "date" not in cols or "open" not in cols:
        raise ValueError(f"原始数据必须包含 date/open，当前字段：{list(df.columns)}")
    df = df.rename(columns={cols["date"]: "date", cols["open"]: "open"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["open"] = pd.to_numeric(df["open"], errors="coerce")
    return (df.dropna(subset=["date", "open"])
              .sort_values("date")
              .drop_duplicates("date", keep="last")
              .reset_index(drop=True))

def normalize_pred(df):
    lookup = {str(c).strip().lower(): c for c in df.columns}
    actual = next((lookup[k] for k in ["actual_open", "real_open", "actualopen", "realopen"] if k in lookup), None)
    pred = next((lookup[k] for k in ["predicted_open", "pred_open", "predictedopen", "predicted"] if k in lookup), None)
    date = lookup.get("date")
    if not all([date, actual, pred]):
        raise ValueError(f"预测文件必须有 date/actual_open(or real_open)/predicted_open，当前字段：{list(df.columns)}")
    out = df.rename(columns={date: "date", actual: "actual_open", pred: "predicted_open"})
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["actual_open"] = pd.to_numeric(out["actual_open"], errors="coerce")
    out["predicted_open"] = pd.to_numeric(out["predicted_open"], errors="coerce")
    return (out.dropna(subset=["date", "actual_open", "predicted_open"])
               .sort_values("date")
               .drop_duplicates("date", keep="last")
               .reset_index(drop=True))

def old_da(y_true, y_pred):
    if len(y_true) < 2:
        return np.nan
    return float(np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred))))

def strict_metrics(pred, market):
    m = market[["date", "open"]].copy()
    m["previous_open"] = m["open"].shift(1)
    x = pred.merge(m[["date", "open", "previous_open"]], on="date", how="left", suffixes=("", "_market"))
    x = x.dropna(subset=["previous_open"]).copy()
    if x.empty:
        raise ValueError("预测日期无法匹配原始数据，请检查日期格式。")
    true_up = x["actual_open"] > x["previous_open"]
    pred_up = x["predicted_open"] > x["previous_open"]
    tp = int((pred_up & true_up).sum())
    tn = int(((~pred_up) & (~true_up)).sum())
    fp = int((pred_up & (~true_up)).sum())
    fn = int(((~pred_up) & true_up).sum())
    n = tp + tn + fp + fn
    acc = (tp + tn) / n if n else np.nan
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    detail = x[["date", "previous_open", "actual_open", "predicted_open"]].copy()
    detail["true_direction"] = np.where(true_up, "UP", "DOWN")
    detail["predicted_direction"] = np.where(pred_up, "UP", "DOWN")
    detail["correct"] = detail["true_direction"] == detail["predicted_direction"]
    return {
        "Strict_Accuracy": acc,
        "Strict_Precision": precision,
        "Strict_Recall": recall,
        "Strict_F1": f1,
        "Test_Samples": n,
        "True_Up_Days": int(true_up.sum()),
        "True_Down_Days": int((~true_up).sum()),
        "Pred_Up_Days": int(pred_up.sum()),
        "Pred_Down_Days": int((~pred_up).sum()),
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "Actual_Open_Max_Abs_Diff": float(np.abs(x["actual_open"] - x["open"]).max()),
    }, detail

def collect_files():
    items = []
    for model, rel in KNOWN:
        p = ROOT / rel
        if p.exists():
            items.append((model, p))
        else:
            print(f"[未找到] {model}: {p}")
    return items

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data_path = find_data()
    market = normalize_market(pd.read_csv(data_path))
    print(f"原始数据：{data_path} | {len(market)} 行")
    files = collect_files()
    if not files:
        raise FileNotFoundError("没有找到预测 CSV。")
    rows = []
    for model, path in files:
        try:
            pred = normalize_pred(pd.read_csv(path))
            strict, detail = strict_metrics(pred, market)
            rows.append({
                "Model": model,
                "Old_DA": old_da(pred["actual_open"].to_numpy(), pred["predicted_open"].to_numpy()),
                **strict,
                "Prediction_File": str(path),
            })
            safe = re.sub(r"[^A-Za-z0-9_-]+", "_", model)
            detail.to_csv(OUT / f"direction_details_{safe}.csv", index=False, encoding="utf-8-sig")
            print(f"\n{model}")
            print(f"  Strict Accuracy : {strict['Strict_Accuracy']:.4f} ({strict['Strict_Accuracy']*100:.2f}%)")
            print(f"  Precision       : {strict['Strict_Precision']:.4f}")
            print(f"  Recall          : {strict['Strict_Recall']:.4f}")
            print(f"  F1              : {strict['Strict_F1']:.4f}")
            print(f"  Test Samples    : {strict['Test_Samples']}")
        except Exception as e:
            print(f"[跳过] {model}: {e}")
    if not rows:
        raise RuntimeError("所有预测文件都处理失败。")
    order = {m: i for i, m in enumerate(["Naive", "ARIMA(5,1,0)", "RAW-LSTM", "RAW-GRU", "Causal Wavelet-L2", "Causal Wavelet-L3", "Causal Wavelet-L4"])}
    out = pd.DataFrame(rows)
    out["_order"] = out["Model"].map(order).fillna(999)
    out = out.sort_values(["_order", "Model"]).drop(columns="_order").reset_index(drop=True)
    csv = OUT / "direction_summary.csv"
    xlsx = OUT / "direction_summary.xlsx"
    out.to_csv(csv, index=False, encoding="utf-8-sig")
    try:
        out.to_excel(xlsx, index=False, sheet_name="Summary")
    except Exception as e:
        print(f"[提示] Excel 保存失败：{e}")
        xlsx = None
    print("\n===== 统一方向指标汇总 =====")
    print(out[["Model", "Old_DA", "Strict_Accuracy", "Strict_Precision", "Strict_Recall", "Strict_F1", "Test_Samples"]].to_string(index=False))
    print(f"\nCSV：{csv}")
    if xlsx:
        print(f"Excel：{xlsx}")

if __name__ == "__main__":
    main()
