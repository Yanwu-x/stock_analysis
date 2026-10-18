# -*- coding: utf-8 -*-
"""贵州茅台次日涨跌：用时间顺序验证集选择收益率阈值，再用测试集做一次性评价。

数据切分（按时间顺序）：前70%训练 / 接下来10%验证 / 最后20%测试。
- LSTM 只在前70%训练；验证集只用于选择涨跌阈值；测试集绝不参与阈值选择。
- 预测目标为次日开盘收益率：(Open[t] - Open[t-1]) / Open[t-1]
- 输入为预测日之前90个交易日的 Open；Scaler 仅在训练段拟合。
- 比较：始终预测不涨的朴素方向基线、收益率预测阈值0、验证集选出的阈值。

注意：由于模型只用前70%训练，本脚本的测试结果不能直接与之前“前80%训练/后20%测试”的模型结果做数值优劣比较。
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import random
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings("ignore")

PAST_DAYS = 90
TRAIN_RATIO = 0.70
VALIDATION_END_RATIO = 0.80  # 70% train + 10% validation; last 20% is final test
EPOCHS = 100
BATCH_SIZE = 32
SEED = 42
OUTPUT_DIR = Path(__file__).resolve().parent / "results_return_threshold_validation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed=SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def find_data_file():
    here = Path(__file__).resolve().parent
    candidates = [
        here / "ashare_600519.csv",
        here / "ashare_600519(1).csv",
        here / "ashare_600519 (1).csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "未找到行情文件。请将 ashare_600519.csv 与本脚本放在同一目录，"
        "或把 DATA_FILE 路径改成你的实际文件名。"
    )


def load_data():
    path = find_data_file()
    df = pd.read_csv(path)
    if not {"date", "open"}.issubset(df.columns):
        raise ValueError(f"数据至少需要 date、open 列；实际列：{list(df.columns)}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["open"] = pd.to_numeric(df["open"], errors="coerce")
    df = (df.dropna(subset=["date", "open"])
            .sort_values("date")
            .drop_duplicates("date", keep="last")
            .reset_index(drop=True))
    if len(df) <= PAST_DAYS + 30:
        raise ValueError(f"有效样本不足：{len(df)} 行")
    return path, df


def build_model():
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(PAST_DAYS, 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(1),
    ])
    model.compile(optimizer=Adam(), loss="mean_squared_error")
    return model


def confusion_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "Accuracy": float((y_true == y_pred).mean()),
        "Balanced_Accuracy": float((recall + specificity) / 2),
        "Precision_Up": precision,
        "Recall_Up": recall,
        "F1_Up": f1,
        "Actual_Up_Days": int(y_true.sum()),
        "Predicted_Up_Days": int(y_pred.sum()),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "Samples": int(len(y_true)),
    }


def select_threshold_by_validation(y_true_up, predicted_return):
    """Choose threshold maximizing validation Balanced Accuracy; ties choose closest to 0."""
    unique = np.unique(np.asarray(predicted_return, dtype=float))
    if len(unique) == 1:
        candidates = np.array([0.0, unique[0]])
    else:
        midpoints = (unique[:-1] + unique[1:]) / 2.0
        candidates = np.unique(np.concatenate([midpoints, np.array([0.0, unique.min() - 1e-12, unique.max() + 1e-12])]))
    rows = []
    for threshold in candidates:
        pred_up = (predicted_return > threshold).astype(int)
        metrics = confusion_metrics(y_true_up, pred_up)
        rows.append({"Threshold": float(threshold), **metrics})
    sweep = pd.DataFrame(rows)
    # Deterministic selection: best balanced accuracy; then threshold closest to 0; then larger F1.
    sweep["Abs_Threshold"] = sweep["Threshold"].abs()
    sweep = sweep.sort_values(
        ["Balanced_Accuracy", "Abs_Threshold", "F1_Up"],
        ascending=[False, True, False]
    ).reset_index(drop=True)
    best = float(sweep.loc[0, "Threshold"])
    return best, sweep


def main():
    set_seed(SEED)
    data_path, df = load_data()
    dates = df["date"].dt.strftime("%Y-%m-%d").to_numpy()
    opens = df["open"].to_numpy(dtype=np.float64)
    n = len(df)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * VALIDATION_END_RATIO)
    test_start = val_end
    if train_end <= PAST_DAYS + 1 or val_end <= train_end or n - test_start < 2:
        raise ValueError("训练/验证/测试切分后样本不足，请检查数据量和比例。")

    # Next-day simple return at target index t is based on open[t] vs open[t-1].
    returns = np.zeros(n, dtype=np.float64)
    returns[1:] = (opens[1:] - opens[:-1]) / opens[:-1]

    # Fit feature scaler on training period only. Later observations are transformed, never fitted.
    input_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_open = np.empty((n, 1), dtype=np.float64)
    scaled_open[:train_end] = input_scaler.fit_transform(opens[:train_end].reshape(-1, 1))
    scaled_open[train_end:] = input_scaler.transform(opens[train_end:].reshape(-1, 1))

    # Fit target scaler on return labels available inside training period only.
    ret_scaler = StandardScaler()
    ret_scaler.fit(returns[PAST_DAYS:train_end].reshape(-1, 1))
    scaled_returns = np.full((n, 1), np.nan, dtype=np.float64)
    scaled_returns[PAST_DAYS:train_end] = ret_scaler.transform(returns[PAST_DAYS:train_end].reshape(-1, 1))
    # Transform validation/test labels for evaluation consistency; this does not fit anything.
    scaled_returns[train_end:] = ret_scaler.transform(returns[train_end:].reshape(-1, 1))

    # Each target t uses only opens[t-PAST_DAYS:t], excluding target-day Open.
    X_all, target_indices = [], []
    for t in range(PAST_DAYS, n):
        X_all.append(scaled_open[t - PAST_DAYS:t, :])
        target_indices.append(t)
    X_all = np.asarray(X_all, dtype=np.float32)
    target_indices = np.asarray(target_indices, dtype=int)
    train_mask = target_indices < train_end
    val_mask = (target_indices >= train_end) & (target_indices < val_end)
    test_mask = target_indices >= test_start

    X_train = X_all[train_mask]
    y_train = scaled_returns[target_indices[train_mask]].astype(np.float32)
    X_val = X_all[val_mask]
    X_test = X_all[test_mask]
    idx_val = target_indices[val_mask]
    idx_test = target_indices[test_mask]

    print(f"数据文件：{data_path}")
    print(f"总样本：{n} | 日期：{dates[0]} 至 {dates[-1]}")
    print(f"训练：前{train_end}行（{dates[0]} 至 {dates[train_end-1]}）")
    print(f"验证：{val_end-train_end}行（{dates[train_end]} 至 {dates[val_end-1]}）")
    print(f"测试：{n-test_start}行（{dates[test_start]} 至 {dates[-1]}）")
    print("注意：阈值只从验证集选择，测试集不参与阈值选择。")

    tf.keras.backend.clear_session()
    set_seed(SEED)
    model = build_model()
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=False, verbose=1)

    val_pred_scaled = model.predict(X_val, verbose=0).reshape(-1, 1)
    test_pred_scaled = model.predict(X_test, verbose=0).reshape(-1, 1)
    val_pred_return = ret_scaler.inverse_transform(val_pred_scaled).reshape(-1)
    test_pred_return = ret_scaler.inverse_transform(test_pred_scaled).reshape(-1)
    val_true_up = (returns[idx_val] > 0).astype(int)
    test_true_up = (returns[idx_test] > 0).astype(int)

    best_threshold, sweep = select_threshold_by_validation(val_true_up, val_pred_return)
    sweep.to_csv(OUTPUT_DIR / "validation_threshold_sweep.csv", index=False, encoding="utf-8-sig")
    val_metrics_zero = confusion_metrics(val_true_up, (val_pred_return > 0).astype(int))
    val_metrics_best = confusion_metrics(val_true_up, (val_pred_return > best_threshold).astype(int))

    # Forecast price using the predicted return. The threshold controls only the direction signal.
    test_previous_open = opens[idx_test - 1]
    test_actual_open = opens[idx_test]
    test_pred_open = test_previous_open * (1.0 + test_pred_return)
    rows = []

    # Majority-down / persistence direction baseline: predicted price = previous actual Open.
    naive_pred_open = test_previous_open.copy()
    naive_pred_up = np.zeros(len(idx_test), dtype=int)
    naive_cls = confusion_metrics(test_true_up, naive_pred_up)
    rows.append({
        "Model": "Naive_Always_Down", "Threshold": np.nan,
        "MAE": float(mean_absolute_error(test_actual_open, naive_pred_open)),
        "RMSE": float(np.sqrt(mean_squared_error(test_actual_open, naive_pred_open))),
        **naive_cls,
    })

    for name, threshold in [("Return_LSTM_Threshold_0", 0.0), ("Return_LSTM_Threshold_ValidationSelected", best_threshold)]:
        pred_up = (test_pred_return > threshold).astype(int)
        cls = confusion_metrics(test_true_up, pred_up)
        rows.append({
            "Model": name,
            "Threshold": float(threshold),
            "MAE": float(mean_absolute_error(test_actual_open, test_pred_open)),
            "RMSE": float(np.sqrt(mean_squared_error(test_actual_open, test_pred_open))),
            **cls,
        })
    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT_DIR / "test_threshold_comparison.csv", index=False, encoding="utf-8-sig")
    try:
        results.to_excel(OUTPUT_DIR / "test_threshold_comparison.xlsx", index=False)
    except Exception as exc:
        print(f"提示：Excel 保存失败（CSV已保存）：{exc}")

    predictions = pd.DataFrame({
        "date": dates[idx_test],
        "previous_actual_open": test_previous_open,
        "actual_open": test_actual_open,
        "predicted_open_from_return": test_pred_open,
        "actual_return": returns[idx_test],
        "predicted_return": test_pred_return,
        "true_up": test_true_up,
        "predicted_up_threshold_0": (test_pred_return > 0).astype(int),
        "predicted_up_selected_threshold": (test_pred_return > best_threshold).astype(int),
    })
    predictions.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False, encoding="utf-8-sig")

    meta = pd.DataFrame([{
        "data_file": data_path.name,
        "rows": n,
        "train_rows": train_end,
        "validation_rows": val_end-train_end,
        "test_rows": n-test_start,
        "train_start": dates[0],
        "train_end": dates[train_end-1],
        "validation_start": dates[train_end],
        "validation_end": dates[val_end-1],
        "test_start": dates[test_start],
        "test_end": dates[-1],
        "selected_threshold": best_threshold,
        "validation_balanced_accuracy_threshold0": val_metrics_zero["Balanced_Accuracy"],
        "validation_balanced_accuracy_selected": val_metrics_best["Balanced_Accuracy"],
        "seed": SEED, "past_days": PAST_DAYS, "epochs": EPOCHS, "batch_size": BATCH_SIZE,
    }])
    meta.to_csv(OUTPUT_DIR / "experiment_config_and_selected_threshold.csv", index=False, encoding="utf-8-sig")

    # Plot the validation threshold sweep; final test results remain in CSV to avoid selecting by test plots.
    plt.figure(figsize=(10, 5))
    plot_sweep = sweep.sort_values("Threshold")
    plt.plot(plot_sweep["Threshold"] * 100, plot_sweep["Balanced_Accuracy"], label="Validation Balanced Accuracy")
    plt.axvline(0, linestyle="--", label="Threshold = 0")
    plt.axvline(best_threshold * 100, linestyle="--", label=f"Selected = {best_threshold:.6f}")
    plt.xlabel("Upward-return threshold (%)")
    plt.ylabel("Balanced Accuracy")
    plt.title("Select Return Threshold on Validation Set Only")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "validation_threshold_sweep.png", dpi=180)
    plt.close()

    print("\n===== 验证集阈值选择 =====")
    print(f"选定阈值：{best_threshold:.8f}（收益率约 {best_threshold*100:.4f}%）")
    print(f"阈值0验证集 Balanced Accuracy：{val_metrics_zero['Balanced_Accuracy']:.4f}")
    print(f"选定阈值验证集 Balanced Accuracy：{val_metrics_best['Balanced_Accuracy']:.4f}")
    print("\n===== 最终测试集结果（仅用于最终评价）=====")
    cols = ["Model", "Threshold", "Accuracy", "Balanced_Accuracy", "Precision_Up", "Recall_Up", "F1_Up", "Actual_Up_Days", "Predicted_Up_Days", "MAE", "RMSE"]
    print(results[cols].to_string(index=False))
    print(f"\n结果目录：{OUTPUT_DIR.resolve()}")
    print("优先查看 test_threshold_comparison.csv、validation_threshold_sweep.csv 和 test_predictions.csv。")
    print("不要根据测试集结果再次调整阈值；若需要改阈值，应重新设计验证流程并留出新的测试集。")


if __name__ == "__main__":
    main()
