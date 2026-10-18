# -*- coding: utf-8 -*-
"""RAW Open + GRU baseline. Keep settings identical to the existing RAW-LSTM."""
import os, random, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, GRU, Dropout, Dense
from tensorflow.keras.optimizers import Adam

DATA_FILE = "ashare_600519.csv"
OUTPUT_DIR = "results_gru"
TRAIN_RATIO = 0.80
PAST_DAYS = 90
EPOCHS = 100
BATCH_SIZE = 32
SEED = 42
os.makedirs(OUTPUT_DIR, exist_ok=True)

def set_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)
    try: tf.config.experimental.enable_op_determinism()
    except Exception: pass

def load_data():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"找不到 {DATA_FILE}，请与本脚本放在同一目录")
    df = pd.read_csv(DATA_FILE)
    for c in ["date", "open"]:
        if c not in df.columns: raise ValueError(f"缺少字段：{c}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["open"] = pd.to_numeric(df["open"], errors="coerce")
    df = df.dropna(subset=["date", "open"]).sort_values("date").reset_index(drop=True)
    print(f"数据量：{len(df)} | {df['date'].iloc[0].date()} -> {df['date'].iloc[-1].date()}")
    return df

def create_train_sequences(scaled, past_days):
    X, y = [], []
    for i in range(past_days, len(scaled)):
        X.append(scaled[i-past_days:i]); y.append(scaled[i])
    return np.array(X), np.array(y)

def create_test_sequences(train_scaled, test_scaled, past_days):
    combined = np.concatenate([train_scaled[-past_days:], test_scaled], axis=0)
    X, y = [], []
    for i in range(past_days, len(combined)):
        X.append(combined[i-past_days:i]); y.append(combined[i])
    return np.array(X), np.array(y)

def build_gru():
    model = Sequential([
        Input(shape=(PAST_DAYS, 1)),
        GRU(50, return_sequences=True), Dropout(0.2),
        GRU(50, return_sequences=True), Dropout(0.2),
        GRU(50, return_sequences=True), Dropout(0.2),
        GRU(50), Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer=Adam(), loss="mean_squared_error")
    return model

def old_da(y_true, y_pred):
    return float(np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred))))

def strict_direction(train_values, y_true, y_pred):
    prev = np.concatenate([[train_values[-1]], y_true[:-1]])
    pred_up = y_pred > prev
    true_up = y_true > prev
    return {
        "Accuracy": float(accuracy_score(true_up, pred_up)),
        "Precision": float(precision_score(true_up, pred_up, zero_division=0)),
        "Recall": float(recall_score(true_up, pred_up, zero_division=0)),
        "F1": float(f1_score(true_up, pred_up, zero_division=0)),
    }

def main():
    set_seed(SEED)
    df = load_data()
    split = int(len(df) * TRAIN_RATIO)
    train = df.iloc[:split].copy(); test = df.iloc[split:].copy()
    train_values = train["open"].to_numpy(dtype=float)
    test_values = test["open"].to_numpy(dtype=float)
    dates = test["date"].dt.strftime("%Y-%m-%d").to_numpy()

    scaler = MinMaxScaler((0, 1))
    train_scaled = scaler.fit_transform(train_values.reshape(-1, 1))
    test_scaled = scaler.transform(test_values.reshape(-1, 1))
    X_train, y_train = create_train_sequences(train_scaled, PAST_DAYS)
    X_test, y_test = create_test_sequences(train_scaled, test_scaled, PAST_DAYS)
    print(f"Train: {len(train_values)} | Test: {len(test_values)}")
    print(f"X_train: {X_train.shape} | X_test: {X_test.shape}")

    set_seed(SEED)
    model = build_gru()
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=False, verbose=1)

    pred_scaled = model.predict(X_test, verbose=0)
    y_pred = scaler.inverse_transform(pred_scaled).ravel()
    y_true = scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    legacy_da = old_da(y_true, y_pred)
    strict = strict_direction(train_values, y_true, y_pred)

    pd.DataFrame({"date": dates, "actual_open": y_true, "predicted_open": y_pred}).to_csv(
        f"{OUTPUT_DIR}/gru_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{
        "Model": "RAW-GRU", "MAE": mae, "RMSE": rmse,
        "Old_DA": legacy_da, "Strict_Accuracy": strict["Accuracy"],
        "Strict_Precision": strict["Precision"], "Strict_Recall": strict["Recall"], "Strict_F1": strict["F1"],
        "Train_Samples": len(X_train), "Test_Samples": len(X_test),
        "Past_Days": PAST_DAYS, "Epochs": EPOCHS, "Batch_Size": BATCH_SIZE, "Seed": SEED
    }]).to_csv(f"{OUTPUT_DIR}/gru_result.csv", index=False, encoding="utf-8-sig")

    print("\n===== RAW-GRU RESULT =====")
    print(f"MAE = {mae:.4f}")
    print(f"RMSE = {rmse:.4f}")
    print(f"旧版 DA = {legacy_da:.4f} ({legacy_da*100:.2f}%)")
    print(f"严格次日 Accuracy = {strict['Accuracy']:.4f} ({strict['Accuracy']*100:.2f}%)")
    print(f"Precision = {strict['Precision']:.4f}")
    print(f"Recall = {strict['Recall']:.4f}")
    print(f"F1 = {strict['F1']:.4f}")
    print(f"结果文件：{OUTPUT_DIR}/gru_result.csv")
    print(f"预测文件：{OUTPUT_DIR}/gru_predictions.csv")

if __name__ == "__main__":
    main()
