import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline


def prepare_features(data, target_mode="binary"):
    """
    Engineer comprehensive technical features using historical information only.
    
    target_mode:
        - "binary": Directional trend ('UP' if next day closes higher, else 'DOWN').
                    Recommended for real trading and cross-asset consistency (AAPL, MSFT, RELIANCE).
        - "3class": Adaptive 3-class ('UP', 'DOWN', 'STABLE') calibrated dynamically to
                    the stock's rolling Average True Range (ATR).
    """
    data = data.copy()

    close = data["Close"]
    high = data["High"]
    low = data["Low"]
    volume = data["Volume"]

    # 1. Multi-period Returns
    for d in [1, 2, 3, 5, 10]:
        data[f"Return_{d}d"] = close.pct_change(d)

    # 2. Moving Average Distances (SMA & EMA)
    for w in [5, 10, 20, 50]:
        data[f"MA_{w}_Distance"] = close / close.rolling(w).mean() - 1
        data[f"EMA_{w}_Distance"] = close / close.ewm(span=w).mean() - 1

    # 3. Relative Strength Index (RSI - 14 period)
    delta = close.diff()
    gain = (delta.clip(lower=0)).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss.replace(0, np.nan))
    data["RSI_14"] = 100 - (100 / (1 + rs))

    # 4. Moving Average Convergence Divergence (MACD 12, 26, 9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    data["MACD"] = macd / close
    data["MACD_Signal"] = signal / close
    data["MACD_Hist"] = (macd - signal) / close

    # 5. Bollinger Bands & Normalized Z-Score
    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    data["Z_Score_20"] = (close - ma20) / std20.replace(0, np.nan)
    data["BB_PctB"] = (close - (ma20 - 2 * std20)) / (4 * std20).replace(0, np.nan)
    data["BB_Width"] = (4 * std20) / ma20

    # 6. Average True Range (ATR - 14 period, normalized by price)
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    data["ATR_14_Pct"] = atr / close

    # 7. Support & Resistance Bounds (20-day rolling channel)
    rolling_low20 = low.rolling(20).min()
    rolling_high20 = high.rolling(20).max()
    data["Support_Distance"] = close / rolling_low20 - 1
    data["Resistance_Distance"] = rolling_high20 / close - 1
    data["Price_Range_20"] = (rolling_high20 - rolling_low20) / rolling_low20

    # 8. Volatility & Consolidation
    data["Volatility_20"] = data["Return_1d"].rolling(20).std()
    data["Consolidation"] = (data["Price_Range_20"] < 0.05).astype(int)

    # 9. Volume Dynamics
    data["Volume_Change"] = volume.pct_change()
    data["Volume_MA20_Ratio"] = volume / (volume.rolling(20).mean().replace(0, np.nan))

    # 10. Candlestick Geometry
    bar_range = (high - low).replace(0, np.nan)
    data["Body_To_Range"] = (close - data["Open"]).abs() / bar_range
    data["Upper_Shadow"] = (high - np.maximum(close, data["Open"])) / bar_range
    data["Lower_Shadow"] = (np.minimum(close, data["Open"]) - low) / bar_range

    # Target calculation
    next_return = close.shift(-1) / close - 1

    if target_mode == "3class":
        # Dynamic ATR threshold: adapts to the asset's natural volatility
        dynamic_threshold = 0.35 * data["ATR_14_Pct"]
        data["Target"] = np.select(
            [next_return > dynamic_threshold, next_return < -dynamic_threshold],
            ["UP", "DOWN"],
            default="STABLE",
        )
    else:
        # Binary Directional Trend: Standard UP vs DOWN
        data["Target"] = np.where(next_return > 0, "UP", "DOWN")

    # The most recent row has unknown next-day return
    data.loc[next_return.isna(), "Target"] = np.nan

    return data


def train_model(target_mode="binary"):
    try:
        data = pd.read_csv("stock_data.csv", index_col=0)

        if data.empty:
            print("Error: stock_data.csv is empty.")
            return

        data = prepare_features(data, target_mode=target_mode)

        excluded_columns = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "Dividends",
            "Stock Splits",
            "Target",
        ]
        feature_columns = [c for c in data.columns if c not in excluded_columns]

        data = data.replace([np.inf, -np.inf], np.nan)

        # Retain the latest session for tomorrow's forecast
        latest_unlabeled = data[
            data["Target"].isna() & data[feature_columns].notna().all(axis=1)
        ]

        # Clean training set
        clean_data = data.dropna(subset=["Target"] + feature_columns)

        print(f"\nTarget distribution across historical data ({target_mode.upper()} mode):")
        print(clean_data["Target"].value_counts())

        split_index = int(len(clean_data) * 0.8)

        train = clean_data.iloc[:split_index].copy()
        test = clean_data.iloc[split_index:].copy()

        if len(train) < 100 or test.empty:
            print("Error: Not enough data to train and test.")
            return

        X_train = train[feature_columns]
        y_train = train["Target"]

        X_test = test[feature_columns]
        y_test = test["Target"]

        # Ensemble Classifier: Random Forest + HistGradientBoosting for cross-asset robustness
        imputer = SimpleImputer(strategy="median")
        X_train_imp = imputer.fit_transform(X_train)
        X_test_imp = imputer.transform(X_test)

        rf = RandomForestClassifier(
            n_estimators=300,
            max_depth=5,
            min_samples_leaf=15,
            max_features="sqrt",
            random_state=42,
        )
        hgb = HistGradientBoostingClassifier(
            max_iter=100,
            max_depth=3,
            min_samples_leaf=20,
            random_state=42,
        )
        model = VotingClassifier([("rf", rf), ("hgb", hgb)], voting="soft")
        model.fit(X_train_imp, y_train)

        predictions = model.predict(X_test_imp)
        probabilities = model.predict_proba(X_test_imp)

        # Baseline
        baseline = DummyClassifier(strategy="most_frequent")
        baseline.fit(X_train_imp, y_train)
        baseline_predictions = baseline.predict(X_test_imp)

        model_accuracy = accuracy_score(y_test, predictions)
        baseline_accuracy = accuracy_score(y_test, baseline_predictions)

        # High-confidence accuracy (conviction >= 54%)
        max_p = np.max(probabilities, axis=1)
        confident_mask = max_p >= 0.54
        if confident_mask.sum() >= 15:
            confident_acc = accuracy_score(
                y_test[confident_mask], predictions[confident_mask]
            )
            conf_str = f"{confident_acc:.2%} ({confident_mask.sum()}/{len(y_test)} sessions)"
        else:
            conf_str = "N/A"

        print("\n--- Model Evaluation ---")
        print(f"Baseline accuracy (Dummy):             {baseline_accuracy:.2%}")
        print(f"Ensemble Model accuracy:               {model_accuracy:.2%}")
        print(f"High-Confidence Accuracy (>=54% prob): {conf_str}")
        print(f"Training records:                      {len(train)}")
        print(f"Testing records:                       {len(test)}")

        print("\nClassification Report:")
        print(classification_report(y_test, predictions, zero_division=0))

        # Save predictions
        results = test[["Close", "Target"]].copy()
        results["Predicted_Trend"] = predictions
        classes = model.classes_
        for i, c in enumerate(classes):
            results[f"Prob_{c}"] = probabilities[:, i]
        results.to_csv("model_predictions.csv")
        print("Predictions saved to model_predictions.csv")

        # Next-Day Forecast
        if not latest_unlabeled.empty:
            latest_features = latest_unlabeled[feature_columns].iloc[[-1]]
            latest_date = latest_unlabeled.index[-1]
        else:
            latest_features = clean_data[feature_columns].iloc[[-1]]
            latest_date = clean_data.index[-1]

        latest_feat_imp = imputer.transform(latest_features)
        next_prediction = model.predict(latest_feat_imp)[0]
        next_proba = model.predict_proba(latest_feat_imp)[0]
        proba_str = " | ".join(
            [f"{c}: {p:.1%}" for c, p in zip(classes, next_proba)]
        )

        print(f"\n--- Next Trading Day Forecast ---")
        print(f"Based on latest trading session: {latest_date}")
        print(f"Predicted Trend:                 {next_prediction}")
        print(f"Confidence Probabilities:        {proba_str}")

    except FileNotFoundError:
        print("Error: stock_data.csv not found. Run main.py first.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train stock price prediction model.")
    parser.add_argument(
        "--mode",
        choices=["binary", "3class"],
        default="binary",
        help="Prediction mode: 'binary' (UP vs DOWN) or '3class' (UP, DOWN, STABLE). Default is 'binary'.",
    )
    args = parser.parse_args()
    train_model(target_mode=args.mode)
