import argparse
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score

from analysis import find_support_resistance, detect_sr_zones
from model import prepare_features
from patterns import detect_patterns
from risk import calculate_risk_metrics


def fetch_stock_data(ticker="RELIANCE.NS", period="5y"):
    """Download historical stock data from Yahoo Finance."""
    print(f"\n[1/4] Downloading {period} historical data for '{ticker}'...")
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period, interval="1d")

        if data.empty:
            print(f"Error: No data found for ticker '{ticker}'. Please verify the symbol.")
            return None

        data.to_csv("stock_data.csv")
        print(
            f"      Downloaded {len(data)} trading records "
            f"(from {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')})."
        )
        return data

    except Exception as e:
        print(f"Error downloading data for '{ticker}': {e}")
        return None


def predict_stock(ticker="RELIANCE.NS", mode="binary"):
    """
    Core backend prediction function.
    Returns a clean, structured dictionary ready for Web API, Frontend, or CLI.
    """
    raw_data = fetch_stock_data(ticker=ticker, period="5y")
    if raw_data is None:
        return None

    # 1. Support and Resistance analysis
    analysis_df = find_support_resistance(raw_data)
    analysis_df.to_csv("stock_analysis.csv")

    # 1b. Swing-pivot zone detection (for API response & dashboard)
    sr_zones = detect_sr_zones(raw_data, top_n=5)

    # 2. Pattern detection
    patterns_df = detect_patterns(raw_data)
    patterns_df.to_csv("stock_patterns.csv")

    # 3. Risk Factor metrics
    risk_data = calculate_risk_metrics(raw_data)

    # 4. Feature engineering & model training
    feature_df = prepare_features(raw_data, target_mode=mode)

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
    feature_columns = [c for c in feature_df.columns if c not in excluded_columns]
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan)

    # Retain the newest session to forecast tomorrow
    latest_unlabeled = feature_df[
        feature_df["Target"].isna() & feature_df[feature_columns].notna().all(axis=1)
    ]

    clean_train_data = feature_df.dropna(subset=["Target"] + feature_columns)

    if len(clean_train_data) < 100:
        return None

    # Chronological train/test split (80% train, 20% test)
    split_index = int(len(clean_train_data) * 0.8)
    train = clean_train_data.iloc[:split_index]
    test = clean_train_data.iloc[split_index:]

    X_train = train[feature_columns]
    y_train = train["Target"]
    X_test = test[feature_columns]
    y_test = test["Target"]

    # Imputer + Ensemble (Random Forest + HistGradientBoosting)
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

    # Test evaluation
    test_predictions = model.predict(X_test_imp)
    test_acc = accuracy_score(y_test, test_predictions)

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train_imp, y_train)
    dummy_acc = accuracy_score(y_test, dummy.predict(X_test_imp))

    # High-confidence accuracy (probability >= 54%)
    probabilities = model.predict_proba(X_test_imp)
    max_p = np.max(probabilities, axis=1)
    confident_mask = max_p >= 0.54
    if confident_mask.sum() >= 15:
        confident_acc = accuracy_score(y_test[confident_mask], test_predictions[confident_mask])
        conf_str = f"{confident_acc:.2%} ({confident_mask.sum()}/{len(y_test)} sessions)"
    else:
        conf_str = f"{test_acc:.2%}"

    # Save test predictions
    results = test[["Close", "Target"]].copy()
    results["Predicted_Trend"] = test_predictions
    classes = model.classes_
    for i, c in enumerate(classes):
        results[f"Prob_{c}"] = probabilities[:, i]
    results.to_csv("model_predictions.csv")

    # Next-Day Prediction
    if not latest_unlabeled.empty:
        latest_row = latest_unlabeled.iloc[[-1]]
    else:
        latest_row = clean_train_data.iloc[[-1]]

    latest_date = (
        latest_row.index[-1].strftime("%Y-%m-%d")
        if hasattr(latest_row.index[-1], "strftime")
        else str(latest_row.index[-1])
    )
    latest_close = float(latest_row["Close"].iloc[0])

    # Support & Resistance values for latest session
    latest_support = float(analysis_df["Support"].iloc[-1])
    latest_resistance = float(analysis_df["Resistance"].iloc[-1])
    support_dist = float((latest_close / latest_support - 1) * 100)
    resist_dist = float((latest_resistance / latest_close - 1) * 100)

    # Indicators
    rsi_val = float(latest_row["RSI_14"].iloc[0]) if "RSI_14" in latest_row else None
    if rsi_val is not None and not np.isnan(rsi_val):
        rsi_status = "Oversold" if rsi_val < 30 else ("Overbought" if rsi_val > 70 else "Neutral")
    else:
        rsi_val = None
        rsi_status = "Neutral"

    atr_pct = float(latest_row["ATR_14_Pct"].iloc[0] * 100) if "ATR_14_Pct" in latest_row else None
    is_consolidation = bool(patterns_df["Consolidation"].iloc[-1])
    is_high_vol = bool(patterns_df["High_Volatility"].iloc[-1])
    vol_regime = "High Volatility" if is_high_vol else ("Consolidation" if is_consolidation else "Normal")

    # Model inference
    features_to_predict = imputer.transform(latest_row[feature_columns])
    predicted_trend = str(model.predict(features_to_predict)[0])
    next_probabilities = model.predict_proba(features_to_predict)[0]
    proba_map = {str(cls): round(float(prob), 4) for cls, prob in zip(classes, next_probabilities)}

    # Determine display currency symbol (INR for .NS/.BO, else USD)
    currency = "₹" if ticker.upper().endswith((".NS", ".BO")) else "$"

    # Recent history for frontend charts (last 60 trading days)
    chart_df = raw_data.tail(60).copy()
    history_records = []
    for idx, row in chart_df.iterrows():
        history_records.append({
            "date": idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx),
            "open": round(float(row["Open"]), 2),
            "high": round(float(row["High"]), 2),
            "low": round(float(row["Low"]), 2),
            "close": round(float(row["Close"]), 2),
            "volume": int(row["Volume"]) if "Volume" in row else 0,
        })

    return {
        "ticker": ticker.upper(),
        "currency": currency,
        "latest_date": latest_date,
        "latest_close": round(latest_close, 2),
        "prediction": {
            "trend": predicted_trend,
            "confidence": proba_map.get(predicted_trend, 0.0),
            "probabilities": proba_map,
            "mode": mode,
        },
        "technical_analysis": {
            "support": round(latest_support, 2),
            "resistance": round(latest_resistance, 2),
            "support_distance_pct": round(support_dist, 2),
            "resistance_distance_pct": round(resist_dist, 2),
            "rsi": round(rsi_val, 1) if rsi_val is not None else None,
            "rsi_status": rsi_status,
            "atr_pct": round(atr_pct, 2) if atr_pct is not None else None,
            "volatility_regime": vol_regime,
        },
        "risk_factors": risk_data,
        "model_metrics": {
            "test_accuracy": round(float(test_acc), 4),
            "baseline_accuracy": round(float(dummy_acc), 4),
            "high_conviction": conf_str,
        },
        "chart_history": history_records,
        "sr_zones": sr_zones,
    }


def run_pipeline_for_ticker(ticker="RELIANCE.NS", mode="binary"):
    """Run pipeline and output human-readable terminal dashboard."""
    result = predict_stock(ticker=ticker, mode=mode)
    if not result:
        print(f"Error: Unable to generate prediction for '{ticker}'.")
        return

    cur = result["currency"]
    tech = result["technical_analysis"]
    pred = result["prediction"]
    met = result["model_metrics"]
    risk = result["risk_factors"]

    print("\n" + "=" * 68)
    print(f"             MARKET SNAPSHOT & TECHNICAL ANALYSIS: {result['ticker']}")
    print("=" * 68)
    print(f"  Latest Session Date   : {result['latest_date']}")
    print(f"  Latest Closing Price  : {cur}{result['latest_close']:,.2f}")
    print(f"  20-Day Support Level  : {cur}{tech['support']:,.2f}  ({tech['support_distance_pct']:+.2f}% from close)")
    print(f"  20-Day Resistance     : {cur}{tech['resistance']:,.2f}  ({tech['resistance_distance_pct']:+.2f}% from close)")
    print(f"  14-Day RSI Indicator  : {tech['rsi']} ({tech['rsi_status']})")
    print(f"  Volatility Regime     : {tech['volatility_regime']} (ATR: {tech['atr_pct']:.2f}% of price)")

    print("-" * 68)
    print(f"                   RISK & GAIN PROFILE (MEMBER 2 MODULE)")
    print("-" * 68)
    print(f"  Risk Level Rating     : {risk['risk_level']}")
    print(f"  Annualized Volatility : {risk['annualized_volatility_pct']}%")
    print(f"  Max Historical Drop   : {risk['max_drawdown_pct']}%")
    print(f"  Daily Value at Risk   : -{risk['daily_var_95_pct']}% (95% confidence)")
    print(f"  Risk / Reward Ratio   : {risk['risk_reward_ratio']}x (Gain to Res: +{risk['potential_gain_pct']}%, Loss to Sup: -{risk['potential_loss_pct']}%)")

    print("-" * 68)
    print(f"                   PREDICTION FOR NEXT TRADING DAY ({mode.upper()} MODE)")
    print("-" * 68)
    print(f"  >>> PREDICTED TREND   : {pred['trend']}")
    print(f"  >>> CONFIDENCE SCORE  : {pred['confidence']:.1%}")
    print(
        f"  >>> PROBABILITIES     : "
        + " | ".join([f"{c}: {p:.1%}" for c, p in sorted(pred['probabilities'].items())])
    )
    print(f"  >>> MODEL RELIABILITY : Test Accuracy {met['test_accuracy']:.2%} (Baseline: {met['baseline_accuracy']:.2%})")
    print(f"  >>> HIGH-CONVICTION   : {met['high_conviction']} on high-confidence setups (>=54% prob)")
    print("=" * 68)
    print("  * Note: Educational demonstration. Not financial or trading advice.\n")


def interactive_mode():
    """Prompt the user for stock tickers interactively in a loop."""
    print("=" * 68)
    print("           WELCOME TO STOCK PRICE TREND PREDICTOR")
    print("=" * 68)

    while True:
        try:
            user_input = input(
                "\nEnter stock ticker symbol (e.g., AAPL, MSFT, RELIANCE.NS, TCS.NS)\n"
                "or press Enter for default [RELIANCE.NS] (type 'q' to quit): "
            ).strip()

            if user_input.lower() in ("q", "quit", "exit"):
                print("Exiting predictor. Goodbye!")
                break

            ticker = user_input if user_input else "RELIANCE.NS"
            run_pipeline_for_ticker(ticker, mode="binary")

        except (KeyboardInterrupt, EOFError):
            print("\nExiting predictor. Goodbye!")
            break


def main():
    parser = argparse.ArgumentParser(
        description="Predict the next trading day's stock price trend using ML."
    )
    parser.add_argument(
        "--ticker",
        "-t",
        type=str,
        default=None,
        help="Stock ticker symbol (e.g. AAPL, MSFT, RELIANCE.NS, TCS.NS).",
    )
    parser.add_argument(
        "--mode",
        "-m",
        choices=["binary", "3class"],
        default="binary",
        help="Prediction mode: 'binary' (UP vs DOWN, recommended) or '3class' (UP, DOWN, STABLE). Default is 'binary'.",
    )
    args = parser.parse_args()

    if args.ticker:
        run_pipeline_for_ticker(args.ticker, mode=args.mode)
    else:
        interactive_mode()


if __name__ == "__main__":
    main()
