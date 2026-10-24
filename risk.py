"""
Risk Factor & Return Potential Module
Provides financial risk indicators and potential gain/loss assessment for equities.
Designed for modular integration into the Stock Price Trend Predictor pipeline.
"""

import numpy as np
import pandas as pd


def calculate_risk_metrics(data, window=20):
    """
    Computes financial risk indicators:
    1. Annualized Volatility (%): Standard deviation of daily returns scaled to 252 trading days.
    2. Maximum Drawdown (MDD %): Largest peak-to-trough price decline over historical period.
    3. Daily Value at Risk (VaR 95% %): Expected maximum single-day loss at 95% confidence.
    4. Qualitative Risk Rating: Categorizes stock into 'Low', 'Moderate', 'High', or 'Very High'.
    5. Risk-Reward Ratio: Potential gain to 20-day resistance vs potential loss to 20-day support.
    """
    if data is None or len(data) < window:
        return {
            "annualized_volatility_pct": None,
            "max_drawdown_pct": None,
            "daily_var_95_pct": None,
            "risk_level": "Unknown",
            "potential_gain_pct": None,
            "potential_loss_pct": None,
            "risk_reward_ratio": None,
        }

    close = data["Close"]
    daily_returns = close.pct_change().dropna()

    # 1. Annualized Volatility (252 trading sessions/year)
    ann_vol = float(daily_returns.std() * np.sqrt(252) * 100)

    # 2. Maximum Drawdown (MDD)
    cumulative = (1 + daily_returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    max_drawdown = float(drawdown.min() * 100)

    # 3. Daily Value at Risk (95% Historical VaR)
    # 5th percentile of daily returns represents the 95% VaR threshold
    var_95 = float(abs(np.percentile(daily_returns, 5)) * 100)

    # 4. Risk Level Categorization
    if ann_vol < 18:
        risk_level = "Low Risk"
    elif ann_vol < 30:
        risk_level = "Moderate Risk"
    elif ann_vol < 45:
        risk_level = "High Risk"
    else:
        risk_level = "Very High / Speculative"

    # 5. Potential Gain vs Potential Loss based on 20-day channel
    latest_close = float(close.iloc[-1])
    recent_support = float(data["Low"].tail(window).min())
    recent_resistance = float(data["High"].tail(window).max())

    potential_gain_pct = max(0.0, (recent_resistance / latest_close - 1) * 100)
    potential_loss_pct = max(0.0, (1 - recent_support / latest_close) * 100)

    risk_reward_ratio = (
        round(potential_gain_pct / potential_loss_pct, 2)
        if potential_loss_pct > 0.05
        else 1.0
    )

    return {
        "annualized_volatility_pct": round(ann_vol, 2),
        "max_drawdown_pct": round(max_drawdown, 2),
        "daily_var_95_pct": round(var_95, 2),
        "risk_level": risk_level,
        "potential_gain_pct": round(potential_gain_pct, 2),
        "potential_loss_pct": round(potential_loss_pct, 2),
        "risk_reward_ratio": risk_reward_ratio,
    }


if __name__ == "__main__":
    try:
        sample_df = pd.read_csv("stock_data.csv", index_col=0)
        metrics = calculate_risk_metrics(sample_df)
        print("\n--- Risk Factor Analysis ---")
        for k, v in metrics.items():
            print(f"  {k:30}: {v}")
    except FileNotFoundError:
        print("stock_data.csv not found. Run main.py first to test risk.py standalone.")
