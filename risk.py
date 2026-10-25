"""
Risk Factor & Return Potential Module
RiskPulse composite score authored by: Arash-Pathania
Provides financial risk indicators and potential gain/loss assessment for equities.
Designed for modular integration into the Stock Price Trend Predictor pipeline.

Composite Risk Score (0-100) contributed by: @Arash-Pathania (RiskPulse)
  Formula: risk_score = round(vol_score * 0.6 + drawdown_score * 0.4)
  Bands:
    Low Risk     -- score < 35   (Stable)
    Moderate Risk -- 35 <= score < 60  (Mediocre)
    High Risk    -- score >= 60  (Unstable)
"""

import numpy as np
import pandas as pd


def _compute_risk_score(ann_vol_pct: float, max_drawdown_pct: float) -> dict:
    """
    Compute RiskPulse composite 0-100 risk score from annualised volatility and
    max drawdown, using Arash's weighted formula:
        vol_score  = min(100, round(ann_vol_pct / 100 * 130 * 100))
        dd_score   = round(abs(max_drawdown_pct))
        risk_score = round(vol_score * 0.6 + dd_score * 0.4)

    Returns dict with: risk_score (int), risk_pulse_band (str), risk_pulse_label (str)
    """
    ann_vol_decimal = ann_vol_pct / 100.0          # convert % back to decimal
    vol_score = int(min(100, round(ann_vol_decimal * 130 * 100)))
    dd_score = int(round(abs(max_drawdown_pct)))
    score = round(vol_score * 0.6 + dd_score * 0.4)

    if score < 35:
        band, label = "Low Risk", "Stable"
    elif score < 60:
        band, label = "Moderate Risk", "Mediocre"
    else:
        band, label = "High Risk", "Unstable"

    return {
        "risk_score": score,
        "risk_pulse_band": band,
        "risk_pulse_label": label,
    }


def calculate_risk_metrics(data, window=20):
    """
    Computes financial risk indicators:
    1. Annualized Volatility (%): Standard deviation of daily returns scaled to 252 trading days.
    2. Maximum Drawdown (MDD %): Largest peak-to-trough price decline over historical period.
    3. Daily Value at Risk (VaR 95% %): Expected maximum single-day loss at 95% confidence.
    4. Qualitative Risk Rating: Categorizes stock into 'Low', 'Moderate', 'High', or 'Very High'.
    5. Risk-Reward Ratio: Potential gain to 20-day resistance vs potential loss to 20-day support.
    6. RiskPulse Composite Score (0-100): Weighted blend of volatility and drawdown (Arash formula).
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
            "risk_score": None,
            "risk_pulse_band": "Unknown",
            "risk_pulse_label": "Unknown",
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

    # 4. Risk Level Categorization (volatility-based qualitative band)
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

    # 6. RiskPulse composite score (Arash's formula)
    pulse = _compute_risk_score(ann_vol, max_drawdown)

    return {
        "annualized_volatility_pct": round(ann_vol, 2),
        "max_drawdown_pct": round(max_drawdown, 2),
        "daily_var_95_pct": round(var_95, 2),
        "risk_level": risk_level,
        "potential_gain_pct": round(potential_gain_pct, 2),
        "potential_loss_pct": round(potential_loss_pct, 2),
        "risk_reward_ratio": risk_reward_ratio,
        **pulse,
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
