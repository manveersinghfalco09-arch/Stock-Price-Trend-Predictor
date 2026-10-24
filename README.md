# Stock Price Trend Predictor

A modular Python machine-learning application and REST API that analyzes historical market data, computes comprehensive technical and risk indicators, and predicts the next trading day's price trend across global assets (NSE, BSE, US markets).

---

## Architecture & Team Modules

| Module | File | Owner / Role | Purpose |
|---|---|---|---|
| **API Server** | `api.py` | Backend Lead | FastAPI REST server with CORS for connecting to any Frontend |
| **Prediction Engine** | `predict.py` | Backend Lead | End-to-end pipeline & prediction function returning structured JSON |
| **Risk & Gain Analysis** | `risk.py` | Analytics Member | Computes Volatility, Max Drawdown, 95% VaR, and Risk-Reward ratio |
| **Technical Analysis** | `analysis.py` | Backend | Rolling 20-day Support & Resistance channel levels |
| **Volatility Patterns** | `patterns.py` | Backend | Consolidation regimes and volatility flags |
| **ML Ensemble Model** | `model.py` | Backend | Technical feature engineering and VotingClassifier ensemble |
| **Data Ingestion** | `main.py` | Backend | 5-year OHLCV data downloader via Yahoo Finance |

---

## Installation

1. Ensure Python 3.10+ is installed.
2. Clone the repository:
   ```bash
   git clone https://github.com/manveersinghfalco09-arch/Stock-Price-Trend-Predictor.git
   cd Stock-Price-Trend-Predictor
   ```
3. Install dependencies:
   ```bash
   python -m pip install -r requirements.txt
   ```

---

## How to Run

### 1. Launching the REST API (For Frontend Integration)

Run the backend server so the frontend can query predictions:

```bash
uvicorn api:app --reload --port 8000
```

* **Interactive API Documentation (Swagger)**: Open [http://localhost:8000/docs](http://localhost:8000/docs) in your browser.
* **Test Endpoint**:
  ```http
  GET http://localhost:8000/api/predict?ticker=AAPL&mode=binary
  ```

#### Sample JSON Response for Frontend:
```json
{
  "ticker": "AAPL",
  "currency": "$",
  "latest_date": "2026-10-23",
  "latest_close": 182.50,
  "prediction": {
    "trend": "UP",
    "confidence": 0.5842,
    "probabilities": { "UP": 0.5842, "DOWN": 0.4158 },
    "mode": "binary"
  },
  "technical_analysis": {
    "support": 178.10,
    "resistance": 189.40,
    "support_distance_pct": 2.47,
    "resistance_distance_pct": 3.78,
    "rsi": 54.6,
    "rsi_status": "Neutral",
    "atr_pct": 1.45,
    "volatility_regime": "Normal"
  },
  "risk_factors": {
    "annualized_volatility_pct": 22.45,
    "max_drawdown_pct": -28.32,
    "daily_var_95_pct": 2.15,
    "risk_level": "Moderate Risk",
    "potential_gain_pct": 3.78,
    "potential_loss_pct": 2.41,
    "risk_reward_ratio": 1.57
  },
  "model_metrics": {
    "test_accuracy": 0.5520,
    "baseline_accuracy": 0.5080,
    "high_conviction": "61.50% (40/250 sessions)"
  },
  "chart_history": [
    { "date": "2026-10-23", "open": 181.2, "high": 183.1, "low": 180.9, "close": 182.5, "volume": 45200000 }
  ]
}
```

---

### 2. Terminal Interactive Mode (CLI)

```bash
python predict.py
```

Or pass arguments directly:

```bash
# US Equities
python predict.py --ticker AAPL --mode binary
python predict.py --ticker MSFT

# Indian Equities
python predict.py --ticker RELIANCE.NS
python predict.py --ticker TCS.NS
```

---

## Team Git & Collaboration Workflow

### Branching Strategy
To avoid conflicts between the frontend, backend, and risk analytics:

1. **`main`**: Protected branch. Always contains stable, working code.
2. **`feature/frontend`**: Branch for frontend code.
3. **`feature/risk-analysis`**: Branch for risk indicators and feature updates.
4. **`feature/backend-api`**: Branch for backend and model updates.

### Merging Process
1. Create a branch: `git checkout -b feature/<name>`
2. Commit and push: `git push origin feature/<name>`
3. Open a **Pull Request (PR)** on GitHub into `main` for review.
