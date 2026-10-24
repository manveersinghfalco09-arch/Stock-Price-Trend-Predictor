"""
FastAPI REST API Server for Stock Price Trend Predictor
Serves real-time stock trend predictions, risk indicators, and technical metrics to any frontend.
Pure Backend Service.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from predict import predict_stock
from risk import calculate_risk_metrics

app = FastAPI(
    title="Stock Price Trend Predictor API",
    description="REST API connecting the Machine Learning trend forecast and risk engine to the frontend.",
    version="1.0.0",
)

# Enable CORS so any frontend (React, Vite, Next.js, HTML/JS) can query without restriction
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "status": "online",
        "service": "Stock Price Trend Predictor API",
        "endpoints": {
            "prediction": "/api/predict?ticker=AAPL&mode=binary",
            "documentation": "/docs",
        },
    }


@app.get("/api/predict")
def get_prediction(
    ticker: str = Query(..., description="Stock ticker symbol (e.g. AAPL, MSFT, RELIANCE.NS, TCS.NS)"),
    mode: str = Query(
        "binary",
        regex="^(binary|3class)$",
        description="Forecast mode: 'binary' (UP vs DOWN) or '3class' (UP, DOWN, STABLE)",
    ),
):
    """
    Get next trading day trend prediction, confidence scores, technical indicators,
    risk factors, and historical chart records for a given stock ticker.
    """
    try:
        data = predict_stock(ticker=ticker.strip(), mode=mode)
        if data is None:
            raise HTTPException(
                status_code=404,
                detail=f"Unable to fetch data or generate prediction for ticker '{ticker}'. Please verify the symbol.",
            )
        return data

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    print("Starting Stock Trend Predictor API on http://localhost:8000 ...")
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
