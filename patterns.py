import pandas as pd


def detect_patterns(data, window=20):
    """
    Detect consolidation zones and high-volatility periods
    using rolling price ranges and daily returns.
    """

    data = data.copy()

    # Daily percentage price change
    data["Daily_Return"] = data["Close"].pct_change() * 100

    # Price range over the last 20 trading days
    rolling_high = data["High"].rolling(window=window).max()
    rolling_low = data["Low"].rolling(window=window).min()

    data["Price_Range_Pct"] = (
        (rolling_high - rolling_low) / rolling_low
    ) * 100

    # Rolling standard deviation of daily returns
    data["Volatility"] = (
        data["Daily_Return"].rolling(window=window).std()
    )

    # Simple initial thresholds; we can refine these later
    data["Consolidation"] = data["Price_Range_Pct"] < 5
    data["High_Volatility"] = data["Volatility"] > 2

    return data


def analyze_patterns():
    try:
        data = pd.read_csv("stock_data.csv", index_col=0)

        if data.empty:
            print("Error: Stock data file is empty.")
            return

        data = detect_patterns(data)

        print("\n--- Consolidation and Volatility Analysis ---")
        print(
            data[
                [
                    "Close",
                    "Daily_Return",
                    "Price_Range_Pct",
                    "Volatility",
                    "Consolidation",
                    "High_Volatility",
                ]
            ].tail(10)
        )

        data.to_csv("stock_patterns.csv")
        print("\nPattern analysis saved to stock_patterns.csv")

    except FileNotFoundError:
        print("Error: stock_data.csv not found. Run main.py first.")


if __name__ == "__main__":
    analyze_patterns()