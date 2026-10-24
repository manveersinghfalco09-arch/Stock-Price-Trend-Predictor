import pandas as pd


def find_support_resistance(data, window=20):
    """
    Identify potential support and resistance levels
    using rolling 20-day price lows and highs.
    """

    data = data.copy()

    data["Support"] = (
        data["Low"]
        .rolling(window=window)
        .min()
    )

    data["Resistance"] = (
        data["High"]
        .rolling(window=window)
        .max()
    )

    return data


def analyze_stock():
    try:
        data = pd.read_csv("stock_data.csv", index_col=0)

        if data.empty:
            print("Error: Stock data file is empty.")
            return

        data = find_support_resistance(data)

        print("\n--- Support and Resistance Analysis ---")
        print(data[["Close", "Support", "Resistance"]].tail(10))

        data.to_csv("stock_analysis.csv")
        print("\nAnalysis saved to stock_analysis.csv")

    except FileNotFoundError:
        print("Error: stock_data.csv not found. Run main.py first.")


if __name__ == "__main__":
    analyze_stock()