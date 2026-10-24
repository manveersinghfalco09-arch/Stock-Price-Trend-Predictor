import sys
import argparse
import warnings
import yfinance as yf

# Suppress library deprecation warnings for clean output
warnings.filterwarnings("ignore")


def download_stock_data(ticker="RELIANCE.NS"):
    """Download five years of historical stock data."""

    print(f"Downloading historical data for {ticker}...")

    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="5y", interval="1d")

        if data.empty:
            print(f"Error: No stock data was downloaded for '{ticker}'.")
            return None

        print(f"Successfully downloaded {len(data)} trading-day records.")
        print(data.tail())

        data.to_csv("stock_data.csv")
        print("Data saved to stock_data.csv")

        return data
    except Exception as e:
        print(f"Error downloading data for '{ticker}': {e}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download 5 years of stock data.")
    parser.add_argument(
        "--ticker",
        "-t",
        type=str,
        default="RELIANCE.NS",
        help="Stock ticker symbol (default: RELIANCE.NS)",
    )
    args = parser.parse_args()

    download_stock_data(ticker=args.ticker)
