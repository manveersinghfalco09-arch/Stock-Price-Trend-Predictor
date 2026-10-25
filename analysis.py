"""
Support & Resistance Analysis Module
-------------------------------------
Two complementary methods:

1. find_support_resistance(data, window=20)
   - Rolling 20-day High/Low channel used as continuous Support &
     Resistance columns fed into the ML feature pipeline.

2. detect_sr_zones(data, prominence_pct=0.02, min_distance=5,
                   tolerance_pct=0.015, top_n=5)
   - Swing-high / swing-low detection via scipy prominence filtering.
   - Clusters nearby pivots into meaningful price zones.
   - Counts historical support & resistance touches per zone.
   - Returns the N zones nearest to the latest close as a list of
     dicts -- ready for the API JSON response and the dashboard UI.
   - Contributed by: @vijaysatwik (vijaysatwik/support_resistance_analysis)
"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


# ---------------------------------------------------------------------------
# Method 1 -- Rolling Channel (used by ML pipeline via predict.py)
# ---------------------------------------------------------------------------

def find_support_resistance(data, window=20):
    """
    Identify potential support and resistance levels
    using rolling 20-day price lows and highs.
    Returns a copy of `data` with added 'Support' and 'Resistance' columns.
    """
    data = data.copy()
    data["Support"] = data["Low"].rolling(window=window).min()
    data["Resistance"] = data["High"].rolling(window=window).max()
    return data


# ---------------------------------------------------------------------------
# Method 2 -- Swing-pivot Zone Detection (used by API response & dashboard)
# ---------------------------------------------------------------------------

def detect_sr_zones(data, prominence_pct=0.02, min_distance=5,
                    tolerance_pct=0.015, top_n=5):
    """
    Detect significant support & resistance price zones from OHLCV data.

    Algorithm (ported from @vijaysatwik's support_resistance_analysis repo):
      1. Find swing highs (resistance pivots) & swing lows (support pivots)
         using scipy.signal.find_peaks with prominence filtering.
      2. Cluster nearby pivots within `tolerance_pct` of each other into zones.
         Zones touched only once are discarded (noise).
      3. Return the `top_n` zones closest to the latest closing price, each
         carrying full metadata (level, type, touches, support/resistance counts).

    Parameters
    ----------
    data            : pd.DataFrame   -- OHLCV data with High, Low, Close columns.
    prominence_pct  : float          -- Minimum peak prominence as % of median close.
    min_distance    : int            -- Minimum bars between two detected peaks.
    tolerance_pct   : float          -- Price tolerance to cluster nearby pivots (1.5%).
    top_n           : int            -- Number of nearest zones to return.

    Returns
    -------
    list[dict]  -- Sorted by distance from latest close (nearest first).
                   Each dict: { level, type, touches, support_touches,
                                 resistance_touches, distance_pct }
    """
    if data is None or len(data) < 15:
        return []

    highs = data["High"].to_numpy()
    lows = data["Low"].to_numpy()
    close = data["Close"].to_numpy()

    prominence = float(np.median(close)) * prominence_pct

    peak_idx, _ = find_peaks(highs, prominence=prominence, distance=min_distance)
    low_idx, _ = find_peaks(-lows, prominence=prominence, distance=min_distance)

    points = (
        [(float(highs[i]), "Resistance") for i in peak_idx]
        + [(float(lows[i]), "Support") for i in low_idx]
    )
    points.sort(key=lambda p: p[0])

    # Cluster nearby pivots
    clusters = []
    for price, kind in points:
        if clusters:
            cluster = clusters[-1]
            center = cluster["price_sum"] / cluster["touches"]
            if abs(price - center) / center <= tolerance_pct:
                cluster["price_sum"] += price
                cluster["touches"] += 1
                cluster[kind] += 1
                continue
        clusters.append({
            "price_sum": price,
            "touches": 1,
            "Support": int(kind == "Support"),
            "Resistance": int(kind == "Resistance"),
        })

    # Build zone records (require at least 2 touches to filter noise)
    zones = []
    for c in clusters:
        if c["touches"] < 2:
            continue
        sc = c["Support"]
        rc = c["Resistance"]
        zone_type = "Both" if sc and rc else ("Support" if sc else "Resistance")
        zones.append({
            "level": round(c["price_sum"] / c["touches"], 2),
            "type": zone_type,
            "touches": c["touches"],
            "support_touches": sc,
            "resistance_touches": rc,
        })

    if not zones:
        return []

    # Return top_n zones nearest to the latest close
    latest_close = float(close[-1])
    for z in zones:
        z["distance_pct"] = round(abs(z["level"] - latest_close) / latest_close * 100, 2)

    zones.sort(key=lambda z: z["distance_pct"])
    return zones[:top_n]


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

def analyze_stock():
    try:
        data = pd.read_csv("stock_data.csv", index_col=0)

        if data.empty:
            print("Error: Stock data file is empty.")
            return

        data = find_support_resistance(data)

        print("\n--- Rolling Support & Resistance (20-day channel) ---")
        print(data[["Close", "Support", "Resistance"]].tail(10))

        data.to_csv("stock_analysis.csv")
        print("\nAnalysis saved to stock_analysis.csv")

        print("\n--- Swing-Pivot Zone Detection ---")
        zones = detect_sr_zones(data)
        if zones:
            for z in zones:
                print(f"  {z['type']:12s}  Level: {z['level']:>10.2f}  "
                      f"Touches: {z['touches']}  "
                      f"(S:{z['support_touches']} / R:{z['resistance_touches']})  "
                      f"Dist: {z['distance_pct']}%")
        else:
            print("  No significant zones detected.")

    except FileNotFoundError:
        print("Error: stock_data.csv not found. Run main.py first.")


if __name__ == "__main__":
    analyze_stock()
