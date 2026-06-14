"""
==============================================================
FEATURE ENGINEERING FOR ORDER BOOK IMBALANCE PREDICTOR
==============================================================

Purpose
-------
Convert raw order book snapshots into predictive market
microstructure features for machine learning.

Input:
------
data/raw/banknifty_orderbook.csv

Output:
-------
data/processed/features.csv

Research Hypothesis
-------------------
Short-term price movements are influenced by supply-demand
imbalances visible in the limit order book.

If buyers dominate the order book, future prices may be more
likely to rise.

If sellers dominate the order book, future prices may be more
likely to fall.

We attempt to quantify these effects using engineered features.
==============================================================
"""

import os
import numpy as np
import pandas as pd

INPUT_FILE = "data/raw/banknifty_orderbook.csv"
OUTPUT_FILE = "data/processed/features.csv"


def calculate_features(df: pd.DataFrame) -> pd.DataFrame:

    # ==========================================================
    # Convert all numerical columns
    # ==========================================================

    numeric_columns = [c for c in df.columns if c != "timestamp"]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ==========================================================
    # FEATURE 1: MID PRICE
    # ==========================================================
    #
    # Formula:
    #
    # Mid = (Best Bid + Best Ask) / 2
    #
    # Intuition:
    #
    # The highest buyer wants to buy at Bid.
    # The lowest seller wants to sell at Ask.
    #
    # The midpoint estimates the market's fair value.
    #
    # Why useful:
    #
    # Future labels will be generated using Mid Price
    # instead of Last Traded Price because it is less noisy.
    #
    # ==========================================================

    df["mid_price"] = (
        df["bid1_price"] +
        df["ask1_price"]
    ) / 2

    # ==========================================================
    # FEATURE 2: SPREAD
    # ==========================================================
    #
    # Formula:
    #
    # Spread = Ask1 - Bid1
    #
    # Intuition:
    #
    # Measures liquidity.
    #
    # Small spread:
    # Market is liquid.
    #
    # Large spread:
    # Market is uncertain.
    #
    # Why useful:
    #
    # Wider spreads often occur during volatility spikes
    # and may precede stronger price movements.
    #
    # ==========================================================

    df["spread"] = (
        (df["ask1_price"] -
        df["bid1_price"]).round(4)
    )

    # ==========================================================
    # FEATURE 3: LEVEL 1 OBI
    # ==========================================================
    #
    # Order Book Imbalance at best bid and best ask.
    #
    # Formula:
    #
    # OBI =
    # (BidQty1 - AskQty1)
    # -------------------
    # (BidQty1 + AskQty1)
    #
    # Range:
    #
    # +1  -> only buyers
    #  0  -> balanced
    # -1  -> only sellers
    #
    # Why useful:
    #
    # Measures immediate buying/selling pressure.
    #
    # ==========================================================

    denominator = (
        df["bid1_qty"] +
        df["ask1_qty"]
    )

    df["obi_l1"] = np.where(
        denominator == 0,
        0,
        (
            df["bid1_qty"] -
            df["ask1_qty"]
        ) / denominator
    )

    # ==========================================================
    # FEATURE 4: TOTAL BID DEPTH
    # ==========================================================
    #
    # Total visible buy-side liquidity.
    #
    # ==========================================================

    df["bid_depth"] = (
        df["bid1_qty"] +
        df["bid2_qty"] +
        df["bid3_qty"] +
        df["bid4_qty"] +
        df["bid5_qty"]
    )

    # ==========================================================
    # FEATURE 5: TOTAL ASK DEPTH
    # ==========================================================
    #
    # Total visible sell-side liquidity.
    #
    # ==========================================================

    df["ask_depth"] = (
        df["ask1_qty"] +
        df["ask2_qty"] +
        df["ask3_qty"] +
        df["ask4_qty"] +
        df["ask5_qty"]
    )

    # ==========================================================
    # FEATURE 6: DEPTH OBI
    # ==========================================================
    #
    # Formula:
    #
    # (Total Bid Depth - Total Ask Depth)
    # -----------------------------------
    # (Total Bid Depth + Total Ask Depth)
    #
    # Why useful:
    #
    # More stable than Level 1 OBI.
    #
    # Looking deeper into the order book reduces
    # sensitivity to temporary top-level noise.
    #
    # ==========================================================

    depth_denominator = (
        df["bid_depth"] +
        df["ask_depth"]
    )

    df["depth_obi"] = np.where(
        depth_denominator == 0,
        0,
        (
            df["bid_depth"] -
            df["ask_depth"]
        ) / depth_denominator
    )

    # ==========================================================
    # FEATURE 7: WEIGHTED BID DEPTH
    # ==========================================================
    #
    # Orders closer to market matter more.
    #
    # Level 1 -> weight 5
    # Level 2 -> weight 4
    # Level 3 -> weight 3
    # Level 4 -> weight 2
    # Level 5 -> weight 1
    #
    # ==========================================================

    df["weighted_bid_depth"] = (
        5 * df["bid1_qty"] +
        4 * df["bid2_qty"] +
        3 * df["bid3_qty"] +
        2 * df["bid4_qty"] +
        1 * df["bid5_qty"]
    )

    # ==========================================================
    # FEATURE 8: WEIGHTED ASK DEPTH
    # ==========================================================

    df["weighted_ask_depth"] = (
        5 * df["ask1_qty"] +
        4 * df["ask2_qty"] +
        3 * df["ask3_qty"] +
        2 * df["ask4_qty"] +
        1 * df["ask5_qty"]
    )

    # ==========================================================
    # FEATURE 9: WEIGHTED OBI
    # ==========================================================
    #
    # Similar to Depth OBI but gives more importance
    # to liquidity near the top of the book.
    #
    # Often more predictive than simple OBI.
    #
    # ==========================================================

    weighted_denominator = (
        df["weighted_bid_depth"] +
        df["weighted_ask_depth"]
    )

    df["weighted_obi"] = np.where(
        weighted_denominator == 0,
        0,
        (
            df["weighted_bid_depth"] -
            df["weighted_ask_depth"]
        ) / weighted_denominator
    )

    # ==========================================================
    # FEATURE 10: BUY/SELL RATIO
    # ==========================================================
    #
    # Formula:
    #
    # TotalBuyQty / TotalSellQty
    #
    # Interpretation:
    #
    # > 1  -> buying pressure dominates
    # < 1  -> selling pressure dominates
    #
    # ==========================================================

    df["buy_sell_ratio"] = np.where(
        df["total_sell_qty"] == 0,
        0,
        df["total_buy_qty"] /
        df["total_sell_qty"]
    )

    # ==========================================================
    # FEATURE 11: OPEN INTEREST CHANGE
    # ==========================================================
    #
    # Measures creation or closing of positions.
    #
    # Price ↑ + OI ↑
    # Often new longs.
    #
    # Price ↓ + OI ↑
    # Often new shorts.
    #
    # ==========================================================

    
    # df["oi_change"] = (
    #     df["open_interest"].diff()
    # )
    # ==========================================================
    # FEATURE 12: VOLUME CHANGE
    # ==========================================================

    # df["volume_change"] = (
    #     df["volume"].diff()
    # )


        # ==========================================================
    # FEATURE 13: OBI CHANGE
    # ==========================================================
    #
    # Measures change in top-level imbalance.
    #
    # Positive:
    # Buyers becoming stronger.
    #
    # Negative:
    # Sellers becoming stronger.
    #
    # ==========================================================

    df["obi_change"] = (
        df["obi_l1"].diff()
    )

    # ==========================================================
    # FEATURE 14: DEPTH OBI CHANGE
    # ==========================================================
    #
    # Measures change in deeper book imbalance.
    #
    # Often more stable than Level-1 OBI change.
    #
    # ==========================================================

    df["depth_obi_change"] = (
        df["depth_obi"].diff()
    )

    # ==========================================================
    # FEATURE 15: WEIGHTED OBI CHANGE
    # ==========================================================
    #
    # Measures acceleration of liquidity imbalance.
    #
    # Large positive:
    # buy-side liquidity increasing rapidly.
    #
    # ==========================================================

    df["weighted_obi_change"] = (
        df["weighted_obi"].diff()
    )

    # ==========================================================
    # FEATURE 16: SPREAD CHANGE
    # ==========================================================
    #
    # Spread widening:
    # liquidity deteriorating.
    #
    # Spread narrowing:
    # liquidity improving.
    #
    # ==========================================================

    df["spread_change"] = (
        df["spread"].diff()
    )

    # ==========================================================
    # FEATURE 17: MID PRICE RETURN
    # ==========================================================
    #
    # Recent short-term momentum.
    #
    # Positive:
    # price recently increased.
    #
    # Negative:
    # price recently decreased.
    #
    # ==========================================================

    df["mid_return"] = (
        df["mid_price"].pct_change()
    )

    # ==========================================================
    # FEATURE 18: BOOK PRESSURE
    # ==========================================================
    #
    # Uses total buy and sell quantities.
    #
    # Formula:
    #
    # (BuyQty - SellQty)
    # ------------------
    # (BuyQty + SellQty)
    #
    # Positive:
    # buy pressure dominates.
    #
    # Negative:
    # sell pressure dominates.
    #
    # ==========================================================

    book_denominator = (
        df["total_buy_qty"] +
        df["total_sell_qty"]
    )

    df["book_pressure"] = np.where(
        book_denominator == 0,
        0,
        (
            df["total_buy_qty"]
            -
            df["total_sell_qty"]
        )
        /
        book_denominator
    )

    # ==========================================================
    # FEATURE 19: BOOK PRESSURE CHANGE
    # ==========================================================
    #
    # Measures change in overall pressure.
    #
    # ==========================================================

    df["book_pressure_change"] = (
        df["book_pressure"].diff()
    )

    # ==========================================================
    # FEATURE 20: VWAP DISTANCE
    # ==========================================================
    #
    # Distance from VWAP.
    #
    # Positive:
    # trading above VWAP.
    #
    # Negative:
    # trading below VWAP.
    #
    # ==========================================================

    # df["vwap_distance"] = (
    #     df["mid_price"]
    #     -
    #     df["vwap"]
    # )
        # ==========================================================
    # FEATURE 21: MID PRICE MOMENTUM
    # ==========================================================

    df["mid_return_3s"] = (
        df["mid_price"]
        -
        df["mid_price"].shift(3)
    )

    df["mid_return_5s"] = (
        df["mid_price"]
        -
        df["mid_price"].shift(5)
    )

    df["mid_return_10s"] = (
        df["mid_price"]
        -
        df["mid_price"].shift(10)
    )

    # ==========================================================
    # FEATURE 22: OBI MOMENTUM
    # ==========================================================

    df["obi_change_3s"] = (
        df["obi_l1"]
        -
        df["obi_l1"].shift(3)
    )

    df["obi_change_5s"] = (
        df["obi_l1"]
        -
        df["obi_l1"].shift(5)
    )

    df["obi_change_10s"] = (
        df["obi_l1"]
        -
        df["obi_l1"].shift(10)
    )

    # ==========================================================
    # FEATURE 23: ROLLING OBI
    # ==========================================================

    df["obi_ma_5"] = (
        df["obi_l1"]
        .rolling(5)
        .mean()
    )

    df["obi_ma_10"] = (
        df["obi_l1"]
        .rolling(10)
        .mean()
    )

    # ==========================================================
    # FEATURE 24: ROLLING WEIGHTED OBI
    # ==========================================================

    df["weighted_obi_ma_5"] = (
        df["weighted_obi"]
        .rolling(5)
        .mean()
    )

    df["weighted_obi_ma_10"] = (
        df["weighted_obi"]
        .rolling(10)
        .mean()
    )

    # ==========================================================
    # FEATURE 25: DEPTH OBI ACCELERATION
    # ==========================================================

    df["depth_obi_acceleration"] = (
        df["depth_obi_change"]
        .diff()
    )

    # ==========================================================
    # FEATURE 26: WEIGHTED OBI ACCELERATION
    # ==========================================================

    df["weighted_obi_acceleration"] = (
        df["weighted_obi_change"]
        .diff()
    )

        # ==========================================================
    # FEATURE 27: SPREAD MOVING AVERAGES
    # ==========================================================
    #
    # Persistent liquidity conditions.
    #
    # High average spread:
    # Market less liquid.
    #
    # Low average spread:
    # Market more liquid.
    #
    # ==========================================================

    df["spread_ma_5"] = (
        df["spread"]
        .rolling(5)
        .mean()
    )

    df["spread_ma_10"] = (
        df["spread"]
        .rolling(10)
        .mean()
    )

    # ==========================================================
    # FEATURE 28: SPREAD VOLATILITY
    # ==========================================================
    #
    # Measures instability in liquidity.
    #
    # High spread volatility often occurs
    # before large moves.
    #
    # ==========================================================

    df["spread_std_5"] = (
        df["spread"]
        .rolling(5)
        .std()
    )

    df["spread_std_10"] = (
        df["spread"]
        .rolling(10)
        .std()
    )

    # ==========================================================
    # FEATURE 29: MID PRICE VOLATILITY
    # ==========================================================
    #
    # Measures short-term market volatility.
    #
    # ==========================================================

    df["mid_volatility_5"] = (
        df["mid_price"]
        .rolling(5)
        .std()
    )

    df["mid_volatility_10"] = (
        df["mid_price"]
        .rolling(10)
        .std()
    )

    # ==========================================================
    # FEATURE 30: WEIGHTED OBI TREND
    # ==========================================================
    #
    # Is imbalance strengthening
    # or weakening over time?
    #
    # Positive:
    # Buy pressure increasing.
    #
    # Negative:
    # Sell pressure increasing.
    #
    # ==========================================================

    df["weighted_obi_slope_5"] = (
        df["weighted_obi"]
        -
        df["weighted_obi"].shift(5)
    )

    df["weighted_obi_slope_10"] = (
        df["weighted_obi"]
        -
        df["weighted_obi"].shift(10)
    )
    # ==========================================================
    # CLEANUP
    # ==========================================================

    df = df.fillna(0)

    return df


def main():

    os.makedirs(
        "data/processed",
        exist_ok=True
    )

    df = pd.read_csv(INPUT_FILE)

    df = calculate_features(df)

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nFeature Engineering Completed")
    print(f"Rows: {len(df)}")
    print(f"Output: {OUTPUT_FILE}")

    print("\nCreated Features:")
    print("- mid_price")
    print("- spread")
    print("- obi_l1")
    print("- bid_depth")
    print("- ask_depth")
    print("- depth_obi")
    print("- weighted_bid_depth")
    print("- weighted_ask_depth")
    print("- weighted_obi")
    print("- buy_sell_ratio")
    # print("- oi_change")
    #print("- volume_change")
    print("- obi_change")
    print("- depth_obi_change")
    print("- weighted_obi_change")
    print("- spread_change")
    print("- mid_return")
    print("- book_pressure")
    print("- book_pressure_change")
  #  print("- vwap_distance")
    print("- mid_return_3s")
    print("- mid_return_5s")
    print("- mid_return_10s")
    print("- obi_change_3s")
    print("- obi_change_5s")
    print("- obi_change_10s")
    print("- obi_ma_5")
    print("- obi_ma_10")
    print("- weighted_obi_ma_5")
    print("- weighted_obi_ma_10")
    print("- depth_obi_acceleration")
    print("- weighted_obi_acceleration")
    print("- spread_ma_5")
    print("- spread_ma_10")
    print("- spread_std_5")
    print("- spread_std_10")
    print("- mid_volatility_5")
    print("- mid_volatility_10")
    print("- weighted_obi_slope_5")
    print("- weighted_obi_slope_10")


if __name__ == "__main__":
    main()