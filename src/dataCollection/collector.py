#!/usr/bin/env python3
"""
==============================================================
GROWW BANKNIFTY FUTURES ORDER BOOK COLLECTOR
==============================================================

Collects top 5 bid/ask levels from Groww market depth API
and saves snapshots every second.

Output:
-------
data/raw/banknifty_orderbook.csv
"""

from __future__ import annotations

import csv
import os
import time
from datetime import datetime

import requests

# ============================================================
# CONFIG
# ============================================================

URL = (
    "https://groww.in/v1/api/stocks_fo_data/v1/tr_live_book/"
    "exchange/NSE/segment/FNO/BANKNIFTY26JUNFUT/latest"
)

CSV_FILE = "data/raw/banknifty_orderbook.csv"

HEADERS = {
    "Accept": "application/json",
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/149.0.0.0 Safari/537.36"
    ),

    # Replace these with your latest values
    "Authorization": "Bearer YOUR_TOKEN",
    "X-REQUEST-ID": "YOUR_REQUEST_ID",
    "X-REQUEST-CHECKSUM": "YOUR_CHECKSUM",
    "X-APP-ID": "growwWeb",
    "X-DEVICE-ID": "YOUR_DEVICE_ID",
    "X-DEVICE-ID-V2": "YOUR_DEVICE_ID",
    "x-platform": "web",
}


# ============================================================
# CSV SETUP
# ============================================================

def create_csv_if_needed():
    os.makedirs("data/raw", exist_ok=True)

    if os.path.exists(CSV_FILE):
        return

    headers = [
        "timestamp",

        "bid1_price", "bid1_qty",
        "bid2_price", "bid2_qty",
        "bid3_price", "bid3_qty",
        "bid4_price", "bid4_qty",
        "bid5_price", "bid5_qty",

        "ask1_price", "ask1_qty",
        "ask2_price", "ask2_qty",
        "ask3_price", "ask3_qty",
        "ask4_price", "ask4_qty",
        "ask5_price", "ask5_qty",

        "spread",
        "midprice",

        "total_buy_qty",
        "total_sell_qty",
    ]

    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)


# ============================================================
# FETCH SNAPSHOT
# ============================================================

def fetch_snapshot():
    response = requests.get(
        URL,
        headers=HEADERS,
        timeout=10,
    )

    response.raise_for_status()

    data = response.json()

    buy = data["buyBook"]
    sell = data["sellBook"]

    bid1 = buy["1"]["price"]
    ask1 = sell["1"]["price"]

    spread = ask1 - bid1
    midprice = (ask1 + bid1) / 2

    total_buy_qty = sum(level["qty"] for level in buy.values())
    total_sell_qty = sum(level["qty"] for level in sell.values())

    row = [
        datetime.utcnow().isoformat(timespec="milliseconds"),

        buy["1"]["price"],
        buy["1"]["qty"],

        buy["2"]["price"],
        buy["2"]["qty"],

        buy["3"]["price"],
        buy["3"]["qty"],

        buy["4"]["price"],
        buy["4"]["qty"],

        buy["5"]["price"],
        buy["5"]["qty"],

        sell["1"]["price"],
        sell["1"]["qty"],

        sell["2"]["price"],
        sell["2"]["qty"],

        sell["3"]["price"],
        sell["3"]["qty"],

        sell["4"]["price"],
        sell["4"]["qty"],

        sell["5"]["price"],
        sell["5"]["qty"],

        spread,
        midprice,

        total_buy_qty,
        total_sell_qty,
    ]

    return row


# ============================================================
# SAVE
# ============================================================

def save_row(row):
    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


# ============================================================
# MAIN
# ============================================================

def main():
    create_csv_if_needed()

    print("=" * 60)
    print("GROWW BANKNIFTY ORDER BOOK COLLECTOR STARTED")
    print(f"Saving to: {CSV_FILE}")
    print("=" * 60)

    while True:
        try:
            row = fetch_snapshot()

            save_row(row)

            print(
                f"{datetime.now().strftime('%H:%M:%S')} | "
                f"BID={row[1]} | "
                f"ASK={row[11]} | "
                f"SPREAD={row[21]:.2f} | "
                f"SAVED"
            )

        except Exception as e:
            print(f"ERROR: {e}")

        time.sleep(1)


if __name__ == "__main__":
    main()