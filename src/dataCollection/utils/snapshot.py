import pandas as pd

CSV_FILE = "data/raw/banknifty_orderbook.csv"


def print_snapshot(row):
    print("\n" + "=" * 60)
    print("BANKNIFTY ORDER BOOK SNAPSHOT")
    print("=" * 60)

    print(f"\nTimestamp : {row['timestamp']}")

    print("\nBUY SIDE")
    print("-" * 60)
    print(f"{'Level':<10}{'Price':<15}{'Qty':<10}")

    for i in range(1, 6):
        print(
            f"B{i:<9}"
            f"{row[f'bid{i}_price']:<15}"
            f"{row[f'bid{i}_qty']:<10}"
        )

    print("\nSELL SIDE")
    print("-" * 60)
    print(f"{'Level':<10}{'Price':<15}{'Qty':<10}")

    for i in range(1, 6):
        print(
            f"S{i:<9}"
            f"{row[f'ask{i}_price']:<15}"
            f"{row[f'ask{i}_qty']:<10}"
        )

    print("\nMARKET DATA")
    print("-" * 60)

    print(f"LTP              : {row['ltp']}")
    print(f"VWAP             : {row['vwap']}")
    print(f"Volume           : {row['volume']}")
    print(f"Open Interest    : {row['open_interest']}")

    print("\nORDER FLOW")
    print("-" * 60)

    print(f"Total Buy Qty    : {row['total_buy_qty']}")
    print(f"Total Sell Qty   : {row['total_sell_qty']}")

    print(f"Buy Percentage   : {row['per_buy_qty']}%")
    print(f"Sell Percentage  : {row['per_sell_qty']}%")

    print("=" * 60)


def main():
    df = pd.read_csv(CSV_FILE)

    if len(df) == 0:
        print("CSV is empty.")
        return

    latest_row = df.iloc[-1]

    print_snapshot(latest_row)


if __name__ == "__main__":
    main()