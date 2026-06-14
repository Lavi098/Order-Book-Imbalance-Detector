import pandas as pd

CSV_FILE = "data/raw/banknifty_orderbook.csv"


def main():

    df = pd.read_csv(CSV_FILE)

    print("=" * 60)
    print("ORDER BOOK DATA QUALITY REPORT")
    print("=" * 60)

    print(f"\nTotal Rows: {len(df)}")

    columns = [
        "bid1_price",
        "ask1_price",
        "bid1_qty",
        "ask1_qty",
        "spread",
        "midprice",
        "total_buy_qty",
        "total_sell_qty",
    ]

    print("\n" + "=" * 60)
    print("UNIQUE VALUES")
    print("=" * 60)

    for col in columns:

        unique_count = df[col].nunique()

        print(
            f"{col:<20} : "
            f"{unique_count}"
        )

    print("\n" + "=" * 60)
    print("MISSING VALUES")
    print("=" * 60)

    for col in columns:

        missing_count = df[col].isna().sum()

        print(
            f"{col:<20} : "
            f"{missing_count}"
        )

    print("\n" + "=" * 60)
    print("DUPLICATE ROWS")
    print("=" * 60)

    duplicate_rows = df.duplicated().sum()

    print(
        f"Duplicate Rows: "
        f"{duplicate_rows}"
    )

    print("\n" + "=" * 60)
    print("TIMESTAMP GAP ANALYSIS")
    print("=" * 60)

    timestamps = pd.to_datetime(
        df["timestamp"]
    )

    time_diff = (
        timestamps.diff()
        .dt.total_seconds()
    )

    print(time_diff.describe())

    print("\n" + "=" * 60)
    print("SPREAD STATISTICS")
    print("=" * 60)

    print(
        df["spread"]
        .describe()
    )

    print("\n" + "=" * 60)
    print("MID PRICE STATISTICS")
    print("=" * 60)

    print(
        df["midprice"]
        .describe()
    )

    print("\n" + "=" * 60)
    print("BUY QUANTITY STATISTICS")
    print("=" * 60)

    print(
        df["total_buy_qty"]
        .describe()
    )

    print("\n" + "=" * 60)
    print("SELL QUANTITY STATISTICS")
    print("=" * 60)

    print(
        df["total_sell_qty"]
        .describe()
    )

    print("\n" + "=" * 60)
    print("FIRST 5 ROWS")
    print("=" * 60)

    print(df.head())

    print("\n" + "=" * 60)
    print("LAST 5 ROWS")
    print("=" * 60)

    print(df.tail())


if __name__ == "__main__":
    main()