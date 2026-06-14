"""
==========================================================
MULTI-CLASS LABEL GENERATION
==========================================================

Goal
----
Create trading-oriented labels.

Research Question
-----------------
Given the current order book state,

Will BANKNIFTY move significantly UP,
significantly DOWN,
or remain SIDEWAYS over the next 10 seconds?

Labels
------

+1 = Future Move >= +4 points

 0 = -4 < Future Move < +4

-1 = Future Move <= -4 points

==========================================================
"""

import os
import pandas as pd

INPUT_FILE = "data/processed/features.csv"
OUTPUT_FILE = "data/processed/labeled_features.csv"

LOOKAHEAD = 20
POINT_THRESHOLD = 4


def create_labels(df):

    # ======================================================
    # FUTURE MID PRICE
    # ======================================================

    df["future_mid_price"] = (
        df["mid_price"]
        .shift(-LOOKAHEAD)
    )

    # ======================================================
    # FUTURE MOVE
    # ======================================================

    df["future_move"] = (
        df["future_mid_price"]
        -
        df["mid_price"]
    )

    # ======================================================
    # MULTI-CLASS LABELS
    # ======================================================

    df["label"] = 0

    df.loc[
        df["future_move"] >= POINT_THRESHOLD,
        "label"
    ] = 1

    df.loc[
        df["future_move"] <= -POINT_THRESHOLD,
        "label"
    ] = -1

    # ======================================================
    # REMOVE ROWS WITHOUT FUTURE DATA
    # ======================================================

    df = df.dropna(
        subset=["future_mid_price"]
    )

    return df


def main():

    os.makedirs(
        "data/processed",
        exist_ok=True
    )

    df = pd.read_csv(INPUT_FILE)

    df = create_labels(df)

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n" + "=" * 60)
    print("LABEL GENERATION COMPLETE")
    print("=" * 60)

    print(f"\nRows: {len(df)}")
    print(f"Lookahead: {LOOKAHEAD}")
    print(f"Point Threshold: {POINT_THRESHOLD}")

    print(f"\nSaved: {OUTPUT_FILE}")

    print("\n" + "=" * 60)
    print("LABEL DISTRIBUTION")
    print("=" * 60)

    print(
        df["label"]
        .value_counts()
        .sort_index()
    )

    print("\n" + "=" * 60)
    print("LABEL PERCENTAGES")
    print("=" * 60)

    print(
        (
            df["label"]
            .value_counts(normalize=True)
            .sort_index()
            * 100
        ).round(2)
    )

    print("\n" + "=" * 60)
    print("FUTURE MOVE STATISTICS")
    print("=" * 60)

    print(
        df["future_move"]
        .describe()
    )


if __name__ == "__main__":
    main()