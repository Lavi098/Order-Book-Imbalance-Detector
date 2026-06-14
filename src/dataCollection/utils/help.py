import pandas as pd

df = pd.read_csv("data/processed/features.csv")

moves = (
    df["mid_price"]
    .diff()
    .abs()
)

print(moves.describe())

print("\n95th percentile:")
print(moves.quantile(0.95))