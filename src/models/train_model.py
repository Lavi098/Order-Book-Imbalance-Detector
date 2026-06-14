"""
=========================================================
MULTI-CLASS MODEL TRAINING
=========================================================

Purpose
-------
Train a classifier to predict:

-1 = Down Move
 0 = Neutral
 1 = Up Move

Input
-----
data/processed/labeled_features.csv

Output
------
models/logistic_regression.pkl

=========================================================
"""

import os
import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

INPUT_FILE = "data/processed/labeled_features.csv"
MODEL_FILE = "models/logistic_regression.pkl"

FEATURES = [
    "spread",

    "obi_l1",
    "depth_obi",
    "weighted_obi",

    "buy_sell_ratio",

    "obi_change",
    "depth_obi_change",
    "weighted_obi_change",

    "spread_change",

    "mid_return",

    "mid_return_3s",
    "mid_return_5s",
    "mid_return_10s",

    "obi_change_3s",
    "obi_change_5s",
    "obi_change_10s",

    "obi_ma_5",
    "obi_ma_10",

    "weighted_obi_ma_5",
    "weighted_obi_ma_10",

    "depth_obi_acceleration",
    "weighted_obi_acceleration",
]

TARGET = "label"


def load_data():

    df = pd.read_csv(INPUT_FILE)

    return df


def chronological_split(df):

    split_idx = int(len(df) * 0.8)

    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    return train_df, test_df


def train_model(X_train, y_train):

    model = LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
        
    )

    model.fit(
        X_train,
        y_train
    )

    return model


def evaluate_model(
    model,
    X_test,
    y_test
):

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print("\n" + "=" * 60)
    print("MODEL RESULTS")
    print("=" * 60)

    print(
        f"\nAccuracy: {accuracy:.4f}"
    )

    print("\nClassification Report")
    print(
        classification_report(
            y_test,
            predictions,
            digits=4
        )
    )

    print("\nConfusion Matrix")

    cm = confusion_matrix(
        y_test,
        predictions
    )

    print(cm)

    return accuracy


def save_feature_importance(model):

    coef_df = pd.DataFrame(
        model.coef_,
        columns=FEATURES
    )

    coef_df.index = [
        "DOWN(-1)",
        "NEUTRAL(0)",
        "UP(+1)"
    ]

    os.makedirs(
        "reports",
        exist_ok=True
    )

    coef_df.to_csv(
        "reports/feature_importance.csv",
        index=True
    )

    print(
        "\nSaved: reports/feature_importance.csv"
    )


def save_model(model):

    os.makedirs(
        "models",
        exist_ok=True
    )

    joblib.dump(
        model,
        MODEL_FILE
    )

    print(
        f"\nModel Saved: {MODEL_FILE}"
    )


def main():

    df = load_data()

    print(
        f"\nTotal Rows: {len(df)}"
    )

    print(
        "\nLabel Distribution:"
    )

    print(
        df[TARGET]
        .value_counts()
        .sort_index()
    )

    train_df, test_df = (
        chronological_split(df)
    )

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]

    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]

    print(
        f"\nTrain Rows: {len(train_df)}"
    )

    print(
        f"Test Rows : {len(test_df)}"
    )

    print(
        "\nClasses:"
    )

    print(
        sorted(
            y_train.unique()
        )
    )

    print(
        "\nTraining Model..."
    )

    model = train_model(
        X_train,
        y_train
    )

    evaluate_model(
        model,
        X_test,
        y_test
    )

    save_feature_importance(
        model
    )

    save_model(
        model
    )


if __name__ == "__main__":
    main()