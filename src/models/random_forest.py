"""
=========================================================
RANDOM FOREST TRAINING
=========================================================

Purpose
-------
Train a Random Forest model to predict:

-1 = Down Move
 0 = Neutral
 1 = Up Move

This model can learn nonlinear relationships
that Logistic Regression cannot.

=========================================================
"""

import os
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

INPUT_FILE = "data/processed/labeled_features.csv"

MODEL_FILE = (
    "models/random_forest.pkl"
)

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
        "spread_ma_5",
    "spread_ma_10",

    "spread_std_5",
    "spread_std_10",

    "mid_volatility_5",
    "mid_volatility_10",

    "weighted_obi_slope_5",
    "weighted_obi_slope_10",
]

TARGET = "label"


def load_data():

    df = pd.read_csv(INPUT_FILE)

    return df


def chronological_split(df):

    split_idx = int(
        len(df) * 0.80
    )

    train_df = df.iloc[:split_idx]

    test_df = df.iloc[split_idx:]

    return train_df, test_df


def train_model(
    X_train,
    y_train
):

    model = RandomForestClassifier(

        n_estimators=500,

        max_depth=10,

        min_samples_leaf=10,

        random_state=42,

        n_jobs=-1
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
    print("RANDOM FOREST RESULTS")
    print("=" * 60)

    print(
        f"\nAccuracy: {accuracy:.4f}"
    )

    print(
        "\nClassification Report"
    )

    print(
        classification_report(
            y_test,
            predictions,
            digits=4,
            zero_division=0
        )
    )

    print(
        "\nConfusion Matrix"
    )

    cm = confusion_matrix(
        y_test,
        predictions
    )

    print(cm)

    return accuracy


def save_feature_importance(
    model
):

    importance_df = pd.DataFrame({

        "feature": FEATURES,

        "importance":
        model.feature_importances_

    })

    importance_df = (
        importance_df
        .sort_values(
            "importance",
            ascending=False
        )
    )

    os.makedirs(
        "reports",
        exist_ok=True
    )

    importance_df.to_csv(

        "reports/random_forest_feature_importance.csv",

        index=False
    )

    print(
        "\nSaved: reports/random_forest_feature_importance.csv"
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

    X_train = train_df[
        FEATURES
    ]

    y_train = train_df[
        TARGET
    ]

    X_test = test_df[
        FEATURES
    ]

    y_test = test_df[
        TARGET
    ]

    print(
        f"\nTrain Rows: {len(train_df)}"
    )

    print(
        f"Test Rows : {len(test_df)}"
    )

    print(
        "\nTraining Random Forest..."
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