"""
=========================================================
MODEL EVALUATION
=========================================================

Purpose
-------
Evaluate predictive performance of the trained model.

Outputs
-------
reports/
├── confusion_matrix.png
├── roc_curve.png
├── feature_importance.csv

Research Questions
------------------
1. Is the model better than random?
2. Which features matter most?
3. Where does the model make mistakes?
4. How separable are the classes?

=========================================================
"""

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
    ConfusionMatrixDisplay,
)

INPUT_FILE = "data/processed/labeled_features.csv"
MODEL_FILE = "models/logistic_regression.pkl"

FEATURES = [
    "spread",
    "obi_l1",
    "depth_obi",
    "weighted_obi",
    "buy_sell_ratio",

    "oi_change",
    "volume_change",

    "obi_change",
    "depth_obi_change",
    "weighted_obi_change",

    "spread_change",

    "mid_return",

    "book_pressure",
    "book_pressure_change",

    "vwap_distance",
]

TARGET = "label"


def load_data():
    return pd.read_csv(INPUT_FILE)


def chronological_split(df):

    split_idx = int(len(df) * 0.8)

    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    return train_df, test_df


def evaluate_predictions(
    y_true,
    y_pred,
    y_prob
):

    print("\n" + "=" * 60)
    print("CLASSIFICATION METRICS")
    print("=" * 60)

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    try:
        roc_auc = roc_auc_score(
            y_true,
            y_prob
        )
    except Exception:
        roc_auc = None

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    if roc_auc is not None:
        print(f"ROC AUC  : {roc_auc:.4f}")

    print("\nClassification Report\n")
    print(
        classification_report(
            y_true,
            y_pred,
            zero_division=0
        )
    )


def save_confusion_matrix(
    y_true,
    y_pred
):

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm
    )

    plt.figure(figsize=(6, 6))
    disp.plot()

    plt.tight_layout()

    plt.savefig(
        "reports/confusion_matrix.png"
    )

    plt.close()

    print(
        "\nSaved: reports/confusion_matrix.png"
    )


def save_roc_curve(
    y_true,
    y_prob
):

    try:

        fpr, tpr, _ = roc_curve(
            y_true,
            y_prob
        )

        auc = roc_auc_score(
            y_true,
            y_prob
        )

        plt.figure(figsize=(8, 6))

        plt.plot(
            fpr,
            tpr,
            label=f"AUC = {auc:.4f}"
        )

        plt.plot(
            [0, 1],
            [0, 1],
            linestyle="--"
        )

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            "reports/roc_curve.png"
        )

        plt.close()

        print(
            "Saved: reports/roc_curve.png"
        )

    except Exception as e:

        print(
            f"ROC curve skipped: {e}"
        )


def save_feature_importance(
    model
):

    if not hasattr(
        model,
        "coef_"
    ):
        print(
            "Model has no coefficients."
        )
        return

    importance = pd.DataFrame({
        "feature": FEATURES,
        "coefficient": model.coef_[0]
    })

    importance["abs_value"] = (
        importance["coefficient"]
        .abs()
    )

    importance = importance.sort_values(
        "abs_value",
        ascending=False
    )

    importance.to_csv(
        "reports/feature_importance.csv",
        index=False
    )

    print(
        "Saved: reports/feature_importance.csv"
    )

    print("\nTop Features")

    print(
        importance[
            ["feature", "coefficient"]
        ].head(10)
    )


def main():

    os.makedirs(
        "reports",
        exist_ok=True
    )

    df = load_data()

    train_df, test_df = (
        chronological_split(df)
    )

    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]

    model = joblib.load(
        MODEL_FILE
    )

    y_pred = model.predict(
        X_test
    )

    y_prob = model.predict_proba(
        X_test
    )[:, 1]

    evaluate_predictions(
        y_test,
        y_pred,
        y_prob
    )

    save_confusion_matrix(
        y_test,
        y_pred
    )

    save_roc_curve(
        y_test,
        y_prob
    )

    save_feature_importance(
        model
    )


if __name__ == "__main__":
    main()