"""Machine Learning Lab 7: Gaussian Naive Bayes on the UCI Raisin dataset.

Place data/Raisin_Dataset.xlsx in the project or allow online UCI fallback.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "Raisin_Dataset.xlsx"
OUTPUT_DIR = BASE_DIR / "outputs"


def load_dataset() -> pd.DataFrame:
    if DATA_PATH.exists():
        return pd.read_excel(DATA_PATH)
    try:
        from ucimlrepo import fetch_ucirepo
        raisin = fetch_ucirepo(id=850)
        features = raisin.data.features.copy()
        labels = raisin.data.targets.squeeze().rename("Class")
        return pd.concat([features, labels], axis=1)
    except Exception as exc:
        raise FileNotFoundError(
            "Dataset missing at data/Raisin_Dataset.xlsx. Download it from "
            "https://archive.ics.uci.edu/dataset/850/raisin and place it there, "
            "or run with internet access and ucimlrepo installed."
        ) from exc


def main() -> None:
    df = load_dataset()
    print("Dataset shape:", df.shape)
    print("\nFirst five rows:\n", df.head())
    print("\nNull counts:\n", df.isna().sum())
    print("Duplicate records:", int(df.duplicated().sum()))
    print("\nClass distribution:\n", df["Class"].value_counts())

    encoder = LabelEncoder()
    y = encoder.fit_transform(df["Class"])
    # DO NOT use the encoded target as an input feature: that leaks the answer.
    X = df.drop(columns=["Class"]).select_dtypes(include=np.number)
    if len(X.columns) != 7:
        raise ValueError(f"Expected 7 numeric features; found {list(X.columns)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print("\nFeatures used:", list(X.columns))
    print("Training samples:", len(X_train), "Test samples:", len(X_test))
    print("Class mapping:", dict(zip(encoder.classes_, range(len(encoder.classes_)))))

    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results = pd.DataFrame({
        "Actual": encoder.inverse_transform(y_test),
        "Predicted": encoder.inverse_transform(y_pred),
    })
    print("\nPrediction preview:\n", results.head(10))

    positive_label = int(encoder.transform(["Kecimen"])[0])
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=positive_label)
    recall = recall_score(y_test, y_pred, pos_label=positive_label)
    f1 = f1_score(y_test, y_pred, pos_label=positive_label)
    print(f"\nAccuracy:  {acc:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall:    {recall:.2%}")
    print(f"F1-score:  {f1:.2%}")
    print("\nClassification report:\n", classification_report(
        y_test, y_pred, target_names=encoder.classes_, zero_division=0
    ))
    matrix = confusion_matrix(y_test, y_pred)
    print("\nConfusion matrix:\n", matrix)

    OUTPUT_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 5))
    ConfusionMatrixDisplay(matrix, display_labels=encoder.classes_).plot(
        ax=ax, cmap="Blues", colorbar=False
    )
    ax.set_title("Raisin Variety Classification — Confusion Matrix")
    fig.tight_layout()
    figure_path = OUTPUT_DIR / "confusion_matrix.png"
    fig.savefig(figure_path, dpi=160, facecolor="white")
    plt.close(fig)
    print(f"\nSaved confusion matrix: {figure_path}")


if __name__ == "__main__":
    main()
