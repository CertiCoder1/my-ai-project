#train.py
import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


def parse_args():
    p = argparse.ArgumentParser(
        description="Train Decision Tree and/or k-NN on Iris and save artifacts to outputs/."
    )
    p.add_argument("--model", choices=["dt", "knn", "both"], default="both",
                   help="Which model to train (default: both).")
    p.add_argument("--test-size", type=float, default=0.2,
                   help="Test split size (default: 0.2).")
    p.add_argument("--random-state", type=int, default=42,
                   help="Random state (default: 42).")
    p.add_argument("--max-depth", type=int, default=None,
                   help="DecisionTree max_depth (default: None).")
    p.add_argument("--n-neighbors", type=int, default=5,
                   help="k-NN n_neighbors (default: 5).")
    p.add_argument("--outputs-dir", type=str, default="outputs",
                   help="Directory to save models/plots (default: outputs).")
    return p.parse_args()


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def save_confusion_matrix(y_true, y_pred, labels, out_path: Path, title: str):
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=labels, cmap=None, ax=ax
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    args = parse_args()
    out_dir = Path(args.outputs_dir)
    ensure_dir(out_dir)

    # Load data
    iris = load_iris(as_frame=True)
    X: pd.DataFrame = iris.data
    y: pd.Series = iris.target
    labels = list(iris.target_names)

    # Optional: quick visibility in console
    print("Feature names:", list(iris.feature_names))
    print("Target names:", labels)
    print(X.head(), "\n", y.head(), "\n")

    # Add species names (kept for parity with your original script)
    species = np.array(iris.target_names)
    df = X.copy()
    df["species"] = species[y]
    print(df.head(10), "\n")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    # ---- Decision Tree ----
    if args.model in ("dt", "both"):
        dt = DecisionTreeClassifier(
            random_state=args.random_state,
            max_depth=args.max_depth
        )
        dt.fit(X_train, y_train)
        y_pred_dt = dt.predict(X_test)
        acc_dt = accuracy_score(y_test, y_pred_dt)
        print(f"[DT] accuracy: {acc_dt:.4f}")
        print("[DT] classification report:\n",
              classification_report(y_test, y_pred_dt, target_names=labels))

        # save model + CM
        dt_model_path = out_dir / "dt_model.joblib"
        joblib.dump(dt, dt_model_path)
        print(f"✅ Saved Decision Tree model -> {dt_model_path}")

        dt_cm_path = out_dir / "dt_confusion_matrix.png"
        save_confusion_matrix(y_test, y_pred_dt, labels, dt_cm_path, "Decision Tree - Confusion Matrix")
        print(f"📊 Saved Decision Tree confusion matrix -> {dt_cm_path}")

    # ---- k-NN ----
    if args.model in ("knn", "both"):
        knn = KNeighborsClassifier(n_neighbors=args.n_neighbors)
        knn.fit(X_train, y_train)
        y_pred_knn = knn.predict(X_test)
        acc_knn = accuracy_score(y_test, y_pred_knn)
        print(f"[kNN] accuracy: {acc_knn:.4f}")
        print("[kNN] classification report:\n",
              classification_report(y_test, y_pred_knn, target_names=labels))

        # save model + CM
        knn_model_path = out_dir / "knn_model.joblib"
        joblib.dump(knn, knn_model_path)
        print(f"✅ Saved k-NN model -> {knn_model_path}")

        knn_cm_path = out_dir / "knn_confusion_matrix.png"
        save_confusion_matrix(y_test, y_pred_knn, labels, knn_cm_path, "k-NN - Confusion Matrix")
        print(f"📊 Saved k-NN confusion matrix -> {knn_cm_path}")


if __name__ == "__main__":
    main()