# src/evaluate.py

"""
Evaluate the saved best model with detailed metrics and plots.

Outputs (saved in /outputs):
- confusion_matrix.png
- roc_curve.png
- threshold_metrics.png

This script helps you understand model performance beyond a single metric.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from data_preprocessing import load_raw_data, clean_and_split_features


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def main() -> None:
    # 1. Load data and recreate the same train/test split
    df = load_raw_data()
    X, y = clean_and_split_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # 2. Load best model pipeline
    root = get_project_root()
    model_path = root / "models" / "best_attrition_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at {model_path}. Train the model first with model_training.py."
        )

    model = joblib.load(model_path)
    print(f"Loaded model from: {model_path.resolve()}")

    # 3. Predictions and probabilities
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # 4. Metrics
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary"
    )
    auc = roc_auc_score(y_test, y_proba)

    print("\n================ METRICS ================")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1-score  : {f1:.4f}")
    print(f"ROC AUC   : {auc:.4f}")
    print("=========================================\n")

    print("Classification report:")
    print(classification_report(y_test, y_pred))

    # 5. Outputs directory
    outputs_dir = root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # 6. Confusion Matrix plot
    fig_cm, ax_cm = plt.subplots(figsize=(5, 5))
    cm_display = ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, ax=ax_cm, cmap="Blues"
    )
    ax_cm.set_title("Confusion Matrix - Attrition Model")
    cm_path = outputs_dir / "confusion_matrix.png"
    fig_cm.tight_layout()
    fig_cm.savefig(cm_path, dpi=120)
    plt.close(fig_cm)
    print(f"Saved confusion matrix to: {cm_path.resolve()}")

    # 7. ROC Curve plot
    fpr, tpr, _ = roc_curve(y_test, y_proba)

    fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
    ax_roc.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.3f})")
    ax_roc.plot([0, 1], [0, 1], linestyle="--", label="Random")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curve - Attrition Model")
    ax_roc.legend()
    roc_path = outputs_dir / "roc_curve.png"
    fig_roc.tight_layout()
    fig_roc.savefig(roc_path, dpi=120)
    plt.close(fig_roc)
    print(f"Saved ROC curve to: {roc_path.resolve()}")

    # 8. Threshold vs precision/recall/F1
    thresholds = np.linspace(0.1, 0.9, 17)
    precisions, recalls, f1_scores = [], [], []

    for thr in thresholds:
        y_thr = (y_proba >= thr).astype(int)
        p, r, f, _ = precision_recall_fscore_support(
            y_test, y_thr, average="binary", zero_division=0
        )
        precisions.append(p)
        recalls.append(r)
        f1_scores.append(f)

    fig_thr, ax_thr = plt.subplots(figsize=(7, 5))
    ax_thr.plot(thresholds, precisions, marker="o", label="Precision")
    ax_thr.plot(thresholds, recalls, marker="o", label="Recall")
    ax_thr.plot(thresholds, f1_scores, marker="o", label="F1-score")
    ax_thr.set_xlabel("Decision Threshold")
    ax_thr.set_ylabel("Score")
    ax_thr.set_title("Threshold vs Precision/Recall/F1")
    ax_thr.grid(True, linestyle="--", alpha=0.5)
    ax_thr.legend()
    thr_path = outputs_dir / "threshold_metrics.png"
    fig_thr.tight_layout()
    fig_thr.savefig(thr_path, dpi=120)
    plt.close(fig_thr)
    print(f"Saved threshold metrics plot to: {thr_path.resolve()}")

    # 9. Simple textual interpretation
    print("\nRESULTS INTERPRETATION (high level):")
    print(
        "- Accuracy shows overall correctness, but in HR we also care about Recall "
        "(catching employees who actually leave)."
    )
    print(
        "- If Recall is low, the model is missing many at-risk employees (false negatives)."
    )
    print(
        "- Threshold plot helps choose a probability cutoff that balances Precision vs Recall "
        "based on business cost."
    )


if __name__ == "__main__":
    main()

    # Key takeaways:
    # - Never rely on a single metric; combine AUC, F1, Precision, and Recall.
    # - Visuals (confusion matrix, ROC) make communication with stakeholders easier.
    # - Threshold analysis is critical when costs of FP vs FN are different.
    # - Keep evaluation separate from training for cleaner code and experimentation.
