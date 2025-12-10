# src/model_training.py

"""
Train classification models to predict employee attrition.

Steps:
1. Load raw data.
2. Clean and split into X, y.
3. Build preprocessing pipeline (scaling + encoding).
4. Train multiple models (Logistic Regression, Random Forest).
5. Compare ROC AUC on a hold-out test set.
6. Save the best model pipeline to /models.
"""

from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import joblib

from data_preprocessing import (
    load_raw_data,
    clean_and_split_features,
    build_preprocessor,
)


def get_project_root() -> Path:
    """Return the project root path."""
    return Path(__file__).resolve().parent.parent


def main() -> None:
    # 1. Load dataset
    df = load_raw_data()
    print(f"Loaded dataset with shape: {df.shape}")

    # 2. Clean + split
    X, y = clean_and_split_features(df)
    print(f"Features shape: {X.shape}, Target length: {len(y)}")

    # 3. Train/Test split (hold-out test for evaluation)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # 4. Build preprocessor based on training data
    preprocessor = build_preprocessor(X_train)

    # 5. Define candidate models
    models = {
        "log_reg": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        ),
    }

    best_model_name = None
    best_pipeline = None
    best_auc = -np.inf

    # 6. Fit each model as a pipeline and evaluate on test set
    for name, clf in models.items():
        print(f"\nTraining model: {name}")
        pipe = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", clf),
            ]
        )
        pipe.fit(X_train, y_train)

        # Predict probabilities for ROC AUC
        y_proba = pipe.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
        print(f"{name} ROC AUC: {auc:.4f}")

        if auc > best_auc:
            best_auc = auc
            best_model_name = name
            best_pipeline = pipe

    print("\n==============================")
    print(f"Best model: {best_model_name} (ROC AUC = {best_auc:.4f})")
    print("==============================\n")

    # Optional: quick classification report for the best model
    y_pred_best = best_pipeline.predict(X_test)
    print("Classification report of best model:")
    print(classification_report(y_test, y_pred_best))

    # 7. Save best pipeline
    root = get_project_root()
    models_dir = root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "best_attrition_model.joblib"
    joblib.dump(best_pipeline, model_path)
    print(f"\nSaved best model pipeline to: {model_path.resolve()}")


if __name__ == "__main__":
    main()

    # Key takeaways:
    # - Use a separate hold-out test set to get unbiased performance.
    # - Wrap preprocessing + model inside a single Pipeline for easy reuse.
    # - Evaluate multiple models on the same metric (here, ROC AUC).
    # - Save the best performing pipeline for later evaluation and deployment.
