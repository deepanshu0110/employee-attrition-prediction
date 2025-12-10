
#---

## 3. `src/data_preprocessing.py`

#This module centralizes how features are prepared (column selections and preprocessor).  
#Other scripts import from here so everything stays consistent.

#```python
# src/data_preprocessing.py

"""
Data preprocessing utilities for the Employee Attrition project.

This file:
- Defines which columns to drop
- Builds the preprocessing pipeline (scaling + encoding)
- Provides helper to load the raw dataset

Other scripts import from here to guarantee consistency.
"""

from pathlib import Path
from typing import Tuple, List

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Name of the target column in hr_data.csv
TARGET_COL = "Attrition"


def get_project_root() -> Path:
    """Return the project root path (two levels above this file)."""
    return Path(__file__).resolve().parent.parent


def load_raw_data() -> pd.DataFrame:
    """
    Load the raw HR dataset from data/raw/hr_data.csv.

    Returns:
        df (pd.DataFrame): Full dataset including the target column.
    """
    root = get_project_root()
    data_path = root / "data" / "raw" / "hr_data.csv"
    df = pd.read_csv(data_path)

    return df


def clean_and_split_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Drop non-informative / identifier columns and split into X (features) and y (target).

    Args:
        df: Original dataframe including target.

    Returns:
        X: Features dataframe
        y: Target series (0/1)
    """
    # Columns that are ID-like or constant in this dataset
    columns_to_drop = [
        "EmployeeCount",
        "Over18",
        "StandardHours",
        "EmployeeNumber",
    ]

    # Only drop if present
    columns_to_drop = [c for c in columns_to_drop if c in df.columns]

    # Map target Attrition Yes/No to 1/0
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataset.")

    y = df[TARGET_COL].map({"Yes": 1, "No": 0})
    if y.isna().any():
        raise ValueError("Unexpected values in Attrition column. Expected only 'Yes' or 'No'.")

    X = df.drop(columns=columns_to_drop + [TARGET_COL])

    return X, y


def get_feature_types(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Identify numeric and categorical columns.

    Args:
        X: Features dataframe.

    Returns:
        numeric_features: list of numeric column names
        categorical_features: list of categorical column names
    """
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    return numeric_features, categorical_features


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Build a ColumnTransformer that:
    - Scales numeric features
    - One-hot encodes categorical features

    Args:
        X: Features dataframe (used only to detect column types).

    Returns:
        preprocessor: ColumnTransformer
    """
    numeric_features, categorical_features = get_feature_types(X)

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor


# Key takeaways (for quick review):
# - Always centralize preprocessing logic so all scripts are consistent.
# - We drop constant / ID columns to avoid noise and leakage.
# - Target mapping (Yes/No -> 1/0) is standardized here.
# - ColumnTransformer keeps numeric and categorical pipelines clean and scalable.
