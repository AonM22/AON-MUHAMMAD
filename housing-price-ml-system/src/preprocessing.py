from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_COLUMN = "price_in_lac"
BASE_FEATURE_COLUMNS = [
    "location",
    "baths",
    "property_type",
    "city",
    "province_name",
    "bedrooms",
    "Total_Area",
]
ENGINEERED_FEATURE_COLUMNS = [
    "area_bins",
    "date_added_year",
    "date_added_month",
    "date_added_day",
    "date_added_dayofweek",
]
TRAINING_FEATURE_COLUMNS = [*BASE_FEATURE_COLUMNS, *ENGINEERED_FEATURE_COLUMNS]


def load_dataset(csv_path: str) -> pd.DataFrame:
    """Load housing data from CSV."""
    return pd.read_csv(csv_path)


def split_feature_types(features: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Return numeric and categorical feature column names."""
    numeric_features = features.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = features.select_dtypes(exclude=["number", "bool"]).columns.tolist()
    return numeric_features, categorical_features


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    """Build sklearn ColumnTransformer for numeric and categorical preprocessing."""
    numeric_features, categorical_features = split_feature_types(features)

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )


def summarize_missing_values(df: pd.DataFrame) -> Dict[str, int]:
    """Return missing-value counts by column."""
    return {column: int(count) for column, count in df.isna().sum().to_dict().items() if int(count) > 0}


def select_training_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Select canonical training features and preserve deterministic order."""
    missing_columns = [column for column in TRAINING_FEATURE_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required features after engineering: {missing_columns}")
    return df[TRAINING_FEATURE_COLUMNS].copy()
