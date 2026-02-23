from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def engineer_features(
    df: pd.DataFrame,
    *,
    include_price_per_sqft: bool = False,
) -> pd.DataFrame:
    """Create engineered features for housing-price modeling."""
    data = df.copy()

    if "Total_Area" in data.columns:
        area = pd.to_numeric(data["Total_Area"], errors="coerce")
        data["area_bins"] = pd.cut(
            area,
            bins=[-np.inf, 5, 10, 20, np.inf],
            labels=["very_small", "small", "medium", "large"],
        ).astype("object")

    if include_price_per_sqft and {"Total_Area", "price_in_lac"}.issubset(data.columns):
        area = pd.to_numeric(data["Total_Area"], errors="coerce").replace(0, np.nan)
        price = pd.to_numeric(data["price_in_lac"], errors="coerce")
        data["price_per_sqft"] = (price / area).replace([np.inf, -np.inf], np.nan)

    if "date_added" in data.columns:
        parsed_date = pd.to_datetime(data["date_added"], errors="coerce")
        data["date_added_year"] = parsed_date.dt.year
        data["date_added_month"] = parsed_date.dt.month
        data["date_added_day"] = parsed_date.dt.day
        data["date_added_dayofweek"] = parsed_date.dt.dayofweek

    return data


def drop_unused_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Drop columns only when they are present in the frame."""
    drop_list = [column for column in columns if column in df.columns]
    return df.drop(columns=drop_list)
