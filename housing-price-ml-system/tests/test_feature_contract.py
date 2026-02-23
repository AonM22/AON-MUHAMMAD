from __future__ import annotations

import pandas as pd

from src.feature_engineering import engineer_features
from src.preprocessing import TRAINING_FEATURE_COLUMNS, select_training_columns


def test_feature_contract_columns_present_and_ordered() -> None:
    frame = pd.DataFrame(
        [
            {
                "location": "DHA Defence",
                "baths": 4,
                "property_type": "House",
                "city": "Islamabad",
                "province_name": "Islamabad Capital",
                "bedrooms": 4,
                "Total_Area": 12.0,
                "date_added": "2023-05-11",
            }
        ]
    )

    engineered = engineer_features(frame)
    selected = select_training_columns(engineered)

    assert list(selected.columns) == TRAINING_FEATURE_COLUMNS
