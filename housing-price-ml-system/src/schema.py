from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class HousePriceInput(BaseModel):
    """Validated request payload for house-price prediction."""

    model_config = ConfigDict(extra="forbid")

    property_type: str = Field(min_length=1, max_length=100)
    location: str = Field(min_length=1, max_length=200)
    city: str = Field(min_length=1, max_length=100)
    province_name: str = Field(min_length=1, max_length=100)
    baths: int = Field(ge=0, le=20)
    bedrooms: int = Field(ge=0, le=20)
    date_added: str = Field(min_length=8, max_length=30)
    Total_Area: float = Field(gt=0)
