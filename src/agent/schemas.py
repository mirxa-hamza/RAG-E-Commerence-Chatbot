"""Bounded schemas shared by HTTP handlers, tools, and final answer validation."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, allow_inf_nan=False)


class CatalogQuery(StrictModel):
    query: str = Field("", max_length=200)
    category: str | None = Field(None, max_length=100)
    brand: str | None = Field(None, max_length=100)
    colors: list[str] = Field(default_factory=list, max_length=10)
    min_price: float | None = Field(None, ge=0)
    max_price: float | None = Field(None, ge=0)
    min_rating: float | None = Field(None, ge=0, le=5)
    limit: int = Field(6, ge=1, le=12)

    @model_validator(mode="after")
    def ranges(self):
        if self.min_price is not None and self.max_price is not None and self.min_price > self.max_price:
            raise ValueError("Minimum price must not exceed maximum price")
        if any(not c.strip() or len(c) > 40 for c in self.colors):
            raise ValueError("Colors must contain 1–40 characters")
        return self


class ReviewQuery(StrictModel):
    question: str = Field(min_length=1, max_length=1000)
    product_ids: list[str] = Field(default_factory=list, max_length=12)
    min_rating: float | None = Field(None, ge=0, le=5)
    max_rating: float | None = Field(None, ge=0, le=5)
    limit: int = Field(4, ge=1, le=8)

    @model_validator(mode="after")
    def ranges(self):
        if self.min_rating is not None and self.max_rating is not None and self.min_rating > self.max_rating:
            raise ValueError("Minimum rating must not exceed maximum rating")
        if any(not p or len(p) > 40 for p in self.product_ids):
            raise ValueError("Invalid product ID")
        return self


class Budget(StrictModel):
    min: float | None = Field(None, ge=0)
    max: float | None = Field(None, ge=0)

    @model_validator(mode="after")
    def ranges(self):
        if self.min is None and self.max is None:
            raise ValueError("Provide at least one budget bound")
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError("Invalid budget range")
        return self


class PreferenceUpdate(StrictModel):
    field: Literal["clothing_size", "budget", "style_notes", "color_preference", "favorite_brands"]
    action: Literal["set", "add", "remove", "clear"]
    value: str | Budget | None = None

    @model_validator(mode="after")
    def validate_action(self):
        if self.action == "clear":
            if self.value is not None:
                raise ValueError("Clear does not accept a value")
            return self
        if self.field in ("color_preference", "favorite_brands"):
            if self.action not in ("add", "remove"):
                raise ValueError("List preferences use add/remove/clear")
        elif self.action != "set":
            raise ValueError("Scalar preferences use set/clear")
        if self.field == "budget":
            if not isinstance(self.value, Budget):
                raise ValueError("Budget must contain min and/or max")
        elif not isinstance(self.value, str) or not self.value.strip() or len(self.value) > (500 if self.field == "style_notes" else 80):
            raise ValueError("Preference text is empty or too long")
        return self


class Product(StrictModel):
    parent_asin: str
    title: str
    brand: str | None = None
    price: float | None = None
    image_url: str | None = None
    average_rating: float | None = None
    rating_number: int = 0
    review_excerpt: str | None = None


class Citation(StrictModel):
    id: str
    parent_asin: str
    review_id: str
    excerpt: str
    rating: float | None = None


class AnswerDraft(StrictModel):
    product_ids: list[str] = Field(default_factory=list, max_length=12)
    citation_ids: list[str] = Field(default_factory=list, max_length=20)
    suggested_relaxations: list[str] = Field(default_factory=list, max_length=4)
    answer: str = Field(min_length=1, max_length=6000)


class ShoppingRequest(StrictModel):
    question: str = Field(min_length=1, max_length=2000)
    session_id: str | None = Field(None, min_length=24, max_length=24)


class ShoppingResponse(StrictModel):
    session_id: str
    answer: str
    products: list[Product] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    suggested_relaxations: list[str] = Field(default_factory=list)
