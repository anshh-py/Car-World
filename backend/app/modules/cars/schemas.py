from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class VariantImport(BaseModel):
	manufacturer_name: str = Field(min_length=1, max_length=255)
	model_name: str = Field(min_length=1, max_length=255)
	variant_name: str = Field(min_length=1, max_length=255)
	ex_showroom_price: Decimal
	displacement: int
	cylinders: int
	fuel_tank_capacity: Decimal
	fuel_type: str = Field(max_length=100)
	height: int
	length: Decimal
	width: int
	body_type: str = Field(max_length=100)
	seating_capacity: int
	transmission_type: str = Field(max_length=100)
	power: Decimal
	torque: Decimal


class ManufacturerRead(BaseModel):
	model_config = ConfigDict(from_attributes=True)

	id: int
	name: str


class ModelRead(BaseModel):
	model_config = ConfigDict(from_attributes=True)

	id: int
	manufacturer_id: int
	name: str


class VariantRead(BaseModel):
	model_config = ConfigDict(from_attributes=True)

	id: int
	model_id: int
	name: str
	ex_showroom_price: Decimal
	displacement: int
	cylinders: int
	fuel_tank_capacity: Decimal
	fuel_type: str
	height: int
	length: Decimal
	width: int
	body_type: str
	seating_capacity: int
	transmission_type: str
	power: Decimal
	torque: Decimal


class CarFilters(BaseModel):
	manufacturer: str | None = Field(default=None, min_length=1, max_length=255)
	model: str | None = Field(default=None, min_length=1, max_length=255)
	fuel_type: str | None = Field(default=None, min_length=1, max_length=100)
	body_type: str | None = Field(default=None, min_length=1, max_length=100)
	transmission_type: str | None = Field(default=None, min_length=1, max_length=100)
	min_price: Decimal | None = Field(default=None, ge=0)
	max_price: Decimal | None = Field(default=None, ge=0)

	@model_validator(mode="after")
	def validate_price_range(self) -> "CarFilters":
		if (
			self.min_price is not None
			and self.max_price is not None
			and self.min_price > self.max_price
		):
			raise ValueError("min_price must be less than or equal to max_price")
		return self


class CarSummary(BaseModel):
	id: int
	manufacturer: str
	model: str
	variant: str
	ex_showroom_price: Decimal
	fuel_type: str
	body_type: str
	transmission_type: str


class CarDetail(CarSummary):
	displacement: int
	cylinders: int
	fuel_tank_capacity: Decimal
	height: int
	length: Decimal
	width: int
	seating_capacity: int
	power: Decimal
	torque: Decimal


class CarPage(BaseModel):
	items: list[CarSummary]
	page: int
	page_size: int
	total_items: int
	total_pages: int


class ManufacturerModelsRead(BaseModel):
	manufacturer: ManufacturerRead
	items: list[ModelRead]


CarSort = Literal["price_asc", "price_desc", "name"]