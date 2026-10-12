import csv
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from math import ceil
from pathlib import Path
from typing import Protocol

from pydantic import ValidationError

from app.modules.cars.repository import CarRecord, CarRepository
from app.modules.cars.schemas import (
	CarDetail,
	CarFilters,
	CarPage,
	CarSummary,
	ManufacturerModelsRead,
	ManufacturerRead,
	ModelRead,
	VariantImport,
)

CSV_COLUMNS = (
	"Make",
	"Model",
	"Variant",
	"Ex-Showroom_Price",
	"Displacement",
	"Cylinders",
	"Fuel_Tank_Capacity",
	"Fuel_Type",
	"Height",
	"Length",
	"Width",
	"Body_Type",
	"Seating_Capacity",
	"Type",
	"Power.1",
	"Torque.1",
)


@dataclass(frozen=True)
class SourceRow:
	line_number: int
	raw_values: tuple[str, ...]
	variant: VariantImport


@dataclass(frozen=True)
class DuplicateSourceRow:
	first_line_number: int
	duplicate_line_number: int


@dataclass(frozen=True)
class SourceDataset:
	rows: list[SourceRow]
	all_rows: list[SourceRow]
	duplicates: list[DuplicateSourceRow]


@dataclass(frozen=True)
class ImportSummary:
	source_row_count: int
	duplicate_rows_skipped: int
	manufacturers_created: int
	models_created: int
	variants_created: int
	variants_already_present: int
	duplicate_rows: list[DuplicateSourceRow]
	zero_value_counts: dict[str, int]
	not_mentioned_counts: dict[str, int]
	short_length_rows: list[SourceRow]
	low_torque_rows: list[SourceRow]


class Repository(Protocol):
	async def get_or_create_manufacturer(self, name: str): ...

	async def get_or_create_model(self, manufacturer_id: int, name: str): ...

	async def get_variant(self, model_id: int, name: str): ...

	async def create_variant(self, model_id: int, row: VariantImport): ...


def parse_source_row(source: dict[str, str], line_number: int) -> SourceRow:
	try:
		price = Decimal(
			source["Ex-Showroom_Price"]
			.replace("₹", "")
			.replace(",", "")
			.strip()
		)
	except (InvalidOperation, AttributeError) as error:
		raise ValueError(
			f"Invalid Ex-Showroom_Price at CSV line {line_number}: "
			f"{source.get('Ex-Showroom_Price')!r}"
		) from error

	try:
		variant = VariantImport.model_validate(
			{
				"manufacturer_name": source["Make"],
				"model_name": source["Model"],
				"variant_name": source["Variant"],
				"ex_showroom_price": price,
				"displacement": source["Displacement"],
				"cylinders": source["Cylinders"],
				"fuel_tank_capacity": source["Fuel_Tank_Capacity"],
				"fuel_type": source["Fuel_Type"],
				"height": source["Height"],
				"length": source["Length"],
				"width": source["Width"],
				"body_type": source["Body_Type"],
				"seating_capacity": source["Seating_Capacity"],
				"transmission_type": source["Type"],
				"power": source["Power.1"],
				"torque": source["Torque.1"],
			}
		)
	except (KeyError, ValidationError) as error:
		raise ValueError(f"Invalid car data at CSV line {line_number}: {error}") from error

	raw_values = tuple(source[column] for column in CSV_COLUMNS)
	return SourceRow(line_number, raw_values, variant)


def read_source_csv(csv_path: Path) -> SourceDataset:
	rows: list[SourceRow] = []
	all_rows: list[SourceRow] = []
	duplicates: list[DuplicateSourceRow] = []
	first_line_by_record: dict[tuple[str, ...], int] = {}

	with csv_path.open("r", newline="", encoding="utf-8-sig") as source_file:
		reader = csv.DictReader(source_file)
		if tuple(reader.fieldnames or ()) != CSV_COLUMNS:
			raise ValueError(
				"CSV columns do not match the inspected dataset. "
				f"Expected {CSV_COLUMNS!r}; found {reader.fieldnames!r}."
			)

		for source in reader:
			line_number = reader.line_num
			if None in source or any(value is None for value in source.values()):
				raise ValueError(f"Malformed CSV row at line {line_number}.")

			row = parse_source_row(source, line_number)
			all_rows.append(row)
			first_line = first_line_by_record.get(row.raw_values)
			if first_line is not None:
				duplicates.append(DuplicateSourceRow(first_line, line_number))
				continue

			first_line_by_record[row.raw_values] = line_number
			rows.append(row)

	return SourceDataset(rows, all_rows, duplicates)


def _variant_values(row: VariantImport) -> dict[str, object]:
	return {
		"ex_showroom_price": row.ex_showroom_price,
		"displacement": row.displacement,
		"cylinders": row.cylinders,
		"fuel_tank_capacity": row.fuel_tank_capacity,
		"fuel_type": row.fuel_type,
		"height": row.height,
		"length": row.length,
		"width": row.width,
		"body_type": row.body_type,
		"seating_capacity": row.seating_capacity,
		"transmission_type": row.transmission_type,
		"power": row.power,
		"torque": row.torque,
	}


async def import_dataset(repository: Repository, csv_path: Path) -> ImportSummary:
	dataset = read_source_csv(csv_path)
	manufacturers_created = 0
	models_created = 0
	variants_created = 0
	variants_already_present = 0

	for source_row in dataset.rows:
		row = source_row.variant
		manufacturer, was_created = await repository.get_or_create_manufacturer(
			row.manufacturer_name
		)
		manufacturers_created += int(was_created)

		model, was_created = await repository.get_or_create_model(
			manufacturer.id, row.model_name
		)
		models_created += int(was_created)

		existing_variant = await repository.get_variant(model.id, row.variant_name)
		if existing_variant is not None:
			differences = {
				field: (getattr(existing_variant, field), value)
				for field, value in _variant_values(row).items()
				if getattr(existing_variant, field) != value
			}
			if differences:
				raise ValueError(
					"Existing variant conflicts with CSV data at line "
					f"{source_row.line_number}: {row.manufacturer_name} / "
					f"{row.model_name} / {row.variant_name}; differences={differences!r}"
				)
			variants_already_present += 1
			continue

		await repository.create_variant(model.id, row)
		variants_created += 1

	zero_value_counts = {
		field: sum(getattr(row.variant, field) == 0 for row in dataset.all_rows)
		for field in (
			"displacement",
			"cylinders",
			"fuel_tank_capacity",
			"height",
			"length",
			"width",
			"seating_capacity",
			"power",
			"torque",
		)
	}
	zero_value_counts = {
		field: count for field, count in zero_value_counts.items() if count
	}
	not_mentioned_counts = {
		field: sum(
			getattr(row.variant, field) == "Not Mentioned" for row in dataset.all_rows
		)
		for field in ("body_type", "transmission_type")
	}
	not_mentioned_counts = {
		field: count for field, count in not_mentioned_counts.items() if count
	}

	return ImportSummary(
		source_row_count=len(dataset.all_rows),
		duplicate_rows_skipped=len(dataset.duplicates),
		manufacturers_created=manufacturers_created,
		models_created=models_created,
		variants_created=variants_created,
		variants_already_present=variants_already_present,
		duplicate_rows=dataset.duplicates,
		zero_value_counts=zero_value_counts,
		not_mentioned_counts=not_mentioned_counts,
		short_length_rows=[
			row for row in dataset.all_rows if row.variant.length < 1000
		],
		low_torque_rows=[
			row for row in dataset.all_rows if row.variant.torque < 20
		],
	)


class InvalidCarFilterError(ValueError):
	pass


class CarCatalogService:
	def __init__(self, repository: CarRepository) -> None:
		self.repository = repository

	@staticmethod
	def _summary(record: CarRecord) -> CarSummary:
		variant = record.variant
		return CarSummary(
			id=variant.id,
			manufacturer=record.manufacturer_name,
			model=record.model_name,
			variant=variant.name,
			ex_showroom_price=variant.ex_showroom_price,
			fuel_type=variant.fuel_type,
			body_type=variant.body_type,
			transmission_type=variant.transmission_type,
		)

	@staticmethod
	def _detail(record: CarRecord) -> CarDetail:
		variant = record.variant
		return CarDetail(
			**CarCatalogService._summary(record).model_dump(),
			displacement=variant.displacement,
			cylinders=variant.cylinders,
			fuel_tank_capacity=variant.fuel_tank_capacity,
			height=variant.height,
			length=variant.length,
			width=variant.width,
			seating_capacity=variant.seating_capacity,
			power=variant.power,
			torque=variant.torque,
		)

	async def _page(
		self,
		filters: CarFilters,
		page: int,
		page_size: int,
		sort: str,
		search: str | None = None,
	) -> CarPage:
		invalid_values = await self.repository.invalid_filter_values(filters)
		if invalid_values:
			raise InvalidCarFilterError(
				"Unknown filter value(s): " + ", ".join(invalid_values)
			)
		records, total_items = await self.repository.list_cars(
			filters, page, page_size, sort, search
		)
		return CarPage(
			items=[self._summary(record) for record in records],
			page=page,
			page_size=page_size,
			total_items=total_items,
			total_pages=ceil(total_items / page_size) if total_items else 0,
		)

	async def list_cars(
		self, filters: CarFilters, page: int, page_size: int, sort: str
	) -> CarPage:
		return await self._page(filters, page, page_size, sort)

	async def search_cars(
		self, query: str, page: int, page_size: int, sort: str
	) -> CarPage:
		query = query.strip()
		if not query:
			raise InvalidCarFilterError("Search query must not be blank")
		return await self._page(CarFilters(), page, page_size, sort, query)

	async def get_car(self, car_id: int) -> CarDetail | None:
		record = await self.repository.get_car(car_id)
		return self._detail(record) if record is not None else None

	async def list_manufacturers(self) -> list[ManufacturerRead]:
		manufacturers = await self.repository.list_manufacturers()
		return [ManufacturerRead.model_validate(item) for item in manufacturers]

	async def get_manufacturer_models(
		self, name: str
	) -> ManufacturerModelsRead | None:
		manufacturer, models = await self.repository.get_manufacturer_models(name)
		if manufacturer is None:
			return None
		return ManufacturerModelsRead(
			manufacturer=ManufacturerRead.model_validate(manufacturer),
			items=[ModelRead.model_validate(model) for model in models],
		)