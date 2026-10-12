import csv
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.modules.cars.models import CarModel, Manufacturer, Variant
from app.modules.cars.service import CSV_COLUMNS, import_dataset, parse_source_row, read_source_csv


def make_source_row(**overrides: str) -> dict[str, str]:
    row = {
        "Make": "Hyundai",
        "Model": "Santro",
        "Variant": "Era Mt",
        "Ex-Showroom_Price": "₹ 429,990.000000000000000",
        "Displacement": "1086",
        "Cylinders": "4",
        "Fuel_Tank_Capacity": "35",
        "Fuel_Type": "Petrol",
        "Height": "1560",
        "Length": "3610",
        "Width": "1645",
        "Body_Type": "Hatchback",
        "Seating_Capacity": "5",
        "Type": "Manual",
        "Power.1": "69",
        "Torque.1": "99.04",
    }
    row.update(overrides)
    return row


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


class MemoryRepository:
    def __init__(self) -> None:
        self.manufacturers: dict[str, SimpleNamespace] = {}
        self.models: dict[tuple[int, str], SimpleNamespace] = {}
        self.variants: dict[tuple[int, str], SimpleNamespace] = {}
        self.next_id = 1

    def _id(self) -> int:
        value = self.next_id
        self.next_id += 1
        return value

    async def get_or_create_manufacturer(self, name: str):
        manufacturer = self.manufacturers.get(name)
        if manufacturer is not None:
            return manufacturer, False
        manufacturer = SimpleNamespace(id=self._id(), name=name)
        self.manufacturers[name] = manufacturer
        return manufacturer, True

    async def get_or_create_model(self, manufacturer_id: int, name: str):
        key = (manufacturer_id, name)
        model = self.models.get(key)
        if model is not None:
            return model, False
        model = SimpleNamespace(id=self._id(), manufacturer_id=manufacturer_id, name=name)
        self.models[key] = model
        return model, True

    async def get_variant(self, model_id: int, name: str):
        return self.variants.get((model_id, name))

    async def create_variant(self, model_id: int, row):
        variant = SimpleNamespace(
            id=self._id(),
            model_id=model_id,
            name=row.variant_name,
            **{
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
            },
        )
        self.variants[(model_id, row.variant_name)] = variant
        return variant


def test_models_have_expected_foreign_keys_and_unique_keys() -> None:
    assert {column.name for column in Manufacturer.__table__.primary_key} == {"id"}
    assert {column.name for column in CarModel.__table__.primary_key} == {"id"}
    assert {column.name for column in Variant.__table__.primary_key} == {"id"}
    assert next(iter(CarModel.__table__.c.manufacturer_id.foreign_keys)).target_fullname == (
        "manufacturers.id"
    )
    assert next(iter(Variant.__table__.c.model_id.foreign_keys)).target_fullname == (
        "models.id"
    )
    assert {constraint.name for constraint in CarModel.__table__.constraints} >= {
        "uq_models_manufacturer_name"
    }
    assert {constraint.name for constraint in Variant.__table__.constraints} >= {
        "uq_variants_model_name"
    }


def test_parse_source_row_preserves_source_values() -> None:
    parsed = parse_source_row(
        make_source_row(Body_Type="Not Mentioned", Displacement="0"), 2
    )

    assert parsed.variant.ex_showroom_price == Decimal("429990.000000000000000")
    assert parsed.variant.body_type == "Not Mentioned"
    assert parsed.variant.displacement == 0
    assert parsed.variant.transmission_type == "Manual"


def test_read_csv_skips_and_reports_exact_duplicate_rows(tmp_path: Path) -> None:
    path = tmp_path / "cars.csv"
    row = make_source_row()
    write_csv(path, [row, row, make_source_row(Variant="Magna Mt")])

    dataset = read_source_csv(path)

    assert len(dataset.all_rows) == 3
    assert len(dataset.rows) == 2
    assert [(item.first_line_number, item.duplicate_line_number) for item in dataset.duplicates] == [
        (2, 3)
    ]


@pytest.mark.asyncio
async def test_import_is_idempotent_and_reports_source_duplicates(tmp_path: Path) -> None:
    path = tmp_path / "cars.csv"
    row = make_source_row()
    write_csv(path, [row, row])
    repository = MemoryRepository()

    first = await import_dataset(repository, path)
    second = await import_dataset(repository, path)

    assert first.duplicate_rows_skipped == second.duplicate_rows_skipped == 1
    assert first.manufacturers_created == 1
    assert first.models_created == 1
    assert first.variants_created == 1
    assert second.manufacturers_created == 0
    assert second.models_created == 0
    assert second.variants_created == 0
    assert second.variants_already_present == 1
    assert len(repository.variants) == 1