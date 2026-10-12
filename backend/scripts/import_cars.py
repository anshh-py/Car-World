import argparse
import asyncio
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = BACKEND_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.core.database import async_session_factory
from app.modules.cars.repository import CarRepository
from app.modules.cars.service import import_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import the CarWorld car dataset.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=PROJECT_DIR / "Data" / "car_dataset.csv",
        help="Path to the source CSV (default: Data/car_dataset.csv).",
    )
    return parser.parse_args()


async def run_import(csv_path: Path) -> None:
    if not csv_path.is_file():
        raise FileNotFoundError(f"Dataset CSV not found: {csv_path}")

    async with async_session_factory() as session:
        async with session.begin():
            repository = CarRepository(session)
            summary = await import_dataset(repository, csv_path)

        counts = await repository.counts()

    print(f"Source rows: {summary.source_row_count}")
    print(f"Exact duplicate rows skipped: {summary.duplicate_rows_skipped}")
    for duplicate in summary.duplicate_rows:
        print(
            "  duplicate CSV line "
            f"{duplicate.duplicate_line_number} (first seen at line "
            f"{duplicate.first_line_number})"
        )
    print(f"Manufacturers created: {summary.manufacturers_created}")
    print(f"Models created: {summary.models_created}")
    print(f"Variants created: {summary.variants_created}")
    print(f"Variants already present unchanged: {summary.variants_already_present}")
    print(
        "Database totals: "
        f"manufacturers={counts['manufacturers']}, "
        f"models={counts['models']}, variants={counts['variants']}"
    )

    print("Data quality observations (values were not changed):")
    print(f"  zero-valued fields: {summary.zero_value_counts}")
    print(f"  'Not Mentioned' fields: {summary.not_mentioned_counts}")
    print(f"  length values below 1000: {len(summary.short_length_rows)}")
    for row in summary.short_length_rows:
        item = row.variant
        print(
            f"    CSV line {row.line_number}: {item.manufacturer_name} / "
            f"{item.model_name} / {item.variant_name}, length={item.length}"
        )
    print(f"  torque values below 20: {len(summary.low_torque_rows)}")
    for row in summary.low_torque_rows:
        item = row.variant
        print(
            f"    CSV line {row.line_number}: {item.manufacturer_name} / "
            f"{item.model_name} / {item.variant_name}, torque={item.torque}"
        )


def main() -> None:
    args = parse_args()
    asyncio.run(run_import(args.csv.resolve()))


if __name__ == "__main__":
    main()