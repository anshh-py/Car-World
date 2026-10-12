from dataclasses import dataclass

from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.cars.models import CarModel, Manufacturer, Variant
from app.modules.cars.schemas import CarFilters, VariantImport


@dataclass(frozen=True)
class CarRecord:
	variant: Variant
	model_name: str
	manufacturer_name: str


class CarRepository:
	def __init__(self, session: AsyncSession) -> None:
		self.session = session

	async def get_or_create_manufacturer(self, name: str) -> tuple[Manufacturer, bool]:
		manufacturer = await self.session.scalar(
			select(Manufacturer).where(Manufacturer.name == name)
		)
		if manufacturer is not None:
			return manufacturer, False

		manufacturer = Manufacturer(name=name)
		self.session.add(manufacturer)
		await self.session.flush()
		return manufacturer, True

	async def get_or_create_model(
		self, manufacturer_id: int, name: str
	) -> tuple[CarModel, bool]:
		model = await self.session.scalar(
			select(CarModel).where(
				CarModel.manufacturer_id == manufacturer_id,
				CarModel.name == name,
			)
		)
		if model is not None:
			return model, False

		model = CarModel(manufacturer_id=manufacturer_id, name=name)
		self.session.add(model)
		await self.session.flush()
		return model, True

	async def get_variant(self, model_id: int, name: str) -> Variant | None:
		return await self.session.scalar(
			select(Variant).where(Variant.model_id == model_id, Variant.name == name)
		)

	async def create_variant(self, model_id: int, row: VariantImport) -> Variant:
		variant = Variant(
			model_id=model_id,
			name=row.variant_name,
			ex_showroom_price=row.ex_showroom_price,
			displacement=row.displacement,
			cylinders=row.cylinders,
			fuel_tank_capacity=row.fuel_tank_capacity,
			fuel_type=row.fuel_type,
			height=row.height,
			length=row.length,
			width=row.width,
			body_type=row.body_type,
			seating_capacity=row.seating_capacity,
			transmission_type=row.transmission_type,
			power=row.power,
			torque=row.torque,
		)
		self.session.add(variant)
		await self.session.flush()
		return variant

	async def counts(self) -> dict[str, int]:
		counts: dict[str, int] = {}
		for label, entity in (
			("manufacturers", Manufacturer),
			("models", CarModel),
			("variants", Variant),
		):
			counts[label] = int(
				await self.session.scalar(select(func.count()).select_from(entity)) or 0
			)
		return counts

	async def invalid_filter_values(self, filters: CarFilters) -> list[str]:
		checks = []
		if filters.manufacturer is not None:
			checks.append((
				"manufacturer",
				select(Manufacturer.id)
				.where(func.lower(Manufacturer.name) == filters.manufacturer.casefold())
				.exists(),
			))
		if filters.model is not None:
			checks.append((
				"model",
				select(CarModel.id)
				.where(func.lower(CarModel.name) == filters.model.casefold())
				.exists(),
			))
		for filter_name, column in (
			("fuel_type", Variant.fuel_type),
			("body_type", Variant.body_type),
			("transmission_type", Variant.transmission_type),
		):
			value = getattr(filters, filter_name)
			if value is not None:
				checks.append((
					filter_name,
					select(Variant.id)
					.where(func.lower(column) == value.casefold())
					.exists(),
				))

		if not checks:
			return []

		result = await self.session.execute(
			select(*(expression.label(name) for name, expression in checks))
		)
		values = result.one()
		return [name for (name, _), exists in zip(checks, values) if not exists]

	@staticmethod
	def _filter_conditions(filters: CarFilters) -> list[object]:
		conditions: list[object] = []
		for value, column in (
			(filters.manufacturer, Manufacturer.name),
			(filters.model, CarModel.name),
			(filters.fuel_type, Variant.fuel_type),
			(filters.body_type, Variant.body_type),
			(filters.transmission_type, Variant.transmission_type),
		):
			if value is not None:
				conditions.append(func.lower(column) == value.casefold())
		if filters.min_price is not None:
			conditions.append(Variant.ex_showroom_price >= filters.min_price)
		if filters.max_price is not None:
			conditions.append(Variant.ex_showroom_price <= filters.max_price)
		return conditions

	@staticmethod
	def _search_condition(query: str) -> object:
		escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
		pattern = f"%{escaped}%"
		return or_(
			Manufacturer.name.ilike(pattern, escape="\\"),
			CarModel.name.ilike(pattern, escape="\\"),
			Variant.name.ilike(pattern, escape="\\"),
		)

	async def list_cars(
		self,
		filters: CarFilters,
		page: int,
		page_size: int,
		sort: str,
		search: str | None = None,
	) -> tuple[list[CarRecord], int]:
		conditions = self._filter_conditions(filters)
		if search is not None:
			conditions.append(self._search_condition(search))

		count_statement = (
			select(func.count(Variant.id))
			.select_from(Variant)
			.join(Variant.model)
			.join(CarModel.manufacturer)
			.where(*conditions)
		)
		total_items = int(await self.session.scalar(count_statement) or 0)

		if sort == "price_asc":
			ordering = (Variant.ex_showroom_price.asc(), Manufacturer.name.asc(), CarModel.name.asc(), Variant.name.asc())
		elif sort == "price_desc":
			ordering = (Variant.ex_showroom_price.desc(), Manufacturer.name.asc(), CarModel.name.asc(), Variant.name.asc())
		else:
			ordering = (Manufacturer.name.asc(), CarModel.name.asc(), Variant.name.asc())

		statement = (
			select(Variant, CarModel.name, Manufacturer.name)
			.select_from(Variant)
			.join(Variant.model)
			.join(CarModel.manufacturer)
			.where(*conditions)
			.order_by(*ordering)
			.offset((page - 1) * page_size)
			.limit(page_size)
		)
		rows = (await self.session.execute(statement)).all()
		return [CarRecord(variant, model_name, manufacturer_name) for variant, model_name, manufacturer_name in rows], total_items

	async def get_car(self, car_id: int) -> CarRecord | None:
		statement = (
			select(Variant, CarModel.name, Manufacturer.name)
			.select_from(Variant)
			.join(Variant.model)
			.join(CarModel.manufacturer)
			.where(Variant.id == car_id)
		)
		row = (await self.session.execute(statement)).one_or_none()
		if row is None:
			return None
		variant, model_name, manufacturer_name = row
		return CarRecord(variant, model_name, manufacturer_name)

	async def list_manufacturers(self) -> list[Manufacturer]:
		result = await self.session.scalars(
			select(Manufacturer).order_by(func.lower(Manufacturer.name), Manufacturer.id)
		)
		return list(result.all())

	async def get_manufacturer_models(
		self, name: str
	) -> tuple[Manufacturer | None, list[CarModel]]:
		manufacturer = await self.session.scalar(
			select(Manufacturer).where(func.lower(Manufacturer.name) == name.casefold())
		)
		if manufacturer is None:
			return None, []

		models = await self.session.scalars(
			select(CarModel)
			.where(CarModel.manufacturer_id == manufacturer.id)
			.order_by(func.lower(CarModel.name), CarModel.id)
		)
		return manufacturer, list(models.all())