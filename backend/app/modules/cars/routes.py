from typing import Annotated
from decimal import Decimal

from fastapi import APIRouter, Depends, HTTPException, Path, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db_session
from app.modules.cars.repository import CarRepository
from app.modules.cars.schemas import (
	CarDetail,
	CarFilters,
	CarPage,
	CarSort,
	ManufacturerModelsRead,
	ManufacturerRead,
)
from app.modules.cars.service import CarCatalogService, InvalidCarFilterError

router = APIRouter()


def get_car_catalog_service(
	session: AsyncSession = Depends(get_db_session),
) -> CarCatalogService:
	return CarCatalogService(CarRepository(session))


def _raise_invalid_filter(error: InvalidCarFilterError) -> HTTPException:
	return HTTPException(status_code=422, detail=str(error))


@router.get("/cars/search", response_model=CarPage, tags=["cars"])
async def search_cars(
	q: Annotated[str, Query(min_length=1, max_length=100)],
	page: Annotated[int, Query(ge=1)] = 1,
	page_size: Annotated[int, Query(ge=1, le=100)] = 20,
	sort: CarSort = "name",
	service: CarCatalogService = Depends(get_car_catalog_service),
) -> CarPage:
	try:
		return await service.search_cars(q, page, page_size, sort)
	except InvalidCarFilterError as error:
		raise _raise_invalid_filter(error) from error


@router.get("/cars", response_model=CarPage, tags=["cars"])
async def list_cars(
	page: Annotated[int, Query(ge=1)] = 1,
	page_size: Annotated[int, Query(ge=1, le=100)] = 20,
	sort: CarSort = "name",
	manufacturer: Annotated[str | None, Query(min_length=1, max_length=255)] = None,
	model: Annotated[str | None, Query(min_length=1, max_length=255)] = None,
	fuel_type: Annotated[str | None, Query(min_length=1, max_length=100)] = None,
	body_type: Annotated[str | None, Query(min_length=1, max_length=100)] = None,
	transmission_type: Annotated[str | None, Query(min_length=1, max_length=100)] = None,
	min_price: Annotated[Decimal | None, Query(ge=0)] = None,
	max_price: Annotated[Decimal | None, Query(ge=0)] = None,
	service: CarCatalogService = Depends(get_car_catalog_service),
) -> CarPage:
	if (
		min_price is not None
		and max_price is not None
		and min_price > max_price
	):
		raise HTTPException(
			status_code=422,
			detail="min_price must be less than or equal to max_price",
		)
	filters = CarFilters(
		manufacturer=manufacturer,
		model=model,
		fuel_type=fuel_type,
		body_type=body_type,
		transmission_type=transmission_type,
		min_price=min_price,
		max_price=max_price,
	)
	try:
		return await service.list_cars(filters, page, page_size, sort)
	except InvalidCarFilterError as error:
		raise _raise_invalid_filter(error) from error


@router.get("/cars/{car_id}", response_model=CarDetail, tags=["cars"])
async def get_car(
	car_id: Annotated[int, Path(ge=1)],
	service: CarCatalogService = Depends(get_car_catalog_service),
) -> CarDetail:
	car = await service.get_car(car_id)
	if car is None:
		raise HTTPException(status_code=404, detail="Car variant not found")
	return car


@router.get("/manufacturers", response_model=list[ManufacturerRead], tags=["manufacturers"])
async def list_manufacturers(
	service: CarCatalogService = Depends(get_car_catalog_service),
) -> list[ManufacturerRead]:
	return await service.list_manufacturers()


@router.get(
	"/manufacturers/{name}/models",
	response_model=ManufacturerModelsRead,
	tags=["manufacturers"],
)
async def list_manufacturer_models(
	name: Annotated[str, Path(min_length=1, max_length=255)],
	service: CarCatalogService = Depends(get_car_catalog_service),
) -> ManufacturerModelsRead:
	result = await service.get_manufacturer_models(name)
	if result is None:
		raise HTTPException(status_code=404, detail="Manufacturer not found")
	return result