import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.core.config import settings
from app.core.database import get_db_session
from app.main import app


@pytest_asyncio.fixture
async def client() -> AsyncClient:
	test_engine = create_async_engine(settings.database_url, poolclass=NullPool)
	test_session_factory = async_sessionmaker(test_engine, expire_on_commit=False)

	async def override_get_db_session():
		async with test_session_factory() as session:
			yield session

	app.dependency_overrides[get_db_session] = override_get_db_session
	try:
		async with AsyncClient(
			transport=ASGITransport(app=app),
			base_url="http://test",
		) as test_client:
			yield test_client
	finally:
		app.dependency_overrides.pop(get_db_session, None)
		await test_engine.dispose()


@pytest.mark.asyncio
async def test_list_cars_paginates_and_filters(client: AsyncClient) -> None:
	response = await client.get(
		"/api/v1/cars",
		params={"manufacturer": "hYunDai", "model": "sAntRo", "page_size": 3},
	)

	assert response.status_code == 200
	data = response.json()
	assert data["page"] == 1
	assert data["page_size"] == 3
	assert data["total_items"] >= len(data["items"]) > 0
	assert len(data["items"]) == 3
	assert data["items"][0]["manufacturer"] == "Hyundai"
	assert data["total_pages"] >= 1


@pytest.mark.asyncio
async def test_car_detail_returns_specs_and_unknown_id_is_404(
	client: AsyncClient,
) -> None:
	listing = await client.get(
		"/api/v1/cars", params={"manufacturer": "Hyundai", "model": "Santro"}
	)
	car_id = listing.json()["items"][0]["id"]
	response = await client.get(f"/api/v1/cars/{car_id}")

	assert response.status_code == 200
	data = response.json()
	assert data["id"] == car_id
	assert data["manufacturer"] == "Hyundai"
	assert data["model"] == "Santro"
	assert {"displacement", "cylinders", "fuel_tank_capacity", "power", "torque"} <= data.keys()
	assert (await client.get("/api/v1/cars/2147483647")).status_code == 404
	assert (await client.get("/api/v1/cars/0")).status_code == 422


@pytest.mark.asyncio
async def test_search_is_case_insensitive(client: AsyncClient) -> None:
	response = await client.get("/api/v1/cars/search", params={"q": "sAnTrO"})

	assert response.status_code == 200
	assert response.json()["total_items"] > 0
	assert all(item["model"].lower() == "santro" for item in response.json()["items"])


@pytest.mark.asyncio
async def test_manufacturers_and_models_endpoints(client: AsyncClient) -> None:
	manufacturers = await client.get("/api/v1/manufacturers")
	models = await client.get("/api/v1/manufacturers/hyUnDaI/models")

	assert manufacturers.status_code == 200
	assert len(manufacturers.json()) == 40
	assert models.status_code == 200
	assert models.json()["manufacturer"]["name"] == "Hyundai"
	assert any(model["name"] == "Santro" for model in models.json()["items"])
	assert (await client.get("/api/v1/manufacturers/not-a-make/models")).status_code == 404


@pytest.mark.asyncio
async def test_invalid_filters_and_pagination_are_rejected(client: AsyncClient) -> None:
	assert (
		await client.get(
			"/api/v1/cars", params={"min_price": 500000, "max_price": 100000}
		)
	).status_code == 422
	assert (
		await client.get("/api/v1/cars", params={"manufacturer": "not-a-make"})
	).status_code == 422
	assert (await client.get("/api/v1/cars", params={"page_size": 101})).status_code == 422
	assert (await client.get("/api/v1/cars", params={"sort": "random"})).status_code == 422