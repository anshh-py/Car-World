from pathlib import Path
import re

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("page", ["index.html", "BMWCars.html", "CarDetails.html"])
def test_api_pages_load_generated_backend_config_before_api_client(page: str) -> None:
	html = (PROJECT_ROOT / page).read_text(encoding="utf-8")
	config = re.search(r"""<script[^>]+src=["']apps/carworld-config\.js(?:\?[^"']*)?["'][^>]*>""", html)
	api = re.search(r"""<script[^>]+src=["']carworld-api\.js(?:\?[^"']*)?["'][^>]*>""", html)

	assert config is not None, f"{page} must load the generated API origin"
	assert api is not None, f"{page} must load the shared API client"
	assert config.start() < api.start(), f"{page} must configure the API before creating its client"
	assert "?v=" in config.group(0), f"{page} must avoid stale cached API configuration"
	assert "?v=" in api.group(0), f"{page} must avoid a stale cached API client"


def test_every_direct_api_page_loads_the_configured_client_in_order() -> None:
	for page in PROJECT_ROOT.glob("*.html"):
		html = page.read_text(encoding="utf-8")
		if not re.search(r"<script[^>]+src=[\"'](?:car-catalog|car-detail)\.js", html):
			continue
		config = re.search(r"<script[^>]+src=[\"']apps/carworld-config\.js(?:\?[^\"']*)?[\"'][^>]*>", html)
		api = re.search(r"<script[^>]+src=[\"']carworld-api\.js(?:\?[^\"']*)?[\"'][^>]*>", html)
		assert config and api and config.start() < api.start(), page.name


def test_api_client_fails_closed_instead_of_calling_the_static_frontend_origin() -> None:
	source = (PROJECT_ROOT / "carworld-api.js").read_text(encoding="utf-8")

	assert "global.CARWORLD_API_BASE_URL" in source
	assert "global.location.origin" not in source
	assert "CarWorld API URL is not configured" in source
	assert 'request("/cars/" + encodeURIComponent(id))' in source
	assert "error.status = response.status" in source


def test_car_detail_page_explains_a_missing_catalog_record() -> None:
	source = (PROJECT_ROOT / "car-detail.js").read_text(encoding="utf-8")

	assert "error.status === 404" in source
	assert "We couldn't find this car in the catalog." in source


def test_legacy_auth_loader_loads_generated_api_config_first() -> None:
	source = (PROJECT_ROOT / "style.js").read_text(encoding="utf-8")

	assert 'appendScript("apps/carworld-config.js?v=20261013-1"' in source
	assert 'appendScript("carworld-api.js?v=20261013-1", loadAuth)' in source
	assert source.index('appendScript("apps/carworld-config.js?v=20261013-1"') < source.index(
		'appendScript("carworld-api.js?v=20261013-1", loadAuth)'
	)
	assert 'appendScript("carworld-auth.js?v=20261012-1")' in source
	assert "if (window.CarWorldAuth) return" in source


def test_every_login_modal_page_loads_auth_directly_or_through_legacy_loader() -> None:
	pages = list(PROJECT_ROOT.glob("*.html"))
	modal_pages = [page for page in pages if 'id="loginmodal"' in page.read_text(encoding="utf-8")]
	assert modal_pages

	for page in modal_pages:
		html = page.read_text(encoding="utf-8")
		assert 'src="style.js"' in html or 'src="carworld-auth.js?v=20261012-1"' in html, page.name


def test_homepage_auth_script_has_a_cache_busting_version() -> None:
	html = (PROJECT_ROOT / "index.html").read_text(encoding="utf-8")
	assert 'src="carworld-auth.js?v=20261012-1"' in html


def test_production_frontend_config_requires_public_https_origin() -> None:
	source = (PROJECT_ROOT / "backend/scripts/render-frontend-config.ps1").read_text(encoding="utf-8")
	assert '$env:FRONTEND_ENV -eq "production"' in source
	assert '$parsedUrl.Scheme -ne "https"' in source
	assert "$parsedUrl.IsLoopback" in source


def test_local_frontend_launcher_uses_local_backend_origin() -> None:
	source = (PROJECT_ROOT / "start-frontend.cmd").read_text(encoding="utf-8")
	assert 'FRONTEND_API_BASE_URL=http://127.0.0.1:8000' in source
	assert 'FRONTEND_ENV=development' in source


def test_car_detail_page_surfaces_actionable_missing_api_configuration() -> None:
	source = (PROJECT_ROOT / "car-detail.js").read_text(encoding="utf-8")
	assert "error.message" in source
	assert "Run start-frontend.cmd from the project folder" in source
