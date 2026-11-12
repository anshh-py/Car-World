"""Tests for the Linux-compatible frontend build (scripts/build-frontend.py)."""

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BUILD_SCRIPT = PROJECT_ROOT / "scripts" / "build-frontend.py"

spec = importlib.util.spec_from_file_location("build_frontend", BUILD_SCRIPT)
build_frontend = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(build_frontend)


@pytest.fixture()
def isolated_build(monkeypatch, tmp_path):
    monkeypatch.setenv("FRONTEND_OUTPUT_DIR", str(tmp_path / "public"))
    monkeypatch.delenv("FRONTEND_API_BASE_URL", raising=False)
    monkeypatch.delenv("FRONTEND_ENV", raising=False)
    return tmp_path / "public"


def test_requires_api_base_url(isolated_build):
    with pytest.raises(build_frontend.BuildError, match="FRONTEND_API_BASE_URL"):
        build_frontend.validate_origin("", production=False)


def test_rejects_invalid_url(isolated_build):
    invalid = [
        "not-a-url",
        "ftp://example.com",
        "https://user:pass@example.com",
        "https://example.com/path",
        "https://example.com?q=1",
        "https://example.com#frag",
    ]
    for url in invalid:
        with pytest.raises(build_frontend.BuildError):
            build_frontend.validate_origin(url, production=False)


def test_requires_https_in_production(isolated_build):
    with pytest.raises(build_frontend.BuildError, match="HTTPS"):
        build_frontend.validate_origin("http://example.com", production=True)


def test_rejects_loopback_in_production(isolated_build):
    for host in ("http://localhost", "https://127.0.0.1", "https://api.localhost"):
        with pytest.raises(build_frontend.BuildError):
            build_frontend.validate_origin(host, production=True)


def test_accepts_public_https_origin(isolated_build):
    origin = build_frontend.validate_origin(
        "https://api.example.com", production=True
    )
    assert origin == "https://api.example.com"


def test_successful_build(isolated_build):
    os.environ["FRONTEND_API_BASE_URL"] = "https://api.example.com"
    os.environ["FRONTEND_ENV"] = "production"

    assert build_frontend.build() == 0
    out = isolated_build
    # Generated config written with the public origin.
    config = out / "apps" / "carworld-config.js"
    assert config.is_file()
    assert "https://api.example.com" in config.read_text(encoding="utf-8")
    # Required HTML page present.
    assert (out / "index.html").is_file()
    # Core first-party assets present.
    for js in build_frontend.REQUIRED_FIRST_PARTY_JS:
        assert (out / js).is_file(), f"missing {js}"
    for css in build_frontend.REQUIRED_CSS:
        assert (out / css).is_file(), f"missing {css}"
    # Asset directories copied with relative paths preserved.
    assert (out / "images" / "logo.png").is_file()
    assert (out / "bootstrap-4.5.3-dist" / "css" / "bootstrap.min.css").is_file()
    assert (out / "FontAwsome" / "all.min.css").is_file()


def test_backend_and_private_files_excluded(isolated_build):
    os.environ["FRONTEND_API_BASE_URL"] = "https://api.example.com"
    assert build_frontend.build() == 0
    out = isolated_build

    prohibited = [
        "backend",
        "Dockerfile",
        "docker-compose.yml",
        "requirements.txt",
        "app.py",
        "car_database.py",
        ".git",
        "docs",
        "pytest.ini",
        ".env",
        ".env.example",
    ]
    for name in prohibited:
        assert not (out / name).exists(), f"should have excluded {name}"

    # No Python files may leak into the output.
    py_files = list(out.rglob("*.py"))
    assert py_files == [], f"Python files leaked into public/: {py_files}"


def test_build_fails_on_missing_required_html(monkeypatch, tmp_path):
    monkeypatch.setenv("FRONTEND_API_BASE_URL", "https://api.example.com")
    monkeypatch.setenv("FRONTEND_OUTPUT_DIR", str(tmp_path / "public"))
    monkeypatch.setattr(build_frontend, "REQUIRED_HTML", ["missing_page.html"])
    assert build_frontend.build() == 3


def test_build_runs_as_executable(monkeypatch, tmp_path):
    monkeypatch.setenv("FRONTEND_API_BASE_URL", "https://api.example.com")
    monkeypatch.setenv("FRONTEND_ENV", "production")
    monkeypatch.setenv("FRONTEND_OUTPUT_DIR", str(tmp_path / "public"))
    result = subprocess.run(
        [sys.executable, str(BUILD_SCRIPT)],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    assert result.returncode == 0
    assert "https://api.example.com" in result.stdout
