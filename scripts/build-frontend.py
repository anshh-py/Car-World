#!/usr/bin/env python3
"""Build the Car World static frontend for deployment (Linux / CI compatible).

Generates ``public/`` from the repository-root static site:

  * requires ``FRONTEND_API_BASE_URL`` (the public backend origin) and renders
    ``apps/carworld-config.js`` from it;
  * in production (``FRONTEND_ENV=production``) requires a public HTTPS origin
    with no credentials, path, query, or fragment;
  * copies every first-party HTML page and its referenced static assets
    (CSS/JS, ``images/``, ``bootstrap-4.5.3-dist/``, ``FontAwsome/``) while
    preserving relative paths;
  * never copies backend/database/docker/git/env/secret files into the output.

Usage:
    FRONTEND_API_BASE_URL=https://api.example.com \
    FRONTEND_ENV=production \
    python scripts/build-frontend.py

Exit codes: 0 success; 2 configuration error; 3 missing required file.
"""

import os
import shutil
import sys
from pathlib import Path
from urllib.parse import urlsplit

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "apps"
CONFIG_FILENAME = "carworld-config.js"
CONFIG_PATH = CONFIG_DIR / CONFIG_FILENAME

# First-party files that must exist for the site to build. Anything missing
# here fails the build (we never fabricate content).
REQUIRED_HTML = ["index.html"]

REQUIRED_FIRST_PARTY_JS = [
    "carworld-api.js",
    "carworld-auth.js",
    "carworld-chat.js",
    "car-catalog.js",
    "car-detail.js",
    "style.js",
]

REQUIRED_CSS = [
    "style.css",
    "car_style.css",
    "carworld-chat.css",
    "brand-style.css",
]

# Whole directories whose contents are copied verbatim (relative paths kept).
ASSET_DIRECTORIES = [
    "images",
    "bootstrap-4.5.3-dist",
    "FontAwsome",
]

# Top-level static files (besides HTML/CSS/JS already covered) to copy.
EXTRA_ROOT_FILES = [
    "brand-style.js",
    "scroll-js-script.js",
    "jqueryui.libs.1.16.1.js",
]

# Path fragments that are never allowed in the public output.
FORBIDDEN_FRAGMENTS = (
    ".git",
    ".env",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".pytest_basetemp",
    ".venv",
    "venv",
)
FORBIDDEN_EXTENSIONS = (".py", ".pyc", ".pyo", ".pem", ".key", ".crt")


class BuildError(Exception):
    """Raised for a configuration or required-file failure."""


def validate_origin(origin: str, production: bool) -> str:
    """Validate ``FRONTEND_API_BASE_URL`` and return the normalized authority."""
    if not origin:
        raise BuildError("FRONTEND_API_BASE_URL is required to build the frontend.")

    parts = urlsplit(origin)
    scheme = parts.scheme.lower()
    host = (parts.hostname or "").lower()

    if not parts.scheme or not host:
        raise BuildError(
            "FRONTEND_API_BASE_URL must be an absolute URL with a host, got: " + origin
        )
    if scheme not in ("http", "https"):
        raise BuildError("FRONTEND_API_BASE_URL scheme must be http or https: " + origin)
    if parts.username or parts.password:
        raise BuildError("FRONTEND_API_BASE_URL must not contain credentials: " + origin)
    if parts.path and parts.path != "/":
        raise BuildError("FRONTEND_API_BASE_URL must have no path: " + origin)
    if parts.query:
        raise BuildError("FRONTEND_API_BASE_URL must have no query string: " + origin)
    if parts.fragment:
        raise BuildError("FRONTEND_API_BASE_URL must have no fragment: " + origin)

    if production:
        if scheme != "https":
            raise BuildError("Production frontend requires an HTTPS origin: " + origin)
        loopback_hosts = ("localhost", "127.0.0.1", "::1")
        if host in loopback_hosts or host.endswith(".localhost"):
            raise BuildError("Production frontend must use a public origin: " + origin)

    return f"{parts.scheme}://{parts.netloc}"


def _fail_if_missing(path: Path, label: str) -> None:
    if not path.is_file():
        raise BuildError(f"Missing required {label}: {path}")


def _copy_file(src: Path, dst_dir: Path) -> None:
    if any(frag in src.parts for frag in FORBIDDEN_FRAGMENTS):
        return
    if src.suffix.lower() in FORBIDDEN_EXTENSIONS:
        return
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst_dir / src.name)


def _is_prohibited(rel: Path) -> bool:
    return any(frag in rel.parts for frag in FORBIDDEN_FRAGMENTS) or (
        rel.suffix.lower() in FORBIDDEN_EXTENSIONS
    )


def render_config(origin: str, out_dir: Path) -> Path:
    """Write ``apps/carworld-config.js`` in both source and output trees."""
    content = (
        "/* Generated from FRONTEND_API_BASE_URL; contains no secret. */\n"
        "window.CARWORLD_API_BASE_URL = " + repr(origin) + ";\n"
    )
    for target_dir in (CONFIG_DIR, out_dir / "apps"):
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / CONFIG_FILENAME).write_text(content, encoding="utf-8")
    return out_dir / "apps" / CONFIG_FILENAME


def copy_html_pages(out_dir: Path) -> int:
    """Copy every first-party HTML page from the repo root."""
    count = 0
    for html in sorted(REPO_ROOT.glob("*.html")):
        rel = html.relative_to(REPO_ROOT)
        if _is_prohibited(rel):
            continue
        _copy_file(html, out_dir)
        count += 1
    return count


def copy_static_assets(out_dir: Path) -> int:
    """Copy all CSS/JS root files and asset directories, preserving paths."""
    count = 0

    for css in REQUIRED_CSS:
        _fail_if_missing(REPO_ROOT / css, f"CSS file {css}")
        _copy_file(REPO_ROOT / css, out_dir)
        count += 1

    for js in REQUIRED_FIRST_PARTY_JS:
        _fail_if_missing(REPO_ROOT / js, f"JS file {js}")
        _copy_file(REPO_ROOT / js, out_dir)
        count += 1

    for rel in EXTRA_ROOT_FILES:
        src = REPO_ROOT / rel
        if src.is_file():
            _copy_file(src, out_dir)
            count += 1

    for name in ASSET_DIRECTORIES:
        src_dir = REPO_ROOT / name
        if not src_dir.is_dir():
            raise BuildError(f"Missing required asset directory: {name}")
        for src in sorted(src_dir.rglob("*")):
            if not src.is_file():
                continue
            rel = src.relative_to(REPO_ROOT)
            if _is_prohibited(rel):
                continue
            dst = out_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            count += 1

    return count


def build() -> int:
    production = os.environ.get("FRONTEND_ENV", "").strip().lower() == "production"
    origin_env = os.environ.get("FRONTEND_API_BASE_URL", "").strip()
    output_dir = Path(os.environ.get("FRONTEND_OUTPUT_DIR", str(REPO_ROOT / "public")))

    try:
        origin = validate_origin(origin_env, production)
    except BuildError as exc:
        print(f"[build-frontend] error: {exc}", file=sys.stderr)
        return 2

    # Required files must exist before we attempt a build (exit 3).
    try:
        for html in REQUIRED_HTML:
            _fail_if_missing(REPO_ROOT / html, f"HTML page {html}")
    except BuildError as exc:
        print(f"[build-frontend] error: {exc}", file=sys.stderr)
        return 3

    # Start from a clean public directory.
    if output_dir.exists():
        shutil.rmtree(output_dir)

    try:
        render_config(origin, output_dir)
        html_count = copy_html_pages(output_dir)
        asset_count = copy_static_assets(output_dir)
    except BuildError as exc:
        print(f"[build-frontend] error: {exc}", file=sys.stderr)
        return 3
    except OSError as exc:
        print(f"[build-frontend] filesystem error: {exc}", file=sys.stderr)
        return 3

    print(
        "[build-frontend] wrote "
        f"{html_count} HTML pages + {asset_count} static assets to {output_dir}"
    )
    print(f"[build-frontend] API origin: {origin}")
    return 0


if __name__ == "__main__":
    sys.exit(build())
