# Car World AI — Production Deployment Guide (M4.1)

This document describes how to host the Car World AI application in production.
It does **not** deploy anything; it is the operator reference for the hosting
run.

## Architecture at a glance

```
Browser (static HTML/CSS/JS)
        │  HTTPS
        ▼
FastAPI backend (uvicorn, Groq primary + Qwen3 Gradio fallback)
        │
        ▼
PostgreSQL (Alembic migrations + car dataset import)
```

- **Backend**: FastAPI app at `backend/app/main.py` (module `app.main:app`).
  Serves the REST API (`/api/v1/*`) and the chat endpoint (`POST /api/chat`).
- **Frontend**: static HTML/CSS/JS in the repository root. It reads the API
  origin from the generated `apps/carworld-config.js` (rendered from
  `FRONTEND_API_BASE_URL`). No bundler; host it on any static host.
- **Database**: PostgreSQL via async SQLAlchemy + asyncpg. Schema is managed by
  Alembic (migrations in `backend/alembic/`).
- **LLM providers**: Groq (primary, configured by `GROQ_API_KEY`); Qwen3 on a
  Gradio Space as fallback (optional, configured via `QWEN3_*`).

## 1. Required environment variables

The backend reads `.env` from the repository root (`backend/app/core/config.py`).
In production all values should be injected through the hosting platform's
secret manager / environment (`CARWORLD_ENV=production`).

**Required in production**

| Variable           | Purpose                                                                                                                                                 |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `CARWORLD_ENV`     | Must be `production` (disables `/docs`, tightens chat limits, requires `DATABASE_URL`).                                                                 |
| `DATABASE_URL`     | PostgreSQL connection string, e.g. `postgresql://USER:PASS@HOST:5432/DB?sslmode=require`. Both the app and Alembic read this. Required when production. |
| `GROQ_API_KEY`     | Groq API key for the primary LLM provider.                                                                                                              |
| `FRONTEND_ORIGINS` | Comma-separated list of exact HTTPS origins serving the frontend (CORS allowlist).                                                                      |

**Optional but recommended**

| Variable                     | Purpose                                 | Default               |
| ---------------------------- | --------------------------------------- | --------------------- |
| `GROQ_MODEL`                 | Groq model id                           | `openai/gpt-oss-120b` |
| `GROQ_COOLDOWN_SECONDS`      | Cooldown between chat requests          | `60`                  |
| `TAVILY_API_KEY`             | Enables web/tool fallback               | (none)                |
| `QWEN3_ENDPOINT_URL`         | Qwen3 Gradio Space endpoint             | (none)                |
| `QWEN3_SPACE_ID`             | Qwen3 Space id                          | (none)                |
| `QWEN3_MODEL_ID`             | Qwen3 model id                          | `qwen3-4b-finetuned`  |
| `QWEN3_TIMEOUT`              | Qwen3 request timeout (seconds)         | `60.0`                |
| `QWEN3_SUPPORTS_TOOLS`       | Whether the Space supports tools        | `false`               |
| `HUGGINGFACE_TOKEN`          | HF token for private/gated Space        | (none)                |
| `SUPABASE_URL`               | Supabase project URL (optional sign-in) | (none)                |
| `SUPABASE_ANON_KEY`          | Supabase anon key (public)              | (none)                |
| `CHAT_RATE_LIMIT_PER_MINUTE` | Per-minute chat limit                   | `5` (prod)            |
| `CHAT_RATE_LIMIT_PER_DAY`    | Per-day chat limit                      | `40` (prod)           |

> **Local dev difference**: in development the backend reads
> `POSTGRES_HOST/PORT/USER/PASSWORD/DB` (defaults to `localhost:5432/carworld`)
> instead of `DATABASE_URL`. `CARWORLD_ENV=development` keeps `/docs` enabled and
> uses looser chat limits (30/min, 300/day).

## 2. Secret handling

- **Never commit** `.env` or any real secret. `.gitignore` excludes `.env`,
  `.env.*`, and local generated frontend config. The only templates that are
  intentionally keepable are `.env.example` (see below) and
  `apps/carworld-config.js` is _generated_ and always gitignored.
- Frontend config (`apps/carworld-config.js`) contains **only** the public API
  origin — never server credentials. The anon Supabase key may appear there
  because it is a public key, but service-role keys and API keys must stay
  server-side.
- Use the platform secret manager (e.g. Render env vars / Railway variables)
  rather than committing secrets.
- The repository contains no hardcoded real credentials; all keys are read from
  the environment or `.env`.

### `.env.example` template

> Note: the Codvian tooling refuses to write/read any `.env*` path for security,
> so the canonical template is reproduced _here_ in this doc. Copy the block
> below into a root `.env.example` file if you want a committed template (the
> `.gitignore` allows `.env.example`).

```env
# --- Runtime environment ----------------------------------------------
CARWORLD_ENV=development

# --- PostgreSQL (dev; use DATABASE_URL in production) ------------------
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=carworld
POSTGRES_PASSWORD=change-me
POSTGRES_DB=carworld
#DATABASE_URL=postgresql://USER:PASS@HOST:5432/DB?sslmode=require

# --- Groq (primary LLM) -----------------------------------------------
GROQ_API_KEY=gsk_YOUR_GROQ_API_KEY
GROQ_MODEL=openai/gpt-oss-120b
GROQ_COOLDOWN_SECONDS=60

# --- Tavily (web fallback) --------------------------------------------
TAVILY_API_KEY=tvly_YOUR_TAVILY_API_KEY

# --- Qwen3 Gradio Space fallback (optional) ---------------------------
QWEN3_ENDPOINT_URL=
QWEN3_SPACE_ID=
QWEN3_MODEL_ID=qwen3-4b-finetuned
QWEN3_TIMEOUT=60.0
QWEN3_SUPPORTS_TOOLS=false
HUGGINGFACE_TOKEN=

# --- Supabase (optional auth) -----------------------------------------
SUPABASE_URL=https://YOUR-PROJECT.supabase.co
SUPABASE_ANON_KEY=YOUR_ANON_KEY

# --- CORS -------------------------------------------------------------
FRONTEND_ORIGINS=https://example-carworld.pages.dev,https://www.example-carworld.pages.dev
```

## 3. Backend startup

**Python / pip runtime** (e.g. Render, Railway, Fly):

```bash
pip install -r requirements.txt
cd backend
alembic upgrade head
python scripts/import_cars.py   # seed the catalog (idempotent)
uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}"
```

**Docker**: the root `Dockerfile` (Python 3.12-slim) builds the image, runs
migrations + import, then starts uvicorn. It binds to `$PORT` if set,
otherwise `8000`:

```bash
docker build -t carworld-api .
docker run -p 8000:8000 \
  -e CARWORLD_ENV=production \
  -e DATABASE_URL="postgresql://..." \
  -e GROQ_API_KEY=... \
  -e FRONTEND_ORIGINS=https://your-site.example \
  carworld-api
```

Use `docker-compose.yml` for local orchestration (postgres + api). The compose
`api` service inherits settings from the environment / `.env`.

## 4. Database setup & migrations

- Create the PostgreSQL database (or use a managed instance).
- **Alembic** migrations live in `backend/alembic/`; `env.py` reads the same
  `DATABASE_URL`. Run:
  ```bash
  cd backend
  alembic upgrade head
  ```
- **Seed data**: `python scripts/import_cars.py` imports `Data/car_dataset.csv`
  (1,276 records). It is idempotent (skips exact duplicates). Run it after the
  first migration / whenever the catalog must be refreshed.
- Migrations are async and run against production PostgreSQL without change;
  no filesystem or non-Postgres assumptions are required at runtime.

## 5. Frontend hosting — Render Static Site

The frontend is a multi-page static site in the repository root. It is deployed
as a **Render Static Site**, built with the Linux-compatible script
`scripts/build-frontend.py` (this supersedes the PowerShell-only
`backend/scripts/render-frontend-config.ps1`, which cannot run in Render's Linux
build environment).

### Render Static Site settings

- **Build command**:
  ```bash
  FRONTEND_ENV=production \
  FRONTEND_API_BASE_URL=https://car-world-agoj.onrender.com \
  python scripts/build-frontend.py
  ```
- **Publish directory**: `public`
- **Environment variables** (set in the Render dashboard):

  | Variable                | Value                                                         |
  | ----------------------- | ------------------------------------------------------------- |
  | `FRONTEND_ENV`          | `production`                                                  |
  | `FRONTEND_API_BASE_URL` | `https://car-world-agoj.onrender.com` (public backend origin) |
  | `FRONTEND_OUTPUT_DIR`   | _(optional)_ defaults to `public`                             |

### What the build does

- Requires `FRONTEND_API_BASE_URL`; fails clearly if it is unset.
- In production (`FRONTEND_ENV=production`) rejects non-HTTPS origins and
  loopback hosts (`localhost`, `127.0.0.1`, `*.localhost`).
- Rejects any origin containing credentials, a path, query, or fragment.
- Generates `apps/carworld-config.js` (in the output `public/apps/`) with the
  configured origin, then copies all first-party HTML pages and assets
  (`images/`, `bootstrap-4.5.3-dist/`, `FontAwsome/`, root CSS/JS) preserving
  relative paths.
- Never copies backend source, database/data files, Python, Docker, Git
  metadata, environment files, secrets, or docs into `public/`.

### CORS on the backend

The backend allows a fixed set of origins. When the Static Site is created,
add its URL to `FRONTEND_ORIGINS` on the backend service, e.g.:

```
FRONTEND_ORIGINS=https://car-world.onrender.com
```

If the Static Site is later given a custom domain, add that HTTPS origin to the
same list (comma-separated). The frontend itself never hardcodes the backend
origin — it is injected at build time from `FRONTEND_API_BASE_URL`.

### Deployment & smoke test

1. Push the repository. Render runs the build command above and publishes
   `public/`.
2. Open the Static Site URL; confirm the page renders and the chat/catalog
   widgets load.
3. In the browser console verify `window.CARWORLD_API_BASE_URL` equals
   `https://car-world-agoj.onrender.com`.
4. Confirm the backend `FRONTEND_ORIGINS` includes the Static Site origin and
   that `GET /health` and `GET /api/v1/health` return `{"status": "ok"}`.
5. Test `carworld-api.js` requests reach the backend (console Network tab shows
   200 responses with `Access-Control-Allow-Origin` set to the Static Site
   origin).

> **Local dev**: run `start-frontend.cmd` or set
> `FRONTEND_API_BASE_URL` + `FRONTEND_ENV=development` and run
> `python scripts/build-frontend.py`. Development allows HTTP for local testing.

## 6. Health checks

The backend exposes two liveness endpoints:

- `GET /health` → `{"status": "ok"}`
- `GET /api/v1/health` → `{"status": "ok"}`

Point the platform health check at `/health` (or `/api/v1/health`).

## 7. Production behavior notes

- `CARWORLD_ENV=production` disables Swagger/ReDoc/OpenAPI (`/docs`,
  `/redoc`, `/openapi.json` → 404) and tightens chat rate limits.
- Security middleware always applies: request-body size limit (64 KiB), security
  headers (`X-Content-Type-Options`, `X-Frame-Options: DENY`,
  `Referrer-Policy`, `Cache-Control: no-store`), and CORS.
- Chat rate limiting is in-memory (per-IP for guests, per-account for signed-in
  users). For multi-instance deployments this is per-process; scale accordingly
  or add a shared store if a hard global limit is required.
- No persistent local filesystem is required at runtime; all state is in
  PostgreSQL. The only filesystem reads are the bundled dataset used by the
  import script (already inside the image).
