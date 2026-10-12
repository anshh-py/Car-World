README.md

 Car World – Car Listing & Comparison PortalAdd commentMore actions

**Car World** is a BCA Final Year Web Project that allows users to explore, compare, and view detailed information about various cars categorized by brand and model.

## 🌟 Features

- Multi-brand car listings (e.g., Audi, BMW, Ferrari)
- Interactive About Us and Contact pages
- Organized car pages with specifications and images
- Hosted online using GitHub Pages

## 🛠 Technologies Used

- HTML5
- CSS3
- JavaScript
- Bootstrap (optional)
- GitHub Pages (for hosting)

## 🎓 Project Purpose

This project was built as part of the BCA Final Year curriculum to showcase website development skills and project structure.

## 🔗 Live Demo

👉 [Click here to visit the live site](https://ansh9057.github.io/Car-World/)

## 📩 Contact

For any queries, contact at:  
📍 137 1st Jagatpura Rd, Prem Nagar, Model Town, Jagatpura, Jaipur, Rajasthan 302017  
anshanand317@gmail.com

## CarWorld AI chat API

The existing static HTML, CSS, and JavaScript site calls the FastAPI service. The
service invokes the same `CarWorldAgent` and tool routing used by the Gradio app.
Visitors can chat as guests; a valid Supabase access token is verified and used
when one is supplied. PostgreSQL, Groq, and Tavily credentials stay server-side.

### Local setup

1. Install Python dependencies from the project root:

   ```powershell
   .venv-1\Scripts\python.exe -m pip install -r requirements.txt
   ```

2. Use `.env.example` as a template for the root `.env`. Preserve existing PostgreSQL
   values and add `GROQ_API_KEY`; set `TAVILY_API_KEY` for current web fallback.
   `SUPABASE_URL` and `SUPABASE_ANON_KEY` are only needed for optional Google sign-in.
   The existing local `.env` currently contains PostgreSQL settings only.
3. Start PostgreSQL and the API. For the API, from the project root run:

   ```powershell
   .venv-1\Scripts\python.exe -m uvicorn --app-dir backend app.main:app --reload --port 8000
   ```

   The API is at `http://127.0.0.1:8000`; `GET /health` checks liveness and
   `GET /api/v1/health` remains available for compatibility.
4. Start the static frontend from the project root:

   ```powershell
   .\start-frontend.cmd
   ```

   The script configures the browser to call `http://127.0.0.1:8000` by default before
   serving the site. To use another backend origin, set `$env:FRONTEND_API_BASE_URL`
   before starting it. Visit `http://127.0.0.1:5500/` and use the CarWorld AI panel
   as a guest. Google / Supabase sign-in remains optional elsewhere on the site. The
   `apps/carworld-config.js` value is public browser configuration; never put server
   credentials there.

Run the retained Gradio interface locally with:

```powershell
.venv-1\Scripts\python.exe app.py
```

### Backend configuration and deployment

The root `.env.example` lists server settings. `FRONTEND_ORIGINS` is a comma-separated
allowlist; set it to the exact HTTPS origins serving the frontend. The static site
has no bundler, so configure the public API origin in `apps/carworld-config.js` at
deployment time by running the config renderer with `FRONTEND_ENV=production` and
`FRONTEND_API_BASE_URL` set to the public HTTPS backend origin. Production rendering
rejects HTTP and loopback origins; the generated local config is ignored by Git and
must be rendered again as part of deployment. Never publish the local config. Set
`CARWORLD_ENV=production` on the backend to use the production
chat limits (5 per minute / 40 per day by default), and adjust the limit variables
only to match the expected traffic and provider budget. Do not include Groq, Tavily,
PostgreSQL, or Supabase service-role secrets in frontend configuration or browser
code.

For container deployment, provide the same server settings through the deployment
secret manager / Compose environment and build with the root `Dockerfile`. The API
image includes the existing `app.py` agent and `car_database.py`; run PostgreSQL
separately or use the Compose database service. Deploy the static site to any HTTPS
static host, set its API origin, and set the backend's `FRONTEND_ORIGINS` to that
site's origin. Keep Supabase URL and anon key configured on the backend for the
existing browser sign-in bootstrap and backend token verification. Run Alembic and
the car import script only when provisioning/updating the catalog database.

Run the backend suite from `backend/` with `python -m pytest -q`. The chat endpoint
is `POST /api/chat` and accepts `{ "message": "...", "history": [...] }` with a
Bearer Supabase access token if available; guests may omit it. It limits message
and history sizes. Guest requests share a rolling per-IP limit, so local visitors
on `127.0.0.1` use the same bucket; verified users are limited by account ID.
Defaults are 30 requests per minute / 300 per day in development and 5 / 40 in
production. Set `CHAT_RATE_LIMIT_PER_MINUTE` and `CHAT_RATE_LIMIT_PER_DAY` to
override them. Concurrent requests are serialized through the limiter lock.
Provider errors are logged with safe diagnostic fields and returned without
credentials or tool arguments.
