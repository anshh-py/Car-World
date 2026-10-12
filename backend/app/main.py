from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import router as v1_router
from app.modules.chat.routes import router as chat_router
from app.core.config import settings

app = FastAPI(title="CarWorld API", version="1.0.0")
app.add_middleware(
	CORSMiddleware,
	allow_origins=settings.frontend_origins,
	allow_methods=["GET", "POST", "OPTIONS"],
	allow_headers=["Authorization", "Content-Type", "Accept"],
)
app.include_router(v1_router, prefix="/api/v1")
app.include_router(chat_router)


@app.get("/health", tags=["health"])
async def health() -> dict[str, str]:
	return {"status": "ok"}
