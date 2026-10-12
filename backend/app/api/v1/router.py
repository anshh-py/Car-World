from fastapi import APIRouter

from app.modules.auth.routes import router as auth_router
from app.modules.cars.routes import router as cars_router

router = APIRouter()


@router.get("/health", tags=["health"])
async def health() -> dict[str, str]:
    return {"status": "ok"}


router.include_router(cars_router)
router.include_router(auth_router, prefix="/auth")